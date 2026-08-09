from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import torch
from torch.nn import functional as F

from .protocol import (
    ALGORITHM_TEMPERATURE,
    INITIAL_MASK_COUNT,
    MAX_CONTEXT_TOKENS,
    MAX_FORWARDS,
    MAX_NEW_TOKENS,
    NUMBER_TRANSFER_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    config_hash,
    width_label,
)


@dataclass(frozen=True)
class TokenizerSpec:
    bos_id: int
    eos_id: int
    mask_id: int
    expand_id: int


def initial_middle(mask_id: int) -> list[int]:
    return [mask_id] * INITIAL_MASK_COUNT


def eligible_mask_ranks(mask_positions: Sequence[int], width: int | None) -> list[int]:
    if not mask_positions:
        return []
    frontier = min(mask_positions)
    if width is None:
        return list(range(len(mask_positions)))
    upper = frontier + width
    return [rank for rank, position in enumerate(mask_positions) if frontier <= position < upper]


def action_name(token_id: int, mask_id: int, expand_id: int, delete_id: int) -> str:
    if token_id == expand_id:
        return "expand"
    if token_id == delete_id:
        return "delete"
    if token_id == mask_id:
        return "mask_noop"
    return "normal"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_hash(token_ids: torch.Tensor, real_length: int) -> str:
    payload = token_ids[0, :real_length].detach().cpu().tolist()
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def physical_line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def newline_token_count(tokenizer: Any, token_ids: Sequence[int]) -> int:
    count = 0
    for token_id in token_ids:
        decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if "\n" in decoded or "\r" in decoded:
            count += 1
    return count


def _gpu_time_start(device: str) -> tuple[Any | None, Any | None]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None, None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _gpu_time_stop(events: tuple[Any | None, Any | None]) -> float | None:
    start, end = events
    if start is None or end is None:
        return None
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


@torch.inference_mode()
def decode_frontier(
    *,
    model: Any,
    tokenizer: Any,
    sample_tokens_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    width: int | None,
    seed: int,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if NUMBER_TRANSFER_TOKENS != 1:
        raise AssertionError("protocol requires one transfer token")
    spec = TokenizerSpec(
        bos_id=int(tokenizer.bos_id),
        eos_id=int(tokenizer.eos_id),
        mask_id=int(tokenizer.mask_id),
        expand_id=int(tokenizer.expand_id),
    )
    set_seed(seed)
    prefix = list(map(int, prefix_ids))
    suffix = list(map(int, suffix_ids))
    input_ids = prefix + initial_middle(spec.mask_id) + suffix
    if len(input_ids) > MAX_CONTEXT_TOKENS:
        raise ValueError(f"input length {len(input_ids)} exceeds {MAX_CONTEXT_TOKENS}")
    prefix_length = len(prefix)
    suffix_length = len(suffix)
    max_tokens = min(
        MAX_CONTEXT_TOKENS,
        len(input_ids) + MAX_NEW_TOKENS - INITIAL_MASK_COUNT,
    )
    if max_tokens != len(input_ids):
        raise AssertionError("locked 64/64 mapping must preserve the initial total tensor length")
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    num_generation_tokens = INITIAL_MASK_COUNT
    expand_budget = MAX_NEW_TOKENS
    initial_dynamic = x[0, prefix_length : prefix_length + num_generation_tokens]
    initial_unresolved = int((initial_dynamic == spec.mask_id).sum().item())
    assert num_generation_tokens == INITIAL_MASK_COUNT
    assert initial_unresolved == INITIAL_MASK_COUNT

    traces: list[dict[str, Any]] = [
        {
            "step": 0,
            "kind": "initial_state",
            "dynamic_length": num_generation_tokens,
            "unresolved_masks": initial_unresolved,
            "mask_token_id": spec.mask_id,
            "continuous_middle": True,
        }
    ]
    action_counts: Counter[str] = Counter()
    broadcast_delete_tail_lengths: list[int] = []
    transition_counts: Counter[str] = Counter()
    state_visits: Counter[str] = Counter()
    peak_unresolved = initial_unresolved
    frontier_violation = 0
    forward_sequence_lengths: list[int] = []
    token_forwards = 0
    stop_reason = "forward_cap"
    wall_start = time.perf_counter()
    gpu_events = _gpu_time_start(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for forward_index in range(MAX_FORWARDS):
        real_length = prefix_length + num_generation_tokens + suffix_length
        attention_1d = torch.ones((1, real_length), dtype=torch.int16, device=x.device)
        attention_1d = F.pad(attention_1d, (0, max_tokens - real_length), value=0)
        mask_index = (x == spec.mask_id) & (attention_1d == 1)
        dynamic_before = x[0, prefix_length : prefix_length + num_generation_tokens]
        unresolved_before = int((dynamic_before == spec.mask_id).sum().item())
        peak_unresolved = max(peak_unresolved, unresolved_before)
        if not bool(mask_index[:, :real_length].any().item()):
            stop_reason = "no_unresolved_masks"
            break

        local_mask_positions = (
            (mask_index[0].nonzero(as_tuple=False).squeeze(1) - prefix_length)
            .detach()
            .cpu()
            .tolist()
        )
        if any(position < 0 or position >= num_generation_tokens for position in local_mask_positions):
            raise AssertionError("visible mask escaped the dynamic middle")
        eligible_ranks = eligible_mask_ranks(local_mask_positions, width)
        if not eligible_ranks:
            raise AssertionError("unresolved masks exist but no position is eligible")
        eligible_positions = [local_mask_positions[rank] for rank in eligible_ranks]
        frontier = min(local_mask_positions)

        tok_idx = attention_1d.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_1d == 0, 1)
        pairwise_attention = torch.logical_and(
            attention_1d.unsqueeze(1).unsqueeze(-2),
            attention_1d.unsqueeze(1).unsqueeze(-1),
        )
        pre_hash = state_hash(x, real_length)
        state_visits[pre_hash] += 1
        output = model(x, pairwise_attention, tok_idx)
        logits = output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        logits = logits[mask_index]
        if real_length == max_tokens or expand_budget == 0:
            logits[:, spec.expand_id] -= 1e9
        confidence, x0 = sample_tokens_fn(
            logits,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            neg_entropy=True,
        )
        candidate_confidence = confidence[
            torch.tensor(eligible_ranks, dtype=torch.long, device=confidence.device)
        ]
        if ALGORITHM_TEMPERATURE:
            raise AssertionError("locked algorithm temperature is zero")
        selected_candidate_rank = int(torch.topk(candidate_confidence, 1).indices.item())
        selected_mask_rank = eligible_ranks[selected_candidate_rank]
        selected_position = int(local_mask_positions[selected_mask_rank])
        if selected_position not in eligible_positions:
            frontier_violation += 1
        assert selected_position in eligible_positions
        proposal_token_id = int(x0[selected_mask_rank].item())
        action = action_name(
            proposal_token_id,
            mask_id=spec.mask_id,
            expand_id=spec.expand_id,
            delete_id=spec.eos_id,
        )

        x0_committed = torch.full_like(x0, spec.mask_id)
        x0_committed[selected_mask_rank] = x0[selected_mask_rank].clone()
        x[mask_index] = x0_committed

        x_seq = x[0]
        eos_indices_all = (x_seq == spec.eos_id).nonzero(as_tuple=True)[0]
        if eos_indices_all.numel() > 0:
            first_eos_idx = int(eos_indices_all[0].item())
            position_mask = torch.arange(x_seq.size(0), device=x.device) >= first_eos_idx
            replace_mask = position_mask & mask_index[0]
            x_seq.masked_fill_(replace_mask, spec.eos_id)
            x = x_seq.unsqueeze(0)

        expand_indices = (x[0] == spec.expand_id).nonzero(as_tuple=False).squeeze(1)
        expand_count = int(expand_indices.numel())
        for index in sorted(expand_indices.tolist(), reverse=True):
            x = torch.cat(
                (
                    x[:, :index],
                    torch.tensor([[spec.mask_id, spec.mask_id]], device=x.device),
                    x[:, index + 1 :],
                ),
                dim=1,
            )
            num_generation_tokens += 1
            expand_budget -= 1
            if x.shape[1] > max_tokens:
                x = x[:, :max_tokens]

        eos_delete_indices = (
            ((x[0] == spec.eos_id) & (mask_index[0] == 1))
            .nonzero(as_tuple=False)
            .squeeze(1)
        )
        delete_count = int(eos_delete_indices.numel())
        if delete_count:
            broadcast_delete_tail_lengths.append(delete_count)
        for index in sorted(eos_delete_indices.tolist(), reverse=True):
            x = torch.cat(
                (
                    x[:, :index],
                    x[:, index + 1 :],
                    torch.tensor([[spec.mask_id]], device=x.device),
                ),
                dim=1,
            )
            num_generation_tokens -= 1

        if action == "expand":
            assert expand_count >= 1
        if action == "delete":
            assert delete_count >= 1
        action_counts[action] += 1
        if delete_count == 1:
            action_counts["single_point_delete"] += 1
        elif delete_count > 1:
            action_counts["broadcast_delete"] += 1

        real_length_after = prefix_length + num_generation_tokens + suffix_length
        post_hash = state_hash(x, real_length_after)
        transition_key = f"{pre_hash}:{selected_position}:{proposal_token_id}:{post_hash}"
        transition_counts[transition_key] += 1
        unresolved_after = int(
            (
                x[0, prefix_length : prefix_length + num_generation_tokens]
                == spec.mask_id
            ).sum().item()
        )
        forward_sequence_lengths.append(real_length)
        token_forwards += real_length
        traces.append(
            {
                "step": forward_index + 1,
                "kind": "commit",
                "frontier": frontier,
                "window": width_label(width),
                "eligible_positions": eligible_positions,
                "commit_position": selected_position,
                "proposal_token_id": proposal_token_id,
                "action": action,
                "dynamic_length_before": int(dynamic_before.numel()),
                "dynamic_length_after": num_generation_tokens,
                "unresolved_before": unresolved_before,
                "unresolved_after": unresolved_after,
                "expand_count": expand_count,
                "delete_count": delete_count,
                "broadcast_delete_tail_length": delete_count if delete_count > 1 else 0,
                "pre_state_hash": pre_hash,
                "post_state_hash": post_hash,
            }
        )

    wall_time = time.perf_counter() - wall_start
    gpu_time = _gpu_time_stop(gpu_events)
    final_middle = x[0, prefix_length : prefix_length + num_generation_tokens]
    final_ids = final_middle.detach().cpu().tolist()
    final_unresolved = int((final_middle == spec.mask_id).sum().item())
    completion = tokenizer.decode(final_ids, skip_special_tokens=True)
    newline_commit_count = 0
    for trace in traces[1:]:
        if trace["action"] != "normal":
            continue
        decoded = tokenizer.decode([trace["proposal_token_id"]], skip_special_tokens=False)
        if "\n" in decoded or "\r" in decoded:
            newline_commit_count += 1
    exact_cycles = sum(count - 1 for count in transition_counts.values() if count > 1)
    repeated_states = sum(count - 1 for count in state_visits.values() if count > 1)
    forward_count = len(forward_sequence_lengths)
    if forward_count == MAX_FORWARDS and final_unresolved:
        stop_reason = "forward_cap"
    elif not final_unresolved:
        stop_reason = "no_unresolved_masks"
    status = "completed" if final_unresolved == 0 else "incomplete"
    peak_memory = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    return {
        "config_hash": config_hash(),
        "width": width_label(width),
        "seed": seed,
        "initial_mask_count": INITIAL_MASK_COUNT,
        "initial_dynamic_length": INITIAL_MASK_COUNT,
        "final_dynamic_length": num_generation_tokens,
        "completion_token_ids": final_ids,
        "completion": completion,
        "status": status,
        "stop_reason": stop_reason,
        "forward_count": forward_count,
        "forward_sequence_lengths": forward_sequence_lengths,
        "token_forwards": token_forwards,
        "wall_time_seconds": wall_time,
        "gpu_time_seconds": gpu_time,
        "peak_cuda_memory_bytes": peak_memory,
        "step_trace": traces,
        "normal_token_count": action_counts["normal"],
        "expand_count": action_counts["expand"],
        "delete_action_count": action_counts["delete"],
        "single_point_delete_count": action_counts["single_point_delete"],
        "broadcast_delete_count": action_counts["broadcast_delete"],
        "broadcast_delete_tail_lengths": broadcast_delete_tail_lengths,
        "mask_noop_count": action_counts["mask_noop"],
        "newline_token_count": newline_token_count(tokenizer, final_ids),
        "newline_commit_count": newline_commit_count,
        "physical_line_count": physical_line_count(completion),
        "peak_unresolved_masks": peak_unresolved,
        "final_unresolved_masks": final_unresolved,
        "exact_cycle_count": exact_cycles,
        "repeated_state_count": repeated_states,
        "oscillation_detected": bool(exact_cycles or repeated_states),
        "frontier_violation": frontier_violation,
        "exception": None,
    }


def scripted_output(logits: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(logits=logits)
