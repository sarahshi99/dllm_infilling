from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
from torch.nn import functional as F


GLOBAL_CONFIDENCE = "confidence_global"
LEFT_TO_RIGHT_FRONTIER = "left_to_right_frontier"


@dataclass(frozen=True)
class Proposal:
    position: int
    token_id: int
    token_type: str


def shift_logits_to_targets(logits: torch.Tensor) -> torch.Tensor:
    """Match DreamOn's released one-token logits-to-target alignment."""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def left_frontier_positions(
    active_mask_positions: Sequence[int], *, requested_k: int
) -> list[int]:
    if requested_k < 1:
        raise ValueError("requested_k must be positive")
    if not active_mask_positions:
        return []
    selected = [int(active_mask_positions[0])]
    for position in active_mask_positions[1:]:
        if len(selected) == requested_k or int(position) != selected[-1] + 1:
            break
        selected.append(int(position))
    return selected


def select_positions(
    active_mask_positions: Sequence[int],
    confidence: Sequence[float],
    *,
    requested_k: int,
    policy: str,
) -> list[int]:
    if len(active_mask_positions) != len(confidence):
        raise ValueError("active positions and confidence must have equal length")
    if requested_k < 1:
        raise ValueError("requested_k must be positive")
    if policy == LEFT_TO_RIGHT_FRONTIER:
        return left_frontier_positions(active_mask_positions, requested_k=requested_k)
    if policy != GLOBAL_CONFIDENCE:
        raise ValueError(f"unknown policy: {policy}")
    ranked = sorted(
        enumerate(confidence), key=lambda item: (-float(item[1]), int(active_mask_positions[item[0]]))
    )
    return [int(active_mask_positions[index]) for index, _ in ranked[:requested_k]]


def apply_left_frontier_barrier(
    proposals: Sequence[Proposal],
) -> tuple[list[Proposal], Proposal | None, int]:
    """Keep normal proposals before the first structural action only."""
    for index, proposal in enumerate(proposals):
        if proposal.token_type in {"expand", "delete"}:
            return list(proposals[:index]), proposal, len(proposals) - index - 1
        if proposal.token_type != "normal":
            raise ValueError(f"unknown proposal token type: {proposal.token_type}")
    return list(proposals), None, 0


def validate_resume_rows(
    rows: Iterable[dict[str, object]], *, variant: str, seed: int
) -> set[str]:
    completed: set[str] = set()
    for row in rows:
        case_key = str(row.get("case_key") or "")
        if not case_key:
            raise RuntimeError("resume row has blank case key")
        if str(row.get("variant")) != variant or int(row.get("seed", -1)) != int(seed):
            raise RuntimeError("resume row has a different variant/seed")
        if case_key in completed:
            raise RuntimeError("resume rows contain a duplicate case key")
        completed.add(case_key)
    return completed


def token_type(token_id: int, *, eos_id: int, expand_id: int, mask_id: int) -> str:
    if token_id == expand_id:
        return "expand"
    if token_id == eos_id:
        return "delete"
    if token_id == mask_id:
        return "mask_noop"
    return "normal"


def _global_ranks(confidence: torch.Tensor, requested_k: int) -> list[int]:
    count = min(int(requested_k), int(confidence.numel()))
    return torch.topk(confidence, count).indices.detach().cpu().tolist()


def _left_frontier_ranks(mask_positions: Sequence[int], requested_k: int) -> list[int]:
    positions = left_frontier_positions(mask_positions, requested_k=requested_k)
    return list(range(len(positions)))


def _apply_one_expand(x: torch.Tensor, index: int, mask_id: int) -> torch.Tensor:
    return torch.cat(
        (x[:, :index], torch.tensor([[mask_id, mask_id]], device=x.device), x[:, index + 1 :]),
        dim=1,
    )


def _apply_one_delete(x: torch.Tensor, index: int, mask_id: int) -> torch.Tensor:
    return torch.cat(
        (x[:, :index], x[:, index + 1 :], torch.tensor([[mask_id]], device=x.device)),
        dim=1,
    )


@torch.inference_mode()
def decode_with_policy(
    *,
    model: Any,
    tokenizer: Any,
    sample_tokens_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    min_gen_len: int,
    max_gen_len: int,
    max_tokens: int,
    steps: int,
    expand_budget: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    requested_k: int,
    policy: str,
    device: str,
) -> dict[str, Any]:
    """Decode one case with official C semantics or the L structural barrier."""
    if policy not in {GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER}:
        raise ValueError(f"unknown policy: {policy}")
    if requested_k not in {1, 2, 4}:
        raise ValueError("requested_k must be one of 1, 2, or 4")
    mask_id, eos_id, expand_id = map(
        int, (tokenizer.mask_id, tokenizer.eos_id, tokenizer.expand_id)
    )
    prefix = list(map(int, prefix_ids))
    suffix = list(map(int, suffix_ids))
    prefix_length = len(prefix)
    input_ids = prefix + [mask_id] * int(min_gen_len) + suffix
    maximum = min(int(max_tokens), len(input_ids) + int(max_gen_len) - int(min_gen_len))
    x = F.pad(torch.tensor([input_ids], device=device), (0, maximum - len(input_ids)), value=mask_id)
    dynamic_length = int(min_gen_len)
    traces: list[dict[str, Any]] = []
    total_normal = total_expand = total_delete = 0
    for step in range(int(steps)):
        real_length = prefix_length + dynamic_length + len(suffix)
        attention_1d = F.pad(
            torch.ones((1, real_length), dtype=torch.int16, device=x.device),
            (0, maximum - real_length),
            value=0,
        )
        mask_index = (x == mask_id) & (attention_1d == 1)
        if not bool(mask_index.any().item()):
            break
        absolute_positions = mask_index[0].nonzero(as_tuple=False).squeeze(1).tolist()
        local_positions = [int(position - prefix_length) for position in absolute_positions]
        tok_idx = attention_1d.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_1d == 0, 1)
        pairwise_attention = torch.logical_and(
            attention_1d.unsqueeze(1).unsqueeze(-2), attention_1d.unsqueeze(1).unsqueeze(-1)
        )
        raw_logits = model(x, pairwise_attention, tok_idx).logits
        logits = shift_logits_to_targets(raw_logits)[mask_index]
        if real_length == maximum or expand_budget == 0:
            logits[:, expand_id] -= 1e9
        confidence, predictions = sample_tokens_fn(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True
        )
        if policy == GLOBAL_CONFIDENCE:
            selected_ranks = _global_ranks(confidence, requested_k)
        else:
            selected_ranks = _left_frontier_ranks(local_positions, requested_k)
        proposals = [
            Proposal(
                position=int(absolute_positions[rank]),
                token_id=int(predictions[rank].item()),
                token_type=token_type(int(predictions[rank].item()), eos_id=eos_id, expand_id=expand_id, mask_id=mask_id),
            )
            for rank in selected_ranks
        ]
        barrier_action: Proposal | None = None
        discarded = 0
        committed = proposals
        if policy == LEFT_TO_RIGHT_FRONTIER:
            committed, barrier_action, discarded = apply_left_frontier_barrier(proposals)
        commit_ranks = [absolute_positions.index(proposal.position) for proposal in committed]
        if barrier_action is not None:
            action_rank = absolute_positions.index(barrier_action.position)
            commit_ranks.append(action_rank)
        committed_tokens = torch.full_like(predictions, mask_id)
        for rank in commit_ranks:
            committed_tokens[rank] = predictions[rank]
        x[mask_index] = committed_tokens
        normal_count = sum(proposal.token_type == "normal" for proposal in committed)
        expand_count = delete_count = 0
        if policy == GLOBAL_CONFIDENCE:
            x_seq = x[0]
            first_eos = (x_seq == eos_id).nonzero(as_tuple=True)[0]
            if first_eos.numel():
                tail = torch.arange(x_seq.size(0), device=x.device) >= int(first_eos[0].item())
                x_seq.masked_fill_(tail & mask_index[0], eos_id)
                x = x_seq.unsqueeze(0)
            expand_indices = (x[0] == expand_id).nonzero(as_tuple=False).squeeze(1).tolist()
            for index in sorted(expand_indices, reverse=True):
                x = _apply_one_expand(x, int(index), mask_id)
                dynamic_length += 1
                expand_budget -= 1
                if x.shape[1] > maximum:
                    x = x[:, :maximum]
            expand_count = len(expand_indices)
            delete_indices = (((x[0] == eos_id) & mask_index[0]).nonzero(as_tuple=False).squeeze(1).tolist())
            for index in sorted(delete_indices, reverse=True):
                x = _apply_one_delete(x, int(index), mask_id)
                dynamic_length -= 1
            delete_count = len(delete_indices)
        elif barrier_action is not None and barrier_action.token_type == "expand":
            x = _apply_one_expand(x, barrier_action.position, mask_id)
            dynamic_length += 1
            expand_budget -= 1
            if x.shape[1] > maximum:
                x = x[:, :maximum]
            expand_count = 1
        elif barrier_action is not None and barrier_action.token_type == "delete":
            x = _apply_one_delete(x, barrier_action.position, mask_id)
            dynamic_length -= 1
            delete_count = 1
        total_normal += normal_count
        total_expand += expand_count
        total_delete += delete_count
        unresolved_after = int((x[0, prefix_length : prefix_length + dynamic_length] == mask_id).sum().item())
        traces.append(
            {
                "step_index": step,
                "active_mask_positions": local_positions,
                "requested_k": requested_k,
                "selected_candidate_count": len(proposals),
                "selected_positions": [proposal.position - prefix_length for proposal in proposals],
                "actual_committed_normal_tokens": normal_count,
                "executed_expand_count": expand_count,
                "executed_delete_count": delete_count,
                "barrier_truncated_proposals": discarded,
                "earliest_structural_action_position": None if barrier_action is None else barrier_action.position - prefix_length,
                "unresolved_after": unresolved_after,
                "canvas_length": dynamic_length,
            }
        )
    completion_ids = x[0, prefix_length : prefix_length + dynamic_length].detach().cpu().tolist()
    return {
        "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "completion_token_ids": completion_ids,
        "final_dynamic_length": dynamic_length,
        "final_remaining_masks": int(sum(token == mask_id for token in completion_ids)),
        "forward_count": len(traces),
        "normal_token_count": total_normal,
        "expand_count": total_expand,
        "delete_count": total_delete,
        "step_trace": traces,
    }
