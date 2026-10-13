"""Frozen 64-token Base transfer; no DreamOn expansion/deletion semantics.

The independent native reference keeps released Dream `_sample` code, with two
explicit patches: a fixed K transfer count, and optional left frontier positions.
EOS remains on the canvas; final text follows the existing native adapter's
skip-special then clean_middle_text rule. The source head's mask is unchanged.
"""
from __future__ import annotations

import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import torch

from experiments.dreamon_singleline_order_parallel import (
    GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER, _global_ranks,
    _left_frontier_ranks, k2_pair_decision, shift_logits_to_targets,
)


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _canvas(model: Any, tokenizer: Any, prefix_ids: list[int], suffix_ids: list[int]):
    values = [*prefix_ids, *([int(tokenizer.mask_token_id)] * 64), *suffix_ids]
    return torch.tensor([values], device=_device(model)), len(prefix_ids)


def _finish(x: torch.Tensor, start: int, tokenizer: Any, **counts) -> dict:
    ids = x[0, start:start + 64].tolist()
    if int(tokenizer.mask_token_id) in ids:
        raise RuntimeError("fixed canvas has unresolved masks")
    valid = set(tokenizer.get_vocab().values())
    if any(token not in valid for token in ids):
        raise RuntimeError("Base sampled an unassigned vocabulary column")
    # Keep the established Base adapter's final processing in every arm.
    from clean_scripts.run_dreamcoder_official_infilling import clean_middle_text
    completion = clean_middle_text(tokenizer.decode(ids, skip_special_tokens=True), tokenizer)
    return dict(completion=completion, completion_token_ids=ids,
                final_dynamic_length=64, final_remaining_masks=0,
                expand_count=0, delete_count=0, recompute_enabled_count=0,
                recompute_forward_count=0, corrected_structural_stop_count=0,
                structural_stop_count=0, **counts)


@torch.no_grad()
def decode_fixed_transfer(
    *, model: Any, tokenizer: Any, sample_tokens_fn: Any,
    prefix_ids: list[int], suffix_ids: list[int], requested_k: int,
    policy: str, row_seed: int, markov_head: Any | None = None,
    temperature: float = 0.2, top_p: float | None = 0.9,
    top_k: int | None = None, eos_penalty: float = 3.0,
    eps: float = 1e-3, steps: int = 64, capture_actions: bool = False,
) -> dict:
    if requested_k not in (1, 2) or policy not in (GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER):
        raise ValueError("only frozen K1/K2 and the two original orders are supported")
    if steps != 64:
        raise ValueError("frozen native Base schedule requires 64 steps")
    _seed(row_seed)
    x, start = _canvas(model, tokenizer, prefix_ids, suffix_ids)
    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
    mask_id = int(tokenizer.mask_token_id)
    special = set(tokenizer.all_special_ids)
    valid = set(tokenizer.get_vocab().values())
    ordinary_ids = valid - special
    calls = adjacent = nonadjacent = enabled = ordinary = special_outputs = 0
    actions = []
    for step in range(steps):
        mask = x == mask_id
        if not bool(mask.any()):
            break
        positions = mask[0].nonzero().flatten().tolist()
        logits = shift_logits_to_targets(model(x, "full", None).logits)[mask]
        calls += 1
        # Released native EOS/pad penalty, with the unchanged 64-step clock.
        logits[:, int(tokenizer.pad_token_id)] += eos_penalty * torch.log(1 - timesteps[step] + eps)
        confidence, predictions = sample_tokens_fn(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
        if policy == GLOBAL_CONFIDENCE:
            # Native topk includes fixed context at -inf. This matters for ties:
            # topk over the compressed mask vector can choose other positions.
            full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype)
            full_confidence[mask] = confidence
            absolute = _global_ranks(full_confidence[0], min(requested_k, len(positions)))
            ranks = [positions.index(position) for position in absolute]
        else:
            ranks = _left_frontier_ranks(positions, requested_k=requested_k)
        pair = k2_pair_decision(selected_ranks=ranks, absolute_positions=positions, policy=policy)
        adjacent += int(pair.markov_eligible)
        nonadjacent += int(len(ranks) == 2 and not pair.markov_eligible)
        if markov_head is not None and pair.markov_eligible:
            left, right = pair.left_rank, pair.right_rank
            left_id, right_id = int(predictions[left]), int(predictions[right])
            if left_id in ordinary_ids and right_id in ordinary_ids:
                local_seed = (row_seed + 1_000_003 * step + 10_007 * positions[left] + positions[right]) % ((1 << 63) - 1)
                devices = [x.device.index or 0] if x.is_cuda else []
                with torch.random.fork_rng(devices=devices, enabled=True):
                    _seed(local_seed)
                    previous = torch.tensor([left_id], device=x.device)
                    correction = markov_head(previous)[0].to(dtype=logits.dtype)
                    _, corrected = sample_tokens_fn(
                        logits[right:right + 1] + correction, temperature=temperature,
                        top_p=top_p, top_k=top_k, neg_entropy=True)
                predictions[right] = corrected[0]
                enabled += 1
        submitted = []
        for rank in ranks:
            token = int(predictions[rank])
            if token not in valid or token == mask_id:
                raise RuntimeError(f"invalid fixed-canvas output token {token} at step {step}")
            x[0, positions[rank]] = token
            ordinary += int(token not in special)
            special_outputs += int(token in special)
            submitted.append([positions[rank], token])
        if capture_actions:
            actions.append(dict(step=step, submitted=submitted, head_eligible=pair.markov_eligible))
    return _finish(x, start, tokenizer, forward_count=calls, base_forward_count=calls,
                   outer_round_count=calls, adjacent_opportunity_count=adjacent,
                   nonadjacent_fallback_count=nonadjacent, markov_enabled_count=enabled,
                   normal_token_count=ordinary, special_output_count=special_outputs,
                   action_trace=actions)


@torch.no_grad()
def native_fixed_reference(
    *, model: Any, tokenizer: Any, prefix_ids: list[int], suffix_ids: list[int],
    requested_k: int, policy: str, row_seed: int, temperature: float = 0.2,
    top_p: float | None = 0.9, top_k: int | None = None,
    eos_penalty: float = 3.0, eps: float = 1e-3, steps: int = 64,
    sample_tokens_fn: Any | None = None,
) -> dict:
    """Released native `_sample` plus explicit fixed-K/order patch, no head.

    No global file or model method is overwritten. `max_length` equals the
    existing prefix+64+suffix canvas, so no additional masks are appended.
    """
    if requested_k not in (1, 2) or policy not in (GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER) or steps != 64:
        raise ValueError("reference requires frozen K1/K2, order and 64-step settings")
    native_model = model if hasattr(model, "_sample") else model.model
    original = type(native_model)._sample
    source = textwrap.dedent(inspect.getsource(original))
    schedule = "number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)"
    if source.count(schedule) != 1:
        raise RuntimeError("native reference schedule differs from audited release")
    source = source.replace(schedule, f"number_transfer_tokens = min({requested_k}, int(num_mask_token))")
    # The entropy branch's selection is the only place needing frontier patch.
    marker = "x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id"
    if source.count(marker) != 2:
        raise RuntimeError("native reference selection differs from audited release")
    if policy == LEFT_TO_RIGHT_FRONTIER:
        index = source.rindex(marker)
        source = source[:index] + "transfer_index = mask_index[0].nonzero().flatten()[:number_transfer_tokens].unsqueeze(0)\n                " + source[index:]
    namespace = dict(original.__globals__)
    exec(compile(source, "<native-fixed-K-reference>", "exec"), namespace)
    x, start = _canvas(model, tokenizer, prefix_ids, suffix_ids)
    config = SimpleNamespace(output_history=False, return_dict_in_generate=True,
                             max_length=x.shape[1], mask_token_id=int(tokenizer.mask_token_id),
                             pad_token_id=int(tokenizer.pad_token_id), steps=steps,
                             eps=eps, alg="entropy", alg_temp=0.0, eos_penalty=eos_penalty,
                             temperature=temperature, top_p=top_p, top_k=top_k)
    _seed(row_seed)
    output = namespace[original.__name__](native_model, x, None, config,
                                           lambda step, x, logits: x,
                                           lambda step, x, logits: logits)
    return _finish(output.sequences, start, tokenizer)
