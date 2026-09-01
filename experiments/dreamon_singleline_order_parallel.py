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


@dataclass(frozen=True)
class PreparedDistribution:
    probabilities: torch.Tensor
    filtered_logits: torch.Tensor
    retained_mask: torch.Tensor


@dataclass
class PendingTransition:
    chain_id: int
    offset: int
    source_step_index: int
    source_position: int
    target_position: int
    remaining_predecessors: int
    next_predecessor_position: int
    source_active_mask_count: int
    source_canvas_length: int
    stale_raw_probabilities: torch.Tensor
    stale_actual_probabilities: torch.Tensor
    stale_raw_reference_log_probability: float | None
    stale_actual_reference_log_probability: float | None
    stale_actual_reference_retained: bool | None
    reference_evaluable: bool
    reference_non_evaluable_reason: str | None
    reference_token_id: int | None
    stale_global_rank_fields: dict[str, Any]
    neighbor_observation_index: int | None = None


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


def prepare_decode_distribution(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> PreparedDistribution:
    """Reproduce released sample_tokens filtering without sampling or RNG use."""
    filtered = logits.clone()
    retained = torch.ones_like(filtered, dtype=torch.bool)
    if temperature > 0:
        filtered = filtered / temperature
    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        remove = torch.zeros_like(filtered, dtype=torch.bool).scatter_(
            -1, sorted_indices, sorted_indices_to_remove
        )
        retained &= ~remove
        filtered = filtered.masked_fill(remove, torch.finfo(filtered.dtype).min)
    if top_k is not None:
        safe_top_k = min(int(top_k), int(filtered.size(-1)))
        remove = filtered < torch.topk(filtered, safe_top_k)[0][..., -1, None]
        retained &= ~remove
        filtered = filtered.masked_fill(remove, torch.finfo(filtered.dtype).min)
    return PreparedDistribution(
        probabilities=torch.softmax(filtered, dim=-1),
        filtered_logits=filtered,
        retained_mask=retained,
    )


def _distribution_summary(probabilities: torch.Tensor) -> dict[str, float | int]:
    values, token_ids = torch.topk(probabilities, min(5, int(probabilities.numel())))
    float_probabilities = probabilities.float()
    entropy = -(
        float_probabilities
        * torch.where(
            float_probabilities > 0,
            float_probabilities.log(),
            torch.zeros_like(float_probabilities),
        )
    ).sum()
    return {
        "entropy": float(entropy.item()),
        "margin": float((values[0] - values[1]).item()) if values.numel() > 1 else float(values[0].item()),
        "top1_token_id": int(token_ids[0].item()),
        "top1_probability": float(values[0].item()),
        "top_token_ids": [int(token_id) for token_id in token_ids.detach().cpu().tolist()],
        "top_probabilities": [float(value) for value in values.detach().cpu().tolist()],
    }


def _token_rank(probabilities: torch.Tensor, token_id: int) -> int:
    value = probabilities[int(token_id)]
    return int((probabilities > value).sum().item()) + 1


def _stable_log_probability(logits: torch.Tensor, token_id: int) -> float:
    values = logits.float()
    return float((values[int(token_id)] - torch.logsumexp(values, dim=-1)).item())


def _global_rank_fields(confidence: torch.Tensor, rank: int) -> dict[str, Any]:
    target = confidence[int(rank)]
    higher = int((confidence > target).sum().item())
    equal_before = int((confidence[: int(rank)] == target).sum().item())
    top2 = set(_global_ranks(confidence, min(2, int(confidence.numel()))))
    top4 = set(_global_ranks(confidence, min(4, int(confidence.numel()))))
    return {
        "global_rank": higher + equal_before + 1,
        "global_confidence": float(target.item()),
        "global_confidence_tie_count": int((confidence == target).sum().item()),
        "in_top2": int(rank) in top2,
        "in_top4": int(rank) in top4,
    }


def _delta_with_infinities(
    stale: float | None, fresh: float | None
) -> tuple[float | None, str]:
    if stale is not None and fresh is not None:
        return fresh - stale, "finite"
    if stale is None and fresh is not None:
        return None, "positive_infinity"
    if stale is not None and fresh is None:
        return None, "negative_infinity"
    return None, "undefined_both_truncated"


def _distribution_pair_metrics(
    *,
    name: str,
    stale_probabilities: torch.Tensor,
    fresh_probabilities: torch.Tensor,
    reference_token_id: int | None,
    stale_reference_log_probability: float | None,
    fresh_reference_log_probability: float | None,
    stale_reference_retained: bool | None,
    fresh_reference_retained: bool | None,
) -> dict[str, Any]:
    stale_summary = _distribution_summary(stale_probabilities)
    fresh_summary = _distribution_summary(fresh_probabilities)
    stale_top1 = int(stale_summary["top1_token_id"])
    fresh_top1 = int(fresh_summary["top1_token_id"])
    payload: dict[str, Any] = {
        f"{name}_total_variation": float(
            0.5
            * torch.abs(stale_probabilities.float() - fresh_probabilities.float()).sum().item()
        ),
        f"{name}_stale_top1_token_id": stale_top1,
        f"{name}_fresh_top1_token_id": fresh_top1,
        f"{name}_top1_agreement": stale_top1 == fresh_top1,
        f"{name}_stale_entropy": stale_summary["entropy"],
        f"{name}_fresh_entropy": fresh_summary["entropy"],
        f"{name}_stale_margin": stale_summary["margin"],
        f"{name}_fresh_margin": fresh_summary["margin"],
        f"{name}_fresh_top1_stale_probability": float(
            stale_probabilities[fresh_top1].item()
        ),
        f"{name}_fresh_top1_stale_rank": _token_rank(stale_probabilities, fresh_top1),
        f"{name}_stale_top_token_ids": stale_summary["top_token_ids"],
        f"{name}_stale_top_probabilities": stale_summary["top_probabilities"],
        f"{name}_fresh_top_token_ids": fresh_summary["top_token_ids"],
        f"{name}_fresh_top_probabilities": fresh_summary["top_probabilities"],
    }
    if reference_token_id is None:
        return payload
    stale_reference_probability = float(stale_probabilities[reference_token_id].item())
    fresh_reference_probability = float(fresh_probabilities[reference_token_id].item())
    delta, delta_kind = _delta_with_infinities(
        stale_reference_log_probability, fresh_reference_log_probability
    )
    payload.update(
        {
            f"{name}_stale_reference_probability": stale_reference_probability,
            f"{name}_fresh_reference_probability": fresh_reference_probability,
            f"{name}_stale_reference_log_probability": stale_reference_log_probability,
            f"{name}_fresh_reference_log_probability": fresh_reference_log_probability,
            f"{name}_reference_log_probability_delta": delta,
            f"{name}_reference_log_probability_delta_kind": delta_kind,
            f"{name}_stale_reference_rank": _token_rank(
                stale_probabilities, reference_token_id
            ),
            f"{name}_fresh_reference_rank": _token_rank(
                fresh_probabilities, reference_token_id
            ),
            f"{name}_reference_rank_improvement": _token_rank(
                stale_probabilities, reference_token_id
            )
            - _token_rank(fresh_probabilities, reference_token_id),
            f"{name}_stale_top1_is_reference": stale_top1 == reference_token_id,
            f"{name}_fresh_top1_is_reference": fresh_top1 == reference_token_id,
            f"{name}_wrong_to_correct": stale_top1 != reference_token_id
            and fresh_top1 == reference_token_id,
            f"{name}_correct_to_wrong": stale_top1 == reference_token_id
            and fresh_top1 != reference_token_id,
            f"{name}_stale_reference_retained_in_top_p": stale_reference_retained,
            f"{name}_fresh_reference_retained_in_top_p": fresh_reference_retained,
        }
    )
    return payload


def _aborted_transition(pending: PendingTransition, reason: str, step: int) -> dict[str, Any]:
    return {
        "record_type": "stale_fresh_transition",
        "chain_id": pending.chain_id,
        "offset": pending.offset,
        "source_step_index": pending.source_step_index,
        "fresh_step_index": step,
        "source_position": pending.source_position,
        "target_position": pending.target_position,
        "source_active_mask_count": pending.source_active_mask_count,
        "source_canvas_length": pending.source_canvas_length,
        "transition_eligible": False,
        "transition_ineligible_reason": reason,
        "reference_evaluable": pending.reference_evaluable,
        "reference_non_evaluable_reason": pending.reference_non_evaluable_reason,
    }


@torch.inference_mode()
def decode_with_policy(
    *,
    model: Any,
    tokenizer: Any,
    sample_tokens_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    reference_ids: Sequence[int] | None = None,
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
    reference = None if reference_ids is None else list(map(int, reference_ids))
    markov_pending: list[PendingTransition] = []
    markov_transitions: list[dict[str, Any]] = []
    global_neighbor_observations: list[dict[str, Any]] = []
    replay_transitions: list[dict[str, Any]] = []
    replay_exclusion_counts: dict[str, int] = {}
    next_chain_id = 0
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
        stale_canvas_before = x[0, :real_length].detach().cpu().tolist()
        raw_logits = model(x, pairwise_attention, tok_idx).logits
        logits = shift_logits_to_targets(raw_logits)[mask_index]
        if real_length == maximum or expand_budget == 0:
            logits[:, expand_id] -= 1e9
        confidence, predictions = sample_tokens_fn(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True
        )
        raw_probabilities = torch.softmax(logits.float(), dim=-1)
        actual = prepare_decode_distribution(
            logits, temperature=temperature, top_p=top_p, top_k=top_k
        )
        still_pending: list[PendingTransition] = []
        for pending in markov_pending:
            if pending.remaining_predecessors != 0:
                still_pending.append(pending)
                continue
            target = pending.target_position
            if target not in absolute_positions:
                markov_transitions.append(
                    _aborted_transition(pending, "target_not_active_on_fresh_forward", step)
                )
                if pending.neighbor_observation_index is not None:
                    global_neighbor_observations[pending.neighbor_observation_index].update(
                        {
                            "fresh_global_rank_available": False,
                            "fresh_global_rank_unavailable_reason": "target_not_active_on_fresh_forward",
                        }
                    )
                continue
            target_rank = absolute_positions.index(target)
            fresh_rank_fields = _global_rank_fields(confidence, target_rank)
            target_local = target - prefix_length
            predecessor_local = target_local - 1
            reference_evaluable = pending.reference_evaluable
            reference_token_id = pending.reference_token_id
            fresh_raw_reference_log_probability = None
            fresh_actual_reference_log_probability = None
            fresh_actual_reference_retained = None
            predecessor_correct = False
            prefix_aligned = False
            if reference_evaluable and reference is not None and reference_token_id is not None:
                fresh_raw_reference_log_probability = _stable_log_probability(
                    logits[target_rank], reference_token_id
                )
                fresh_actual_reference_retained = bool(
                    actual.retained_mask[target_rank, reference_token_id].item()
                )
                if fresh_actual_reference_retained:
                    fresh_actual_reference_log_probability = _stable_log_probability(
                        actual.filtered_logits[target_rank], reference_token_id
                    )
                predecessor_token = int(x[0, prefix_length + predecessor_local].item())
                predecessor_correct = predecessor_token == int(reference[predecessor_local])
                generated_prefix = x[
                    0, prefix_length : prefix_length + predecessor_local + 1
                ].detach().cpu().tolist()
                prefix_aligned = generated_prefix == reference[: predecessor_local + 1]
            row: dict[str, Any] = {
                "record_type": "stale_fresh_transition",
                "chain_id": pending.chain_id,
                "offset": pending.offset,
                "source_step_index": pending.source_step_index,
                "fresh_step_index": step,
                "source_position": pending.source_position,
                "target_position": target_local,
                "source_active_mask_count": pending.source_active_mask_count,
                "source_canvas_length": pending.source_canvas_length,
                "transition_eligible": True,
                "transition_ineligible_reason": None,
                "reference_evaluable": reference_evaluable,
                "reference_non_evaluable_reason": pending.reference_non_evaluable_reason,
                "target_reference_token_id": reference_token_id,
                "predecessor_token_matches_reference": predecessor_correct
                if reference_evaluable
                else None,
                "reference_prefix_aligned_through_predecessor": prefix_aligned
                if reference_evaluable
                else None,
                "stale_global_rank": pending.stale_global_rank_fields["global_rank"],
                "fresh_global_rank": fresh_rank_fields["global_rank"],
                "stale_global_confidence": pending.stale_global_rank_fields[
                    "global_confidence"
                ],
                "fresh_global_confidence": fresh_rank_fields["global_confidence"],
                "stale_global_confidence_tie_count": pending.stale_global_rank_fields[
                    "global_confidence_tie_count"
                ],
                "fresh_global_confidence_tie_count": fresh_rank_fields[
                    "global_confidence_tie_count"
                ],
                "stale_in_top2": pending.stale_global_rank_fields["in_top2"],
                "fresh_in_top2": fresh_rank_fields["in_top2"],
                "stale_in_top4": pending.stale_global_rank_fields["in_top4"],
                "fresh_in_top4": fresh_rank_fields["in_top4"],
                "global_rank_improvement": pending.stale_global_rank_fields["global_rank"]
                - fresh_rank_fields["global_rank"],
                "outside_to_top2": not pending.stale_global_rank_fields["in_top2"]
                and fresh_rank_fields["in_top2"],
                "outside_to_top4": not pending.stale_global_rank_fields["in_top4"]
                and fresh_rank_fields["in_top4"],
                "top2_retained": pending.stale_global_rank_fields["in_top2"]
                and fresh_rank_fields["in_top2"],
                "top4_retained": pending.stale_global_rank_fields["in_top4"]
                and fresh_rank_fields["in_top4"],
            }
            row.update(
                _distribution_pair_metrics(
                    name="raw",
                    stale_probabilities=pending.stale_raw_probabilities,
                    fresh_probabilities=raw_probabilities[target_rank],
                    reference_token_id=reference_token_id,
                    stale_reference_log_probability=pending.stale_raw_reference_log_probability,
                    fresh_reference_log_probability=fresh_raw_reference_log_probability,
                    stale_reference_retained=True if reference_evaluable else None,
                    fresh_reference_retained=True if reference_evaluable else None,
                )
            )
            row.update(
                _distribution_pair_metrics(
                    name="actual",
                    stale_probabilities=pending.stale_actual_probabilities,
                    fresh_probabilities=actual.probabilities[target_rank],
                    reference_token_id=reference_token_id,
                    stale_reference_log_probability=pending.stale_actual_reference_log_probability,
                    fresh_reference_log_probability=fresh_actual_reference_log_probability,
                    stale_reference_retained=pending.stale_actual_reference_retained,
                    fresh_reference_retained=fresh_actual_reference_retained,
                )
            )
            markov_transitions.append(row)
            if pending.neighbor_observation_index is not None:
                global_neighbor_observations[pending.neighbor_observation_index].update(
                    {
                        "fresh_global_rank_available": True,
                        "fresh_global_rank_unavailable_reason": None,
                        "fresh_global_rank": fresh_rank_fields["global_rank"],
                        "fresh_in_top2": fresh_rank_fields["in_top2"],
                        "fresh_in_top4": fresh_rank_fields["in_top4"],
                        "global_rank_improvement": pending.stale_global_rank_fields[
                            "global_rank"
                        ]
                        - fresh_rank_fields["global_rank"],
                        "outside_to_top2": not pending.stale_global_rank_fields["in_top2"]
                        and fresh_rank_fields["in_top2"],
                        "outside_to_top4": not pending.stale_global_rank_fields["in_top4"]
                        and fresh_rank_fields["in_top4"],
                        "top2_retained": pending.stale_global_rank_fields["in_top2"]
                        and fresh_rank_fields["in_top2"],
                        "top4_retained": pending.stale_global_rank_fields["in_top4"]
                        and fresh_rank_fields["in_top4"],
                    }
                )
        markov_pending = still_pending
        top4_ranks = _global_ranks(confidence, min(4, len(absolute_positions)))
        top4 = [
            {
                "position": int(absolute_positions[rank] - prefix_length),
                "token_id": int(predictions[rank].item()),
                "confidence": float(confidence[rank].item()),
                "token_type": token_type(int(predictions[rank].item()), eos_id=eos_id, expand_id=expand_id, mask_id=mask_id),
            }
            for rank in top4_ranks
        ]
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
        broadcast_delete = policy == GLOBAL_CONFIDENCE or (
            barrier_action is not None and barrier_action.token_type == "delete"
        )
        if broadcast_delete:
            x_seq = x[0]
            first_eos = (x_seq == eos_id).nonzero(as_tuple=True)[0]
            if first_eos.numel():
                tail = torch.arange(x_seq.size(0), device=x.device) >= int(first_eos[0].item())
                x_seq.masked_fill_(tail & mask_index[0], eos_id)
                x = x_seq.unsqueeze(0)
        if policy == GLOBAL_CONFIDENCE:
            expand_indices = (x[0] == expand_id).nonzero(as_tuple=False).squeeze(1).tolist()
            for index in sorted(expand_indices, reverse=True):
                x = _apply_one_expand(x, int(index), mask_id)
                dynamic_length += 1
                expand_budget -= 1
                if x.shape[1] > maximum:
                    x = x[:, :maximum]
            expand_count = len(expand_indices)
        elif barrier_action is not None and barrier_action.token_type == "expand":
            x = _apply_one_expand(x, barrier_action.position, mask_id)
            dynamic_length += 1
            expand_budget -= 1
            if x.shape[1] > maximum:
                x = x[:, :maximum]
            expand_count = 1
        if broadcast_delete:
            delete_indices = (
                ((x[0] == eos_id) & mask_index[0])
                .nonzero(as_tuple=False)
                .squeeze(1)
                .tolist()
            )
            for index in sorted(delete_indices, reverse=True):
                x = _apply_one_delete(x, int(index), mask_id)
                dynamic_length -= 1
            delete_count = len(delete_indices)
        total_normal += normal_count
        total_expand += expand_count
        total_delete += delete_count
        ordinary_commit = normal_count == 1 and barrier_action is None and not expand_count and not delete_count
        committed_position = committed[0].position if ordinary_commit else None
        if requested_k == 1 and ordinary_commit and committed_position is not None:
            target_position = committed_position + 1
            fresh_canvas = x[0, :real_length].detach().cpu().tolist()
            if (
                target_position in absolute_positions
                and int(fresh_canvas[target_position]) == mask_id
                and len(fresh_canvas) == len(stale_canvas_before)
            ):
                predecessor_local = committed_position - prefix_length
                target_local = target_position - prefix_length
                reference_token_id = -1
                predecessor_matches_reference = False
                reference_prefix_aligned = False
                if reference is not None and 0 <= target_local < len(reference):
                    reference_token_id = int(reference[target_local])
                    predecessor_matches_reference = (
                        0 <= predecessor_local < len(reference)
                        and int(fresh_canvas[committed_position])
                        == int(reference[predecessor_local])
                    )
                    generated_prefix = fresh_canvas[
                        prefix_length : committed_position + 1
                    ]
                    reference_prefix_aligned = (
                        generated_prefix == reference[: predecessor_local + 1]
                    )
                active_count = len(absolute_positions)
                generation_stage = (
                    "early" if active_count > 42 else "middle" if active_count > 21 else "late"
                )
                replay_transitions.append(
                    {
                        "record_type": "replay_transition",
                        "source_step_index": step,
                        "trajectory_policy": policy,
                        "stale_input_ids": stale_canvas_before,
                        "fresh_input_ids": fresh_canvas,
                        "previous_token_id": int(fresh_canvas[committed_position]),
                        "target_position": int(target_position),
                        "target_local_position": int(target_local),
                        "reference_token_id": reference_token_id,
                        "predecessor_token_matches_reference": predecessor_matches_reference,
                        "reference_prefix_aligned_through_predecessor": reference_prefix_aligned,
                        "active_mask_count": active_count,
                        "generation_stage": generation_stage,
                        "canvas_length": real_length,
                        "expand_action_masked": real_length == maximum or expand_budget == 0,
                    }
                )
            else:
                replay_exclusion_counts["right_neighbor_not_fresh_active_mask"] = (
                    replay_exclusion_counts.get("right_neighbor_not_fresh_active_mask", 0) + 1
                )
        elif requested_k == 1:
            reason = (
                "structural_or_coordinate_change"
                if barrier_action is not None or expand_count or delete_count
                else "non_single_ordinary_commit"
            )
            replay_exclusion_counts[reason] = replay_exclusion_counts.get(reason, 0) + 1
        if policy == GLOBAL_CONFIDENCE and ordinary_commit and committed_position is not None:
            right_neighbor = committed_position + 1
            observation: dict[str, Any] = {
                "record_type": "global_right_neighbor_observation",
                "source_step_index": step,
                "source_position": committed_position - prefix_length,
                "source_active_mask_count": len(absolute_positions),
                "source_canvas_length": dynamic_length,
                "committed_position_is_ordinary": True,
                "right_neighbor_still_mask": right_neighbor in absolute_positions,
                "left_to_right_contiguous_short_chain": right_neighbor in absolute_positions,
                "source_right_neighbor_in_top2": False,
                "source_right_neighbor_in_top4": False,
                "fresh_global_rank_available": False,
                "fresh_global_rank_unavailable_reason": "right_neighbor_not_active_mask",
            }
            if right_neighbor in absolute_positions:
                right_rank = absolute_positions.index(right_neighbor)
                right_fields = _global_rank_fields(confidence, right_rank)
                observation.update(
                    {
                        "source_right_neighbor_global_rank": right_fields["global_rank"],
                        "source_right_neighbor_in_top2": right_fields["in_top2"],
                        "source_right_neighbor_in_top4": right_fields["in_top4"],
                        "fresh_global_rank_unavailable_reason": "pending_fresh_forward",
                    }
                )
            global_neighbor_observations.append(observation)
        if requested_k == 1 and ordinary_commit and committed_position is not None:
            committed_position = committed[0].position
            advanced_pending: list[PendingTransition] = []
            for pending in markov_pending:
                if pending.next_predecessor_position == committed_position:
                    pending.remaining_predecessors -= 1
                    pending.next_predecessor_position = committed_position + 1
                    advanced_pending.append(pending)
                else:
                    markov_transitions.append(
                        _aborted_transition(pending, "gap_or_nonconsecutive_commit", step)
                    )
                    if pending.neighbor_observation_index is not None:
                        global_neighbor_observations[pending.neighbor_observation_index].update(
                            {
                                "fresh_global_rank_available": False,
                                "fresh_global_rank_unavailable_reason": "gap_or_nonconsecutive_commit",
                            }
                        )
            markov_pending = advanced_pending
            for offset in (1, 2, 3):
                target = committed_position + offset
                if target not in absolute_positions:
                    continue
                target_rank = absolute_positions.index(target)
                target_local = target - prefix_length
                reference_evaluable = reference is not None and target_local < len(reference)
                reference_reason = None
                reference_token_id = None
                stale_raw_reference_log_probability = None
                stale_actual_reference_log_probability = None
                stale_actual_reference_retained = None
                if reference is None:
                    reference_reason = "reference_not_provided"
                elif target_local >= len(reference):
                    reference_reason = "target_reference_token_out_of_range"
                else:
                    reference_token_id = int(reference[target_local])
                    stale_raw_reference_log_probability = _stable_log_probability(
                        logits[target_rank], reference_token_id
                    )
                    stale_actual_reference_retained = bool(
                        actual.retained_mask[target_rank, reference_token_id].item()
                    )
                    if stale_actual_reference_retained:
                        stale_actual_reference_log_probability = _stable_log_probability(
                            actual.filtered_logits[target_rank], reference_token_id
                        )
                neighbor_index = None
                if policy == GLOBAL_CONFIDENCE and offset == 1:
                    neighbor_index = len(global_neighbor_observations) - 1
                markov_pending.append(
                    PendingTransition(
                        chain_id=next_chain_id,
                        offset=offset,
                        source_step_index=step,
                        source_position=committed_position - prefix_length,
                        target_position=target,
                        remaining_predecessors=offset - 1,
                        next_predecessor_position=committed_position + 1,
                        source_active_mask_count=len(absolute_positions),
                        source_canvas_length=dynamic_length,
                        stale_raw_probabilities=raw_probabilities[target_rank].detach().clone(),
                        stale_actual_probabilities=actual.probabilities[target_rank]
                        .detach()
                        .clone(),
                        stale_raw_reference_log_probability=stale_raw_reference_log_probability,
                        stale_actual_reference_log_probability=stale_actual_reference_log_probability,
                        stale_actual_reference_retained=stale_actual_reference_retained,
                        reference_evaluable=reference_evaluable,
                        reference_non_evaluable_reason=reference_reason,
                        reference_token_id=reference_token_id,
                        stale_global_rank_fields=_global_rank_fields(confidence, target_rank),
                        neighbor_observation_index=neighbor_index,
                    )
                )
                next_chain_id += 1
        else:
            for pending in markov_pending:
                markov_transitions.append(
                    _aborted_transition(pending, "structural_or_coordinate_change", step)
                )
                if pending.neighbor_observation_index is not None:
                    global_neighbor_observations[pending.neighbor_observation_index].update(
                        {
                            "fresh_global_rank_available": False,
                            "fresh_global_rank_unavailable_reason": "structural_or_coordinate_change",
                        }
                    )
            markov_pending = []
        unresolved_after = int((x[0, prefix_length : prefix_length + dynamic_length] == mask_id).sum().item())
        traces.append(
            {
                "step_index": step,
                "active_mask_positions": local_positions,
                "confidence_top4": top4,
                "leftmost_contiguous_positions": left_frontier_positions(local_positions, requested_k=4),
                "requested_k": requested_k,
                "selected_candidate_count": len(proposals),
                "selected_positions": [proposal.position - prefix_length for proposal in proposals],
                "selected_token_ids": [proposal.token_id for proposal in proposals],
                "selected_token_types": [proposal.token_type for proposal in proposals],
                "actual_committed_normal_tokens": normal_count,
                "executed_expand_count": expand_count,
                "executed_delete_count": delete_count,
                "barrier_truncated_proposals": discarded,
                "earliest_structural_action_position": None if barrier_action is None else barrier_action.position - prefix_length,
                "unresolved_after": unresolved_after,
                "canvas_length": dynamic_length,
                "canvas_state_after": x[
                    0, prefix_length : prefix_length + dynamic_length
                ].detach().cpu().tolist(),
            }
        )
    for pending in markov_pending:
        markov_transitions.append(
            _aborted_transition(pending, "case_terminated_before_fresh_forward", len(traces))
        )
        if pending.neighbor_observation_index is not None:
            global_neighbor_observations[pending.neighbor_observation_index].update(
                {
                    "fresh_global_rank_available": False,
                    "fresh_global_rank_unavailable_reason": "case_terminated_before_fresh_forward",
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
        "markov_transitions": markov_transitions,
        "global_neighbor_observations": global_neighbor_observations,
        "replay_transitions": replay_transitions,
        "replay_exclusion_counts": replay_exclusion_counts,
    }
