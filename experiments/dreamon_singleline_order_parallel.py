from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


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
