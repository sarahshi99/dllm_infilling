from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch


@dataclass
class GlobalGapStoppingConfig:
    min_stop_step: int = 32
    gap_threshold: float = 0.50
    top1_threshold: float = 0.70
    max_remaining_mask_ratio: float = 0.50


@dataclass
class StoppingDecision:
    should_stop: bool
    reason: str
    step: int
    remaining_masks: int
    remaining_mask_ratio: float
    mean_gap: Optional[float]
    min_gap: Optional[float]
    mean_top1: Optional[float]
    max_top1: Optional[float]
    min_top1: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_global_gap_stopping_config(cfg) -> GlobalGapStoppingConfig:
    return GlobalGapStoppingConfig(
        min_stop_step=int(getattr(cfg.decode, "stop_min_step", 32)),
        gap_threshold=float(getattr(cfg.decode, "stop_gap_threshold", 0.50)),
        top1_threshold=float(getattr(cfg.decode, "stop_top1_threshold", 0.70)),
        max_remaining_mask_ratio=float(getattr(cfg.decode, "stop_max_remaining_mask_ratio", 0.50)),
    )


def validate_stopping_config(stop_cfg: GlobalGapStoppingConfig, total_steps: int) -> None:
    if stop_cfg.min_stop_step < 0:
        raise ValueError(f"stop_min_step must be non-negative, got {stop_cfg.min_stop_step}")
    if stop_cfg.min_stop_step >= total_steps:
        raise ValueError(
            f"stop_min_step must be smaller than total_steps. "
            f"got stop_min_step={stop_cfg.min_stop_step}, total_steps={total_steps}"
        )
    if not (0.0 <= stop_cfg.gap_threshold <= 1.0):
        raise ValueError(f"gap_threshold must be in [0, 1], got {stop_cfg.gap_threshold}")
    if not (0.0 <= stop_cfg.top1_threshold <= 1.0):
        raise ValueError(f"top1_threshold must be in [0, 1], got {stop_cfg.top1_threshold}")
    if not (0.0 <= stop_cfg.max_remaining_mask_ratio <= 1.0):
        raise ValueError(
            f"max_remaining_mask_ratio must be in [0, 1], got {stop_cfg.max_remaining_mask_ratio}"
        )


def evaluate_global_gap_stopping(
    *,
    step: int,
    middle_probs: torch.Tensor,
    middle_mask_idx: torch.Tensor,
    selected_mask_length: int,
    stop_cfg: GlobalGapStoppingConfig,
) -> StoppingDecision:
    if selected_mask_length <= 0:
        raise ValueError(f"selected_mask_length must be positive, got {selected_mask_length}")

    remaining_masks = int(middle_mask_idx.sum().item())
    remaining_mask_ratio = remaining_masks / float(selected_mask_length)

    if remaining_masks == 0:
        return StoppingDecision(
            should_stop=False,
            reason="no_remaining_masks",
            step=step,
            remaining_masks=0,
            remaining_mask_ratio=0.0,
            mean_gap=None,
            min_gap=None,
            mean_top1=None,
            max_top1=None,
            min_top1=None,
        )

    top2_values, _ = torch.topk(middle_probs, k=2, dim=-1)
    top1_probs = top2_values[..., 0]
    top2_probs = top2_values[..., 1]
    gaps = top1_probs - top2_probs

    masked_top1 = top1_probs[middle_mask_idx]
    masked_gaps = gaps[middle_mask_idx]

    mean_gap = float(masked_gaps.mean().item())
    min_gap = float(masked_gaps.min().item())
    mean_top1 = float(masked_top1.mean().item())
    max_top1 = float(masked_top1.max().item())
    min_top1 = float(masked_top1.min().item())

    if step < stop_cfg.min_stop_step:
        reason = "before_min_stop_step"
        should_stop = False
    elif remaining_mask_ratio > stop_cfg.max_remaining_mask_ratio:
        reason = "too_many_remaining_masks"
        should_stop = False
    elif mean_gap < stop_cfg.gap_threshold:
        reason = "mean_gap_below_threshold"
        should_stop = False
    elif mean_top1 < stop_cfg.top1_threshold:
        reason = "mean_top1_below_threshold"
        should_stop = False
    else:
        reason = "global_gap_early_commit"
        should_stop = True

    return StoppingDecision(
        should_stop=should_stop,
        reason=reason,
        step=step,
        remaining_masks=remaining_masks,
        remaining_mask_ratio=remaining_mask_ratio,
        mean_gap=mean_gap,
        min_gap=min_gap,
        mean_top1=mean_top1,
        max_top1=max_top1,
        min_top1=min_top1,
    )