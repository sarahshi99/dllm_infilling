from __future__ import annotations

from typing import List, Optional, Sequence, Set

import torch

from .ast_utils import StructuralUnit, middle_char_span_to_token_indices
from .config import PolicyConfig


def linear_target_masks(total_masks: int, total_steps: int, step: int) -> int:
    return int(total_masks * (total_steps - 1 - step) / total_steps)


def select_low_confidence_mask_positions(
    max_probs: torch.Tensor,
    current_mask_idx: torch.Tensor,
    target_masks: int,
) -> List[int]:
    if target_masks <= 0:
        return []

    mask_probs = max_probs.clone()
    mask_probs[~current_mask_idx] = float("inf")
    _, indices_to_mask = torch.topk(mask_probs, target_masks, dim=-1, largest=False)
    return indices_to_mask[0].tolist()


def structural_units_to_canvas_indices(
    tokenizer,
    middle_text: str,
    middle_length_tokens: int,
    units: Sequence[StructuralUnit],
) -> List[int]:
    all_indices: Set[int] = set()
    for unit in units:
        relative_indices = middle_char_span_to_token_indices(
            tokenizer=tokenizer,
            middle_text=middle_text,
            middle_length_tokens=middle_length_tokens,
            char_start=unit.char_start,
            char_end=unit.char_end,
        )
        all_indices.update(relative_indices)
    return sorted(all_indices)


# 旧实现（保留对照，不再直接使用）：
# 旧版只是做简单并集，然后按索引排序截断。
# 这样虽然不会再像更早版本那样爆预算，
# 但 semantic stage 下结构修复仍可能被低置信度 remask 稀释掉。
# def merge_policy_indices(
#     low_confidence_indices: Sequence[int],
#     structural_indices: Sequence[int],
#     max_keep: Optional[int] = None,
# ) -> List[int]:
#     merged = list(sorted(set(low_confidence_indices).union(structural_indices)))
#     if max_keep is not None:
#         return merged[:max_keep]
#     return merged

def _ordered_unique(indices: Sequence[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
    return ordered


def merge_policy_indices(
    low_confidence_indices: Sequence[int],
    structural_indices: Sequence[int],
    max_keep: Optional[int] = None,
    structural_min_keep: int = 0,
) -> List[int]:
    """
    新实现：
    1) 结构索引优先进入预算池；
    2) semantic stage 可通过 structural_min_keep 保证最小结构修复配额；
    3) 低置信度 remask 只在剩余预算里补齐。
    """
    structural_order = _ordered_unique(structural_indices)
    low_conf_order = _ordered_unique(low_confidence_indices)

    selected: List[int] = []
    selected_set = set()

    def add_many(items: Sequence[int], budget: Optional[int]) -> None:
        for idx in items:
            if idx in selected_set:
                continue
            if budget is not None and len(selected) >= budget:
                return
            selected.append(idx)
            selected_set.add(idx)

    if structural_min_keep > 0:
        add_many(structural_order[:structural_min_keep], max_keep)

    add_many(structural_order, max_keep)
    add_many(low_conf_order, max_keep)
    return selected