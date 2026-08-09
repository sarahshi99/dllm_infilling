from __future__ import annotations

from collections import Counter

import pytest

from experiments.frontier_gated_dreamon.core import (
    action_name,
    eligible_mask_ranks,
    initial_middle,
)
from experiments.frontier_gated_dreamon.manifests import largest_remainder_allocation
from experiments.frontier_gated_dreamon.protocol import (
    INITIAL_MASK_COUNT,
    MAX_NEW_TOKENS,
    WIDTHS,
    stable_sample_seed,
)
from experiments.frontier_gated_dreamon.schema import validate_result_row


def test_locked_protocol_constants() -> None:
    assert INITIAL_MASK_COUNT == 64
    assert MAX_NEW_TOKENS == 64
    assert WIDTHS == (1, 4, 8, 16, None)


def test_initial_middle_is_one_contiguous_64_mask_canvas() -> None:
    middle = initial_middle(mask_id=12)
    assert middle == [12] * 64
    assert len(middle) == 64


@pytest.mark.parametrize(
    ("mask_positions", "width", "expected"),
    [
        ([2, 3, 7, 9], 1, [0]),
        ([2, 3, 7, 9], 4, [0, 1]),
        ([2, 3, 7, 9], 8, [0, 1, 2, 3]),
        ([2, 3, 7, 9], 16, [0, 1, 2, 3]),
        ([2, 3, 7, 9], None, [0, 1, 2, 3]),
    ],
)
def test_frontier_eligibility_uses_dynamic_positions(
    mask_positions: list[int], width: int | None, expected: list[int]
) -> None:
    assert eligible_mask_ranks(mask_positions, width) == expected


def test_frontier_eligibility_recomputes_after_dynamic_shift() -> None:
    assert eligible_mask_ranks([4, 8, 9], 4) == [0]
    assert eligible_mask_ranks([3, 4, 8, 9], 4) == [0, 1]


def test_newline_is_a_normal_action() -> None:
    assert action_name(token_id=198, mask_id=12, expand_id=13, delete_id=14) == "normal"


def test_expand_delete_and_mask_actions_are_only_token_identity_based() -> None:
    assert action_name(token_id=13, mask_id=12, expand_id=13, delete_id=14) == "expand"
    assert action_name(token_id=14, mask_id=12, expand_id=13, delete_id=14) == "delete"
    assert action_name(token_id=12, mask_id=12, expand_id=13, delete_id=14) == "mask_noop"


def test_largest_remainder_allocation_is_exact_and_proportional() -> None:
    sizes = {"a": 8, "b": 5, "c": 2}
    allocation = largest_remainder_allocation(sizes, total=7, seed="fixed")
    assert sum(allocation.values()) == 7
    assert all(0 <= allocation[key] <= sizes[key] for key in sizes)
    assert allocation["a"] >= allocation["b"] >= allocation["c"]


def test_largest_remainder_tie_break_is_stable() -> None:
    sizes = {"a": 1, "b": 1, "c": 1}
    first = largest_remainder_allocation(sizes, total=2, seed="fixed")
    second = largest_remainder_allocation(dict(reversed(list(sizes.items()))), total=2, seed="fixed")
    assert first == second
    assert Counter(first.values()) == Counter({1: 2, 0: 1})


def test_seed_mapping_is_width_independent() -> None:
    sample_id = "MultiLineInfilling/HumanEval/0/L0_L2"
    seeds = {stable_sample_seed(sample_id, width) for width in WIDTHS}
    assert len(seeds) == 1


def test_result_schema_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        validate_result_row({}, scored=False)


def test_result_schema_rejects_frontier_violation() -> None:
    row = {
        "manifest_id": "m",
        "manifest_role": "pilot",
        "sample_id": "s",
        "task_id": "s",
        "base_task_id": "0",
        "group_id": "0",
        "w": "4",
        "seed": 1,
        "config_hash": "c",
        "model_path": "model",
        "model_revision": "revision",
        "initial_mask_count": 64,
        "initial_dynamic_length": 64,
        "final_dynamic_length": 64,
        "completion_token_ids": [],
        "completion": "",
        "pass_at_1": None,
        "compiled": False,
        "completed": False,
        "status": "incomplete",
        "stop_reason": "forward_cap",
        "forward_count": 1,
        "wall_time_seconds": 1.0,
        "gpu_time_seconds": 1.0,
        "token_forwards": 100,
        "step_trace": [
            {
                "step": 0,
                "kind": "initial_state",
                "dynamic_length": 64,
                "unresolved_masks": 64,
            },
            {
                "step": 1,
                "kind": "commit",
                "eligible_positions": [0, 1],
                "commit_position": 3,
            },
        ],
        "normal_token_count": 1,
        "expand_count": 0,
        "delete_action_count": 0,
        "single_point_delete_count": 0,
        "broadcast_delete_count": 0,
        "broadcast_delete_tail_lengths": [],
        "newline_token_count": 0,
        "physical_line_count": 0,
        "peak_unresolved_masks": 64,
        "final_unresolved_masks": 63,
        "exact_cycle_count": 0,
        "oscillation_detected": False,
        "frontier_violation": 0,
        "exception": None,
        "evaluable": None,
        "unevaluable_reason": None,
    }
    with pytest.raises(ValueError, match="frontier violation"):
        validate_result_row(row, scored=False)
