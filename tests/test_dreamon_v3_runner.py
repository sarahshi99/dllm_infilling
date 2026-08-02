from __future__ import annotations

import json

import pytest

from repro_scripts.dreamon_slot_generator import Method
from repro_scripts.run_dreamon_progressive_v2 import (
    TARGETED_STAGE_SIZES,
    build_method_config,
    classify_protocol_outcome,
    load_targeted_generation_population,
    stable_json_hash,
    stage_size,
    validate_resume_rows,
)


def _row(task_id: str) -> dict[str, str]:
    return {
        "task_id": task_id,
        "base_problem_id": task_id.split("/")[2],
        "prompt": "prefix",
        "suffix": "suffix",
    }


def test_v3_targeted_stage_sizes_are_separate_from_frozen_main_stages():
    assert TARGETED_STAGE_SIZES == {"cycle5": 5}
    assert stage_size("cycle5") == 5
    assert stage_size("pilot") == 30


def test_targeted_population_loader_preserves_order_and_exact_size(tmp_path):
    rows = [_row(f"MultiLineInfilling/HumanEval/{index}/L0_L2") for index in range(5)]
    path = tmp_path / "cycle5.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert load_targeted_generation_population(path, 5) == rows
    with pytest.raises(RuntimeError, match="expected 6 unique tasks"):
        load_targeted_generation_population(path, 6)


def test_targeted_population_rejects_reference_fields(tmp_path):
    row = _row("MultiLineInfilling/HumanEval/1/L0_L2")
    row["canonical_solution"] = "answer"
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden generation field"):
        load_targeted_generation_population(path, 1)


def test_v3_budgeted_config_freezes_official_budget_semantics():
    config = build_method_config(
        Method.V3_HARD_BUDGETED, "pilot", "runner-commit"
    )
    assert config["protocol_name"] == "dreamon_progressive_v3_hard_budgeted"
    assert config["protocol_version"] == 3
    assert config["initial_expand_budget"] == 64
    assert config["expand_budget_scope"] == "per_sample_global"
    assert config["expand_budget_reset_per_slot"] is False
    assert config["expand_budget_refund_on_delete"] is False
    assert config["expand_budget_logit_mask_at_zero"] is True
    assert config["nonempty_guard"] is False


def test_v3_config_hash_changes_with_budget_semantics():
    budgeted = build_method_config(
        Method.V3_HARD_BUDGETED, "pilot", "runner-commit"
    )
    changed = dict(budgeted, initial_expand_budget=63)
    assert stable_json_hash(budgeted) != stable_json_hash(changed)


def test_v3_resume_rejects_v2_row():
    with pytest.raises(RuntimeError, match="method/config mismatch"):
        validate_resume_rows(
            [
                {
                    "task_id": "task",
                    "method": Method.V2_HARD_V2_BOUNDARY.value,
                    "config_hash": "v2",
                }
            ],
            Method.V3_HARD_BUDGETED,
            "v3",
            ["task"],
        )


@pytest.mark.parametrize(
    "flag", ["exact_deterministic_cycle", "forward_cap_with_unresolved_masks"]
)
def test_v3_controlled_terminal_failures_are_method_failures(flag):
    row = {
        "method": Method.V3_HARD_BUDGETED.value,
        "status": "protocol_error",
        "protocol_flags": [flag],
        "unresolved_mask_count": 2,
        "completion": "",
    }
    assert classify_protocol_outcome(row) == "terminal_method_failure"


def test_v3_invariant_failure_is_protocol_violation():
    row = {
        "method": Method.V3_HARD_BUDGETED.value,
        "status": "protocol_error",
        "protocol_flags": ["expand_budget_conservation_failed"],
        "unresolved_mask_count": 2,
        "completion": "",
    }
    assert classify_protocol_outcome(row) == "protocol_violation"
