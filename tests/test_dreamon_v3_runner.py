from __future__ import annotations

import json

import pytest

from repro_scripts.dreamon_slot_generator import Method
from repro_scripts.run_dreamon_progressive_v2 import (
    TARGETED_PREFIX_STAGES,
    TARGETED_STAGE_SIZES,
    build_method_config,
    classify_protocol_outcome,
    load_stage_population,
    load_targeted_generation_population,
    nonempty_isolation_audit,
    pure_newline_isolation_audit,
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
    assert TARGETED_STAGE_SIZES == {"cycle5": 5, "affected6": 6, "pure30": 30}
    assert TARGETED_PREFIX_STAGES == {
        "v3c_smoke": {"population_rows": 30, "selected_rows": 5}
    }
    assert stage_size("cycle5") == 5
    assert stage_size("v3c_smoke") == 5
    assert stage_size("pure30") == 30
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


def test_v3c_smoke_selects_first_five_from_sealed_pure30(tmp_path):
    rows = [_row(f"MultiLineInfilling/HumanEval/{index}/L0_L2") for index in range(30)]
    path = tmp_path / "pure30.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert load_stage_population(path, "v3c_smoke") == rows[:5]
    assert load_stage_population(path, "pure30") == rows


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


def test_nonempty_config_differs_only_by_identity_and_guard():
    budgeted = build_method_config(
        Method.V3_HARD_BUDGETED, "pilot", "runner-commit"
    )
    nonempty = build_method_config(
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE, "pilot", "runner-commit"
    )
    allowed_differences = {"protocol_name", "method", "nonempty_guard"}
    assert {
        key for key in budgeted if budgeted[key] != nonempty[key]
    } == allowed_differences
    assert nonempty["nonempty_guard"] is True


def test_v3c_configs_freeze_only_the_two_authorized_modes():
    a = build_method_config(Method.V3_HARD_BUDGETED, "pilot", "runner")
    c0 = build_method_config(
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        "pure30",
        "runner",
    )
    c = build_method_config(
        Method.V3_C_BUDGETED_NONCONSUMING_BLANKLINE,
        "pure30",
        "runner",
    )
    for candidate, mode in ((c0, "oneshot_veto"), (c, "nonconsuming_blankline")):
        assert candidate["protocol_version"] == 4
        assert candidate["pure_newline_guard_mode"] == mode
        assert candidate["pure_newline_guard_budget_per_slot"] == 1
        assert candidate["pure_newline_guard_global_budget"] == 3
        assert candidate["nonempty_guard"] is False
        assert candidate["initial_expand_budget"] == a["initial_expand_budget"]
        assert candidate["max_global_middle_tokens"] == a["max_global_middle_tokens"]
        assert candidate["max_hard_slot_tokens"] == a["max_hard_slot_tokens"]
    assert stable_json_hash(c0) != stable_json_hash(c)


def test_full_stage_config_records_run_authorization_without_false_gate_label():
    full = build_method_config(
        Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO,
        "full",
        "runner",
    )
    assert full["run_authorization"] == "preregistered_full_gate_authorized"


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


def test_nonempty_no_valid_action_is_controlled_terminal_failure():
    row = {
        "method": Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE.value,
        "status": "protocol_error",
        "protocol_flags": ["nonempty_guard_no_valid_action"],
        "unresolved_mask_count": 1,
        "completion": "",
    }
    assert classify_protocol_outcome(row) == "terminal_method_failure"


def test_nonempty_isolation_audit_requires_exact_match_without_rejection():
    control = [
        {
            "task_id": "same",
            "completion": "x",
            "score": {"passed": True, "compile_passed": True},
            "total_forwards": 1,
            "step_trace": [
                {
                    "selected_position": 1,
                    "selected_region": "HARD_SLOT_0",
                    "proposal_token_id": 7,
                    "action": "normal",
                }
            ],
        }
    ]
    candidate = [
        {
            **control[0],
            "nonempty_guard_rejection_count": 0,
        }
    ]
    audit = nonempty_isolation_audit(candidate, control)
    assert audit["rows_without_guard_activation"] == 1
    assert audit["all_inactive_rows_match"] is True
    candidate[0]["completion"] = "different"
    assert nonempty_isolation_audit(candidate, control)[
        "all_inactive_rows_match"
    ] is False


def test_pure_newline_isolation_requires_compact_actions_score_and_compute_match():
    control = [
        {
            "task_id": "same",
            "method": Method.V3_HARD_BUDGETED.value,
            "completion": "x",
            "completion_token_ids": [1],
            "region_token_ids": {"HARD_SLOT_0": [1]},
            "score": {"passed": True, "compile_passed": True, "exact_match": False},
            "total_forwards": 1,
            "token_forwards": 10,
            "forward_sequence_lengths": [10],
            "selected_update_counts": {"HARD_SLOT_0": 1},
            "normal_update_counts": {"HARD_SLOT_0": 1},
            "expand_counts": {"HARD_SLOT_0": 0},
            "delete_counts": {"HARD_SLOT_0": 0},
            "mask_noop_counts": {"HARD_SLOT_0": 0},
            "newline_boundary_event_details": [],
            "region_local_eos_event_details": [],
        }
    ]
    candidate = [
        {
            **control[0],
            "method": Method.V3_C0_BUDGETED_ONESHOT_PURE_NEWLINE_VETO.value,
            "pure_newline_guard_trigger_count": 0,
        }
    ]
    audit = pure_newline_isolation_audit(candidate, control)
    assert audit["rows_without_trigger"] == 1
    assert audit["all_inactive_rows_match"] is True
    candidate[0]["token_forwards"] = 11
    assert pure_newline_isolation_audit(candidate, control)[
        "all_inactive_rows_match"
    ] is False
