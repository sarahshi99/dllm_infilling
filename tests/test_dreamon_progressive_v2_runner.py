from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repro_scripts.dreamon_slot_generator import Method
from repro_scripts.run_dreamon_progressive_v2 import (
    STAGE_SIZES,
    atomic_write_json,
    build_method_config,
    classify_protocol_outcome,
    load_generation_population,
    prediction_key,
    stable_json_hash,
    validate_resume_rows,
    write_discard_reference_diagnostics,
    run_gate,
)


def sample_population():
    return [
        {
            "task_id": f"MultiLineInfilling/HumanEval/{index}/L0_L2",
            "base_problem_id": str(index),
            "prompt": f"prefix-{index}",
            "suffix": f"suffix-{index}",
        }
        for index in range(642)
    ]


def test_stage_sizes_are_frozen():
    assert STAGE_SIZES == {"smoke": 5, "pilot": 30, "full": 642}


def test_population_loader_preserves_order_and_rejects_reference_fields(tmp_path):
    path = tmp_path / "population.jsonl"
    rows = sample_population()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert load_generation_population(path) == rows
    rows[0]["canonical_solution"] = "answer"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden generation field"):
        load_generation_population(path)


def test_config_hash_is_stable_and_method_specific():
    hard = build_method_config(Method.V2_HARD, "full", "runner-commit")
    open_tail = build_method_config(Method.V2_OPENTAIL, "full", "runner-commit")
    assert stable_json_hash(hard) == stable_json_hash(dict(reversed(list(hard.items()))))
    assert stable_json_hash(hard) != stable_json_hash(open_tail)


def test_boundary_v2_config_has_distinct_method_and_protocol_version():
    boundary = build_method_config(
        Method.V2_HARD_V2_BOUNDARY, "full", "runner-commit"
    )
    hard_v1 = build_method_config(Method.V2_HARD, "full", "runner-commit")
    assert boundary["method"] == "v2_hard_v2_boundary"
    assert boundary["protocol_version"] == 2
    assert boundary["newline_boundary_action"] is True
    assert boundary["region_local_eos_broadcast"] is True
    assert boundary["exact_transition_cycle_detection"] is True
    assert stable_json_hash(boundary) != stable_json_hash(hard_v1)


def test_smoke_pilot_and_full_configs_cannot_mix():
    hashes = {
        stable_json_hash(build_method_config(Method.V2_HARD, stage, "runner-commit"))
        for stage in STAGE_SIZES
    }
    assert len(hashes) == 3


def test_prediction_key_includes_task_method_and_config_hash():
    row = {
        "task_id": "task",
        "method": "v2_hard",
        "config_hash": "abc",
    }
    assert prediction_key(row) == ("task", "v2_hard", "abc")


def test_resume_validation_rejects_duplicate_key():
    rows = [
        {"task_id": "a", "method": "v2_hard", "config_hash": "hash"},
        {"task_id": "a", "method": "v2_hard", "config_hash": "hash"},
    ]
    with pytest.raises(RuntimeError, match="duplicate prediction key"):
        validate_resume_rows(rows, Method.V2_HARD, "hash", ["a"])


def test_resume_validation_rejects_wrong_method_or_config():
    with pytest.raises(RuntimeError, match="method/config mismatch"):
        validate_resume_rows(
            [{"task_id": "a", "method": "joint_opentail", "config_hash": "x"}],
            Method.V2_HARD,
            "hash",
            ["a"],
        )


def test_resume_validation_rejects_task_outside_stage():
    with pytest.raises(RuntimeError, match="outside selected stage"):
        validate_resume_rows(
            [{"task_id": "b", "method": "v2_hard", "config_hash": "hash"}],
            Method.V2_HARD,
            "hash",
            ["a"],
        )


def test_atomic_progress_write_round_trips(tmp_path):
    path = tmp_path / "progress.json"
    payload = {"completed": 7, "total": 30, "status": "running"}
    atomic_write_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "progress.json.tmp").exists()


def test_forward_cap_is_explicit_terminal_failure_not_implementation_violation():
    row = {
        "status": "protocol_error",
        "protocol_flags": ["forward_cap_with_unresolved_masks"],
        "unresolved_mask_count": 3,
        "completion": "",
    }
    assert classify_protocol_outcome(row) == "allowed_terminal_protocol_failure"


@pytest.mark.parametrize(
    "flag", ["forward_cap_with_unresolved_masks", "exact_deterministic_cycle"]
)
def test_boundary_v2_terminal_failures_are_method_failures_not_audit_success(flag):
    row = {
        "method": "v2_hard_v2_boundary",
        "status": "protocol_error",
        "protocol_flags": [flag],
        "unresolved_mask_count": 3,
        "completion": "",
    }
    assert classify_protocol_outcome(row) == "terminal_method_failure"


def test_boundary_v2_invariant_failure_remains_protocol_violation():
    row = {
        "method": "v2_hard_v2_boundary",
        "status": "protocol_error",
        "protocol_flags": ["future_region_mutated_before_activation"],
        "unresolved_mask_count": 1,
        "completion": "",
    }
    assert classify_protocol_outcome(row) == "protocol_violation"


def test_silent_unresolved_or_invariant_error_is_protocol_violation():
    silent = {
        "status": "completed",
        "protocol_flags": [],
        "unresolved_mask_count": 1,
        "completion": "code",
    }
    invariant = {
        "status": "protocol_error",
        "protocol_flags": ["future_region_mutated_before_activation"],
        "unresolved_mask_count": 1,
        "completion": "",
    }
    assert classify_protocol_outcome(silent) == "protocol_violation"
    assert classify_protocol_outcome(invariant) == "protocol_violation"


def test_runner_source_has_no_reference_access_in_generation_function():
    root = Path(__file__).resolve().parents[1]
    source = (root / "repro_scripts/run_dreamon_progressive_v2.py").read_text(
        encoding="utf-8"
    )
    generation_source = source.split("def run_generation", 1)[1].split(
        "def score_completion", 1
    )[0]
    forbidden = [
        "canonical_solution",
        "reference_middle",
        "reference_lines",
        "check_correctness",
    ]
    assert not any(term in generation_source for term in forbidden)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


def test_discard_reference_diagnostic_preserves_segments_and_scores_groups(tmp_path):
    row = {
        "task_id": "MultiLineInfilling/HumanEval/0/L0_L2",
        "base_problem_id": "0",
        "score": {"passed": True, "compile_passed": True, "exact_match": False},
        "normal_update_counts": {
            "HARD_SLOT_0": 2,
            "HARD_SLOT_1": 1,
            "HARD_SLOT_2": 1,
        },
        "boundary_retokenized_left_token_count": 1,
        "region_text": {
            "HARD_SLOT_0": "line0",
            "HARD_SLOT_1": "line1",
            "HARD_SLOT_2": "line2",
        },
        "newline_boundary_event_details": [
            {
                "slot": "HARD_SLOT_0",
                "proposal_token_id": 510,
                "decoded_text": ":\n",
                "left_text": ":",
                "right_text": "line1",
                "discarded_contiguous_segments": ["line1", "fragment"],
                "discarded_complete_text": None,
                "discarded_masks_after_boundary": 1,
                "discarded_resolved_tokens_after_boundary": 2,
                "discarded_resolved_segments_after_boundary": [
                    {"token_ids": [1, 2], "text": "fragment"}
                ],
            }
        ],
    }
    problems = {
        row["task_id"]: {"canonical_solution": "line0\nline1\nline2\n"}
    }
    summary = write_discard_reference_diagnostics(
        tmp_path, [row], problems, CharacterTokenizer()
    )
    assert summary["event_count"] == 1
    assert summary["rows_with_resolved_discard"] == 1
    assert summary["leading_segment_reference"]["prefix_rate"] == 1.0
    event = json.loads(
        (tmp_path / "discard_reference_events.jsonl").read_text().splitlines()[0]
    )
    assert event["discarded_contiguous_segments"] == ["line1", "fragment"]
    assert event["discarded_resolved_segments"][0]["text"] == "fragment"


def test_smoke_gate_requires_all_completed_and_sequential(tmp_path):
    required = {
        "newline_boundary_events": 0,
        "mixed_newline_token_events": 0,
        "pure_newline_token_events": 0,
        "multiple_newline_token_events": 0,
        "boundary_retokenized_left_token_count": 0,
        "discarded_internal_right_text": [],
        "discarded_masks_after_boundary": 0,
        "discarded_resolved_token_ids_after_boundary": [],
        "discarded_resolved_segments_after_boundary": [],
        "discarded_resolved_tokens_after_boundary": 0,
        "boundary_events_with_left_masks_remaining": 0,
        "region_local_eos_events": 0,
        "region_local_eos_deleted_masks": 0,
        "max_masks_deleted_by_one_eos": 0,
        "cross_region_delete_attempts": 0,
        "exact_cycle_events": [],
    }
    predictions = []
    for index in range(5):
        trace = [{"forward_index": item} for item in range(1, 4)]
        predictions.append(
            {
                "task_id": f"task-{index}",
                "status": "completed",
                "protocol_flags": [],
                "unresolved_mask_count": 0,
                "cross_region_delete_attempts": 0,
                "active_region_history": [
                    "HARD_SLOT_0",
                    "HARD_SLOT_1",
                    "HARD_SLOT_2",
                ],
                "total_forwards": 3,
                "step_trace": trace,
                **required,
            }
        )
    (tmp_path / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    (tmp_path / "scored.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "passed": 0,
                "compile_passed": 0,
                "exact_cycle_count": 0,
                "forward_cap_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "completeness_audit.json").write_text(
        json.dumps({"complete": True, "posthoc_truncation_fields_present": False}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        method=Method.V2_HARD_V2_BOUNDARY.value,
        stage="smoke",
        output_dir=tmp_path,
    )
    run_gate(args)
    assert json.loads((tmp_path / "gate.json").read_text())["passed"] is True
