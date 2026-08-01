from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro_scripts.dreamon_slot_generator import Method
from repro_scripts.run_dreamon_progressive_v2 import (
    STAGE_SIZES,
    atomic_write_json,
    build_method_config,
    load_generation_population,
    prediction_key,
    stable_json_hash,
    validate_resume_rows,
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
