#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PREDICTIONS = (
    ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_all642/predictions.jsonl"
)
SOURCE_SCORED = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_all642/scored.jsonl"
SOURCE_CONFIG = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_all642/config.json"
SOURCE_AUDIT = (
    ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_all642/completeness_audit.json"
)
GENERATION_POPULATION = (
    ROOT / "repro_results/dreamon_progressive_v2_protocol/generation_population.jsonl"
)
OUTPUT_DIR = ROOT / "manifests"

PURE30 = OUTPUT_DIR / "v3_pure_newline_blank30.jsonl"
META = OUTPUT_DIR / "v3_pure_newline_blank30.meta.json"
SLOT2 = OUTPUT_DIR / "v3_pure_newline_slot2_12.jsonl"
EARLY = OUTPUT_DIR / "v3_pure_newline_early_slot01_18.jsonl"

EXPECTED_ROWS = 30
EXPECTED_BASE_PROBLEMS = 21
EXPECTED_SLOTS = {"HARD_SLOT_0": 6, "HARD_SLOT_1": 12, "HARD_SLOT_2": 12}
EXPECTED_PURE_NEWLINE_TOKEN_ID = 198
SCRIPT_VERSION = 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_strict_pure_newline_blank_event(
    row: Mapping[str, Any], event: Mapping[str, Any]
) -> bool:
    if event["normalized_text"] != "\n":
        return False
    if event["left_text"] != "" or event["right_text"] != "":
        return False
    if event["left_token_ids"] != []:
        return False
    if int(event["region_length_after"]) != 0:
        return False
    if not bool(row["blank_region_flags"].get(event["slot"], False)):
        return False
    consumed_region_positions = (
        1
        + int(event["discarded_masks_after_boundary"])
        + int(event["discarded_resolved_tokens_after_boundary"])
    )
    if consumed_region_positions != int(event["region_length_before"]):
        return False
    return True


def main() -> None:
    predictions = read_jsonl(SOURCE_PREDICTIONS)
    scored = read_jsonl(SOURCE_SCORED)
    generation_rows = read_jsonl(GENERATION_POPULATION)
    config = read_json(SOURCE_CONFIG)
    audit = read_json(SOURCE_AUDIT)

    assert len(predictions) == len(scored) == len(generation_rows) == 642
    task_order = [str(row["task_id"]) for row in generation_rows]
    assert task_order == [str(row["task_id"]) for row in predictions]
    assert task_order == [str(row["task_id"]) for row in scored]
    assert len(set(task_order)) == 642
    assert audit["complete"] is True
    assert audit["config_hash"] == config["config_hash"]

    selected: list[dict[str, Any]] = []
    for population_index, row in enumerate(predictions):
        matches = [
            (event_index, event)
            for event_index, event in enumerate(row["newline_boundary_event_details"])
            if is_strict_pure_newline_blank_event(row, event)
        ]
        if not matches:
            continue
        assert len(matches) == 1, (row["task_id"], len(matches))
        event_index, event = matches[0]
        selected.append(
            {
                "population_index": population_index,
                "task_id": row["task_id"],
                "base_problem_id": row["base_problem_id"],
                "slot": event["slot"],
                "event_index": event_index,
                "proposal_token_id": event["proposal_token_id"],
                "proposal_decoded": event["decoded_text"],
                "pre_state_hash": event["pre_state_hash"],
                "post_state_hash": event["post_state_hash"],
                "region_length_before": event["region_length_before"],
                "region_length_after": event["region_length_after"],
                "discarded_masks": event["discarded_masks_after_boundary"],
                "discarded_resolved_tokens": event[
                    "discarded_resolved_tokens_after_boundary"
                ],
                "discarded_resolved_segments": event[
                    "discarded_resolved_segments_after_boundary"
                ],
                "has_resolved_right_token": bool(
                    event["discarded_resolved_tokens_after_boundary"]
                ),
                "has_nonwhitespace_resolved_right_token": any(
                    segment["text"].strip()
                    for segment in event["discarded_resolved_segments_after_boundary"]
                ),
            }
        )

    assert len(selected) == EXPECTED_ROWS, len(selected)
    assert len({row["base_problem_id"] for row in selected}) == EXPECTED_BASE_PROBLEMS
    assert Counter(row["slot"] for row in selected) == Counter(EXPECTED_SLOTS)
    assert {row["proposal_token_id"] for row in selected} == {
        EXPECTED_PURE_NEWLINE_TOKEN_ID
    }

    selected_ids = [str(row["task_id"]) for row in selected]
    selected_id_set = set(selected_ids)
    pure30_rows = [row for row in generation_rows if row["task_id"] in selected_id_set]
    assert [row["task_id"] for row in pure30_rows] == selected_ids
    slot_by_task = {str(row["task_id"]): str(row["slot"]) for row in selected}
    slot2_rows = [row for row in pure30_rows if slot_by_task[row["task_id"]] == "HARD_SLOT_2"]
    early_rows = [row for row in pure30_rows if slot_by_task[row["task_id"]] != "HARD_SLOT_2"]
    assert len(slot2_rows) == 12
    assert len(early_rows) == 18

    write_jsonl(PURE30, pure30_rows)
    write_jsonl(SLOT2, slot2_rows)
    write_jsonl(EARLY, early_rows)

    score_by_task = {str(row["task_id"]): row["score"] for row in scored}
    source_runner_commits = sorted({str(row["runner_commit"]) for row in predictions})
    meta: dict[str, Any] = {
        "schema_version": 1,
        "construction_script": str(Path(__file__).relative_to(ROOT)),
        "construction_script_version": SCRIPT_VERSION,
        "posthoc_mechanism_diagnostic": True,
        "selection_source": "V3-A Full newline_boundary_event_details only; B outcomes are not read",
        "selection_predicate": {
            "normalized_proposal": "\\n",
            "left_text": "",
            "right_text": "",
            "left_edge_proof": "1 selected + discarded masks + discarded resolved tokens equals pre-region length",
            "would_immediately_blank": "region_length_after == 0",
            "final_slot_blank": True,
            "earliest_blank_event": "exactly one strict event per selected row",
        },
        "source_a": {
            "predictions_path": str(SOURCE_PREDICTIONS.relative_to(ROOT)),
            "predictions_sha256": sha256(SOURCE_PREDICTIONS),
            "scored_path": str(SOURCE_SCORED.relative_to(ROOT)),
            "scored_sha256": sha256(SOURCE_SCORED),
            "config_path": str(SOURCE_CONFIG.relative_to(ROOT)),
            "config_hash": config["config_hash"],
            "runner_commits": source_runner_commits,
            "manifest_sha256": audit["manifest_sha256"],
        },
        "population": {
            "path": str(GENERATION_POPULATION.relative_to(ROOT)),
            "sha256": sha256(GENERATION_POPULATION),
            "rows": len(generation_rows),
        },
        "pure_newline_token_id_observed": EXPECTED_PURE_NEWLINE_TOKEN_ID,
        "rows": len(selected),
        "base_problems": len({row["base_problem_id"] for row in selected}),
        "slot_distribution": dict(sorted(Counter(row["slot"] for row in selected).items())),
        "a_baseline": {
            "passed": sum(bool(score_by_task[task_id]["passed"]) for task_id in selected_ids),
            "compiled": sum(
                bool(score_by_task[task_id]["compile_passed"]) for task_id in selected_ids
            ),
            "exact": sum(
                bool(score_by_task[task_id]["exact_match"]) for task_id in selected_ids
            ),
        },
        "resolved_right_token_rows": sum(
            bool(row["has_resolved_right_token"]) for row in selected
        ),
        "nonwhitespace_resolved_right_token_rows": sum(
            bool(row["has_nonwhitespace_resolved_right_token"]) for row in selected
        ),
        "task_order": selected_ids,
        "selection_rows": selected,
        "manifests": {},
    }
    write_json(META, meta)
    meta["manifests"] = {
        "pure30": {"path": str(PURE30.relative_to(ROOT)), "rows": 30, "sha256": sha256(PURE30)},
        "slot2_12": {"path": str(SLOT2.relative_to(ROOT)), "rows": 12, "sha256": sha256(SLOT2)},
        "early_slot01_18": {"path": str(EARLY.relative_to(ROOT)), "rows": 18, "sha256": sha256(EARLY)},
    }
    write_json(META, meta)

    print(
        json.dumps(
            {
                "rows": meta["rows"],
                "base_problems": meta["base_problems"],
                "slot_distribution": meta["slot_distribution"],
                "a_baseline": meta["a_baseline"],
                "pure30_sha256": meta["manifests"]["pure30"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
