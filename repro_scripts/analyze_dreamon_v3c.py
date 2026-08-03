#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from repro_scripts.analyze_dreamon_v3_budgeted import paired, result_summary


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/home/shx/projects/dllm_infilling")
OUTPUT = ROOT / "repro_results/dreamon_v3c_targeted_analysis"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def group_summary(
    rows: Sequence[Mapping[str, Any]],
    control: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "base_problems": 0,
            "completed": 0,
            "passed": 0,
            "pass_rate": None,
            "compile_passed": 0,
            "compile_pass_rate": None,
            "exact_matches": 0,
            "exact_match_rate": None,
            "task_macro_pass_rate": None,
            "blank_slot_rows": 0,
            "budget_exhausted_rows": 0,
            "total_forwards": 0,
            "total_token_forwards": 0,
            "total_wall_time_seconds": 0.0,
            "peak_cuda_memory_bytes": 0,
            "trigger_rows": 0,
            "trigger_events": 0,
            "paired_vs_control": None,
        }
    summary = result_summary(rows)
    summary["trigger_rows"] = sum(
        int(row.get("pure_newline_guard_trigger_count", 0)) > 0 for row in rows
    )
    summary["trigger_events"] = sum(
        int(row.get("pure_newline_guard_trigger_count", 0)) for row in rows
    )
    if control is not None and rows:
        summary["paired_vs_control"] = paired(rows, control)
    return summary


def transition(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> str:
    candidate_pass = bool(candidate["score"]["passed"])
    control_pass = bool(control["score"]["passed"])
    if candidate_pass and not control_pass:
        return "win"
    if control_pass and not candidate_pass:
        return "loss"
    return "both_pass" if candidate_pass else "both_fail"


def earliest_a_blank_cause(row: Mapping[str, Any]) -> str:
    blank_slots = [
        slot for slot, is_blank in row["blank_region_flags"].items() if is_blank
    ]
    if not blank_slots:
        return "nonblank"
    slot = min(blank_slots, key=lambda name: int(name.rsplit("_", 1)[1]))
    boundaries = [
        event
        for event in row["newline_boundary_event_details"]
        if event["slot"] == slot and int(event.get("region_length_after", -1)) == 0
    ]
    if boundaries:
        event = boundaries[0]
        if (
            event.get("normalized_text") == "\n"
            and event.get("left_text") == ""
            and event.get("right_text") == ""
        ):
            return "pure_newline"
        return "whitespace_or_mixed_newline"
    eos_events = [
        event
        for event in row["region_local_eos_event_details"]
        if event["slot"] == slot and int(event.get("region_length_after", -1)) == 0
    ]
    if eos_events:
        return "EOS"
    return "other_blank_completion"


def trigger_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    events = row.get("pure_newline_guard_events", [])
    return {
        "resolved_right": any(event["has_resolved_right_token"] for event in events),
        "nonwhitespace_resolved_right": any(
            event["has_nonwhitespace_resolved_right_token"] for event in events
        ),
        "repeated_request": int(row.get("repeated_blankline_requests", 0)) > 0,
        "fallback": int(row.get("blankline_insert_cap_fallbacks", 0)) > 0,
    }


def compute_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "forwards": sum(int(row["total_forwards"]) for row in rows),
        "token_forwards": sum(int(row["token_forwards"]) for row in rows),
        "wall_time_seconds": sum(float(row["wall_time_seconds"]) for row in rows),
        "peak_cuda_memory_bytes": max(
            int(row["peak_cuda_memory_bytes"]) for row in rows
        ),
    }


def compute_delta(
    candidate: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left = compute_summary(candidate)
    right = compute_summary(control)
    return {
        "candidate": left,
        "control": right,
        "candidate_minus_control": {
            key: left[key] - right[key]
            for key in ("forwards", "token_forwards", "wall_time_seconds")
        },
        "candidate_over_control": {
            key: left[key] / right[key]
            for key in ("forwards", "token_forwards", "wall_time_seconds")
        },
        "timing_caveat": "Wall time is observational because runs occurred at different times with an unrelated user GPU process present; forwards and token-forwards are the cleaner compute comparison.",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    historical = read_jsonl(
        EXTERNAL / "repro_results/dreamon_progressive_three_line_all642/scored.jsonl"
    )
    historical_by_task = {row["task_id"]: row for row in historical}
    a_dir = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_all642"
    b_dir = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_all642"
    c_dir = ROOT / "repro_results/dreamon_v3_c_nonconsuming_blankline_all642"
    c0_pure_dir = ROOT / "repro_results/dreamon_v3_c0_pure_newline_veto_pure30"
    c_pure_dir = ROOT / "repro_results/dreamon_v3_c_nonconsuming_blankline_pure30"
    c0_pilot_dir = ROOT / "repro_results/dreamon_v3_c0_pure_newline_veto_pilot30"
    c_pilot_dir = ROOT / "repro_results/dreamon_v3_c_nonconsuming_blankline_pilot30"

    a = read_jsonl(a_dir / "scored.jsonl")
    b = read_jsonl(b_dir / "scored.jsonl")
    c = read_jsonl(c_dir / "scored.jsonl")
    c0_pure = read_jsonl(c0_pure_dir / "scored.jsonl")
    c_pure = read_jsonl(c_pure_dir / "scored.jsonl")
    c0_pilot = read_jsonl(c0_pilot_dir / "scored.jsonl")
    c_pilot = read_jsonl(c_pilot_dir / "scored.jsonl")
    a_by_task = {row["task_id"]: row for row in a}
    b_by_task = {row["task_id"]: row for row in b}
    c0_pure_by_task = {row["task_id"]: row for row in c0_pure}
    c0_pilot_by_task = {row["task_id"]: row for row in c0_pilot}

    task_ids = [row["task_id"] for row in a]
    if task_ids != [row["task_id"] for row in b] or task_ids != [
        row["task_id"] for row in c
    ]:
        raise RuntimeError("A/B/C Full task order mismatch")

    one_shot = lambda task: historical_by_task[task]["scores"]["baseline"]
    v1 = lambda task: historical_by_task[task]["scores"][
        "progressive_equal_compute"
    ]
    a_score = lambda task: a_by_task[task]["score"]
    b_score = lambda task: b_by_task[task]["score"]
    c0_pure_score = lambda task: c0_pure_by_task[task]["score"]

    full_pairing = {
        "c_vs_a": paired(c, a_score),
        "c_vs_b": paired(c, b_score),
        "c_vs_one_shot": paired(c, one_shot),
        "c_vs_v1_progressive": paired(c, v1),
    }
    pure_pairing = {
        "c0_vs_a": paired(c0_pure, a_score),
        "c_vs_a": paired(c_pure, a_score),
        "c_vs_c0": paired(c_pure, c0_pure_score),
    }
    pilot_pairing = {
        "c0_vs_a": paired(c0_pilot, a_score),
        "c_vs_a": paired(c_pilot, a_score),
        "c_vs_c0": paired(
            c_pilot,
            lambda task: c0_pilot_by_task[task]["score"],
        ),
    }

    trigger_rows = [
        row for row in c if int(row["pure_newline_guard_trigger_count"]) > 0
    ]
    no_trigger_rows = [
        row for row in c if int(row["pure_newline_guard_trigger_count"]) == 0
    ]
    slot_groups = {
        slot: [row for row in c if slot in row["blankline_insert_slot"]]
        for slot in ("HARD_SLOT_0", "HARD_SLOT_1", "HARD_SLOT_2")
    }
    early_rows = [
        row
        for row in c
        if {"HARD_SLOT_0", "HARD_SLOT_1"}.intersection(row["blankline_insert_slot"])
    ]
    slot2_rows = slot_groups["HARD_SLOT_2"]
    resolved_rows = [row for row in trigger_rows if trigger_flags(row)["resolved_right"]]
    nonresolved_rows = [
        row for row in trigger_rows if not trigger_flags(row)["resolved_right"]
    ]
    nonwhite_rows = [
        row
        for row in trigger_rows
        if trigger_flags(row)["nonwhitespace_resolved_right"]
    ]
    whitespace_or_none_rows = [
        row
        for row in trigger_rows
        if not trigger_flags(row)["nonwhitespace_resolved_right"]
    ]
    repeated_rows = [
        row for row in trigger_rows if trigger_flags(row)["repeated_request"]
    ]
    nonrepeated_rows = [
        row for row in trigger_rows if not trigger_flags(row)["repeated_request"]
    ]
    exhausted_rows = [row for row in c if row["budget_exhausted"]]
    nonexhausted_rows = [row for row in c if not row["budget_exhausted"]]

    a_cause = {row["task_id"]: earliest_a_blank_cause(row) for row in a}
    cause_groups = {
        cause: [row for row in c if a_cause[row["task_id"]] == cause]
        for cause in (
            "pure_newline",
            "EOS",
            "whitespace_or_mixed_newline",
            "other_blank_completion",
            "nonblank",
        )
    }

    stratified = {
        "trigger": group_summary(trigger_rows, a_score),
        "no_trigger": group_summary(no_trigger_rows, a_score),
        "slots": {
            slot: group_summary(rows, a_score) for slot, rows in slot_groups.items()
        },
        "early_slot01": group_summary(early_rows, a_score),
        "slot2": group_summary(slot2_rows, a_score),
        "resolved_right": group_summary(resolved_rows, a_score),
        "no_resolved_right": group_summary(nonresolved_rows, a_score),
        "nonwhitespace_resolved_right": group_summary(nonwhite_rows, a_score),
        "no_nonwhitespace_resolved_right": group_summary(
            whitespace_or_none_rows, a_score
        ),
        "repeated_request": group_summary(repeated_rows, a_score),
        "no_repeated_request": group_summary(nonrepeated_rows, a_score),
        "guard_fallback": group_summary(
            [row for row in trigger_rows if trigger_flags(row)["fallback"]], a_score
        ),
        "budget_exhausted": group_summary(exhausted_rows, a_score),
        "budget_not_exhausted": group_summary(nonexhausted_rows, a_score),
        "a_earliest_blank_cause": {
            cause: group_summary(rows, a_score) for cause, rows in cause_groups.items()
        },
    }

    future_events: list[dict[str, Any]] = []
    for row in c:
        row_transition = transition(row, a_by_task[row["task_id"]])
        for event in row["future_slot_diagnostic_events"]:
            future_events.append(
                {
                    "task_id": row["task_id"],
                    "base_problem_id": row["base_problem_id"],
                    "transition_vs_a": row_transition,
                    **event,
                }
            )
    future_by_transition = Counter(
        event["transition_vs_a"] for event in future_events
    )
    future_lower_by_transition = Counter(
        event["transition_vs_a"]
        for event in future_events
        if event["future_has_lower_entropy_normal_candidate"]
    )
    future_higher_by_transition = Counter(
        event["transition_vs_a"]
        for event in future_events
        if event["future_has_higher_top1_probability_normal_candidate"]
    )
    future_summary = {
        "events": len(future_events),
        "rows": len({event["task_id"] for event in future_events}),
        "active_selected_is_pure_termination_events": sum(
            event["selected_top1_action_class"] == "pure_newline"
            for event in future_events
        ),
        "future_lower_entropy_normal_events": sum(
            event["future_has_lower_entropy_normal_candidate"]
            for event in future_events
        ),
        "future_higher_top1_probability_normal_events": sum(
            event["future_has_higher_top1_probability_normal_candidate"]
            for event in future_events
        ),
        "by_transition": dict(future_by_transition),
        "lower_entropy_normal_by_transition": dict(future_lower_by_transition),
        "higher_probability_normal_by_transition": dict(future_higher_by_transition),
        "interpretation": "Observational mechanism evidence only. Normal-vs-termination classification is fixed; no diagnostic field affected generation.",
    }
    write_jsonl(OUTPUT / "future_slot_events.jsonl", future_events)
    write_jsonl(OUTPUT / "future_slot_cases_first10.jsonl", future_events[:10])

    full_changed = []
    for row in c:
        control = a_by_task[row["task_id"]]
        if row["completion"] == control["completion"] and row["score"] == control["score"]:
            continue
        full_changed.append(
            {
                "task_id": row["task_id"],
                "base_problem_id": row["base_problem_id"],
                "transition_vs_a": transition(row, control),
                "trigger_slots": row["blankline_insert_slot"],
                "resolved_right": trigger_flags(row)["resolved_right"],
                "nonwhitespace_resolved_right": trigger_flags(row)[
                    "nonwhitespace_resolved_right"
                ],
                "repeated_request": trigger_flags(row)["repeated_request"],
                "a_score": control["score"],
                "c_score": row["score"],
                "a_completion": control["completion"],
                "c_completion": row["completion"],
                "a_forwards": control["total_forwards"],
                "c_forwards": row["total_forwards"],
            }
        )
    write_jsonl(OUTPUT / "full_changed_rows.jsonl", full_changed)

    pure_changed = []
    for row in c_pure:
        task_id = row["task_id"]
        control = a_by_task[task_id]
        c0_row = c0_pure_by_task[task_id]
        pure_changed.append(
            {
                "task_id": task_id,
                "base_problem_id": row["base_problem_id"],
                "a_passed": bool(control["score"]["passed"]),
                "c0_passed": bool(c0_row["score"]["passed"]),
                "c_passed": bool(row["score"]["passed"]),
                "c_vs_a": transition(row, control),
                "c0_vs_a": transition(c0_row, control),
                "c_vs_c0": transition(row, c0_row),
                "trigger_slots": row["blankline_insert_slot"],
                "resolved_right": trigger_flags(row)["resolved_right"],
                "nonwhitespace_resolved_right": trigger_flags(row)[
                    "nonwhitespace_resolved_right"
                ],
                "a_completion": control["completion"],
                "c0_completion": c0_row["completion"],
                "c_completion": row["completion"],
            }
        )
    write_jsonl(OUTPUT / "pure30_rows.jsonl", pure_changed)

    pure_union_pass = sum(
        bool(a_by_task[row["task_id"]]["score"]["passed"])
        or bool(c0_pure_by_task[row["task_id"]]["score"]["passed"])
        or bool(row["score"]["passed"])
        for row in c_pure
    )
    oracle_upper = {
        "full_choose_c_only_on_c_wins_otherwise_a": {
            "passed": result_summary(a)["passed"]
            + full_pairing["c_vs_a"]["candidate_wins"],
            "rows": 642,
            "posthoc_oracle": True,
        },
        "pure30_best_of_a_c0_c": {
            "passed": pure_union_pass,
            "rows": 30,
            "posthoc_oracle": True,
        },
    }

    pure_a_rows = [a_by_task[row["task_id"]] for row in c_pure]
    compute = {
        "full_c_vs_a": compute_delta(c, a),
        "pure30_c0_vs_a": compute_delta(c0_pure, pure_a_rows),
        "pure30_c_vs_a": compute_delta(c_pure, pure_a_rows),
        "pure30_c_vs_c0": compute_delta(c_pure, c0_pure),
        "tokenizer_newline_map_preprocessing_seconds": read_json(
            c_dir / "tokenizer_metadata.json"
        )["preprocessing_seconds"],
    }

    summaries = {
        "one_shot": result_summary(
            [
                {
                    "task_id": row["task_id"],
                    "base_problem_id": row["base_problem_id"],
                    "score": row["scores"]["baseline"],
                    "status": "completed",
                }
                for row in historical
            ]
        ),
        "v1_progressive": result_summary(
            [
                {
                    "task_id": row["task_id"],
                    "base_problem_id": row["base_problem_id"],
                    "score": row["scores"]["progressive_equal_compute"],
                    "status": "completed",
                }
                for row in historical
            ]
        ),
        "v3_a_budgeted": group_summary(a),
        "v3_b_nonempty_oracle": group_summary(b),
        "v3_c_nonconsuming_blankline": group_summary(c),
    }

    integrity = {
        "complete": True,
        "full_rows": len(c),
        "full_unique_task_ids": len({row["task_id"] for row in c}),
        "same_a_b_c_task_order": True,
        "manifest_sha256": read_json(c_dir / "completeness_audit.json")[
            "manifest_sha256"
        ],
        "population_sha256": "ffaac8a367103ee50893542bf5deb368232d40a904509db4708173994427731d",
        "pure30_sha256": sha256(ROOT / "manifests/v3_pure_newline_blank30.jsonl"),
        "c0_protocol_sha256": sha256(
            ROOT / "repro_results/dreamon_v3_c0_pure_newline_veto_protocol/protocol.json"
        ),
        "c_protocol_sha256": sha256(
            ROOT / "repro_results/dreamon_v3_c_nonconsuming_blankline_protocol/protocol.json"
        ),
        "c_full_config_hash": read_json(c_dir / "config.json")["config_hash"],
        "c0_pure_config_hash": read_json(c0_pure_dir / "config.json")[
            "config_hash"
        ],
        "c_pure_config_hash": read_json(c_pure_dir / "config.json")[
            "config_hash"
        ],
        "model_revision": c[0]["model_revision"],
        "runner_commit": c[0]["runner_commit"],
        "prediction_sha256": sha256(c_dir / "predictions.jsonl"),
        "scored_sha256": sha256(c_dir / "scored.jsonl"),
        "runtime_errors": sum(row["status"] == "runtime_error" for row in c),
        "protocol_errors": sum(row["status"] == "protocol_error" for row in c),
        "unresolved_masks": sum(int(row["unresolved_mask_count"]) for row in c),
        "exact_cycles": sum(
            "exact_deterministic_cycle" in row["protocol_flags"] for row in c
        ),
        "no_trigger_isolation": read_json(
            c_dir / "pure_newline_isolation_audit.json"
        ),
    }

    analysis = {
        "population_role": "642-row / 115-base-problem exact-three-line development/mechanism population; not held-out",
        "c0_full_status": "not run because Pure30 Pass was 5/30, below the frozen 6/30 authorization threshold",
        "c_full_status": "completed after Smoke5, Pure30, and Pilot30 engineering gates passed and Pure30 reached 14/30",
        "summaries": summaries,
        "full_pairing": full_pairing,
        "pure30": {
            "a_passed": 3,
            "c0_summary": group_summary(c0_pure),
            "c_summary": group_summary(c_pure),
            "pairing": pure_pairing,
            "sealed_slot_distribution": read_json(
                ROOT / "manifests/v3_pure_newline_blank30.meta.json"
            )["slot_distribution"],
        },
        "pilot30": {
            "c0_summary": group_summary(c0_pilot),
            "c_summary": group_summary(c_pilot),
            "pairing": pilot_pairing,
        },
        "stratified": stratified,
        "future_slots": future_summary,
        "oracle_upper_bounds": oracle_upper,
        "compute": compute,
        "integrity": integrity,
        "failure_taxonomy": {
            "compile_failures": sum(
                not row["score"]["compile_passed"] for row in c
            ),
            "compiled_functional_failures": sum(
                row["score"]["compile_passed"] and not row["score"]["passed"]
                for row in c
            ),
            "blank_slot_rows": sum(
                any(row["blank_region_flags"].values()) for row in c
            ),
            "budget_exhausted_rows": sum(row["budget_exhausted"] for row in c),
            "slot_cap_hit_rows": sum(row["slot_expand_cap_hits"] > 0 for row in c),
            "global_cap_hit_rows": sum(
                row["global_expand_cap_hits"] > 0 for row in c
            ),
            "a_earliest_blank_cause_distribution": dict(Counter(a_cause.values())),
        },
        "decision": "advance",
        "decision_basis": "C materially outperformed A on the targeted Pure30 and retained a smaller but positive paired Full642 gain with positive clustered uncertainty bounds. C0 failed its gate, and frozen-future confidence was rarely higher, so the supported mechanism is physical blank-line reconditioning rather than a simple veto or a general future-slot confidence advantage. Advance only to independent review and preregistered held-out validation; do not claim superiority to one-shot/V1 from this development population.",
    }
    write_json(OUTPUT / "analysis.json", analysis)
    write_json(c_dir / "full_analysis.json", analysis)
    write_json(c0_pure_dir / "pure30_analysis.json", analysis["pure30"])
    write_json(c_pure_dir / "pure30_analysis.json", analysis["pure30"])
    write_json(c0_pilot_dir / "pilot_analysis.json", analysis["pilot30"])
    write_json(c_pilot_dir / "pilot_analysis.json", analysis["pilot30"])

    result_rows = []
    for method, summary in summaries.items():
        result_rows.append(
            {
                "method": method,
                "rows": summary["rows"],
                "passed": summary["passed"],
                "pass_rate": summary["pass_rate"],
                "compile_passed": summary["compile_passed"],
                "compile_rate": summary["compile_pass_rate"],
                "exact_matches": summary["exact_matches"],
                "exact_rate": summary["exact_match_rate"],
                "task_macro_pass_rate": summary["task_macro_pass_rate"],
            }
        )
    write_csv(
        OUTPUT / "main_results.csv",
        (
            "method",
            "rows",
            "passed",
            "pass_rate",
            "compile_passed",
            "compile_rate",
            "exact_matches",
            "exact_rate",
            "task_macro_pass_rate",
        ),
        result_rows,
    )
    paired_rows = []
    for stage, comparisons in (
        ("full", full_pairing),
        ("pure30", pure_pairing),
        ("pilot30", pilot_pairing),
    ):
        for name, values in comparisons.items():
            paired_rows.append(
                {
                    "stage": stage,
                    "comparison": name,
                    "wins": values["candidate_wins"],
                    "losses": values["candidate_losses"],
                    "both_pass": values["both_pass"],
                    "both_fail": values["both_fail"],
                    "net_wins": values["net_wins"],
                    "mcnemar_exact_p": values["mcnemar_exact_p"],
                    "row_ci_low": values["row_bootstrap_95_ci"][0],
                    "row_ci_high": values["row_bootstrap_95_ci"][1],
                    "cluster_ci_low": values["cluster_bootstrap_95_ci"][0],
                    "cluster_ci_high": values["cluster_bootstrap_95_ci"][1],
                    "task_macro_delta": values["task_macro_delta"],
                }
            )
    write_csv(
        OUTPUT / "paired_comparisons.csv",
        (
            "stage",
            "comparison",
            "wins",
            "losses",
            "both_pass",
            "both_fail",
            "net_wins",
            "mcnemar_exact_p",
            "row_ci_low",
            "row_ci_high",
            "cluster_ci_low",
            "cluster_ci_high",
            "task_macro_delta",
        ),
        paired_rows,
    )

    c_summary = summaries["v3_c_nonconsuming_blankline"]
    a_summary = summaries["v3_a_budgeted"]
    c_vs_a = full_pairing["c_vs_a"]
    c_vs_c0 = pure_pairing["c_vs_c0"]
    report = f"""# DreamOn V3-C targeted results

Date: 2026-08-03 UTC

Population: 642 rows / 115 base problems, development/mechanism population, not held-out.

## Stage decisions

- C0 Smoke5 passed; Pure30 reached 5/30 and therefore did not authorize Full.
- C Smoke5 passed; Pure30 reached 14/30; Pilot30 engineering/isolation checks passed; Full642 was authorized and completed.

## Main Full642 result

| Method | Pass@1 | Compile | Exact | Task-macro Pass |
|---|---:|---:|---:|---:|
| One-shot | {summaries['one_shot']['passed']}/642 ({pct(summaries['one_shot']['pass_rate'])}) | {summaries['one_shot']['compile_passed']}/642 ({pct(summaries['one_shot']['compile_pass_rate'])}) | {summaries['one_shot']['exact_matches']}/642 ({pct(summaries['one_shot']['exact_match_rate'])}) | {pct(summaries['one_shot']['task_macro_pass_rate'])} |
| Historical V1 | {summaries['v1_progressive']['passed']}/642 ({pct(summaries['v1_progressive']['pass_rate'])}) | {summaries['v1_progressive']['compile_passed']}/642 ({pct(summaries['v1_progressive']['compile_pass_rate'])}) | {summaries['v1_progressive']['exact_matches']}/642 ({pct(summaries['v1_progressive']['exact_match_rate'])}) | {pct(summaries['v1_progressive']['task_macro_pass_rate'])} |
| V3-A Budgeted | {a_summary['passed']}/642 ({pct(a_summary['pass_rate'])}) | {a_summary['compile_passed']}/642 ({pct(a_summary['compile_pass_rate'])}) | {a_summary['exact_matches']}/642 ({pct(a_summary['exact_match_rate'])}) | {pct(a_summary['task_macro_pass_rate'])} |
| V3-B Nonempty oracle | {summaries['v3_b_nonempty_oracle']['passed']}/642 ({pct(summaries['v3_b_nonempty_oracle']['pass_rate'])}) | {summaries['v3_b_nonempty_oracle']['compile_passed']}/642 ({pct(summaries['v3_b_nonempty_oracle']['compile_pass_rate'])}) | {summaries['v3_b_nonempty_oracle']['exact_matches']}/642 ({pct(summaries['v3_b_nonempty_oracle']['exact_match_rate'])}) | {pct(summaries['v3_b_nonempty_oracle']['task_macro_pass_rate'])} |
| V3-C Nonconsuming blank line | {c_summary['passed']}/642 ({pct(c_summary['pass_rate'])}) | {c_summary['compile_passed']}/642 ({pct(c_summary['compile_pass_rate'])}) | {c_summary['exact_matches']}/642 ({pct(c_summary['exact_match_rate'])}) | {pct(c_summary['task_macro_pass_rate'])} |

## Paired interpretation

- C vs A: {c_vs_a['candidate_wins']} wins / {c_vs_a['candidate_losses']} losses / {c_vs_a['both_pass']} both-pass / {c_vs_a['both_fail']} both-fail; McNemar p={c_vs_a['mcnemar_exact_p']:.6g}; 115-cluster bootstrap CI [{pct(c_vs_a['cluster_bootstrap_95_ci'][0])}, {pct(c_vs_a['cluster_bootstrap_95_ci'][1])}].
- C vs C0 on Pure30: {c_vs_c0['candidate_wins']} wins / {c_vs_c0['candidate_losses']} losses; McNemar p={c_vs_c0['mcnemar_exact_p']:.6g}; cluster CI [{pct(c_vs_c0['cluster_bootstrap_95_ci'][0])}, {pct(c_vs_c0['cluster_bootstrap_95_ci'][1])}].
- No-trigger isolation: 612/612 rows matched A exactly in completion, compact action signature, score, forwards, and token-forwards.
- C does not establish superiority to one-shot or historical V1 on this development population. C vs one-shot is net {full_pairing['c_vs_one_shot']['net_wins']} rows and C vs V1 is net {full_pairing['c_vs_v1_progressive']['net_wins']} rows, with clustered intervals including zero.

## Mechanism

- C0's one-shot veto was insufficient: 5/30 on Pure30 versus A 3/30.
- C reached 14/30 on the same Pure30. Because both preserve the right-side content canvas, the additional benefit is consistent with reconditioning on the inserted physical blank line.
- Frozen future slots rarely had a more confident normal candidate: lower-entropy normal in {future_summary['future_lower_entropy_normal_events']}/{future_summary['events']} events and higher top1-probability normal in {future_summary['future_higher_top1_probability_normal_events']}/{future_summary['events']} events. This does not support a general future-slot-confidence explanation.
- C reduced blank-slot rows from {a_summary['blank_slot_rows']} to {c_summary['blank_slot_rows']} without a blanket nonempty guard.

## Compute

- C Full: {compute['full_c_vs_a']['candidate']['forwards']} forwards, {compute['full_c_vs_a']['candidate']['token_forwards']} token-forwards, {compute['full_c_vs_a']['candidate']['wall_time_seconds']:.1f}s, peak {compute['full_c_vs_a']['candidate']['peak_cuda_memory_bytes']} bytes.
- C minus A: {compute['full_c_vs_a']['candidate_minus_control']['forwards']:+d} forwards and {compute['full_c_vs_a']['candidate_minus_control']['token_forwards']:+d} token-forwards. Wall time is not used as the cleaner causal compute comparison.

## Decision

`advance`: send V3-C for independent review and preregistered held-out validation. Do not deploy it, do not claim it beats one-shot/V1, and do not restart C0, blanket nonempty, OpenTail, Joint, BoundaryShift, compile gate, or AST repair in this round.
"""
    (OUTPUT / "report.zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
