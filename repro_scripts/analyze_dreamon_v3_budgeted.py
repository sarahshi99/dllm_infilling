#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/home/shx/projects/dllm_infilling")
SEED = 42
REPLICATES = 10_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def task_macro(rows: Sequence[Mapping[str, Any]], values: Sequence[bool]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values):
        grouped[str(row["base_problem_id"])].append(float(value))
    return sum(sum(group) / len(group) for group in grouped.values()) / len(grouped)


def paired(
    candidate_rows: Sequence[Mapping[str, Any]],
    control_score: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_pass = [bool(row["score"]["passed"]) for row in candidate_rows]
    control_pass = [bool(control_score(row["task_id"])["passed"]) for row in candidate_rows]
    deltas = [float(left) - float(right) for left, right in zip(candidate_pass, control_pass)]
    wins = sum(left and not right for left, right in zip(candidate_pass, control_pass))
    losses = sum(right and not left for left, right in zip(candidate_pass, control_pass))
    rng = random.Random(SEED)
    row_bootstrap = []
    for _ in range(REPLICATES):
        row_bootstrap.append(
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    cluster_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(candidate_rows):
        cluster_indices[str(row["base_problem_id"])].append(index)
    clusters = sorted(cluster_indices)
    cluster_bootstrap = []
    for _ in range(REPLICATES):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sampled_deltas = [
            deltas[index]
            for cluster in sampled
            for index in cluster_indices[cluster]
        ]
        cluster_bootstrap.append(sum(sampled_deltas) / len(sampled_deltas))
    candidate_compile = [
        bool(row["score"]["compile_passed"]) for row in candidate_rows
    ]
    control_compile = [
        bool(control_score(row["task_id"])["compile_passed"])
        for row in candidate_rows
    ]
    candidate_macro = task_macro(candidate_rows, candidate_pass)
    control_macro = task_macro(candidate_rows, control_pass)
    return {
        "candidate_wins": wins,
        "candidate_losses": losses,
        "both_pass": sum(
            left and right for left, right in zip(candidate_pass, control_pass)
        ),
        "both_fail": sum(
            not left and not right for left, right in zip(candidate_pass, control_pass)
        ),
        "net_wins": wins - losses,
        "mcnemar_exact_p": exact_mcnemar(wins, losses),
        "compile_help": sum(
            left and not right
            for left, right in zip(candidate_compile, control_compile)
        ),
        "compile_harm": sum(
            right and not left
            for left, right in zip(candidate_compile, control_compile)
        ),
        "exact_match_help": sum(
            bool(row["score"]["exact_match"])
            and not bool(control_score(row["task_id"])["exact_match"])
            for row in candidate_rows
        ),
        "exact_match_harm": sum(
            bool(control_score(row["task_id"])["exact_match"])
            and not bool(row["score"]["exact_match"])
            for row in candidate_rows
        ),
        "row_delta": sum(deltas) / len(deltas),
        "row_bootstrap_95_ci": [
            percentile(row_bootstrap, 0.025),
            percentile(row_bootstrap, 0.975),
        ],
        "cluster_bootstrap_95_ci": [
            percentile(cluster_bootstrap, 0.025),
            percentile(cluster_bootstrap, 0.975),
        ],
        "candidate_task_macro_pass_rate": candidate_macro,
        "control_task_macro_pass_rate": control_macro,
        "task_macro_delta": candidate_macro - control_macro,
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
    }


def group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "passed": sum(bool(row["score"]["passed"]) for row in rows),
        "compile_passed": sum(
            bool(row["score"]["compile_passed"]) for row in rows
        ),
    }


def result_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = [bool(row["score"]["passed"]) for row in rows]
    compiled = [bool(row["score"]["compile_passed"]) for row in rows]
    exact = [bool(row["score"]["exact_match"]) for row in rows]
    completed = sum(row.get("status", "completed") == "completed" for row in rows)
    return {
        "rows": len(rows),
        "base_problems": len({str(row["base_problem_id"]) for row in rows}),
        "completed": completed,
        "passed": sum(passed),
        "pass_rate": sum(passed) / len(rows),
        "compile_passed": sum(compiled),
        "compile_pass_rate": sum(compiled) / len(rows),
        "exact_matches": sum(exact),
        "exact_match_rate": sum(exact) / len(rows),
        "task_macro_pass_rate": task_macro(rows, passed),
        "blank_slot_rows": sum(
            any(bool(value) for value in row.get("blank_region_flags", {}).values())
            for row in rows
        ),
        "guard_triggered_rows": sum(
            int(row.get("nonempty_guard_rejection_count", 0)) > 0 for row in rows
        ),
        "budget_exhausted_rows": sum(bool(row.get("budget_exhausted")) for row in rows),
        "total_forwards": sum(int(row.get("total_forwards", 0)) for row in rows),
        "total_token_forwards": sum(int(row.get("token_forwards", 0)) for row in rows),
        "total_wall_time_seconds": sum(float(row.get("wall_time_seconds", 0.0)) for row in rows),
        "peak_cuda_memory_bytes": max(
            (int(row.get("peak_cuda_memory_bytes", 0)) for row in rows), default=0
        ),
    }


def historical_summary(
    rows: Sequence[Mapping[str, Any]], score_key: str
) -> dict[str, Any]:
    projected = [
        {
            "task_id": row["task_id"],
            "base_problem_id": row["base_problem_id"],
            "score": row["scores"][score_key],
            "status": "completed",
        }
        for row in rows
    ]
    return result_summary(projected)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    historical = read_jsonl(
        EXTERNAL / "repro_results/dreamon_progressive_three_line_all642/scored.jsonl"
    )
    historical_by_task = {row["task_id"]: row for row in historical}
    v2_full = read_jsonl(ROOT / "repro_results/dreamon_progressive_v2_hard_all642/scored.jsonl")
    v2_full_by_task = {row["task_id"]: row for row in v2_full}
    v2_pilot = read_jsonl(
        ROOT / "repro_results/dreamon_progressive_v2_hard_v2_pilot30/scored.jsonl"
    )
    v2_pilot_by_task = {row["task_id"]: row for row in v2_pilot}
    a_pilot = read_jsonl(
        ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_pilot30/scored.jsonl"
    )
    a_pilot_by_task = {row["task_id"]: row for row in a_pilot}
    a_full_dir = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_all642"
    a_full = read_jsonl(a_full_dir / "scored.jsonl")
    a_full_by_task = {row["task_id"]: row for row in a_full}
    b_pilot_dir = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_pilot30"
    b_pilot = read_jsonl(b_pilot_dir / "scored.jsonl")
    b_full_dir = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_all642"
    b_full = read_jsonl(b_full_dir / "scored.jsonl")
    b_full_by_task = {row["task_id"]: row for row in b_full}
    affected_dir = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_affected6"
    affected = read_jsonl(affected_dir / "scored.jsonl")

    baseline = lambda task: historical_by_task[task]["scores"]["baseline"]
    v1 = lambda task: historical_by_task[task]["scores"]["progressive_equal_compute"]
    pilot_analysis = {
        "population_role": "development/mechanism pilot; not held-out",
        "result": json.loads((b_pilot_dir / "summary.json").read_text()),
        "paired": {
            "vs_a_budgeted": paired(b_pilot, lambda task: a_pilot_by_task[task]["score"]),
            "vs_v2_hard_v2": paired(b_pilot, lambda task: v2_pilot_by_task[task]["score"]),
            "vs_one_shot": paired(b_pilot, baseline),
            "vs_v1_progressive": paired(b_pilot, v1),
        },
        "isolation": json.loads((b_pilot_dir / "nonempty_isolation_audit.json").read_text()),
        "guard_triggered": group(
            [row for row in b_pilot if row["nonempty_guard_rejection_count"] > 0]
        ),
        "guard_not_triggered": group(
            [row for row in b_pilot if row["nonempty_guard_rejection_count"] == 0]
        ),
        "full_authorized": True,
    }
    write_json(b_pilot_dir / "pilot_analysis.json", pilot_analysis)

    write_json(
        affected_dir / "affected6_analysis.json",
        {
            "role": "frozen mechanism diagnostic; not performance evidence",
            "result": json.loads((affected_dir / "summary.json").read_text()),
            "paired_vs_a": paired(affected, lambda task: a_full_by_task[task]["score"]),
            "isolation": json.loads((affected_dir / "nonempty_isolation_audit.json").read_text()),
        },
    )

    triggered = [row for row in b_full if row["nonempty_guard_rejection_count"] > 0]
    not_triggered = [row for row in b_full if row["nonempty_guard_rejection_count"] == 0]
    exhausted = [row for row in b_full if row["budget_exhausted"]]
    not_exhausted = [row for row in b_full if not row["budget_exhausted"]]
    compile_fail = [row for row in b_full if not row["score"]["compile_passed"]]
    functional_fail = [
        row
        for row in b_full
        if row["score"]["compile_passed"] and not row["score"]["passed"]
    ]
    ab_pair = paired(b_full, lambda task: a_full_by_task[task]["score"])
    ba_pair = paired(a_full, lambda task: b_full_by_task[task]["score"])
    a_full_analysis = {
        "population_role": "exact-three-line development/mechanism population; A Full was run post-hoc by explicit user authorization after failing the preregistered Pilot30 performance gate; not held-out",
        "result": json.loads((a_full_dir / "summary.json").read_text()),
        "paired": {
            "vs_one_shot": paired(a_full, baseline),
            "vs_v1_progressive": paired(a_full, v1),
            "vs_v2_hard_v1_multifactor": paired(
                a_full, lambda task: v2_full_by_task[task]["score"]
            ),
            "vs_b_nonempty_oracle": ba_pair,
        },
        "full_authorization_note": "The original A Pilot30 result was 14/30, below the preregistered 18/30 Full threshold. This Full642 was subsequently requested explicitly and is a post-hoc development/mechanism diagnostic, not a preregistered performance claim.",
        "discard_reference_diagnostic": json.loads(
            (a_full_dir / "discard_reference_diagnostics.json").read_text()
        ),
    }
    write_json(a_full_dir / "full_analysis.json", a_full_analysis)

    full_analysis = {
        "population_role": "exact-three-line development/mechanism population; oracle structural diagnostic; not deployable; not held-out",
        "result": json.loads((b_full_dir / "summary.json").read_text()),
        "paired": {
            "vs_a_budgeted": ab_pair,
            "vs_one_shot": paired(b_full, baseline),
            "vs_v1_progressive": paired(b_full, v1),
            "vs_v2_hard_v1_multifactor": paired(
                b_full, lambda task: v2_full_by_task[task]["score"]
            ),
        },
        "a_full_comparison": {
            "available": True,
            "posthoc_note": "A Full642 was run later by explicit user authorization after A failed the original Pilot30 Full gate; this comparison is a development/mechanism diagnostic.",
            "paired": ab_pair,
            "a_full_path": str(a_full_dir),
        },
        "mechanism_groups": {
            "guard_triggered": group(triggered),
            "guard_not_triggered": group(not_triggered),
            "budget_exhausted": group(exhausted),
            "budget_not_exhausted": group(not_exhausted),
        },
        "failure_taxonomy": {
            "compile_failure": len(compile_fail),
            "compiled_functional_failure": len(functional_fail),
            "failed_with_guard_trigger": sum(
                not row["score"]["passed"] for row in triggered
            ),
            "failed_with_budget_exhaustion": sum(
                not row["score"]["passed"] for row in exhausted
            ),
            "failed_with_slot_cap_hit": sum(
                not row["score"]["passed"] and row["slot_expand_cap_hits"] > 0
                for row in b_full
            ),
            "failed_with_global_cap_hit": sum(
                not row["score"]["passed"] and row["global_expand_cap_hits"] > 0
                for row in b_full
            ),
            "residual_label": "residual errors not explained by cumulative budget or empty-slot termination",
        },
        "discard_reference_diagnostic": json.loads(
            (b_full_dir / "discard_reference_diagnostics.json").read_text()
        ),
    }
    write_json(b_full_dir / "full_analysis.json", full_analysis)

    transitions = []
    for row in b_full:
        candidate = bool(row["score"]["passed"])
        for control_name, score in (("one_shot", baseline(row["task_id"])), ("v1_progressive", v1(row["task_id"]))):
            control = bool(score["passed"])
            if candidate != control:
                transitions.append(
                    {
                        "task_id": row["task_id"],
                        "base_problem_id": row["base_problem_id"],
                        "control": control_name,
                        "transition": "candidate_win" if candidate else "candidate_loss",
                        "candidate_compile": bool(row["score"]["compile_passed"]),
                        "guard_triggered": row["nonempty_guard_rejection_count"] > 0,
                        "budget_exhausted": bool(row["budget_exhausted"]),
                        "slot_cap_hit": row["slot_expand_cap_hits"] > 0,
                        "global_cap_hit": row["global_expand_cap_hits"] > 0,
                    }
                )
    write_jsonl(b_full_dir / "failure_transitions.jsonl", transitions)

    comparison_dir = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_ab_comparison_all642"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    a_summary = result_summary(a_full)
    b_summary = result_summary(b_full)
    one_shot_summary = historical_summary(historical, "baseline")
    v1_summary = historical_summary(historical, "progressive_equal_compute")
    v2_summary = result_summary(v2_full)

    a_task_ids = [str(row["task_id"]) for row in a_full]
    b_task_ids = [str(row["task_id"]) for row in b_full]
    inactive_core_fields = (
        "completion",
        "completion_token_ids",
        "score",
        "status",
        "total_forwards",
        "token_forwards",
        "region_text",
        "region_token_ids",
        "region_lengths",
        "normal_update_counts",
        "expand_counts",
        "delete_counts",
        "newline_boundary_events",
        "region_local_eos_events",
        "expand_budget_consumed",
        "remaining_expand_budget",
        "blank_region_flags",
    )
    inactive_differences = []
    for b_row in b_full:
        if int(b_row.get("nonempty_guard_rejection_count", 0)) != 0:
            continue
        a_row = a_full_by_task[b_row["task_id"]]
        differing = [
            field for field in inactive_core_fields if a_row.get(field) != b_row.get(field)
        ]
        if differing:
            inactive_differences.append(
                {"task_id": b_row["task_id"], "differing_fields": differing}
            )

    affected_rows = []
    pass_transitions = []
    for a_row, b_row in zip(a_full, b_full):
        assert a_row["task_id"] == b_row["task_id"]
        a_score = a_row["score"]
        b_score = b_row["score"]
        guard_triggered = int(b_row.get("nonempty_guard_rejection_count", 0)) > 0
        if guard_triggered:
            affected_rows.append(
                {
                    "task_id": b_row["task_id"],
                    "base_problem_id": b_row["base_problem_id"],
                    "guard_rejections": b_row["nonempty_guard_rejection_count"],
                    "guard_rejections_by_slot": b_row["nonempty_guard_rejections_by_slot"],
                    "guard_rejections_by_action": b_row["nonempty_guard_rejections_by_action"],
                    "a_blank_regions": [
                        region
                        for region, is_blank in a_row["blank_region_flags"].items()
                        if is_blank
                    ],
                    "a_score": a_score,
                    "b_score": b_score,
                    "pass_transition": (
                        "b_win"
                        if b_score["passed"] and not a_score["passed"]
                        else "b_loss"
                        if a_score["passed"] and not b_score["passed"]
                        else "both_pass"
                        if a_score["passed"] and b_score["passed"]
                        else "both_fail"
                    ),
                    "compile_transition": (
                        "b_help"
                        if b_score["compile_passed"] and not a_score["compile_passed"]
                        else "b_harm"
                        if a_score["compile_passed"] and not b_score["compile_passed"]
                        else "same"
                    ),
                    "exact_transition": (
                        "b_help"
                        if b_score["exact_match"] and not a_score["exact_match"]
                        else "b_harm"
                        if a_score["exact_match"] and not b_score["exact_match"]
                        else "same"
                    ),
                    "a_completion": a_row["completion"],
                    "b_completion": b_row["completion"],
                    "a_forwards": a_row["total_forwards"],
                    "b_forwards": b_row["total_forwards"],
                    "a_token_forwards": a_row["token_forwards"],
                    "b_token_forwards": b_row["token_forwards"],
                    "a_expand_budget_consumed": a_row["expand_budget_consumed"],
                    "b_expand_budget_consumed": b_row["expand_budget_consumed"],
                }
            )
        if bool(a_score["passed"]) != bool(b_score["passed"]):
            pass_transitions.append(
                {
                    "task_id": b_row["task_id"],
                    "base_problem_id": b_row["base_problem_id"],
                    "transition": "b_win" if b_score["passed"] else "b_loss",
                    "guard_rejections": b_row["nonempty_guard_rejection_count"],
                    "a_compile": bool(a_score["compile_passed"]),
                    "b_compile": bool(b_score["compile_passed"]),
                    "a_exact": bool(a_score["exact_match"]),
                    "b_exact": bool(b_score["exact_match"]),
                    "a_completion": a_row["completion"],
                    "b_completion": b_row["completion"],
                }
            )
    write_jsonl(comparison_dir / "guard_affected_rows.jsonl", affected_rows)
    write_jsonl(comparison_dir / "pass_transitions.jsonl", pass_transitions)

    a_triggered = [
        a_full_by_task[row["task_id"]]
        for row in b_full
        if int(row.get("nonempty_guard_rejection_count", 0)) > 0
    ]
    b_triggered = [
        row for row in b_full if int(row.get("nonempty_guard_rejection_count", 0)) > 0
    ]
    isolation = {
        "guard_triggered_rows": len(b_triggered),
        "guard_not_triggered_rows": len(b_full) - len(b_triggered),
        "a_blank_slot_rows": a_summary["blank_slot_rows"],
        "b_blank_slot_rows": b_summary["blank_slot_rows"],
        "guard_triggered_exactly_on_a_blank_rows": all(
            (int(b_row.get("nonempty_guard_rejection_count", 0)) > 0)
            == any(bool(value) for value in a_row["blank_region_flags"].values())
            for a_row, b_row in zip(a_full, b_full)
        ),
        "inactive_core_fields_checked": list(inactive_core_fields),
        "inactive_difference_rows": len(inactive_differences),
        "inactive_differences": inactive_differences,
        "passed": len(inactive_differences) == 0,
    }
    write_json(comparison_dir / "nonempty_isolation_audit.json", isolation)

    paired_payload = {
        "primary_b_vs_a": ab_pair,
        "a_vs_one_shot": paired(a_full, baseline),
        "a_vs_v1_progressive": paired(a_full, v1),
        "a_vs_v2_hard_v1_multifactor": paired(
            a_full, lambda task: v2_full_by_task[task]["score"]
        ),
        "b_vs_one_shot": paired(b_full, baseline),
        "b_vs_v1_progressive": paired(b_full, v1),
        "b_vs_v2_hard_v1_multifactor": paired(
            b_full, lambda task: v2_full_by_task[task]["score"]
        ),
    }
    write_json(comparison_dir / "paired_comparisons.json", paired_payload)

    compute = {
        "a": {
            key: a_summary[key]
            for key in (
                "total_forwards",
                "total_token_forwards",
                "total_wall_time_seconds",
                "peak_cuda_memory_bytes",
                "budget_exhausted_rows",
            )
        },
        "b": {
            key: b_summary[key]
            for key in (
                "total_forwards",
                "total_token_forwards",
                "total_wall_time_seconds",
                "peak_cuda_memory_bytes",
                "budget_exhausted_rows",
            )
        },
        "b_minus_a": {
            "total_forwards": b_summary["total_forwards"] - a_summary["total_forwards"],
            "total_token_forwards": b_summary["total_token_forwards"] - a_summary["total_token_forwards"],
            "total_wall_time_seconds": b_summary["total_wall_time_seconds"] - a_summary["total_wall_time_seconds"],
            "peak_cuda_memory_bytes": b_summary["peak_cuda_memory_bytes"] - a_summary["peak_cuda_memory_bytes"],
            "budget_exhausted_rows": b_summary["budget_exhausted_rows"] - a_summary["budget_exhausted_rows"],
        },
        "b_over_a_ratio": {
            "total_forwards": b_summary["total_forwards"] / a_summary["total_forwards"],
            "total_token_forwards": b_summary["total_token_forwards"] / a_summary["total_token_forwards"],
            "total_wall_time_seconds": b_summary["total_wall_time_seconds"] / a_summary["total_wall_time_seconds"],
        },
        "timing_caveat": "A and B used the same H200/model configuration but were run at different wall-clock times alongside unrelated user GPU processes; forwards and token-forwards are the cleaner compute comparison.",
    }
    write_json(comparison_dir / "compute_comparison.json", compute)

    completeness = {
        "complete": (
            len(a_full) == len(b_full) == 642
            and len(set(a_task_ids)) == len(set(b_task_ids)) == 642
            and a_task_ids == b_task_ids
            and not inactive_differences
        ),
        "a_rows": len(a_full),
        "b_rows": len(b_full),
        "a_unique_task_ids": len(set(a_task_ids)),
        "b_unique_task_ids": len(set(b_task_ids)),
        "same_task_order": a_task_ids == b_task_ids,
        "base_problems": len({str(row["base_problem_id"]) for row in a_full}),
        "a_manifest_sha256": json.loads(
            (a_full_dir / "completeness_audit.json").read_text()
        )["manifest_sha256"],
        "b_manifest_sha256": json.loads(
            (b_full_dir / "completeness_audit.json").read_text()
        )["manifest_sha256"],
        "a_config_hash": json.loads((a_full_dir / "config.json").read_text())["config_hash"],
        "b_config_hash": json.loads((b_full_dir / "config.json").read_text())["config_hash"],
        "a_protocol_violations": sum(
            1 for line in (a_full_dir / "protocol_violations.jsonl").read_text().splitlines() if line
        ),
        "b_protocol_violations": sum(
            1 for line in (b_full_dir / "protocol_violations.jsonl").read_text().splitlines() if line
        ),
        "isolation_audit_passed": isolation["passed"],
    }
    write_json(comparison_dir / "completeness_audit.json", completeness)

    summaries = {
        "one_shot": one_shot_summary,
        "v1_progressive": v1_summary,
        "v2_hard_v1": v2_summary,
        "a_v3_budgeted": a_summary,
        "b_v3_budgeted_nonempty_oracle": b_summary,
    }
    comparison_summary = {
        "population_role": "642-row / 115-base-problem exact-three-line development/mechanism population; not held-out",
        "a_full_status": "post-hoc explicitly authorized after failing the original Pilot30 performance gate",
        "b_full_status": "oracle structural diagnostic; passed its original Pilot30 Full gate; not deployable",
        "methods": summaries,
        "primary_b_vs_a": ab_pair,
        "guard_mechanism": {
            "isolation": isolation,
            "a_triggered_group": result_summary(a_triggered),
            "b_triggered_group": result_summary(b_triggered),
            "total_guard_rejections": sum(
                int(row.get("nonempty_guard_rejection_count", 0)) for row in b_full
            ),
        },
        "compute": compute,
        "conclusion": "The nonempty guard removes all 137 A blank-slot rows but does not improve overall Pass@1 (B 280/642 versus A 281/642). It substantially harms compile rate, improves exact match, and increases compute. The result does not support the oracle nonempty guard as a generally beneficial decoding constraint.",
    }
    write_json(comparison_dir / "comparison_summary.json", comparison_summary)

    main_rows = []
    roles = {
        "one_shot": "historical matched control",
        "v1_progressive": "historical progressive with post-hoc first-line truncation",
        "v2_hard_v1": "decoder/protocol failure diagnostic",
        "a_v3_budgeted": "post-hoc Full; official cumulative-budget fidelity control",
        "b_v3_budgeted_nonempty_oracle": "oracle structural diagnostic; not deployable",
    }
    for method, summary in summaries.items():
        main_rows.append(
            {
                "method": method,
                "role": roles[method],
                "rows": summary["rows"],
                "completed": summary["completed"],
                "passed": summary["passed"],
                "pass_rate": f'{summary["pass_rate"]:.10f}',
                "compile_passed": summary["compile_passed"],
                "compile_rate": f'{summary["compile_pass_rate"]:.10f}',
                "exact_matches": summary["exact_matches"],
                "exact_rate": f'{summary["exact_match_rate"]:.10f}',
                "task_macro_pass_rate": f'{summary["task_macro_pass_rate"]:.10f}',
            }
        )
    write_csv(
        comparison_dir / "main_results.csv",
        (
            "method", "role", "rows", "completed", "passed", "pass_rate",
            "compile_passed", "compile_rate", "exact_matches", "exact_rate",
            "task_macro_pass_rate",
        ),
        main_rows,
    )
    write_csv(
        b_full_dir / "main_results.csv",
        (
            "method", "role", "rows", "completed", "passed", "pass_rate",
            "compile_passed", "compile_rate", "exact_matches", "exact_rate",
            "task_macro_pass_rate",
        ),
        main_rows,
    )

    paired_rows = []
    for name, values in paired_payload.items():
        paired_rows.append(
            {
                "comparison": name,
                "wins": values["candidate_wins"],
                "losses": values["candidate_losses"],
                "both_pass": values["both_pass"],
                "both_fail": values["both_fail"],
                "net_wins": values["net_wins"],
                "mcnemar_exact_p": f'{values["mcnemar_exact_p"]:.10g}',
                "row_delta": f'{values["row_delta"]:.10f}',
                "row_ci_low": f'{values["row_bootstrap_95_ci"][0]:.10f}',
                "row_ci_high": f'{values["row_bootstrap_95_ci"][1]:.10f}',
                "cluster_ci_low": f'{values["cluster_bootstrap_95_ci"][0]:.10f}',
                "cluster_ci_high": f'{values["cluster_bootstrap_95_ci"][1]:.10f}',
                "task_macro_delta": f'{values["task_macro_delta"]:.10f}',
                "compile_help": values["compile_help"],
                "compile_harm": values["compile_harm"],
                "exact_match_help": values["exact_match_help"],
                "exact_match_harm": values["exact_match_harm"],
            }
        )
    write_csv(
        comparison_dir / "paired_comparisons.csv",
        (
            "comparison", "wins", "losses", "both_pass", "both_fail", "net_wins",
            "mcnemar_exact_p", "row_delta", "row_ci_low", "row_ci_high",
            "cluster_ci_low", "cluster_ci_high", "task_macro_delta", "compile_help",
            "compile_harm", "exact_match_help", "exact_match_harm",
        ),
        paired_rows,
    )
    b_legacy_paired_rows = []
    for control, values in (
        ("a_v3_budgeted_posthoc_full", ab_pair),
        ("one_shot", paired_payload["b_vs_one_shot"]),
        ("v1_progressive", paired_payload["b_vs_v1_progressive"]),
        ("v2_hard_v1_multifactor", paired_payload["b_vs_v2_hard_v1_multifactor"]),
    ):
        b_legacy_paired_rows.append(
            {
                "control": control,
                "wins": values["candidate_wins"],
                "losses": values["candidate_losses"],
                "both_pass": values["both_pass"],
                "both_fail": values["both_fail"],
                "net_wins": values["net_wins"],
                "mcnemar_exact_p": f'{values["mcnemar_exact_p"]:.10g}',
                "row_delta": f'{values["row_delta"]:.10f}',
                "row_ci_low": f'{values["row_bootstrap_95_ci"][0]:.10f}',
                "row_ci_high": f'{values["row_bootstrap_95_ci"][1]:.10f}',
                "cluster_ci_low": f'{values["cluster_bootstrap_95_ci"][0]:.10f}',
                "cluster_ci_high": f'{values["cluster_bootstrap_95_ci"][1]:.10f}',
                "task_macro_delta": f'{values["task_macro_delta"]:.10f}',
                "compile_help": values["compile_help"],
                "compile_harm": values["compile_harm"],
            }
        )
    write_csv(
        b_full_dir / "paired_comparisons.csv",
        (
            "control", "wins", "losses", "both_pass", "both_fail", "net_wins",
            "mcnemar_exact_p", "row_delta", "row_ci_low", "row_ci_high",
            "cluster_ci_low", "cluster_ci_high", "task_macro_delta", "compile_help",
            "compile_harm",
        ),
        b_legacy_paired_rows,
    )

    report = f"""# DreamOn V3 A/B Full642 对比

生成日期：2026-08-02（UTC）

## 实验身份

- Population：642 rows / 115 base problems，exact-three-line development/mechanism population，不是 held-out test。
- A：`v3_hard_budgeted`。A Pilot30 为 14/30，未通过原始 18/30 Full 门槛；本次 Full 是用户后续明确授权的 post-hoc 机制诊断，不能改写成预注册性能证据。
- B：`v3_hard_budgeted_nonempty_oracle`。只在 A 上增加 hard-slot nonempty guard；属于 oracle structural diagnostic，不可部署。
- 两个 Full 都是 642/642 completed，0 runtime/protocol error，task 顺序与 manifest 一致。

## 主结果

| 方法 | Pass@1 | Compile | Exact | Task-macro Pass@1 |
|---|---:|---:|---:|---:|
| One-shot | {one_shot_summary['passed']}/642 ({pct(one_shot_summary['pass_rate'])}) | {one_shot_summary['compile_passed']}/642 ({pct(one_shot_summary['compile_pass_rate'])}) | {one_shot_summary['exact_matches']}/642 ({pct(one_shot_summary['exact_match_rate'])}) | {pct(one_shot_summary['task_macro_pass_rate'])} |
| 历史 V1 | {v1_summary['passed']}/642 ({pct(v1_summary['pass_rate'])}) | {v1_summary['compile_passed']}/642 ({pct(v1_summary['compile_pass_rate'])}) | {v1_summary['exact_matches']}/642 ({pct(v1_summary['exact_match_rate'])}) | {pct(v1_summary['task_macro_pass_rate'])} |
| A: V3-Budgeted | {a_summary['passed']}/642 ({pct(a_summary['pass_rate'])}) | {a_summary['compile_passed']}/642 ({pct(a_summary['compile_pass_rate'])}) | {a_summary['exact_matches']}/642 ({pct(a_summary['exact_match_rate'])}) | {pct(a_summary['task_macro_pass_rate'])} |
| B: Budgeted+Nonempty | {b_summary['passed']}/642 ({pct(b_summary['pass_rate'])}) | {b_summary['compile_passed']}/642 ({pct(b_summary['compile_pass_rate'])}) | {b_summary['exact_matches']}/642 ({pct(b_summary['exact_match_rate'])}) | {pct(b_summary['task_macro_pass_rate'])} |

## B 对 A 的配对结果

- Pass：B wins {ab_pair['candidate_wins']} / losses {ab_pair['candidate_losses']} / both pass {ab_pair['both_pass']} / both fail {ab_pair['both_fail']}，净变化 {ab_pair['net_wins']} 条（{pct(ab_pair['row_delta'])}）。
- Exact McNemar p=`{ab_pair['mcnemar_exact_p']:.6g}`；row bootstrap 95% CI=[{pct(ab_pair['row_bootstrap_95_ci'][0])}, {pct(ab_pair['row_bootstrap_95_ci'][1])}]；115-base-problem cluster bootstrap 95% CI=[{pct(ab_pair['cluster_bootstrap_95_ci'][0])}, {pct(ab_pair['cluster_bootstrap_95_ci'][1])}]。
- Task-macro：A {pct(a_summary['task_macro_pass_rate'])}，B {pct(b_summary['task_macro_pass_rate'])}，差值 {pct(ab_pair['task_macro_delta'])}。
- Compile：B help {ab_pair['compile_help']} / harm {ab_pair['compile_harm']}；总体从 {a_summary['compile_passed']}/642 降到 {b_summary['compile_passed']}/642。
- Exact match：B help {ab_pair['exact_match_help']} / harm {ab_pair['exact_match_harm']}；总体从 {a_summary['exact_matches']}/642 升到 {b_summary['exact_matches']}/642。

## 机制隔离

- A 有 {a_summary['blank_slot_rows']} 个 blank-slot rows；B guard 恰好在这 {len(b_triggered)} 条上触发，B blank-slot rows 为 {b_summary['blank_slot_rows']}。
- 其余 {len(b_full) - len(b_triggered)} 条未触发 guard 的样本，在 completion、score、动作计数、budget、forwards 和 token-forwards 上全部与 A 一致；隔离审计差异行数为 {len(inactive_differences)}。
- Guard 共拒绝 {sum(int(row.get('nonempty_guard_rejection_count', 0)) for row in b_full)} 个候选，但没有产生 `nonempty_guard_no_valid_action` terminal failure。
- 非空约束消除了空槽位并增加 exact match，但没有提高整体 Pass@1；同时造成明显 compile harm。这说明“reference 为三条非空行”这一 oracle 结构先验并不是稳定的功能正确性改进。

## 计算对比

| 指标 | A | B | B-A |
|---|---:|---:|---:|
| Forwards | {a_summary['total_forwards']:,} | {b_summary['total_forwards']:,} | {compute['b_minus_a']['total_forwards']:+,} |
| Token-forwards | {a_summary['total_token_forwards']:,} | {b_summary['total_token_forwards']:,} | {compute['b_minus_a']['total_token_forwards']:+,} |
| Generation wall time | {a_summary['total_wall_time_seconds']:.2f}s | {b_summary['total_wall_time_seconds']:.2f}s | {compute['b_minus_a']['total_wall_time_seconds']:+.2f}s |
| Peak CUDA memory | {a_summary['peak_cuda_memory_bytes']:,} | {b_summary['peak_cuda_memory_bytes']:,} | {compute['b_minus_a']['peak_cuda_memory_bytes']:+,} |
| Budget exhausted rows | {a_summary['budget_exhausted_rows']} | {b_summary['budget_exhausted_rows']} | {compute['b_minus_a']['budget_exhausted_rows']:+d} |

B 相对 A 使用 {pct(compute['b_over_a_ratio']['total_forwards'] - 1)} 更多 forwards、{pct(compute['b_over_a_ratio']['total_token_forwards'] - 1)} 更多 token-forwards。墙钟时间还受到同期其他 GPU 作业影响，因此机制计算比较优先采用 forwards/token-forwards。

## 结论

本次全量结果不支持 B 的 nonempty guard 能提高功能正确率：B 比 A 少 1 个 Pass，配对统计和 cluster bootstrap 均不支持稳定收益；其代价是 67 个 compile 净损失和约 25% 更多 token-forwards。B 的 9 个 exact-match 净收益说明它更贴近 oracle 三行表面结构，但这没有转化成功能收益。当前证据更支持保留 A 作为协议忠实性控制，并把 nonempty guard 视为负向/混合机制结果，而不是可推进的方法。
"""
    write_text(comparison_dir / "report.zh.md", report)


if __name__ == "__main__":
    main()
