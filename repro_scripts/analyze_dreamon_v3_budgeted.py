#!/usr/bin/env python3
from __future__ import annotations

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
    b_pilot_dir = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_pilot30"
    b_pilot = read_jsonl(b_pilot_dir / "scored.jsonl")
    b_full_dir = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_all642"
    b_full = read_jsonl(b_full_dir / "scored.jsonl")
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
            "paired_vs_a": paired(affected, lambda task: a_pilot_by_task[task]["score"]),
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
    full_analysis = {
        "population_role": "exact-three-line development/mechanism population; oracle structural diagnostic; not deployable; not held-out",
        "result": json.loads((b_full_dir / "summary.json").read_text()),
        "paired": {
            "vs_one_shot": paired(b_full, baseline),
            "vs_v1_progressive": paired(b_full, v1),
            "vs_v2_hard_v1_multifactor": paired(
                b_full, lambda task: v2_full_by_task[task]["score"]
            ),
        },
        "a_full_comparison": {
            "available": False,
            "reason": "A Pilot30 scored 14/30, below the preregistered 18/30 threshold, so A Full642 was not authorized.",
            "pilot_comparison_path": str(b_pilot_dir / "pilot_analysis.json"),
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


if __name__ == "__main__":
    main()
