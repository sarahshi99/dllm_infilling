#!/usr/bin/env python3
"""Cluster-aware statistics for the official 6707-span second-regime results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1"
POLICIES = (
    "control_fixed64",
    "best_deployable_cal_lite_alpha006",
    "oracle_sufficient_canvas",
)
POLICY_LABELS = {
    "control_fixed64": "control_fixed64",
    "best_deployable_cal_lite_alpha006": "cal_lite",
    "oracle_sufficient_canvas": "oracle_ceiling",
}
SEED = 20260713


def boolish(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def floatish(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_group"])].append(row)
    return groups


def macro_accuracy(rows: Sequence[Mapping[str, Any]], policy: str) -> float:
    values = policy_group_accuracies(rows, policy)
    return sum(values) / len(values) if values else math.nan


def span_micro_accuracy(rows: Sequence[Mapping[str, Any]], policy: str) -> float:
    policy_rows = [row for row in rows if row["policy"] == policy]
    return sum(boolish(row["passed"]) for row in policy_rows) / len(policy_rows) if policy_rows else math.nan


def policy_group_accuracies(rows: Sequence[Mapping[str, Any]], policy: str) -> list[float]:
    groups = group_rows(rows)
    values: list[float] = []
    for group in sorted(groups):
        group_policy = [row for row in groups[group] if row["policy"] == policy]
        if not group_policy:
            return []
        values.append(sum(boolish(row["passed"]) for row in group_policy) / len(group_policy))
    return values


def policy_group_metric(rows: Sequence[Mapping[str, Any]], policy: str, metric) -> list[float]:
    groups = group_rows(rows)
    values: list[float] = []
    for group in sorted(groups):
        group_policy = [row for row in groups[group] if row["policy"] == policy]
        if not group_policy:
            continue
        values.append(float(metric(group_policy)))
    return values


def cluster_bootstrap_values(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    estimate = float(array.mean()) if len(array) else math.nan
    if not len(array):
        return {
            "estimate": estimate,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "bootstrap_replicates_requested": replicates,
            "bootstrap_replicates_valid": 0,
            "cluster_count": 0,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(array), size=(replicates, len(array)))
    samples = array[sample_indices].mean(axis=1)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid": int(len(samples)),
        "cluster_count": int(len(array)),
        "seed": seed,
    }


def paired_task_outcomes(rows: Sequence[Mapping[str, Any]], policy_a: str, policy_b: str) -> list[dict[str, Any]]:
    groups = group_rows(rows)
    output: list[dict[str, Any]] = []
    for group in sorted(groups):
        group_rows_ = groups[group]
        a_rows = [row for row in group_rows_ if row["policy"] == policy_a]
        b_rows = [row for row in group_rows_ if row["policy"] == policy_b]
        if not a_rows or not b_rows:
            raise ValueError(f"Missing policy in task group {group}")
        a = sum(boolish(row["passed"]) for row in a_rows) / len(a_rows)
        b = sum(boolish(row["passed"]) for row in b_rows) / len(b_rows)
        output.append(
            {
                "task_group": group,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "policy_a_accuracy": a,
                "policy_b_accuracy": b,
                "delta": a - b,
                "outcome": "win" if a > b else "loss" if a < b else "tie",
                "span_count": len(a_rows),
            }
        )
    return output


def exact_sign_test_two_sided(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    low = min(wins, losses)
    probability = sum(math.comb(total, k) for k in range(low + 1)) / (2**total)
    return min(1.0, 2.0 * probability)


def group_label_swap_test(task_rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    deltas = np.asarray([float(row["delta"]) for row in task_rows], dtype=np.float64)
    observed = float(deltas.mean()) if len(deltas) else math.nan
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(replicates, len(deltas)))
    null = (signs * deltas).mean(axis=1) if len(deltas) else np.asarray([], dtype=np.float64)
    extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-15))
    return {
        "observed_macro_delta": observed,
        "two_sided_p": (extreme + 1) / (len(null) + 1),
        "permutations_requested": replicates,
        "permutations_valid": int(len(null)),
        "cluster_count": len(task_rows),
        "seed": seed,
        "test": "task_group_label_swap_sign_flip",
    }


def policy_summary(rows: Sequence[Mapping[str, Any]], policy: str, replicates: int) -> dict[str, Any]:
    policy_rows = [row for row in rows if row["policy"] == policy]
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "span_count": len(policy_rows),
        "task_group_count": len({row["task_group"] for row in policy_rows}),
        "span_micro_accuracy": span_micro_accuracy(rows, policy),
        "task_macro_cluster_bootstrap": cluster_bootstrap_values(
            policy_group_accuracies(rows, policy),
            replicates=replicates,
            seed=SEED + POLICIES.index(policy),
        ),
        "mean_total_sec_including_probe": mean_numeric(policy_rows, "total_sec_including_probe"),
        "mean_decode_sec": mean_numeric(policy_rows, "decode_sec"),
        "mean_length_probe_sec": mean_numeric(policy_rows, "length_probe_sec"),
        "mean_selected_canvas_tokens": mean_numeric(policy_rows, "selected_canvas_tokens"),
    }


def mean_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [floatish(row.get(key)) for row in rows]
    usable = [value for value in values if math.isfinite(value)]
    return sum(usable) / len(usable) if usable else math.nan


def stratum_rows(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    value: str,
    replicates: int,
) -> list[dict[str, Any]]:
    if field == "error_type" and value == "<none>":
        subset = [row for row in rows if not str(row.get(field, ""))]
    else:
        subset = [row for row in rows if str(row.get(field, "")) == value]
    records: list[dict[str, Any]] = []
    for policy in POLICIES:
        policy_subset = [row for row in subset if row["policy"] == policy]
        records.append(
            {
                "stratum_field": field,
                "stratum_value": value,
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "span_count": len(policy_subset),
                "task_group_count": len({row["task_group"] for row in policy_subset}),
                "span_micro_accuracy": span_micro_accuracy(subset, policy),
                **cluster_bootstrap_values(
                    policy_group_metric(
                        subset,
                        policy,
                        lambda group_policy: sum(boolish(row["passed"]) for row in group_policy) / len(group_policy),
                    ),
                    replicates=replicates,
                    seed=SEED + 100 + POLICIES.index(policy),
                ),
            }
        )
    return records


def taxonomic_intersection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_index: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_index[str(row["manifest_index"])][str(row["policy"])] = row
    counts: Counter[tuple[bool, bool, bool]] = Counter()
    for index, policy_rows in by_index.items():
        if set(policy_rows) != set(POLICIES):
            raise ValueError(f"Incomplete policy intersection for manifest_index={index}")
        flags = tuple(boolish(policy_rows[policy]["passed"]) for policy in POLICIES)
        counts[flags] += 1
    out: list[dict[str, Any]] = []
    for flags in [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)]:
        control, cal_lite, oracle = flags
        out.append(
            {
                "control_passed": control,
                "cal_lite_passed": cal_lite,
                "oracle_passed": oracle,
                "intersection_label": f"control={int(control)}|cal_lite={int(cal_lite)}|oracle={int(oracle)}",
                "span_count": counts[flags],
                "interpretation": intersection_interpretation(control, cal_lite, oracle),
            }
        )
    return out


def intersection_interpretation(control: bool, cal_lite: bool, oracle: bool) -> str:
    if not control and not cal_lite and not oracle:
        return "rescue_or_noncanvas_limited"
    if not control and not cal_lite and oracle:
        return "oracle_only_canvas_recoverable"
    if not control and cal_lite and oracle:
        return "cal_lite_and_oracle_recover_control_failure"
    if not control and cal_lite and not oracle:
        return "cal_lite_recovers_but_oracle_regresses"
    if control and not cal_lite and oracle:
        return "cal_lite_harm_oracle_keeps_control"
    if control and not cal_lite and not oracle:
        return "cal_lite_and_oracle_harm_control"
    if control and cal_lite and not oracle:
        return "oracle_harm_control"
    return "all_pass"


def audit_inputs(rows: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows_by_key = {(str(row["manifest_index"]), str(row["policy"])) for row in rows}
    expected = {(str(row["manifest_index"]), policy) for row in manifest for policy in POLICIES}
    duplicate_count = len(rows) - len(rows_by_key)
    errors = [row for row in rows if row.get("status") != "ok"]
    groups = {str(row["task_group"]) for row in rows}
    return {
        "manifest_span_count": len(manifest),
        "result_row_count": len(rows),
        "expected_result_row_count": len(expected),
        "unique_result_key_count": len(rows_by_key),
        "missing_result_count": len(expected - rows_by_key),
        "extra_result_count": len(rows_by_key - expected),
        "duplicate_result_count": duplicate_count,
        "error_row_count": len(errors),
        "task_group_count": len(groups),
        "expected_task_group_count": 148,
        "frozen_rows": sum(boolish(row.get("frozen_controller_test_row")) for row in manifest),
        "passed": len(manifest) == 6707
        and len(rows) == 20121
        and len(rows_by_key) == len(expected)
        and not (expected - rows_by_key)
        and not (rows_by_key - expected)
        and duplicate_count == 0
        and len(errors) == 0
        and len(groups) == 148
        and sum(boolish(row.get("frozen_controller_test_row")) for row in manifest) == 0,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# P2.1 Grouped Statistics: Official Full-Allowed Second Regime",
        "",
        f"Input integrity: `{summary['input_audit']['passed']}`.",
        "",
        "Primary estimand: equal-weight base-task macro accuracy. For each policy and base task, calculate span accuracy within task, then average equally over task groups. Span-micro rates are descriptive only; no row-independent significance test is reported.",
        "",
        "## Overall",
        "",
        "| Policy | Task-macro accuracy (95% cluster CI) | Span-micro accuracy | Mean sec/span |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["policy_summary"]:
        ci = row["task_macro_cluster_bootstrap"]
        lines.append(
            f"| `{row['policy_label']}` | `{ci['estimate']:.4f}` [`{ci['ci_low']:.4f}`, `{ci['ci_high']:.4f}`] | "
            f"`{row['span_micro_accuracy']:.4f}` | `{row['mean_total_sec_including_probe']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Cluster Tests",
            "",
            "| Comparison | Macro delta | 95% cluster CI | Task wins/losses/ties | Label-swap p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["comparisons"]:
        ci = row["macro_delta_cluster_bootstrap"]
        paired = row["task_paired"]
        permutation = row["label_swap_permutation"]
        lines.append(
            f"| `{row['policy_a_label']} - {row['policy_b_label']}` | `{ci['estimate']:.4f}` | "
            f"[`{ci['ci_low']:.4f}`, `{ci['ci_high']:.4f}`] | `{paired['wins']}/{paired['losses']}/{paired['ties']}` | `{permutation['two_sided_p']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- The 148 base tasks are the inferential clusters; 6707 spans and 20121 policy rows remain descriptive accounting totals.",
            "- `task_paired_outcomes.csv` contains one paired accuracy comparison per task group; `strata_cluster_ci.csv` reports both span and group counts for config, length, and error strata.",
            "- `intersection_8cell.csv` is the control/CAL-lite/oracle outcome intersection. `accuracy_cost_frontier.csv` is paper-ready plotting data.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    args = parser.parse_args()

    rows = read_csv(args.input_dir / "results.csv")
    manifest = read_csv(args.input_dir / "manifest.csv")
    audit = audit_inputs(rows, manifest)
    if not audit["passed"]:
        raise RuntimeError(f"Input audit failed: {json.dumps(audit, sort_keys=True)}")

    policy_rows = [policy_summary(rows, policy, args.bootstrap_replicates) for policy in POLICIES]
    comparison_rows: list[dict[str, Any]] = []
    paired_records: list[dict[str, Any]] = []
    for policy_a, policy_b in [
        ("best_deployable_cal_lite_alpha006", "control_fixed64"),
        ("oracle_sufficient_canvas", "control_fixed64"),
        ("oracle_sufficient_canvas", "best_deployable_cal_lite_alpha006"),
    ]:
        task_rows = paired_task_outcomes(rows, policy_a, policy_b)
        paired_records.extend(task_rows)
        wins = sum(row["outcome"] == "win" for row in task_rows)
        losses = sum(row["outcome"] == "loss" for row in task_rows)
        ties = sum(row["outcome"] == "tie" for row in task_rows)
        comparison_rows.append(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "policy_a_label": POLICY_LABELS[policy_a],
                "policy_b_label": POLICY_LABELS[policy_b],
                "macro_delta_cluster_bootstrap": cluster_bootstrap_values(
                    [float(row["delta"]) for row in task_rows],
                    replicates=args.bootstrap_replicates,
                    seed=SEED + 30 + len(comparison_rows),
                ),
                "task_paired": {
                    "task_group_count": len(task_rows),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "net": wins - losses,
                    "exact_sign_test_two_sided_p": exact_sign_test_two_sided(wins, losses),
                },
                "label_swap_permutation": group_label_swap_test(
                    task_rows,
                    replicates=args.permutation_replicates,
                    seed=SEED + 60 + len(comparison_rows),
                ),
            }
        )

    strata: list[dict[str, Any]] = []
    for field in ("source_config", "length_bucket", "error_type"):
        values = sorted({str(row.get(field, "")) or "<none>" for row in rows})
        for value in values:
            strata.extend(stratum_rows(rows, field, value, args.bootstrap_replicates))

    frontier = []
    for item in policy_rows:
        ci = item["task_macro_cluster_bootstrap"]
        frontier.append(
            {
                "policy": item["policy"],
                "policy_label": item["policy_label"],
                "task_macro_accuracy": ci["estimate"],
                "task_macro_ci_low": ci["ci_low"],
                "task_macro_ci_high": ci["ci_high"],
                "span_micro_accuracy_descriptive": item["span_micro_accuracy"],
                "mean_total_sec_including_probe": item["mean_total_sec_including_probe"],
                "mean_decode_sec": item["mean_decode_sec"],
                "mean_length_probe_sec": item["mean_length_probe_sec"],
                "mean_selected_canvas_tokens": item["mean_selected_canvas_tokens"],
            }
        )
    intersections = taxonomic_intersection(rows)
    summary = {
        "analysis": "P2.1 grouped statistics",
        "input_dir": str(args.input_dir),
        "primary_estimand": "equal_weight_base_task_macro_accuracy",
        "secondary_descriptive_metric": "span_micro_accuracy",
        "prohibited": ["row_independent_significance", "row_level_independent_bootstrap"],
        "input_audit": audit,
        "policy_summary": policy_rows,
        "comparisons": comparison_rows,
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": SEED,
        "frozen_test_status": "sealed_not_touched",
        "test_evaluation_count": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "policy_summary.csv", [
        {
            "policy": row["policy"],
            "policy_label": row["policy_label"],
            "span_count": row["span_count"],
            "task_group_count": row["task_group_count"],
            "span_micro_accuracy_descriptive": row["span_micro_accuracy"],
            "task_macro_accuracy": row["task_macro_cluster_bootstrap"]["estimate"],
            "task_macro_ci_low": row["task_macro_cluster_bootstrap"]["ci_low"],
            "task_macro_ci_high": row["task_macro_cluster_bootstrap"]["ci_high"],
            "mean_total_sec_including_probe": row["mean_total_sec_including_probe"],
            "mean_decode_sec": row["mean_decode_sec"],
            "mean_length_probe_sec": row["mean_length_probe_sec"],
            "mean_selected_canvas_tokens": row["mean_selected_canvas_tokens"],
        }
        for row in policy_rows
    ])
    write_csv(args.output_dir / "pairwise_comparisons.csv", [
        {
            "policy_a": row["policy_a"],
            "policy_b": row["policy_b"],
            "macro_delta": row["macro_delta_cluster_bootstrap"]["estimate"],
            "macro_delta_ci_low": row["macro_delta_cluster_bootstrap"]["ci_low"],
            "macro_delta_ci_high": row["macro_delta_cluster_bootstrap"]["ci_high"],
            **row["task_paired"],
            "label_swap_two_sided_p": row["label_swap_permutation"]["two_sided_p"],
        }
        for row in comparison_rows
    ])
    write_csv(args.output_dir / "task_paired_outcomes.csv", paired_records)
    write_csv(args.output_dir / "strata_cluster_ci.csv", strata)
    write_csv(args.output_dir / "accuracy_cost_frontier.csv", frontier)
    write_csv(args.output_dir / "intersection_8cell.csv", intersections)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "input_audit": audit, "policy_summary": policy_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
