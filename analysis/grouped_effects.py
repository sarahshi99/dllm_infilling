"""Shared cluster-aware effect summaries for the M1--M4 candidate portfolio.

The base task (`task_group`) is always the inferential cluster.  Row-level
rates are retained as descriptive accounting only.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_SEED = 20260715


def boolish(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * float(quantile)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def cluster_bootstrap(
    values: Sequence[float],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {
            "estimate": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "cluster_count": 0,
            "bootstrap_replicates": int(replicates),
            "seed": int(seed),
        }
    rng = random.Random(int(seed))
    n = len(values)
    samples = [mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(int(replicates))]
    return {
        "estimate": mean(values),
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "cluster_count": n,
        "bootstrap_replicates": int(replicates),
        "seed": int(seed),
    }


def _by_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_group"])].append(row)
    return groups


def task_accuracies(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        group: mean([float(boolish(row.get("passed"))) for row in group_rows])
        for group, group_rows in _by_group(rows).items()
    }


def _mean_metric(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float:
    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics") or {}
        for key in keys:
            value = number(metrics.get(key, row.get(key)))
            if value is not None:
                values.append(value)
                break
    return mean(values)


def method_summary(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    usable = [row for row in rows if row.get("status") == "ok"]
    task_values = task_accuracies(usable)
    ci = cluster_bootstrap(
        list(task_values.values()), replicates=bootstrap_replicates, seed=seed
    )
    return {
        "method": method,
        "row_count": len(rows),
        "ok_row_count": len(usable),
        "error_row_count": len(rows) - len(usable),
        "task_group_count": len(task_values),
        "equal_weight_task_macro_accuracy": ci["estimate"],
        "task_macro_ci_low": ci["ci_low"],
        "task_macro_ci_high": ci["ci_high"],
        "bootstrap_replicates": ci["bootstrap_replicates"],
        "bootstrap_seed": ci["seed"],
        "span_micro_accuracy_descriptive": mean(
            [float(boolish(row.get("passed"))) for row in usable]
        ),
        "mean_standalone_forward_count": _mean_metric(
            usable, ("standalone_actual_forward_count", "actual_forward_count")
        ),
        "mean_incremental_forward_count": _mean_metric(
            usable, ("shared_bank_incremental_forward_count", "actual_forward_count")
        ),
        "mean_token_forward_budget": _mean_metric(
            usable,
            ("standalone_token_budget", "actual_token_forward_budget", "token_budget"),
        ),
        "mean_wall_sec": _mean_metric(
            usable, ("total_sec_including_probe", "wall_sec", "total_sec")
        ),
        "mean_peak_memory_bytes": _mean_metric(usable, ("peak_memory_bytes",)),
    }


def paired_effect(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    method_a: str,
    method_b: str,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left = {str(row["row_key"]): row for row in rows_a if row.get("status") == "ok"}
    right = {str(row["row_key"]): row for row in rows_b if row.get("status") == "ok"}
    if set(left) != set(right):
        raise RuntimeError(f"paired methods have different rows: {method_a} vs {method_b}")
    by_group: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    help_count = harm_count = 0
    for key in sorted(left):
        a, b = left[key], right[key]
        if str(a["task_group"]) != str(b["task_group"]):
            raise RuntimeError("paired rows disagree on task_group")
        by_group[str(a["task_group"])].append((a, b))
        help_count += int(boolish(a.get("passed")) and not boolish(b.get("passed")))
        harm_count += int(not boolish(a.get("passed")) and boolish(b.get("passed")))
    task_rows: list[dict[str, Any]] = []
    for group, pairs in sorted(by_group.items()):
        accuracy_a = mean([float(boolish(a.get("passed"))) for a, _ in pairs])
        accuracy_b = mean([float(boolish(b.get("passed"))) for _, b in pairs])
        delta = accuracy_a - accuracy_b
        task_rows.append(
            {
                "task_group": group,
                "method_a": method_a,
                "method_b": method_b,
                "method_a_accuracy": accuracy_a,
                "method_b_accuracy": accuracy_b,
                "task_macro_delta": delta,
                "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie",
                "span_count": len(pairs),
            }
        )
    deltas = [float(row["task_macro_delta"]) for row in task_rows]
    ci = cluster_bootstrap(deltas, replicates=bootstrap_replicates, seed=seed)
    rng = random.Random(int(seed) + 991)
    null = [
        mean([delta if rng.randrange(2) else -delta for delta in deltas])
        for _ in range(int(bootstrap_replicates))
    ]
    observed = ci["estimate"]
    extreme = sum(abs(value) >= abs(observed) - 1e-15 for value in null)
    return (
        {
            "method_a": method_a,
            "method_b": method_b,
            "task_macro_delta": observed,
            "task_macro_ci_low": ci["ci_low"],
            "task_macro_ci_high": ci["ci_high"],
            "task_group_count": len(task_rows),
            "wins": sum(row["outcome"] == "win" for row in task_rows),
            "losses": sum(row["outcome"] == "loss" for row in task_rows),
            "ties": sum(row["outcome"] == "tie" for row in task_rows),
            "span_help": help_count,
            "span_harm": harm_count,
            "group_label_swap_two_sided_p": (extreme + 1) / (len(null) + 1),
            "permutation_replicates": int(bootstrap_replicates),
            "permutation_seed": int(seed) + 991,
        },
        task_rows,
    )


def length_bucket_summary(rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method, rows in rows_by_method.items():
        buckets = sorted({str(row.get("length_bucket", "unknown")) for row in rows})
        for bucket in buckets:
            subset = [row for row in rows if str(row.get("length_bucket", "unknown")) == bucket and row.get("status") == "ok"]
            records.append(
                {
                    "method": method,
                    "length_bucket": bucket,
                    "span_count": len(subset),
                    "task_group_count": len({str(row["task_group"]) for row in subset}),
                    "span_micro_accuracy_descriptive": mean([float(boolish(row.get("passed"))) for row in subset]),
                    "mean_forward_count": _mean_metric(subset, ("standalone_actual_forward_count", "actual_forward_count")),
                    "mean_token_forward_budget": _mean_metric(subset, ("standalone_token_budget", "actual_token_forward_budget", "token_budget")),
                }
            )
    return records


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_grouped_effect_report(
    *,
    output_dir: Path,
    title: str,
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    comparisons: Sequence[tuple[str, str]],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        method_summary(rows, method, bootstrap_replicates=bootstrap_replicates, seed=seed + index)
        for index, (method, rows) in enumerate(rows_by_method.items())
    ]
    effects: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for index, (method_a, method_b) in enumerate(comparisons):
        effect, task_rows = paired_effect(
            rows_by_method[method_a],
            rows_by_method[method_b],
            method_a=method_a,
            method_b=method_b,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 100 + index,
        )
        effects.append(effect)
        paired.extend(task_rows)
    buckets = length_bucket_summary(rows_by_method)
    summary = {
        "title": title,
        "primary_estimand": "equal_weight_base_task_macro_accuracy",
        "span_micro_role": "descriptive_only",
        "inference_unit": "task_group",
        "bootstrap_replicates": int(bootstrap_replicates),
        "method_summary": summaries,
        "paired_effects": effects,
    }
    write_csv(output_dir / "method_summary.csv", summaries)
    write_csv(output_dir / "paired_effects.csv", effects)
    write_csv(output_dir / "paired_task_outcomes.csv", paired)
    write_csv(output_dir / "length_bucket_summary.csv", buckets)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {title}",
        "",
        "Primary estimand: equal-weight base-task macro accuracy with task_group as the cluster. Span-micro accuracy is descriptive only.",
        "",
        "| Method | Task macro (95% cluster CI) | Span micro | Standalone forwards |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['method']}` | `{row['equal_weight_task_macro_accuracy']:.4f}` "
            f"[`{row['task_macro_ci_low']:.4f}`, `{row['task_macro_ci_high']:.4f}`] | "
            f"`{row['span_micro_accuracy_descriptive']:.4f}` | `{row['mean_standalone_forward_count']:.1f}` |"
        )
    lines.extend(["", "| Comparison | Task macro delta | wins/losses/ties | help/harm | label-swap p |", "|---|---:|---:|---:|---:|"])
    for row in effects:
        lines.append(
            f"| `{row['method_a']} - {row['method_b']}` | `{row['task_macro_delta']:.4f}` | "
            f"`{row['wins']}/{row['losses']}/{row['ties']}` | `{row['span_help']}/{row['span_harm']}` | "
            f"`{row['group_label_swap_two_sided_p']:.4f}` |"
        )
    (output_dir / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
