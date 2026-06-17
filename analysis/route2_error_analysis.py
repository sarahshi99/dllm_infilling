#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl, metric


JsonDict = Dict[str, Any]


NUMERIC_FEATURES = [
    "baseline_selected_length",
    "route2_primary_selected_length",
    "route2_selected_length",
    "route2_rescue_length",
    "route2_trace_top1_last",
    "route2_trace_top1_median",
    "route2_trace_confidence_max",
    "route2_trace_max_remaining_plateau_steps",
    "route2_best_len",
    "route2_best_long_len",
    "route2_long_score_max",
    "route2_long_ratio",
]


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    parsed = _num(value)
    if parsed is None:
        return None
    return int(parsed)


def _bool(value: Any) -> bool:
    return bool(value)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def bucket_oracle_length(oracle_length: Optional[int]) -> str:
    if oracle_length is None:
        return "unknown"
    oracle = int(oracle_length)
    if oracle <= 8:
        return "<=8"
    if oracle <= 12:
        return "9-12"
    if oracle <= 16:
        return "13-16"
    if oracle <= 24:
        return "17-24"
    return "25+"


def classify_pairwise(*, route_passed: bool, baseline_passed: bool) -> str:
    if route_passed and not baseline_passed:
        return "win"
    if (not route_passed) and baseline_passed:
        return "loss"
    if route_passed and baseline_passed:
        return "tie_pass"
    return "tie_fail"


def rescue_length_sufficient(*, rescue_length: Optional[int], oracle_length: Optional[int]) -> Optional[bool]:
    if rescue_length is None or oracle_length is None:
        return None
    return int(rescue_length) >= int(oracle_length)


def _oracle_len(baseline_row: Mapping[str, Any], route_row: Mapping[str, Any]) -> Optional[int]:
    return _int(metric(route_row, "oracle_mask_length", metric(baseline_row, "oracle_mask_length")))


def _route_triggered(route_row: Mapping[str, Any]) -> bool:
    return _bool(metric(route_row, "route2_trace_rescue_triggered", False))


def classify_row(baseline_row: Mapping[str, Any], route_row: Mapping[str, Any]) -> JsonDict:
    baseline_passed = _bool(metric(baseline_row, "passed", False))
    route_passed = _bool(metric(route_row, "passed", False))
    oracle_len = _oracle_len(baseline_row, route_row)
    true_long = bool(oracle_len is not None and oracle_len >= 17)
    short_or_medium = bool(oracle_len is not None and oracle_len < 17)
    triggered = _route_triggered(route_row)
    pairwise = classify_pairwise(route_passed=route_passed, baseline_passed=baseline_passed)

    classes: List[str] = []
    if pairwise == "win":
        classes.append("route2_win")
    elif pairwise == "loss":
        classes.append("route2_loss")
    elif pairwise == "tie_pass":
        classes.append("tie_pass")
    else:
        classes.append("tie_fail")

    if true_long and triggered and not route_passed:
        classes.append("triggered_failed_long")
    if true_long and triggered and (not baseline_passed) and route_passed:
        classes.append("triggered_rescued_long")
    if true_long and (not baseline_passed) and not triggered:
        classes.append("missed_failed_long")
    if short_or_medium and (not baseline_passed) and route_passed:
        classes.append("short_or_medium_win")
    if (not baseline_passed) and (not route_passed):
        classes.append("unchanged_fail")

    rescue_len = _int(metric(route_row, "route2_rescue_length"))
    return {
        "pairwise": pairwise,
        "classes": classes,
        "oracle_length": oracle_len,
        "oracle_bucket": bucket_oracle_length(oracle_len),
        "route2_triggered": triggered,
        "route2_rescue_length": rescue_len,
        "rescue_len_ge_oracle": rescue_length_sufficient(
            rescue_length=rescue_len,
            oracle_length=oracle_len,
        ),
        "baseline_passed": baseline_passed,
        "route2_passed": route_passed,
    }


def feature_row(baseline_row: Mapping[str, Any], route_row: Mapping[str, Any]) -> JsonDict:
    oracle_len = _oracle_len(baseline_row, route_row)
    return {
        "task_id": str(route_row.get("task_id")),
        "oracle_bucket": bucket_oracle_length(oracle_len),
        "baseline_selected_length": _int(metric(baseline_row, "selected_mask_length")),
        "route2_primary_selected_length": _int(metric(route_row, "route2_primary_selected_mask_length")),
        "route2_selected_length": _int(metric(route_row, "selected_mask_length")),
        "route2_rescue_length": _int(metric(route_row, "route2_rescue_length")),
        "route2_selected_minus_oracle_length": _int(metric(route_row, "selected_minus_oracle_length")),
        "route2_abs_selected_minus_oracle_length": _int(metric(route_row, "abs_selected_minus_oracle_length")),
        "route2_primary_stop_reason": metric(route_row, "route2_primary_stop_reason"),
        "route2_final_source": metric(route_row, "route2_final_source"),
        "route2_trace_top1_last": _num(metric(route_row, "route2_trace_top1_last")),
        "route2_trace_top1_median": _num(metric(route_row, "route2_trace_top1_median")),
        "route2_trace_confidence_max": _num(metric(route_row, "route2_trace_confidence_max")),
        "route2_trace_max_remaining_plateau_steps": _num(
            metric(route_row, "route2_trace_max_remaining_plateau_steps")
        ),
        "route2_best_len": _num(metric(route_row, "best_len")),
        "route2_best_long_len": _num(metric(route_row, "best_long_len")),
        "route2_long_score_max": _num(metric(route_row, "best_long_score", metric(route_row, "long_score_max"))),
        "route2_long_ratio": _num(metric(route_row, "long_ratio")),
    }


def taxonomy_row(baseline_row: Mapping[str, Any], route_row: Mapping[str, Any]) -> JsonDict:
    labels = classify_row(baseline_row, route_row)
    features = feature_row(baseline_row, route_row)
    oracle_len = labels["oracle_length"]
    return {
        **features,
        "classes": ";".join(labels["classes"]),
        "pairwise": labels["pairwise"],
        "oracle_length": oracle_len,
        "baseline_passed": labels["baseline_passed"],
        "route2_passed": labels["route2_passed"],
        "route2_triggered": labels["route2_triggered"],
        "rescue_len_ge_oracle": labels["rescue_len_ge_oracle"],
    }


def _rows_by_task(rows: Iterable[Mapping[str, Any]], *, name: str) -> Dict[str, Mapping[str, Any]]:
    by_task: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id is None:
            raise ValueError(f"Missing task_id in {name}")
        by_task[str(task_id)] = row
    return by_task


def _counter_to_ordered_dict(counter: Counter[str], keys: Sequence[str]) -> Dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in keys}


def _class_contains(row: Mapping[str, Any], class_name: str) -> bool:
    classes = str(row.get("classes", "")).split(";")
    return class_name in classes


def _feature_contrasts(taxonomy: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    interesting_classes = [
        "route2_win",
        "short_or_medium_win",
        "triggered_rescued_long",
        "triggered_failed_long",
        "missed_failed_long",
        "unchanged_fail",
    ]
    rows: List[JsonDict] = []
    for class_name in interesting_classes:
        subset = [row for row in taxonomy if _class_contains(row, class_name)]
        for feature in NUMERIC_FEATURES:
            values = [parsed for parsed in (_num(row.get(feature)) for row in subset) if parsed is not None]
            if not values:
                continue
            rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "count": len(values),
                    "mean": mean(values),
                    "median": median(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return rows


def _decide(summary: Mapping[str, Any]) -> Tuple[str, str]:
    class_counts = summary.get("class_counts") or {}
    triggered_failed = int(class_counts.get("triggered_failed_long", 0))
    missed_failed = int(class_counts.get("missed_failed_long", 0))
    sufficient = int(summary.get("triggered_failed_long_rescue_ge_oracle", 0))
    insufficient = triggered_failed - sufficient

    if triggered_failed and sufficient >= max(insufficient, 1) and missed_failed:
        return (
            "mixed_rescue_quality_and_gate_recall",
            "rescue_generation_quality+gate_recall",
        )
    if triggered_failed and sufficient > insufficient:
        return ("rescue_generation_quality", "rescue_generation_quality")
    if missed_failed:
        return ("gate_recall", "gate_recall")
    if triggered_failed and insufficient > sufficient:
        return ("length_insufficiency", "adaptive_length")
    if int(class_counts.get("route2_win", 0)) > 0:
        return ("incremental_safe_gain", "route2_polish_only")
    return ("no_useful_error_signal", "stop_true_long_current_signals")


def summarize_joined_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    route2_rows: Sequence[Mapping[str, Any]],
) -> Tuple[JsonDict, List[JsonDict]]:
    baseline = _rows_by_task(baseline_rows, name="baseline")
    route2 = _rows_by_task(route2_rows, name="route2")
    common_task_ids = sorted(set(baseline).intersection(route2))
    taxonomy = [taxonomy_row(baseline[task_id], route2[task_id]) for task_id in common_task_ids]

    pairwise_counts = Counter(str(row["pairwise"]) for row in taxonomy)
    class_counts: Counter[str] = Counter()
    bucket_pairwise: Dict[str, Counter[str]] = defaultdict(Counter)
    trigger_by_bucket: Counter[str] = Counter()
    for row in taxonomy:
        bucket = str(row.get("oracle_bucket"))
        bucket_pairwise[bucket][str(row["pairwise"])] += 1
        if row.get("route2_triggered"):
            trigger_by_bucket[bucket] += 1
        for class_name in str(row.get("classes", "")).split(";"):
            if class_name:
                class_counts[class_name] += 1

    triggered_failed_long = [row for row in taxonomy if _class_contains(row, "triggered_failed_long")]
    rescue_ge_oracle = sum(1 for row in triggered_failed_long if row.get("rescue_len_ge_oracle") is True)
    rescue_lt_oracle = sum(1 for row in triggered_failed_long if row.get("rescue_len_ge_oracle") is False)
    rescue_unknown = sum(1 for row in triggered_failed_long if row.get("rescue_len_ge_oracle") is None)

    summary: JsonDict = {
        "joined_rows": len(common_task_ids),
        "baseline_rows": len(baseline_rows),
        "route2_rows": len(route2_rows),
        "pairwise_counts": _counter_to_ordered_dict(pairwise_counts, ["win", "loss", "tie_pass", "tie_fail"]),
        "class_counts": dict(sorted((key, int(value)) for key, value in class_counts.items())),
        "triggered_count": sum(1 for row in taxonomy if row.get("route2_triggered")),
        "triggered_by_bucket": dict(sorted((key, int(value)) for key, value in trigger_by_bucket.items())),
        "bucket_pairwise": {
            bucket: _counter_to_ordered_dict(counter, ["win", "loss", "tie_pass", "tie_fail"])
            for bucket, counter in sorted(bucket_pairwise.items())
        },
        "triggered_failed_long_rescue_ge_oracle": rescue_ge_oracle,
        "triggered_failed_long_rescue_lt_oracle": rescue_lt_oracle,
        "triggered_failed_long_rescue_unknown": rescue_unknown,
    }
    bottleneck, recommendation = _decide(summary)
    summary["dominant_bottleneck"] = bottleneck
    summary["recommended_next_path"] = recommendation
    return summary, taxonomy


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report_markdown(summary: Mapping[str, Any], taxonomy: Sequence[Mapping[str, Any]]) -> str:
    class_counts = summary.get("class_counts") or {}
    pairwise = summary.get("pairwise_counts") or {}
    lines = [
        "# Route2 Error Analysis",
        "",
        f"Decision: `{summary.get('dominant_bottleneck')}`",
        "",
        f"Recommended next path: `{summary.get('recommended_next_path')}`",
        "",
        "## Overview",
        "",
        f"- Joined rows: `{summary.get('joined_rows')}`",
        f"- Pairwise W/L/TP/TF: `{pairwise.get('win', 0)}/{pairwise.get('loss', 0)}/{pairwise.get('tie_pass', 0)}/{pairwise.get('tie_fail', 0)}`",
        f"- Route2 triggers: `{summary.get('triggered_count')}`",
        f"- Triggered failed-long rows: `{class_counts.get('triggered_failed_long', 0)}`",
        f"- Missed failed-long rows: `{class_counts.get('missed_failed_long', 0)}`",
        f"- Rescue length >= oracle among triggered failed-long: `{summary.get('triggered_failed_long_rescue_ge_oracle')}/{class_counts.get('triggered_failed_long', 0)}`",
        "",
        "## Bucket Pairwise",
        "",
        "| Bucket | Win | Loss | Tie pass | Tie fail | Triggers |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    triggered_by_bucket = summary.get("triggered_by_bucket") or {}
    for bucket, counts in (summary.get("bucket_pairwise") or {}).items():
        lines.append(
            f"| `{bucket}` | `{counts.get('win', 0)}` | `{counts.get('loss', 0)}` | "
            f"`{counts.get('tie_pass', 0)}` | `{counts.get('tie_fail', 0)}` | "
            f"`{triggered_by_bucket.get(bucket, 0)}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The diagnostic separates gate recall from rescue quality. If most triggered failed-long rows already have rescue length greater than or equal to oracle length, another blind length increase is not the default next step.",
            "",
            "The current result supports a mixed next direction when both triggered failed-long and missed failed-long counts are substantial: inspect rescue generation/selection quality and expand probe-trace fusion for missed failed-long rows.",
            "",
            "## Example Rows",
            "",
            "| Class | Task | Bucket | Pairwise | Triggered | Primary len | Rescue len | Final len |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    interesting = [
        "route2_win",
        "triggered_failed_long",
        "missed_failed_long",
        "short_or_medium_win",
    ]
    for class_name in interesting:
        examples = [row for row in taxonomy if _class_contains(row, class_name)][:3]
        for row in examples:
            lines.append(
                f"| `{class_name}` | `{row.get('task_id')}` | `{row.get('oracle_bucket')}` | "
                f"`{row.get('pairwise')}` | `{row.get('route2_triggered')}` | "
                f"`{row.get('route2_primary_selected_length')}` | `{row.get('route2_rescue_length')}` | "
                f"`{row.get('route2_selected_length')}` |"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir: Path | str, summary: Mapping[str, Any], taxonomy: Sequence[Mapping[str, Any]]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    taxonomy_rows = [dict(row) for row in taxonomy]
    feature_contrasts = _feature_contrasts(taxonomy_rows)

    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "error_taxonomy.csv", taxonomy_rows)
    _write_csv(output / "triggered_failed_long.csv", [row for row in taxonomy_rows if _class_contains(row, "triggered_failed_long")])
    _write_csv(output / "missed_failed_long.csv", [row for row in taxonomy_rows if _class_contains(row, "missed_failed_long")])
    _write_csv(output / "wins.csv", [row for row in taxonomy_rows if _class_contains(row, "route2_win")])
    _write_csv(output / "feature_contrast.csv", feature_contrasts)
    (output / "report.md").write_text(_report_markdown(summary, taxonomy_rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only Route2 error analysis diagnostic.")
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--route2-results", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_rows = load_jsonl(args.baseline_results)
    route2_rows = load_jsonl(args.route2_results)
    summary, taxonomy = summarize_joined_rows(baseline_rows, route2_rows)
    summary["baseline_results"] = args.baseline_results
    summary["route2_results"] = args.route2_results
    write_outputs(Path(args.output_dir), summary, taxonomy)

    pairwise = summary["pairwise_counts"]
    class_counts = summary["class_counts"]
    print("===== Route2 Error Analysis =====", flush=True)
    print(f"output_dir: {args.output_dir}", flush=True)
    print(f"joined_rows: {summary['joined_rows']}", flush=True)
    print(
        "pairwise W/L/TP/TF: "
        f"{pairwise['win']}/{pairwise['loss']}/{pairwise['tie_pass']}/{pairwise['tie_fail']}",
        flush=True,
    )
    print(f"triggered_count: {summary['triggered_count']}", flush=True)
    print(f"triggered_failed_long: {class_counts.get('triggered_failed_long', 0)}", flush=True)
    print(f"missed_failed_long: {class_counts.get('missed_failed_long', 0)}", flush=True)
    print(
        "triggered_failed_long_rescue_ge_oracle: "
        f"{summary['triggered_failed_long_rescue_ge_oracle']}/{class_counts.get('triggered_failed_long', 0)}",
        flush=True,
    )
    print(f"dominant_bottleneck: {summary['dominant_bottleneck']}", flush=True)
    print(f"recommended_next_path: {summary['recommended_next_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
