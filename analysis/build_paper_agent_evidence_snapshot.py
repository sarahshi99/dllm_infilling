#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis import diagnose_long_underestimate_policy as long_diag


BUCKET_ORDER = ("<=8", "9-12", "13-16", "17-24", "25+")
DEFAULT_OUTPUT_JSON = Path("docs/paper_agent/evidence_snapshot.json")
DEFAULT_OUTPUT_MD = Path("docs/paper_agent/evidence_snapshot.md")

DEFAULT_RUNS = {
    "a6000_control": "/home/shx/projects/dllm_infilling/outputs_clean/"
    "full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529/"
    "results.jsonl",
    "midcons": "/home/shx/projects/dllm_infilling/outputs_clean/"
    "full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626/"
    "results.jsonl",
    "mid_precision": "/home/shx/projects/dllm_infilling/outputs_clean/"
    "full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517/"
    "results.jsonl",
    "true_long": "/home/shx/projects/dllm_infilling/outputs_clean/"
    "full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755/"
    "results.jsonl",
    "combined": "/home/shx/projects/dllm_infilling/outputs_clean/"
    "full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756/"
    "results.jsonl",
}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "task_id" not in row:
                raise ValueError(f"Missing task_id in {path} line {line_no}")
            rows.append(row)
    return rows


def metric(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    metrics = row.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        return default
    return metrics.get(key, default)


def passed(row: Mapping[str, Any]) -> bool:
    return bool(metric(row, "passed", False))


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def oracle_bucket(length: int | None) -> str:
    if length is None:
        return "unknown"
    if length <= 8:
        return "<=8"
    if length <= 12:
        return "9-12"
    if length <= 16:
        return "13-16"
    if length <= 24:
        return "17-24"
    return "25+"


def ordered_counter(counter: Counter[str]) -> Dict[str, int]:
    ordered: Dict[str, int] = {}
    for key in BUCKET_ORDER:
        if counter.get(key, 0):
            ordered[key] = int(counter[key])
    for key in sorted(counter):
        if key not in ordered:
            ordered[key] = int(counter[key])
    return ordered


def ordered_rates(counts: Counter[str], passes: Counter[str]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key in BUCKET_ORDER:
        if counts.get(key, 0):
            values[key] = float(passes[key]) / float(counts[key])
    for key in sorted(counts):
        if key not in values and counts[key]:
            values[key] = float(passes[key]) / float(counts[key])
    return values


def summarize_run(name: str, raw_path: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bucket_counts: Counter[str] = Counter()
    bucket_passes: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for row in rows:
        bucket = oracle_bucket(optional_int(metric(row, "oracle_mask_length")))
        bucket_counts[bucket] += 1
        if passed(row):
            bucket_passes[bucket] += 1
        source_counts[str(metric(row, "final_source", "unknown"))] += 1

    pass_count = sum(1 for row in rows if passed(row))
    return {
        "name": name,
        "raw_path": raw_path,
        "rows": len(rows),
        "pass_count": pass_count,
        "pass_rate": rate(pass_count, len(rows)),
        "oracle_bucket_counts": ordered_counter(bucket_counts),
        "oracle_bucket_pass_rates": ordered_rates(bucket_counts, bucket_passes),
        "final_source_counts": dict(sorted(source_counts.items())),
    }


def _index_by_task_id(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in indexed:
            raise ValueError(f"Duplicate task_id: {task_id}")
        indexed[task_id] = row
    return indexed


def _pairwise_row(task_id: str, base: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "oracle_mask_length": optional_int(metric(candidate, "oracle_mask_length")),
        "bucket": oracle_bucket(optional_int(metric(candidate, "oracle_mask_length"))),
        "base_selected_length": optional_int(metric(base, "selected_mask_length")),
        "candidate_selected_length": optional_int(metric(candidate, "selected_mask_length")),
        "candidate_source": str(metric(candidate, "final_source", "unknown")),
    }


def compare_runs(
    base_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    base_by_id = _index_by_task_id(base_rows)
    candidate_by_id = _index_by_task_id(candidate_rows)
    common_ids = sorted(set(base_by_id) & set(candidate_by_id))
    missing_from_base = sorted(set(candidate_by_id) - set(base_by_id))
    missing_from_candidate = sorted(set(base_by_id) - set(candidate_by_id))

    wins: List[Dict[str, Any]] = []
    losses: List[Dict[str, Any]] = []
    wins_by_bucket: Counter[str] = Counter()
    losses_by_bucket: Counter[str] = Counter()

    for task_id in common_ids:
        base = base_by_id[task_id]
        candidate = candidate_by_id[task_id]
        base_passed = passed(base)
        candidate_passed = passed(candidate)
        if candidate_passed and not base_passed:
            item = _pairwise_row(task_id, base, candidate)
            wins.append(item)
            wins_by_bucket[item["bucket"]] += 1
        elif base_passed and not candidate_passed:
            item = _pairwise_row(task_id, base, candidate)
            losses.append(item)
            losses_by_bucket[item["bucket"]] += 1

    return {
        "common_rows": len(common_ids),
        "missing_from_base": missing_from_base,
        "missing_from_candidate": missing_from_candidate,
        "wins": len(wins),
        "losses": len(losses),
        "net": len(wins) - len(losses),
        "wins_by_bucket": ordered_counter(wins_by_bucket),
        "losses_by_bucket": ordered_counter(losses_by_bucket),
        "win_rows": wins,
        "loss_rows": losses,
    }


def summarize_long_failures(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    long_rows = [
        row
        for row in rows
        if (optional_int(metric(row, "oracle_mask_length")) is not None)
        and int(metric(row, "oracle_mask_length")) >= 17
    ]
    failed_long = [row for row in long_rows if not passed(row)]
    underselected = [
        row
        for row in failed_long
        if optional_int(metric(row, "selected_mask_length")) is not None
        and optional_int(metric(row, "oracle_mask_length")) is not None
        and int(metric(row, "selected_mask_length")) < int(metric(row, "oracle_mask_length"))
    ]
    selected_hist = Counter(str(metric(row, "selected_mask_length", "unknown")) for row in failed_long)

    return {
        "long_total": len(long_rows),
        "failed_long_total": len(failed_long),
        "underselected_failed_long": len(underselected),
        "underselected_failed_long_rate": rate(len(underselected), len(failed_long)),
        "failed_long_base_source": sum(1 for row in failed_long if metric(row, "final_source") == "base"),
        "failed_long_selected_length_histogram": dict(sorted(selected_hist.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 9999)),
    }


def summarize_long_sweep(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sweep = long_diag.sweep_rules(rows)
    strict = [
        row
        for row in sweep
        if row["trigger_count"] >= 10
        and (row["true_long_precision"] or 0.0) >= 0.60
        and (row["short_risk_rate"] or 0.0) <= 0.05
        and row["failed_long_trigger_count"] >= 10
    ]
    best = sweep[0] if sweep else None
    return {
        "evaluated_rules": len(sweep),
        "strict_viable_rules": len(strict),
        "best_rule": best["rule"] if best else None,
        "best_trigger_count": best["trigger_count"] if best else 0,
        "best_true_long_precision": best["true_long_precision"] if best else None,
        "best_failed_long_recall": best["failed_long_recall"] if best else None,
        "best_short_risk_rate": best["short_risk_rate"] if best else None,
        "best_current_pass_risk_rate": best["current_pass_risk_rate"] if best else None,
    }


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def build_snapshot(run_paths: Mapping[str, str]) -> Dict[str, Any]:
    loaded = {name: load_jsonl(path) for name, path in run_paths.items()}
    if "a6000_control" not in loaded:
        raise ValueError("run_paths must include a6000_control")

    summaries = {
        name: summarize_run(name, run_paths[name], rows)
        for name, rows in loaded.items()
    }
    comparisons = {
        name: compare_runs(loaded["a6000_control"], rows)
        for name, rows in loaded.items()
        if name != "a6000_control"
    }
    current_name = "midcons" if "midcons" in loaded else next(iter(loaded))
    return {
        "snapshot_version": 1,
        "runs": summaries,
        "comparisons_vs_a6000_control": comparisons,
        "current_checkpoint": current_name,
        "current_checkpoint_long_failures": summarize_long_failures(loaded[current_name]),
        "current_checkpoint_long_sweep": summarize_long_sweep(loaded[current_name]),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = payload["runs"]
    comparisons = payload["comparisons_vs_a6000_control"]
    current_name = payload["current_checkpoint"]
    long_failures = payload["current_checkpoint_long_failures"]
    sweep = payload["current_checkpoint_long_sweep"]

    lines = [
        "# Paper-Agent Evidence Snapshot",
        "",
        "Generated from existing local `results.jsonl` files. Raw outputs are not copied here.",
        "",
        "## Runs",
        "",
        "| Run | Rows | Pass | Rate | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in runs.items():
        bucket_rates = summary["oracle_bucket_pass_rates"]
        lines.append(
            f"| `{name}` | {summary['rows']} | {summary['pass_count']} | {pct(summary['pass_rate'])} | "
            f"{pct(bucket_rates.get('<=8'))} | {pct(bucket_rates.get('9-12'))} | "
            f"{pct(bucket_rates.get('13-16'))} | {pct(bucket_rates.get('17-24'))} | "
            f"{pct(bucket_rates.get('25+'))} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Against A6000 Control",
            "",
            "| Candidate | Common Rows | Wins | Losses | Net | Wins By Bucket | Losses By Bucket |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for name, comparison in comparisons.items():
        lines.append(
            f"| `{name}` | {comparison['common_rows']} | {comparison['wins']} | "
            f"{comparison['losses']} | {comparison['net']} | "
            f"`{comparison['wins_by_bucket']}` | `{comparison['losses_by_bucket']}` |"
        )

    lines.extend(
        [
            "",
            f"## Long-Failure Summary For `{current_name}`",
            "",
            f"- long_total: `{long_failures['long_total']}`",
            f"- failed_long_total: `{long_failures['failed_long_total']}`",
            f"- underselected_failed_long: `{long_failures['underselected_failed_long']}` "
            f"({pct(long_failures['underselected_failed_long_rate'])})",
            f"- failed_long_base_source: `{long_failures['failed_long_base_source']}`",
            f"- failed_long_selected_length_histogram: `{long_failures['failed_long_selected_length_histogram']}`",
            "",
            "## Long-Underestimate Sweep",
            "",
            f"- evaluated_rules: `{sweep['evaluated_rules']}`",
            f"- strict_viable_rules: `{sweep['strict_viable_rules']}`",
            f"- best_rule: `{sweep['best_rule']}`",
            f"- best_true_long_precision: `{pct(sweep['best_true_long_precision'])}`",
            f"- best_failed_long_recall: `{pct(sweep['best_failed_long_recall'])}`",
            f"- best_short_risk_rate: `{pct(sweep['best_short_risk_rate'])}`",
            f"- best_current_pass_risk_rate: `{pct(sweep['best_current_pass_risk_rate'])}`",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact paper-agent evidence snapshot.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=RESULTS_JSONL",
        help="Override or add a run path. Must include a6000_control if any --run is supplied.",
    )
    return parser.parse_args()


def parse_run_overrides(items: Iterable[str]) -> Dict[str, str]:
    if not items:
        return dict(DEFAULT_RUNS)
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--run must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        parsed[name] = path
    return parsed


def main() -> None:
    args = parse_args()
    run_paths = parse_run_overrides(args.run)
    payload = build_snapshot(run_paths)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    current = payload["runs"][payload["current_checkpoint"]]
    print(
        f"wrote {args.output_json} and {args.output_md}; "
        f"current={payload['current_checkpoint']} pass={current['pass_count']}/{current['rows']} "
        f"rate={pct(current['pass_rate'])}"
    )


if __name__ == "__main__":
    main()
