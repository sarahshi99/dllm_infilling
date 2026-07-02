#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl, metric, oracle_bucket
from experiments.action_ceiling.action_ceiling_matrix import current_branch, current_commit, git_capture


RUNS = {
    "control": "/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529",
    "midcons": "/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626",
    "route2_len32": "/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516",
    "v6_short_override": "/home/shx/projects/dllm_infilling/git_workspace/outputs_clean/full_route2_v6_short_override_gpu2_20260620_124754",
}


def passed(row: Mapping[str, Any]) -> bool:
    return bool(metric(row, "passed", False))


def task_id(row: Mapping[str, Any]) -> str:
    return str(row["task_id"])


def read_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return list(load_jsonl(str(path)))


def read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def by_task(rows: Iterable[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        tid = task_id(row)
        if tid in result:
            raise ValueError(f"duplicate task_id in {name}: {tid}")
        result[tid] = row
    return result


def pairwise(candidate: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    common = sorted(set(candidate) & set(baseline))
    counts = Counter()
    for tid in common:
        cand_pass = passed(candidate[tid])
        base_pass = passed(baseline[tid])
        if cand_pass and not base_pass:
            counts["wins"] += 1
        elif (not cand_pass) and base_pass:
            counts["losses"] += 1
        elif cand_pass and base_pass:
            counts["tie_pass"] += 1
        else:
            counts["tie_fail"] += 1
    counts["common"] = len(common)
    return dict(counts)


def bucket_rows(name: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets = sorted({oracle_bucket(metric(row, "oracle_mask_length")) for row in rows})
    out = []
    for bucket in buckets:
        subset = [row for row in rows if oracle_bucket(metric(row, "oracle_mask_length")) == bucket]
        pass_count = sum(1 for row in subset if passed(row))
        out.append(
            {
                "run": name,
                "oracle_bucket": bucket,
                "rows": len(subset),
                "pass_count": pass_count,
                "pass_rate": None if not subset else pass_count / len(subset),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(
    summary: Mapping[str, Any],
    paired_rows: Sequence[Mapping[str, Any]],
    bucket_table: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# CAL Full Same-Protocol Comparison",
        "",
        f"cal_full_status: `{summary.get('cal_full_status')}`",
        f"cal_run_dir: `{summary.get('cal_run_dir')}`",
        "",
        "## Overall",
        "",
        "| Run | Rows | Pass | Pass Rate |",
        "|---|---:|---:|---:|",
    ]
    for run in summary.get("runs", []):
        lines.append(f"| `{run['name']}` | {run['rows']} | {run['pass_count']} | {run['pass_rate']:.4f} |")
    lines.extend(
        [
            "",
            "## Paired Against CAL",
            "",
            "| Baseline | Common | CAL Wins | CAL Losses | Tie Pass | Tie Fail |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| `{row['baseline']}` | {row['common']} | {row['cal_wins']} | {row['cal_losses']} | "
            f"{row['tie_pass']} | {row['tie_fail']} |"
        )
    lines.extend(
        [
            "",
            "## Oracle Bucket Pass Rate",
            "",
            "| Run | Bucket | Rows | Pass | Rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in bucket_table:
        rate = row["pass_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.4f}"
        lines.append(f"| `{row['run']}` | `{row['oracle_bucket']}` | {row['rows']} | {row['pass_count']} | {rate_text} |")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This is a same-repository, same-protocol local CAL comparison after the 10-case sanity verdict.",
            "- It is not an external official-code reproduction claim.",
            "- Raw 1033-row JSONL outputs remain in `outputs_clean/`; this directory contains compact, auditable summaries only.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact comparison artifacts for a full local CAL run.")
    parser.add_argument("--cal-run-dir", required=True)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-root", default="analysis_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"cal_full_same_protocol_compare_{timestamp}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    run_dirs = {"cal_full": args.cal_run_dir, **RUNS}
    rows_by_name = {name: read_rows(Path(path)) for name, path in run_dirs.items()}
    maps = {name: by_task(rows, name) for name, rows in rows_by_name.items()}

    run_summary_rows = []
    for name, rows in rows_by_name.items():
        pass_count = sum(1 for row in rows if passed(row))
        run_summary_rows.append(
            {
                "name": name,
                "path": run_dirs[name],
                "rows": len(rows),
                "pass_count": pass_count,
                "pass_rate": pass_count / len(rows) if rows else 0.0,
            }
        )

    paired_rows = []
    cal_map = maps["cal_full"]
    for baseline in ("control", "midcons", "route2_len32", "v6_short_override"):
        counts = pairwise(cal_map, maps[baseline])
        paired_rows.append(
            {
                "candidate": "cal_full",
                "baseline": baseline,
                "common": counts.get("common", 0),
                "cal_wins": counts.get("wins", 0),
                "cal_losses": counts.get("losses", 0),
                "tie_pass": counts.get("tie_pass", 0),
                "tie_fail": counts.get("tie_fail", 0),
            }
        )

    bucket_table = []
    for name, rows in rows_by_name.items():
        bucket_table.extend(bucket_rows(name, rows))

    cal_summary = read_summary(Path(args.cal_run_dir))
    summary = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "cal_full_status": "completed",
        "cal_run_dir": args.cal_run_dir,
        "cal_summary": {
            "num_samples": cal_summary.get("num_samples"),
            "pass_rate": cal_summary.get("pass_rate"),
            "required_metrics": cal_summary.get("required_metrics"),
            "avg_total_sec_including_probe": cal_summary.get("avg_total_sec_including_probe"),
        },
        "runs": run_summary_rows,
        "paired_against_cal": paired_rows,
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }

    write_csv(output_dir / "run_summary.csv", run_summary_rows)
    write_csv(output_dir / "paired_comparison.csv", paired_rows)
    write_csv(output_dir / "bucket_table.csv", bucket_table)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_report(summary, paired_rows, bucket_table), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "cal_full_status": "completed"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
