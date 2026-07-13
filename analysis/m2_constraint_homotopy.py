#!/usr/bin/env python3
"""Compact grouped analysis for M2 Constraint-Homotopy V0."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


METHODS = ("m2_gradual_constraints", "m2_abrupt_constraints")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_group"])].append(row)
    group_accuracy = [_mean([float(bool(item.get("passed"))) for item in group_rows]) for group_rows in groups.values()]
    metrics = [row.get("metrics") or {} for row in rows]
    return {
        "method": method,
        "task_group_count": len(groups),
        "equal_weight_task_macro_accuracy": _mean(group_accuracy),
        "span_micro_accuracy_descriptive": _mean([float(bool(row.get("passed"))) for row in rows]),
        "mean_forward_count": _mean([float(metric.get("actual_forward_count") or 0) for metric in metrics]),
        "mean_token_budget": _mean([float(metric.get("token_budget") or 0) for metric in metrics]),
        "mean_wall_sec": _mean([float(metric.get("total_sec_including_probe") or 0) for metric in metrics]),
    }


def run_analysis(gradual_raw: Path, abrupt_raw: Path, output_dir: Path) -> dict[str, Any]:
    by_method = {
        METHODS[0]: read_jsonl(gradual_raw),
        METHODS[1]: read_jsonl(abrupt_raw),
    }
    keys = {method: {str(row["row_key"]) for row in rows} for method, rows in by_method.items()}
    if keys[METHODS[0]] != keys[METHODS[1]]:
        raise RuntimeError("M2 gradual and abrupt populations differ")
    for method, rows in by_method.items():
        if any(row.get("status") != "ok" for row in rows):
            raise RuntimeError(f"M2 {method} includes error rows")
        if any(int((row.get("metrics") or {}).get("actual_forward_count") or 0) != 64 for row in rows):
            raise RuntimeError(f"M2 {method} violates the fixed 64-forward budget")
    gradual = {str(row["row_key"]): row for row in by_method[METHODS[0]]}
    abrupt = {str(row["row_key"]): row for row in by_method[METHODS[1]]}
    paired = []
    for key in sorted(gradual):
        left, right = gradual[key], abrupt[key]
        paired.append(
            {
                "row_key": key,
                "task_group": left["task_group"],
                "length_bucket": left["length_bucket"],
                "gradual_passed": bool(left.get("passed")),
                "abrupt_passed": bool(right.get("passed")),
                "gradual_forward_count": (left.get("metrics") or {}).get("actual_forward_count"),
                "abrupt_forward_count": (right.get("metrics") or {}).get("actual_forward_count"),
            }
        )
    wins = sum(item["gradual_passed"] and not item["abrupt_passed"] for item in paired)
    losses = sum(not item["gradual_passed"] and item["abrupt_passed"] for item in paired)
    summaries = [summarize(by_method[method], method) for method in METHODS]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_results.csv", paired)
    write_csv(output_dir / "method_summary.csv", summaries)
    summary = {
        "methods": summaries,
        "paired_gradual_vs_abrupt": {"wins": wins, "losses": losses, "net": wins - losses},
        "primary_estimand": "equal_weight_task_macro_accuracy",
        "span_micro_role": "descriptive_only",
        "claim_boundary": "candidate_method_development_not_heldout_sota",
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        "# M2 Constraint-Homotopy V0\n\n"
        "Gradual and abrupt constraints use the same 64-forward budget.\n\n"
        f"- gradual vs abrupt: `{wins}` wins / `{losses}` losses / net `{wins - losses}`\n"
        "- primary: equal-weight task macro accuracy; span-micro is descriptive.\n",
        encoding="utf-8",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Analyze M2 Constraint-Homotopy V0")
    root.add_argument("--gradual-raw", required=True)
    root.add_argument("--abrupt-raw", required=True)
    root.add_argument("--output-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(Path(args.gradual_raw).resolve(), Path(args.abrupt_raw).resolve(), Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
