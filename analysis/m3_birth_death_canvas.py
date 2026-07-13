#!/usr/bin/env python3
"""Grouped, equal-compute analysis for M3 Birth-Death Canvas Diffusion V0."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def macro(rows: Sequence[Mapping[str, Any]]) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_group"])].append(bool(row.get("passed")))
    values = [sum(group) / len(group) for group in groups.values()]
    return sum(values) / len(values) if values else 0.0


def run_analysis(uniform_raw: Path, birth_raw: Path, output_dir: Path) -> dict[str, Any]:
    uniform, birth = read_jsonl(uniform_raw), read_jsonl(birth_raw)
    by_uniform, by_birth = {str(row["row_key"]): row for row in uniform}, {str(row["row_key"]): row for row in birth}
    if set(by_uniform) != set(by_birth):
        raise RuntimeError("M3 populations differ")
    if any(int((row.get("metrics") or {}).get("actual_forward_count") or 0) != 256 for row in [*uniform, *birth] if row.get("status") == "ok"):
        raise RuntimeError("M3 equal-compute budget violation")
    paired = []
    for key in sorted(by_uniform):
        left, right = by_uniform[key], by_birth[key]
        paired.append({"row_key": key, "task_group": left["task_group"], "uniform_passed": bool(left.get("passed")), "birth_death_passed": bool(right.get("passed")), "uniform_canvas": left.get("canvas_tokens"), "birth_death_canvas": right.get("canvas_tokens")})
    wins = sum(row["birth_death_passed"] and not row["uniform_passed"] for row in paired)
    losses = sum(not row["birth_death_passed"] and row["uniform_passed"] for row in paired)
    summary = {
        "uniform_fixed_grid_task_macro_accuracy": macro(uniform),
        "birth_death_task_macro_accuracy": macro(birth),
        "uniform_span_micro_descriptive": sum(bool(row.get("passed")) for row in uniform) / len(uniform) if uniform else 0.0,
        "birth_death_span_micro_descriptive": sum(bool(row.get("passed")) for row in birth) / len(birth) if birth else 0.0,
        "paired_birth_death_vs_uniform": {"wins": wins, "losses": losses, "net": wins - losses},
        "fixed_total_forward_budget_per_method": 256,
        "primary_estimand": "equal_weight_task_macro_accuracy",
        "span_micro_role": "descriptive_only",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_results.csv", paired)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        "# M3 Birth-Death Canvas Diffusion V0\n\n"
        f"- birth-death vs uniform: `{wins}` wins / `{losses}` losses / net `{wins - losses}`\n"
        "- Both methods use exactly 256 model forwards per task.\n",
        encoding="utf-8",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Analyze M3 Birth-Death Canvas Diffusion V0")
    root.add_argument("--uniform-raw", required=True)
    root.add_argument("--birth-death-raw", required=True)
    root.add_argument("--output-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(Path(args.uniform_raw).resolve(), Path(args.birth_death_raw).resolve(), Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
