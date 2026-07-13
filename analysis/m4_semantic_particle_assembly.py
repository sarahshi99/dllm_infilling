#!/usr/bin/env python3
"""Grouped comparison for M4 best-single, assembly, and fixed-budget repair."""

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


def macro(rows: Sequence[Mapping[str, Any]]) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_group"])].append(bool(row.get("passed")))
    values = [sum(values) / len(values) for values in groups.values()]
    return sum(values) / len(values) if values else 0.0


def run_analysis(best_raw: Path, assembly_raw: Path, repair_raw: Path, output_dir: Path) -> dict[str, Any]:
    by_method = {
        "best_single_particle": read_jsonl(best_raw),
        "assembly_without_repair": read_jsonl(assembly_raw),
        "assembly_with_repair": read_jsonl(repair_raw),
    }
    populations = [{str(row["row_key"]) for row in rows} for rows in by_method.values()]
    if len({frozenset(population) for population in populations}) != 1:
        raise RuntimeError("M4 method populations differ")
    if any(int((row.get("metrics") or {}).get("actual_forward_count") or 0) != 64 for row in by_method["assembly_with_repair"] if row.get("status") == "ok"):
        raise RuntimeError("M4 repair budget is not 64 forwards")
    paired = []
    index = {name: {str(row["row_key"]): row for row in rows} for name, rows in by_method.items()}
    for key in sorted(index["best_single_particle"]):
        paired.append({"row_key": key, "task_group": index["best_single_particle"][key]["task_group"], **{f"{name}_passed": bool(rows[key].get("passed")) for name, rows in index.items()}})
    repair_wins = sum(row["assembly_with_repair_passed"] and not row["best_single_particle_passed"] for row in paired)
    repair_losses = sum(not row["assembly_with_repair_passed"] and row["best_single_particle_passed"] for row in paired)
    summary = {
        "task_macro_accuracy": {name: macro(rows) for name, rows in by_method.items()},
        "span_micro_descriptive": {name: sum(bool(row.get("passed")) for row in rows) / len(rows) if rows else 0.0 for name, rows in by_method.items()},
        "repair_vs_best_single": {"wins": repair_wins, "losses": repair_losses, "net": repair_wins - repair_losses},
        "repair_forward_budget": 64,
        "primary_estimand": "equal_weight_task_macro_accuracy",
        "span_micro_role": "descriptive_only",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_results.csv", paired)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "# M4 Semantic Particle Assembly V0\n\n"
        f"- repair vs best-single: `{repair_wins}` wins / `{repair_losses}` losses / net `{repair_wins - repair_losses}`\n"
        "- only assembly-with-repair has the fixed 64-forward repair decode.\n",
        encoding="utf-8",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Analyze M4 Semantic Particle Assembly V0")
    root.add_argument("--best-raw", required=True)
    root.add_argument("--assembly-raw", required=True)
    root.add_argument("--repair-raw", required=True)
    root.add_argument("--output-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(Path(args.best_raw).resolve(), Path(args.assembly_raw).resolve(), Path(args.repair_raw).resolve(), Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
