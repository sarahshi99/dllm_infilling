#!/usr/bin/env python3
"""Shared grouped analysis for M4 best-single, assembly, and repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.grouped_effects import write_grouped_effect_report


METHODS = ("m4_best_single_particle", "m4_assembly_without_repair", "m4_assembly_with_repair")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_analysis(best_raw: Path, assembly_raw: Path, repair_raw: Path, output_dir: Path) -> dict[str, Any]:
    rows_by_method = {
        METHODS[0]: read_jsonl(best_raw),
        METHODS[1]: read_jsonl(assembly_raw),
        METHODS[2]: read_jsonl(repair_raw),
    }
    populations = [{str(row["row_key"]) for row in rows} for rows in rows_by_method.values()]
    if len({frozenset(keys) for keys in populations}) != 1:
        raise RuntimeError("M4 method populations differ")
    for method, rows in rows_by_method.items():
        if any(row.get("status") != "ok" for row in rows):
            raise RuntimeError(f"M4 {method} includes error rows")
    if any(int((row.get("metrics") or {}).get("standalone_actual_forward_count") or 0) != 576 for row in rows_by_method[METHODS[2]]):
        raise RuntimeError("M4 repair must report 512+64 standalone forwards")
    return write_grouped_effect_report(
        output_dir=output_dir,
        title="M4 Semantic Particle Assembly V0",
        rows_by_method=rows_by_method,
        comparisons=((METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[2], METHODS[1])),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
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
