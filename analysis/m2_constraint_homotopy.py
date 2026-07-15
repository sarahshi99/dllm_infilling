#!/usr/bin/env python3
"""Shared grouped analysis for M2 Constraint-Homotopy V0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.grouped_effects import write_grouped_effect_report


METHODS = ("m2_vanilla_fixed64", "m2_gradual_constraints", "m2_abrupt_constraints")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_analysis(vanilla_raw: Path, gradual_raw: Path, abrupt_raw: Path, output_dir: Path) -> dict[str, Any]:
    rows_by_method = {
        METHODS[0]: read_jsonl(vanilla_raw),
        METHODS[1]: read_jsonl(gradual_raw),
        METHODS[2]: read_jsonl(abrupt_raw),
    }
    populations = [{str(row["row_key"]) for row in rows} for rows in rows_by_method.values()]
    if len({frozenset(keys) for keys in populations}) != 1:
        raise RuntimeError("M2 method populations differ")
    for method, rows in rows_by_method.items():
        if any(row.get("status") != "ok" for row in rows):
            raise RuntimeError(f"M2 {method} includes error rows")
        if any(int((row.get("metrics") or {}).get("actual_forward_count") or 0) != 64 for row in rows):
            raise RuntimeError(f"M2 {method} violates the 64-forward budget")
    return write_grouped_effect_report(
        output_dir=output_dir,
        title="M2 Constraint-Homotopy V0",
        rows_by_method=rows_by_method,
        comparisons=((METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[1], METHODS[2])),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--vanilla-raw", required=True)
    root.add_argument("--gradual-raw", required=True)
    root.add_argument("--abrupt-raw", required=True)
    root.add_argument("--output-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(Path(args.vanilla_raw).resolve(), Path(args.gradual_raw).resolve(), Path(args.abrupt_raw).resolve(), Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
