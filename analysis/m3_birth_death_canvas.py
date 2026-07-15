#!/usr/bin/env python3
"""Grouped equal-forward (not equal-token) analysis for M3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.grouped_effects import write_grouped_effect_report


METHODS = ("m3_uniform_fixed_grid", "m3_birth_death_canvas")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_analysis(uniform_raw: Path, birth_raw: Path, output_dir: Path) -> dict[str, Any]:
    rows_by_method = {METHODS[0]: read_jsonl(uniform_raw), METHODS[1]: read_jsonl(birth_raw)}
    if {str(row["row_key"]) for row in rows_by_method[METHODS[0]]} != {str(row["row_key"]) for row in rows_by_method[METHODS[1]]}:
        raise RuntimeError("M3 populations differ")
    for method, rows in rows_by_method.items():
        if any(row.get("status") != "ok" for row in rows):
            raise RuntimeError(f"M3 {method} includes error rows")
        if any(int((row.get("metrics") or {}).get("actual_forward_count") or 0) != 256 for row in rows):
            raise RuntimeError(f"M3 {method} violates equal-forward accounting")
    summary = write_grouped_effect_report(
        output_dir=output_dir,
        title="M3 Birth-Death Canvas Diffusion V0",
        rows_by_method=rows_by_method,
        comparisons=((METHODS[1], METHODS[0]),),
    )
    summary["compute_claim"] = "equal_forward_only; actual token-forward totals are descriptive and may differ"
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
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
