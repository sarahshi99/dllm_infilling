#!/usr/bin/env python3
"""Shared grouped analysis for M4 best-single, assembly, and repair."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.grouped_effects import DEFAULT_BOOTSTRAP_REPLICATES, write_grouped_effect_report


METHODS = ("m4_best_single_particle", "m4_assembly_without_repair", "m4_assembly_with_repair")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _histogram(values: Sequence[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {key: int(counts[key]) for key in sorted(counts)}


def activation_audit(rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    best, assembly, repair = (rows_by_method[method] for method in METHODS)
    by_key = [{str(row["row_key"]): row for row in rows} for rows in (best, assembly, repair)]
    if len({frozenset(items) for items in by_key}) != 1:
        raise RuntimeError("M4 activation audit requires aligned rows")
    assembly_meta = [row.get("assembly_metadata") or {} for row in assembly]
    repair_meta = [row.get("assembly_metadata") or {} for row in repair]
    return {
        "contains_generated_code": False,
        "row_count": len(by_key[0]),
        "assembly_vs_best_candidate_hash_difference_count": sum(str(by_key[0][key].get("candidate_middle_sha256") or "") != str(by_key[1][key].get("candidate_middle_sha256") or "") for key in by_key[0]),
        "repair_vs_assembly_candidate_hash_difference_count": sum(str(by_key[2][key].get("candidate_middle_sha256") or "") != str(by_key[1][key].get("candidate_middle_sha256") or "") for key in by_key[0]),
        "fragment_count_distribution": _histogram([meta.get("fragment_count", 0) for meta in assembly_meta]),
        "provider_candidate_count_distribution": _histogram([meta.get("provider_candidate_count", 0) for meta in assembly_meta]),
        "assembly_fallback_count": sum(bool(meta.get("assembly_fallback_to_best")) for meta in assembly_meta),
        "repair_fallback_count": sum(bool(meta.get("assembly_fallback_to_best")) for meta in repair_meta),
        "connector_index_count_distribution": _histogram([len(row.get("connector_indices") or []) for row in repair]),
    }


def write_frontier(output_dir: Path, summary: Mapping[str, Any]) -> None:
    rows = [
        {
            "method": item["method"],
            "equal_weight_task_macro_accuracy": item["equal_weight_task_macro_accuracy"],
            "actual_forwards": item["mean_standalone_forward_count"],
            "actual_token_forward_budget": item["mean_token_forward_budget"],
            "mean_wall_sec": item["mean_wall_sec"],
        }
        for item in summary["method_summary"]
    ]
    with (output_dir / "accuracy_cost_frontier.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(best_raw: Path, assembly_raw: Path, repair_raw: Path, output_dir: Path, *, bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES, run_manifest: Path | None = None) -> dict[str, Any]:
    rows_by_method = {
        METHODS[0]: read_jsonl(best_raw),
        METHODS[1]: read_jsonl(assembly_raw),
        METHODS[2]: read_jsonl(repair_raw),
    }
    populations = [{str(row["row_key"]) for row in rows} for rows in rows_by_method.values()]
    if len({frozenset(keys) for keys in populations}) != 1:
        raise RuntimeError("M4 method populations differ")
    for method, rows in rows_by_method.items():
        if len(rows) != 148 or len({str(row["task_group"]) for row in rows}) != 148:
            raise RuntimeError("M4 grouped analysis requires exactly 148 aligned task groups per arm")
        if any(row.get("status") != "ok" for row in rows):
            raise RuntimeError(f"M4 {method} includes error rows")
    if any(int((row.get("metrics") or {}).get("standalone_actual_forward_count") or 0) != 576 for row in rows_by_method[METHODS[2]]):
        raise RuntimeError("M4 repair must report 512+64 standalone forwards")
    summary = write_grouped_effect_report(
        output_dir=output_dir,
        title="M4 Semantic Particle Assembly V0",
        rows_by_method=rows_by_method,
        comparisons=((METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[2], METHODS[1])),
        bootstrap_replicates=bootstrap_replicates,
    )
    summary["primary_comparison"] = [METHODS[1], METHODS[0]]
    summary["cost_contract"] = {METHODS[0]: 512, METHODS[1]: 512, METHODS[2]: 576}
    if run_manifest is not None:
        full = json.loads(run_manifest.read_text(encoding="utf-8")).get("full") or {}
        summary["run_peak_cuda_memory_bytes"] = full.get("peak_cuda_memory_bytes")
        summary["run_full_wall_sec"] = full.get("wall_sec")
    activation = activation_audit(rows_by_method)
    (output_dir / "activation_audit.json").write_text(json.dumps(activation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_frontier(output_dir, summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--best-raw", required=True)
    root.add_argument("--assembly-raw", required=True)
    root.add_argument("--repair-raw", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    root.add_argument("--run-manifest")
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(Path(args.best_raw).resolve(), Path(args.assembly_raw).resolve(), Path(args.repair_raw).resolve(), Path(args.output_dir).resolve(), bootstrap_replicates=int(args.bootstrap_replicates), run_manifest=Path(args.run_manifest).resolve() if args.run_manifest else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
