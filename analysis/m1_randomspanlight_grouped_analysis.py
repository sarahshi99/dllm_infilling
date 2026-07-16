#!/usr/bin/env python3
"""Outcome-blind grouped analysis for M1 RandomSpanLight 12->148 runs.

Selection rules and cost contracts are fixed before this program reads the
completed 148-case outcome files.  The oracle arm is an offline ceiling only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.grouped_effects import write_grouped_effect_report
from analysis.phase6_abductive_bridge_v1 import select_v1_candidate


METHODS = (
    "fixed64_seed0",
    "ordinary_confidence_best_of_grid",
    "equal_compute_generic_remask",
    "m1_score_only_abductive_selector",
    "m1_dependency_cone_full",
    "oracle_ceiling_offline_only",
)
GRID_KIND = "deployable_grid"
ORACLE_KIND = "oracle_sufficient_diagnostic_ceiling"
FIXED64_FORWARDS = 64
GRID_SELECTOR_FORWARDS = 512
REFINEMENT_FORWARDS = 576
FIXED64_TOKEN_FORWARDS = 64 * 64
GRID_SELECTOR_TOKEN_FORWARDS = 2 * (16 + 32 + 64 + 128) * 64


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_wall_sec(row: Mapping[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    return float(metrics.get("total_sec_including_probe") or row.get("wall_sec") or 0.0)


def mean_confidence(row: Mapping[str, Any]) -> float:
    values = [float(value) for value in row.get("final_token_confidences") or []]
    return sum(values) / len(values) if values else 0.0


def sorted_grid(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: (int(row["canvas_tokens"]), int(row["seed"])))
    expected = {(16, 0), (16, 1), (32, 0), (32, 1), (64, 0), (64, 1), (128, 0), (128, 1)}
    observed = {(int(row["canvas_tokens"]), int(row["seed"])) for row in ordered}
    if len(ordered) != 8 or observed != expected:
        raise RuntimeError("M1 grouped analysis requires exactly the eight fixed stage-one candidates")
    return ordered


def score_only_choice(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered = sorted_grid(rows)
    context = ordered[0]
    selected, _ = select_v1_candidate(
        str(context.get("prefix_text") or ""),
        str(context.get("suffix_text") or ""),
        [
            {
                "candidate_ordinal": ordinal,
                "middle_text": str(row.get("middle_text") or ""),
                "canvas_tokens": int(row["canvas_tokens"]),
                "seed": int(row["seed"]),
            }
            for ordinal, row in enumerate(ordered)
        ],
    )
    return ordered[int(selected["candidate_ordinal"])]


def stage1_selection_record(
    method: str,
    selected: Mapping[str, Any],
    grid_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach predeclared standalone cost to one score-only selection."""
    ordered = sorted_grid(grid_rows)
    fixed64 = method == METHODS[0]
    required = (selected,) if fixed64 else tuple(ordered)
    forwards = FIXED64_FORWARDS if fixed64 else GRID_SELECTOR_FORWARDS
    token_budget = FIXED64_TOKEN_FORWARDS if fixed64 else GRID_SELECTOR_TOKEN_FORWARDS
    selected_wall = candidate_wall_sec(selected)
    total_wall = sum(candidate_wall_sec(row) for row in required)
    return {
        "row_key": str(selected["row_key"]),
        "task_group": str(selected["task_group"]),
        "length_bucket": str(selected["length_bucket"]),
        "status": str(selected.get("status") or ""),
        "passed": bool(selected.get("passed", False)),
        "candidate_key": str(selected.get("candidate_key") or ""),
        "selected_canvas_tokens": int(selected["canvas_tokens"]),
        "selected_seed": int(selected["seed"]),
        "method": method,
        "metrics": {
            "actual_forward_count": forwards,
            "standalone_actual_forward_count": forwards,
            "shared_bank_incremental_forward_count": 0,
            "token_budget": token_budget,
            "standalone_token_budget": token_budget,
            "selected_candidate_wall_sec": selected_wall,
            "required_candidate_wall_sec": total_wall,
            "total_sec_including_probe": total_wall,
        },
    }


def refinement_record(method: str, row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("metrics") or {})
    forwards = int(metrics.get("standalone_actual_forward_count") or metrics.get("actual_forward_count") or 0)
    if forwards != REFINEMENT_FORWARDS:
        raise RuntimeError(f"{method} requires standalone 576 forwards, got {forwards}")
    return {
        "row_key": str(row["row_key"]),
        "task_group": str(row["task_group"]),
        "length_bucket": str(row["length_bucket"]),
        "status": str(row.get("status") or ""),
        "passed": bool(row.get("passed", False)),
        "candidate_key": str(row.get("candidate_key") or ""),
        "selected_canvas_tokens": int(row["canvas_tokens"]),
        "selected_seed": int(row["seed"]),
        "method": method,
        "metrics": metrics,
    }


def oracle_record(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("metrics") or {})
    forwards = int(metrics.get("actual_forward_count") or row.get("total_steps") or 0)
    if forwards != FIXED64_FORWARDS:
        raise RuntimeError("oracle ceiling must retain its 64-forward diagnostic decode")
    canvas = int(row["canvas_tokens"])
    return {
        "row_key": str(row["row_key"]),
        "task_group": str(row["task_group"]),
        "length_bucket": str(row["length_bucket"]),
        "status": str(row.get("status") or ""),
        "passed": bool(row.get("passed", False)),
        "candidate_key": str(row.get("candidate_key") or ""),
        "selected_canvas_tokens": canvas,
        "selected_seed": int(row["seed"]),
        "method": METHODS[5],
        "offline_only": True,
        "metrics": {
            **metrics,
            "actual_forward_count": forwards,
            "standalone_actual_forward_count": forwards,
            "shared_bank_incremental_forward_count": 0,
            "token_budget": canvas * forwards,
            "standalone_token_budget": canvas * forwards,
            "total_sec_including_probe": candidate_wall_sec(row),
        },
    }


def unique_index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    counts = Counter(str(row.get("row_key") or "") for row in rows)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if duplicates:
        raise RuntimeError(f"{name} has duplicate or blank row keys: {duplicates[:5]}")
    return {str(row["row_key"]): row for row in rows}


def assemble_methods(
    stage1_rows: Sequence[Mapping[str, Any]],
    generic_rows: Sequence[Mapping[str, Any]],
    m1_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    stage_grid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    stage_oracle: list[Mapping[str, Any]] = []
    for row in stage1_rows:
        kind = row.get("candidate_kind")
        if kind == GRID_KIND:
            stage_grid[str(row["row_key"])].append(row)
        elif kind == ORACLE_KIND:
            stage_oracle.append(row)
    if not stage_grid:
        raise RuntimeError("M1 stage-one raw has no deployable grid rows")
    oracle_by_key = unique_index(stage_oracle, "oracle stage-one raw")
    generic_by_key = unique_index(generic_rows, "generic refinement raw")
    m1_by_key = unique_index(m1_rows, "M1 refinement raw")
    expected = set(stage_grid)
    if set(oracle_by_key) != expected or set(generic_by_key) != expected or set(m1_by_key) != expected:
        raise RuntimeError("M1 grouped analysis requires matched stage-one/oracle/generic/M1 populations")

    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row_key in sorted(expected):
        grid = sorted_grid(stage_grid[row_key])
        if any(row.get("status") != "ok" for row in grid):
            raise RuntimeError(f"stage-one grid contains an error row for {row_key}")
        if generic_by_key[row_key].get("status") != "ok" or m1_by_key[row_key].get("status") != "ok":
            raise RuntimeError(f"M1 refinement contains an error row for {row_key}")
        if oracle_by_key[row_key].get("status") != "ok":
            raise RuntimeError(f"oracle ceiling contains an error row for {row_key}")
        fixed64 = next(row for row in grid if int(row["canvas_tokens"]) == 64 and int(row["seed"]) == 0)
        confidence = max(grid, key=lambda row: (mean_confidence(row), -int(row["canvas_tokens"]), -int(row["seed"])))
        rows_by_method[METHODS[0]].append(stage1_selection_record(METHODS[0], fixed64, grid))
        rows_by_method[METHODS[1]].append(stage1_selection_record(METHODS[1], confidence, grid))
        rows_by_method[METHODS[2]].append(refinement_record(METHODS[2], generic_by_key[row_key]))
        rows_by_method[METHODS[3]].append(stage1_selection_record(METHODS[3], score_only_choice(grid), grid))
        rows_by_method[METHODS[4]].append(refinement_record(METHODS[4], m1_by_key[row_key]))
        rows_by_method[METHODS[5]].append(oracle_record(oracle_by_key[row_key]))
    return rows_by_method


def write_figure_data(output_dir: Path, summary: Mapping[str, Any], peak_memory_bytes: int | None) -> None:
    fields = [
        "method", "task_macro_accuracy", "task_macro_ci_low", "task_macro_ci_high",
        "span_micro_accuracy_descriptive", "standalone_forwards", "token_forward_budget",
        "wall_sec", "run_peak_cuda_memory_bytes",
    ]
    with (output_dir / "figure_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["method_summary"]:
            writer.writerow(
                {
                    "method": row["method"],
                    "task_macro_accuracy": row["equal_weight_task_macro_accuracy"],
                    "task_macro_ci_low": row["task_macro_ci_low"],
                    "task_macro_ci_high": row["task_macro_ci_high"],
                    "span_micro_accuracy_descriptive": row["span_micro_accuracy_descriptive"],
                    "standalone_forwards": row["mean_standalone_forward_count"],
                    "token_forward_budget": row["mean_token_forward_budget"],
                    "wall_sec": row["mean_wall_sec"],
                    "run_peak_cuda_memory_bytes": peak_memory_bytes if peak_memory_bytes is not None else "",
                }
            )


def peak_memory_from_run_manifest(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    full = payload.get("full") or {}
    smoke = payload.get("smoke") or {}
    value = full.get("peak_cuda_memory_bytes", smoke.get("peak_cuda_memory_bytes"))
    return int(value) if value is not None else None


def run_analysis(
    stage1_raw: Path,
    generic_raw: Path,
    m1_raw: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = 10_000,
    run_manifest: Path | None = None,
) -> dict[str, Any]:
    rows_by_method = assemble_methods(read_jsonl(stage1_raw), read_jsonl(generic_raw), read_jsonl(m1_raw))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = write_grouped_effect_report(
        output_dir=output_dir,
        title="M1 Abductive Program-State Bridge / RandomSpanLight",
        rows_by_method=rows_by_method,
        comparisons=tuple((method, METHODS[0]) for method in METHODS[1:]),
        bootstrap_replicates=bootstrap_replicates,
    )
    peak_memory = peak_memory_from_run_manifest(run_manifest)
    summary.update(
        {
            "analysis_protocol": "fixed_before_reading_148_case_outcomes",
            "primary_estimand": "equal_weight_base_task_macro_accuracy",
            "span_micro_role": "descriptive_only",
            "inference": "10000 fixed-seed task-group cluster bootstrap and group-aware paired label-swap test",
            "standalone_cost_contract": {
                METHODS[0]: {"forwards": FIXED64_FORWARDS, "token_forwards": FIXED64_TOKEN_FORWARDS},
                METHODS[1]: {"forwards": GRID_SELECTOR_FORWARDS, "token_forwards": GRID_SELECTOR_TOKEN_FORWARDS},
                METHODS[3]: {"forwards": GRID_SELECTOR_FORWARDS, "token_forwards": GRID_SELECTOR_TOKEN_FORWARDS},
                METHODS[2]: {"forwards": REFINEMENT_FORWARDS},
                METHODS[4]: {"forwards": REFINEMENT_FORWARDS},
            },
            "oracle_role": "offline ceiling only; never deployable or a compute-matched control",
            "run_peak_cuda_memory_bytes": peak_memory,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_figure_data(output_dir, summary, peak_memory)
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--stage1-raw", required=True)
    root.add_argument("--generic-raw", required=True)
    root.add_argument("--m1-raw", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--run-manifest")
    root.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(
        Path(args.stage1_raw).resolve(),
        Path(args.generic_raw).resolve(),
        Path(args.m1_raw).resolve(),
        Path(args.output_dir).resolve(),
        bootstrap_replicates=int(args.bootstrap_replicates),
        run_manifest=Path(args.run_manifest).resolve() if args.run_manifest else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
