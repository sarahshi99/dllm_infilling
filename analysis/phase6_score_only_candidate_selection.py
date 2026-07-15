#!/usr/bin/env python3
"""Fixed score-only analysis of the completed Phase6 MultiLine candidate bank.

This is explicitly a historical M1 precursor, not a true M1 full run.  All
selectors are fixed before reading outcomes; best-of-8 is an offline ceiling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.grouped_effects import paired_effect, write_grouped_effect_report
from analysis.phase5_premise_falsification import bridge_features, deployable_proxy_scores
from analysis.phase6_abductive_bridge_v1 import select_v1_candidate


METHODS = (
    "fixed64_seed0",
    "ordinary_confidence_best_of_grid",
    "phase5_prefix_only_deterministic_proxy",
    "phase5_combined_deterministic_proxy",
    "fixed_score_only_abductive_selector",
    "best_of_8_oracle_ceiling_offline_only",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_grid_rows(path: Path) -> list[dict[str, Any]]:
    """Stream the large raw JSONL and retain only analysis-required fields."""
    rows: list[dict[str, Any]] = []
    fields = (
        "candidate_key", "row_key", "task_group", "length_bucket", "passed",
        "canvas_tokens", "seed", "prefix_text", "middle_text", "suffix_text",
        "middle_token_ids", "final_token_confidences", "metrics", "wall_sec",
        "candidate_kind", "status",
    )
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("candidate_kind") == "deployable_grid" and row.get("status") == "ok":
                rows.append({field: row.get(field) for field in fields})
    return rows


def mean_confidence(row: Mapping[str, Any]) -> float:
    values = [float(value) for value in row.get("final_token_confidences") or []]
    return sum(values) / len(values) if values else 0.0


def proxy_scores(row: Mapping[str, Any]) -> dict[str, float]:
    features = bridge_features(str(row["prefix_text"]), str(row.get("middle_text") or ""), str(row["suffix_text"]))
    canvas = max(1, int(row["canvas_tokens"]))
    return deployable_proxy_scores(
        {
            **features,
            "candidate_canvas_fill_ratio": min(1.0, len(row.get("middle_token_ids") or []) / canvas),
            "ordinary_confidence": mean_confidence(row),
        }
    )


def tie_key(row: Mapping[str, Any]) -> tuple[float, int, int]:
    return (mean_confidence(row), -int(row["canvas_tokens"]), -int(row["seed"]))


def choose_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: (int(row["canvas_tokens"]), int(row["seed"])))
    if len(ordered) != 8:
        raise RuntimeError("each span requires exactly eight deployable candidates")
    fixed64 = next(row for row in ordered if int(row["canvas_tokens"]) == 64 and int(row["seed"]) == 0)
    confidence = max(ordered, key=tie_key)
    enriched = [{**dict(row), "_scores": proxy_scores(row)} for row in ordered]
    prefix = max(enriched, key=lambda row: (float(row["_scores"]["deployable_proxy_prefix_only"]), -int(row["canvas_tokens"]), -int(row["seed"])))
    combined = max(enriched, key=lambda row: (float(row["_scores"]["deployable_proxy_combined"]), -int(row["canvas_tokens"]), -int(row["seed"])))
    selected, _ = select_v1_candidate(
        str(fixed64["prefix_text"]),
        str(fixed64["suffix_text"]),
        [
            {
                "candidate_ordinal": index,
                "middle_text": str(row.get("middle_text") or ""),
                "canvas_tokens": int(row["canvas_tokens"]),
                "seed": int(row["seed"]),
            }
            for index, row in enumerate(ordered)
        ],
    )
    abductive = ordered[int(selected["candidate_ordinal"])]
    oracle = max(ordered, key=lambda row: (int(bool(row.get("passed"))), *tie_key(row)))
    return {
        METHODS[0]: fixed64,
        METHODS[1]: confidence,
        METHODS[2]: prefix,
        METHODS[3]: combined,
        METHODS[4]: abductive,
        METHODS[5]: oracle,
    }


def selection_record(method: str, row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("metrics") or {})
    return {
        "row_key": str(row["row_key"]),
        "task_group": str(row["task_group"]),
        "length_bucket": str(row["length_bucket"]),
        "status": "ok",
        "passed": bool(row.get("passed")),
        "candidate_key": str(row["candidate_key"]),
        "selected_canvas_tokens": int(row["canvas_tokens"]),
        "selected_seed": int(row["seed"]),
        "metrics": {
            "actual_forward_count": 512,
            "standalone_actual_forward_count": 512,
            "shared_bank_incremental_forward_count": 0,
            "token_budget": 2 * (16 + 32 + 64 + 128) * 64,
            "total_sec_including_probe": float(metrics.get("total_sec_including_probe") or row.get("wall_sec") or 0.0),
        },
        "method": method,
    }


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    low = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(low + 1)) / (2**n))


def payload_for_span(row_key: str, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chosen = choose_rows(candidates)
    return {
        "row_key": row_key,
        "task_group": chosen[METHODS[0]]["task_group"],
        "length_bucket": chosen[METHODS[0]]["length_bucket"],
        "methods": {method: selection_record(method, candidate) for method, candidate in chosen.items()},
    }


def write_selection_chunk(raw_path: Path, output_path: Path, start: int, stop: int) -> int:
    raw = load_grid_rows(raw_path)
    by_span: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw:
        by_span[str(row["row_key"])].append(row)
    keys = sorted(by_span)[int(start) : int(stop)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row_key in keys:
            handle.write(json.dumps(payload_for_span(row_key, by_span[row_key]), ensure_ascii=False, sort_keys=True) + "\n")
    return len(keys)


def summarize_payloads(payloads: Sequence[Mapping[str, Any]], output_dir: Path, *, bootstrap_replicates: int) -> dict[str, Any]:
    if len({str(row["row_key"]) for row in payloads}) != 5079:
        raise RuntimeError("selection payloads must cover exactly 5079 unique spans")
    selections: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    selection_table: list[dict[str, Any]] = []
    for payload in sorted(payloads, key=lambda row: str(row["row_key"])):
        record: dict[str, Any] = {"row_key": payload["row_key"], "task_group": payload["task_group"], "length_bucket": payload["length_bucket"]}
        for method in METHODS:
            candidate = dict(payload["methods"][method])
            selections[method].append(candidate)
            record[f"{method}_candidate_key"] = candidate["candidate_key"]
            record[f"{method}_canvas"] = candidate["selected_canvas_tokens"]
            record[f"{method}_passed"] = candidate["passed"]
        selection_table.append(record)
    comparisons = [(method, METHODS[0]) for method in METHODS[1:]]
    summary = write_grouped_effect_report(output_dir=output_dir, title="Phase6 score-only candidate selection / M1 historical precursor", rows_by_method=selections, comparisons=comparisons, bootstrap_replicates=bootstrap_replicates)
    effects = []
    for index, (method, baseline) in enumerate(comparisons):
        effect, _ = paired_effect(selections[method], selections[baseline], method_a=method, method_b=baseline, bootstrap_replicates=bootstrap_replicates, seed=20260715 + index)
        effect["exact_task_sign_test_p"] = exact_sign_p(int(effect["wins"]), int(effect["losses"]))
        effect["oracle_regret_span_count"] = sum(int(oracle["passed"]) - int(candidate["passed"]) for oracle, candidate in zip(selections[METHODS[5]], selections[method]))
        effects.append(effect)
    canvas_rows = []
    for method, rows in selections.items():
        counts: dict[int, int] = defaultdict(int)
        for row in rows:
            counts[int(row["selected_canvas_tokens"])] += 1
        canvas_rows.extend({"method": method, "canvas_tokens": canvas, "span_count": count} for canvas, count in sorted(counts.items()))
    primary = next(row for row in effects if row["method_a"] == METHODS[4])
    verdict = "positive" if primary["task_macro_ci_low"] > 0 else "weak" if primary["task_macro_delta"] > 0 else "negative"
    summary.update({"mechanism_name": "Phase6 score-only candidate selection / M1 historical precursor", "not_true_m1_full": True, "candidate_bank_integrity": {"span_count": 5079, "candidate_rows": 40632, "candidate_coverage": "8/8 for every span"}, "comparisons_with_sign_test": effects, "fixed_selector_rule": "no score/threshold/fusion changed after outcomes", "conclusion": verdict, "oracle_ceiling_role": "offline analysis upper bound only"})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for filename, rows, fields in (("selections.csv", selection_table, list(selection_table[0])), ("selected_canvas.csv", canvas_rows, ["method", "canvas_tokens", "span_count"])):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    with (output_dir / "figure_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "task_macro_accuracy", "ci_low", "ci_high", "span_micro_accuracy", "standalone_forwards"], lineterminator="\n"); writer.writeheader()
        for item in summary["method_summary"]:
            writer.writerow({"method": item["method"], "task_macro_accuracy": item["equal_weight_task_macro_accuracy"], "ci_low": item["task_macro_ci_low"], "ci_high": item["task_macro_ci_high"], "span_micro_accuracy": item["span_micro_accuracy_descriptive"], "standalone_forwards": item["mean_standalone_forward_count"]})
    (output_dir / "report.zh.md").write_text(f"# Phase6 score-only candidate selection / M1 historical precursor\n\nConclusion: `{verdict}`. This is not M1 full; it performs no second-stage remasking.\n\nPrimary estimand is equal-weight base-task macro accuracy; span-micro is descriptive. `best_of_8_oracle_ceiling_offline_only` is an analysis ceiling, never deployable.\n", encoding="utf-8")
    return summary


def run_from_chunks(chunk_paths: Sequence[Path], output_dir: Path, *, bootstrap_replicates: int) -> dict[str, Any]:
    payloads = [row for path in chunk_paths for row in read_jsonl(path)]
    return summarize_payloads(payloads, output_dir, bootstrap_replicates=bootstrap_replicates)


def run_analysis(raw_path: Path, output_dir: Path, *, bootstrap_replicates: int = 10_000) -> dict[str, Any]:
    raw = load_grid_rows(raw_path)
    print(json.dumps({"phase": "load", "candidate_rows": len(raw)}), flush=True)
    by_span: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw:
        by_span[str(row["row_key"])].append(row)
    if len(by_span) != 5079 or len(raw) != 40632:
        raise RuntimeError(f"expected completed 5079x8 bank, got spans={len(by_span)} rows={len(raw)}")
    selections: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    selection_table: list[dict[str, Any]] = []
    for row_key, candidates in sorted(by_span.items()):
        chosen = choose_rows(candidates)
        record: dict[str, Any] = {"row_key": row_key, "task_group": chosen[METHODS[0]]["task_group"], "length_bucket": chosen[METHODS[0]]["length_bucket"]}
        for method, candidate in chosen.items():
            selections[method].append(selection_record(method, candidate))
            record[f"{method}_candidate_key"] = candidate["candidate_key"]
            record[f"{method}_canvas"] = candidate["canvas_tokens"]
            record[f"{method}_passed"] = bool(candidate.get("passed"))
        selection_table.append(record)
        if len(selection_table) % 1000 == 0:
            print(json.dumps({"phase": "select", "spans": len(selection_table)}), flush=True)
    comparisons = [(method, METHODS[0]) for method in METHODS[1:]]
    summary = write_grouped_effect_report(
        output_dir=output_dir,
        title="Phase6 score-only candidate selection / M1 historical precursor",
        rows_by_method=selections,
        comparisons=comparisons,
        bootstrap_replicates=bootstrap_replicates,
    )
    print(json.dumps({"phase": "grouped_effects_written"}), flush=True)
    effects = []
    for index, (method, baseline) in enumerate(comparisons):
        effect, _ = paired_effect(
            selections[method], selections[baseline], method_a=method, method_b=baseline,
            bootstrap_replicates=bootstrap_replicates, seed=20260715 + index,
        )
        effect["exact_task_sign_test_p"] = exact_sign_p(int(effect["wins"]), int(effect["losses"]))
        effect["oracle_regret_span_count"] = sum(
            int(oracle["passed"]) - int(candidate["passed"])
            for oracle, candidate in zip(selections[METHODS[5]], selections[method])
        )
        effects.append(effect)
    canvas_rows = []
    for method, rows in selections.items():
        counts: dict[int, int] = defaultdict(int)
        for row in rows:
            counts[int(row["selected_canvas_tokens"])] += 1
        canvas_rows.extend({"method": method, "canvas_tokens": canvas, "span_count": count} for canvas, count in sorted(counts.items()))
    primary = next(row for row in effects if row["method_a"] == METHODS[4])
    verdict = "positive" if primary["task_macro_ci_low"] > 0 else "weak" if primary["task_macro_delta"] > 0 else "negative"
    summary.update(
        {
            "mechanism_name": "Phase6 score-only candidate selection / M1 historical precursor",
            "not_true_m1_full": True,
            "candidate_bank_integrity": {"span_count": len(by_span), "candidate_rows": len(raw), "candidate_coverage": "8/8 for every span"},
            "comparisons_with_sign_test": effects,
            "fixed_selector_rule": "no score/threshold/fusion changed after outcomes",
            "conclusion": verdict,
            "oracle_ceiling_role": "offline analysis upper bound only",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "selections.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(selection_table[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(selection_table)
    with (output_dir / "selected_canvas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "canvas_tokens", "span_count"], lineterminator="\n")
        writer.writeheader(); writer.writerows(canvas_rows)
    with (output_dir / "figure_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "task_macro_accuracy", "ci_low", "ci_high", "span_micro_accuracy", "standalone_forwards"], lineterminator="\n")
        writer.writeheader()
        for item in summary["method_summary"]:
            writer.writerow({"method": item["method"], "task_macro_accuracy": item["equal_weight_task_macro_accuracy"], "ci_low": item["task_macro_ci_low"], "ci_high": item["task_macro_ci_high"], "span_micro_accuracy": item["span_micro_accuracy_descriptive"], "standalone_forwards": item["mean_standalone_forward_count"]})
    report = ["# Phase6 score-only candidate selection / M1 historical precursor", "", f"Conclusion: `{verdict}`. This is not M1 full; it performs no second-stage remasking.", "", "Primary estimand is equal-weight base-task macro accuracy; span-micro is descriptive. `best_of_8_oracle_ceiling_offline_only` is an analysis ceiling, never deployable."]
    (output_dir / "report.zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw")
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--chunk-output")
    parser.add_argument("--chunk-start", type=int)
    parser.add_argument("--chunk-stop", type=int)
    parser.add_argument("--selection-chunks", nargs="+")
    args = parser.parse_args()
    if args.chunk_output:
        if not args.raw or args.chunk_start is None or args.chunk_stop is None:
            parser.error("chunk mode requires --raw --chunk-start --chunk-stop")
        write_selection_chunk(Path(args.raw).resolve(), Path(args.chunk_output).resolve(), args.chunk_start, args.chunk_stop)
        return 0
    if args.selection_chunks:
        if not args.output_dir:
            parser.error("aggregation mode requires --output-dir")
        run_from_chunks([Path(path).resolve() for path in args.selection_chunks], Path(args.output_dir).resolve(), bootstrap_replicates=args.bootstrap_replicates)
        return 0
    if not args.raw or not args.output_dir:
        parser.error("full mode requires --raw and --output-dir")
    run_analysis(Path(args.raw).resolve(), Path(args.output_dir).resolve(), bootstrap_replicates=args.bootstrap_replicates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
