#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


VARIANTS = ("C1", "C2", "C4", "L1", "L2", "L4")
COMPARISONS = (
    ("C2-C1", "C2", "C1"),
    ("C4-C1", "C4", "C1"),
    ("L2-L1", "L2", "L1"),
    ("L4-L1", "L4", "L1"),
    ("L1-C1", "L1", "C1"),
    ("L2-C2", "L2", "C2"),
    ("L4-C4", "L4", "C4"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    index = (len(values) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def stage(active_masks: int) -> str:
    if active_masks > 42:
        return "early"
    if active_masks > 21:
        return "middle"
    return "late"


def group_values(rows: list[dict[str, Any]], variant: str) -> dict[str, list[int]]:
    values: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row["variant"] == variant:
            values[str(row["task_group"])].append(int(bool(row["passed"])))
    return values


def grouped_ci_and_test(
    rows: list[dict[str, Any]], left: str, right: str, *, seed: int = 20260823
) -> dict[str, float]:
    a, b = group_values(rows, left), group_values(rows, right)
    groups = sorted(set(a) & set(b))
    deltas = [mean(a[group]) - mean(b[group]) for group in groups]
    observed = mean(deltas)
    rng = random.Random(seed + sum(map(ord, left + right)))
    bootstrap = []
    sign_flips = []
    for _ in range(10_000):
        sample = [deltas[rng.randrange(len(deltas))] for _ in groups]
        bootstrap.append(mean(sample))
        sign_flips.append(mean(delta if rng.randrange(2) else -delta for delta in deltas))
    bootstrap.sort()
    p_two_sided = sum(abs(value) >= abs(observed) for value in sign_flips) / len(sign_flips)
    return {
        "group_count": len(groups),
        "group_macro_delta_pp": observed * 100,
        "ci95_low_pp": percentile(bootstrap, 0.025) * 100,
        "ci95_high_pp": percentile(bootstrap, 0.975) * 100,
        "paired_sign_flip_p": p_two_sided,
    }


def failure_category(reason: str) -> str:
    lower = reason.lower()
    if "syntax" in lower or "indent" in lower or "invalid" in lower:
        return "syntax_or_compile"
    if "timeout" in lower:
        return "timeout"
    return "functional_or_other"


def analyze(output: Path) -> None:
    cases = read_jsonl(output / "per_case_results.jsonl")
    traces = read_jsonl(output / "per_step_trace.jsonl")
    markov = read_jsonl(output / "markov_transitions.jsonl")
    expected_keys = {(row["task_id"], row["variant"]) for row in cases}
    duplicate_count = len(cases) - len(expected_keys)
    counts = Counter(row["variant"] for row in cases)
    task_sets = {variant: {row["task_id"] for row in cases if row["variant"] == variant} for variant in VARIANTS}
    common_tasks = set.intersection(*task_sets.values())
    completeness = {
        "status": "passed" if all(counts[variant] == 1033 for variant in VARIANTS) and not duplicate_count and len(common_tasks) == 1033 else "failed",
        "case_rows": len(cases),
        "trace_rows": len(traces),
        "markov_transition_rows": len(markov),
        "variant_counts": dict(counts),
        "duplicate_case_variant_keys": duplicate_count,
        "common_task_count": len(common_tasks),
        "task_group_count": len({row["task_group"] for row in cases}),
        "failure_journal_present": (output / "failure_journal.jsonl").exists(),
        "note": "The v2 output is the only valid experiment population; 20260822 v1 is excluded due to audited delete-semantics drift.",
    }
    write_json(output / "completeness_audit.json", completeness)
    write_json(output / "manifest.snapshot.json", {"population": "HumanEval-Infilling SingleLine development/full-allowed", "rows": 1033, "task_groups": completeness["task_group_count"], "task_ids_common_to_all_variants": len(common_tasks)})

    quality_rows = []
    for variant in VARIANTS:
        subset = [row for row in cases if row["variant"] == variant]
        passed = sum(bool(row["passed"]) for row in subset)
        quality_rows.append({"record_type": "variant", "comparison": variant, "variant": variant, "n": len(subset), "pass_count": passed, "pass_rate_percent": passed / len(subset) * 100, "task_groups": len({row["task_group"] for row in subset})})
    by_task_variant = {(row["task_id"], row["variant"]): row for row in cases}
    for label, left, right in COMPARISONS:
        help_count = harm_count = tie_pass = tie_fail = 0
        for task_id in common_tasks:
            a, b = bool(by_task_variant[task_id, left]["passed"]), bool(by_task_variant[task_id, right]["passed"])
            if a and not b:
                help_count += 1
            elif b and not a:
                harm_count += 1
            elif a:
                tie_pass += 1
            else:
                tie_fail += 1
        quality_rows.append({"record_type": "comparison", "comparison": label, "left": left, "right": right, "help": help_count, "harm": harm_count, "tie_pass": tie_pass, "tie_fail": tie_fail, **grouped_ci_and_test(cases, left, right)})
    write_csv(output / "quality_summary.csv", quality_rows)

    traces_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        traces_by_variant[str(trace["variant"])].append(trace)
    efficiency_rows = []
    for variant in VARIANTS:
        subset, step_rows = [row for row in cases if row["variant"] == variant], traces_by_variant[variant]
        wall = [float(row["wall_sec"]) for row in subset]
        forwards = [float(row["forward_count"]) for row in subset]
        selected = [float(row["selected_candidate_count"]) for row in step_rows]
        normal = [float(row["actual_committed_normal_tokens"]) for row in step_rows]
        resolved = [float(row["actual_committed_normal_tokens"] + row["executed_expand_count"] + row["executed_delete_count"]) for row in step_rows]
        mask_reduction = [len(row["active_mask_positions"]) - int(row["unresolved_after"]) for row in step_rows]
        requested = int(variant[1])
        barrier = sum(int(row["barrier_truncated_proposals"]) > 0 for row in step_rows)
        short_mask = sum(int(row["selected_candidate_count"]) < requested for row in step_rows)
        effective = Counter(int(row["actual_committed_normal_tokens"]) for row in step_rows)
        efficiency_rows.append({
            "variant": variant, "cases": len(subset), "steps": len(step_rows), "forward_per_case_mean": mean(forwards), "forward_per_case_median": median(forwards), "wall_sec_mean": mean(wall), "wall_sec_median": median(wall), "wall_sec_p90": percentile(wall, .9), "wall_sec_p95": percentile(wall, .95), "selected_per_forward_mean": mean(selected), "committed_normal_per_forward_mean": mean(normal), "resolved_decisions_per_forward_mean": mean(resolved), "net_mask_reduction_per_forward_mean": mean(mask_reduction), "structural_barrier_step_fraction": barrier / len(step_rows), "insufficient_mask_step_fraction": short_mask / len(step_rows), **{f"committed_{value}_fraction": effective[value] / len(step_rows) for value in range(5)},
        })
    write_csv(output / "efficiency_summary.csv", efficiency_rows)

    passed_by_case = {row["case_key"]: bool(row["passed"]) for row in cases}
    topk_rows = []
    for variant in ("C1", "C2", "C4"):
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for trace in traces_by_variant[variant]:
            context = "structural" if trace["executed_expand_count"] or trace["executed_delete_count"] else "normal"
            buckets[(stage(len(trace["active_mask_positions"])), "pass" if passed_by_case[trace["case_key"]] else "fail", context)].append(trace)
        for (stage_name, verdict, context), values in buckets.items():
            for k in (2, 4):
                eligible = [value for value in values if len(value["confidence_top4"]) >= k]
                if not eligible:
                    continue
                equal, adjacent, run, span, prefix = [], [], [], [], []
                for value in eligible:
                    positions = [entry["position"] for entry in value["confidence_top4"][:k]]
                    ordered = sorted(positions)
                    left = value["leftmost_contiguous_positions"][:k]
                    equal.append(set(positions) == set(left) and len(left) == k)
                    adjacent.append(sum(b - a == 1 for a, b in zip(ordered, ordered[1:])))
                    longest = current = 1
                    for a, b in zip(ordered, ordered[1:]):
                        current = current + 1 if b == a + 1 else 1
                        longest = max(longest, current)
                    run.append(longest)
                    span.append(ordered[-1] - ordered[0] + 1)
                    prefix.append(len(set(positions) & set(left)))
                topk_rows.append({"variant": variant, "k": k, "stage": stage_name, "final_verdict": verdict, "step_context": context, "steps": len(eligible), "topk_equals_leftmostk_fraction": mean(equal), "adjacent_pair_count_mean": mean(adjacent), "longest_contiguous_run_mean": mean(run), "span_width_mean": mean(span), "left_prefix_coverage_mean": mean(prefix)})
    write_csv(output / "topk_spatial_summary.csv", topk_rows)

    trace_lookup = {(row["case_key"], int(row["step_index"])): row for row in traces}
    markov_rows = []
    stage_rows: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in markov:
        source = trace_lookup.get((row["case_key"], int(row["source_step_index"])))
        source_stage = stage(len(source["active_mask_positions"])) if source else "unknown"
        verdict = "pass" if passed_by_case[row["case_key"]] else "fail"
        stage_rows[(int(row["offset"]), source_stage, verdict)].append(row)
    for offset in (1, 2, 3):
        values = [row for row in markov if int(row["offset"]) == offset]
        if values:
            markov_rows.append({"offset": offset, "transitions": len(values), "cases": len({row["case_key"] for row in values}), "task_groups": len({row["task_group"] for row in values}), "top1_agreement_fraction": mean(bool(row["top1_agreement"]) for row in values), "tv_mean": mean(float(row["total_variation"]) for row in values), "tv_median": median(float(row["total_variation"]) for row in values), "entropy_delta_fresh_minus_stale_mean": mean(float(row["fresh_entropy"]) - float(row["stale_entropy"]) for row in values), "margin_delta_fresh_minus_stale_mean": mean(float(row["fresh_margin"]) - float(row["stale_margin"]) for row in values), "fresh_top1_stale_rank_mean": mean(float(row["fresh_top1_stale_rank"]) for row in values), "fresh_top1_stale_probability_mean": mean(float(row["fresh_top1_stale_probability"]) for row in values)})
    write_csv(output / "markov_premise_by_offset.csv", markov_rows)
    markov_stage_rows = []
    for (offset, stage_name, verdict), values in sorted(stage_rows.items()):
        markov_stage_rows.append({"offset": offset, "stage": stage_name, "final_verdict": verdict, "transitions": len(values), "top1_agreement_fraction": mean(bool(row["top1_agreement"]) for row in values), "tv_mean": mean(float(row["total_variation"]) for row in values), "entropy_delta_fresh_minus_stale_mean": mean(float(row["fresh_entropy"]) - float(row["stale_entropy"]) for row in values)})
    write_csv(output / "markov_premise_by_stage.csv", markov_stage_rows)
    write_json(output / "markov_premise_summary.json", {"scope": "L1 only; online full-vocabulary stale-to-fresh distribution comparisons", "by_offset": markov_rows, "oracle_direction_available": False, "interpretation_constraint": "Distribution movement alone does not demonstrate correctness-direction improvement; this run cannot authorize Markov-head training."})

    failures = []
    for row in cases:
        if not row["passed"]:
            failures.append({"case_key": row["case_key"], "task_id": row["task_id"], "task_group": row["task_group"], "variant": row["variant"], "failure_reason": row["failure_reason"], "failure_category": failure_category(str(row["failure_reason"])), "forward_count": row["forward_count"], "final_remaining_masks": row["final_remaining_masks"], "expand_count": row["expand_count"], "delete_count": row["delete_count"]})
    with (output / "failure_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {"status": "completed_valid_v2", "completeness": completeness, "quality": quality_rows, "efficiency": efficiency_rows, "markov": markov_rows, "limitations": ["No reference-token oracle diagnostics were logged; correctness direction of stale-to-fresh changes is unidentified.", "No trace field encodes suffix boundary, so suffix-adjacency top-K statistics are unavailable.", "Evaluator output did not retain a compile boolean; failure categories are conservative string-based labels."]}
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text("# DreamOn SingleLine order × parallelism × Markov diagnostic\n\nValid run: 2026-08-23 v2. The 2026-08-22 v1 output is retained separately as an invalidated delete-semantics audit and is not used here. Run `python analysis/analyze_dreamon_order_parallel.py --output-dir <this-dir>` to regenerate compact summaries.\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    analyze(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
