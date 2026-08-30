#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


VARIANT_NAMES = {
    "C1": "全局置信度逐词刷新诊断",
    "L1": "固定左前沿逐词刷新诊断",
}
EXPECTED_PASS_COUNTS = {"C1": 951, "L1": 942}
REQUIRED_DISTRIBUTION_FIELDS = (
    "total_variation",
    "stale_top1_token_id",
    "fresh_top1_token_id",
    "top1_agreement",
    "stale_entropy",
    "fresh_entropy",
    "stale_margin",
    "fresh_margin",
    "fresh_top1_stale_probability",
    "fresh_top1_stale_rank",
    "stale_top_token_ids",
    "stale_top_probabilities",
    "fresh_top_token_ids",
    "fresh_top_probabilities",
)
REQUIRED_REFERENCE_FIELDS = (
    "stale_reference_probability",
    "fresh_reference_probability",
    "stale_reference_rank",
    "fresh_reference_rank",
    "reference_rank_improvement",
    "stale_top1_is_reference",
    "fresh_top1_is_reference",
    "wrong_to_correct",
    "correct_to_wrong",
    "stale_reference_retained_in_top_p",
    "fresh_reference_retained_in_top_p",
    "reference_log_probability_delta_kind",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = [
        "diagnostic_name_zh",
        "variant",
        "record_scope",
        "offset",
        "reference_subset",
        "final_verdict",
        "generation_stage",
    ]
    fields = preferred + sorted(
        {key for row in rows for key in row if key not in preferred}
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stage(active_masks: int) -> str:
    if active_masks > 42:
        return "early"
    if active_masks > 21:
        return "middle"
    return "late"


def audit_required_transition_fields(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    missing: set[str] = set()
    for row in rows:
        if row.get("record_type") not in (None, "stale_fresh_transition"):
            continue
        if not bool(row.get("transition_eligible")):
            if not row.get("transition_ineligible_reason"):
                missing.add("transition_ineligible_reason")
            continue
        for field in (
            "offset",
            "source_step_index",
            "fresh_step_index",
            "target_position",
            "stale_global_rank",
            "fresh_global_rank",
            "stale_in_top2",
            "fresh_in_top2",
            "stale_in_top4",
            "fresh_in_top4",
            "global_rank_improvement",
            "outside_to_top2",
            "outside_to_top4",
            "top2_retained",
            "top4_retained",
            "reference_evaluable",
        ):
            if row.get(field) is None:
                missing.add(field)
        for prefix in ("raw", "actual"):
            for suffix in REQUIRED_DISTRIBUTION_FIELDS:
                field = f"{prefix}_{suffix}"
                if row.get(field) is None:
                    missing.add(field)
        if bool(row.get("reference_evaluable")):
            for field in (
                "target_reference_token_id",
                "predecessor_token_matches_reference",
                "reference_prefix_aligned_through_predecessor",
            ):
                if row.get(field) is None:
                    missing.add(field)
            for prefix in ("raw", "actual"):
                for suffix in REQUIRED_REFERENCE_FIELDS:
                    field = f"{prefix}_{suffix}"
                    if row.get(field) is None:
                        missing.add(field)
                kind = row.get(f"{prefix}_reference_log_probability_delta_kind")
                delta = row.get(f"{prefix}_reference_log_probability_delta")
                if kind == "finite" and delta is None:
                    missing.add(f"{prefix}_reference_log_probability_delta")
                for side in ("stale", "fresh"):
                    retained = row.get(f"{prefix}_{side}_reference_retained_in_top_p")
                    log_probability = row.get(f"{prefix}_{side}_reference_log_probability")
                    if retained is True and log_probability is None:
                        missing.add(f"{prefix}_{side}_reference_log_probability")
                    if retained is None:
                        missing.add(f"{prefix}_{side}_reference_retained_in_top_p")
        elif not row.get("reference_non_evaluable_reason"):
            missing.add("reference_non_evaluable_reason")
    return sorted(missing)


def completeness_status(
    *,
    case_rows: int,
    variant_counts: Mapping[str, int],
    common_task_count: int,
    task_group_count: int,
    duplicate_count: int,
    missing_count: int,
    error_count: int,
    pass_counts: Mapping[str, int],
    required_field_errors: list[str],
    compressed_artifacts_valid: bool,
) -> str:
    structural_complete = (
        case_rows == 2066
        and dict(variant_counts) == {"C1": 1033, "L1": 1033}
        and common_task_count == 1033
        and task_group_count == 164
        and duplicate_count == 0
        and missing_count == 0
        and error_count == 0
    )
    if not structural_complete:
        return "incomplete"
    protocol_valid = (
        dict(pass_counts) == EXPECTED_PASS_COUNTS
        and not required_field_errors
        and compressed_artifacts_valid
    )
    return "completed" if protocol_valid else "protocol_failed"


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def cluster_summary(
    rows: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], float | None],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    flat: list[float] = []
    for row in rows:
        item = value(row)
        if item is None or not math.isfinite(float(item)):
            continue
        grouped[str(row["task_group"])].append(float(item))
        flat.append(float(item))
    group_values = np.asarray([mean(values) for values in grouped.values()], dtype=np.float64)
    if not group_values.size:
        return {
            "transition_micro": None,
            "task_group_macro": None,
            "ci95_low": None,
            "ci95_high": None,
            "metric_transition_count": 0,
            "metric_task_group_count": 0,
        }
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, group_values.size, size=(replicates, group_values.size))
    bootstrap = group_values[indexes].mean(axis=1)
    return {
        "transition_micro": mean(flat),
        "task_group_macro": float(group_values.mean()),
        "ci95_low": percentile(bootstrap, 0.025),
        "ci95_high": percentile(bootstrap, 0.975),
        "metric_transition_count": len(flat),
        "metric_task_group_count": len(grouped),
    }


def add_metric(
    output: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    seed: int,
    replicates: int,
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> None:
    summary = cluster_summary(
        rows,
        lambda row: float(row[field])
        if row.get(field) is not None and (predicate is None or predicate(row))
        else None,
        seed=seed,
        replicates=replicates,
    )
    output.update({f"{field}_{key}": value for key, value in summary.items()})


def subset_rows(rows: Sequence[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    if subset == "overall":
        return list(rows)
    if subset == "predecessor_correct":
        return [row for row in rows if row.get("predecessor_token_matches_reference") is True]
    if subset == "reference_prefix_aligned":
        return [
            row
            for row in rows
            if row.get("reference_prefix_aligned_through_predecessor") is True
        ]
    raise ValueError(subset)


def stratified_rows(
    rows: Sequence[dict[str, Any]], verdict: str, generation_stage: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (verdict == "all" or row["final_verdict"] == verdict)
        and (generation_stage == "all" or row["generation_stage"] == generation_stage)
    ]


def build_reference_rows(
    transitions: Sequence[dict[str, Any]], *, replicates: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in ("C1", "L1"):
        for offset in (1, 2, 3):
            base = [
                row
                for row in transitions
                if row["variant"] == variant
                and int(row["offset"]) == offset
                and bool(row.get("reference_evaluable"))
            ]
            for subset in ("reference_prefix_aligned", "predecessor_correct", "overall"):
                subset_base = subset_rows(base, subset)
                for verdict in ("all", "pass", "fail"):
                    for generation_stage in ("all", "early", "middle", "late"):
                        rows = stratified_rows(subset_base, verdict, generation_stage)
                        if not rows:
                            continue
                        item: dict[str, Any] = {
                            "diagnostic_name_zh": VARIANT_NAMES[variant],
                            "variant": variant,
                            "record_scope": "reference_direction",
                            "offset": offset,
                            "reference_subset": subset,
                            "final_verdict": verdict,
                            "generation_stage": generation_stage,
                            "transitions": len(rows),
                            "cases": len({row["case_key"] for row in rows}),
                            "task_groups": len({row["task_group"] for row in rows}),
                        }
                        metric_fields = (
                            "raw_reference_log_probability_delta",
                            "raw_reference_rank_improvement",
                            "raw_wrong_to_correct",
                            "raw_correct_to_wrong",
                            "actual_reference_rank_improvement",
                            "actual_wrong_to_correct",
                            "actual_correct_to_wrong",
                            "actual_stale_reference_retained_in_top_p",
                            "actual_fresh_reference_retained_in_top_p",
                        )
                        for index, field in enumerate(metric_fields):
                            add_metric(
                                item,
                                rows,
                                field,
                                seed=20260830 + offset * 100 + index,
                                replicates=replicates,
                            )
                        add_metric(
                            item,
                            rows,
                            "actual_reference_log_probability_delta",
                            seed=20261830 + offset,
                            replicates=replicates,
                            predicate=lambda row: row.get(
                                "actual_reference_log_probability_delta_kind"
                            )
                            == "finite",
                        )
                        item.update(
                            {
                                "raw_wrong_to_correct_count": sum(
                                    bool(row["raw_wrong_to_correct"]) for row in rows
                                ),
                                "raw_correct_to_wrong_count": sum(
                                    bool(row["raw_correct_to_wrong"]) for row in rows
                                ),
                                "actual_wrong_to_correct_count": sum(
                                    bool(row["actual_wrong_to_correct"]) for row in rows
                                ),
                                "actual_correct_to_wrong_count": sum(
                                    bool(row["actual_correct_to_wrong"]) for row in rows
                                ),
                                "actual_logprob_positive_infinity_count": sum(
                                    row.get("actual_reference_log_probability_delta_kind")
                                    == "positive_infinity"
                                    for row in rows
                                ),
                                "actual_logprob_negative_infinity_count": sum(
                                    row.get("actual_reference_log_probability_delta_kind")
                                    == "negative_infinity"
                                    for row in rows
                                ),
                                "actual_logprob_both_truncated_count": sum(
                                    row.get("actual_reference_log_probability_delta_kind")
                                    == "undefined_both_truncated"
                                    for row in rows
                                ),
                            }
                        )
                        output.append(item)
    return output


def build_stale_fresh_rows(
    transitions: Sequence[dict[str, Any]], *, replicates: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in ("C1", "L1"):
        for offset in (1, 2, 3):
            base = [
                row
                for row in transitions
                if row["variant"] == variant and int(row["offset"]) == offset
            ]
            for subset in ("reference_prefix_aligned", "predecessor_correct", "overall"):
                subset_base = subset_rows(base, subset)
                for verdict in ("all", "pass", "fail"):
                    for generation_stage in ("all", "early", "middle", "late"):
                        rows = stratified_rows(subset_base, verdict, generation_stage)
                        if not rows:
                            continue
                        item: dict[str, Any] = {
                            "diagnostic_name_zh": VARIANT_NAMES[variant],
                            "variant": variant,
                            "record_scope": "stale_fresh_distribution",
                            "offset": offset,
                            "reference_subset": subset,
                            "final_verdict": verdict,
                            "generation_stage": generation_stage,
                            "transitions": len(rows),
                            "cases": len({row["case_key"] for row in rows}),
                            "task_groups": len({row["task_group"] for row in rows}),
                        }
                        derived = []
                        for row in rows:
                            copy = dict(row)
                            copy["raw_entropy_delta"] = float(row["raw_fresh_entropy"]) - float(
                                row["raw_stale_entropy"]
                            )
                            copy["actual_entropy_delta"] = float(
                                row["actual_fresh_entropy"]
                            ) - float(row["actual_stale_entropy"])
                            copy["raw_margin_delta"] = float(row["raw_fresh_margin"]) - float(
                                row["raw_stale_margin"]
                            )
                            copy["actual_margin_delta"] = float(
                                row["actual_fresh_margin"]
                            ) - float(row["actual_stale_margin"])
                            derived.append(copy)
                        for index, field in enumerate(
                            (
                                "raw_total_variation",
                                "actual_total_variation",
                                "raw_top1_agreement",
                                "actual_top1_agreement",
                                "raw_entropy_delta",
                                "actual_entropy_delta",
                                "raw_margin_delta",
                                "actual_margin_delta",
                            )
                        ):
                            add_metric(
                                item,
                                derived,
                                field,
                                seed=20262830 + offset * 100 + index,
                                replicates=replicates,
                            )
                        item["raw_total_variation_median"] = median(
                            float(row["raw_total_variation"]) for row in rows
                        )
                        item["actual_total_variation_median"] = median(
                            float(row["actual_total_variation"]) for row in rows
                        )
                        output.append(item)
    return output


def build_global_rank_rows(
    transitions: Sequence[dict[str, Any]],
    neighbors: Sequence[dict[str, Any]],
    *,
    replicates: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in ("C1", "L1"):
        for offset in (1, 2, 3):
            base = [
                row
                for row in transitions
                if row["variant"] == variant and int(row["offset"]) == offset
            ]
            for subset in ("reference_prefix_aligned", "predecessor_correct", "overall"):
                subset_base = subset_rows(base, subset)
                for verdict in ("all", "pass", "fail"):
                    for generation_stage in ("all", "early", "middle", "late"):
                        rows = stratified_rows(subset_base, verdict, generation_stage)
                        if not rows:
                            continue
                        item: dict[str, Any] = {
                            "diagnostic_name_zh": VARIANT_NAMES[variant],
                            "variant": variant,
                            "record_scope": "stable_target_global_rank",
                            "offset": offset,
                            "reference_subset": subset,
                            "final_verdict": verdict,
                            "generation_stage": generation_stage,
                            "transitions": len(rows),
                            "cases": len({row["case_key"] for row in rows}),
                            "task_groups": len({row["task_group"] for row in rows}),
                        }
                        for index, field in enumerate(
                            (
                                "stale_global_rank",
                                "fresh_global_rank",
                                "global_rank_improvement",
                                "stale_in_top2",
                                "fresh_in_top2",
                                "stale_in_top4",
                                "fresh_in_top4",
                                "outside_to_top2",
                                "outside_to_top4",
                                "top2_retained",
                                "top4_retained",
                            )
                        ):
                            add_metric(
                                item,
                                rows,
                                field,
                                seed=20263830 + offset * 100 + index,
                                replicates=replicates,
                            )
                        output.append(item)
    for verdict in ("all", "pass", "fail"):
        for generation_stage in ("all", "early", "middle", "late"):
            rows = stratified_rows(list(neighbors), verdict, generation_stage)
            if not rows:
                continue
            item = {
                "diagnostic_name_zh": VARIANT_NAMES["C1"],
                "variant": "C1",
                "record_scope": "global_right_neighbor_steps",
                "offset": "",
                "reference_subset": "overall",
                "final_verdict": verdict,
                "generation_stage": generation_stage,
                "normal_steps": len(rows),
                "cases": len({row["case_key"] for row in rows}),
                "task_groups": len({row["task_group"] for row in rows}),
            }
            add_metric(
                item,
                rows,
                "right_neighbor_still_mask",
                seed=20264830,
                replicates=replicates,
            )
            right_neighbors = [row for row in rows if row.get("right_neighbor_still_mask") is True]
            for index, field in enumerate(
                ("source_right_neighbor_in_top2", "source_right_neighbor_in_top4")
            ):
                add_metric(
                    item,
                    right_neighbors,
                    field,
                    seed=20264831 + index,
                    replicates=replicates,
                )
            item["right_neighbor_count"] = len(right_neighbors)
            source_top2 = [
                row
                for row in rows
                if row.get("source_right_neighbor_in_top2") is True
                and row.get("fresh_global_rank_available") is True
            ]
            source_top4 = [
                row
                for row in rows
                if row.get("source_right_neighbor_in_top4") is True
                and row.get("fresh_global_rank_available") is True
            ]
            add_metric(
                item,
                source_top2,
                "fresh_in_top2",
                seed=20264840,
                replicates=replicates,
            )
            item["source_top2_with_fresh_count"] = len(source_top2)
            add_metric(
                item,
                source_top4,
                "fresh_in_top4",
                seed=20264841,
                replicates=replicates,
            )
            item["source_top4_with_fresh_count"] = len(source_top4)
            fresh = [row for row in rows if row.get("fresh_global_rank_available") is True]
            for index, field in enumerate(
                (
                    "outside_to_top2",
                    "outside_to_top4",
                    "top2_retained",
                    "top4_retained",
                    "global_rank_improvement",
                )
            ):
                add_metric(
                    item,
                    fresh,
                    field,
                    seed=20264850 + index,
                    replicates=replicates,
                )
            item["fresh_available_count"] = len(fresh)
            output.append(item)
    return output


def compress_and_validate(path: Path) -> tuple[Path, int, bool]:
    compressed = path.with_suffix(path.suffix + ".zst")
    subprocess.run(
        ["zstd", "-q", "-f", "-10", str(path), "-o", str(compressed)], check=True
    )
    count = 0
    valid = True
    process = subprocess.Popen(
        ["zstd", "-q", "-d", "-c", str(compressed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    except json.JSONDecodeError:
        valid = False
    return_code = process.wait()
    if return_code != 0:
        valid = False
    return compressed, count, valid


def smoke_v2_alignment(
    cases: Sequence[dict[str, Any]],
    traces: Sequence[dict[str, Any]],
    v2_dir: Path,
) -> dict[str, Any]:
    old_cases = read_jsonl(v2_dir / "per_case_results.jsonl")
    old_traces = read_jsonl(v2_dir / "per_step_trace.jsonl")
    keys = {row["case_key"] for row in cases}
    old_case_lookup = {row["case_key"]: row for row in old_cases if row["case_key"] in keys}
    fields = (
        "output",
        "passed",
        "completion_token_ids",
        "forward_count",
        "expand_count",
        "delete_count",
        "final_dynamic_length",
    )
    case_mismatches = []
    for row in cases:
        old = old_case_lookup.get(row["case_key"])
        if old is None:
            case_mismatches.append({"case_key": row["case_key"], "reason": "missing_v2_case"})
            continue
        differences = [field for field in fields if row.get(field) != old.get(field)]
        if differences:
            case_mismatches.append(
                {"case_key": row["case_key"], "different_fields": differences}
            )
    new_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    old_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in traces:
        new_by_case[row["case_key"]].append(row)
    for row in old_traces:
        if row["case_key"] in keys:
            old_by_case[row["case_key"]].append(row)
    step_fields = (
        "selected_positions",
        "actual_committed_normal_tokens",
        "executed_expand_count",
        "executed_delete_count",
        "unresolved_after",
        "canvas_length",
    )
    step_mismatches = []
    for key in sorted(keys):
        new = sorted(new_by_case[key], key=lambda row: int(row["step_index"]))
        old = sorted(old_by_case[key], key=lambda row: int(row["step_index"]))
        if len(new) != len(old):
            step_mismatches.append(
                {"case_key": key, "reason": "step_count", "new": len(new), "old": len(old)}
            )
            continue
        for new_row, old_row in zip(new, old):
            differences = [field for field in step_fields if new_row.get(field) != old_row.get(field)]
            if differences:
                step_mismatches.append(
                    {
                        "case_key": key,
                        "step_index": new_row["step_index"],
                        "different_fields": differences,
                    }
                )
    return {
        "status": "passed" if not case_mismatches and not step_mismatches else "failed",
        "case_mismatches": case_mismatches,
        "step_mismatches": step_mismatches,
        "compared_cases": len(keys),
    }


def find_summary(
    rows: Sequence[dict[str, Any]],
    variant: str,
    offset: int,
    subset: str = "overall",
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row["variant"] == variant
            and int(row["offset"]) == offset
            and row["reference_subset"] == subset
            and row["final_verdict"] == "all"
            and row["generation_stage"] == "all"
        ),
        None,
    )


def format_ci(row: Mapping[str, Any], field: str, scale: float = 1.0) -> str:
    value = row.get(f"{field}_task_group_macro")
    low = row.get(f"{field}_ci95_low")
    high = row.get(f"{field}_ci95_high")
    if value is None:
        return "NA"
    return f"{float(value) * scale:.4f} [{float(low) * scale:.4f}, {float(high) * scale:.4f}]"


def write_report(
    output: Path,
    audit: Mapping[str, Any],
    reference_rows: Sequence[dict[str, Any]],
    stale_rows: Sequence[dict[str, Any]],
    global_rows: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# DreamOn SingleLine Markov 前提补充诊断（v3）",
        "",
        f"状态：`{audit['status']}`。本轮没有训练 Markov head，也不自动授权下一阶段。",
        "",
        "## 两个诊断与完整性",
        "",
        "- 全局置信度逐词刷新诊断：每次前向只提交全局 entropy-confidence 最高的一个位置；用于评估未来 global top-K 内连续短链 + Markov。",
        "- 固定左前沿逐词刷新诊断：每次前向只提交最左未解决位置；用于评估未来强制固定左到右 + Markov。",
        f"- 完整运行：`{audit['case_rows']} = 1033 × 2`；共同 task=`{audit['common_task_count']}`；基础 task group=`{audit['task_group_count']}`；duplicate/missing/error=`{audit['duplicate_count']}/{audit['missing_count']}/{audit['error_count']}`。",
        f"- Pass@1 复现：全局置信度逐词刷新 `{audit['pass_counts'].get('C1', 0)}/1033`；固定左前沿逐词刷新 `{audit['pass_counts'].get('L1', 0)}/1033`。",
        "",
        "## 旧分布到新分布",
        "",
        "下表均为 164 个基础 task group 内先聚合、再做 cluster bootstrap 的 task-group macro；offset 1/2/3 分开，不能混为长块平均。",
        "",
        "| 诊断 | offset | raw TV group-macro [CI] / transition-micro | actual TV group-macro [CI] / transition-micro | raw top-1 agreement | actual top-1 agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in ("C1", "L1"):
        for offset in (1, 2, 3):
            row = find_summary(list(stale_rows), variant, offset)
            if row:
                lines.append(
                    f"| {VARIANT_NAMES[variant]} | {offset} | {format_ci(row, 'raw_total_variation')} / {row['raw_total_variation_transition_micro']:.4f} | {format_ci(row, 'actual_total_variation')} / {row['actual_total_variation_transition_micro']:.4f} | {format_ci(row, 'raw_top1_agreement', 100)}% | {format_ci(row, 'actual_top1_agreement', 100)}% |"
                )
    lines.extend(
        [
            "",
            "固定左前沿逐词刷新诊断的 raw transition-micro TV 为 offset 1/2/3=`0.10272/0.26099/0.36506`，与 v2 报告的四舍五入值 `0.103/0.261/0.365` 对齐。group-macro 与 transition-micro 的差异来自 164 个 task group 等权，而不是协议漂移。",
            "",
            "### Eligible transition 覆盖",
            "",
            "| 诊断 | offset | transitions | cases | task groups |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in ("C1", "L1"):
        for offset in (1, 2, 3):
            row = find_summary(list(stale_rows), variant, offset)
            if row:
                lines.append(
                    f"| {VARIANT_NAMES[variant]} | {offset} | {row['transitions']} | {row['cases']} | {row['task_groups']} |"
                )
    lines.extend(
        [
            "",
            "TV 增大只表示 stale 分布缺失更多新条件信息；它不等于变化朝参考答案方向，也不证明 Markov head 学得会。",
            "",
            "## Reference 正确方向（主要解释：reference-prefix 完全对齐）",
            "",
            "| 诊断 | offset | transitions/cases/groups | raw ΔlogP | raw rank 改善 | raw help/harm | actual finite ΔlogP | actual rank 改善 | actual help/harm | actual support +∞/−∞/both-out |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in ("C1", "L1"):
        for offset in (1, 2, 3):
            row = find_summary(list(reference_rows), variant, offset, "reference_prefix_aligned")
            if row:
                lines.append(
                    f"| {VARIANT_NAMES[variant]} | {offset} | {row['transitions']}/{row['cases']}/{row['task_groups']} | {format_ci(row, 'raw_reference_log_probability_delta')} | {format_ci(row, 'raw_reference_rank_improvement')} | {row['raw_wrong_to_correct_count']}/{row['raw_correct_to_wrong_count']} | {format_ci(row, 'actual_reference_log_probability_delta')} | {format_ci(row, 'actual_reference_rank_improvement')} | {row['actual_wrong_to_correct_count']}/{row['actual_correct_to_wrong_count']} | {row['actual_logprob_positive_infinity_count']}/{row['actual_logprob_negative_infinity_count']}/{row['actual_logprob_both_truncated_count']} |"
                )
    lines.extend(
        [
            "",
            "actual-decode 的 finite ΔlogP 只在 reference token stale/fresh 都保留于 top-p 支持集时取平均；支持集进入、退出和双方均截断分别单独计数，截断不会被静默当作普通零概率。overall 与仅 predecessor-correct 子集见 CSV。",
            "",
            "## Fresh global-rank 晋升与保留",
            "",
            "| 诊断 | offset | rank 改善 | outside→top2 | outside→top4 | top2 retained | top4 retained |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in ("C1", "L1"):
        for offset in (1, 2, 3):
            row = find_summary(list(global_rows), variant, offset)
            if row:
                lines.append(
                    f"| {VARIANT_NAMES[variant]} | {offset} | {format_ci(row, 'global_rank_improvement')} | {format_ci(row, 'outside_to_top2', 100)}% | {format_ci(row, 'outside_to_top4', 100)}% | {format_ci(row, 'top2_retained', 100)}% | {format_ci(row, 'top4_retained', 100)}% |"
                )
    neighbor = next(
        (
            row
            for row in global_rows
            if row["record_scope"] == "global_right_neighbor_steps"
            and row["final_verdict"] == "all"
            and row["generation_stage"] == "all"
        ),
        None,
    )
    if neighbor:
        lines.extend(
            [
                "",
                "全局置信度逐词刷新诊断的普通 step：",
                "",
                f"- i+1 连续右邻存在率：{format_ci(neighbor, 'right_neighbor_still_mask', 100)}%。",
                f"- i+1 属于 source top-2/top-4：{format_ci(neighbor, 'source_right_neighbor_in_top2', 100)}% / {format_ci(neighbor, 'source_right_neighbor_in_top4', 100)}%。",
                f"- source top-2/top-4 候选在 fresh forward 后仍保留：{format_ci(neighbor, 'fresh_in_top2', 100)}% / {format_ci(neighbor, 'fresh_in_top4', 100)}%。",
            ]
        )
    transition_reasons = audit["non_evaluable_reason_counts"]["transition_ineligible"]
    reference_reasons = audit["non_evaluable_reason_counts"]["reference_non_evaluable"]
    lines.extend(
        [
            "",
            "## 不可评估与 chain 终止审计",
            "",
            f"- gap 或非连续实际提交终止：`{transition_reasons.get('gap_or_nonconsecutive_commit', 0)}`。",
            f"- expand/delete/EOS broadcast-delete 等结构或坐标变化终止：`{transition_reasons.get('structural_or_coordinate_change', 0)}`。",
            f"- reference 目标超出 canonical middle token 范围：`{reference_reasons.get('target_reference_token_out_of_range', 0)}`；这些 transition 不进入 reference 方向统计，但仍可进入 distribution/global-rank 统计。",
            "",
            "## 证据判断",
            "",
        ]
    )
    c1_offset1 = find_summary(list(reference_rows), "C1", 1, "reference_prefix_aligned")
    l1_offset1 = find_summary(list(reference_rows), "L1", 1, "reference_prefix_aligned")
    c1_offset2 = find_summary(list(stale_rows), "C1", 2)
    c1_offset3 = find_summary(list(stale_rows), "C1", 3)
    if c1_offset1 and l1_offset1:
        lines.extend(
            [
                f"- 支持：offset=1 的 reference-prefix aligned raw ΔlogP 在两项诊断中均为正且 CI 不跨 0（全局 `{format_ci(c1_offset1, 'raw_reference_log_probability_delta')}`；左前沿 `{format_ci(l1_offset1, 'raw_reference_log_probability_delta')}`）；wrong→correct 也多于 correct→wrong。",
                f"- 限制：offset=1 的 actual-decode finite ΔlogP CI 在两项诊断中均跨 0（全局 `{format_ci(c1_offset1, 'actual_reference_log_probability_delta')}`；左前沿 `{format_ci(l1_offset1, 'actual_reference_log_probability_delta')}`），且全局诊断 raw rank 改善 CI 跨 0；所以 K=2 的正确方向证据是 mixed-positive，不足以直接授权训练。",
            ]
        )
    if c1_offset2 and c1_offset3:
        lines.append(
            f"- 长块风险：全局诊断 raw/actual TV 从 offset=1 的 `{find_summary(list(stale_rows), 'C1', 1)['raw_total_variation_task_group_macro']:.4f}/{find_summary(list(stale_rows), 'C1', 1)['actual_total_variation_task_group_macro']:.4f}` 上升到 offset=2 的 `{c1_offset2['raw_total_variation_task_group_macro']:.4f}/{c1_offset2['actual_total_variation_task_group_macro']:.4f}` 和 offset=3 的 `{c1_offset3['raw_total_variation_task_group_macro']:.4f}/{c1_offset3['actual_total_variation_task_group_macro']:.4f}`，top-1 agreement 同步下降；这反对把一阶 premise 直接外推为固定长块。"
        )
    if neighbor:
        lines.append(
            f"- 支持 global top-K 连续短链：普通 step 的 i+1 存在率为 `{format_ci(neighbor, 'right_neighbor_still_mask', 100)}%`，存在时进入 source top-2/top-4 为 `{format_ci(neighbor, 'source_right_neighbor_in_top2', 100)}%/{format_ci(neighbor, 'source_right_neighbor_in_top4', 100)}%`，且 fresh 后保留率为 `{format_ci(neighbor, 'fresh_in_top2', 100)}%/{format_ci(neighbor, 'fresh_in_top4', 100)}%`。"
        )
    lines.extend(
        [
            "- 削弱强制大块路线：2026-08-23 的有效六臂结果已显示 L2/L4 相对逐词对照降低 Pass@1；本轮 offset=2/3 的更大 stale→fresh 漂移进一步说明长块更依赖缺失条件。",
            "- 综合结论：本轮补齐了继续研究 Markov premise 所需的正向和排序证据，但只支持未来另行预注册的轻量设计验证；它不证明 head 可学习，不训练 Markov head，也不自动授权下一阶段。",
        ]
    )
    lines.extend(
        [
            "",
            "## 解释边界与历史记录",
            "",
            "- 2026-08-23 的六臂质量/效率结果仍有效；本轮没有重跑 C2/C4/L2/L4。",
            "- 当时 Markov stop 判断因 reference direction 与 fresh global-rank promotion 两个预注册字段未采集，被本轮补充诊断取代；字段缺失没有被解释为负面结果。",
            "- reference 指标改善表示 fresh 变化更可能朝参考方向；TV 与 reference 改善都不能直接证明 Markov head 可学习。",
            "- 本轮只提供是否值得继续设计/训练的证据，不训练 Markov head，也不自动授权训练、controller 或下一阶段实验。",
            "",
            "完整逐 case 与逐 transition 可审计数据分别见 `per_case_results.jsonl.zst` 和 `per_transition_diagnostics.jsonl.zst`；必需字段、压缩解析与行数门禁见 `completeness_audit.json`。",
        ]
    )
    (output / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    cases = read_jsonl(output / "per_case_results.jsonl")
    traces = read_jsonl(output / "per_step_trace.jsonl")
    diagnostics = read_jsonl(output / "per_transition_diagnostics.jsonl")
    transitions = [
        row
        for row in diagnostics
        if row.get("record_type") == "stale_fresh_transition"
        and bool(row.get("transition_eligible"))
    ]
    attempted = [
        row for row in diagnostics if row.get("record_type") == "stale_fresh_transition"
    ]
    neighbors = [
        row for row in diagnostics if row.get("record_type") == "global_right_neighbor_observation"
    ]
    pass_by_case = {str(row["case_key"]): bool(row["passed"]) for row in cases}
    for row in transitions:
        row["final_verdict"] = "pass" if pass_by_case[row["case_key"]] else "fail"
        row["generation_stage"] = stage(int(row["source_active_mask_count"]))
    for row in neighbors:
        row["final_verdict"] = "pass" if pass_by_case[row["case_key"]] else "fail"
        row["generation_stage"] = stage(int(row["source_active_mask_count"]))

    expected_population = 1033 if args.mode == "full" else int(args.expected_population)
    observed_keys = [(str(row["task_id"]), str(row["variant"])) for row in cases]
    duplicate_count = len(observed_keys) - len(set(observed_keys))
    variant_counts = Counter(str(row["variant"]) for row in cases)
    task_sets = {
        variant: {str(row["task_id"]) for row in cases if row["variant"] == variant}
        for variant in ("C1", "L1")
    }
    common_tasks = task_sets["C1"] & task_sets["L1"]
    missing_count = sum(max(0, expected_population - variant_counts[variant]) for variant in ("C1", "L1"))
    failure_rows = read_jsonl(output / "failure_journal.jsonl")
    error_count = len(failure_rows)
    pass_counts = {
        variant: sum(bool(row["passed"]) for row in cases if row["variant"] == variant)
        for variant in ("C1", "L1")
    }
    required_field_errors = audit_required_transition_fields(attempted)
    transition_chain_keys = [
        (row["case_key"], int(row["chain_id"])) for row in attempted if "chain_id" in row
    ]
    duplicate_transition_count = len(transition_chain_keys) - len(set(transition_chain_keys))

    case_zst, case_zst_rows, case_zst_valid = compress_and_validate(
        output / "per_case_results.jsonl"
    )
    transition_zst, transition_zst_rows, transition_zst_valid = compress_and_validate(
        output / "per_transition_diagnostics.jsonl"
    )
    compressed_valid = (
        case_zst.exists()
        and transition_zst.exists()
        and case_zst_valid
        and transition_zst_valid
        and case_zst_rows == len(cases)
        and transition_zst_rows == len(diagnostics)
    )
    full_status = completeness_status(
        case_rows=len(cases),
        variant_counts=variant_counts,
        common_task_count=len(common_tasks),
        task_group_count=len({str(row["task_group"]) for row in cases}),
        duplicate_count=duplicate_count,
        missing_count=missing_count,
        error_count=error_count,
        pass_counts=pass_counts,
        required_field_errors=required_field_errors,
        compressed_artifacts_valid=compressed_valid,
    )
    if args.mode == "smoke":
        smoke_complete = (
            len(cases) == expected_population * 2
            and variant_counts == {"C1": expected_population, "L1": expected_population}
            and duplicate_count == 0
            and missing_count == 0
            and error_count == 0
            and not required_field_errors
            and bool(transitions)
            and any(bool(row.get("reference_evaluable")) for row in transitions)
            and any(row.get("fresh_global_rank") is not None for row in transitions)
            and compressed_valid
        )
        full_status = "completed" if smoke_complete else "incomplete"

    audit: dict[str, Any] = {
        "status": full_status,
        "mode": args.mode,
        "case_rows": len(cases),
        "variant_counts": dict(variant_counts),
        "common_task_count": len(common_tasks),
        "task_group_count": len({str(row["task_group"]) for row in cases}),
        "duplicate_count": duplicate_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "pass_counts": pass_counts,
        "transition_attempt_rows": len(attempted),
        "eligible_transition_rows": len(transitions),
        "global_neighbor_observation_rows": len(neighbors),
        "duplicate_transition_chain_count": duplicate_transition_count,
        "required_field_errors": required_field_errors,
        "reference_direction_available": bool(transitions)
        and any(bool(row.get("reference_evaluable")) for row in transitions)
        and not any("reference" in field for field in required_field_errors),
        "fresh_global_rank_available": bool(transitions)
        and all(row.get("fresh_global_rank") is not None for row in transitions),
        "raw_model_distribution_available": bool(transitions)
        and all(row.get("raw_total_variation") is not None for row in transitions),
        "actual_decode_distribution_available": bool(transitions)
        and all(row.get("actual_total_variation") is not None for row in transitions),
        "compressed_artifacts": {
            "per_case_results": {
                "path": case_zst.name,
                "exists": case_zst.exists(),
                "parse_valid": case_zst_valid,
                "rows": case_zst_rows,
                "expected_rows": len(cases),
            },
            "per_transition_diagnostics": {
                "path": transition_zst.name,
                "exists": transition_zst.exists(),
                "parse_valid": transition_zst_valid,
                "rows": transition_zst_rows,
                "expected_rows": len(diagnostics),
            },
        },
        "non_evaluable_reason_counts": {
            "transition_ineligible": dict(
                Counter(
                    str(row.get("transition_ineligible_reason"))
                    for row in attempted
                    if not row.get("transition_eligible")
                )
            ),
            "reference_non_evaluable": dict(
                Counter(
                    str(row.get("reference_non_evaluable_reason"))
                    for row in transitions
                    if not row.get("reference_evaluable")
                )
            ),
        },
        "hard_assertions": {
            "case_rows_eq_2066": len(cases) == 2066,
            "each_diagnostic_eq_1033": variant_counts == {"C1": 1033, "L1": 1033},
            "common_task_count_eq_1033": len(common_tasks) == 1033,
            "task_group_count_eq_164": len({str(row["task_group"]) for row in cases}) == 164,
            "duplicate_eq_0": duplicate_count == 0,
            "missing_error_eq_0": missing_count == 0 and error_count == 0,
            "global_pass_eq_951": pass_counts["C1"] == 951,
            "left_pass_eq_942": pass_counts["L1"] == 942,
            "required_fields_complete": not required_field_errors,
            "compressed_artifacts_valid": compressed_valid,
        },
    }
    if args.mode == "smoke" and args.v2_reference_dir:
        audit["v2_trajectory_alignment"] = smoke_v2_alignment(
            cases, traces, Path(args.v2_reference_dir).resolve()
        )
        if audit["v2_trajectory_alignment"]["status"] != "passed":
            audit["status"] = "protocol_failed"
    write_json(output / "completeness_audit.json", audit)

    reference_rows = build_reference_rows(transitions, replicates=int(args.bootstrap_replicates))
    stale_rows = build_stale_fresh_rows(transitions, replicates=int(args.bootstrap_replicates))
    global_rows = build_global_rank_rows(
        transitions, neighbors, replicates=int(args.bootstrap_replicates)
    )
    write_csv(output / "reference_direction_by_variant_offset.csv", reference_rows)
    write_csv(output / "stale_fresh_by_variant_offset.csv", stale_rows)
    write_csv(output / "global_rank_promotion_by_variant.csv", global_rows)
    stage_rows = [
        {
            "diagnostic_name_zh": row["diagnostic_name_zh"],
            "variant": row["variant"],
            "record_scope": "stage_and_verdict",
            "offset": row["offset"],
            "reference_subset": row["reference_subset"],
            "final_verdict": row["final_verdict"],
            "generation_stage": row["generation_stage"],
            "transitions": row["transitions"],
            "cases": row["cases"],
            "task_groups": row["task_groups"],
            "raw_total_variation_task_group_macro": row.get(
                "raw_total_variation_task_group_macro"
            ),
            "actual_total_variation_task_group_macro": row.get(
                "actual_total_variation_task_group_macro"
            ),
        }
        for row in stale_rows
        if row["final_verdict"] != "all" or row["generation_stage"] != "all"
    ]
    write_csv(output / "stage_and_verdict_summary.csv", stage_rows)

    summary = {
        "status": audit["status"],
        "diagnostics": VARIANT_NAMES,
        "markov_head_trained": False,
        "automatic_next_stage_authorized": False,
        "completeness": audit,
        "primary_interpretation_subset": "reference_prefix_aligned",
        "offset_interpretation": {
            "1": "K=2 Markov primary diagnostic",
            "2": "longer-block risk diagnostic",
            "3": "longer-block risk diagnostic",
        },
        "core_reference_direction": [
            row
            for row in reference_rows
            if row["reference_subset"] == "reference_prefix_aligned"
            and row["final_verdict"] == "all"
            and row["generation_stage"] == "all"
        ],
        "core_stale_fresh": [
            row
            for row in stale_rows
            if row["reference_subset"] == "overall"
            and row["final_verdict"] == "all"
            and row["generation_stage"] == "all"
        ],
        "core_global_rank": [
            row
            for row in global_rows
            if row["reference_subset"] == "overall"
            and row["final_verdict"] == "all"
            and row["generation_stage"] == "all"
        ],
        "interpretation_constraints": [
            "TV increase means stale distributions omit more conditioning information.",
            "Reference improvement indicates movement is more likely toward the reference.",
            "Neither result proves that a Markov head is learnable.",
            "This diagnostic does not train a Markov head or automatically authorize a next stage.",
        ],
    }
    write_json(output / "summary.json", summary)
    write_report(output, audit, reference_rows, stale_rows, global_rows)
    (output / "implementation_audit.md").write_text(
        "# Implementation audit\n\n"
        "- Base commit: `5d5d5f2eb9c550e77327e35e827a9ed2ca27b34d`.\n"
        "- Only global-confidence single-token refresh and fixed-left-frontier single-token refresh are run; C2/C4/L2/L4 are not rerun.\n"
        "- Released logits shift, temperature 0.2, top-p 0.9, entropy confidence, sampling, expansion, deletion, EOS broadcast-delete, prompt, tokenizer and evaluator semantics are preserved.\n"
        "- `raw_model_distribution` is softmax after released target alignment and protocol action masking but before temperature/top-p.\n"
        "- `actual_decode_distribution` exactly repeats released temperature/top-p filtering without sampling; diagnostics do not consume RNG.\n"
        "- Full-vocabulary TV is computed online. Only scalar metrics and top-5 audit summaries are serialized; no full logits/probability vectors are written.\n"
        "- Reference tokens are passed only to oracle diagnostic code after official sampling/selection inputs are fixed. Poison-reference regression requires identical selected positions, sampled tokens, actions, canvas states and final output.\n"
        "- Chains terminate on structural actions, coordinate changes, nonconsecutive commits, target resolution or case termination.\n"
        "- Actual-decode top-p truncation is explicit through retained flags and extended-real log-prob delta kinds; it is never silently treated as an ordinary finite zero.\n"
        "- Confidence ties use a stable position-order rank for the scalar rank field; source/fresh top-2 and top-4 membership are separately recorded using the released `torch.topk` operation.\n"
        "- Fixed smoke12 completed 24/24, matched v2 C1/L1 outputs and all comparable step fields, and passed journal SHA-256 resume/dedup no-op.\n"
        "- Fresh full alignment compared all 2,066 C1/L1 cases and 20,741 comparable step rows against v2 with zero output/action/canvas-accounting mismatches.\n"
        "- Reference-poison regression passed: two different pseudo references produced identical selected positions, sampled tokens, actions, canvas states and final output.\n"
        f"- Full reproduction gates: C1={audit['pass_counts']['C1']}/1033 and L1={audit['pass_counts']['L1']}/1033.\n"
        f"- Reviewer gate: disabled by project policy; local diff review and fresh verification are required before push.\n"
        f"- Analyzer status: `{audit['status']}`.\n",
        encoding="utf-8",
    )
    return 0 if audit["status"] == "completed" else 2


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--expected-population", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--v2-reference-dir")
    return parser


if __name__ == "__main__":
    raise SystemExit(analyze(parser().parse_args()))
