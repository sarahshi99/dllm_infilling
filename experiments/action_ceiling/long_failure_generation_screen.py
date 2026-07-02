#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl
from experiments.action_ceiling.action_ceiling_matrix import (
    DEFAULT_BASELINE,
    DEFAULT_ROUTE2,
    canvas_is_oracle_sufficient,
    current_branch,
    current_commit,
    environment_info,
    git_capture,
    oracle_sufficient_length,
    parse_task_id_group,
    reset_action_seed,
    rows_by_task,
    stable_task_seed,
    verification_error_type,
    write_csv,
    write_jsonl,
)
from experiments.action_ceiling.distinct_candidate_ceiling import (
    DIAGNOSTIC_ACTIONS,
    compact_result_record,
    decode_fixed_canvas,
    result_record,
    run_trace_remask,
    run_trace_span_remask,
)
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import load_humaneval_infilling
from expvision_dllm_clean.modeling import load_model_and_tokenizer


JsonDict = Dict[str, Any]

EXPECTED_TRIGGERED_FAILED_LONG = 33
EXPECTED_MISSED_FAILED_LONG = 56
EXPECTED_POSITIVE_CONTROLS = 6
SEED0_ACTIONS = (
    "E_oracle_sufficient_no_early_commit",
    "F_oracle_sufficient_trace_remask",
    "G_oracle_sufficient_trace_span_remask",
)
MULTISEED_SEEDS = (1, 2)
MAX_MULTISEED_CASES = 30
MAX_ACTION_BUDGET = 600


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else sum(values) / len(values)


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def read_csv_rows(path: Path) -> List[JsonDict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def make_output_dir(base_dir: str, timestamp: Optional[str]) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"long_failure_generation_screen_{stamp}"
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def original_humaneval_task(task_id: str) -> str:
    return parse_task_id_group(task_id)


def historical_error_type(task_id: str, baseline_by_task: Mapping[str, Mapping[str, Any]], route2_by_task: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    route2_error = verification_error_type(route2_by_task[task_id])
    if route2_error is not None:
        return route2_error
    return verification_error_type(baseline_by_task[task_id])


def case_from_history_row(
    row: Mapping[str, Any],
    *,
    pool: str,
    baseline_by_task: Mapping[str, Mapping[str, Any]],
    route2_by_task: Mapping[str, Mapping[str, Any]],
    max_canvas_length: int,
) -> JsonDict:
    task_id = str(row["task_id"])
    oracle_len = _to_int(row.get("oracle_length"))
    primary_len = _to_int(row.get("baseline_selected_length") or row.get("route2_primary_selected_length"))
    route2_len = _to_int(row.get("route2_rescue_length") or row.get("route2_selected_length"))
    sufficient_len = oracle_sufficient_length(
        primary_len=primary_len,
        route2_len=route2_len,
        oracle_len=oracle_len,
        max_length=max_canvas_length,
    )
    if sufficient_len is None:
        sufficient_len = oracle_len
    return {
        "task_id": task_id,
        "original_humaneval_task": original_humaneval_task(task_id),
        "task_group": original_humaneval_task(task_id),
        "pool": pool,
        "case_pool": pool,
        "oracle_length": oracle_len,
        "primary_length": primary_len,
        "primary_selected_length": primary_len,
        "route2_length": route2_len,
        "route2_rescue_length": route2_len,
        "oracle_sufficient_length": sufficient_len,
        "max_canvas_length": max_canvas_length,
        "canvas_is_oracle_sufficient": canvas_is_oracle_sufficient(sufficient_len, oracle_len),
        "historical_primary_pass": _to_bool(row.get("baseline_passed")),
        "baseline_passed": _to_bool(row.get("baseline_passed")),
        "historical_route2_pass": _to_bool(row.get("route2_passed")),
        "route2_passed": _to_bool(row.get("route2_passed")),
        "trigger_status": "triggered" if _to_bool(row.get("route2_triggered")) else "missed",
        "route2_triggered": _to_bool(row.get("route2_triggered")),
        "historical_error_type": historical_error_type(task_id, baseline_by_task, route2_by_task),
        "route2_trace_top1_last": row.get("route2_trace_top1_last"),
        "route2_trace_top1_median": row.get("route2_trace_top1_median"),
        "route2_trace_confidence_max": row.get("route2_trace_confidence_max"),
        "route2_trace_max_remaining_plateau_steps": row.get("route2_trace_max_remaining_plateau_steps"),
    }


def build_95_case_manifest(args: argparse.Namespace) -> Tuple[List[JsonDict], Optional[str]]:
    analysis_dir = Path(args.route2_error_analysis_dir)
    triggered = read_csv_rows(analysis_dir / "triggered_failed_long.csv")
    missed = read_csv_rows(analysis_dir / "missed_failed_long.csv")
    wins = read_csv_rows(analysis_dir / "wins.csv")
    baseline_by_task = rows_by_task(load_jsonl(args.baseline_results), name="baseline")
    route2_by_task = rows_by_task(load_jsonl(args.route2_results), name="route2")

    rows: List[JsonDict] = []
    rows.extend(
        case_from_history_row(
            row,
            pool="triggered_failed_long",
            baseline_by_task=baseline_by_task,
            route2_by_task=route2_by_task,
            max_canvas_length=args.max_canvas_length,
        )
        for row in triggered
    )
    rows.extend(
        case_from_history_row(
            row,
            pool="missed_failed_long",
            baseline_by_task=baseline_by_task,
            route2_by_task=route2_by_task,
            max_canvas_length=args.max_canvas_length,
        )
        for row in missed
    )
    positive_rows = sorted(
        wins,
        key=lambda row: (0 if str(row.get("classes", "")).find("triggered_rescued_long") >= 0 else 1, str(row["task_id"])),
    )[:EXPECTED_POSITIVE_CONTROLS]
    rows.extend(
        case_from_history_row(
            row,
            pool="positive_control_rescued",
            baseline_by_task=baseline_by_task,
            route2_by_task=route2_by_task,
            max_canvas_length=args.max_canvas_length,
        )
        for row in positive_rows
    )

    counts = Counter(row["pool"] for row in rows)
    duplicates = [task_id for task_id, count in Counter(row["task_id"] for row in rows).items() if count > 1]
    mismatch_reasons = []
    if counts["triggered_failed_long"] != EXPECTED_TRIGGERED_FAILED_LONG:
        mismatch_reasons.append(f"triggered_failed_long expected {EXPECTED_TRIGGERED_FAILED_LONG}, got {counts['triggered_failed_long']}")
    if counts["missed_failed_long"] != EXPECTED_MISSED_FAILED_LONG:
        mismatch_reasons.append(f"missed_failed_long expected {EXPECTED_MISSED_FAILED_LONG}, got {counts['missed_failed_long']}")
    if counts["positive_control_rescued"] < EXPECTED_POSITIVE_CONTROLS:
        mismatch_reasons.append(f"positive_control_rescued expected >= {EXPECTED_POSITIVE_CONTROLS}, got {counts['positive_control_rescued']}")
    if duplicates:
        mismatch_reasons.append(f"duplicate task_ids: {duplicates}")
    if len(rows) != EXPECTED_TRIGGERED_FAILED_LONG + EXPECTED_MISSED_FAILED_LONG + EXPECTED_POSITIVE_CONTROLS:
        mismatch_reasons.append(f"total expected 95, got {len(rows)}")
    return rows, "\n".join(mismatch_reasons) if mismatch_reasons else None


def build_action_manifest(cases: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for case in cases:
        for action_id in SEED0_ACTIONS:
            rows.append(
                {
                    "task_id": case["task_id"],
                    "pool": case["pool"],
                    "action_id": action_id,
                    "experimental_seeds": "0",
                    "oracle_sufficient_length": case["oracle_sufficient_length"],
                    "planned_steps": 64 if action_id.startswith("E_") else (80 if action_id.startswith("F_") else 88),
                }
            )
    return rows


def result_for_action(
    *,
    action_id: str,
    case: Mapping[str, Any],
    task: Any,
    tokenizer: Any,
    model: Any,
    cfg: ExperimentConfig,
    experimental_seed: int,
    source_commit: str,
) -> JsonDict:
    task_seed = stable_task_seed(42, str(case["task_id"]), experimental_seed)
    reset_action_seed(task_seed)
    canvas_len = int(case["oracle_sufficient_length"])
    if action_id == "E_oracle_sufficient_no_early_commit":
        result = decode_fixed_canvas(
            task=task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            canvas_len=canvas_len,
            total_steps=64,
            early_commit_enabled=False,
            phase_name="screen_e_no_early_commit",
        )
        note = "executed_oracle_sufficient_no_early_commit"
    elif action_id == "F_oracle_sufficient_trace_remask":
        result = run_trace_remask(
            task=task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            canvas_len=canvas_len,
            total_steps=64,
            refinement_steps=16,
        )
        note = "executed_trace_token_remask_refinement"
    elif action_id == "G_oracle_sufficient_trace_span_remask":
        result = run_trace_span_remask(
            task=task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            canvas_len=canvas_len,
            total_steps=64,
            refinement_steps=24,
        )
        note = "executed_trace_span_remask_refinement"
    else:
        raise ValueError(f"unknown screen action: {action_id}")
    return result_record(
        case=case,
        action_id=action_id,
        result=result,
        experimental_seed=experimental_seed,
        task_seed=task_seed,
        source_commit=source_commit,
        action_was_executed=True,
        execution_note=note,
    )


def fill_c_reference_flags(rows: Sequence[JsonDict]) -> None:
    stage1_by_task_seed: Dict[Tuple[str, int], str] = {}
    for row in rows:
        if row.get("stage1_hash"):
            key = (str(row["task_id"]), int(row.get("experimental_seed") or 0))
            stage1_by_task_seed.setdefault(key, str(row["stage1_hash"]))
    for row in rows:
        key = (str(row["task_id"]), int(row.get("experimental_seed") or 0))
        c_hash = stage1_by_task_seed.get(key)
        row["c_reference_hash"] = c_hash
        row["output_changed_vs_C"] = None if not c_hash else row.get("generated_text_sha256") != c_hash


def run_actions_for_cases(
    *,
    cases: Sequence[Mapping[str, Any]],
    actions: Sequence[str],
    seeds: Sequence[int],
    by_task: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    cfg: ExperimentConfig,
    source_commit: str,
) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for case_index, case in enumerate(cases, start=1):
        task_id = str(case["task_id"])
        task = by_task[task_id]
        for seed in seeds:
            for action_id in actions:
                print(
                    f"[{case_index}/{len(cases)}] seed={seed} action={action_id} task_id={task_id}",
                    flush=True,
                )
                rows.append(
                    result_for_action(
                        action_id=action_id,
                        case=case,
                        task=task,
                        tokenizer=tokenizer,
                        model=model,
                        cfg=cfg,
                        experimental_seed=seed,
                        source_commit=source_commit,
                    )
                )
    fill_c_reference_flags(rows)
    return rows


def g_action_sanity(
    *,
    case_by_id: Mapping[str, Mapping[str, Any]],
    by_task: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    cfg: ExperimentConfig,
    source_commit: str,
) -> Tuple[List[JsonDict], JsonDict]:
    sanity_cases = [case_by_id["SingleLineInfilling/HumanEval/85/L0"], case_by_id["SingleLineInfilling/HumanEval/113/L3"]]
    rows = run_actions_for_cases(
        cases=sanity_cases,
        actions=("F_oracle_sufficient_trace_remask", "G_oracle_sufficient_trace_span_remask"),
        seeds=(0,),
        by_task=by_task,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        source_commit=source_commit,
    )
    by_case = defaultdict(list)
    for row in rows:
        by_case[str(row["task_id"])].append(row)
    case_status = {}
    for task_id, task_rows in by_case.items():
        g = next(row for row in task_rows if row["action_id"] == "G_oracle_sufficient_trace_span_remask")
        f = next(row for row in task_rows if row["action_id"] == "F_oracle_sufficient_trace_remask")
        case_status[task_id] = {
            "g_hash": g.get("generated_text_sha256"),
            "f_hash": f.get("generated_text_sha256"),
            "c_hash": g.get("c_reference_hash") or f.get("c_reference_hash"),
            "g_hash_distinct_from_c_or_f": g.get("generated_text_sha256") not in {g.get("c_reference_hash"), f.get("generated_text_sha256")},
            "g_remasked_span_width": g.get("remasked_span_width"),
            "g_remasked_token_count": g.get("remasked_token_count"),
            "g_effective_update_steps": g.get("effective_update_steps"),
            "g_span_refinement_executed": g.get("span_refinement_executed"),
            "g_mechanism_real_executed": bool(g.get("span_refinement_executed"))
            and int(g.get("remasked_token_count") or 0) > 0
            and int(g.get("remasked_span_width") or 0) > 0,
        }
    passed = all(status["g_mechanism_real_executed"] for status in case_status.values())
    return rows, {"passed": passed, "cases": case_status}


def summarize_seed0(cases: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_case = defaultdict(list)
    for row in rows:
        by_case[str(row["task_id"])].append(row)
    case_by_id = {str(case["task_id"]): case for case in cases}
    case_summaries: Dict[str, JsonDict] = {}
    for task_id, task_rows in sorted(by_case.items()):
        hashes = {row.get("generated_text_sha256") for row in task_rows if row.get("generated_text_sha256")}
        case = case_by_id[task_id]
        compile_repaired = case.get("historical_error_type") == "SyntaxError" and any(row.get("compile_passed") for row in task_rows)
        case_summaries[task_id] = {
            "pool": case["pool"],
            "historical_error_type": case.get("historical_error_type"),
            "unique_candidate_hash_count": len(hashes),
            "any_correct": any(bool(row.get("passed")) for row in task_rows),
            "compile_repaired": compile_repaired,
            "all_actions_unchanged": all(row.get("output_changed_vs_C") is False for row in task_rows),
            "all_actions_distinct_but_wrong": len(hashes) == len(task_rows) and not any(bool(row.get("passed")) for row in task_rows),
            "error_types": dict(Counter(str(row.get("error_type")) for row in task_rows)),
            "actions_passed": [row["action_id"] for row in task_rows if row.get("passed")],
            "max_trace_instability": max(
                int(row.get("remasked_token_count") or 0) + int(row.get("effective_update_steps") or 0)
                for row in task_rows
            ),
        }
    pool_summary: Dict[str, JsonDict] = {}
    for pool in ["positive_control_rescued", "triggered_failed_long", "missed_failed_long"]:
        task_ids = [task_id for task_id, summary in case_summaries.items() if summary["pool"] == pool]
        pool_rows = [row for row in rows if str(row["task_id"]) in set(task_ids)]
        costs = [float(row["total_sec_including_probe"]) for row in pool_rows if row.get("total_sec_including_probe") is not None]
        pool_summary[pool] = {
            "case_count": len(task_ids),
            "action_count": len(pool_rows),
            "new_correct_candidate_cases": sum(1 for task_id in task_ids if case_summaries[task_id]["any_correct"]),
            "unique_candidate_case_count": sum(1 for task_id in task_ids if case_summaries[task_id]["unique_candidate_hash_count"] > 1),
            "unchanged_by_all_actions": sum(1 for task_id in task_ids if case_summaries[task_id]["all_actions_unchanged"]),
            "all_actions_distinct_but_wrong": sum(1 for task_id in task_ids if case_summaries[task_id]["all_actions_distinct_but_wrong"]),
            "compile_repair_cases": sum(1 for task_id in task_ids if case_summaries[task_id]["compile_repaired"]),
            "compile_success_actions": sum(1 for row in pool_rows if row.get("compile_passed")),
            "mean_cost_sec": _mean(costs),
            "p50_cost_sec": _percentile(costs, 50),
            "p95_cost_sec": _percentile(costs, 95),
        }
    action_summary: Dict[str, JsonDict] = {}
    for action_id in SEED0_ACTIONS:
        action_rows = [row for row in rows if row.get("action_id") == action_id]
        costs = [float(row["total_sec_including_probe"]) for row in action_rows if row.get("total_sec_including_probe") is not None]
        action_summary[action_id] = {
            "action_count": len(action_rows),
            "passed_cases": sum(1 for row in action_rows if row.get("passed")),
            "hard_passed_cases": sum(1 for row in action_rows if row.get("passed") and row.get("case_pool") != "positive_control_rescued"),
            "compile_success": sum(1 for row in action_rows if row.get("compile_passed")),
            "output_changed_vs_C": sum(1 for row in action_rows if row.get("output_changed_vs_C") is True),
            "avg_remasked_token_count": _mean([float(row.get("remasked_token_count") or 0) for row in action_rows]),
            "avg_remasked_span_width": _mean([float(row.get("remasked_span_width") or 0) for row in action_rows]),
            "avg_effective_update_steps": _mean([float(row.get("effective_update_steps") or 0) for row in action_rows]),
            "mean_cost_sec": _mean(costs),
            "p50_cost_sec": _percentile(costs, 50),
            "p95_cost_sec": _percentile(costs, 95),
        }
    hard_cases = [summary for summary in case_summaries.values() if summary["pool"] != "positive_control_rescued"]
    hard_correct = [summary for summary in hard_cases if summary["any_correct"]]
    diversity_without_correctness = [
        summary for summary in hard_cases if summary["unique_candidate_hash_count"] > 1 and not summary["any_correct"]
    ]
    all_actions_unchanged = [summary for summary in hard_cases if summary["all_actions_unchanged"]]
    verdict = choose_final_verdict(case_summaries, rows, multiseed_rows=[])
    return {
        "mode": "long_failure_generation_screen_seed0",
        "case_count": len(cases),
        "hard_case_count": len(hard_cases),
        "positive_control_count": sum(1 for case in cases if case["pool"] == "positive_control_rescued"),
        "seed0_action_count": len(rows),
        "seed0_generation_actions_expected": len(cases) * len(SEED0_ACTIONS),
        "pool_summary": pool_summary,
        "action_summary": action_summary,
        "case_summary": case_summaries,
        "hard_new_correct_candidate_count": len(hard_correct),
        "diversity_without_correctness_count": len(diversity_without_correctness),
        "all_actions_unchanged_count": len(all_actions_unchanged),
        "final_verdict_seed0_only": verdict,
    }


def choose_multiseed_cases(seed0_summary: Mapping[str, Any]) -> List[JsonDict]:
    selected: List[Tuple[Tuple[int, float, str], JsonDict]] = []
    for task_id, summary in (seed0_summary.get("case_summary") or {}).items():
        priority = 99
        if summary.get("any_correct"):
            priority = 1
        elif summary.get("compile_repaired"):
            priority = 2
        elif "None" in (summary.get("error_types") or {}) and summary.get("historical_error_type") not in {None, "None"}:
            priority = 3
        elif int(summary.get("unique_candidate_hash_count") or 0) >= 2:
            priority = 4
        else:
            priority = 5
        selected.append(((priority, -float(summary.get("max_trace_instability") or 0.0), str(task_id)), {"task_id": task_id, **summary, "selection_priority": priority}))
    selected.sort(key=lambda item: item[0])
    return [item[1] for item in selected[:MAX_MULTISEED_CASES]]


def summarize_multiseed(rows: Sequence[Mapping[str, Any]], selected_cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_case = defaultdict(list)
    for row in rows:
        by_case[str(row["task_id"])].append(row)
    return {
        "mode": "conditional_multiseed_diagnostic",
        "selection_note": "Conditional diagnostic only; do not interpret selected-case Pass@3 as an unbiased benchmark metric.",
        "selected_case_count": len(selected_cases),
        "multiseed_action_count": len(rows),
        "seeds": list(MULTISEED_SEEDS),
        "case_summary": {
            task_id: {
                "unique_hash_count": len({row.get("generated_text_sha256") for row in task_rows}),
                "any_correct": any(bool(row.get("passed")) for row in task_rows),
                "actions_passed": [row.get("action_id") for row in task_rows if row.get("passed")],
                "error_types": dict(Counter(str(row.get("error_type")) for row in task_rows)),
            }
            for task_id, task_rows in sorted(by_case.items())
        },
    }


def candidate_equivalence_rows(rows: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    by_task = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    output: List[JsonDict] = []
    for task_id, task_rows in sorted(by_task.items()):
        clusters = defaultdict(list)
        for row in task_rows:
            clusters[str(row.get("generated_text_sha256"))].append(str(row.get("action_id")))
        for digest, actions in sorted(clusters.items()):
            output.append(
                {
                    "task_id": task_id,
                    "generated_text_sha256": digest,
                    "action_count": len(actions),
                    "actions": ";".join(sorted(actions)),
                }
            )
    return output


def per_case_best_action_rows(cases: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    by_task = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    output: List[JsonDict] = []
    for case in cases:
        task_rows = by_task[str(case["task_id"])]
        passed = [row for row in task_rows if row.get("passed")]
        compile_ok = [row for row in task_rows if row.get("compile_passed")]
        best = (passed or compile_ok or task_rows)[0]
        output.append(
            {
                "task_id": case["task_id"],
                "pool": case["pool"],
                "best_action": best.get("action_id"),
                "best_passed": best.get("passed"),
                "best_compile_passed": best.get("compile_passed"),
                "best_error_type": best.get("error_type"),
                "best_hash": best.get("generated_text_sha256"),
            }
        )
    return output


def error_transition_rows(cases: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    case_by_id = {str(case["task_id"]): case for case in cases}
    counter: Counter[Tuple[str, str, str, str]] = Counter()
    for row in rows:
        case = case_by_id[str(row["task_id"])]
        counter[
            (
                str(case["pool"]),
                str(case.get("historical_error_type")),
                str(row.get("action_id")),
                str(row.get("error_type")),
            )
        ] += 1
    return [
        {
            "pool": pool,
            "historical_error_type": before,
            "action_id": action_id,
            "new_error_type": after,
            "count": count,
        }
        for (pool, before, action_id, after), count in sorted(counter.items())
    ]


def choose_final_verdict(
    case_summaries: Mapping[str, Mapping[str, Any]],
    seed0_rows: Sequence[Mapping[str, Any]],
    multiseed_rows: Sequence[Mapping[str, Any]],
) -> str:
    hard_seed0_correct = [
        row for row in seed0_rows if row.get("case_pool") != "positive_control_rescued" and row.get("passed")
    ]
    hard_multiseed_correct = [
        row for row in multiseed_rows if row.get("case_pool") != "positive_control_rescued" and row.get("passed")
    ]
    if not hard_seed0_correct and hard_multiseed_correct:
        return "multi_seed_signal"
    if hard_seed0_correct:
        mechanisms = {row.get("action_id") for row in hard_seed0_correct}
        if len(mechanisms) > 1 or hard_multiseed_correct:
            return "mixed_generation_signal"
        if "G_oracle_sufficient_trace_span_remask" in mechanisms:
            return "trace_span_remask_signal"
        if "F_oracle_sufficient_trace_remask" in mechanisms:
            return "trace_token_remask_signal"
        return "alternate_trajectory_signal"
    hard_cases = [summary for summary in case_summaries.values() if summary.get("pool") != "positive_control_rescued"]
    diversity = [summary for summary in hard_cases if int(summary.get("unique_candidate_hash_count") or 0) > 1]
    if diversity:
        return "diversity_without_correctness_at_scale"
    unchanged = [summary for summary in hard_cases if summary.get("all_actions_unchanged")]
    if len(unchanged) == len(hard_cases):
        return "strong_current_action_family_ceiling"
    return "no_generation_ceiling_signal"


def render_report(seed0_summary: Mapping[str, Any], multiseed_summary: Mapping[str, Any], final_verdict: str) -> str:
    lines = [
        "# Long-Failure Generation Screen",
        "",
        f"final_verdict: `{final_verdict}`",
        "",
        "## Coverage",
        "",
        f"- case_count: `{seed0_summary.get('case_count')}`",
        f"- hard_case_count: `{seed0_summary.get('hard_case_count')}`",
        f"- positive_control_count: `{seed0_summary.get('positive_control_count')}`",
        f"- seed0_action_count: `{seed0_summary.get('seed0_action_count')}`",
        "",
        "## Pool Summary",
        "",
        "| Pool | Cases | New Correct | Unique Candidate Cases | Compile Repairs | Unchanged | Mean Sec | P95 Sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pool, info in (seed0_summary.get("pool_summary") or {}).items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                pool,
                info.get("case_count"),
                info.get("new_correct_candidate_cases"),
                info.get("unique_candidate_case_count"),
                info.get("compile_repair_cases"),
                info.get("unchanged_by_all_actions"),
                info.get("mean_cost_sec"),
                info.get("p95_cost_sec"),
            )
        )
    lines.extend(["", "## Action Summary", "", "| Action | Rows | Hard Correct | Compile Success | Changed vs C | Mean Sec | P95 Sec |", "|---|---:|---:|---:|---:|---:|---:|"])
    for action_id, info in (seed0_summary.get("action_summary") or {}).items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                action_id,
                info.get("action_count"),
                info.get("hard_passed_cases"),
                info.get("compile_success"),
                info.get("output_changed_vs_C"),
                info.get("mean_cost_sec"),
                info.get("p95_cost_sec"),
            )
        )
    lines.extend(
        [
            "",
            "## Conditional Multi-Seed",
            "",
            "This section is conditional diagnostic only; it is not an unbiased benchmark Pass@3.",
            "",
            f"- selected_case_count: `{multiseed_summary.get('selected_case_count')}`",
            f"- multiseed_action_count: `{multiseed_summary.get('multiseed_action_count')}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_run_manifest(args: argparse.Namespace, output_dir: Path, status: str, started_at: str, **extra: Any) -> JsonDict:
    command = shlex.join([sys.executable, *sys.argv])
    env_prefix = {
        key: os.environ[key]
        for key in ("CUDA_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        if key in os.environ
    }
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env_prefix.items()))
    return {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": command,
        "repro_command": f"{env_text} {command}".strip(),
        "env_command_prefix": env_prefix,
        "timestamp": args.timestamp,
        "started_at": started_at,
        "execution_status": status,
        "output_dir": str(output_dir),
        "environment": environment_info(args.model_path),
        "dataset": {"subset": args.dataset_subset, "split": args.split},
        "route2_error_analysis_dir": args.route2_error_analysis_dir,
        "action_budget": {"max": MAX_ACTION_BUDGET, **extra.pop("action_budget", {})},
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
        **extra,
    }


def execute(args: argparse.Namespace) -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    output_dir = make_output_dir(args.output_dir, args.timestamp)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(build_run_manifest(args, output_dir, "running", started_at), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        cases, mismatch = build_95_case_manifest(args)
        write_csv(output_dir / "case_manifest.csv", cases)
        write_csv(output_dir / "action_manifest.csv", build_action_manifest(cases))
        if mismatch:
            (output_dir / "mismatch_report.md").write_text(f"# Manifest Mismatch\n\n{mismatch}\n", encoding="utf-8")
            raise RuntimeError(mismatch)

        cfg = ExperimentConfig()
        cfg.model.model_path = args.model_path
        cfg.data.split = args.split
        cfg.data.dataset_subset = args.dataset_subset
        cfg.decode.lcas_policy = args.lcas_policy
        tokenizer, model = load_model_and_tokenizer(cfg.model)
        tasks = load_humaneval_infilling(split=args.split, dataset_subset=args.dataset_subset)
        by_task = {task.task_id: task for task in tasks}
        missing = [case["task_id"] for case in cases if case["task_id"] not in by_task]
        if missing:
            raise ValueError(f"manifest tasks missing from dataset: {missing[:10]}")
        source_commit = current_commit()
        case_by_id = {str(case["task_id"]): case for case in cases}

        sanity_rows, sanity = g_action_sanity(
            case_by_id=case_by_id,
            by_task=by_task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            source_commit=source_commit,
        )
        write_jsonl(output_dir / "g_action_distinctness_sanity.jsonl", sanity_rows)
        (output_dir / "g_action_distinctness_sanity.json").write_text(
            json.dumps(sanity, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not sanity.get("passed"):
            raise RuntimeError("G action mechanism did not execute distinctly enough for screen")

        seed0_rows = run_actions_for_cases(
            cases=cases,
            actions=SEED0_ACTIONS,
            seeds=(0,),
            by_task=by_task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            source_commit=source_commit,
        )
        seed0_expected = len(cases) * len(SEED0_ACTIONS)
        if len(seed0_rows) != seed0_expected:
            raise RuntimeError(f"seed0 action count mismatch: expected {seed0_expected}, got {len(seed0_rows)}")
        seed0_summary = summarize_seed0(cases, seed0_rows)

        selected_multiseed = choose_multiseed_cases(seed0_summary)
        selected_case_ids = {str(row["task_id"]) for row in selected_multiseed}
        selected_cases = [case for case in cases if str(case["task_id"]) in selected_case_ids]
        write_csv(output_dir / "multiseed_case_manifest.csv", selected_multiseed)
        multiseed_rows = run_actions_for_cases(
            cases=selected_cases,
            actions=SEED0_ACTIONS,
            seeds=MULTISEED_SEEDS,
            by_task=by_task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            source_commit=source_commit,
        )
        multiseed_summary = summarize_multiseed(multiseed_rows, selected_multiseed)
        final_verdict = choose_final_verdict(seed0_summary["case_summary"], seed0_rows, multiseed_rows)
        seed0_summary["final_verdict"] = final_verdict
        total_actions = len(seed0_rows) + len(multiseed_rows) + len(sanity_rows)
        if total_actions > MAX_ACTION_BUDGET:
            raise RuntimeError(f"action budget exceeded: {total_actions} > {MAX_ACTION_BUDGET}")

        write_jsonl(output_dir / "seed0_results.jsonl", seed0_rows)
        write_csv(output_dir / "seed0_results.csv", [compact_result_record(row) for row in seed0_rows])
        (output_dir / "seed0_summary.json").write_text(
            json.dumps(seed0_summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_csv(output_dir / "per_case_best_action.csv", per_case_best_action_rows(cases, seed0_rows))
        write_csv(output_dir / "candidate_equivalence.csv", candidate_equivalence_rows(seed0_rows))
        write_csv(output_dir / "error_transition_table.csv", error_transition_rows(cases, seed0_rows))
        write_jsonl(output_dir / "multiseed_results.jsonl", multiseed_rows)
        write_csv(output_dir / "multiseed_results.csv", [compact_result_record(row) for row in multiseed_rows])
        (output_dir / "multiseed_summary.json").write_text(
            json.dumps(multiseed_summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "report.md").write_text(render_report(seed0_summary, multiseed_summary, final_verdict), encoding="utf-8")
        manifest = build_run_manifest(
            args,
            output_dir,
            "completed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            action_budget={
                "g_sanity_actions": len(sanity_rows),
                "seed0_actions": len(seed0_rows),
                "multiseed_actions": len(multiseed_rows),
                "total_actions": total_actions,
            },
            coverage={
                "hard_cases_expected": EXPECTED_TRIGGERED_FAILED_LONG + EXPECTED_MISSED_FAILED_LONG,
                "hard_cases_executed": EXPECTED_TRIGGERED_FAILED_LONG + EXPECTED_MISSED_FAILED_LONG,
                "positive_controls_executed": EXPECTED_POSITIVE_CONTROLS,
            },
            final_verdict=final_verdict,
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "final_verdict": final_verdict}, ensure_ascii=False, indent=2))
    except BaseException:
        manifest = build_run_manifest(
            args,
            output_dir,
            "failed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            failure_traceback=traceback.format_exc(),
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 95-case long-failure generation ceiling screen.")
    parser.add_argument("--route2-error-analysis-dir", default="analysis_outputs/route2_error_analysis_20260617_165806")
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE)
    parser.add_argument("--route2-results", default=DEFAULT_ROUTE2)
    parser.add_argument("--output-dir", default="analysis_outputs")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--max-canvas-length", type=int, default=64)
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-subset", default="HumanEval-SingleLineInfilling")
    parser.add_argument("--lcas-policy", default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    return parser.parse_args()


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
