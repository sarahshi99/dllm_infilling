#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.action_ceiling.action_ceiling_matrix import (
    DEFAULT_BASELINE,
    DEFAULT_ROUTE2,
    canvas_is_oracle_sufficient,
    compile_passed,
    current_branch,
    current_commit,
    environment_info,
    git_capture,
    oracle_sufficient_length,
    parse_task_id_group,
    reset_action_seed,
    rows_by_task,
    sha256_text,
    stable_task_seed,
    verification_error_message,
    verification_error_type,
    write_csv,
    write_jsonl,
)
from analysis.trace_long_rescue_features import load_jsonl, metric, oracle_bucket
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask, compute_oracle_mask_length, load_humaneval_infilling
from expvision_dllm_clean.decode import (
    _segment_decode,
    build_reconstruction_diagnostics,
    linear_target_masks,
    prepare_model_inputs,
    select_low_confidence_mask_positions,
)
from expvision_dllm_clean.modeling import load_model_and_tokenizer
from expvision_dllm_clean.runner_lcas_v3 import choose_lcas_v3_config
from expvision_dllm_clean.stopping import evaluate_global_gap_stopping, validate_stopping_config
from expvision_dllm_clean.verifier import run_verifier_stack


JsonDict = Dict[str, Any]

TASK_IDS = (
    "SingleLineInfilling/HumanEval/116/L0",
    "SingleLineInfilling/HumanEval/85/L0",
    "SingleLineInfilling/HumanEval/113/L3",
)
SMOKE_TASK_ID = "SingleLineInfilling/HumanEval/85/L0"
EXPERIMENTAL_SEEDS = (0, 1, 2)
SANITY_ACTIONS = ("A_primary", "B_route2_len32")
DIAGNOSTIC_ACTIONS = (
    "C_oracle_sufficient",
    "E_oracle_sufficient_no_early_commit",
    "F_oracle_sufficient_trace_remask",
)
ALL_ACTIONS = SANITY_ACTIONS + DIAGNOSTIC_ACTIONS
VERDICTS = {
    "invalid_action_not_distinct",
    "no_candidate_diversity",
    "candidate_diversity_without_correctness",
    "no_early_commit_signal",
    "trace_remask_signal",
    "multi_seed_candidate_signal",
    "mixed_distinct_candidate_signal",
    "positive_control_only",
}


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(values: Sequence[bool]) -> Optional[float]:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def make_output_dir(base_dir: str, timestamp: Optional[str]) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"distinct_candidate_ceiling_{stamp}"
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def parse_task_ids_csv(value: Optional[str]) -> List[str]:
    if not value:
        return list(TASK_IDS)
    return [item.strip() for item in value.split(",") if item.strip()]


def build_case_manifest(
    baseline_rows: Sequence[Mapping[str, Any]],
    route2_rows: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[str],
    max_canvas_length: int,
) -> List[JsonDict]:
    baseline = rows_by_task(baseline_rows, name="baseline")
    route2 = rows_by_task(route2_rows, name="route2")
    rows: List[JsonDict] = []
    for task_id in task_ids:
        if task_id not in baseline or task_id not in route2:
            raise ValueError(f"requested task id missing from historical rows: {task_id}")
        baseline_row = baseline[task_id]
        route2_row = route2[task_id]
        oracle_len = _to_int(metric(baseline_row, "oracle_mask_length", metric(route2_row, "oracle_mask_length")))
        primary_len = _to_int(metric(baseline_row, "selected_mask_length"))
        route2_len = _to_int(metric(route2_row, "route2_rescue_length", metric(route2_row, "selected_mask_length")))
        sufficient_len = oracle_sufficient_length(
            primary_len=primary_len,
            route2_len=route2_len,
            oracle_len=oracle_len,
            max_length=max_canvas_length,
        )
        if sufficient_len is None:
            raise ValueError(f"oracle-sufficient length unavailable within max canvas for {task_id}")
        baseline_passed = bool(metric(baseline_row, "passed", False))
        route2_passed = bool(metric(route2_row, "passed", False))
        route2_triggered = bool(metric(route2_row, "route2_trace_rescue_triggered", False))
        if (not baseline_passed) and route2_passed:
            pool = "positive_control_rescued"
        elif oracle_len is not None and oracle_len >= 17 and (not baseline_passed) and route2_triggered:
            pool = "triggered_failed_long"
        elif oracle_len is not None and oracle_len >= 17 and (not baseline_passed) and (not route2_triggered):
            pool = "missed_failed_long"
        else:
            pool = "manual_requested"
        rows.append(
            {
                "task_id": task_id,
                "task_group": parse_task_id_group(task_id),
                "case_pool": pool,
                "oracle_length": oracle_len,
                "oracle_bucket": oracle_bucket(oracle_len),
                "primary_selected_length": primary_len,
                "route2_triggered": route2_triggered,
                "route2_rescue_length": route2_len,
                "oracle_sufficient_length": sufficient_len,
                "max_canvas_length": max_canvas_length,
                "canvas_is_oracle_sufficient": canvas_is_oracle_sufficient(sufficient_len, oracle_len),
                "baseline_passed": baseline_passed,
                "route2_passed": route2_passed,
            }
        )
    return rows


def build_action_manifest(case_manifest: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    specs = {
        "A_primary": ("deployable_sanity_replay", "current primary", 64, [0]),
        "B_route2_len32": ("deployable_sanity_replay", "current Route2 when historically triggered", 64, [0]),
        "C_oracle_sufficient": ("offline_ceiling_oracle_canvas", "oracle-sufficient canvas with early commit", 64, list(EXPERIMENTAL_SEEDS)),
        "E_oracle_sufficient_no_early_commit": (
            "offline_ceiling_oracle_canvas",
            "oracle-sufficient canvas with early commit disabled",
            64,
            list(EXPERIMENTAL_SEEDS),
        ),
        "F_oracle_sufficient_trace_remask": (
            "offline_ceiling_trace_remask",
            "oracle-sufficient C candidate plus fixed internal-trace remask refinement",
            80,
            list(EXPERIMENTAL_SEEDS),
        ),
    }
    rows: List[JsonDict] = []
    for case in case_manifest:
        for action_id in ALL_ACTIONS:
            deployability, description, steps, seeds = specs[action_id]
            if action_id == "A_primary":
                planned_canvas = case.get("primary_selected_length")
            elif action_id == "B_route2_len32":
                planned_canvas = case.get("route2_rescue_length") or case.get("primary_selected_length")
            else:
                planned_canvas = case.get("oracle_sufficient_length")
            rows.append(
                {
                    "task_id": case["task_id"],
                    "case_pool": case["case_pool"],
                    "action_id": action_id,
                    "deployability": deployability,
                    "planned_canvas_length": planned_canvas,
                    "planned_total_steps": steps,
                    "experimental_seeds": ",".join(str(seed) for seed in seeds),
                    "description": description,
                }
            )
    return rows


def compact_step_trace(trace: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    return [
        {
            "step": row.get("step"),
            "remaining_masks_before": row.get("remaining_masks_before"),
            "remaining_masks_after": row.get("remaining_masks_after"),
            "target_masks": row.get("target_masks"),
            "token_change_count": row.get("token_change_count"),
            "low_confidence_count": len(row.get("low_confidence_indices") or []),
            "stop_reason": row.get("stop_reason"),
            "should_stop": row.get("should_stop"),
        }
        for row in trace
    ]


def decode_fixed_canvas(
    *,
    task: CodeTask,
    tokenizer: Any,
    model: Any,
    cfg: ExperimentConfig,
    canvas_len: int,
    total_steps: int,
    early_commit_enabled: bool,
    initial_middle_ids: Optional[Sequence[int]] = None,
    initial_mask_indices: Optional[Sequence[int]] = None,
    schedule_length: Optional[int] = None,
    phase_name: str,
) -> JsonDict:
    length_meta = {
        "candidate_scores": None,
        "length_probe_sec": 0.0,
        "selected_mask_length": int(canvas_len),
        "selected_score": None,
        "oracle_mask_length": compute_oracle_mask_length(task, tokenizer, add_special_tokens=False),
        "mask_length": int(canvas_len),
        "mask_length_source": "distinct_candidate_fixed_oracle_canvas",
        "probe_lengths": [int(canvas_len)],
        "tie_break": "fixed",
    }
    prepared = prepare_model_inputs(task, tokenizer, int(canvas_len), cfg)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    middle_start = int(prepared["middle_start"])
    middle_end = int(prepared["middle_end"])
    mask_token_id = int(prepared["mask_token_id"])
    if initial_middle_ids is not None:
        if len(initial_middle_ids) != int(canvas_len):
            raise ValueError("initial_middle_ids length must match canvas_len")
        x_t[0, middle_start:middle_end] = torch.tensor(list(initial_middle_ids), dtype=torch.long, device=device)
        for idx in initial_mask_indices or []:
            x_t[0, middle_start + int(idx)] = mask_token_id

    selected_for_schedule = int(schedule_length or canvas_len)
    stop_cfg = choose_lcas_v3_config(int(canvas_len), str(getattr(cfg.decode, "lcas_policy", "lcas_v3b")))
    if early_commit_enabled and stop_cfg is not None:
        validate_stopping_config(stop_cfg, total_steps=total_steps)

    step_trace: List[JsonDict] = []
    stopped = False
    stop_step: Optional[int] = None
    stop_reason = "not_stopped"
    stop_decision_at_stop: Optional[JsonDict] = None
    no_remaining_masks_first_step: Optional[int] = None
    top1_history: List[List[int]] = []
    top1_prob_history: List[List[float]] = []
    total_token_changes = 0
    effective_update_steps = 0
    decode_start = time.perf_counter()

    for step in range(int(total_steps)):
        before_middle = x_t[0, middle_start:middle_end].detach().clone()
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)
        current_mask_idx = x_t == mask_token_id
        middle_probs = probs[:, middle_start:middle_end, :]
        middle_max_probs = max_probs[:, middle_start:middle_end]
        middle_mask_idx = current_mask_idx[:, middle_start:middle_end]
        remaining_before = int(middle_mask_idx.sum().item())
        top1_history.append([int(value) for value in preds[0, middle_start:middle_end].detach().cpu().tolist()])
        top1_prob_history.append([float(value) for value in middle_max_probs[0].detach().cpu().tolist()])

        if remaining_before == 0 and no_remaining_masks_first_step is None:
            no_remaining_masks_first_step = step

        if early_commit_enabled and stop_cfg is not None:
            decision = evaluate_global_gap_stopping(
                step=step,
                middle_probs=middle_probs,
                middle_mask_idx=middle_mask_idx,
                selected_mask_length=int(canvas_len),
                stop_cfg=stop_cfg,
            )
            decision_dict = decision.to_dict()
        else:
            ratio = remaining_before / float(max(1, selected_for_schedule))
            decision_dict = {
                "should_stop": False,
                "reason": "early_commit_disabled",
                "step": step,
                "remaining_masks": remaining_before,
                "remaining_mask_ratio": ratio,
                "mean_gap": None,
                "min_gap": None,
                "mean_top1": None,
                "max_top1": None,
                "min_top1": None,
            }

        should_stop = bool(decision_dict.get("should_stop", False))
        low_conf_indices: List[int] = []
        if should_stop:
            x_t[current_mask_idx] = preds[current_mask_idx]
            stopped = True
            stop_step = step
            stop_reason = str(decision_dict.get("reason"))
            stop_decision_at_stop = dict(decision_dict)
        else:
            if early_commit_enabled and decision_dict.get("reason") == "no_remaining_masks":
                stop_reason = "no_remaining_masks"
                stop_decision_at_stop = dict(decision_dict)
                after_middle = x_t[0, middle_start:middle_end].detach().clone()
                token_change_count = int((after_middle != before_middle).sum().item())
                step_trace.append(
                    {
                        "phase": phase_name,
                        "step": step,
                        "target_masks": 0,
                        "remaining_masks_before": remaining_before,
                        "remaining_masks_after": remaining_before,
                        "token_change_count": token_change_count,
                        "low_confidence_indices": [],
                        "stop_reason": stop_reason,
                        "should_stop": False,
                    }
                )
                break
            target_masks = linear_target_masks(selected_for_schedule, int(total_steps), step)
            if target_masks > 0:
                low_conf_indices = select_low_confidence_mask_positions(
                    middle_max_probs,
                    middle_mask_idx,
                    target_masks,
                )
                x_t[current_mask_idx] = preds[current_mask_idx]
                if low_conf_indices:
                    x_t[0, [middle_start + idx for idx in low_conf_indices]] = mask_token_id
            else:
                x_t[current_mask_idx] = preds[current_mask_idx]

        after_middle = x_t[0, middle_start:middle_end].detach().clone()
        token_change_count = int((after_middle != before_middle).sum().item())
        remaining_after = int((after_middle == mask_token_id).sum().item())
        if token_change_count > 0:
            effective_update_steps += 1
            total_token_changes += token_change_count
        step_trace.append(
            {
                "phase": phase_name,
                "step": step,
                "target_masks": 0 if should_stop else linear_target_masks(selected_for_schedule, int(total_steps), step),
                "remaining_masks_before": remaining_before,
                "remaining_masks_after": remaining_after,
                "token_change_count": token_change_count,
                "low_confidence_indices": low_conf_indices,
                "stop_reason": str(decision_dict.get("reason")),
                "should_stop": should_stop,
            }
        )
        if should_stop:
            break

    total_decode_sec = time.perf_counter() - decode_start
    final_middle_ids = [int(value) for value in x_t[0, middle_start:middle_end].detach().cpu().tolist()]
    final_segments = _segment_decode(
        tokenizer=tokenizer,
        prefix_ids=prepared["prefix_ids"],
        middle_ids=final_middle_ids,
        suffix_ids=prepared["suffix_ids"],
    )
    final_verification = run_verifier_stack(
        task=task,
        full_code=final_segments["full_text"],
        completion_without_suffix=final_segments["middle_text"],
    )
    verification_sec = sum(item.duration_sec for item in final_verification.values())
    tier3 = final_verification.get("tier3_unit_tests")
    passed = bool(tier3.passed) if tier3 else False
    actual_forward_steps = len(step_trace)
    effective_steps = (stop_step + 1) if stopped and stop_step is not None else actual_forward_steps
    return {
        "task_id": task.task_id,
        "code": final_segments["full_text"],
        "middle_text": final_segments["middle_text"],
        "final_middle_ids": final_middle_ids,
        "metrics": {
            "passed": passed,
            "decode_sec": total_decode_sec,
            "verification_sec": verification_sec,
            "total_sec": total_decode_sec + verification_sec,
            "total_sec_including_probe": total_decode_sec + verification_sec + float(length_meta["length_probe_sec"]),
            "length_probe_sec": float(length_meta["length_probe_sec"]),
            "total_steps": int(total_steps),
            "actual_forward_steps": actual_forward_steps,
            "effective_steps": effective_steps,
            "effective_update_steps": effective_update_steps,
            "total_token_changes": total_token_changes,
            "stopped": stopped,
            "stop_step": stop_step,
            "stop_reason": stop_reason,
            "no_remaining_masks_first_step": no_remaining_masks_first_step,
            "mask_length": int(canvas_len),
            "oracle_mask_length": length_meta["oracle_mask_length"],
            "selected_mask_length": int(canvas_len),
            "mask_length_source": length_meta["mask_length_source"],
            "early_commit_enabled": bool(early_commit_enabled),
            "full_schedule_completed": actual_forward_steps == int(total_steps),
        },
        "verification": {key: value.to_dict() for key, value in final_verification.items()},
        "diagnostics": build_reconstruction_diagnostics(task, final_segments),
        "trajectory": {
            "phase": phase_name,
            "step_trace": compact_step_trace(step_trace),
            "top1_history": top1_history,
            "top1_prob_history": top1_prob_history,
            "stop_decision": stop_decision_at_stop,
        },
    }


def remask_count_for_canvas(canvas_len: int) -> int:
    return max(1, min(4, int(math.ceil(0.10 * int(canvas_len)))))


def select_trace_remask_positions(stage1: Mapping[str, Any], *, canvas_len: int) -> List[JsonDict]:
    trajectory = stage1.get("trajectory") or {}
    top1_history = trajectory.get("top1_history") or []
    top1_prob_history = trajectory.get("top1_prob_history") or []
    final_probs = top1_prob_history[-1] if top1_prob_history else [0.0] * int(canvas_len)
    rows: List[JsonDict] = []
    for idx in range(int(canvas_len)):
        seq = [step[idx] for step in top1_history if idx < len(step)]
        flips = sum(1 for left, right in zip(seq, seq[1:]) if left != right)
        final_conf = float(final_probs[idx]) if idx < len(final_probs) else 0.0
        rows.append(
            {
                "index": idx,
                "token_flip_count": flips,
                "final_confidence": final_conf,
                "score_tuple": [int(flips), -final_conf, -idx],
            }
        )
    rows.sort(key=lambda row: (-int(row["token_flip_count"]), float(row["final_confidence"]), int(row["index"])))
    return rows[: remask_count_for_canvas(int(canvas_len))]


def run_trace_remask(
    *,
    task: CodeTask,
    tokenizer: Any,
    model: Any,
    cfg: ExperimentConfig,
    canvas_len: int,
    total_steps: int,
    refinement_steps: int,
) -> JsonDict:
    stage1 = decode_fixed_canvas(
        task=task,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        canvas_len=canvas_len,
        total_steps=total_steps,
        early_commit_enabled=True,
        phase_name="stage1_c_oracle_sufficient",
    )
    selected = select_trace_remask_positions(stage1, canvas_len=canvas_len)
    remask_indices = [int(row["index"]) for row in selected]
    refine = decode_fixed_canvas(
        task=task,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        canvas_len=canvas_len,
        total_steps=refinement_steps,
        early_commit_enabled=False,
        initial_middle_ids=stage1["final_middle_ids"],
        initial_mask_indices=remask_indices,
        schedule_length=len(remask_indices),
        phase_name="stage2_trace_remask_refinement",
    )
    combined = copy.deepcopy(refine)
    stage1_metrics = stage1.get("metrics") or {}
    refine_metrics = refine.get("metrics") or {}
    combined["metrics"] = {
        **refine_metrics,
        "total_steps": int(total_steps) + int(refinement_steps),
        "stage1_steps": int(total_steps),
        "refinement_steps": int(refinement_steps),
        "actual_forward_steps": int(stage1_metrics.get("actual_forward_steps") or 0)
        + int(refine_metrics.get("actual_forward_steps") or 0),
        "total_sec_including_probe": float(stage1_metrics.get("total_sec_including_probe") or 0.0)
        + float(refine_metrics.get("total_sec_including_probe") or 0.0),
        "decode_sec": float(stage1_metrics.get("decode_sec") or 0.0) + float(refine_metrics.get("decode_sec") or 0.0),
        "verification_sec": float(stage1_metrics.get("verification_sec") or 0.0)
        + float(refine_metrics.get("verification_sec") or 0.0),
        "remasked_token_count": len(remask_indices),
        "refinement_executed": True,
        "stage1_hash": sha256_text(stage1.get("middle_text")),
    }
    combined["trajectory"] = {
        "stage1_summary": {
            key: stage1_metrics.get(key)
            for key in [
                "actual_forward_steps",
                "effective_update_steps",
                "total_token_changes",
                "stopped",
                "stop_reason",
                "no_remaining_masks_first_step",
            ]
        },
        "stage2_summary": {
            key: refine_metrics.get(key)
            for key in [
                "actual_forward_steps",
                "effective_update_steps",
                "total_token_changes",
                "stopped",
                "stop_reason",
                "no_remaining_masks_first_step",
            ]
        },
        "stage1_step_trace": (stage1.get("trajectory") or {}).get("step_trace"),
        "stage2_step_trace": (refine.get("trajectory") or {}).get("step_trace"),
        "remask_rule": "rank token_flip_count desc, final_confidence asc, index asc; k=max(1,min(4,ceil(0.10*canvas_len)))",
        "remasked_token_indices": remask_indices,
        "remask_scores": selected,
        "remask_rule_uses_test_result": False,
    }
    return combined


def result_error_type(result: Mapping[str, Any]) -> Optional[str]:
    return verification_error_type(result)


def result_record(
    *,
    case: Mapping[str, Any],
    action_id: str,
    result: Mapping[str, Any],
    experimental_seed: int,
    task_seed: int,
    source_commit: str,
    action_was_executed: bool,
    execution_note: str,
    c_reference: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    metrics = result.get("metrics") or {}
    generated_text = result.get("middle_text")
    error_message = verification_error_message(result)
    selected_len = _to_int(metrics.get("selected_mask_length", metrics.get("mask_length")))
    oracle_len = _to_int(metrics.get("oracle_mask_length", case.get("oracle_length")))
    c_hash = None if c_reference is None else sha256_text(c_reference.get("middle_text"))
    generated_hash = sha256_text(generated_text)
    return {
        "task_id": str(case["task_id"]),
        "task_group": case.get("task_group"),
        "case_pool": case.get("case_pool"),
        "action_id": action_id,
        "experimental_seed": int(experimental_seed),
        "seed": int(task_seed),
        "passed": bool(metrics.get("passed")),
        "generated_text": generated_text,
        "generated_text_sha256": generated_hash,
        "generated_text_len": None if generated_text is None else len(generated_text),
        "code_sha256": sha256_text(result.get("code")),
        "selected_mask_length": selected_len,
        "actual_canvas_length": _to_int(metrics.get("mask_length", selected_len)),
        "oracle_mask_length": oracle_len,
        "canvas_is_oracle_sufficient": canvas_is_oracle_sufficient(selected_len, oracle_len),
        "decode_steps": _to_int(metrics.get("total_steps")),
        "actual_forward_steps": _to_int(metrics.get("actual_forward_steps")),
        "effective_steps": _to_int(metrics.get("effective_steps")),
        "effective_update_steps": _to_int(metrics.get("effective_update_steps")),
        "total_token_changes": _to_int(metrics.get("total_token_changes")),
        "early_commit_enabled": metrics.get("early_commit_enabled"),
        "early_commit_triggered": metrics.get("stop_reason") == "global_gap_early_commit",
        "full_schedule_completed": metrics.get("full_schedule_completed"),
        "stop_reason": metrics.get("stop_reason"),
        "no_remaining_masks_first_step": metrics.get("no_remaining_masks_first_step"),
        "remasked_token_count": metrics.get("remasked_token_count"),
        "refinement_steps": metrics.get("refinement_steps"),
        "refinement_executed": metrics.get("refinement_executed"),
        "stage1_hash": metrics.get("stage1_hash"),
        "output_changed_vs_C": None if c_hash is None else generated_hash != c_hash,
        "total_sec_including_probe": metrics.get("total_sec_including_probe"),
        "compile_passed": compile_passed(result),
        "error_type": result_error_type(result),
        "error_message_short": None if error_message is None else str(error_message).replace("\n", " ")[:240],
        "action_was_executed": bool(action_was_executed),
        "execution_note": execution_note,
        "source_commit": source_commit,
        "trajectory": result.get("trajectory"),
        "verification": result.get("verification"),
        "diagnostics": result.get("diagnostics"),
    }


def compact_result_record(row: Mapping[str, Any]) -> JsonDict:
    excluded = {"generated_text", "trajectory", "verification", "diagnostics"}
    return {key: value for key, value in row.items() if key not in excluded}


def pass_at_k(rows: Sequence[Mapping[str, Any]], k: int) -> bool:
    return any(bool(row.get("passed")) for row in rows[:k])


def cluster_hashes(rows: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    clusters: Dict[str, List[JsonDict]] = defaultdict(list)
    for row in rows:
        clusters[str(row.get("generated_text_sha256"))].append(
            {
                "task_id": row.get("task_id"),
                "action_id": row.get("action_id"),
                "experimental_seed": row.get("experimental_seed"),
                "passed": row.get("passed"),
            }
        )
    return [
        {"generated_text_sha256": digest, "count": len(items), "members": items}
        for digest, items in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


def summarize_diversity(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_case: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_action: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("action_id") in DIAGNOSTIC_ACTIONS:
            by_case[str(row["task_id"])].append(row)
            by_action[str(row["action_id"])].append(row)
    case_summary: Dict[str, JsonDict] = {}
    for task_id, task_rows in sorted(by_case.items()):
        hashes = {row.get("generated_text_sha256") for row in task_rows if row.get("generated_text_sha256")}
        compile_values = [bool(row.get("compile_passed")) for row in task_rows if row.get("compile_passed") is not None]
        pass_at_1_by_seed = {
            str(seed): {
                str(row.get("action_id")): bool(row.get("passed"))
                for row in task_rows
                if int(row.get("experimental_seed") or 0) == seed
            }
            for seed in sorted({int(row.get("experimental_seed") or 0) for row in task_rows})
        }
        pass_at_3_by_seed = {
            str(seed): any(
                bool(row.get("passed"))
                for row in task_rows
                if int(row.get("experimental_seed") or 0) == seed
            )
            for seed in sorted({int(row.get("experimental_seed") or 0) for row in task_rows})
        }
        case_summary[task_id] = {
            "unique_candidate_hash_count": len(hashes),
            "pass_at_1_by_seed": pass_at_1_by_seed,
            "candidate_existence_pass_at_3_by_seed": pass_at_3_by_seed,
            "candidate_existence_pass_at_3_any_seed": any(pass_at_3_by_seed.values()),
            "candidate_existence_pass_any": any(bool(row.get("passed")) for row in task_rows),
            "compile_rate": _rate(compile_values),
            "error_type_distribution": dict(Counter(str(row.get("error_type")) for row in task_rows)),
            "candidate_length_values": [row.get("generated_text_len") for row in task_rows],
            "hash_clusters": cluster_hashes(task_rows),
        }
    action_summary: Dict[str, JsonDict] = {}
    for action_id, action_rows in sorted(by_action.items()):
        hashes = {row.get("generated_text_sha256") for row in action_rows if row.get("generated_text_sha256")}
        compile_values = [bool(row.get("compile_passed")) for row in action_rows if row.get("compile_passed") is not None]
        costs = [
            float(row["total_sec_including_probe"])
            for row in action_rows
            if row.get("total_sec_including_probe") is not None
        ]
        action_summary[action_id] = {
            "unique_candidate_hash_count": len(hashes),
            "pass_count": sum(1 for row in action_rows if row.get("passed")),
            "compile_rate": _rate(compile_values),
            "error_type_distribution": dict(Counter(str(row.get("error_type")) for row in action_rows)),
            "avg_total_sec_including_probe": None if not costs else sum(costs) / len(costs),
        }
    return {
        "mode": "distinct_candidate_diversity_summary",
        "case_summary": case_summary,
        "action_summary": action_summary,
    }


def action_equivalence(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "mode": "distinct_candidate_action_equivalence",
        "clusters": cluster_hashes([row for row in rows if row.get("action_id") in DIAGNOSTIC_ACTIONS]),
    }


def smoke_gate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_action = {row["action_id"]: row for row in rows if row.get("task_id") == SMOKE_TASK_ID}
    c = by_action.get("C_oracle_sufficient")
    e = by_action.get("E_oracle_sufficient_no_early_commit")
    f = by_action.get("F_oracle_sufficient_trace_remask")
    c_hash = None if c is None else c.get("generated_text_sha256")
    e_hash = None if e is None else e.get("generated_text_sha256")
    f_hash = None if f is None else f.get("generated_text_sha256")
    e_disabled = bool(e and e.get("early_commit_enabled") is False)
    e_more_forwards = bool(e and c and int(e.get("actual_forward_steps") or 0) > int(c.get("actual_forward_steps") or 0))
    f_remasked = bool(f and int(f.get("remasked_token_count") or 0) > 0)
    f_refined = bool(f and f.get("refinement_executed") is True)
    hash_distinct = (e_hash is not None and e_hash != c_hash) or (f_hash is not None and f_hash != c_hash)
    auditable_distinct = e_more_forwards or (f_remasked and f_refined)
    passed = e_disabled and f_remasked and f_refined and (hash_distinct or auditable_distinct)
    return {
        "task_id": SMOKE_TASK_ID,
        "passed": passed,
        "c_hash": c_hash,
        "e_hash": e_hash,
        "f_hash": f_hash,
        "e_early_commit_disabled": e_disabled,
        "e_more_actual_forwards_than_c": e_more_forwards,
        "f_remasked_token_count": None if f is None else f.get("remasked_token_count"),
        "f_refinement_executed": None if f is None else f.get("refinement_executed"),
        "hash_distinct_from_c": hash_distinct,
        "auditable_distinct_trajectory": auditable_distinct,
    }


def choose_verdict(rows: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]) -> str:
    if not gate.get("passed"):
        return "invalid_action_not_distinct"
    hard_rows = [
        row
        for row in rows
        if row.get("case_pool") != "positive_control_rescued" and row.get("action_id") in DIAGNOSTIC_ACTIONS
    ]
    hard_correct = [row for row in hard_rows if row.get("passed")]
    if hard_correct:
        mechanisms = {row.get("action_id") for row in hard_correct}
        seeds = {int(row.get("experimental_seed") or 0) for row in hard_correct}
        if len(mechanisms) > 1:
            return "mixed_distinct_candidate_signal"
        if seeds - {0} and not any(int(row.get("experimental_seed") or 0) == 0 for row in hard_correct):
            return "multi_seed_candidate_signal"
        if "E_oracle_sufficient_no_early_commit" in mechanisms:
            return "no_early_commit_signal"
        if "F_oracle_sufficient_trace_remask" in mechanisms:
            return "trace_remask_signal"
        return "multi_seed_candidate_signal"
    hard_hashes_by_task: Dict[str, set[str]] = defaultdict(set)
    for row in hard_rows:
        if row.get("generated_text_sha256"):
            hard_hashes_by_task[str(row["task_id"])].add(str(row["generated_text_sha256"]))
    if any(len(values) > 1 for values in hard_hashes_by_task.values()):
        return "candidate_diversity_without_correctness"
    positive_correct = any(
        row.get("case_pool") == "positive_control_rescued"
        and row.get("action_id") in DIAGNOSTIC_ACTIONS
        and row.get("passed")
        for row in rows
    )
    if positive_correct:
        return "positive_control_only"
    return "no_candidate_diversity"


def render_report(summary: Mapping[str, Any], diversity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Distinct-Candidate Generation Ceiling Pilot",
        "",
        f"verdict: `{summary.get('verdict')}`",
        "",
        "## Action-Distinctness Gate",
        "",
        f"- gate: `{summary.get('action_distinctness_gate')}`",
        "",
        "## Per-Case Diagnostic Actions",
        "",
        "| Task | Action | Seed | Pass | Hash | Compile | Error | Changed vs C | Sec |",
        "|---|---|---:|---|---|---|---|---|---:|",
    ]
    for row in rows:
        if row.get("action_id") not in DIAGNOSTIC_ACTIONS:
            continue
        lines.append(
            "| `{}` | `{}` | {} | `{}` | `{}` | `{}` | `{}` | `{}` | {:.3f} |".format(
                row.get("task_id"),
                row.get("action_id"),
                row.get("experimental_seed"),
                row.get("passed"),
                str(row.get("generated_text_sha256"))[:12],
                row.get("compile_passed"),
                row.get("error_type"),
                row.get("output_changed_vs_C"),
                float(row.get("total_sec_including_probe") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Diversity",
            "",
            f"- case_summary: `{diversity.get('case_summary')}`",
            f"- action_summary: `{diversity.get('action_summary')}`",
            "",
            "## Required Questions",
            "",
            f"1. new actions distinct: `{summary.get('new_actions_distinct')}`",
            f"2. multi-seed diversity: `{summary.get('multi_seed_diversity')}`",
            f"3. non-positive-control correct candidate: `{summary.get('hard_case_candidate_exists')}`",
            f"4. 85/L0 SyntaxError outcome: `{summary.get('case_85_error_evolution')}`",
            f"5. 113/L3 UnitTestFailure outcome: `{summary.get('case_113_error_evolution')}`",
            f"6. changed mechanism: `{summary.get('changed_mechanisms')}`",
            f"7. change type: `{summary.get('change_type_note')}`",
            f"8. action costs: `{summary.get('action_costs')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest(args: argparse.Namespace, output_dir: Path, status: str, started_at: str, **extra: Any) -> JsonDict:
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
        "task_ids": parse_task_ids_csv(args.task_ids_csv),
        "experimental_seeds": list(EXPERIMENTAL_SEEDS),
        "action_definitions": list(ALL_ACTIONS),
        "historical_results": {"baseline": args.baseline_results, "route2": args.route2_results},
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
        **extra,
    }


def run_ab_sanity(
    *,
    case: Mapping[str, Any],
    task: CodeTask,
    tokenizer: Any,
    model: Any,
    cfg: Any,
    settings: Any,
    route2_runner: Any,
    source_commit: str,
) -> List[JsonDict]:
    from experiments.action_ceiling.action_ceiling_matrix import run_single_action

    rows: List[JsonDict] = []
    task_seed = stable_task_seed(42, str(case["task_id"]), 0)
    primary_result = None
    for action_id in SANITY_ACTIONS:
        result, action_was_executed, execution_note = run_single_action(
            action_id=action_id,
            case=case,
            task=task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            settings=settings,
            route2_runner=route2_runner,
            seed=task_seed,
            primary_result=primary_result,
        )
        if action_id == "A_primary":
            primary_result = result
        rows.append(
            result_record(
                case=case,
                action_id=action_id,
                result=result,
                experimental_seed=0,
                task_seed=task_seed,
                source_commit=source_commit,
                action_was_executed=action_was_executed,
                execution_note=execution_note,
            )
        )
    return rows


def run_diagnostic_actions(
    *,
    case: Mapping[str, Any],
    task: CodeTask,
    tokenizer: Any,
    model: Any,
    cfg: ExperimentConfig,
    experimental_seed: int,
    source_commit: str,
    actions: Sequence[str] = DIAGNOSTIC_ACTIONS,
) -> List[JsonDict]:
    task_seed = stable_task_seed(42, str(case["task_id"]), experimental_seed)
    canvas_len = int(case["oracle_sufficient_length"])
    rows: List[JsonDict] = []
    c_result: Optional[JsonDict] = None
    for action_id in actions:
        reset_action_seed(task_seed)
        if action_id == "C_oracle_sufficient":
            result = decode_fixed_canvas(
                task=task,
                tokenizer=tokenizer,
                model=model,
                cfg=cfg,
                canvas_len=canvas_len,
                total_steps=64,
                early_commit_enabled=True,
                phase_name="c_oracle_sufficient",
            )
            c_result = result
            note = "executed_oracle_sufficient_early_commit"
        elif action_id == "E_oracle_sufficient_no_early_commit":
            result = decode_fixed_canvas(
                task=task,
                tokenizer=tokenizer,
                model=model,
                cfg=cfg,
                canvas_len=canvas_len,
                total_steps=64,
                early_commit_enabled=False,
                phase_name="e_no_early_commit",
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
            note = "executed_trace_remask_refinement"
        else:
            raise ValueError(f"unknown diagnostic action: {action_id}")
        rows.append(
            result_record(
                case=case,
                action_id=action_id,
                result=result,
                experimental_seed=experimental_seed,
                task_seed=task_seed,
                source_commit=source_commit,
                action_was_executed=True,
                execution_note=note,
                c_reference=c_result,
            )
        )
    return rows


def build_pilot_summary(rows: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]) -> JsonDict:
    diversity = summarize_diversity(rows)
    verdict = choose_verdict(rows, gate)
    hard_case_candidate_exists = any(
        row.get("case_pool") != "positive_control_rescued"
        and row.get("action_id") in DIAGNOSTIC_ACTIONS
        and row.get("passed")
        for row in rows
    )
    changed_mechanisms = sorted(
        {
            str(row.get("action_id"))
            for row in rows
            if row.get("action_id") in {"E_oracle_sufficient_no_early_commit", "F_oracle_sufficient_trace_remask"}
            and row.get("output_changed_vs_C") is True
        }
    )
    action_costs: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get("action_id") in DIAGNOSTIC_ACTIONS and row.get("total_sec_including_probe") is not None:
            action_costs[str(row["action_id"])].append(float(row["total_sec_including_probe"]))
    action_cost_summary = {
        action_id: {"avg_sec": sum(values) / len(values), "count": len(values)}
        for action_id, values in sorted(action_costs.items())
    }
    by_task = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id"))].append(row)
    return {
        "mode": "distinct_candidate_generation_ceiling",
        "verdict": verdict,
        "action_distinctness_gate": gate,
        "new_actions_distinct": bool(gate.get("passed")),
        "multi_seed_diversity": any(
            info.get("unique_candidate_hash_count", 0) > 1
            for info in (diversity.get("case_summary") or {}).values()
        ),
        "hard_case_candidate_exists": hard_case_candidate_exists,
        "case_85_error_evolution": [
            {"action": row.get("action_id"), "seed": row.get("experimental_seed"), "error": row.get("error_type"), "passed": row.get("passed")}
            for row in by_task.get("SingleLineInfilling/HumanEval/85/L0", [])
            if row.get("action_id") in DIAGNOSTIC_ACTIONS
        ],
        "case_113_error_evolution": [
            {"action": row.get("action_id"), "seed": row.get("experimental_seed"), "error": row.get("error_type"), "passed": row.get("passed")}
            for row in by_task.get("SingleLineInfilling/HumanEval/113/L3", [])
            if row.get("action_id") in DIAGNOSTIC_ACTIONS
        ],
        "changed_mechanisms": changed_mechanisms,
        "change_type_note": "hash-level and verifier-level differences are reported separately; no unit-test label was used to choose remask positions.",
        "action_costs": action_cost_summary,
        "candidate_diversity_summary": diversity,
    }


def execute(args: argparse.Namespace) -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    output_dir = make_output_dir(args.output_dir, args.timestamp)
    manifest_path = output_dir / "run_manifest.json"
    manifest = build_manifest(args, output_dir, "running", started_at)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    try:
        baseline_rows = load_jsonl(args.baseline_results)
        route2_rows = load_jsonl(args.route2_results)
        task_ids = parse_task_ids_csv(args.task_ids_csv)
        case_manifest = build_case_manifest(
            baseline_rows,
            route2_rows,
            task_ids=task_ids,
            max_canvas_length=args.max_canvas_length,
        )
        action_manifest = build_action_manifest(case_manifest)
        write_csv(output_dir / "case_manifest.csv", case_manifest)
        write_csv(output_dir / "action_manifest.csv", action_manifest)
        config = {
            "baseline_results": args.baseline_results,
            "route2_results": args.route2_results,
            "task_ids": task_ids,
            "experimental_seeds": list(EXPERIMENTAL_SEEDS),
            "max_canvas_length": args.max_canvas_length,
            "model_path": args.model_path,
            "split": args.split,
            "dataset_subset": args.dataset_subset,
            "lcas_policy": args.lcas_policy,
            "trace_remask_rule": "token_flip_count desc, final_confidence asc, index asc; k=max(1,min(4,ceil(0.10*canvas_len)))",
        }
        (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        from clean_scripts import run_route2_trace_rescue as route2_runner

        route2_args = SimpleNamespace(
            model_path=args.model_path,
            split=args.split,
            dataset_subset=args.dataset_subset,
            max_samples=None,
            total_steps=64,
            seed=42,
            lcas_policy=args.lcas_policy,
            save_full_text_per_step=False,
            output_dir=str(output_dir),
            experiment_name="distinct_candidate_ceiling_internal",
        )
        route2_cfg = route2_runner.make_cfg(route2_args)
        route2_cfg.decode.save_step_traces = False
        route2_cfg.decode.save_full_text_per_step = False
        cfg = ExperimentConfig()
        cfg.model.model_path = args.model_path
        cfg.data.split = args.split
        cfg.data.dataset_subset = args.dataset_subset
        cfg.decode.lcas_policy = args.lcas_policy
        reset_action_seed(42)
        tokenizer, model = load_model_and_tokenizer(cfg.model)
        settings = route2_runner.default_midcons_settings()
        tasks = load_humaneval_infilling(split=args.split, dataset_subset=args.dataset_subset)
        by_task = {task.task_id: task for task in tasks}
        source_commit = current_commit()

        case_by_id = {str(case["task_id"]): case for case in case_manifest}
        smoke_case = case_by_id[SMOKE_TASK_ID]
        smoke_task = by_task[SMOKE_TASK_ID]
        rows: List[JsonDict] = []
        smoke_rows = run_diagnostic_actions(
            case=smoke_case,
            task=smoke_task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            experimental_seed=0,
            source_commit=source_commit,
        )
        rows.extend(smoke_rows)
        gate = smoke_gate(smoke_rows)
        if gate.get("passed"):
            for case in case_manifest:
                task = by_task[str(case["task_id"])]
                rows.extend(
                    run_ab_sanity(
                        case=case,
                        task=task,
                        tokenizer=tokenizer,
                        model=model,
                        cfg=route2_cfg,
                        settings=settings,
                        route2_runner=route2_runner,
                        source_commit=source_commit,
                    )
                )
                for seed in EXPERIMENTAL_SEEDS:
                    if str(case["task_id"]) == SMOKE_TASK_ID and seed == 0:
                        continue
                    rows.extend(
                        run_diagnostic_actions(
                            case=case,
                            task=task,
                            tokenizer=tokenizer,
                            model=model,
                            cfg=cfg,
                            experimental_seed=seed,
                            source_commit=source_commit,
                        )
                    )
        diversity = summarize_diversity(rows)
        equivalence = action_equivalence(rows)
        summary = build_pilot_summary(rows, gate)
        if summary["verdict"] not in VERDICTS:
            raise ValueError(f"invalid verdict: {summary['verdict']}")
        write_jsonl(output_dir / "pilot_results.jsonl", rows)
        write_csv(output_dir / "pilot_results.csv", [compact_result_record(row) for row in rows])
        (output_dir / "candidate_diversity_summary.json").write_text(
            json.dumps(diversity, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "action_equivalence.json").write_text(
            json.dumps(equivalence, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "pilot_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "pilot_report.md").write_text(render_report(summary, diversity, rows), encoding="utf-8")
        manifest = build_manifest(
            args,
            output_dir,
            "completed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            verdict=summary["verdict"],
            action_distinctness_gate=gate,
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Distinct-candidate ceiling output written to {output_dir}")
        print(json.dumps({"verdict": summary["verdict"], "gate": gate}, ensure_ascii=False, indent=2))
    except BaseException:
        manifest = build_manifest(
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
    parser = argparse.ArgumentParser(description="Run Phase 1b distinct-candidate generation ceiling pilot.")
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE)
    parser.add_argument("--route2-results", default=DEFAULT_ROUTE2)
    parser.add_argument("--output-dir", default="analysis_outputs")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--task-ids-csv", default=",".join(TASK_IDS))
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
