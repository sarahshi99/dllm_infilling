from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import ExperimentConfig
from .data_loader import CodeTask
from .decode_vanilla import _segment_decode, build_reconstruction_diagnostics, prepare_model_inputs
from .policies import linear_target_masks, select_low_confidence_mask_positions
from .snapshot_selector import (
    build_raw_snapshot,
    build_sample_trajectory_summary,
    build_snapshot_records,
    resolve_sampling_steps,
    select_best_snapshot,
    should_capture_step,
)
from .verifier import run_verifier_stack, tier1_parse_and_compile, tier2_smoke_exec, tier3_unit_tests


# =========================
# 旧实现（保留对照，不再直接使用）
# =========================
# def _resolve_start_step(total_steps: int, start_ratio: float) -> int:
#     if total_steps <= 1:
#         return 0
#     raw = int(total_steps * start_ratio)
#     return max(0, min(total_steps - 1, raw))
#
# def _should_capture(step: int, total_steps: int, start_step: int, stride: int) -> bool:
#     if step < start_step:
#         return False
#     safe_stride = max(1, stride)
#     if (step - start_step) % safe_stride == 0:
#         return True
#     return step == total_steps - 1
#
# 删除原因：
# 当前已经冻结成显式 sampling policy，而不是 start_ratio/stride 的隐式组合。


def _proxy_snapshot_verification(full_text: str) -> Dict[str, Any]:
    tier1 = tier1_parse_and_compile(full_text)
    result = {tier1.tier: tier1}
    if tier1.passed:
        tier2 = tier2_smoke_exec(full_text)
        result[tier2.tier] = tier2
    return result


def _build_run_metadata(run_id: str, cfg: ExperimentConfig) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset_subset": cfg.decode.dataset_subset,
        "split": cfg.decode.dataset_split,
        "max_samples": cfg.decode.max_samples,
        "seed": cfg.decode.seed,
        "total_steps": cfg.decode.total_steps,
        "sampling_policy_version": cfg.decode.s_sampling_policy_version,
        "feature_schema_version": cfg.decode.s_feature_schema_version,
        "selector_version": cfg.decode.s_selector_version,
        "verifier_version": cfg.decode.verifier_version,
        "baseline_chain_version": cfg.decode.baseline_chain_version,
    }


def _compute_snapshot_tier3_summary(
    task: CodeTask,
    raw_snapshots: Sequence[Dict[str, Any]],
    timeout: float,
) -> Dict[str, Any]:
    first_tier3_step: Optional[int] = None
    backslide_from_step: Optional[int] = None
    oracle_first_pass_step: Optional[int] = None
    saw_tier3_pass = False
    final_snapshot_step = int(raw_snapshots[-1]["step"])
    final_snapshot_tier3_pass = False

    for snapshot in raw_snapshots:
        step = int(snapshot["step"])
        tier3 = tier3_unit_tests(task, str(snapshot["middle_text"]), timeout=timeout)
        passed = bool(tier3.passed)
        if passed and first_tier3_step is None:
            first_tier3_step = step
            oracle_first_pass_step = step
        if first_tier3_step is not None and not passed and backslide_from_step is None and step > first_tier3_step:
            backslide_from_step = step
        if passed:
            saw_tier3_pass = True
        if step == final_snapshot_step:
            final_snapshot_tier3_pass = passed

    if not saw_tier3_pass:
        backslide_happened = False
    else:
        backslide_happened = not final_snapshot_tier3_pass

    return {
        "tier3_snapshot_summary_available": True,
        "first_tier3_step": first_tier3_step,
        "backslide_happened": backslide_happened,
        "backslide_from_step": backslide_from_step,
        "oracle_first_pass_step": oracle_first_pass_step,
    }


# =========================
# 旧实现（保留对照，不再直接使用）
# =========================
# def _rank_snapshot(snapshot: Dict[str, Any]) -> Tuple[Any, ...]:
#     proxy = snapshot["proxy_verification"]
#     tier1_pass = int(bool(proxy.get("tier1_parse_compile", {}).get("passed", False)))
#     tier2_pass = int(bool(proxy.get("tier2_smoke_exec", {}).get("passed", False)))
#     stable_run_length = int(snapshot.get("stable_run_length", 1))
#     mean_conf = float(snapshot.get("mean_confidence") or 0.0)
#     step = int(snapshot.get("step", 0))
#     return (tier2_pass, tier1_pass, stable_run_length, -step, mean_conf)
#
# def _select_snapshot_proxy(...)
# def _select_snapshot_consistency(...)
#
# 删除原因：
# 旧版直接在 decode 文件里做排序；当前已冻结“selector-specific score 只进入 SelectorDecision”，
# 因此统一迁移到 snapshot_selector.py。


def run_s_series_decode(task: CodeTask, tokenizer, model, cfg: ExperimentConfig, run_id: str) -> Dict[str, Any]:
    mode = str(cfg.decode.s_control_mode).upper()

    # =========================
    # 旧实现（保留对照，不再直接使用）
    # =========================
    # if mode not in {"S0", "S1", "S2"}:
    #     raise ValueError(f"Unsupported S mode: {mode}")
    #
    # 删除原因：
    # 当前需要在原有 S0/S1/S2 之外扩展 S3/S4/S5，并统一在同一 schema 下记录决策日志。
    if mode not in {"S0", "S1", "S2", "S3", "S4", "S5"}:
        raise ValueError(f"Unsupported S mode: {mode}")

    prepared = prepare_model_inputs(task, tokenizer, cfg)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    mask_token_id = prepared["mask_token_id"]
    middle_start = prepared["middle_start"]
    middle_end = prepared["middle_end"]
    num_mask_tokens = cfg.decode.num_mask_tokens
    total_steps = cfg.decode.total_steps

    sampling_steps = resolve_sampling_steps(total_steps, cfg.decode.s_sampling_policy_version)
    sampling_step_set = set(sampling_steps)

    step_traces: List[Dict[str, Any]] = []
    raw_snapshots: List[Dict[str, Any]] = []
    decode_start = time.perf_counter()
    final_segments: Optional[Dict[str, str]] = None
    last_middle_mean_confidence: Optional[float] = None

    inference_verifier_calls = 0
    inference_verifier_sec = 0.0

    for step in range(total_steps):
        current_mask_idx = (x_t == mask_token_id)
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)

        filled_tensor = x_t.clone()
        filled_tensor[current_mask_idx] = preds[current_mask_idx]
        target_masks = linear_target_masks(num_mask_tokens, total_steps, step)
        low_conf_indices = select_low_confidence_mask_positions(
            max_probs=max_probs[:, middle_start:middle_end],
            current_mask_idx=current_mask_idx[:, middle_start:middle_end],
            target_masks=target_masks,
        ) if target_masks > 0 else []
        next_tensor = filled_tensor.clone()
        if target_masks > 0 and low_conf_indices:
            next_tensor[0, [middle_start + idx for idx in low_conf_indices]] = mask_token_id
        x_t = next_tensor

        segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=filled_tensor[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )
        final_segments = segments

        detached_middle_conf = max_probs[0, middle_start:middle_end].detach().float().cpu()
        last_middle_mean_confidence = float(detached_middle_conf.mean().item()) if detached_middle_conf.numel() > 0 else 0.0
        current_min_confidence = float(detached_middle_conf.min().item()) if detached_middle_conf.numel() > 0 else 0.0
        current_low_conf_count = int((detached_middle_conf < float(cfg.decode.s_low_conf_threshold)).sum().item()) if detached_middle_conf.numel() > 0 else 0

        captured_snapshot = None
        if should_capture_step(step, sampling_step_set):
            verifier_start = time.perf_counter()
            proxy_verification_obj = _proxy_snapshot_verification(segments["full_text"])
            inference_verifier_sec += time.perf_counter() - verifier_start
            inference_verifier_calls += len(proxy_verification_obj)
            captured_snapshot = build_raw_snapshot(
                full_text=segments["full_text"],
                middle_text=segments["middle_text"],
                step=step,
                mean_confidence=last_middle_mean_confidence,
                min_confidence=current_min_confidence,
                low_conf_count=current_low_conf_count,
                proxy_verification={k: v.to_dict() for k, v in proxy_verification_obj.items()},
            )
            raw_snapshots.append(copy.deepcopy(captured_snapshot))

        step_traces.append({
            "task_id": task.task_id,
            "step": step,
            "s_control_mode": mode,
            "sampling_policy_version": cfg.decode.s_sampling_policy_version,
            "target_masks": target_masks,
            "remaining_masks": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
            "low_confidence_indices": low_conf_indices,
            "snapshot_captured": captured_snapshot,
            "full_text": segments["full_text"],
            "middle_text": segments["middle_text"],
            "reconstruction_diagnostics": build_reconstruction_diagnostics(task, segments),
        })

    total_decode_sec = time.perf_counter() - decode_start
    if final_segments is None:
        final_segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=x_t[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )

    if not raw_snapshots:
        proxy_verification_obj = _proxy_snapshot_verification(final_segments["full_text"])
        raw_snapshots.append(
            build_raw_snapshot(
                full_text=final_segments["full_text"],
                middle_text=final_segments["middle_text"],
                step=total_steps - 1,
                mean_confidence=last_middle_mean_confidence if last_middle_mean_confidence is not None else 0.0,
                min_confidence=0.0,
                low_conf_count=0,
                proxy_verification={k: v.to_dict() for k, v in proxy_verification_obj.items()},
            )
        )

    run_metadata = _build_run_metadata(run_id, cfg)
    selector_version = f"{mode.lower()}::{cfg.decode.s_selector_version}"
    snapshot_records = build_snapshot_records(
        run_id=run_id,
        sample_id=task.task_id,
        dataset_subset=cfg.decode.dataset_subset,
        split=cfg.decode.dataset_split,
        max_samples=cfg.decode.max_samples,
        seed=cfg.decode.seed,
        total_steps=cfg.decode.total_steps,
        sampling_policy_version=cfg.decode.s_sampling_policy_version,
        feature_schema_version=cfg.decode.s_feature_schema_version,
        selector_version=selector_version,
        verifier_version=cfg.decode.verifier_version,
        baseline_chain_version=cfg.decode.baseline_chain_version,
        raw_snapshots=raw_snapshots,
        recent_window=cfg.decode.s_recent_window,
    )

    selected_record, selector_decision = select_best_snapshot(
        run_id=run_id,
        sample_id=task.task_id,
        records=snapshot_records,
        mode=mode,
        selector_version=selector_version,
        consistency_window=cfg.decode.s_consistency_window,
    )

    selected_snapshot = next(snapshot for snapshot in raw_snapshots if int(snapshot["step"]) == int(selected_record.step))
    selected_snapshot = copy.deepcopy(selected_snapshot)
    selected_snapshot["selection_reason"] = selector_decision.tie_break_reason

    final_verification = run_verifier_stack(
        task=task,
        full_code=selected_snapshot["full_text"],
        completion_without_suffix=selected_snapshot["middle_text"],
        timeout=cfg.verifier.timeout,
    )
    final_verifier_sec = sum(v.duration_sec for v in final_verification.values())
    final_verifier_calls = len(final_verification)
    final_tier3 = final_verification.get("tier3_unit_tests")
    final_passed = bool(final_tier3.passed) if final_tier3 is not None else False

    tier3_summary = {
        "tier3_snapshot_summary_available": False,
        "first_tier3_step": None,
        "backslide_happened": None,
        "backslide_from_step": None,
        "oracle_first_pass_step": None,
    }
    if cfg.decode.s_capture_snapshot_tier3_summary:
        tier3_summary = _compute_snapshot_tier3_summary(
            task=task,
            raw_snapshots=raw_snapshots,
            timeout=cfg.verifier.timeout,
        )

    trajectory_summary = build_sample_trajectory_summary(
        run_id=run_id,
        sample_id=task.task_id,
        dataset_subset=cfg.decode.dataset_subset,
        split=cfg.decode.dataset_split,
        max_samples=cfg.decode.max_samples,
        seed=cfg.decode.seed,
        total_steps=cfg.decode.total_steps,
        sampling_policy_version=cfg.decode.s_sampling_policy_version,
        feature_schema_version=cfg.decode.s_feature_schema_version,
        selector_version=selector_version,
        verifier_version=cfg.decode.verifier_version,
        baseline_chain_version=cfg.decode.baseline_chain_version,
        records=snapshot_records,
        final_tier3_pass=final_passed,
        first_tier3_step=tier3_summary["first_tier3_step"],
        backslide_happened=tier3_summary["backslide_happened"],
        backslide_from_step=tier3_summary["backslide_from_step"],
        oracle_first_pass_step=tier3_summary["oracle_first_pass_step"],
        tier3_snapshot_summary_available=tier3_summary["tier3_snapshot_summary_available"],
    )

    return {
        "task_id": task.task_id,
        "task": task,
        "dataset_subset": cfg.decode.dataset_subset,
        "run_metadata": run_metadata,
        "code": selected_snapshot["full_text"],
        "prefix_text": task.prefix,
        "middle_text": selected_snapshot["middle_text"],
        "suffix_text": task.suffix,
        "step_traces": step_traces,
        "metrics": {
            "passed": final_passed,
            "decode_sec": total_decode_sec,
            "verification_sec": final_verifier_sec,
            "total_sec": total_decode_sec + inference_verifier_sec + final_verifier_sec,
            "early_stop": False,
            "first_pass_step": None,
            "final_pass_step": int(selected_snapshot["step"]),
            "backslide": trajectory_summary.backslide_happened,
            "rescue": False,
            "wasted_steps_after_first_pass": None,
            "inference_verifier_calls": inference_verifier_calls,
            "inference_verifier_sec": inference_verifier_sec,
            "final_verifier_calls": final_verifier_calls,
            "final_verifier_sec": final_verifier_sec,
            "verifier_calls_total": inference_verifier_calls + final_verifier_calls,
            "verifier_total_sec": inference_verifier_sec + final_verifier_sec,
            "mean_final_confidence": last_middle_mean_confidence,
            "s_control_mode": mode,
            "snapshot_count": len(raw_snapshots),
            "selected_snapshot_step": int(selected_snapshot["step"]),
            "sampling_policy_version": cfg.decode.s_sampling_policy_version,
            "feature_schema_version": cfg.decode.s_feature_schema_version,
            "selector_version": selector_version,
            "verifier_version": cfg.decode.verifier_version,
            "baseline_chain_version": cfg.decode.baseline_chain_version,
        },
        "verification": {k: v.to_dict() for k, v in final_verification.items()},
        "diagnostics": {
            **build_reconstruction_diagnostics(
                task,
                {
                    "prefix_text": task.prefix,
                    "middle_text": selected_snapshot["middle_text"],
                    "suffix_text": task.suffix,
                    "full_text": selected_snapshot["full_text"],
                },
            ),
            "s_control_mode": mode,
            "sampling_policy_version": cfg.decode.s_sampling_policy_version,
            "sampling_steps": sampling_steps,
            "raw_snapshots": raw_snapshots,
            "selected_snapshot": selected_snapshot,
            "selector_decision": selector_decision.to_dict(),
            "trajectory_summary": trajectory_summary.to_dict(),
        },
        "snapshot_records": [record.to_dict() for record in snapshot_records],
        "selector_decision": selector_decision.to_dict(),
        "trajectory_summary": trajectory_summary.to_dict(),
        "prepared": prepared,
    }