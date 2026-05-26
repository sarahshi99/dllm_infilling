from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

from .config import ExperimentConfig
from .dataset import CodeTask
from .decode import (
    _segment_decode,
    build_reconstruction_diagnostics,
    linear_target_masks,
    prepare_model_inputs,
    resolve_mask_length,
    select_low_confidence_mask_positions,
)
from .stopping import (
    GlobalGapStoppingConfig,
    StoppingDecision,
    evaluate_global_gap_stopping,
    validate_stopping_config,
)
from .verifier import run_verifier_stack


def choose_lcas_config(selected_mask_length: int, policy_name: str) -> Optional[GlobalGapStoppingConfig]:
    if policy_name == "lcas_v1":
        if selected_mask_length <= 12:
            return GlobalGapStoppingConfig(
                min_stop_step=32,
                gap_threshold=0.50,
                top1_threshold=0.75,
                max_remaining_mask_ratio=0.50,
            )
        if selected_mask_length <= 16:
            return GlobalGapStoppingConfig(
                min_stop_step=40,
                gap_threshold=0.50,
                top1_threshold=0.75,
                max_remaining_mask_ratio=0.375,
            )
        return None

    if policy_name == "lcas_v2":
        if selected_mask_length <= 12:
            return GlobalGapStoppingConfig(
                min_stop_step=28,
                gap_threshold=0.40,
                top1_threshold=0.60,
                max_remaining_mask_ratio=0.625,
            )
        if selected_mask_length <= 16:
            return GlobalGapStoppingConfig(
                min_stop_step=40,
                gap_threshold=0.50,
                top1_threshold=0.75,
                max_remaining_mask_ratio=0.375,
            )
        return None

    raise ValueError(f"Unknown LCAS policy: {policy_name}")


def no_stop_decision(step: int, remaining_masks: int, selected_mask_length: int) -> Dict[str, Any]:
    ratio = remaining_masks / float(selected_mask_length) if selected_mask_length > 0 else 0.0
    return StoppingDecision(
        should_stop=False,
        reason="lcas_policy_no_stop",
        step=step,
        remaining_masks=remaining_masks,
        remaining_mask_ratio=ratio,
        mean_gap=None,
        min_gap=None,
        mean_top1=None,
        max_top1=None,
        min_top1=None,
    ).to_dict()


def run_decode_with_lcas(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    policy_name = str(getattr(cfg.decode, "lcas_policy", "lcas_v1"))

    length_meta = resolve_mask_length(task, tokenizer, model, cfg)
    prepared = prepare_model_inputs(task, tokenizer, length_meta["mask_length"], cfg)

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    mask_token_id = prepared["mask_token_id"]
    middle_start = prepared["middle_start"]
    middle_end = prepared["middle_end"]

    selected_mask_length = int(length_meta["mask_length"])
    total_steps = int(cfg.decode.total_steps)

    stop_cfg = choose_lcas_config(selected_mask_length, policy_name)
    if stop_cfg is not None:
        validate_stopping_config(stop_cfg, total_steps=total_steps)

    lcas_bucket = (
        "short_selected_<=12"
        if selected_mask_length <= 12
        else "medium_selected_13_16"
        if selected_mask_length <= 16
        else "long_selected_17plus"
    )

    step_traces: List[Dict[str, Any]] = []
    stopping_trace: List[Dict[str, Any]] = []

    stopped = False
    stop_step: Optional[int] = None
    stop_reason = "not_stopped"
    stop_decision_at_stop: Optional[Dict[str, Any]] = None

    final_mean_confidence: Optional[float] = None
    final_mean_gap: Optional[float] = None
    final_mean_top1: Optional[float] = None
    final_remaining_masks: Optional[int] = None
    final_remaining_mask_ratio: Optional[float] = None

    decode_start = time.perf_counter()

    for step in range(total_steps):
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)

        current_mask_idx = x_t == mask_token_id
        middle_probs = probs[:, middle_start:middle_end, :]
        middle_max_probs = max_probs[:, middle_start:middle_end]
        middle_mask_idx = current_mask_idx[:, middle_start:middle_end]
        remaining_masks = int(middle_mask_idx.sum().item())

        if stop_cfg is None:
            decision_dict = no_stop_decision(
                step=step,
                remaining_masks=remaining_masks,
                selected_mask_length=selected_mask_length,
            )
        else:
            decision = evaluate_global_gap_stopping(
                step=step,
                middle_probs=middle_probs,
                middle_mask_idx=middle_mask_idx,
                selected_mask_length=selected_mask_length,
                stop_cfg=stop_cfg,
            )
            decision_dict = decision.to_dict()

        decision_dict["lcas_policy"] = policy_name
        decision_dict["lcas_bucket"] = lcas_bucket
        decision_dict["lcas_stop_enabled"] = stop_cfg is not None
        stopping_trace.append(decision_dict)

        final_mean_gap = decision_dict.get("mean_gap")
        final_mean_top1 = decision_dict.get("mean_top1")
        final_remaining_masks = decision_dict.get("remaining_masks")
        final_remaining_mask_ratio = decision_dict.get("remaining_mask_ratio")

        if bool(decision_dict.get("should_stop", False)):
            x_t[current_mask_idx] = preds[current_mask_idx]
            stopped = True
            stop_step = step
            stop_reason = str(decision_dict.get("reason"))
            stop_decision_at_stop = decision_dict
            final_mean_confidence = decision_dict.get("mean_top1")
            break

        if decision_dict.get("reason") == "no_remaining_masks":
            stop_reason = "no_remaining_masks"
            stop_decision_at_stop = decision_dict
            final_mean_confidence = None
            break

        target_masks = linear_target_masks(selected_mask_length, total_steps, step)

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
            low_conf_indices = []

        final_mean_confidence = float(middle_max_probs.mean().item())

        if cfg.decode.save_step_traces:
            trace: Dict[str, Any] = {
                "task_id": task.task_id,
                "step": step,
                "target_masks": target_masks,
                "remaining_masks_before_update": int(middle_mask_idx.sum().item()),
                "remaining_masks_after_update": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
                "low_confidence_indices": low_conf_indices,
                "mean_confidence": final_mean_confidence,
                "stop_decision": decision_dict,
            }
            if cfg.decode.save_full_text_per_step:
                segments = _segment_decode(
                    tokenizer=tokenizer,
                    prefix_ids=prepared["prefix_ids"],
                    middle_ids=x_t[0, middle_start:middle_end].tolist(),
                    suffix_ids=prepared["suffix_ids"],
                )
                trace["middle_text"] = segments["middle_text"]
                trace["full_text"] = segments["full_text"]
            step_traces.append(trace)

    effective_steps = (stop_step + 1) if stopped and stop_step is not None else total_steps
    total_decode_sec = time.perf_counter() - decode_start

    final_segments = _segment_decode(
        tokenizer=tokenizer,
        prefix_ids=prepared["prefix_ids"],
        middle_ids=x_t[0, middle_start:middle_end].tolist(),
        suffix_ids=prepared["suffix_ids"],
    )

    final_verification = run_verifier_stack(
        task=task,
        full_code=final_segments["full_text"],
        completion_without_suffix=final_segments["middle_text"],
    )

    verification_sec = sum(item.duration_sec for item in final_verification.values())
    total_sec = total_decode_sec + verification_sec
    total_sec_including_probe = total_sec + float(length_meta["length_probe_sec"])

    tier3 = final_verification.get("tier3_unit_tests")
    passed = bool(tier3.passed) if tier3 else False

    stop_config_dict = None if stop_cfg is None else asdict(stop_cfg)

    return {
        "task_id": task.task_id,
        "task": task,
        "code": final_segments["full_text"],
        "prefix_text": final_segments["prefix_text"],
        "middle_text": final_segments["middle_text"],
        "suffix_text": final_segments["suffix_text"],
        "step_traces": step_traces,
        "stopping_trace": stopping_trace,
        "length_probe": {
            "candidate_scores": length_meta["candidate_scores"],
            "length_probe_sec": length_meta["length_probe_sec"],
            "selected_mask_length": length_meta["selected_mask_length"],
            "selected_score": length_meta["selected_score"],
            "selected_minus_oracle_length": length_meta.get("selected_minus_oracle_length"),
            "abs_selected_minus_oracle_length": length_meta.get("abs_selected_minus_oracle_length"),
            "probe_lengths": length_meta["probe_lengths"],
            "tie_break": length_meta["tie_break"],
        },
        "stopping": {
            "enabled": stop_cfg is not None,
            "method": "length_conditioned_global_gap_early_commit",
            "policy": policy_name,
            "bucket": lcas_bucket,
            "stopped": stopped,
            "stop_step": stop_step,
            "stop_reason": stop_reason,
            "effective_steps": effective_steps,
            "stop_decision": stop_decision_at_stop,
            "config": stop_config_dict,
        },
        "metrics": {
            "passed": passed,
            "decode_sec": total_decode_sec,
            "verification_sec": verification_sec,
            "total_sec": total_sec,
            "total_sec_including_probe": total_sec_including_probe,
            "length_probe_sec": float(length_meta["length_probe_sec"]),
            "total_steps": total_steps,
            "effective_steps": effective_steps,
            "stopped": stopped,
            "stop_step": stop_step,
            "stop_reason": stop_reason,
            "mean_final_confidence": final_mean_confidence,
            "mean_gap_at_stop_or_final": final_mean_gap,
            "mean_top1_at_stop_or_final": final_mean_top1,
            "remaining_masks_at_stop_or_final": final_remaining_masks,
            "remaining_mask_ratio_at_stop_or_final": final_remaining_mask_ratio,
            "mask_length": length_meta["mask_length"],
            "oracle_mask_length": length_meta["oracle_mask_length"],
            "selected_mask_length": length_meta["selected_mask_length"],
            "selected_score": length_meta["selected_score"],
            "selected_minus_oracle_length": length_meta.get("selected_minus_oracle_length"),
            "abs_selected_minus_oracle_length": length_meta.get("abs_selected_minus_oracle_length"),
            "mask_length_source": length_meta["mask_length_source"],
            "lcas_policy": policy_name,
            "lcas_bucket": lcas_bucket,
            "lcas_stop_enabled": stop_cfg is not None,
            "lcas_min_stop_step": None if stop_cfg is None else stop_cfg.min_stop_step,
            "lcas_gap_threshold": None if stop_cfg is None else stop_cfg.gap_threshold,
            "lcas_top1_threshold": None if stop_cfg is None else stop_cfg.top1_threshold,
            "lcas_max_remaining_mask_ratio": None if stop_cfg is None else stop_cfg.max_remaining_mask_ratio,
        },
        "verification": {key: value.to_dict() for key, value in final_verification.items()},
        "diagnostics": build_reconstruction_diagnostics(task, final_segments),
    }