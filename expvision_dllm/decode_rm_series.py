from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import ExperimentConfig
from .data_loader import CodeTask
from .decode_structured import (
    choose_units_from_verifier,
    _pick_boundary_control_flow_unit,
    _project_unit_to_middle_or_local_fallback,
    _empty_repair_state,
)
from .decode_vanilla import _segment_decode, build_reconstruction_diagnostics, prepare_model_inputs
from .policies import linear_target_masks, select_low_confidence_mask_positions, structural_units_to_canvas_indices
from .verifier import run_verifier_stack, tier3_unit_tests


def _resolve_start_step(total_steps: int, start_ratio: float) -> int:
    if total_steps <= 1:
        return 0
    raw = int(total_steps * start_ratio)
    return max(0, min(total_steps - 1, raw))


def _should_probe(step: int, total_steps: int, start_step: int, stride: int) -> bool:
    if step < start_step:
        return False
    safe_stride = max(1, stride)
    if (step - start_step) % safe_stride == 0:
        return True
    return step == total_steps - 1


def _validate_projected_unit(unit, max_tokens: int, middle_length_tokens: int) -> Tuple[bool, str]:
    if unit is None:
        return False, "unit_none"
    if unit.unit_type == "statement":
        return False, "statement_rejected"
    if unit.char_end <= unit.char_start:
        return False, "empty_char_span"
    if max_tokens <= 0:
        return False, "nonpositive_span_budget"
    return True, "ok"


def _pick_subtree_locator_unit(
    *,
    full_text: str,
    prefix_text: str,
    middle_text: str,
    verifier_results,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    debug: Dict[str, Any] = {"locator_mode": "subtree_locator"}
    units, selection_debug, _weak = choose_units_from_verifier(
        full_text=full_text,
        prefix_text=prefix_text,
        middle_text=middle_text,
        verifier_results=verifier_results,
        cfg=copy.deepcopy(ExperimentConfig()),
        repair_state=_empty_repair_state(),
        issue_signature=None,
    )
    debug["selection_debug"] = selection_debug
    if not units:
        debug["locator_result"] = "no_unit"
        return None, debug
    unit = units[0]
    debug["locator_result"] = "ok"
    debug["unit"] = unit.to_dict()
    return unit, debug


def _pick_boundary_fragment_locator_unit(
    *,
    full_text: str,
    prefix_text: str,
    middle_text: str,
    verifier_results,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    debug: Dict[str, Any] = {"locator_mode": "boundary_control_flow"}
    tier3 = verifier_results.get("tier3_unit_tests")
    if tier3 is None:
        debug["locator_result"] = "missing_tier3"
        return None, debug
    boundary_unit, boundary_debug = _pick_boundary_control_flow_unit(
        full_text=full_text,
        prefix_text=prefix_text,
        middle_text=middle_text,
        tier3=tier3,
    )
    debug.update(boundary_debug)
    if boundary_unit is None:
        debug["locator_result"] = "no_boundary_unit"
        return None, debug
    projected_unit, projection_debug = _project_unit_to_middle_or_local_fallback(
        unit=boundary_unit,
        prefix_text=prefix_text,
        middle_text=middle_text,
    )
    debug.update(projection_debug)
    debug["locator_result"] = "ok"
    debug["unit"] = projected_unit.to_dict()
    return projected_unit, debug


def _unit_to_indices(tokenizer, middle_text: str, middle_length_tokens: int, unit) -> List[int]:
    return structural_units_to_canvas_indices(
        tokenizer=tokenizer,
        middle_text=middle_text,
        middle_length_tokens=middle_length_tokens,
        units=[unit],
    )


def _execute_strict_span(structural_indices: Sequence[int], target_masks: int) -> List[int]:
    # 旧思路如果硬受 target_masks 限制，会把“定位好的小结构”再次裁碎。
    # 删除原因：R 系列固定执行器需要一个足够朴素、不会偷偷引入 merge 技巧的版本。
    # return list(structural_indices)[:target_masks]
    return sorted(set(int(idx) for idx in structural_indices))


def _execute_global_union(low_conf_indices: Sequence[int], structural_indices: Sequence[int], target_masks: int) -> List[int]:
    # 新增：显式保留原始坏执行器，用于 M0 对照。
    merged = sorted(set(int(idx) for idx in low_conf_indices).union(int(idx) for idx in structural_indices))
    if target_masks <= 0:
        return []
    return merged[:target_masks]


def _execute_in_span_confidence(
    *,
    structural_indices: Sequence[int],
    middle_max_probs: torch.Tensor,
    target_masks: int,
) -> List[int]:
    unique_structural = sorted(set(int(idx) for idx in structural_indices))
    if not unique_structural or target_masks <= 0:
        return []
    scored = [(float(middle_max_probs[idx].item()), idx) for idx in unique_structural]
    scored.sort(key=lambda x: x[0])
    keep = min(target_masks, len(scored))
    return [idx for _, idx in scored[:keep]]


def _run_inference_verifier_stack(task: CodeTask, segments: Dict[str, str], cfg: ExperimentConfig):
    return run_verifier_stack(
        task=task,
        full_code=segments["full_text"],
        completion_without_suffix=segments["middle_text"],
        timeout=cfg.verifier.timeout,
    )


def _mode_defaults(mode: str) -> Dict[str, Any]:
    mode = str(mode).upper()
    defaults = {
        "R0": {"repair_enabled": False, "locator_mode": "none", "executor_mode": "none"},
        "R1": {"repair_enabled": True, "locator_mode": "subtree_locator", "executor_mode": "strict_span_only"},
        "R2": {"repair_enabled": True, "locator_mode": "boundary_control_flow", "executor_mode": "strict_span_only"},
        "M0": {"repair_enabled": True, "locator_mode": "subtree_locator", "executor_mode": "global_union"},
        "M1": {"repair_enabled": True, "locator_mode": "subtree_locator", "executor_mode": "strict_span_only"},
        "M2": {"repair_enabled": True, "locator_mode": "subtree_locator", "executor_mode": "in_span_confidence"},
    }
    if mode not in defaults:
        raise ValueError(f"Unsupported RM mode: {mode}")
    return defaults[mode]


def run_rm_series_decode(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    mode = str(cfg.decode.rm_control_mode).upper()
    defaults = _mode_defaults(mode)

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
    middle_length_tokens = middle_end - middle_start

    probe_start_step = _resolve_start_step(total_steps, cfg.decode.rm_probe_start_ratio)
    step_traces: List[Dict[str, Any]] = []
    decode_start = time.perf_counter()

    first_pass_step: Optional[int] = None
    first_pass_code: Optional[str] = None
    first_pass_middle_text: Optional[str] = None
    early_stopped = False
    repair_applied = False
    repair_step: Optional[int] = None
    repair_debug: Optional[Dict[str, Any]] = None

    inference_verifier_calls = 0
    inference_verifier_sec = 0.0
    probe_steps: List[int] = []
    last_middle_mean_confidence: Optional[float] = None
    final_segments: Optional[Dict[str, str]] = None

    for step in range(total_steps):
        current_mask_idx = (x_t == mask_token_id)
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)
        target_masks = linear_target_masks(num_mask_tokens, total_steps, step)
        middle_max_probs = max_probs[0, middle_start:middle_end]

        low_conf_indices = select_low_confidence_mask_positions(
            max_probs=max_probs[:, middle_start:middle_end],
            current_mask_idx=current_mask_idx[:, middle_start:middle_end],
            target_masks=target_masks,
        ) if target_masks > 0 else []

        # 旧实现先构造 filled_tensor，再在 filled candidate 上做 probe，导致 R0 与 G1 不对齐。
        # 删除原因：G1 的 stopping 判定发生在“默认 low-confidence remask 已应用之后”的轨迹状态上。
        # 为了让 R0 严格等价于 G1，R/M 链也必须先完整执行这一默认更新；
        # 只有当 Tier3 fail 且满足 repair 条件时，才允许回滚到 filled candidate 并替换 remask 动作。
        filled_tensor = x_t.clone()
        filled_tensor[current_mask_idx] = preds[current_mask_idx]

        default_next_x = filled_tensor.clone()
        if target_masks > 0 and low_conf_indices:
            default_next_x[0, [middle_start + idx for idx in low_conf_indices]] = mask_token_id
        x_t = default_next_x

        middle_conf = max_probs[:, middle_start:middle_end]
        last_middle_mean_confidence = float(middle_conf.mean().item())

        segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=x_t[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )
        final_segments = segments

        inference_probe = None
        probe_called = False
        step_repair_debug = None

        if _should_probe(step, total_steps, probe_start_step, cfg.decode.rm_probe_stride):
            probe_called = True
            probe_steps.append(step)

            # 先对齐 G1：默认只做 Tier3 probe。
            probe_start = time.perf_counter()
            tier3_probe = tier3_unit_tests(task, segments["middle_text"], timeout=cfg.verifier.timeout)
            inference_verifier_sec += time.perf_counter() - probe_start
            inference_verifier_calls += 1
            inference_probe = {"tier3_unit_tests": tier3_probe}

            if tier3_probe.passed and first_pass_step is None:
                first_pass_step = step
                first_pass_code = segments["full_text"]
                first_pass_middle_text = segments["middle_text"]

            if tier3_probe.passed:
                early_stopped = True
                step_traces.append({
                    "task_id": task.task_id,
                    "step": step,
                    "rm_control_mode": mode,
                    "target_masks": target_masks,
                    "remaining_masks": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
                    "low_confidence_indices": low_conf_indices,
                    "verifier_called": True,
                    "verifier_tier": "tier3_unit_tests",
                    "inference_probe": {k: v.to_dict() for k, v in inference_probe.items()},
                    "repair_debug": None,
                    "full_text": segments["full_text"],
                    "middle_text": segments["middle_text"],
                    "reconstruction_diagnostics": build_reconstruction_diagnostics(task, segments),
                    "early_stop_triggered": True,
                })
                break

            can_try_repair = (
                defaults["repair_enabled"]
                and not repair_applied
                and target_masks > 0
                and step >= probe_start_step
            )

            if can_try_repair:
                # repair 模式才额外调用 full stack；R0 不会走到这里，因此不会再与 G1 漂移。
                verifier_start = time.perf_counter()
                full_probe = _run_inference_verifier_stack(task, segments, cfg)
                inference_verifier_sec += time.perf_counter() - verifier_start
                # Tier3 在前面已经调用过一次，这里只把额外层的调用数补进去，避免重复算 probe 次数。
                inference_verifier_calls += max(0, len(full_probe) - 1)
                inference_probe = full_probe

                can_try_repair = (
                    full_probe.get("tier3_unit_tests") is not None
                    and not full_probe["tier3_unit_tests"].passed
                )
                if cfg.decode.rm_trigger_requires_tier12_pass:
                    can_try_repair = can_try_repair and bool(full_probe.get("tier1_parse_compile", None) and full_probe["tier1_parse_compile"].passed) and bool(full_probe.get("tier2_smoke_exec", None) and full_probe["tier2_smoke_exec"].passed)

            if can_try_repair:
                if defaults["locator_mode"] == "subtree_locator":
                    unit, locator_debug = _pick_subtree_locator_unit(
                        full_text=segments["full_text"],
                        prefix_text=segments["prefix_text"],
                        middle_text=segments["middle_text"],
                        verifier_results=inference_probe,
                    )
                elif defaults["locator_mode"] == "boundary_control_flow":
                    unit, locator_debug = _pick_boundary_fragment_locator_unit(
                        full_text=segments["full_text"],
                        prefix_text=segments["prefix_text"],
                        middle_text=segments["middle_text"],
                        verifier_results=inference_probe,
                    )
                else:
                    unit, locator_debug = None, {"locator_mode": defaults["locator_mode"], "locator_result": "disabled"}

                structural_indices = _unit_to_indices(tokenizer, segments["middle_text"], middle_length_tokens, unit) if unit is not None else []
                structural_indices = sorted(set(int(idx) for idx in structural_indices if 0 <= int(idx) < middle_length_tokens))
                valid_unit, validate_reason = _validate_projected_unit(unit, cfg.decode.rm_repair_max_token_span, middle_length_tokens)
                if valid_unit and len(structural_indices) > cfg.decode.rm_repair_max_token_span:
                    valid_unit = False
                    validate_reason = "token_span_exceeds_limit"

                executor_mode = defaults["executor_mode"]
                remask_indices = list(low_conf_indices)
                if valid_unit:
                    if executor_mode == "strict_span_only":
                        remask_indices = _execute_strict_span(structural_indices, target_masks)
                    elif executor_mode == "global_union":
                        remask_indices = _execute_global_union(low_conf_indices, structural_indices, target_masks)
                    elif executor_mode == "in_span_confidence":
                        remask_indices = _execute_in_span_confidence(
                            structural_indices=structural_indices,
                            middle_max_probs=middle_max_probs,
                            target_masks=target_masks,
                        )

                step_repair_debug = {
                    "repair_attempted": True,
                    "locator_debug": locator_debug,
                    "validate_reason": validate_reason,
                    "valid_unit": valid_unit,
                    "executor_mode": executor_mode,
                    "low_confidence_indices": list(low_conf_indices),
                    "structural_indices": structural_indices,
                    "applied_remask_indices": list(remask_indices),
                    "repair_applied": bool(valid_unit and remask_indices != list(low_conf_indices)),
                }

                if valid_unit and remask_indices != list(low_conf_indices):
                    # 新增：回滚到 filled candidate，再用 repair remask 覆盖默认 low-confidence remask。
                    x_t = filled_tensor.clone()
                    if remask_indices:
                        x_t[0, [middle_start + idx for idx in remask_indices]] = mask_token_id
                    repair_applied = True
                    repair_step = step
                    repair_debug = copy.deepcopy(step_repair_debug)

        step_traces.append({
            "task_id": task.task_id,
            "step": step,
            "rm_control_mode": mode,
            "target_masks": target_masks,
            "remaining_masks": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
            "low_confidence_indices": low_conf_indices,
            "verifier_called": probe_called,
            "verifier_tier": "full_stack" if (probe_called and inference_probe is not None and len(inference_probe) > 1) else ("tier3_unit_tests" if probe_called else None),
            "inference_probe": {k: v.to_dict() for k, v in inference_probe.items()} if inference_probe is not None else None,
            "repair_debug": step_repair_debug,
            "full_text": segments["full_text"],
            "middle_text": segments["middle_text"],
            "reconstruction_diagnostics": build_reconstruction_diagnostics(task, segments),
            "early_stop_triggered": False,
        })

    total_decode_sec = time.perf_counter() - decode_start
    if final_segments is None:
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
        timeout=cfg.verifier.timeout,
    )
    final_verifier_sec = sum(v.duration_sec for v in final_verification.values())
    final_verifier_calls = len(final_verification)
    final_tier3 = final_verification.get("tier3_unit_tests")
    final_passed = bool(final_tier3.passed) if final_tier3 is not None else False

    backslide = False
    wasted_steps_after_first_pass = None
    if first_pass_step is not None:
        backslide = not final_passed and (not early_stopped)
        wasted_steps_after_first_pass = (total_steps - 1 - first_pass_step) if not early_stopped else 0

    return {
        "task_id": task.task_id,
        "task": task,
        "dataset_subset": cfg.decode.dataset_subset,
        "code": final_segments["full_text"],
        "prefix_text": final_segments["prefix_text"],
        "middle_text": final_segments["middle_text"],
        "suffix_text": final_segments["suffix_text"],
        "step_traces": step_traces,
        "metrics": {
            "passed": final_passed,
            "decode_sec": total_decode_sec,
            "verification_sec": final_verifier_sec,
            "total_sec": total_decode_sec + inference_verifier_sec + final_verifier_sec,
            "early_stop": early_stopped,
            "first_pass_step": first_pass_step,
            "final_pass_step": first_pass_step if early_stopped and first_pass_step is not None else total_steps - 1,
            "backslide": backslide,
            "rescue": False,
            "wasted_steps_after_first_pass": wasted_steps_after_first_pass,
            "inference_verifier_calls": inference_verifier_calls,
            "inference_verifier_sec": inference_verifier_sec,
            "final_verifier_calls": final_verifier_calls,
            "final_verifier_sec": final_verifier_sec,
            "verifier_calls_total": inference_verifier_calls + final_verifier_calls,
            "verifier_total_sec": inference_verifier_sec + final_verifier_sec,
            "mean_final_confidence": last_middle_mean_confidence,
            "rm_control_mode": mode,
            "repair_applied": repair_applied,
            "repair_step": repair_step,
        },
        "verification": {k: v.to_dict() for k, v in final_verification.items()},
        "diagnostics": {
            **build_reconstruction_diagnostics(task, final_segments),
            "rm_control_mode": mode,
            "probe_start_step": probe_start_step,
            "probe_steps": probe_steps,
            "repair_debug": repair_debug,
            "first_pass_middle_text": first_pass_middle_text,
            "first_pass_code": first_pass_code,
        },
        "prepared": prepared,
    }