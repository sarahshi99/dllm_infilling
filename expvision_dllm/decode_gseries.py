from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import ExperimentConfig
from .data_loader import CodeTask
from .decode_vanilla import _segment_decode, build_reconstruction_diagnostics, prepare_model_inputs
from .model_utils import resolve_mask_token_id
from .policies import linear_target_masks, select_low_confidence_mask_positions
from .verifier import run_verifier_stack, tier1_parse_and_compile, tier2_smoke_exec, tier3_unit_tests


def _resolve_probe_start_step(total_steps: int, start_ratio: float) -> int:
    if total_steps <= 1:
        return 0
    # 新增：probe 默认只从后段开始，避免把 Tier 3 检查过早压进高噪声阶段。
    raw = int(total_steps * start_ratio)
    return max(0, min(total_steps - 1, raw))


def _should_run_tier3_probe(step: int, total_steps: int, start_step: int, stride: int) -> bool:
    if step < start_step:
        return False
    safe_stride = max(1, stride)
    if (step - start_step) % safe_stride == 0:
        return True
    return step == total_steps - 1


def _resolve_sampling_start_step(total_steps: int, start_ratio: float) -> int:
    if total_steps <= 1:
        return 0
    raw = int(total_steps * start_ratio)
    return max(0, min(total_steps - 1, raw))


def _sample_masked_predictions(
    logits: torch.Tensor,
    current_mask_idx: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
) -> torch.Tensor:
    """
    仅对当前仍为 [MASK] 的位置进行 top-k 采样，其余位置保持 argmax。
    设计原因：
    1. G3 需要真正的 restart 多样性；
    2. 只在 masked 位置采样，尽量不破坏已确定 token；
    3. top-k + temperature 让随机性受控，而不是全词表无约束采样。
    """
    safe_temperature = max(float(temperature), 1e-5)
    probs = torch.softmax(logits / safe_temperature, dim=-1)
    max_probs, argmax_preds = torch.max(probs, dim=-1)
    sampled_preds = argmax_preds.clone()

    masked_positions = torch.nonzero(current_mask_idx[0], as_tuple=False).flatten().tolist()
    if not masked_positions:
        return max_probs, sampled_preds

    vocab_size = probs.shape[-1]
    safe_top_k = max(1, min(int(top_k), int(vocab_size)))
    for pos in masked_positions:
        row_probs = probs[0, pos]
        topk_probs, topk_indices = torch.topk(row_probs, safe_top_k, dim=-1)
        normalized = topk_probs / torch.clamp(topk_probs.sum(), min=1e-12)
        sampled_local = torch.multinomial(normalized, num_samples=1).item()
        sampled_token_id = int(topk_indices[sampled_local].item())
        sampled_preds[0, pos] = sampled_token_id
        max_probs[0, pos] = float(row_probs[sampled_token_id].item())

    return max_probs, sampled_preds


def _extract_tier_pass_tuple(verification: Dict[str, Dict[str, Any]]) -> Tuple[int, int, int]:
    tier1 = int(bool(verification.get("tier1_parse_compile", {}).get("passed", False)))
    tier2 = int(bool(verification.get("tier2_smoke_exec", {}).get("passed", False)))
    tier3 = int(bool(verification.get("tier3_unit_tests", {}).get("passed", False)))
    return tier1, tier2, tier3


def _run_proxy_verifier_stack(full_code: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    tier1 = tier1_parse_and_compile(full_code)
    results[tier1.tier] = tier1
    if not tier1.passed:
        return results

    tier2 = tier2_smoke_exec(full_code)
    results[tier2.tier] = tier2
    return results


def _run_single_token_decode(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
    *,
    final_verifier_mode: str = "full",
    stochastic_decode: bool = False,
) -> Dict[str, Any]:
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

    step_traces: List[Dict[str, Any]] = []
    decode_start = time.perf_counter()

    first_pass_step: Optional[int] = None
    first_pass_code: Optional[str] = None
    first_pass_middle_text: Optional[str] = None
    early_stopped = False

    inference_verifier_calls = 0
    inference_verifier_sec = 0.0
    probe_steps: List[int] = []
    start_step = _resolve_probe_start_step(total_steps, cfg.decode.inference_tier3_probe_start_ratio)
    sampling_start_step = _resolve_sampling_start_step(total_steps, cfg.decode.g3_sample_start_ratio)
    last_middle_mean_confidence: Optional[float] = None
    final_segments: Optional[Dict[str, str]] = None

    for step in range(total_steps):
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        current_mask_idx = (x_t == mask_token_id)
        if stochastic_decode and step >= sampling_start_step:
            max_probs, preds = _sample_masked_predictions(
                logits=logits,
                current_mask_idx=current_mask_idx,
                top_k=cfg.decode.g3_top_k,
                temperature=cfg.decode.g3_temperature,
            )
        else:
            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
        target_masks = linear_target_masks(num_mask_tokens, total_steps, step)

        if target_masks > 0:
            low_conf_indices = select_low_confidence_mask_positions(
                max_probs=max_probs[:, middle_start:middle_end],
                current_mask_idx=current_mask_idx[:, middle_start:middle_end],
                target_masks=target_masks,
            )
            x_t[current_mask_idx] = preds[current_mask_idx]
            x_t[0, [middle_start + idx for idx in low_conf_indices]] = mask_token_id
        else:
            x_t[current_mask_idx] = preds[current_mask_idx]
            low_conf_indices = []

        middle_conf = max_probs[:, middle_start:middle_end]
        last_middle_mean_confidence = float(middle_conf.mean().item())

        segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=x_t[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )
        final_segments = segments

        probe_result = None
        probe_called = False
        if cfg.decode.enable_inference_tier3_probe and _should_run_tier3_probe(
            step=step,
            total_steps=total_steps,
            start_step=start_step,
            stride=cfg.decode.inference_tier3_probe_stride,
        ):
            probe_called = True
            probe_steps.append(step)
            probe_start = time.perf_counter()
            probe_result = tier3_unit_tests(task, segments["middle_text"], timeout=cfg.verifier.timeout)
            inference_verifier_sec += time.perf_counter() - probe_start
            inference_verifier_calls += 1

            if probe_result.passed and first_pass_step is None:
                first_pass_step = step
                first_pass_code = segments["full_text"]
                first_pass_middle_text = segments["middle_text"]

            if probe_result.passed and cfg.decode.early_stop_on_tier3_pass:
                early_stopped = True
                step_traces.append(
                    {
                        "task_id": task.task_id,
                        "step": step,
                        "g_control_mode": cfg.decode.g_control_mode,
                        "target_masks": target_masks,
                        "remaining_masks": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
                        "low_confidence_indices": low_conf_indices,
                        "verifier_called": True,
                        "verifier_tier": "tier3_unit_tests",
                        "inference_probe": probe_result.to_dict(),
                        "full_text": segments["full_text"],
                        "middle_text": segments["middle_text"],
                        "reconstruction_diagnostics": build_reconstruction_diagnostics(task, segments),
                        "early_stop_triggered": True,
                    }
                )
                break

        step_trace = {
            "task_id": task.task_id,
            "step": step,
            "g_control_mode": cfg.decode.g_control_mode,
            "target_masks": target_masks,
            "remaining_masks": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
            "low_confidence_indices": low_conf_indices,
            "verifier_called": probe_called,
            "verifier_tier": "tier3_unit_tests" if probe_called else None,
            "inference_probe": probe_result.to_dict() if probe_result is not None else None,
            "full_text": segments["full_text"],
            "middle_text": segments["middle_text"],
            "reconstruction_diagnostics": build_reconstruction_diagnostics(task, segments),
            "early_stop_triggered": False,
        }
        step_traces.append(step_trace)

    total_decode_sec = time.perf_counter() - decode_start
    if final_segments is None:
        final_segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=x_t[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )

    if final_verifier_mode == "proxy":
        final_verification = _run_proxy_verifier_stack(final_segments["full_text"])
    elif final_verifier_mode == "full":
        final_verification = run_verifier_stack(
            task=task,
            full_code=final_segments["full_text"],
            completion_without_suffix=final_segments["middle_text"],
            timeout=cfg.verifier.timeout,
        )
    else:
        raise ValueError(f"Unsupported final_verifier_mode: {final_verifier_mode}")
    final_verifier_sec = sum(v.duration_sec for v in final_verification.values())
    final_verifier_calls = len(final_verification)

    final_tier3 = final_verification.get("tier3_unit_tests")
    final_passed = bool(final_tier3.passed) if final_tier3 is not None else False
    backslide = False
    rescue = False
    wasted_steps_after_first_pass = None
    if first_pass_step is not None:
        # 旧实现只在 final_tier3 存在时才把样本记为 backslide。
        # 删除原因：
        # 1. 若样本中途 Tier3 已通过，但最终已经退化到 Tier1/Tier2 失败，final_tier3 会缺失；
        # 2. 这种“先过测试、后退化到语法/执行失败”的情况本质上仍是 backslide；
        # 3. 之前的统计会系统性低估 early stop 的价值。
        backslide = not final_passed and (not early_stopped)
        wasted_steps_after_first_pass = (total_steps - 1 - first_pass_step) if not early_stopped else 0
    elif final_passed:
        rescue = False

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
            "rescue": rescue,
            "wasted_steps_after_first_pass": wasted_steps_after_first_pass,
            "inference_verifier_calls": inference_verifier_calls,
            "inference_verifier_sec": inference_verifier_sec,
            "final_verifier_calls": final_verifier_calls,
            "final_verifier_sec": final_verifier_sec,
            "verifier_calls_total": inference_verifier_calls + final_verifier_calls,
            "verifier_total_sec": inference_verifier_sec + final_verifier_sec,
            "mean_final_confidence": last_middle_mean_confidence,
            "g_control_mode": cfg.decode.g_control_mode,
        },
        "verification": {k: v.to_dict() for k, v in final_verification.items()},
        "diagnostics": {
            **build_reconstruction_diagnostics(task, final_segments),
            "g_control_mode": cfg.decode.g_control_mode,
            "probe_start_step": start_step,
            "probe_steps": probe_steps,
            "first_pass_middle_text": first_pass_middle_text,
            "first_pass_code": first_pass_code,
            "stochastic_decode": stochastic_decode,
            "g3_sampling_start_step": sampling_start_step if stochastic_decode else None,
            "g3_top_k": cfg.decode.g3_top_k if stochastic_decode else None,
            "g3_temperature": cfg.decode.g3_temperature if stochastic_decode else None,
        },
        "prepared": prepared,
    }


def run_gseries_decode(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    return _run_single_token_decode(task, tokenizer, model, cfg)


def _candidate_proxy_rank(candidate_result: Dict[str, Any], selection_rule: str) -> Tuple[Any, ...]:
    verification = candidate_result.get("verification", {})
    tier1_pass, tier2_pass, tier3_pass = _extract_tier_pass_tuple(verification)
    mean_final_confidence = float(candidate_result.get("metrics", {}).get("mean_final_confidence") or 0.0)
    decode_sec = float(candidate_result.get("metrics", {}).get("decode_sec") or 0.0)

    if selection_rule == "tier3":
        return (tier3_pass, tier2_pass, tier1_pass, mean_final_confidence, -decode_sec)

    # proxy 规则严格不依赖 Tier 3 通过与否，只使用 Tier1/Tier2 和模型置信度。
    return (tier2_pass, tier1_pass, mean_final_confidence, -decode_sec)


def run_g3_restart_decode(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    # 旧实现不存在 G3；这里新增 restart-style vanilla 控制组。
    num_restarts = max(1, int(cfg.decode.g3_num_restarts))
    seed_stride = max(1, int(cfg.decode.g3_seed_stride))
    selection_rule = cfg.decode.g3_selection_rule

    all_candidates: List[Dict[str, Any]] = []
    for restart_idx in range(num_restarts):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.decode.enable_inference_tier3_probe = False
        local_cfg.decode.early_stop_on_tier3_pass = False
        local_cfg.decode.g_control_mode = "G3"
        local_cfg.decode.seed = int(cfg.decode.seed) + restart_idx * seed_stride
        torch.manual_seed(local_cfg.decode.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_cfg.decode.seed)
        candidate = _run_single_token_decode(
            task,
            tokenizer,
            model,
            local_cfg,
            final_verifier_mode="proxy" if selection_rule == "proxy" else "full",
            stochastic_decode=bool(local_cfg.decode.g3_do_sample),
        )
        candidate["restart_index"] = restart_idx
        candidate["restart_seed"] = local_cfg.decode.seed
        all_candidates.append(candidate)

    ranked = sorted(
        all_candidates,
        key=lambda x: _candidate_proxy_rank(x, selection_rule),
        reverse=True,
    )
    selected = copy.deepcopy(ranked[0])

    # 旧实现只保留“被选中候选”的 total_sec，
    # 删除原因：G3 的真实成本应当包括所有 restart 候选的生成与筛选成本，
    # 否则 summary 会系统性低估 best-of-k search 的实际代价。
    aggregate_decode_sec = sum(float(cand["metrics"].get("decode_sec", 0.0)) for cand in all_candidates)
    aggregate_inference_verifier_sec = sum(float(cand["metrics"].get("inference_verifier_sec", 0.0)) for cand in all_candidates)
    aggregate_verifier_total_sec = sum(float(cand["metrics"].get("verifier_total_sec", 0.0)) for cand in all_candidates)
    aggregate_total_sec = sum(float(cand["metrics"].get("total_sec", 0.0)) for cand in all_candidates)
    aggregate_verifier_calls_total = sum(int(cand["metrics"].get("verifier_calls_total", 0)) for cand in all_candidates)

    if selection_rule == "proxy":
        # 新增：proxy 选择阶段不使用 Tier 3；选中后再补一次官方最终评测，避免把官方测试提前用于候选排序。
        selected_full_verification = run_verifier_stack(
            task=task,
            full_code=selected["code"],
            completion_without_suffix=selected["middle_text"],
            timeout=cfg.verifier.timeout,
        )
        selected_final_verifier_sec = sum(v.duration_sec for v in selected_full_verification.values())
        selected_final_verifier_calls = len(selected_full_verification)
        selected["verification"] = {k: v.to_dict() for k, v in selected_full_verification.items()}
        selected_tier3 = selected_full_verification.get("tier3_unit_tests")
        selected["metrics"]["passed"] = bool(selected_tier3.passed) if selected_tier3 is not None else False
        aggregate_verifier_total_sec += selected_final_verifier_sec
        aggregate_total_sec += selected_final_verifier_sec
        aggregate_verifier_calls_total += selected_final_verifier_calls
        selected["metrics"]["final_verifier_calls"] = selected_final_verifier_calls
        selected["metrics"]["final_verifier_sec"] = selected_final_verifier_sec
    else:
        selected["metrics"]["final_verifier_calls"] = int(selected["metrics"].get("final_verifier_calls", 0))
        selected["metrics"]["final_verifier_sec"] = float(selected["metrics"].get("final_verifier_sec", 0.0))

    selected["metrics"]["decode_sec"] = aggregate_decode_sec
    selected["metrics"]["inference_verifier_sec"] = aggregate_inference_verifier_sec
    selected["metrics"]["verification_sec"] = aggregate_verifier_total_sec
    selected["metrics"]["verifier_total_sec"] = aggregate_verifier_total_sec
    selected["metrics"]["verifier_calls_total"] = aggregate_verifier_calls_total
    selected["metrics"]["total_sec"] = aggregate_total_sec

    selected["metrics"]["g_control_mode"] = "G3"
    unique_candidate_codes = len({cand["code"] for cand in all_candidates})
    selected["diagnostics"]["g3_selection_rule"] = selection_rule
    selected["diagnostics"]["g3_num_restarts"] = num_restarts
    selected["diagnostics"]["g3_unique_candidate_codes"] = unique_candidate_codes
    selected["diagnostics"]["g3_candidates"] = [
        {
            "restart_index": cand["restart_index"],
            "restart_seed": cand["restart_seed"],
            "proxy_rank": _candidate_proxy_rank(cand, selection_rule),
            "passed": cand["metrics"]["passed"],
            "mean_final_confidence": cand["metrics"].get("mean_final_confidence"),
            "decode_sec": cand["metrics"].get("decode_sec"),
            "total_sec": cand["metrics"].get("total_sec"),
            "verification": cand["verification"],
        }
        for cand in ranked
    ]
    selected["metrics"]["g3_num_restarts"] = num_restarts
    selected["metrics"]["g3_selection_rule"] = selection_rule
    selected["metrics"]["g3_unique_candidate_codes"] = unique_candidate_codes
    return selected
