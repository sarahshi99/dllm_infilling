from __future__ import annotations

import time
from typing import Any, Dict, List

import torch

from .config import ExperimentConfig
from .data_loader import CodeTask
from .model_utils import resolve_mask_token_id
from .policies import linear_target_masks, select_low_confidence_mask_positions
from .verifier import parse_compile_diagnostics, run_verifier_stack


def _segment_decode(tokenizer, prefix_ids: List[int], middle_ids: List[int], suffix_ids: List[int]) -> Dict[str, str]:
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    middle_text = tokenizer.decode(middle_ids, skip_special_tokens=True)
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    return {
        "prefix_text": prefix_text,
        "middle_text": middle_text,
        "suffix_text": suffix_text,
        "full_text": prefix_text + middle_text + suffix_text,
    }


# 新增：把 reconstruction 与 verifier 物理输入的关系写进日志，后面核查不用再靠离线推断。
def build_reconstruction_diagnostics(task: CodeTask, segments: Dict[str, str]) -> Dict[str, Any]:
    full_text = segments["full_text"]
    middle_text = segments["middle_text"]
    expected_full = task.prefix + middle_text + task.suffix
    full_diag = parse_compile_diagnostics(full_text, compile_mode="exec")
    harness_diag = parse_compile_diagnostics(
        task.prefix + middle_text + task.suffix + "\n" + task.test_code + "\n" + f"check({task.entry_point})",
        compile_mode="exec",
    )
    return {
        "task_prefix_equals_decoded_prefix": task.prefix == segments["prefix_text"],
        "task_suffix_equals_decoded_suffix": task.suffix == segments["suffix_text"],
        "full_equals_task_reconstruction": full_text == expected_full,
        "tier3_problem": {
            "prompt": task.prefix,
            "suffix": task.suffix,
            "entry_point": task.entry_point,
        },
        "tier3_completion": middle_text,
        "final_full_code_parse_passed": full_diag["parse_passed"],
        "final_full_code_compile_passed": full_diag["compile_passed"],
        "final_full_code_parse_error": full_diag["parse_error"],
        "final_full_code_compile_error": full_diag["compile_error"],
        "expected_harness_program_parse_passed": harness_diag["parse_passed"],
        "expected_harness_program_compile_passed": harness_diag["compile_passed"],
        "expected_harness_program_parse_error": harness_diag["parse_error"],
        "expected_harness_program_compile_error": harness_diag["compile_error"],
    }


def prepare_model_inputs(task: CodeTask, tokenizer, cfg: ExperimentConfig) -> Dict[str, Any]:
    mask_token_id = resolve_mask_token_id(tokenizer)
    prefix_ids = tokenizer.encode(task.prefix, add_special_tokens=cfg.decode.add_special_tokens_to_prefix)
    suffix_ids = tokenizer.encode(task.suffix, add_special_tokens=cfg.decode.add_special_tokens_to_suffix)
    input_ids = prefix_ids + [mask_token_id] * cfg.decode.num_mask_tokens + suffix_ids
    return {
        "mask_token_id": mask_token_id,
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids,
        "input_ids": input_ids,
        "middle_start": len(prefix_ids),
        "middle_end": len(prefix_ids) + cfg.decode.num_mask_tokens,
    }


def run_vanilla_decode(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
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

    for step in range(total_steps):
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)
        current_mask_idx = (x_t == mask_token_id)
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

        segments = _segment_decode(
            tokenizer=tokenizer,
            prefix_ids=prepared["prefix_ids"],
            middle_ids=x_t[0, middle_start:middle_end].tolist(),
            suffix_ids=prepared["suffix_ids"],
        )
        step_trace = {
            "task_id": task.task_id,
            "step": step,
            "target_masks": target_masks,
            "remaining_masks": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
            "low_confidence_indices": low_conf_indices,
            "verifier_called": False,
            "full_text": segments["full_text"],
            "middle_text": segments["middle_text"],
            "reconstruction_diagnostics": build_reconstruction_diagnostics(task, segments),
        }
        step_traces.append(step_trace)

    total_decode_sec = time.perf_counter() - decode_start
    final_segments = _segment_decode(
        tokenizer=tokenizer,
        prefix_ids=prepared["prefix_ids"],
        middle_ids=x_t[0, middle_start:middle_end].tolist(),
        suffix_ids=prepared["suffix_ids"],
    )

    # 旧实现（保留对照，不再直接使用）：
    # final_verification = tier3_unit_tests(task, final_segments["full_text"], timeout=cfg.verifier.timeout)
    #
    # 删除原因：
    # 这会把 full_text 当成 completion 喂给标准 HumanEval harness，
    # 对 infilling 任务是错位的。

    final_verification = run_verifier_stack(
        task=task,
        full_code=final_segments["full_text"],
        completion_without_suffix=final_segments["middle_text"],
        timeout=cfg.verifier.timeout,
    )

    return {
        "task_id": task.task_id,
        "task": task,
        "code": final_segments["full_text"],
        "prefix_text": final_segments["prefix_text"],
        "middle_text": final_segments["middle_text"],
        "suffix_text": final_segments["suffix_text"],
        "step_traces": step_traces,
        "metrics": {
            "passed": final_verification.get("tier3_unit_tests").passed if final_verification.get("tier3_unit_tests") else False,
            "decode_sec": total_decode_sec,
            "verification_sec": sum(v.duration_sec for v in final_verification.values()),
            "total_sec": total_decode_sec + sum(v.duration_sec for v in final_verification.values()),
            "early_stop": False,
            "first_pass_step": None,
            "final_pass_step": total_steps - 1,
            "backslide": False,
            "rescue": False,
            "wasted_steps_after_first_pass": None,
        },
        "verification": {k: v.to_dict() for k, v in final_verification.items()},
        "diagnostics": build_reconstruction_diagnostics(task, final_segments),
        "prepared": prepared,
    }