from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from .config import ExperimentConfig
from .dataset import CodeTask, compute_oracle_mask_length, infer_reference_middle_text
from .length_probe import select_mask_length_cal_lite
from .modeling import resolve_mask_token_id
from .verifier import parse_compile_diagnostics, run_verifier_stack


def linear_target_masks(num_mask_tokens: int, total_steps: int, step: int) -> int:
    if total_steps <= 1:
        return 0
    remaining_ratio = max(0.0, 1.0 - (step + 1) / total_steps)
    return int(round(num_mask_tokens * remaining_ratio))


def select_low_confidence_mask_positions(
    max_probs: torch.Tensor,
    current_mask_idx: torch.Tensor,
    target_masks: int,
) -> List[int]:
    masked_positions = torch.nonzero(current_mask_idx[0], as_tuple=False).flatten().tolist()
    if target_masks <= 0 or not masked_positions:
        return []

    scores = [(float(max_probs[0, idx].item()), int(idx)) for idx in masked_positions]
    scores.sort(key=lambda item: (item[0], item[1]))
    return [idx for _, idx in scores[: min(target_masks, len(scores))]]


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


def build_reconstruction_diagnostics(task: CodeTask, segments: Dict[str, str]) -> Dict[str, Any]:
    full_text = segments["full_text"]
    middle_text = segments["middle_text"]
    full_diag = parse_compile_diagnostics(full_text, compile_mode="exec")

    return {
        "task_prefix_equals_decoded_prefix": task.prefix == segments["prefix_text"],
        "task_suffix_equals_decoded_suffix": task.suffix == segments["suffix_text"],
        "final_full_code_parse_passed": full_diag["parse_passed"],
        "final_full_code_compile_passed": full_diag["compile_passed"],
        "final_full_code_parse_error": full_diag["parse_error"],
        "final_full_code_compile_error": full_diag["compile_error"],
        "reference_middle_text": infer_reference_middle_text(task),
        "decoded_middle_text": middle_text,
    }


def prepare_model_inputs(task: CodeTask, tokenizer, mask_length: int, cfg: ExperimentConfig) -> Dict[str, Any]:
    if mask_length <= 0:
        raise ValueError(f"mask_length must be positive, got {mask_length}")

    mask_token_id = resolve_mask_token_id(tokenizer)
    prefix_ids = tokenizer.encode(task.prefix, add_special_tokens=cfg.decode.add_special_tokens_to_prefix)
    suffix_ids = tokenizer.encode(task.suffix, add_special_tokens=cfg.decode.add_special_tokens_to_suffix)
    input_ids = prefix_ids + [mask_token_id] * mask_length + suffix_ids

    return {
        "mask_token_id": mask_token_id,
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids,
        "input_ids": input_ids,
        "middle_start": len(prefix_ids),
        "middle_end": len(prefix_ids) + mask_length,
    }


def _length_diff(selected_length: Optional[int], oracle_length: Optional[int]) -> Dict[str, Optional[int]]:
    if selected_length is None or oracle_length is None:
        return {
            "selected_minus_oracle_length": None,
            "abs_selected_minus_oracle_length": None,
        }

    diff = int(selected_length) - int(oracle_length)
    return {
        "selected_minus_oracle_length": diff,
        "abs_selected_minus_oracle_length": abs(diff),
    }


def resolve_mask_length(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    oracle_mask_length = compute_oracle_mask_length(task, tokenizer, add_special_tokens=False)
    source = cfg.decode.mask_length_source

    base = {
        "oracle_mask_length": None if oracle_mask_length is None else int(oracle_mask_length),
        "mask_length_source": source,
        "selected_mask_length": None,
        "selected_score": None,
        "selected_raw_score": None,
        "selected_adjusted_score": None,
        "candidate_scores": None,
        "length_probe_sec": 0.0,
        "probe_lengths": None,
        "tie_break": None,
        "score_mode": None,
        "length_alpha": None,
    }

    if source == "fixed":
        mask_length = int(cfg.decode.fixed_mask_length)
        base["selected_mask_length"] = mask_length
        return {**base, "mask_length": mask_length, **_length_diff(mask_length, base["oracle_mask_length"])}

    if source == "oracle":
        if oracle_mask_length is None:
            raise ValueError(f"Oracle mask length unavailable for task {task.task_id}")
        mask_length = int(oracle_mask_length)
        base["selected_mask_length"] = mask_length
        return {**base, "mask_length": mask_length, **_length_diff(mask_length, base["oracle_mask_length"])}

    if source == "cal_lite":
        selection = select_mask_length_cal_lite(task, tokenizer, model, cfg)
        selected = int(selection["selected_mask_length"])
        return {
            **base,
            "mask_length": selected,
            "selected_mask_length": selected,
            "selected_score": float(selection["selected_score"]),
            "selected_raw_score": float(selection["selected_raw_score"]),
            "selected_adjusted_score": float(selection["selected_adjusted_score"]),
            "candidate_scores": selection["candidate_scores"],
            "length_probe_sec": float(selection["length_probe_sec"]),
            "probe_lengths": selection["probe_lengths"],
            "tie_break": selection["tie_break"],
            "score_mode": selection["score_mode"],
            "length_alpha": selection["length_alpha"],
            **_length_diff(selected, base["oracle_mask_length"]),
        }

    raise ValueError(f"Unsupported mask_length_source: {source}")


def run_vanilla_decode(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    length_meta = resolve_mask_length(task, tokenizer, model, cfg)
    prepared = prepare_model_inputs(task, tokenizer, length_meta["mask_length"], cfg)

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    mask_token_id = prepared["mask_token_id"]
    middle_start = prepared["middle_start"]
    middle_end = prepared["middle_end"]
    num_mask_tokens = length_meta["mask_length"]
    total_steps = cfg.decode.total_steps

    step_traces: List[Dict[str, Any]] = []
    decode_start = time.perf_counter()
    final_mean_confidence: Optional[float] = None

    for step in range(total_steps):
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)
        current_mask_idx = x_t == mask_token_id
        target_masks = linear_target_masks(num_mask_tokens, total_steps, step)

        middle_probs = max_probs[:, middle_start:middle_end]
        middle_mask_idx = current_mask_idx[:, middle_start:middle_end]

        if target_masks > 0:
            low_conf_indices = select_low_confidence_mask_positions(middle_probs, middle_mask_idx, target_masks)
            x_t[current_mask_idx] = preds[current_mask_idx]
            if low_conf_indices:
                x_t[0, [middle_start + idx for idx in low_conf_indices]] = mask_token_id
        else:
            x_t[current_mask_idx] = preds[current_mask_idx]
            low_conf_indices = []

        final_mean_confidence = float(middle_probs.mean().item())

        if cfg.decode.save_step_traces:
            trace: Dict[str, Any] = {
                "task_id": task.task_id,
                "step": step,
                "target_masks": target_masks,
                "remaining_masks": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
                "low_confidence_indices": low_conf_indices,
                "mean_confidence": final_mean_confidence,
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

    return {
        "task_id": task.task_id,
        "task": task,
        "code": final_segments["full_text"],
        "prefix_text": final_segments["prefix_text"],
        "middle_text": final_segments["middle_text"],
        "suffix_text": final_segments["suffix_text"],
        "step_traces": step_traces,
        "length_probe": {
            "candidate_scores": length_meta["candidate_scores"],
            "length_probe_sec": length_meta["length_probe_sec"],
            "selected_mask_length": length_meta["selected_mask_length"],
            "selected_score": length_meta["selected_score"],
            "selected_raw_score": length_meta["selected_raw_score"],
            "selected_adjusted_score": length_meta["selected_adjusted_score"],
            "selected_minus_oracle_length": length_meta["selected_minus_oracle_length"],
            "abs_selected_minus_oracle_length": length_meta["abs_selected_minus_oracle_length"],
            "probe_lengths": length_meta["probe_lengths"],
            "tie_break": length_meta["tie_break"],
            "score_mode": length_meta["score_mode"],
            "length_alpha": length_meta["length_alpha"],
        },
        "metrics": {
            "passed": passed,
            "decode_sec": total_decode_sec,
            "verification_sec": verification_sec,
            "total_sec": total_sec,
            "total_sec_including_probe": total_sec_including_probe,
            "length_probe_sec": float(length_meta["length_probe_sec"]),
            "total_steps": total_steps,
            "mean_final_confidence": final_mean_confidence,
            "mask_length": length_meta["mask_length"],
            "oracle_mask_length": length_meta["oracle_mask_length"],
            "selected_mask_length": length_meta["selected_mask_length"],
            "selected_score": length_meta["selected_score"],
            "selected_raw_score": length_meta["selected_raw_score"],
            "selected_adjusted_score": length_meta["selected_adjusted_score"],
            "selected_minus_oracle_length": length_meta["selected_minus_oracle_length"],
            "abs_selected_minus_oracle_length": length_meta["abs_selected_minus_oracle_length"],
            "mask_length_source": length_meta["mask_length_source"],
            "score_mode": length_meta["score_mode"],
            "length_alpha": length_meta["length_alpha"],
        },
        "verification": {key: value.to_dict() for key, value in final_verification.items()},
        "diagnostics": build_reconstruction_diagnostics(task, final_segments),
    }