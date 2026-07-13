#!/usr/bin/env python3
"""State-capturing fixed-canvas decoding used by the Phase 6 M1 protocol.

This module deliberately contains no selection logic.  It executes a fixed
canvas decode from either an all-mask canvas or a supplied candidate state and
returns only the generated state plus post-generation evaluator results.
"""

from __future__ import annotations

import io
import re
import time
import tokenize
from typing import Any, Mapping, Sequence

import torch

from expvision_dllm_clean.dataset import CodeTask
from expvision_dllm_clean.decode import (
    linear_target_masks,
    prepare_model_inputs,
    select_low_confidence_mask_positions,
)
from expvision_dllm_clean.verifier import parse_compile_diagnostics, run_verifier_stack


def remask_count_for_canvas(canvas_tokens: int) -> int:
    """Use a small, fixed remask budget for both generic and M1 refinement."""
    return max(1, min(4, (int(canvas_tokens) + 9) // 10))


def _segments(tokenizer: Any, prepared: Mapping[str, Any], middle_ids: Sequence[int]) -> dict[str, str]:
    prefix = tokenizer.decode(prepared["prefix_ids"], skip_special_tokens=True)
    middle = tokenizer.decode(list(middle_ids), skip_special_tokens=True)
    suffix = tokenizer.decode(prepared["suffix_ids"], skip_special_tokens=True)
    return {
        "prefix_text": prefix,
        "middle_text": middle,
        "suffix_text": suffix,
        "full_text": prefix + middle + suffix,
    }


def _diagnostics(task: CodeTask, segments: Mapping[str, str]) -> dict[str, Any]:
    parsed = parse_compile_diagnostics(str(segments["full_text"]), compile_mode="exec")
    return {
        "task_prefix_equals_decoded_prefix": task.prefix == segments["prefix_text"],
        "task_suffix_equals_decoded_suffix": task.suffix == segments["suffix_text"],
        "final_full_code_parse_passed": parsed["parse_passed"],
        "final_full_code_compile_passed": parsed["compile_passed"],
        "final_full_code_parse_error": parsed["parse_error"],
        "final_full_code_compile_error": parsed["compile_error"],
    }


def decode_fixed_canvas_state(
    *,
    task: CodeTask,
    tokenizer: Any,
    model: Any,
    cfg: Any,
    canvas_tokens: int,
    total_steps: int,
    phase_name: str,
    initial_middle_ids: Sequence[int] | None = None,
    initial_mask_indices: Sequence[int] | None = None,
    schedule_length: int | None = None,
) -> dict[str, Any]:
    """Run a fixed 64-step decode and retain the final token state locally.

    ``initial_middle_ids`` makes stage two a real refinement: only the supplied
    positions are remasked, and all other first-stage tokens are preserved as
    the initial state.  The verifier runs only after decoding has completed.
    """
    if int(canvas_tokens) <= 0 or int(total_steps) <= 0:
        raise ValueError("canvas_tokens and total_steps must be positive")
    prepared = prepare_model_inputs(task, tokenizer, int(canvas_tokens), cfg)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    middle_start = int(prepared["middle_start"])
    middle_end = int(prepared["middle_end"])
    mask_token_id = int(prepared["mask_token_id"])
    remask_indices = sorted({int(index) for index in initial_mask_indices or []})
    if any(index < 0 or index >= int(canvas_tokens) for index in remask_indices):
        raise ValueError("initial_mask_indices contain an out-of-canvas position")
    if initial_middle_ids is not None:
        if len(initial_middle_ids) != int(canvas_tokens):
            raise ValueError("initial_middle_ids length must equal canvas_tokens")
        x_t[0, middle_start:middle_end] = torch.tensor(list(initial_middle_ids), dtype=torch.long, device=device)
        if remask_indices:
            x_t[0, [middle_start + index for index in remask_indices]] = mask_token_id

    selected_schedule_length = int(schedule_length or canvas_tokens)
    if selected_schedule_length <= 0:
        raise ValueError("schedule_length must be positive")
    decode_started = time.perf_counter()
    total_token_changes = 0
    effective_update_steps = 0
    final_confidences: list[float] = [0.0] * int(canvas_tokens)
    step_trace: list[dict[str, Any]] = []

    for step in range(int(total_steps)):
        before_middle = x_t[0, middle_start:middle_end].detach().clone()
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probabilities = torch.softmax(logits, dim=-1)
        max_probs, predictions = torch.max(probabilities, dim=-1)
        current_mask_idx = x_t == mask_token_id
        middle_probs = max_probs[:, middle_start:middle_end]
        middle_mask_idx = current_mask_idx[:, middle_start:middle_end]
        target_masks = linear_target_masks(selected_schedule_length, int(total_steps), step)
        if target_masks > 0:
            low_confidence = select_low_confidence_mask_positions(middle_probs, middle_mask_idx, target_masks)
            x_t[current_mask_idx] = predictions[current_mask_idx]
            if low_confidence:
                x_t[0, [middle_start + index for index in low_confidence]] = mask_token_id
        else:
            low_confidence = []
            x_t[current_mask_idx] = predictions[current_mask_idx]
        after_middle = x_t[0, middle_start:middle_end].detach().clone()
        token_changes = int((after_middle != before_middle).sum().item())
        if token_changes:
            effective_update_steps += 1
            total_token_changes += token_changes
        final_confidences = [float(value) for value in middle_probs[0].detach().cpu().tolist()]
        step_trace.append(
            {
                "phase": phase_name,
                "step": step,
                "target_masks": target_masks,
                "remaining_masks_before": int(middle_mask_idx.sum().item()),
                "remaining_masks_after": int((after_middle == mask_token_id).sum().item()),
                "token_change_count": token_changes,
                "low_confidence_indices": [int(index) for index in low_confidence],
            }
        )

    decode_sec = time.perf_counter() - decode_started
    final_middle_ids = [int(value) for value in x_t[0, middle_start:middle_end].detach().cpu().tolist()]
    segments = _segments(tokenizer, prepared, final_middle_ids)
    verification = run_verifier_stack(
        task=task,
        full_code=str(segments["full_text"]),
        completion_without_suffix=str(segments["middle_text"]),
    )
    verification_sec = sum(item.duration_sec for item in verification.values())
    tier3 = verification.get("tier3_unit_tests")
    passed = bool(tier3.passed) if tier3 else False
    return {
        "code": segments["full_text"],
        "prefix_text": segments["prefix_text"],
        "middle_text": segments["middle_text"],
        "suffix_text": segments["suffix_text"],
        "middle_token_ids": final_middle_ids,
        "final_token_confidences": final_confidences,
        "metrics": {
            "passed": passed,
            "decode_sec": decode_sec,
            "verification_sec": verification_sec,
            "total_sec": decode_sec + verification_sec,
            "total_sec_including_probe": decode_sec + verification_sec,
            "total_steps": int(total_steps),
            "actual_forward_count": int(total_steps),
            "effective_update_steps": effective_update_steps,
            "total_token_changes": total_token_changes,
            "mean_final_confidence": sum(final_confidences) / len(final_confidences) if final_confidences else 0.0,
            "canvas_tokens": int(canvas_tokens),
            "remasked_initial_token_count": len(remask_indices),
        },
        "verification": {name: result.to_dict() for name, result in verification.items()},
        "diagnostics": _diagnostics(task, segments),
        "trajectory": {
            "phase": phase_name,
            "step_trace": step_trace,
        },
    }


def generic_low_confidence_indices(confidences: Sequence[float], canvas_tokens: int) -> list[int]:
    if len(confidences) != int(canvas_tokens):
        raise ValueError("generic remask requires one confidence value per canvas token")
    count = remask_count_for_canvas(int(canvas_tokens))
    return sorted(
        sorted(range(int(canvas_tokens)), key=lambda index: (float(confidences[index]), index))[:count]
    )


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\\n", text):
        offsets.append(match.end())
    return offsets


def identifier_character_spans(text: str, names: Sequence[str]) -> list[tuple[int, int, str]]:
    wanted = {str(name) for name in names}
    if not wanted:
        return []
    try:
        offsets = _line_offsets(text)
        spans: list[tuple[int, int, str]] = []
        for item in tokenize.generate_tokens(io.StringIO(text).readline):
            if item.type != tokenize.NAME or item.string not in wanted:
                continue
            start = offsets[item.start[0] - 1] + item.start[1]
            end = offsets[item.end[0] - 1] + item.end[1]
            spans.append((start, end, item.string))
        return spans
    except (tokenize.TokenError, IndentationError):
        pattern = re.compile(r"\\b(" + "|".join(re.escape(name) for name in sorted(wanted, key=len, reverse=True)) + r")\\b")
        return [(match.start(), match.end(), match.group(1)) for match in pattern.finditer(text)]


def decoded_token_ranges(tokenizer: Any, token_ids: Sequence[int]) -> tuple[str, list[tuple[int, int]]]:
    """Map decoded character intervals back to token positions without source code."""
    previous = ""
    ranges: list[tuple[int, int]] = []
    ids = [int(token_id) for token_id in token_ids]
    for index in range(len(ids)):
        current = tokenizer.decode(ids[: index + 1], skip_special_tokens=True)
        if current.startswith(previous):
            ranges.append((len(previous), len(current)))
        else:
            # Tokenizer cleanup can rarely break prefix monotonicity.  Mark this
            # token as unmappable rather than inventing a target span.
            ranges.append((-1, -1))
        previous = current
    return previous, ranges


def dependency_cone_token_indices(
    tokenizer: Any,
    token_ids: Sequence[int],
    dependency_names: Sequence[str],
) -> list[int]:
    decoded, token_ranges = decoded_token_ranges(tokenizer, token_ids)
    spans = identifier_character_spans(decoded, dependency_names)
    if not spans:
        return []
    indices: list[int] = []
    for index, (left, right) in enumerate(token_ranges):
        if left < 0 or right <= left:
            continue
        if any(left < span_right and span_left < right for span_left, span_right, _ in spans):
            indices.append(index)
    return sorted(set(indices))
