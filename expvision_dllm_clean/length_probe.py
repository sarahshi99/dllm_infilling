from __future__ import annotations

import time
import math
from typing import Any, Dict, List

import torch

from .config import ExperimentConfig
from .dataset import CodeTask
from .modeling import resolve_mask_token_id


def parse_probe_lengths(csv_text: str) -> List[int]:
    if not isinstance(csv_text, str) or not csv_text.strip():
        raise ValueError("cal_lite_probe_lengths_csv must be a non-empty string")

    values: List[int] = []
    seen = set()

    for raw in csv_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError(f"Probe mask lengths must be positive, got {value}")
        if value not in seen:
            seen.add(value)
            values.append(value)

    if not values:
        raise ValueError("No valid probe lengths were parsed")

    return values


def _resolve_model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _build_probe_input(task: CodeTask, tokenizer, cfg: ExperimentConfig, mask_length: int) -> Dict[str, Any]:
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


def adjust_length_probe_score(raw_score: float, mask_length: int, cfg: ExperimentConfig) -> float:
    score_mode = cfg.decode.cal_lite_score_mode
    alpha = float(cfg.decode.cal_lite_length_alpha)

    if score_mode == "raw":
        return float(raw_score)

    if score_mode == "length_power":
        return float(raw_score) * (float(mask_length) ** alpha)

    if score_mode == "length_power_proportional":
        ref_length = max(float(getattr(cfg.decode, "cal_lite_length_prop_ref_length", 12.0)), 1.0)
        cap_length = getattr(cfg.decode, "cal_lite_length_prop_cap_length", None)
        effective_length = float(mask_length)
        if cap_length is not None:
            effective_length = min(effective_length, max(float(cap_length), 1.0))
        ratio = max(effective_length / ref_length, 1.0)
        beta = float(getattr(cfg.decode, "cal_lite_length_prop_beta", 0.0))
        effective_alpha = alpha + beta * math.log(ratio)
        return float(raw_score) * (float(mask_length) ** effective_alpha)

    raise ValueError(f"Unsupported cal_lite_score_mode: {score_mode}")


def probe_mask_length_score(task: CodeTask, tokenizer, model, cfg: ExperimentConfig, mask_length: int) -> Dict[str, Any]:
    prepared = _build_probe_input(task, tokenizer, cfg, mask_length)
    device = _resolve_model_device(model)

    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(x_t)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    probs = torch.softmax(logits, dim=-1)
    middle_probs = probs[:, prepared["middle_start"] : prepared["middle_end"], :]

    max_probs, _ = torch.max(middle_probs, dim=-1)
    top2_values, _ = torch.topk(middle_probs, k=2, dim=-1)

    raw_score = float(max_probs.mean().item())
    adjusted_score = adjust_length_probe_score(
        raw_score=raw_score,
        mask_length=mask_length,
        cfg=cfg,
    )
    mean_top2_gap = float((top2_values[..., 0] - top2_values[..., 1]).mean().item())
    probe_sec = time.perf_counter() - start

    return {
        "mask_length": int(mask_length),
        "score": adjusted_score,
        "raw_score": raw_score,
        "adjusted_score": adjusted_score,
        "mean_top1_prob": raw_score,
        "mean_top2_gap": mean_top2_gap,
        "score_mode": cfg.decode.cal_lite_score_mode,
        "length_alpha": float(cfg.decode.cal_lite_length_alpha),
        "length_prop_beta": float(getattr(cfg.decode, "cal_lite_length_prop_beta", 0.0)),
        "length_prop_ref_length": float(getattr(cfg.decode, "cal_lite_length_prop_ref_length", 12.0)),
        "length_prop_cap_length": getattr(cfg.decode, "cal_lite_length_prop_cap_length", None),
        "probe_sec": probe_sec,
    }


def _pick_best_candidate(candidates: List[Dict[str, Any]], tie_break: str) -> Dict[str, Any]:
    if tie_break not in {"shorter", "longer"}:
        raise ValueError(f"Unsupported cal_lite_tie_break: {tie_break}")

    if tie_break == "shorter":
        candidates = sorted(candidates, key=lambda item: (-float(item["score"]), int(item["mask_length"])))
    else:
        candidates = sorted(candidates, key=lambda item: (-float(item["score"]), -int(item["mask_length"])))

    return candidates[0]


def select_mask_length_cal_lite(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    probe_lengths = parse_probe_lengths(cfg.decode.cal_lite_probe_lengths_csv)

    candidate_scores: List[Dict[str, Any]] = []
    total_probe_sec = 0.0

    for mask_length in probe_lengths:
        result = probe_mask_length_score(task, tokenizer, model, cfg, mask_length)
        candidate_scores.append(result)
        total_probe_sec += float(result["probe_sec"])

    best = _pick_best_candidate(candidate_scores, cfg.decode.cal_lite_tie_break)

    return {
        "selected_mask_length": int(best["mask_length"]),
        "selected_score": float(best["score"]),
        "selected_raw_score": float(best["raw_score"]),
        "selected_adjusted_score": float(best["adjusted_score"]),
        "candidate_scores": candidate_scores,
        "length_probe_sec": total_probe_sec,
        "probe_lengths": probe_lengths,
        "tie_break": cfg.decode.cal_lite_tie_break,
        "score_mode": cfg.decode.cal_lite_score_mode,
        "length_alpha": float(cfg.decode.cal_lite_length_alpha),
    }
