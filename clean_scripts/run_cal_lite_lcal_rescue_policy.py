#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import (
    CodeTask,
    compute_oracle_mask_length,
    load_humaneval_infilling,
)
from expvision_dllm_clean.evaluation import summarize_results
from expvision_dllm_clean.length_probe import parse_probe_lengths, select_mask_length_cal_lite
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.runner_lcas_v3 import (
    annotate_lcas_v3_result,
    choose_lcas_v3_config,
    summarize_lcas_v3_results,
)

import expvision_dllm_clean.decode_lcas as decode_lcas


DEFAULT_COMPACT_GRID = "3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24"
DEFAULT_WEAK_GRID = "13,14,15,16"
DEFAULT_STRONG_GRID = "13,14,15,16,20,24,28,32,40"
DEFAULT_RATIO_TRIGGER_THRESHOLD = 0.97
DEFAULT_LONG_SCORE_FLOOR = 0.55
DEFAULT_RAW_RATIO_THRESHOLD = 0.97
DEFAULT_SUPPORT_COUNT_THRESHOLD = 2

_ACTIVE_SETTINGS: Optional["LcalV3Settings"] = None
_LCAL_META_BY_TASK_ID: Dict[str, Dict[str, Any]] = {}


@dataclass
class LcalV3Settings:
    base_probe_lengths_csv: str
    base_alpha: float
    weak_probe_lengths_csv: str
    strong_probe_lengths_csv: str
    long_alpha: float
    strong_min_len: int
    weak_min_base_len: int
    weak_max_base_len: int
    ratio_trigger_threshold: float
    long_score_floor: float
    raw_ratio_threshold: float
    support_count_threshold: int
    cap_base_le8: int
    cap_base_9_12: int
    short_safe_policy: str
    correction_selection_rule: str
    shortest_supported_ratio: float
    tie_break: str
    score_mode: str
    length_prop_beta: float = 0.0
    length_prop_ref_length: float = 12.0
    length_prop_cap_length: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LCAL rescue policies with LCAS-v3 stopping."
    )
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-probe-lengths", type=str, default=DEFAULT_COMPACT_GRID)
    parser.add_argument("--base-alpha", type=float, default=0.06)
    parser.add_argument("--weak-probe-lengths", type=str, default=DEFAULT_WEAK_GRID)
    parser.add_argument("--strong-probe-lengths", type=str, default=DEFAULT_STRONG_GRID)
    parser.add_argument("--long-alpha", type=float, default=0.10)
    parser.add_argument("--strong-min-len", type=int, default=13)
    parser.add_argument("--weak-min-base-len", type=int, default=8)
    parser.add_argument("--weak-max-base-len", type=int, default=12)
    parser.add_argument("--ratio-trigger-threshold", type=float, default=DEFAULT_RATIO_TRIGGER_THRESHOLD)
    parser.add_argument("--long-score-floor", type=float, default=DEFAULT_LONG_SCORE_FLOOR)
    parser.add_argument("--raw-ratio-threshold", type=float, default=DEFAULT_RAW_RATIO_THRESHOLD)
    parser.add_argument("--support-count-threshold", type=int, default=DEFAULT_SUPPORT_COUNT_THRESHOLD)
    parser.add_argument("--cap-base-le8", type=int, default=14)
    parser.add_argument("--cap-base-9-12", type=int, default=16)
    parser.add_argument(
        "--short-safe-policy",
        type=str,
        default="s3",
        choices=["s1", "s2", "s3", "s4"],
        help="S1=weak ratio, S2=weak ratio with jump cap, S3=raw-confirmed S2, S4=support-count S2.",
    )
    parser.add_argument("--tie-break", type=str, default="shorter", choices=["shorter", "longer"])
    parser.add_argument("--score-mode", type=str, default="length_power", choices=["raw", "length_power", "length_power_proportional"])
    parser.add_argument("--length-prop-beta", type=float, default=0.0)
    parser.add_argument("--length-prop-ref-length", type=float, default=12.0)
    parser.add_argument("--length-prop-cap-length", type=float, default=None)
    parser.add_argument(
        "--correction-selection-rule",
        type=str,
        default="shortest_supported",
        choices=["argmax", "shortest_supported"],
        help="How to select the correction length after a long/weak trigger.",
    )
    parser.add_argument(
        "--shortest-supported-ratio",
        type=float,
        default=0.985,
        help="For shortest_supported, choose the shortest candidate within this fraction of the best correction score.",
    )

    parser.add_argument("--lcas-policy", type=str, default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Optional baseline results.jsonl used for short-loss and long-win summary metrics.",
    )

    parser.add_argument("--output-dir", type=str, default="outputs_clean")
    parser.add_argument("--experiment-name", type=str, default="lcal_v3_clean")
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--save-full-text-per-step", action="store_true")
    return parser.parse_args()


def _avg(values: Iterable[float | int | None]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else None


def _rate(values: Iterable[bool]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _hist(values: Iterable[Any]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        counter[str(value)] += 1
    return dict(sorted(counter.items()))


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


def _oracle_bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    value = int(length)
    if value <= 8:
        return "<=8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    if value <= 24:
        return "17-24"
    return "25+"


def _clone_probe_cfg(
    cfg: ExperimentConfig,
    probe_lengths_csv: str,
    alpha: float,
    settings: LcalV3Settings,
) -> ExperimentConfig:
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg.decode.mask_length_source = "cal_lite"
    probe_cfg.decode.cal_lite_probe_lengths_csv = probe_lengths_csv
    probe_cfg.decode.cal_lite_length_alpha = float(alpha)
    probe_cfg.decode.cal_lite_tie_break = settings.tie_break
    probe_cfg.decode.cal_lite_score_mode = settings.score_mode
    probe_cfg.decode.cal_lite_length_prop_beta = float(settings.length_prop_beta)
    probe_cfg.decode.cal_lite_length_prop_ref_length = float(settings.length_prop_ref_length)
    probe_cfg.decode.cal_lite_length_prop_cap_length = settings.length_prop_cap_length
    return probe_cfg


def _candidate_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("score", candidate.get("adjusted_score", candidate.get("raw_score"))))


def _raw_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("raw_score", candidate.get("mean_top1_prob", candidate.get("score"))))


def _candidate_length(candidate: Dict[str, Any]) -> int:
    return int(candidate["mask_length"])


def _best_candidate(candidates: List[Dict[str, Any]], score_fn=_candidate_score) -> Dict[str, Any]:
    if _ACTIVE_SETTINGS is None:
        raise RuntimeError("LCAL-v3 settings were not initialized")
    if not candidates:
        raise ValueError("Cannot pick best candidate from an empty list")
    reverse_len_sign = -1 if _ACTIVE_SETTINGS.tie_break == "shorter" else 1
    return max(
        candidates,
        key=lambda item: (
            score_fn(item),
            reverse_len_sign * int(item["mask_length"]),
        ),
    )


def _select_correction_candidate(
    correction_selection: Dict[str, Any],
    base_selected_length: int,
    settings: LcalV3Settings,
) -> Dict[str, Any]:
    candidates = [
        item
        for item in correction_selection["candidate_scores"]
        if _candidate_length(item) >= max(int(base_selected_length), int(settings.strong_min_len))
    ]
    if not candidates:
        raise ValueError(
            "Correction probe produced no usable candidates at or above "
            f"max(base_len={base_selected_length}, strong_min_len={settings.strong_min_len})"
        )
    best = _best_candidate(candidates)
    if settings.correction_selection_rule == "argmax":
        return best
    if settings.correction_selection_rule != "shortest_supported":
        raise ValueError(f"Unsupported correction_selection_rule={settings.correction_selection_rule}")
    best_score = _candidate_score(best)
    supported = [
        item
        for item in candidates
        if _candidate_score(item) >= float(settings.shortest_supported_ratio) * best_score
    ]
    if not supported:
        return best
    return min(supported, key=_candidate_length)


def _apply_correction_selection(
    correction_selection: Dict[str, Any],
    base_selected_length: int,
    settings: LcalV3Settings,
) -> Dict[str, Any]:
    selected_candidate = _select_correction_candidate(correction_selection, base_selected_length, settings)
    adjusted = copy.deepcopy(correction_selection)
    adjusted["selected_mask_length"] = int(selected_candidate["mask_length"])
    adjusted["selected_score"] = float(selected_candidate.get("score", selected_candidate.get("adjusted_score")))
    adjusted["selected_raw_score"] = float(
        selected_candidate.get("raw_score", selected_candidate.get("mean_top1_prob"))
    )
    adjusted["selected_adjusted_score"] = float(
        selected_candidate.get("score", selected_candidate.get("adjusted_score", adjusted["selected_score"]))
    )
    adjusted["correction_selection_rule"] = settings.correction_selection_rule
    adjusted["shortest_supported_ratio"] = float(settings.shortest_supported_ratio)
    adjusted["correction_argmax_selected_mask_length"] = int(correction_selection["selected_mask_length"])
    return adjusted


def _base_best_info(base_selection: Dict[str, Any]) -> Dict[str, Any]:
    candidates = list(base_selection["candidate_scores"])
    best = _best_candidate(candidates)
    if _ACTIVE_SETTINGS is None:
        raise RuntimeError("LCAL-v3 settings were not initialized")
    settings = _ACTIVE_SETTINGS
    long_candidates = [item for item in candidates if int(item["mask_length"]) >= int(settings.strong_min_len)]
    best_long = _best_candidate(long_candidates)
    best_raw = _best_candidate(candidates, score_fn=_raw_score)
    best_long_raw = _best_candidate(long_candidates, score_fn=_raw_score)
    best_score = _candidate_score(best)
    best_long_score = _candidate_score(best_long)
    best_raw_score = _raw_score(best_raw)
    best_long_raw_score = _raw_score(best_long_raw)
    support_count = sum(
        1
        for item in long_candidates
        if _candidate_score(item) >= float(settings.ratio_trigger_threshold) * best_score
    )
    return {
        "best_len": int(best["mask_length"]),
        "best_score": best_score,
        "best_long_len": int(best_long["mask_length"]),
        "best_long_score": best_long_score,
        "long_ratio": best_long_score / best_score if best_score else None,
        "best_raw_len": int(best_raw["mask_length"]),
        "best_raw_score": best_raw_score,
        "best_long_raw_len": int(best_long_raw["mask_length"]),
        "best_long_raw_score": best_long_raw_score,
        "raw_long_ratio": best_long_raw_score / best_raw_score if best_raw_score else None,
        "support_count": int(support_count),
    }


def _jump_cap_for_base(base_len: int, settings: LcalV3Settings) -> Optional[int]:
    if base_len <= 8:
        return int(settings.cap_base_le8)
    if 9 <= base_len <= 12:
        return int(settings.cap_base_9_12)
    return None


def _effective_weak_grid(base_len: int, settings: LcalV3Settings) -> str:
    weak_lengths = parse_probe_lengths(settings.weak_probe_lengths_csv)
    if settings.short_safe_policy == "s1":
        return ",".join(str(length) for length in weak_lengths)
    cap = _jump_cap_for_base(base_len, settings)
    if cap is not None:
        weak_lengths = [length for length in weak_lengths if length <= cap]
    if not weak_lengths:
        raise ValueError(f"Empty weak grid after applying cap for base_len={base_len}")
    return ",".join(str(length) for length in weak_lengths)


def _trigger_decision(
    base_selected_length: int,
    best_info: Dict[str, Any],
    settings: LcalV3Settings,
) -> Dict[str, Any]:
    long_score_floor_passed = float(best_info["best_long_score"]) >= float(settings.long_score_floor)
    long_ratio = best_info["long_ratio"]
    ratio_passed = (
        long_ratio is not None
        and float(long_ratio) >= float(settings.ratio_trigger_threshold)
        and long_score_floor_passed
    )
    raw_ratio = best_info["raw_long_ratio"]
    raw_ratio_passed = raw_ratio is not None and float(raw_ratio) >= float(settings.raw_ratio_threshold)
    support_passed = int(best_info["support_count"]) >= int(settings.support_count_threshold)
    strong_trigger = int(base_selected_length) >= int(settings.strong_min_len)
    weak_window = int(settings.weak_min_base_len) <= int(base_selected_length) <= int(settings.weak_max_base_len)

    if strong_trigger:
        return {
            "triggered": True,
            "strong_triggered": True,
            "weak_triggered": False,
            "trigger_reason": "strong_base_len",
            "correction_kind": "strong_full",
            "correction_probe_lengths_csv": settings.strong_probe_lengths_csv,
            "long_score_floor_passed": bool(long_score_floor_passed),
            "ratio_passed": bool(ratio_passed),
            "raw_ratio_passed": bool(raw_ratio_passed),
            "support_passed": bool(support_passed),
            "weak_window": bool(weak_window),
        }

    weak_triggered = False
    trigger_reason = "none"
    if weak_window and ratio_passed:
        if settings.short_safe_policy == "s1":
            weak_triggered = True
            trigger_reason = "weak_ratio"
        elif settings.short_safe_policy == "s2":
            weak_triggered = True
            trigger_reason = "weak_ratio_jump_cap"
        elif settings.short_safe_policy == "s3" and raw_ratio_passed:
            weak_triggered = True
            trigger_reason = "weak_ratio_raw_confirmed_jump_cap"
        elif settings.short_safe_policy == "s4" and support_passed:
            weak_triggered = True
            trigger_reason = "weak_ratio_support_count_jump_cap"
        elif settings.short_safe_policy not in {"s1", "s2", "s3", "s4"}:
            raise ValueError(f"Unsupported short_safe_policy={settings.short_safe_policy}")

    return {
        "triggered": bool(weak_triggered),
        "strong_triggered": False,
        "weak_triggered": bool(weak_triggered),
        "trigger_reason": trigger_reason,
        "correction_kind": "weak_grid" if weak_triggered and settings.short_safe_policy == "s1" else (
            "weak_grid_capped" if weak_triggered else "none"
        ),
        "correction_probe_lengths_csv": _effective_weak_grid(base_selected_length, settings) if weak_triggered else None,
        "long_score_floor_passed": bool(long_score_floor_passed),
        "ratio_passed": bool(ratio_passed),
        "raw_ratio_passed": bool(raw_ratio_passed),
        "support_passed": bool(support_passed),
        "weak_window": bool(weak_window),
    }


def resolve_mask_length_lcal_v3(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    if _ACTIVE_SETTINGS is None:
        raise RuntimeError("LCAL-v3 settings were not initialized")

    settings = _ACTIVE_SETTINGS
    oracle_mask_length = compute_oracle_mask_length(task, tokenizer, add_special_tokens=False)

    base_cfg = _clone_probe_cfg(
        cfg=cfg,
        probe_lengths_csv=settings.base_probe_lengths_csv,
        alpha=settings.base_alpha,
        settings=settings,
    )
    base_selection = select_mask_length_cal_lite(task, tokenizer, model, base_cfg)
    base_selected = int(base_selection["selected_mask_length"])
    best_info = _base_best_info(base_selection)

    trigger = _trigger_decision(base_selected, best_info, settings)
    long_triggered = bool(trigger["triggered"])

    long_selection: Optional[Dict[str, Any]] = None
    final_source = "base"
    final_selection = base_selection
    total_probe_sec = float(base_selection["length_probe_sec"])

    if long_triggered:
        long_cfg = _clone_probe_cfg(
            cfg=cfg,
            probe_lengths_csv=str(trigger["correction_probe_lengths_csv"]),
            alpha=settings.long_alpha,
            settings=settings,
        )
        raw_long_selection = select_mask_length_cal_lite(task, tokenizer, model, long_cfg)
        long_selection = _apply_correction_selection(
            correction_selection=raw_long_selection,
            base_selected_length=base_selected,
            settings=settings,
        )
        long_selected = int(long_selection["selected_mask_length"])
        if long_selected >= base_selected:
            final_selection = long_selection
            final_source = "strong_correction" if trigger["strong_triggered"] else "weak_correction"
        else:
            final_selection = base_selection
            final_source = "base_max_after_long"
        total_probe_sec += float(long_selection["length_probe_sec"])

    selected = int(final_selection["selected_mask_length"])
    if long_selection is not None:
        selected = max(base_selected, int(long_selection["selected_mask_length"]))
    diff = _length_diff(selected, None if oracle_mask_length is None else int(oracle_mask_length))
    base_diff = _length_diff(base_selected, None if oracle_mask_length is None else int(oracle_mask_length))

    meta: Dict[str, Any] = {
        "version": "lcal_rescue_policy",
        "short_safe_policy": settings.short_safe_policy,
        "final_source": final_source,
        "long_triggered": long_triggered,
        "strong_triggered": bool(trigger["strong_triggered"]),
        "weak_triggered": bool(trigger["weak_triggered"]),
        "correction_kind": trigger["correction_kind"],
        "correction_selection_rule": settings.correction_selection_rule,
        "shortest_supported_ratio": float(settings.shortest_supported_ratio),
        "correction_probe_lengths": (
            None
            if trigger["correction_probe_lengths_csv"] is None
            else parse_probe_lengths(str(trigger["correction_probe_lengths_csv"]))
        ),
        "long_trigger_reasons": [trigger["trigger_reason"]] if long_triggered else [],
        "trigger_reason": str(trigger["trigger_reason"]),
        "best_score": float(best_info["best_score"]),
        "best_len": int(best_info["best_len"]),
        "best_long_score": float(best_info["best_long_score"]),
        "best_long_len": int(best_info["best_long_len"]),
        "long_ratio": None if best_info["long_ratio"] is None else float(best_info["long_ratio"]),
        "best_raw_score": float(best_info["best_raw_score"]),
        "best_raw_len": int(best_info["best_raw_len"]),
        "best_long_raw_score": float(best_info["best_long_raw_score"]),
        "best_long_raw_len": int(best_info["best_long_raw_len"]),
        "raw_long_ratio": None if best_info["raw_long_ratio"] is None else float(best_info["raw_long_ratio"]),
        "long_score_floor_passed": bool(trigger["long_score_floor_passed"]),
        "ratio_passed": bool(trigger["ratio_passed"]),
        "raw_ratio_passed": bool(trigger["raw_ratio_passed"]),
        "support_count": int(best_info["support_count"]),
        "support_passed": bool(trigger["support_passed"]),
        "weak_window": bool(trigger["weak_window"]),
        "ratio_trigger_threshold": float(settings.ratio_trigger_threshold),
        "long_score_floor": float(settings.long_score_floor),
        "raw_ratio_threshold": float(settings.raw_ratio_threshold),
        "support_count_threshold": int(settings.support_count_threshold),
        "strong_min_len": int(settings.strong_min_len),
        "weak_min_base_len": int(settings.weak_min_base_len),
        "weak_max_base_len": int(settings.weak_max_base_len),
        "cap_base_le8": int(settings.cap_base_le8),
        "cap_base_9_12": int(settings.cap_base_9_12),
        "base_probe_lengths": parse_probe_lengths(settings.base_probe_lengths_csv),
        "base_alpha": float(settings.base_alpha),
        "base_selected_length": base_selected,
        "base_selected_mask_length": base_selected,
        "base_selected_score": float(base_selection["selected_score"]),
        "base_selected_raw_score": float(base_selection["selected_raw_score"]),
        "base_selected_adjusted_score": float(base_selection["selected_adjusted_score"]),
        "base_length_probe_sec": float(base_selection["length_probe_sec"]),
        "base_selected_minus_oracle_length": base_diff["selected_minus_oracle_length"],
        "base_abs_selected_minus_oracle_length": base_diff["abs_selected_minus_oracle_length"],
        "base_candidate_scores": base_selection["candidate_scores"],
        "weak_probe_lengths": parse_probe_lengths(settings.weak_probe_lengths_csv),
        "strong_probe_lengths": parse_probe_lengths(settings.strong_probe_lengths_csv),
        "long_probe_lengths": (
            None
            if trigger["correction_probe_lengths_csv"] is None
            else parse_probe_lengths(str(trigger["correction_probe_lengths_csv"]))
        ),
        "long_alpha": float(settings.long_alpha),
        "long_selected_mask_length": None,
        "long_selected_score": None,
        "long_selected_raw_score": None,
        "long_selected_adjusted_score": None,
        "long_length_probe_sec": 0.0,
        "long_candidate_scores": None,
        "long_selected_length": None,
        "long_argmax_selected_length": None,
        "final_selected_length": selected,
        "final_minus_base_length": selected - base_selected,
    }

    if long_selection is not None:
        meta.update(
            {
                "long_selected_length": int(long_selection["selected_mask_length"]),
                "long_selected_mask_length": int(long_selection["selected_mask_length"]),
                "long_selected_score": float(long_selection["selected_score"]),
                "long_selected_raw_score": float(long_selection["selected_raw_score"]),
                "long_selected_adjusted_score": float(long_selection["selected_adjusted_score"]),
                "long_length_probe_sec": float(long_selection["length_probe_sec"]),
                "long_candidate_scores": long_selection["candidate_scores"],
                "long_argmax_selected_length": int(long_selection["correction_argmax_selected_mask_length"]),
            }
        )

    output = {
        "oracle_mask_length": None if oracle_mask_length is None else int(oracle_mask_length),
        "mask_length_source": "lcal_rescue_policy",
        "mask_length": selected,
        "selected_mask_length": selected,
        "selected_score": float(final_selection["selected_score"]),
        "selected_raw_score": float(final_selection["selected_raw_score"]),
        "selected_adjusted_score": float(final_selection["selected_adjusted_score"]),
        "candidate_scores": final_selection["candidate_scores"],
        "length_probe_sec": total_probe_sec,
        "probe_lengths": final_selection["probe_lengths"],
        "tie_break": final_selection["tie_break"],
        "score_mode": final_selection["score_mode"],
        "length_alpha": final_selection["length_alpha"],
        **diff,
        "lcal_v3": meta,
    }
    _LCAL_META_BY_TASK_ID[task.task_id] = output
    return output


def patch_decode_for_lcal_v3(settings: LcalV3Settings) -> None:
    global _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = settings
    decode_lcas.resolve_mask_length = resolve_mask_length_lcal_v3
    decode_lcas.choose_lcas_config = choose_lcas_v3_config


def annotate_lcal_v3_result(result: Dict[str, Any]) -> None:
    meta = _LCAL_META_BY_TASK_ID.pop(result["task_id"], None)
    if meta is None:
        return

    lcal_meta = meta["lcal_v3"]
    result["lcal_v3"] = lcal_meta

    result["length_probe"].update(
        {
            "lcal_v3": {
                key: value
                for key, value in lcal_meta.items()
                if key not in {"base_candidate_scores", "long_candidate_scores"}
            },
            "selected_raw_score": meta["selected_raw_score"],
            "selected_adjusted_score": meta["selected_adjusted_score"],
            "base_candidate_scores": lcal_meta["base_candidate_scores"],
            "long_candidate_scores": lcal_meta["long_candidate_scores"],
        }
    )

    result["metrics"].update(
        {
            "selected_raw_score": meta["selected_raw_score"],
            "selected_adjusted_score": meta["selected_adjusted_score"],
            "score_mode": meta["score_mode"],
            "length_alpha": meta["length_alpha"],
            "best_score": lcal_meta["best_score"],
            "best_len": lcal_meta["best_len"],
            "best_long_score": lcal_meta["best_long_score"],
            "best_long_len": lcal_meta["best_long_len"],
            "long_ratio": lcal_meta["long_ratio"],
            "best_raw_score": lcal_meta["best_raw_score"],
            "best_raw_len": lcal_meta["best_raw_len"],
            "best_long_raw_score": lcal_meta["best_long_raw_score"],
            "best_long_raw_len": lcal_meta["best_long_raw_len"],
            "raw_long_ratio": lcal_meta["raw_long_ratio"],
            "long_score_floor_passed": lcal_meta["long_score_floor_passed"],
            "ratio_passed": lcal_meta["ratio_passed"],
            "raw_ratio_passed": lcal_meta["raw_ratio_passed"],
            "support_count": lcal_meta["support_count"],
            "support_passed": lcal_meta["support_passed"],
            "weak_window": lcal_meta["weak_window"],
            "trigger_reason": lcal_meta["trigger_reason"],
            "base_selected_length": lcal_meta["base_selected_length"],
            "long_selected_length": lcal_meta["long_selected_length"],
            "final_selected_length": lcal_meta["final_selected_length"],
            "final_source": lcal_meta["final_source"],
            "correction_kind": lcal_meta["correction_kind"],
            "correction_selection_rule": lcal_meta["correction_selection_rule"],
            "shortest_supported_ratio": lcal_meta["shortest_supported_ratio"],
            "long_argmax_selected_length": lcal_meta["long_argmax_selected_length"],
            "short_safe_policy": lcal_meta["short_safe_policy"],
            "lcal_v3_final_source": lcal_meta["final_source"],
            "lcal_v3_long_triggered": lcal_meta["long_triggered"],
            "lcal_v3_strong_triggered": lcal_meta["strong_triggered"],
            "lcal_v3_weak_triggered": lcal_meta["weak_triggered"],
            "lcal_v3_correction_kind": lcal_meta["correction_kind"],
            "lcal_v3_correction_selection_rule": lcal_meta["correction_selection_rule"],
            "lcal_v3_long_argmax_selected_length": lcal_meta["long_argmax_selected_length"],
            "lcal_v3_short_safe_policy": lcal_meta["short_safe_policy"],
            "lcal_v3_base_selected_mask_length": lcal_meta["base_selected_mask_length"],
            "lcal_v3_base_selected_score": lcal_meta["base_selected_score"],
            "lcal_v3_base_selected_minus_oracle_length": lcal_meta["base_selected_minus_oracle_length"],
            "lcal_v3_base_abs_selected_minus_oracle_length": lcal_meta[
                "base_abs_selected_minus_oracle_length"
            ],
            "lcal_v3_long_selected_mask_length": lcal_meta["long_selected_mask_length"],
            "lcal_v3_final_minus_base_length": lcal_meta["final_minus_base_length"],
        }
    )


def load_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _baseline_comparison(
    results: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    baseline_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not baseline_rows:
        return None

    baseline_map = {row["task_id"]: row for row in baseline_rows}
    common: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
        (result, baseline_map[result["task_id"]])
        for result in results
        if result["task_id"] in baseline_map
    ]

    if not common:
        return {
            "baseline_results": baseline_path,
            "common_samples": 0,
        }

    def new_pass(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> bool:
        return bool(pair[0]["metrics"].get("passed", False))

    def base_pass(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> bool:
        return bool(pair[1]["metrics"].get("passed", False))

    def oracle_len(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> Optional[int]:
        value = pair[0]["metrics"].get("oracle_mask_length")
        return None if value is None else int(value)

    short_pairs = [pair for pair in common if oracle_len(pair) is not None and int(oracle_len(pair)) <= 8]
    long_pairs = [pair for pair in common if oracle_len(pair) is not None and int(oracle_len(pair)) >= 17]

    wins = [pair for pair in common if new_pass(pair) and not base_pass(pair)]
    losses = [pair for pair in common if base_pass(pair) and not new_pass(pair)]
    short_losses = [pair for pair in short_pairs if base_pass(pair) and not new_pass(pair)]
    long_wins = [pair for pair in long_pairs if new_pass(pair) and not base_pass(pair)]

    return {
        "baseline_results": baseline_path,
        "common_samples": len(common),
        "overall_win": len(wins),
        "overall_loss": len(losses),
        "overall_tie_pass": sum(1 for pair in common if new_pass(pair) and base_pass(pair)),
        "overall_tie_fail": sum(1 for pair in common if not new_pass(pair) and not base_pass(pair)),
        "short_bucket_common_samples": len(short_pairs),
        "short_bucket_loss": len(short_losses),
        "short_bucket_loss_rate": len(short_losses) / len(short_pairs) if short_pairs else None,
        "long_bucket_common_samples": len(long_pairs),
        "long_bucket_win": len(long_wins),
        "long_bucket_win_rate": len(long_wins) / len(long_pairs) if long_pairs else None,
        "win_oracle_bucket_histogram": _hist(_oracle_bucket(oracle_len(pair)) for pair in wins),
        "loss_oracle_bucket_histogram": _hist(_oracle_bucket(oracle_len(pair)) for pair in losses),
    }


def summarize_lcal_v3_results(
    results: List[Dict[str, Any]],
    baseline_rows: Optional[List[Dict[str, Any]]] = None,
    baseline_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not results:
        return {"num_samples": 0}

    summary = summarize_lcas_v3_results(results)
    metrics = [item["metrics"] for item in results]

    oracle_buckets = sorted({_oracle_bucket(m.get("oracle_mask_length")) for m in metrics})
    summary.update(
        {
            "oracle_bucket_histogram": _hist(_oracle_bucket(m.get("oracle_mask_length")) for m in metrics),
            "oracle_bucket_pass_rate": {
                bucket: _rate(
                    bool(m.get("passed", False))
                    for m in metrics
                    if _oracle_bucket(m.get("oracle_mask_length")) == bucket
                )
                for bucket in oracle_buckets
            },
            "oracle_bucket_avg_selected_minus_oracle_length": {
                bucket: _avg(
                    m.get("selected_minus_oracle_length")
                    for m in metrics
                    if _oracle_bucket(m.get("oracle_mask_length")) == bucket
                )
                for bucket in oracle_buckets
            },
            "lcal_v3_long_trigger_rate": _rate(bool(m.get("lcal_v3_long_triggered", False)) for m in metrics),
            "lcal_v3_strong_trigger_rate": _rate(
                bool(m.get("lcal_v3_strong_triggered", False)) for m in metrics
            ),
            "lcal_v3_weak_trigger_rate": _rate(bool(m.get("lcal_v3_weak_triggered", False)) for m in metrics),
            "lcal_v3_final_source_histogram": _hist(m.get("lcal_v3_final_source") for m in metrics),
            "lcal_v3_trigger_reason_histogram": _hist(m.get("trigger_reason") for m in metrics),
            "lcal_v3_correction_kind_histogram": _hist(m.get("lcal_v3_correction_kind") for m in metrics),
            "lcal_v3_avg_base_selected_mask_length": _avg(
                m.get("lcal_v3_base_selected_mask_length") for m in metrics
            ),
            "lcal_v3_avg_final_minus_base_length": _avg(m.get("lcal_v3_final_minus_base_length") for m in metrics),
        }
    )

    oracle_17plus = [
        m for m in metrics if m.get("oracle_mask_length") is not None and int(m["oracle_mask_length"]) >= 17
    ]
    oracle_le8 = [
        m for m in metrics if m.get("oracle_mask_length") is not None and int(m["oracle_mask_length"]) <= 8
    ]
    summary["under_select_rate_17plus"] = _rate(
        int(m.get("selected_mask_length")) < int(m.get("oracle_mask_length")) for m in oracle_17plus
    )
    summary["over_select_rate_le8"] = _rate(
        int(m.get("selected_mask_length")) > int(m.get("oracle_mask_length")) for m in oracle_le8
    )
    summary["short_large_jump_rate"] = _rate(
        int(m.get("lcal_v3_final_minus_base_length", 0)) >= 4 for m in oracle_le8
    )
    summary["avg_final_minus_base_short"] = _avg(
        m.get("lcal_v3_final_minus_base_length") for m in oracle_le8
    )
    weak_triggered = [m for m in metrics if bool(m.get("lcal_v3_weak_triggered", False))]
    summary["weak_trigger_precision_true_long"] = (
        _rate(int(m.get("oracle_mask_length")) >= 17 for m in weak_triggered)
        if weak_triggered
        else None
    )
    summary["weak_trigger_count"] = len(weak_triggered)
    summary["strong_trigger_count"] = sum(1 for m in metrics if bool(m.get("lcal_v3_strong_triggered", False)))

    comparison = _baseline_comparison(results, baseline_rows or [], baseline_path)
    if comparison is not None:
        summary["baseline_comparison"] = comparison

    # Keep the exact names from the experiment note easy to grep.
    summary["required_metrics"] = {
        "overall_pass_rate": summary.get("pass_rate"),
        "<=8_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("<=8"),
        "9-12_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("9-12"),
        "13-16_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("13-16"),
        "17-24_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("17-24"),
        "25+_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("25+"),
        "short_bucket_loss": None if comparison is None else comparison.get("short_bucket_loss"),
        "long_bucket_win": None if comparison is None else comparison.get("long_bucket_win"),
        "avg_selected_minus_oracle_length": summary.get("avg_selected_minus_oracle_length"),
        "under_select_rate_17plus": summary.get("under_select_rate_17plus"),
        "over_select_rate_le8": summary.get("over_select_rate_le8"),
        "short_large_jump_rate": summary.get("short_large_jump_rate"),
        "avg_final_minus_base_short": summary.get("avg_final_minus_base_short"),
        "weak_trigger_precision_true_long": summary.get("weak_trigger_precision_true_long"),
    }
    return summary


def run_lcal_v3_experiment(
    cfg: ExperimentConfig,
    settings: LcalV3Settings,
    baseline_results_path: Optional[str] = None,
) -> Dict[str, Any]:
    patch_decode_for_lcal_v3(settings)

    set_global_seed(cfg.decode.seed)
    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)

    config_payload = cfg.to_dict()
    config_payload["decode"]["mask_length_source"] = "lcal_rescue_policy"
    config_payload["decode"]["lcas_policy"] = getattr(cfg.decode, "lcas_policy", None)
    config_payload["lcal_v3"] = asdict(settings)
    config_payload["baseline_results"] = baseline_results_path
    logger.save_config(config_payload)

    print("=" * 80, flush=True)
    print("Starting LCAL rescue policy experiment", flush=True)
    print(f"experiment_name = {cfg.logging.experiment_name}", flush=True)
    print(f"output_dir       = {cfg.logging.output_dir}", flush=True)
    print(f"dataset_subset   = {cfg.data.dataset_subset}", flush=True)
    print(f"split            = {cfg.data.split}", flush=True)
    print(f"max_samples      = {cfg.data.max_samples}", flush=True)
    print(f"model_path       = {cfg.model.model_path}", flush=True)
    print(f"short_policy     = {settings.short_safe_policy}", flush=True)
    print(f"correction_rule  = {settings.correction_selection_rule}", flush=True)
    print(f"supported_ratio  = {settings.shortest_supported_ratio}", flush=True)
    print(f"base_grid        = {settings.base_probe_lengths_csv}", flush=True)
    print(f"base_alpha       = {settings.base_alpha}", flush=True)
    print(f"weak_grid        = {settings.weak_probe_lengths_csv}", flush=True)
    print(f"strong_grid      = {settings.strong_probe_lengths_csv}", flush=True)
    print(f"long_alpha       = {settings.long_alpha}", flush=True)
    print(f"strong_min_len   = {settings.strong_min_len}", flush=True)
    print(f"weak_base_window = {settings.weak_min_base_len}..{settings.weak_max_base_len}", flush=True)
    print(f"ratio_threshold  = {settings.ratio_trigger_threshold}", flush=True)
    print(f"long_score_floor = {settings.long_score_floor}", flush=True)
    print(f"raw_ratio_thr    = {settings.raw_ratio_threshold}", flush=True)
    print(f"support_count    = {settings.support_count_threshold}", flush=True)
    print(f"cap_base_le8     = {settings.cap_base_le8}", flush=True)
    print(f"cap_base_9_12    = {settings.cap_base_9_12}", flush=True)
    print(f"lcas_policy      = {getattr(cfg.decode, 'lcas_policy', None)}", flush=True)
    print(f"total_steps      = {cfg.decode.total_steps}", flush=True)
    print(f"seed             = {cfg.decode.seed}", flush=True)
    print(f"baseline_results = {baseline_results_path}", flush=True)
    print("=" * 80, flush=True)

    tokenizer, model = load_model_and_tokenizer(cfg.model)
    tasks = load_humaneval_infilling(
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        dataset_subset=cfg.data.dataset_subset,
    )

    total_tasks = len(tasks)
    print(f"Loaded {total_tasks} tasks", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("-" * 80, flush=True)

    results: List[Dict[str, Any]] = []

    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{total_tasks}] task_id={task.task_id} | start", flush=True)

        result = decode_lcas.run_decode_with_lcas(task, tokenizer, model, cfg)
        annotate_lcal_v3_result(result)
        annotate_lcas_v3_result(result)

        result["dataset_subset"] = cfg.data.dataset_subset
        results.append(result)

        payload = {
            "task_id": result["task_id"],
            "dataset_subset": cfg.data.dataset_subset,
            "metrics": result["metrics"],
            "verification": result["verification"],
            "length_probe": result["length_probe"],
            "stopping": result["stopping"],
            "lcal_v3": result.get("lcal_v3"),
            "code": result["code"],
            "diagnostics": result["diagnostics"],
        }
        logger.log_result(payload)

        for trace in result["step_traces"]:
            logger.log_trace(trace)

        m = result["metrics"]
        print(
            f"[{idx}/{total_tasks}] task_id={task.task_id} | "
            f"{'PASS' if m['passed'] else 'FAIL'} | "
            f"total_sec={m['total_sec']:.3f} | "
            f"total_sec_including_probe={m['total_sec_including_probe']:.3f} | "
            f"base_len={m['lcal_v3_base_selected_mask_length']} | "
            f"final_len={m['selected_mask_length']} | "
            f"oracle_len={m['oracle_mask_length']} | "
            f"diff={m['selected_minus_oracle_length']} | "
            f"long_triggered={m['lcal_v3_long_triggered']} | "
            f"kind={m['correction_kind']} | "
            f"trigger_reason={m['trigger_reason']} | "
            f"ratio={m['long_ratio']:.4f} | "
            f"raw_ratio={m['raw_long_ratio']:.4f} | "
            f"support={m['support_count']} | "
            f"stopped={m['stopped']} | "
            f"stop_step={m['stop_step']} | "
            f"effective_steps={m['effective_steps']} | "
            f"stop_reason={m['stop_reason']}",
            flush=True,
        )

    baseline_rows: List[Dict[str, Any]] = []
    if baseline_results_path:
        if os.path.exists(baseline_results_path):
            baseline_rows = load_jsonl(baseline_results_path)
        else:
            print(
                f"WARNING: baseline results not found, skipping baseline comparison: {baseline_results_path}",
                flush=True,
            )
    summary = summarize_lcal_v3_results(
        results=results,
        baseline_rows=baseline_rows,
        baseline_path=baseline_results_path,
    )
    logger.save_json("summary.json", summary)

    print("-" * 80, flush=True)
    print("LCAL rescue policy experiment finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)

    return {
        "results": results,
        "summary": summary,
        "run_dir": logger.run_dir,
    }


def main() -> None:
    args = parse_args()

    settings = LcalV3Settings(
        base_probe_lengths_csv=args.base_probe_lengths,
        base_alpha=args.base_alpha,
        weak_probe_lengths_csv=args.weak_probe_lengths,
        strong_probe_lengths_csv=args.strong_probe_lengths,
        long_alpha=args.long_alpha,
        strong_min_len=args.strong_min_len,
        weak_min_base_len=args.weak_min_base_len,
        weak_max_base_len=args.weak_max_base_len,
        ratio_trigger_threshold=args.ratio_trigger_threshold,
        long_score_floor=args.long_score_floor,
        raw_ratio_threshold=args.raw_ratio_threshold,
        support_count_threshold=args.support_count_threshold,
        cap_base_le8=args.cap_base_le8,
        cap_base_9_12=args.cap_base_9_12,
        short_safe_policy=args.short_safe_policy,
        correction_selection_rule=args.correction_selection_rule,
        shortest_supported_ratio=args.shortest_supported_ratio,
        tie_break=args.tie_break,
        score_mode=args.score_mode,
        length_prop_beta=args.length_prop_beta,
        length_prop_ref_length=args.length_prop_ref_length,
        length_prop_cap_length=args.length_prop_cap_length,
    )

    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.data.split = args.split
    cfg.data.dataset_subset = args.dataset_subset
    cfg.data.max_samples = args.max_samples

    cfg.decode.mask_length_source = "lcal_rescue_policy"
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed
    cfg.decode.save_step_traces = args.save_step_traces
    cfg.decode.save_full_text_per_step = args.save_full_text_per_step
    cfg.decode.cal_lite_probe_lengths_csv = args.base_probe_lengths
    cfg.decode.cal_lite_tie_break = args.tie_break
    cfg.decode.cal_lite_score_mode = args.score_mode
    cfg.decode.cal_lite_length_alpha = args.base_alpha
    cfg.decode.cal_lite_length_prop_beta = args.length_prop_beta
    cfg.decode.cal_lite_length_prop_ref_length = args.length_prop_ref_length
    cfg.decode.cal_lite_length_prop_cap_length = args.length_prop_cap_length
    cfg.decode.lcas_policy = args.lcas_policy

    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name

    output = run_lcal_v3_experiment(
        cfg=cfg,
        settings=settings,
        baseline_results_path=args.baseline_results,
    )

    print("===== LCAL Rescue Policy Summary =====")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")
    print(f"\nRun directory: {output['run_dir']}")


if __name__ == "__main__":
    main()
