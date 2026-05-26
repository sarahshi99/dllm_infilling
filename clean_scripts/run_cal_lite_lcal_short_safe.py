#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import expvision_dllm_clean.decode_lcas as decode_lcas
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask, compute_oracle_mask_length, load_humaneval_infilling
from expvision_dllm_clean.length_probe import parse_probe_lengths, select_mask_length_cal_lite
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.runner_lcas_v3 import (
    annotate_lcas_v3_result,
    choose_lcas_v3_config,
    summarize_lcas_v3_results,
)

# Reuse mature utility functions from the existing T2 runner without changing that file.
import run_cal_lite_lcal_v3_t2_ratio as t2_utils


DEFAULT_BASE_GRID = "3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24"
DEFAULT_STRONG_GRID = "13,14,15,16,20,24,28,32,40"
DEFAULT_WEAK_GRID = "13,14,15,16"

_ACTIVE_SETTINGS: Optional["ShortSafeSettings"] = None
_META_BY_TASK_ID: Dict[str, Dict[str, Any]] = {}


@dataclass
class ShortSafeSettings:
    base_probe_lengths_csv: str
    base_alpha: float
    strong_probe_lengths_csv: str
    weak_probe_lengths_csv: str
    long_alpha: float
    short_safe_policy: str
    strong_min_length: int
    weak_min_base_length: int
    weak_max_base_length: int
    ratio_trigger_threshold: float
    long_score_floor: float
    raw_ratio_threshold: float
    support_count_threshold: int
    cap_base_le8: int
    cap_base_9_12: int
    tie_break: str
    score_mode: str

    # Compatibility with the old T2 helper print/config conventions.
    @property
    def long_probe_lengths_csv(self) -> str:
        return self.strong_probe_lengths_csv

    @property
    def long_trigger_min_length(self) -> int:
        return self.strong_min_length

    @property
    def long_trigger_extra_lengths_csv(self) -> str:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short-safe LCAL: strong trigger + weak-grid/jump-cap variants + LCAS-v3 stopping."
    )
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-probe-lengths", type=str, default=DEFAULT_BASE_GRID)
    parser.add_argument("--base-alpha", type=float, default=0.06)
    parser.add_argument("--strong-probe-lengths", type=str, default=DEFAULT_STRONG_GRID)
    parser.add_argument("--weak-probe-lengths", type=str, default=DEFAULT_WEAK_GRID)
    parser.add_argument("--long-alpha", type=float, default=0.10)
    parser.add_argument("--short-safe-policy", type=str, default="s2", choices=["s1", "s2", "s3", "s4"])
    parser.add_argument("--strong-min-length", type=int, default=13)
    parser.add_argument("--weak-min-base-length", type=int, default=8)
    parser.add_argument("--weak-max-base-length", type=int, default=12)
    parser.add_argument("--ratio-trigger-threshold", type=float, default=0.97)
    parser.add_argument("--long-score-floor", type=float, default=0.55)
    parser.add_argument("--raw-ratio-threshold", type=float, default=0.97)
    parser.add_argument("--support-count-threshold", type=int, default=2)
    parser.add_argument("--cap-base-le8", type=int, default=14)
    parser.add_argument("--cap-base-9-12", type=int, default=16)
    parser.add_argument("--tie-break", type=str, default="shorter", choices=["shorter", "longer"])
    parser.add_argument("--score-mode", type=str, default="length_power", choices=["raw", "length_power"])

    parser.add_argument("--lcas-policy", type=str, default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    parser.add_argument("--baseline-results", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs_clean")
    parser.add_argument("--experiment-name", type=str, default="lcal_short_safe")
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--save-full-text-per-step", action="store_true")
    return parser.parse_args()


def _clone_probe_cfg(cfg: ExperimentConfig, probe_lengths_csv: str, alpha: float, settings: ShortSafeSettings) -> ExperimentConfig:
    return t2_utils._clone_probe_cfg(
        cfg=cfg,
        probe_lengths_csv=probe_lengths_csv,
        alpha=alpha,
        settings=settings,
    )


def _candidate_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("score", candidate.get("adjusted_score", candidate.get("raw_score"))))


def _raw_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("raw_score", candidate.get("mean_top1_prob", candidate.get("score"))))


def _best_candidate(candidates: List[Dict[str, Any]], tie_break: str) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("Cannot pick best candidate from empty candidate list")
    if tie_break == "shorter":
        return max(candidates, key=lambda item: (_candidate_score(item), -int(item["mask_length"])))
    if tie_break == "longer":
        return max(candidates, key=lambda item: (_candidate_score(item), int(item["mask_length"])))
    raise ValueError(f"Unsupported tie_break={tie_break}")


def _base_best_info(base_selection: Dict[str, Any], settings: ShortSafeSettings) -> Dict[str, Any]:
    candidates = list(base_selection["candidate_scores"])
    best = _best_candidate(candidates, settings.tie_break)
    long_candidates = [item for item in candidates if int(item["mask_length"]) >= settings.strong_min_length]
    best_long = _best_candidate(long_candidates, settings.tie_break)

    raw_best = max(candidates, key=lambda item: (_raw_score(item), -int(item["mask_length"])))
    raw_long_best = max(long_candidates, key=lambda item: (_raw_score(item), -int(item["mask_length"])))

    best_score = _candidate_score(best)
    best_long_score = _candidate_score(best_long)
    best_raw_score = _raw_score(raw_best)
    best_long_raw_score = _raw_score(raw_long_best)
    support_count = sum(
        1
        for item in long_candidates
        if _candidate_score(item) >= float(settings.ratio_trigger_threshold) * best_score
    )

    return {
        "best_len": int(best["mask_length"]),
        "best_score": float(best_score),
        "best_long_len": int(best_long["mask_length"]),
        "best_long_score": float(best_long_score),
        "long_ratio": best_long_score / best_score if best_score else None,
        "best_raw_len": int(raw_best["mask_length"]),
        "best_raw_score": float(best_raw_score),
        "best_long_raw_len": int(raw_long_best["mask_length"]),
        "best_long_raw_score": float(best_long_raw_score),
        "raw_long_ratio": best_long_raw_score / best_raw_score if best_raw_score else None,
        "long_support_count": int(support_count),
    }


def _length_diff(selected_length: Optional[int], oracle_length: Optional[int]) -> Dict[str, Optional[int]]:
    if selected_length is None or oracle_length is None:
        return {"selected_minus_oracle_length": None, "abs_selected_minus_oracle_length": None}
    diff = int(selected_length) - int(oracle_length)
    return {"selected_minus_oracle_length": diff, "abs_selected_minus_oracle_length": abs(diff)}


def _cap_for_weak_base(base_len: int, settings: ShortSafeSettings) -> Optional[int]:
    if base_len <= 8:
        return int(settings.cap_base_le8)
    if 9 <= base_len <= 12:
        return int(settings.cap_base_9_12)
    return None


def _weak_trigger_passed(best_info: Dict[str, Any], settings: ShortSafeSettings) -> Tuple[bool, Dict[str, Any]]:
    long_ratio = best_info["long_ratio"]
    raw_long_ratio = best_info["raw_long_ratio"]
    ratio_passed = (
        long_ratio is not None
        and float(long_ratio) >= float(settings.ratio_trigger_threshold)
        and float(best_info["best_long_score"]) >= float(settings.long_score_floor)
    )
    raw_ratio_passed = raw_long_ratio is not None and float(raw_long_ratio) >= float(settings.raw_ratio_threshold)
    support_passed = int(best_info["long_support_count"]) >= int(settings.support_count_threshold)

    if settings.short_safe_policy == "s1":
        passed = ratio_passed
    elif settings.short_safe_policy == "s2":
        passed = ratio_passed
    elif settings.short_safe_policy == "s3":
        passed = ratio_passed and raw_ratio_passed
    elif settings.short_safe_policy == "s4":
        passed = ratio_passed and support_passed
    else:
        raise ValueError(f"Unsupported short_safe_policy={settings.short_safe_policy}")

    return passed, {
        "ratio_passed": bool(ratio_passed),
        "raw_ratio_passed": bool(raw_ratio_passed),
        "support_passed": bool(support_passed),
    }


def _select_with_grid(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
    settings: ShortSafeSettings,
    probe_lengths_csv: str,
    alpha: float,
) -> Dict[str, Any]:
    probe_cfg = _clone_probe_cfg(cfg, probe_lengths_csv=probe_lengths_csv, alpha=alpha, settings=settings)
    return select_mask_length_cal_lite(task, tokenizer, model, probe_cfg)


def resolve_mask_length_short_safe(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    if _ACTIVE_SETTINGS is None:
        raise RuntimeError("Short-safe LCAL settings are not initialized")
    settings = _ACTIVE_SETTINGS
    oracle_mask_length = compute_oracle_mask_length(task, tokenizer, add_special_tokens=False)

    base_selection = _select_with_grid(
        task=task,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        settings=settings,
        probe_lengths_csv=settings.base_probe_lengths_csv,
        alpha=settings.base_alpha,
    )
    base_selected = int(base_selection["selected_mask_length"])
    best_info = _base_best_info(base_selection, settings)

    trigger_reason = "none"
    correction_kind = "none"
    triggered = False
    weak_triggered = False
    strong_triggered = False
    correction_selection: Optional[Dict[str, Any]] = None
    correction_probe_lengths_csv: Optional[str] = None
    final_selection = base_selection
    total_probe_sec = float(base_selection["length_probe_sec"])

    if base_selected >= int(settings.strong_min_length):
        triggered = True
        strong_triggered = True
        trigger_reason = "strong_base_len"
        correction_kind = "strong_full_grid"
        correction_probe_lengths_csv = settings.strong_probe_lengths_csv
    else:
        weak_window = int(settings.weak_min_base_length) <= base_selected <= int(settings.weak_max_base_length)
        weak_passed, weak_flags = _weak_trigger_passed(best_info, settings)
        if weak_window and weak_passed:
            triggered = True
            weak_triggered = True
            trigger_reason = f"weak_{settings.short_safe_policy}"
            correction_kind = "weak_grid"
            weak_lengths = parse_probe_lengths(settings.weak_probe_lengths_csv)
            if settings.short_safe_policy in {"s2", "s3", "s4"}:
                cap = _cap_for_weak_base(base_selected, settings)
                if cap is not None:
                    weak_lengths = [length for length in weak_lengths if length <= cap]
            correction_probe_lengths_csv = ",".join(str(length) for length in weak_lengths)
        else:
            weak_flags = _weak_trigger_passed(best_info, settings)[1]

    if triggered and correction_probe_lengths_csv:
        correction_selection = _select_with_grid(
            task=task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            settings=settings,
            probe_lengths_csv=correction_probe_lengths_csv,
            alpha=settings.long_alpha,
        )
        total_probe_sec += float(correction_selection["length_probe_sec"])
        correction_selected = int(correction_selection["selected_mask_length"])
        if correction_selected >= base_selected:
            final_selection = correction_selection
            final_source = correction_kind
        else:
            final_selection = base_selection
            final_source = "base_max_after_correction"
    else:
        final_source = "base"

    selected = int(final_selection["selected_mask_length"])
    if correction_selection is not None:
        selected = max(base_selected, int(correction_selection["selected_mask_length"]))

    diff = _length_diff(selected, None if oracle_mask_length is None else int(oracle_mask_length))
    base_diff = _length_diff(base_selected, None if oracle_mask_length is None else int(oracle_mask_length))

    weak_flags = _weak_trigger_passed(best_info, settings)[1]
    meta: Dict[str, Any] = {
        "version": f"lcal_short_safe_{settings.short_safe_policy}",
        "short_safe_policy": settings.short_safe_policy,
        "final_source": final_source,
        "long_triggered": bool(triggered),
        "strong_triggered": bool(strong_triggered),
        "weak_triggered": bool(weak_triggered),
        "trigger_reason": trigger_reason,
        "correction_kind": correction_kind,
        "best_score": float(best_info["best_score"]),
        "best_len": int(best_info["best_len"]),
        "best_long_score": float(best_info["best_long_score"]),
        "best_long_len": int(best_info["best_long_len"]),
        "long_ratio": None if best_info["long_ratio"] is None else float(best_info["long_ratio"]),
        "best_raw_score": float(best_info["best_raw_score"]),
        "best_long_raw_score": float(best_info["best_long_raw_score"]),
        "raw_long_ratio": None if best_info["raw_long_ratio"] is None else float(best_info["raw_long_ratio"]),
        "long_score_floor_passed": float(best_info["best_long_score"]) >= float(settings.long_score_floor),
        "ratio_trigger_threshold": float(settings.ratio_trigger_threshold),
        "long_score_floor": float(settings.long_score_floor),
        "raw_ratio_threshold": float(settings.raw_ratio_threshold),
        "ratio_passed": weak_flags["ratio_passed"],
        "raw_ratio_passed": weak_flags["raw_ratio_passed"],
        "support_count": int(best_info["long_support_count"]),
        "support_count_threshold": int(settings.support_count_threshold),
        "support_passed": weak_flags["support_passed"],
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
        "strong_probe_lengths": parse_probe_lengths(settings.strong_probe_lengths_csv),
        "weak_probe_lengths": parse_probe_lengths(settings.weak_probe_lengths_csv),
        "correction_probe_lengths": [] if correction_probe_lengths_csv is None else parse_probe_lengths(correction_probe_lengths_csv),
        "long_alpha": float(settings.long_alpha),
        "correction_selected_mask_length": None,
        "correction_selected_score": None,
        "correction_selected_raw_score": None,
        "correction_selected_adjusted_score": None,
        "correction_length_probe_sec": 0.0,
        "correction_candidate_scores": None,
        "final_selected_length": selected,
        "final_minus_base_length": selected - base_selected,
    }

    if correction_selection is not None:
        meta.update(
            {
                "correction_selected_mask_length": int(correction_selection["selected_mask_length"]),
                "correction_selected_score": float(correction_selection["selected_score"]),
                "correction_selected_raw_score": float(correction_selection["selected_raw_score"]),
                "correction_selected_adjusted_score": float(correction_selection["selected_adjusted_score"]),
                "correction_length_probe_sec": float(correction_selection["length_probe_sec"]),
                "correction_candidate_scores": correction_selection["candidate_scores"],
            }
        )

    output = {
        "oracle_mask_length": None if oracle_mask_length is None else int(oracle_mask_length),
        "mask_length_source": f"lcal_short_safe_{settings.short_safe_policy}",
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
    _META_BY_TASK_ID[task.task_id] = output
    return output


def patch_decode(settings: ShortSafeSettings) -> None:
    global _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = settings
    decode_lcas.resolve_mask_length = resolve_mask_length_short_safe
    decode_lcas.choose_lcas_config = choose_lcas_v3_config


def annotate_short_safe_result(result: Dict[str, Any]) -> None:
    meta = _META_BY_TASK_ID.pop(result["task_id"], None)
    if meta is None:
        return
    lcal_meta = meta["lcal_v3"]
    result["lcal_v3"] = lcal_meta
    result["length_probe"].update(
        {
            "lcal_v3": {
                key: value
                for key, value in lcal_meta.items()
                if key not in {"base_candidate_scores", "correction_candidate_scores"}
            },
            "selected_raw_score": meta["selected_raw_score"],
            "selected_adjusted_score": meta["selected_adjusted_score"],
            "base_candidate_scores": lcal_meta["base_candidate_scores"],
            "correction_candidate_scores": lcal_meta["correction_candidate_scores"],
        }
    )
    result["metrics"].update(
        {
            "selected_raw_score": meta["selected_raw_score"],
            "selected_adjusted_score": meta["selected_adjusted_score"],
            "score_mode": meta["score_mode"],
            "length_alpha": meta["length_alpha"],
            "short_safe_policy": lcal_meta["short_safe_policy"],
            "best_score": lcal_meta["best_score"],
            "best_len": lcal_meta["best_len"],
            "best_long_score": lcal_meta["best_long_score"],
            "best_long_len": lcal_meta["best_long_len"],
            "long_ratio": lcal_meta["long_ratio"],
            "raw_long_ratio": lcal_meta["raw_long_ratio"],
            "long_score_floor_passed": lcal_meta["long_score_floor_passed"],
            "ratio_passed": lcal_meta["ratio_passed"],
            "raw_ratio_passed": lcal_meta["raw_ratio_passed"],
            "support_count": lcal_meta["support_count"],
            "support_passed": lcal_meta["support_passed"],
            "trigger_reason": lcal_meta["trigger_reason"],
            "correction_kind": lcal_meta["correction_kind"],
            "strong_triggered": lcal_meta["strong_triggered"],
            "weak_triggered": lcal_meta["weak_triggered"],
            "base_selected_length": lcal_meta["base_selected_length"],
            "correction_selected_length": lcal_meta["correction_selected_mask_length"],
            "final_selected_length": lcal_meta["final_selected_length"],
            "final_source": lcal_meta["final_source"],
            "lcal_v3_final_source": lcal_meta["final_source"],
            "lcal_v3_long_triggered": lcal_meta["long_triggered"],
            "lcal_v3_base_selected_mask_length": lcal_meta["base_selected_mask_length"],
            "lcal_v3_base_selected_score": lcal_meta["base_selected_score"],
            "lcal_v3_base_selected_minus_oracle_length": lcal_meta["base_selected_minus_oracle_length"],
            "lcal_v3_base_abs_selected_minus_oracle_length": lcal_meta["base_abs_selected_minus_oracle_length"],
            "lcal_v3_long_selected_mask_length": lcal_meta["correction_selected_mask_length"],
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


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rate(values: Iterable[bool]) -> Optional[float]:
    vals = list(values)
    return sum(1 for v in vals if v) / len(vals) if vals else None


def _hist(values: Iterable[Any]) -> Dict[str, int]:
    counter: Counter[str] = Counter(str(v) for v in values)
    return dict(sorted(counter.items()))


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


def _baseline_comparison(results: List[Dict[str, Any]], baseline_rows: List[Dict[str, Any]], baseline_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not baseline_rows:
        return None
    baseline_map = {row["task_id"]: row for row in baseline_rows}
    common: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
        (result, baseline_map[result["task_id"]])
        for result in results
        if result["task_id"] in baseline_map
    ]
    if not common:
        return {"baseline_results": baseline_path, "common_samples": 0}

    def new_pass(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> bool:
        return bool(pair[0]["metrics"].get("passed", False))

    def base_pass(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> bool:
        return bool(pair[1]["metrics"].get("passed", False))

    def oracle_len(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> Optional[int]:
        value = pair[0]["metrics"].get("oracle_mask_length")
        return None if value is None else int(value)

    wins = [pair for pair in common if new_pass(pair) and not base_pass(pair)]
    losses = [pair for pair in common if base_pass(pair) and not new_pass(pair)]
    short_pairs = [pair for pair in common if oracle_len(pair) is not None and int(oracle_len(pair)) <= 8]
    long_pairs = [pair for pair in common if oracle_len(pair) is not None and int(oracle_len(pair)) >= 17]
    return {
        "baseline_results": baseline_path,
        "common_samples": len(common),
        "overall_win": len(wins),
        "overall_loss": len(losses),
        "overall_tie_pass": sum(1 for pair in common if new_pass(pair) and base_pass(pair)),
        "overall_tie_fail": sum(1 for pair in common if not new_pass(pair) and not base_pass(pair)),
        "short_bucket_common_samples": len(short_pairs),
        "short_bucket_loss": sum(1 for pair in short_pairs if base_pass(pair) and not new_pass(pair)),
        "long_bucket_common_samples": len(long_pairs),
        "long_bucket_win": sum(1 for pair in long_pairs if new_pass(pair) and not base_pass(pair)),
        "win_oracle_bucket_histogram": _hist(_oracle_bucket(oracle_len(pair)) for pair in wins),
        "loss_oracle_bucket_histogram": _hist(_oracle_bucket(oracle_len(pair)) for pair in losses),
    }


def summarize_short_safe_results(results: List[Dict[str, Any]], baseline_rows: List[Dict[str, Any]], baseline_path: Optional[str]) -> Dict[str, Any]:
    summary = summarize_lcas_v3_results(results)
    metrics = [row["metrics"] for row in results]
    oracle_17plus = [m for m in metrics if m.get("oracle_mask_length") is not None and int(m["oracle_mask_length"]) >= 17]
    oracle_le8 = [m for m in metrics if m.get("oracle_mask_length") is not None and int(m["oracle_mask_length"]) <= 8]

    buckets = sorted({_oracle_bucket(m.get("oracle_mask_length")) for m in metrics})
    summary.update(
        {
            "oracle_bucket_histogram": _hist(_oracle_bucket(m.get("oracle_mask_length")) for m in metrics),
            "oracle_bucket_pass_rate": {
                bucket: _rate(bool(m.get("passed", False)) for m in metrics if _oracle_bucket(m.get("oracle_mask_length")) == bucket)
                for bucket in buckets
            },
            "short_safe_policy_histogram": _hist(m.get("short_safe_policy") for m in metrics),
            "trigger_reason_histogram_lcal": _hist(m.get("trigger_reason") for m in metrics),
            "final_source_histogram_lcal": _hist(m.get("final_source") for m in metrics),
            "strong_trigger_rate": _rate(bool(m.get("strong_triggered", False)) for m in metrics),
            "weak_trigger_rate": _rate(bool(m.get("weak_triggered", False)) for m in metrics),
            "lcal_v3_long_trigger_rate": _rate(bool(m.get("lcal_v3_long_triggered", False)) for m in metrics),
            "avg_final_minus_base_length": _avg(m.get("lcal_v3_final_minus_base_length") for m in metrics),
            "under_select_rate_17plus": _rate(
                int(m.get("selected_mask_length")) < int(m.get("oracle_mask_length")) for m in oracle_17plus
            ),
            "under_by3_rate_17plus": _rate(
                int(m.get("selected_mask_length")) <= int(m.get("oracle_mask_length")) - 3 for m in oracle_17plus
            ),
            "over_select_rate_le8": _rate(
                int(m.get("selected_mask_length")) > int(m.get("oracle_mask_length")) for m in oracle_le8
            ),
            "short_large_jump_rate": _rate(
                int(m.get("lcal_v3_final_minus_base_length", 0)) >= 4 for m in oracle_le8
            ),
        }
    )
    comparison = _baseline_comparison(results, baseline_rows, baseline_path)
    if comparison is not None:
        summary["baseline_comparison"] = comparison
    return summary


def run_experiment(cfg: ExperimentConfig, settings: ShortSafeSettings, baseline_results_path: Optional[str]) -> Dict[str, Any]:
    patch_decode(settings)
    set_global_seed(cfg.decode.seed)
    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)

    config_payload = cfg.to_dict()
    config_payload["decode"]["mask_length_source"] = f"lcal_short_safe_{settings.short_safe_policy}"
    config_payload["decode"]["lcas_policy"] = getattr(cfg.decode, "lcas_policy", None)
    config_payload["short_safe_lcal"] = asdict(settings)
    config_payload["baseline_results"] = baseline_results_path
    logger.save_config(config_payload)

    print("=" * 80, flush=True)
    print("Starting short-safe LCAL experiment", flush=True)
    print(f"experiment_name   = {cfg.logging.experiment_name}", flush=True)
    print(f"model_path        = {cfg.model.model_path}", flush=True)
    print(f"dataset_subset    = {cfg.data.dataset_subset}", flush=True)
    print(f"max_samples       = {cfg.data.max_samples}", flush=True)
    print(f"policy            = {settings.short_safe_policy}", flush=True)
    print(f"base_grid         = {settings.base_probe_lengths_csv}", flush=True)
    print(f"strong_grid       = {settings.strong_probe_lengths_csv}", flush=True)
    print(f"weak_grid         = {settings.weak_probe_lengths_csv}", flush=True)
    print(f"ratio_threshold   = {settings.ratio_trigger_threshold}", flush=True)
    print(f"score_floor       = {settings.long_score_floor}", flush=True)
    print(f"lcas_policy       = {getattr(cfg.decode, 'lcas_policy', None)}", flush=True)
    print(f"Run directory     = {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)

    tokenizer, model = load_model_and_tokenizer(cfg.model)
    tasks = load_humaneval_infilling(
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        dataset_subset=cfg.data.dataset_subset,
    )

    results: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{len(tasks)}] task_id={task.task_id} | start", flush=True)
        result = decode_lcas.run_decode_with_lcas(task, tokenizer, model, cfg)
        annotate_short_safe_result(result)
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
        ratio = m.get("long_ratio")
        print(
            f"[{idx}/{len(tasks)}] task_id={task.task_id} | "
            f"{'PASS' if m['passed'] else 'FAIL'} | "
            f"base={m.get('base_selected_length')} | final={m.get('selected_mask_length')} | "
            f"oracle={m.get('oracle_mask_length')} | diff={m.get('selected_minus_oracle_length')} | "
            f"trig={m.get('trigger_reason')} | ratio={None if ratio is None else f'{ratio:.4f}'} | "
            f"stopped={m.get('stopped')} | eff_steps={m.get('effective_steps')}",
            flush=True,
        )

    summary = summarize_short_safe_results(results, load_jsonl(baseline_results_path), baseline_results_path)
    logger.save_json("summary.json", summary)
    print("-" * 80, flush=True)
    print("Short-safe LCAL experiment finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)
    return {"results": results, "summary": summary, "run_dir": logger.run_dir}


def main() -> None:
    args = parse_args()
    settings = ShortSafeSettings(
        base_probe_lengths_csv=args.base_probe_lengths,
        base_alpha=args.base_alpha,
        strong_probe_lengths_csv=args.strong_probe_lengths,
        weak_probe_lengths_csv=args.weak_probe_lengths,
        long_alpha=args.long_alpha,
        short_safe_policy=args.short_safe_policy,
        strong_min_length=args.strong_min_length,
        weak_min_base_length=args.weak_min_base_length,
        weak_max_base_length=args.weak_max_base_length,
        ratio_trigger_threshold=args.ratio_trigger_threshold,
        long_score_floor=args.long_score_floor,
        raw_ratio_threshold=args.raw_ratio_threshold,
        support_count_threshold=args.support_count_threshold,
        cap_base_le8=args.cap_base_le8,
        cap_base_9_12=args.cap_base_9_12,
        tie_break=args.tie_break,
        score_mode=args.score_mode,
    )

    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.data.split = args.split
    cfg.data.dataset_subset = args.dataset_subset
    cfg.data.max_samples = args.max_samples
    cfg.decode.mask_length_source = f"lcal_short_safe_{args.short_safe_policy}"
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed
    cfg.decode.save_step_traces = args.save_step_traces
    cfg.decode.save_full_text_per_step = args.save_full_text_per_step
    cfg.decode.cal_lite_probe_lengths_csv = args.base_probe_lengths
    cfg.decode.cal_lite_tie_break = args.tie_break
    cfg.decode.cal_lite_score_mode = args.score_mode
    cfg.decode.cal_lite_length_alpha = args.base_alpha
    cfg.decode.lcas_policy = args.lcas_policy
    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name

    output = run_experiment(cfg, settings, args.baseline_results)
    print("===== Short-safe LCAL Summary =====")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")
    print(f"\nRun directory: {output['run_dir']}")


if __name__ == "__main__":
    main()
