#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask, compute_oracle_mask_length, load_humaneval_infilling
from expvision_dllm_clean.evaluation import summarize_results
from expvision_dllm_clean.length_probe import parse_probe_lengths, select_mask_length_cal_lite
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.runner_lcas_v3 import annotate_lcas_v3_result, choose_lcas_v3_config, summarize_lcas_v3_results

import expvision_dllm_clean.decode_lcas as decode_lcas
import run_cal_lite_lcal_rescue_policy as rescue
import run_cal_official_lcas_v3 as official_cal


DEFAULT_COMPACT_GRID = rescue.DEFAULT_COMPACT_GRID
DEFAULT_WEAK_GRID = rescue.DEFAULT_WEAK_GRID
DEFAULT_STRONG_GRID = rescue.DEFAULT_STRONG_GRID
DEFAULT_BIAS_PARAMS = official_cal.DEFAULT_BIAS_PARAMS

_ACTIVE_SETTINGS: Optional["BoundedRepairSettings"] = None
_META_BY_TASK_ID: Dict[str, Dict[str, Any]] = {}


@dataclass
class BoundedRepairSettings:
    lcal: rescue.LcalV3Settings
    official: official_cal.OfficialCalSettings
    official_eval_max_s3_len: Optional[int] = None
    repair_max_s3_len: int = 5
    repair_min_official_len: int = 6
    repair_max_official_len: int = 11
    repair_min_delta: int = 1
    repair_max_delta: int = 8
    suspicion_max_s3_len: Optional[int] = None
    suspicion_min_official_len: Optional[int] = None
    suspicion_max_official_len: int = 64
    suspicion_min_delta: int = 1
    mid_rescue_max_s3_len: Optional[int] = None
    mid_rescue_min_official_len: Optional[int] = None
    mid_rescue_max_official_len: int = 13
    mid_rescue_min_delta: int = 3
    mid_rescue_max_delta: int = 7
    mid_rescue_min_long_ratio: Optional[float] = None
    mid_rescue_source: str = "base"
    mid_rescue_min_support_count: Optional[int] = None
    mid_rescue_best_long_lens_csv: Optional[str] = None
    mid_rescue_veto_s3_le: Optional[int] = None
    mid_rescue_veto_official_len: Optional[int] = None
    true_long_max_s3_len: Optional[int] = None
    true_long_min_official_len: Optional[int] = None
    true_long_max_official_len: int = 64
    true_long_min_delta: int = 8
    true_long_min_long_ratio: Optional[float] = None
    true_long_min_raw_ratio: Optional[float] = None
    true_long_min_support_count: Optional[int] = None
    true_long_best_long_lens_csv: Optional[str] = None
    true_long_source: str = "any"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S3 short-safe LCAL, then bounded official-CAL repair, with LCAS-v3 stopping."
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
    parser.add_argument("--ratio-trigger-threshold", type=float, default=rescue.DEFAULT_RATIO_TRIGGER_THRESHOLD)
    parser.add_argument("--long-score-floor", type=float, default=rescue.DEFAULT_LONG_SCORE_FLOOR)
    parser.add_argument("--raw-ratio-threshold", type=float, default=rescue.DEFAULT_RAW_RATIO_THRESHOLD)
    parser.add_argument("--support-count-threshold", type=int, default=rescue.DEFAULT_SUPPORT_COUNT_THRESHOLD)
    parser.add_argument("--cap-base-le8", type=int, default=14)
    parser.add_argument("--cap-base-9-12", type=int, default=16)
    parser.add_argument("--short-safe-policy", type=str, default="s3", choices=["s1", "s2", "s3", "s4"])
    parser.add_argument("--tie-break", type=str, default="shorter", choices=["shorter", "longer"])
    parser.add_argument("--score-mode", type=str, default="length_power", choices=["raw", "length_power"])
    parser.add_argument(
        "--correction-selection-rule",
        type=str,
        default="shortest_supported",
        choices=["argmax", "shortest_supported"],
    )
    parser.add_argument("--shortest-supported-ratio", type=float, default=0.985)

    parser.add_argument("--official-initial-length", type=int, default=8)
    parser.add_argument("--official-span", type=int, default=1)
    parser.add_argument("--official-max-length", type=int, default=64)
    parser.add_argument("--official-dstep", type=int, default=4)
    parser.add_argument("--official-no-bias", action="store_true")
    parser.add_argument("--official-bias-params", type=str, default=DEFAULT_BIAS_PARAMS)

    parser.add_argument("--official-eval-max-s3-len", type=int, default=None)
    parser.add_argument("--repair-max-s3-len", type=int, default=5)
    parser.add_argument("--repair-min-official-len", type=int, default=6)
    parser.add_argument("--repair-max-official-len", type=int, default=11)
    parser.add_argument("--repair-min-delta", type=int, default=1)
    parser.add_argument("--repair-max-delta", type=int, default=8)
    parser.add_argument("--suspicion-max-s3-len", type=int, default=None)
    parser.add_argument("--suspicion-min-official-len", type=int, default=None)
    parser.add_argument("--suspicion-max-official-len", type=int, default=64)
    parser.add_argument("--suspicion-min-delta", type=int, default=1)
    parser.add_argument("--mid-rescue-max-s3-len", type=int, default=None)
    parser.add_argument("--mid-rescue-min-official-len", type=int, default=None)
    parser.add_argument("--mid-rescue-max-official-len", type=int, default=13)
    parser.add_argument("--mid-rescue-min-delta", type=int, default=3)
    parser.add_argument("--mid-rescue-max-delta", type=int, default=7)
    parser.add_argument("--mid-rescue-min-long-ratio", type=float, default=None)
    parser.add_argument("--mid-rescue-source", type=str, default="base")
    parser.add_argument("--mid-rescue-min-support-count", type=int, default=None)
    parser.add_argument("--mid-rescue-best-long-lens", type=str, default=None)
    parser.add_argument("--mid-rescue-veto-s3-le", type=int, default=None)
    parser.add_argument("--mid-rescue-veto-official-len", type=int, default=None)
    parser.add_argument("--true-long-max-s3-len", type=int, default=None)
    parser.add_argument("--true-long-min-official-len", type=int, default=None)
    parser.add_argument("--true-long-max-official-len", type=int, default=64)
    parser.add_argument("--true-long-min-delta", type=int, default=8)
    parser.add_argument("--true-long-min-long-ratio", type=float, default=None)
    parser.add_argument("--true-long-min-raw-ratio", type=float, default=None)
    parser.add_argument("--true-long-min-support-count", type=int, default=None)
    parser.add_argument("--true-long-best-long-lens", type=str, default=None)
    parser.add_argument("--true-long-source", type=str, default="any")

    parser.add_argument("--lcas-policy", type=str, default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    parser.add_argument("--baseline-results", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs_clean")
    parser.add_argument("--experiment-name", type=str, default="lcal_official_bounded_repair")
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


def _source_matches(source: Optional[str], allowed: str) -> bool:
    allowed = str(allowed or "").strip()
    if not allowed or allowed == "any":
        return True
    return str(source or "") in {item.strip() for item in allowed.split(",") if item.strip()}


def _parse_optional_int_set(csv_text: Optional[str]) -> Optional[set[int]]:
    if csv_text is None or not str(csv_text).strip():
        return None
    return {int(part.strip()) for part in str(csv_text).split(",") if part.strip()}


def _mid_rescue_precision_passed(
    *,
    s3_selected: int,
    official_selected: int,
    settings: BoundedRepairSettings,
    support_count: Optional[int],
    best_long_len: Optional[int],
) -> Dict[str, Any]:
    support_passed = (
        settings.mid_rescue_min_support_count is None
        or (support_count is not None and int(support_count) >= int(settings.mid_rescue_min_support_count))
    )
    allowed_lens = _parse_optional_int_set(settings.mid_rescue_best_long_lens_csv)
    best_long_len_passed = allowed_lens is None or (
        best_long_len is not None and int(best_long_len) in allowed_lens
    )
    vetoed_short_jump = (
        settings.mid_rescue_veto_s3_le is not None
        and settings.mid_rescue_veto_official_len is not None
        and int(s3_selected) <= int(settings.mid_rescue_veto_s3_le)
        and int(official_selected) >= int(settings.mid_rescue_veto_official_len)
        and not support_passed
    )
    return {
        "support_passed": bool(support_passed),
        "best_long_len_passed": bool(best_long_len_passed),
        "vetoed_short_jump": bool(vetoed_short_jump),
        "passed": bool(support_passed and best_long_len_passed and not vetoed_short_jump),
    }


def _repair_decision(
    s3_selected: int,
    official_selected: Optional[int],
    settings: BoundedRepairSettings,
    *,
    pre_repair_source: Optional[str] = None,
    long_ratio: Optional[float] = None,
    raw_long_ratio: Optional[float] = None,
    support_count: Optional[int] = None,
    best_long_len: Optional[int] = None,
) -> Dict[str, Any]:
    official_eval_max = (
        int(settings.official_eval_max_s3_len)
        if settings.official_eval_max_s3_len is not None
        else int(settings.repair_max_s3_len)
    )
    considered = int(s3_selected) <= official_eval_max
    if not considered:
        return {
            "considered": False,
            "triggered": False,
            "reason": "s3_len_above_repair_max"
            if settings.official_eval_max_s3_len is None
            else "s3_len_above_official_eval_max",
            "official_len_passed": None,
            "delta_passed": None,
            "suspicion_len_passed": None,
            "suspicion_delta_passed": None,
            "mid_rescue_len_passed": None,
            "mid_rescue_delta_passed": None,
            "mid_rescue_ratio_passed": None,
            "mid_rescue_source_passed": None,
            "mid_rescue_precision_support_passed": None,
            "mid_rescue_precision_best_long_len_passed": None,
            "mid_rescue_precision_vetoed_short_jump": None,
            "true_long_len_passed": None,
            "true_long_delta_passed": None,
            "true_long_ratio_passed": None,
            "true_long_raw_ratio_passed": None,
            "true_long_support_passed": None,
            "true_long_best_long_len_passed": None,
            "true_long_source_passed": None,
            "delta": None,
        }
    if official_selected is None:
        return {
            "considered": True,
            "triggered": False,
            "reason": "official_not_run",
            "official_len_passed": None,
            "delta_passed": None,
            "suspicion_len_passed": None,
            "suspicion_delta_passed": None,
            "mid_rescue_len_passed": None,
            "mid_rescue_delta_passed": None,
            "mid_rescue_ratio_passed": None,
            "mid_rescue_source_passed": None,
            "mid_rescue_precision_support_passed": None,
            "mid_rescue_precision_best_long_len_passed": None,
            "mid_rescue_precision_vetoed_short_jump": None,
            "true_long_len_passed": None,
            "true_long_delta_passed": None,
            "true_long_ratio_passed": None,
            "true_long_raw_ratio_passed": None,
            "true_long_support_passed": None,
            "true_long_best_long_len_passed": None,
            "true_long_source_passed": None,
            "delta": None,
        }
    delta = int(official_selected) - int(s3_selected)
    official_len_passed = (
        int(s3_selected) <= int(settings.repair_max_s3_len)
        and
        int(settings.repair_min_official_len)
        <= int(official_selected)
        <= int(settings.repair_max_official_len)
    )
    delta_passed = int(settings.repair_min_delta) <= int(delta) <= int(settings.repair_max_delta)
    bounded_triggered = bool(official_len_passed and delta_passed)

    suspicion_enabled = settings.suspicion_min_official_len is not None
    suspicion_len_passed = False
    suspicion_delta_passed = False
    suspicion_triggered = False
    if suspicion_enabled:
        suspicion_max_s3_len = (
            int(settings.suspicion_max_s3_len)
            if settings.suspicion_max_s3_len is not None
            else int(settings.repair_max_s3_len)
        )
        suspicion_len_passed = (
            int(s3_selected) <= suspicion_max_s3_len
            and
            int(settings.suspicion_min_official_len)
            <= int(official_selected)
            <= int(settings.suspicion_max_official_len)
        )
        suspicion_delta_passed = int(delta) >= int(settings.suspicion_min_delta)
        suspicion_triggered = bool(suspicion_len_passed and suspicion_delta_passed)

    mid_rescue_enabled = settings.mid_rescue_min_official_len is not None
    mid_rescue_len_passed = False
    mid_rescue_delta_passed = False
    mid_rescue_ratio_passed = False
    mid_rescue_source_passed = False
    mid_rescue_precision = {
        "support_passed": True,
        "best_long_len_passed": True,
        "vetoed_short_jump": False,
        "passed": True,
    }
    mid_rescue_triggered = False
    if mid_rescue_enabled:
        mid_rescue_max_s3_len = (
            int(settings.mid_rescue_max_s3_len)
            if settings.mid_rescue_max_s3_len is not None
            else official_eval_max
        )
        mid_rescue_len_passed = (
            int(s3_selected) <= mid_rescue_max_s3_len
            and int(settings.mid_rescue_min_official_len)
            <= int(official_selected)
            <= int(settings.mid_rescue_max_official_len)
        )
        mid_rescue_delta_passed = (
            int(settings.mid_rescue_min_delta) <= int(delta) <= int(settings.mid_rescue_max_delta)
        )
        mid_rescue_ratio_passed = (
            settings.mid_rescue_min_long_ratio is None
            or (long_ratio is not None and float(long_ratio) >= float(settings.mid_rescue_min_long_ratio))
        )
        mid_rescue_source_passed = _source_matches(pre_repair_source, settings.mid_rescue_source)
        mid_rescue_precision = _mid_rescue_precision_passed(
            s3_selected=s3_selected,
            official_selected=int(official_selected),
            settings=settings,
            support_count=support_count,
            best_long_len=best_long_len,
        )
        mid_rescue_triggered = bool(
            mid_rescue_len_passed
            and mid_rescue_delta_passed
            and mid_rescue_ratio_passed
            and mid_rescue_source_passed
            and mid_rescue_precision["passed"]
        )

    true_long_enabled = settings.true_long_min_official_len is not None
    true_long_len_passed = False
    true_long_delta_passed = False
    true_long_ratio_passed = False
    true_long_raw_ratio_passed = False
    true_long_support_passed = False
    true_long_best_long_len_passed = False
    true_long_source_passed = False
    true_long_triggered = False
    if true_long_enabled:
        true_long_max_s3_len = (
            int(settings.true_long_max_s3_len)
            if settings.true_long_max_s3_len is not None
            else official_eval_max
        )
        true_long_len_passed = (
            int(s3_selected) <= true_long_max_s3_len
            and int(settings.true_long_min_official_len)
            <= int(official_selected)
            <= int(settings.true_long_max_official_len)
        )
        true_long_delta_passed = int(delta) >= int(settings.true_long_min_delta)
        true_long_ratio_passed = (
            settings.true_long_min_long_ratio is None
            or (long_ratio is not None and float(long_ratio) >= float(settings.true_long_min_long_ratio))
        )
        true_long_raw_ratio_passed = (
            settings.true_long_min_raw_ratio is None
            or (
                raw_long_ratio is not None
                and float(raw_long_ratio) >= float(settings.true_long_min_raw_ratio)
            )
        )
        true_long_support_passed = (
            settings.true_long_min_support_count is None
            or (support_count is not None and int(support_count) >= int(settings.true_long_min_support_count))
        )
        true_long_lens = _parse_optional_int_set(settings.true_long_best_long_lens_csv)
        true_long_best_long_len_passed = true_long_lens is None or (
            best_long_len is not None and int(best_long_len) in true_long_lens
        )
        true_long_source_passed = _source_matches(pre_repair_source, settings.true_long_source)
        true_long_triggered = bool(
            true_long_len_passed
            and true_long_delta_passed
            and true_long_ratio_passed
            and true_long_raw_ratio_passed
            and true_long_support_passed
            and true_long_best_long_len_passed
            and true_long_source_passed
        )

    triggered = bool(bounded_triggered or suspicion_triggered or mid_rescue_triggered or true_long_triggered)
    if triggered:
        if bounded_triggered:
            reason = "official_bounded_repair"
        elif suspicion_triggered:
            reason = "official_long_suspicion"
        elif mid_rescue_triggered:
            reason = "official_mid_rescue"
        else:
            reason = "official_true_long_rescue"
    elif not official_len_passed:
        reason = "official_len_out_of_bounds"
    else:
        reason = "official_delta_out_of_bounds"
    return {
        "considered": True,
        "triggered": triggered,
        "reason": reason,
        "official_len_passed": bool(official_len_passed),
        "delta_passed": bool(delta_passed),
        "suspicion_len_passed": bool(suspicion_len_passed) if suspicion_enabled else None,
        "suspicion_delta_passed": bool(suspicion_delta_passed) if suspicion_enabled else None,
        "mid_rescue_len_passed": bool(mid_rescue_len_passed) if mid_rescue_enabled else None,
        "mid_rescue_delta_passed": bool(mid_rescue_delta_passed) if mid_rescue_enabled else None,
        "mid_rescue_ratio_passed": bool(mid_rescue_ratio_passed) if mid_rescue_enabled else None,
        "mid_rescue_source_passed": bool(mid_rescue_source_passed) if mid_rescue_enabled else None,
        "mid_rescue_precision_support_passed": (
            bool(mid_rescue_precision["support_passed"]) if mid_rescue_enabled else None
        ),
        "mid_rescue_precision_best_long_len_passed": (
            bool(mid_rescue_precision["best_long_len_passed"]) if mid_rescue_enabled else None
        ),
        "mid_rescue_precision_vetoed_short_jump": (
            bool(mid_rescue_precision["vetoed_short_jump"]) if mid_rescue_enabled else None
        ),
        "true_long_len_passed": bool(true_long_len_passed) if true_long_enabled else None,
        "true_long_delta_passed": bool(true_long_delta_passed) if true_long_enabled else None,
        "true_long_ratio_passed": bool(true_long_ratio_passed) if true_long_enabled else None,
        "true_long_raw_ratio_passed": bool(true_long_raw_ratio_passed) if true_long_enabled else None,
        "true_long_support_passed": bool(true_long_support_passed) if true_long_enabled else None,
        "true_long_best_long_len_passed": (
            bool(true_long_best_long_len_passed) if true_long_enabled else None
        ),
        "true_long_source_passed": bool(true_long_source_passed) if true_long_enabled else None,
        "delta": int(delta),
    }


def _run_s3_selection(task: CodeTask, tokenizer, model, cfg: ExperimentConfig, settings: BoundedRepairSettings) -> Dict[str, Any]:
    lcal = settings.lcal
    oracle_mask_length = compute_oracle_mask_length(task, tokenizer, add_special_tokens=False)

    base_cfg = rescue._clone_probe_cfg(
        cfg=cfg,
        probe_lengths_csv=lcal.base_probe_lengths_csv,
        alpha=lcal.base_alpha,
        settings=lcal,
    )
    base_selection = select_mask_length_cal_lite(task, tokenizer, model, base_cfg)
    base_selected = int(base_selection["selected_mask_length"])
    best_info = rescue._base_best_info(base_selection)
    trigger = rescue._trigger_decision(base_selected, best_info, lcal)

    long_selection: Optional[Dict[str, Any]] = None
    final_source = "base"
    final_selection = base_selection
    total_probe_sec = float(base_selection["length_probe_sec"])

    if bool(trigger["triggered"]):
        long_cfg = rescue._clone_probe_cfg(
            cfg=cfg,
            probe_lengths_csv=str(trigger["correction_probe_lengths_csv"]),
            alpha=lcal.long_alpha,
            settings=lcal,
        )
        raw_long_selection = select_mask_length_cal_lite(task, tokenizer, model, long_cfg)
        long_selection = rescue._apply_correction_selection(
            correction_selection=raw_long_selection,
            base_selected_length=base_selected,
            settings=lcal,
        )
        if int(long_selection["selected_mask_length"]) >= base_selected:
            final_selection = long_selection
            final_source = "strong_correction" if trigger["strong_triggered"] else "weak_correction"
        else:
            final_selection = base_selection
            final_source = "base_max_after_long"
        total_probe_sec += float(long_selection["length_probe_sec"])

    s3_selected = int(final_selection["selected_mask_length"])
    if long_selection is not None:
        s3_selected = max(base_selected, int(long_selection["selected_mask_length"]))
    diff = rescue._length_diff(s3_selected, None if oracle_mask_length is None else int(oracle_mask_length))
    base_diff = rescue._length_diff(base_selected, None if oracle_mask_length is None else int(oracle_mask_length))

    meta: Dict[str, Any] = {
        "version": "lcal_official_bounded_repair",
        "short_safe_policy": lcal.short_safe_policy,
        "pre_repair_source": final_source,
        "final_source": final_source,
        "long_triggered": bool(trigger["triggered"]),
        "strong_triggered": bool(trigger["strong_triggered"]),
        "weak_triggered": bool(trigger["weak_triggered"]),
        "correction_kind": trigger["correction_kind"],
        "correction_selection_rule": lcal.correction_selection_rule,
        "shortest_supported_ratio": float(lcal.shortest_supported_ratio),
        "correction_probe_lengths": (
            None if trigger["correction_probe_lengths_csv"] is None else parse_probe_lengths(str(trigger["correction_probe_lengths_csv"]))
        ),
        "long_trigger_reasons": [trigger["trigger_reason"]] if trigger["triggered"] else [],
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
        "ratio_trigger_threshold": float(lcal.ratio_trigger_threshold),
        "long_score_floor": float(lcal.long_score_floor),
        "raw_ratio_threshold": float(lcal.raw_ratio_threshold),
        "support_count_threshold": int(lcal.support_count_threshold),
        "strong_min_len": int(lcal.strong_min_len),
        "weak_min_base_len": int(lcal.weak_min_base_len),
        "weak_max_base_len": int(lcal.weak_max_base_len),
        "cap_base_le8": int(lcal.cap_base_le8),
        "cap_base_9_12": int(lcal.cap_base_9_12),
        "base_probe_lengths": parse_probe_lengths(lcal.base_probe_lengths_csv),
        "base_alpha": float(lcal.base_alpha),
        "base_selected_length": base_selected,
        "base_selected_mask_length": base_selected,
        "base_selected_score": float(base_selection["selected_score"]),
        "base_selected_raw_score": float(base_selection["selected_raw_score"]),
        "base_selected_adjusted_score": float(base_selection["selected_adjusted_score"]),
        "base_length_probe_sec": float(base_selection["length_probe_sec"]),
        "base_selected_minus_oracle_length": base_diff["selected_minus_oracle_length"],
        "base_abs_selected_minus_oracle_length": base_diff["abs_selected_minus_oracle_length"],
        "base_candidate_scores": base_selection["candidate_scores"],
        "weak_probe_lengths": parse_probe_lengths(lcal.weak_probe_lengths_csv),
        "strong_probe_lengths": parse_probe_lengths(lcal.strong_probe_lengths_csv),
        "long_probe_lengths": (
            None if trigger["correction_probe_lengths_csv"] is None else parse_probe_lengths(str(trigger["correction_probe_lengths_csv"]))
        ),
        "long_alpha": float(lcal.long_alpha),
        "long_selected_mask_length": None,
        "long_selected_score": None,
        "long_selected_raw_score": None,
        "long_selected_adjusted_score": None,
        "long_length_probe_sec": 0.0,
        "long_candidate_scores": None,
        "long_selected_length": None,
        "long_argmax_selected_length": None,
        "pre_repair_selected_length": s3_selected,
        "s3_selected_length": s3_selected,
        "final_selected_length": s3_selected,
        "final_minus_base_length": s3_selected - base_selected,
        "final_minus_s3_length": 0,
        "official_repair_considered": False,
        "official_repair_triggered": False,
        "official_repair_reason": "not_evaluated",
        "official_repair_delta": None,
        "official_repair_official_len_passed": None,
        "official_repair_delta_passed": None,
        "official_repair_suspicion_len_passed": None,
        "official_repair_suspicion_delta_passed": None,
        "official_repair_mid_rescue_len_passed": None,
        "official_repair_mid_rescue_delta_passed": None,
        "official_repair_mid_rescue_ratio_passed": None,
        "official_repair_mid_rescue_source_passed": None,
        "official_repair_mid_rescue_precision_support_passed": None,
        "official_repair_mid_rescue_precision_best_long_len_passed": None,
        "official_repair_mid_rescue_precision_vetoed_short_jump": None,
        "official_repair_true_long_len_passed": None,
        "official_repair_true_long_delta_passed": None,
        "official_repair_true_long_ratio_passed": None,
        "official_repair_true_long_raw_ratio_passed": None,
        "official_repair_true_long_support_passed": None,
        "official_repair_true_long_best_long_len_passed": None,
        "official_repair_true_long_source_passed": None,
        "official_eval_max_s3_len": settings.official_eval_max_s3_len,
        "repair_max_s3_len": int(settings.repair_max_s3_len),
        "repair_min_official_len": int(settings.repair_min_official_len),
        "repair_max_official_len": int(settings.repair_max_official_len),
        "repair_min_delta": int(settings.repair_min_delta),
        "repair_max_delta": int(settings.repair_max_delta),
        "suspicion_max_s3_len": settings.suspicion_max_s3_len,
        "suspicion_min_official_len": settings.suspicion_min_official_len,
        "suspicion_max_official_len": int(settings.suspicion_max_official_len),
        "suspicion_min_delta": int(settings.suspicion_min_delta),
        "mid_rescue_max_s3_len": settings.mid_rescue_max_s3_len,
        "mid_rescue_min_official_len": settings.mid_rescue_min_official_len,
        "mid_rescue_max_official_len": int(settings.mid_rescue_max_official_len),
        "mid_rescue_min_delta": int(settings.mid_rescue_min_delta),
        "mid_rescue_max_delta": int(settings.mid_rescue_max_delta),
        "mid_rescue_min_long_ratio": settings.mid_rescue_min_long_ratio,
        "mid_rescue_source": settings.mid_rescue_source,
        "mid_rescue_min_support_count": settings.mid_rescue_min_support_count,
        "mid_rescue_best_long_lens_csv": settings.mid_rescue_best_long_lens_csv,
        "mid_rescue_veto_s3_le": settings.mid_rescue_veto_s3_le,
        "mid_rescue_veto_official_len": settings.mid_rescue_veto_official_len,
        "true_long_max_s3_len": settings.true_long_max_s3_len,
        "true_long_min_official_len": settings.true_long_min_official_len,
        "true_long_max_official_len": int(settings.true_long_max_official_len),
        "true_long_min_delta": int(settings.true_long_min_delta),
        "true_long_min_long_ratio": settings.true_long_min_long_ratio,
        "true_long_min_raw_ratio": settings.true_long_min_raw_ratio,
        "true_long_min_support_count": settings.true_long_min_support_count,
        "true_long_best_long_lens_csv": settings.true_long_best_long_lens_csv,
        "true_long_source": settings.true_long_source,
        "official_selected_length": None,
        "official_selected_score": None,
        "official_selected_raw_score": None,
        "official_selected_adjusted_score": None,
        "official_length_probe_sec": 0.0,
        "official_search_steps": None,
        "official_candidate_scores": None,
        "official_cal": None,
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

    return {
        "oracle_mask_length": None if oracle_mask_length is None else int(oracle_mask_length),
        "base_selection": base_selection,
        "final_selection": final_selection,
        "s3_selected": s3_selected,
        "base_selected": base_selected,
        "total_probe_sec": total_probe_sec,
        "meta": meta,
        "diff": diff,
    }


def resolve_mask_length_bounded_repair(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    if _ACTIVE_SETTINGS is None:
        raise RuntimeError("Bounded repair settings were not initialized")
    settings = _ACTIVE_SETTINGS

    s3 = _run_s3_selection(task, tokenizer, model, cfg, settings)
    meta = s3["meta"]
    final_selection = s3["final_selection"]
    selected = int(s3["s3_selected"])
    total_probe_sec = float(s3["total_probe_sec"])

    official_selection: Optional[Dict[str, Any]] = None
    decision = _repair_decision(
        selected,
        None,
        settings,
        pre_repair_source=meta.get("pre_repair_source"),
        long_ratio=meta.get("long_ratio"),
        raw_long_ratio=meta.get("raw_long_ratio"),
        support_count=meta.get("support_count"),
        best_long_len=meta.get("best_long_len"),
    )
    if bool(decision["considered"]):
        official_selection = official_cal.select_mask_length_official_cal(task, tokenizer, model, cfg)
        official_selected = int(official_selection["selected_mask_length"])
        decision = _repair_decision(
            selected,
            official_selected,
            settings,
            pre_repair_source=meta.get("pre_repair_source"),
            long_ratio=meta.get("long_ratio"),
            raw_long_ratio=meta.get("raw_long_ratio"),
            support_count=meta.get("support_count"),
            best_long_len=meta.get("best_long_len"),
        )
        total_probe_sec += float(official_selection["length_probe_sec"])
        official_meta = copy.deepcopy(official_cal._CAL_META_BY_TASK_ID.get(task.task_id, {}))
        meta.update(
            {
                "official_selected_length": official_selected,
                "official_selected_score": float(official_selection["selected_score"]),
                "official_selected_raw_score": float(official_selection["selected_raw_score"]),
                "official_selected_adjusted_score": float(official_selection["selected_adjusted_score"]),
                "official_length_probe_sec": float(official_selection["length_probe_sec"]),
                "official_search_steps": official_meta.get("search_steps"),
                "official_candidate_scores": official_selection["candidate_scores"],
                "official_cal": official_meta,
            }
        )
        if bool(decision["triggered"]):
            final_selection = official_selection
            selected = official_selected
            meta["final_source"] = str(decision["reason"])

    meta.update(
        {
            "official_repair_considered": bool(decision["considered"]),
            "official_repair_triggered": bool(decision["triggered"]),
            "official_repair_reason": str(decision["reason"]),
            "official_repair_delta": decision["delta"],
            "official_repair_official_len_passed": decision["official_len_passed"],
            "official_repair_delta_passed": decision["delta_passed"],
            "official_repair_suspicion_len_passed": decision["suspicion_len_passed"],
            "official_repair_suspicion_delta_passed": decision["suspicion_delta_passed"],
            "official_repair_mid_rescue_len_passed": decision["mid_rescue_len_passed"],
            "official_repair_mid_rescue_delta_passed": decision["mid_rescue_delta_passed"],
            "official_repair_mid_rescue_ratio_passed": decision["mid_rescue_ratio_passed"],
            "official_repair_mid_rescue_source_passed": decision["mid_rescue_source_passed"],
            "official_repair_mid_rescue_precision_support_passed": decision[
                "mid_rescue_precision_support_passed"
            ],
            "official_repair_mid_rescue_precision_best_long_len_passed": decision[
                "mid_rescue_precision_best_long_len_passed"
            ],
            "official_repair_mid_rescue_precision_vetoed_short_jump": decision[
                "mid_rescue_precision_vetoed_short_jump"
            ],
            "official_repair_true_long_len_passed": decision["true_long_len_passed"],
            "official_repair_true_long_delta_passed": decision["true_long_delta_passed"],
            "official_repair_true_long_ratio_passed": decision["true_long_ratio_passed"],
            "official_repair_true_long_raw_ratio_passed": decision["true_long_raw_ratio_passed"],
            "official_repair_true_long_support_passed": decision["true_long_support_passed"],
            "official_repair_true_long_best_long_len_passed": decision[
                "true_long_best_long_len_passed"
            ],
            "official_repair_true_long_source_passed": decision["true_long_source_passed"],
            "final_selected_length": selected,
            "final_minus_base_length": selected - int(s3["base_selected"]),
            "final_minus_s3_length": selected - int(s3["s3_selected"]),
        }
    )

    diff = rescue._length_diff(selected, s3["oracle_mask_length"])
    output = {
        "oracle_mask_length": s3["oracle_mask_length"],
        "mask_length_source": "lcal_official_bounded_repair",
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


def patch_decode(settings: BoundedRepairSettings) -> None:
    global _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = settings
    rescue._ACTIVE_SETTINGS = settings.lcal
    official_cal._ACTIVE_SETTINGS = settings.official
    _META_BY_TASK_ID.clear()
    official_cal._CAL_META_BY_TASK_ID.clear()
    decode_lcas.resolve_mask_length = resolve_mask_length_bounded_repair
    decode_lcas.choose_lcas_config = choose_lcas_v3_config


def annotate_result(result: Dict[str, Any]) -> None:
    meta = _META_BY_TASK_ID.pop(result["task_id"], None)
    if meta is None:
        return

    lcal_meta = meta["lcal_v3"]
    result["lcal_v3"] = lcal_meta
    result["official_cal"] = copy.deepcopy(lcal_meta.get("official_cal") or {})

    result["length_probe"].update(
        {
            "lcal_v3": {
                key: value
                for key, value in lcal_meta.items()
                if key not in {"base_candidate_scores", "long_candidate_scores", "official_candidate_scores"}
            },
            "selected_raw_score": meta["selected_raw_score"],
            "selected_adjusted_score": meta["selected_adjusted_score"],
            "base_candidate_scores": lcal_meta["base_candidate_scores"],
            "long_candidate_scores": lcal_meta["long_candidate_scores"],
            "official_candidate_scores": lcal_meta["official_candidate_scores"],
            "official_cal": result["official_cal"],
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
            "lcal_v3_base_abs_selected_minus_oracle_length": lcal_meta["base_abs_selected_minus_oracle_length"],
            "lcal_v3_long_selected_mask_length": lcal_meta["long_selected_mask_length"],
            "lcal_v3_final_minus_base_length": lcal_meta["final_minus_base_length"],
            "pre_repair_selected_length": lcal_meta["pre_repair_selected_length"],
            "s3_selected_length": lcal_meta["s3_selected_length"],
            "lcal_v3_final_minus_s3_length": lcal_meta["final_minus_s3_length"],
            "official_repair_considered": lcal_meta["official_repair_considered"],
            "official_repair_triggered": lcal_meta["official_repair_triggered"],
            "official_repair_reason": lcal_meta["official_repair_reason"],
            "official_repair_delta": lcal_meta["official_repair_delta"],
            "official_repair_official_len_passed": lcal_meta["official_repair_official_len_passed"],
            "official_repair_delta_passed": lcal_meta["official_repair_delta_passed"],
            "official_repair_suspicion_len_passed": lcal_meta["official_repair_suspicion_len_passed"],
            "official_repair_suspicion_delta_passed": lcal_meta["official_repair_suspicion_delta_passed"],
            "official_repair_mid_rescue_len_passed": lcal_meta["official_repair_mid_rescue_len_passed"],
            "official_repair_mid_rescue_delta_passed": lcal_meta["official_repair_mid_rescue_delta_passed"],
            "official_repair_mid_rescue_ratio_passed": lcal_meta["official_repair_mid_rescue_ratio_passed"],
            "official_repair_mid_rescue_source_passed": lcal_meta["official_repair_mid_rescue_source_passed"],
            "official_repair_mid_rescue_precision_support_passed": lcal_meta[
                "official_repair_mid_rescue_precision_support_passed"
            ],
            "official_repair_mid_rescue_precision_best_long_len_passed": lcal_meta[
                "official_repair_mid_rescue_precision_best_long_len_passed"
            ],
            "official_repair_mid_rescue_precision_vetoed_short_jump": lcal_meta[
                "official_repair_mid_rescue_precision_vetoed_short_jump"
            ],
            "official_repair_true_long_len_passed": lcal_meta["official_repair_true_long_len_passed"],
            "official_repair_true_long_delta_passed": lcal_meta[
                "official_repair_true_long_delta_passed"
            ],
            "official_repair_true_long_ratio_passed": lcal_meta[
                "official_repair_true_long_ratio_passed"
            ],
            "official_repair_true_long_raw_ratio_passed": lcal_meta[
                "official_repair_true_long_raw_ratio_passed"
            ],
            "official_repair_true_long_support_passed": lcal_meta[
                "official_repair_true_long_support_passed"
            ],
            "official_repair_true_long_best_long_len_passed": lcal_meta[
                "official_repair_true_long_best_long_len_passed"
            ],
            "official_repair_true_long_source_passed": lcal_meta[
                "official_repair_true_long_source_passed"
            ],
            "official_eval_max_s3_len": lcal_meta["official_eval_max_s3_len"],
            "suspicion_max_s3_len": lcal_meta["suspicion_max_s3_len"],
            "suspicion_min_official_len": lcal_meta["suspicion_min_official_len"],
            "suspicion_max_official_len": lcal_meta["suspicion_max_official_len"],
            "suspicion_min_delta": lcal_meta["suspicion_min_delta"],
            "mid_rescue_max_s3_len": lcal_meta["mid_rescue_max_s3_len"],
            "mid_rescue_min_official_len": lcal_meta["mid_rescue_min_official_len"],
            "mid_rescue_max_official_len": lcal_meta["mid_rescue_max_official_len"],
            "mid_rescue_min_delta": lcal_meta["mid_rescue_min_delta"],
            "mid_rescue_max_delta": lcal_meta["mid_rescue_max_delta"],
            "mid_rescue_min_long_ratio": lcal_meta["mid_rescue_min_long_ratio"],
            "mid_rescue_source": lcal_meta["mid_rescue_source"],
            "mid_rescue_min_support_count": lcal_meta["mid_rescue_min_support_count"],
            "mid_rescue_best_long_lens_csv": lcal_meta["mid_rescue_best_long_lens_csv"],
            "mid_rescue_veto_s3_le": lcal_meta["mid_rescue_veto_s3_le"],
            "mid_rescue_veto_official_len": lcal_meta["mid_rescue_veto_official_len"],
            "true_long_max_s3_len": lcal_meta["true_long_max_s3_len"],
            "true_long_min_official_len": lcal_meta["true_long_min_official_len"],
            "true_long_max_official_len": lcal_meta["true_long_max_official_len"],
            "true_long_min_delta": lcal_meta["true_long_min_delta"],
            "true_long_min_long_ratio": lcal_meta["true_long_min_long_ratio"],
            "true_long_min_raw_ratio": lcal_meta["true_long_min_raw_ratio"],
            "true_long_min_support_count": lcal_meta["true_long_min_support_count"],
            "true_long_best_long_lens_csv": lcal_meta["true_long_best_long_lens_csv"],
            "true_long_source": lcal_meta["true_long_source"],
            "official_selected_length": lcal_meta["official_selected_length"],
            "official_selected_score": lcal_meta["official_selected_score"],
            "official_selected_raw_score": lcal_meta["official_selected_raw_score"],
            "official_selected_adjusted_score": lcal_meta["official_selected_adjusted_score"],
            "official_length_probe_sec": lcal_meta["official_length_probe_sec"],
            "official_search_steps": lcal_meta["official_search_steps"],
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


def summarize_bounded_repair_results(
    results: List[Dict[str, Any]],
    baseline_rows: Optional[List[Dict[str, Any]]] = None,
    baseline_path: Optional[str] = None,
) -> Dict[str, Any]:
    summary = rescue.summarize_lcal_v3_results(
        results=results,
        baseline_rows=baseline_rows or [],
        baseline_path=baseline_path,
    )
    metrics = [item["metrics"] for item in results]
    repair_considered = [m for m in metrics if bool(m.get("official_repair_considered", False))]
    repair_triggered = [m for m in metrics if bool(m.get("official_repair_triggered", False))]
    suspicion_triggered = [
        m for m in repair_triggered if str(m.get("official_repair_reason")) == "official_long_suspicion"
    ]
    mid_rescue_triggered = [
        m for m in repair_triggered if str(m.get("official_repair_reason")) == "official_mid_rescue"
    ]
    true_long_triggered = [
        m for m in repair_triggered if str(m.get("official_repair_reason")) == "official_true_long_rescue"
    ]
    summary.update(
        {
            "mask_length_source": "lcal_official_bounded_repair",
            "official_repair_considered_count": len(repair_considered),
            "official_repair_considered_rate": len(repair_considered) / len(metrics) if metrics else None,
            "official_repair_trigger_count": len(repair_triggered),
            "official_repair_trigger_rate": len(repair_triggered) / len(metrics) if metrics else None,
            "official_long_suspicion_trigger_count": len(suspicion_triggered),
            "official_long_suspicion_trigger_rate": len(suspicion_triggered) / len(metrics) if metrics else None,
            "official_mid_rescue_trigger_count": len(mid_rescue_triggered),
            "official_mid_rescue_trigger_rate": len(mid_rescue_triggered) / len(metrics) if metrics else None,
            "official_true_long_rescue_trigger_count": len(true_long_triggered),
            "official_true_long_rescue_trigger_rate": len(true_long_triggered) / len(metrics) if metrics else None,
            "official_repair_reason_histogram": _hist(m.get("official_repair_reason") for m in metrics),
            "official_repair_source_histogram": _hist(m.get("final_source") for m in metrics),
            "official_repair_oracle_bucket_histogram": _hist(
                rescue._oracle_bucket(m.get("oracle_mask_length")) for m in repair_triggered
            ),
            "official_long_suspicion_oracle_bucket_histogram": _hist(
                rescue._oracle_bucket(m.get("oracle_mask_length")) for m in suspicion_triggered
            ),
            "official_mid_rescue_oracle_bucket_histogram": _hist(
                rescue._oracle_bucket(m.get("oracle_mask_length")) for m in mid_rescue_triggered
            ),
            "official_true_long_rescue_oracle_bucket_histogram": _hist(
                rescue._oracle_bucket(m.get("oracle_mask_length")) for m in true_long_triggered
            ),
            "official_repair_pass_rate": _rate(bool(m.get("passed", False)) for m in repair_triggered),
            "official_long_suspicion_pass_rate": _rate(
                bool(m.get("passed", False)) for m in suspicion_triggered
            ),
            "official_mid_rescue_pass_rate": _rate(
                bool(m.get("passed", False)) for m in mid_rescue_triggered
            ),
            "official_true_long_rescue_pass_rate": _rate(
                bool(m.get("passed", False)) for m in true_long_triggered
            ),
            "official_repair_true_long_precision": _rate(
                int(m.get("oracle_mask_length")) >= 17
                for m in repair_triggered
                if m.get("oracle_mask_length") is not None
            ),
            "official_long_suspicion_true_long_precision": _rate(
                int(m.get("oracle_mask_length")) >= 17
                for m in suspicion_triggered
                if m.get("oracle_mask_length") is not None
            ),
            "official_mid_rescue_true_long_precision": _rate(
                int(m.get("oracle_mask_length")) >= 17
                for m in mid_rescue_triggered
                if m.get("oracle_mask_length") is not None
            ),
            "official_true_long_rescue_true_long_precision": _rate(
                int(m.get("oracle_mask_length")) >= 17
                for m in true_long_triggered
                if m.get("oracle_mask_length") is not None
            ),
            "official_repair_nonshort_precision": _rate(
                int(m.get("oracle_mask_length")) >= 9
                for m in repair_triggered
                if m.get("oracle_mask_length") is not None
            ),
            "official_repair_avg_final_minus_s3_length": _avg(
                m.get("lcal_v3_final_minus_s3_length") for m in repair_triggered
            ),
            "official_repair_avg_search_steps": _avg(m.get("official_search_steps") for m in repair_considered),
        }
    )
    summary["required_metrics"]["official_repair_trigger_rate"] = summary["official_repair_trigger_rate"]
    summary["required_metrics"]["official_long_suspicion_trigger_rate"] = summary[
        "official_long_suspicion_trigger_rate"
    ]
    summary["required_metrics"]["official_mid_rescue_trigger_rate"] = summary[
        "official_mid_rescue_trigger_rate"
    ]
    summary["required_metrics"]["official_true_long_rescue_trigger_rate"] = summary[
        "official_true_long_rescue_trigger_rate"
    ]
    summary["required_metrics"]["official_repair_true_long_precision"] = summary[
        "official_repair_true_long_precision"
    ]
    summary["required_metrics"]["official_long_suspicion_true_long_precision"] = summary[
        "official_long_suspicion_true_long_precision"
    ]
    summary["required_metrics"]["official_mid_rescue_true_long_precision"] = summary[
        "official_mid_rescue_true_long_precision"
    ]
    summary["required_metrics"]["official_true_long_rescue_true_long_precision"] = summary[
        "official_true_long_rescue_true_long_precision"
    ]
    summary["required_metrics"]["official_repair_nonshort_precision"] = summary[
        "official_repair_nonshort_precision"
    ]
    return summary


def run_experiment(
    cfg: ExperimentConfig,
    settings: BoundedRepairSettings,
    baseline_results_path: Optional[str] = None,
) -> Dict[str, Any]:
    patch_decode(settings)
    set_global_seed(cfg.decode.seed)
    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)

    config_payload = cfg.to_dict()
    config_payload["decode"]["mask_length_source"] = "lcal_official_bounded_repair"
    config_payload["decode"]["lcas_policy"] = getattr(cfg.decode, "lcas_policy", None)
    config_payload["lcal_official_bounded_repair"] = {
        "lcal": asdict(settings.lcal),
        "official": {
            **asdict(settings.official),
            "bias_params": list(settings.official.bias_params),
        },
        "repair": {
            "official_eval_max_s3_len": settings.official_eval_max_s3_len,
            "repair_max_s3_len": settings.repair_max_s3_len,
            "repair_min_official_len": settings.repair_min_official_len,
            "repair_max_official_len": settings.repair_max_official_len,
            "repair_min_delta": settings.repair_min_delta,
            "repair_max_delta": settings.repair_max_delta,
            "suspicion_max_s3_len": settings.suspicion_max_s3_len,
            "suspicion_min_official_len": settings.suspicion_min_official_len,
            "suspicion_max_official_len": settings.suspicion_max_official_len,
            "suspicion_min_delta": settings.suspicion_min_delta,
            "mid_rescue_max_s3_len": settings.mid_rescue_max_s3_len,
            "mid_rescue_min_official_len": settings.mid_rescue_min_official_len,
            "mid_rescue_max_official_len": settings.mid_rescue_max_official_len,
            "mid_rescue_min_delta": settings.mid_rescue_min_delta,
            "mid_rescue_max_delta": settings.mid_rescue_max_delta,
            "mid_rescue_min_long_ratio": settings.mid_rescue_min_long_ratio,
            "mid_rescue_source": settings.mid_rescue_source,
            "mid_rescue_min_support_count": settings.mid_rescue_min_support_count,
            "mid_rescue_best_long_lens_csv": settings.mid_rescue_best_long_lens_csv,
            "mid_rescue_veto_s3_le": settings.mid_rescue_veto_s3_le,
            "mid_rescue_veto_official_len": settings.mid_rescue_veto_official_len,
            "true_long_max_s3_len": settings.true_long_max_s3_len,
            "true_long_min_official_len": settings.true_long_min_official_len,
            "true_long_max_official_len": settings.true_long_max_official_len,
            "true_long_min_delta": settings.true_long_min_delta,
            "true_long_min_long_ratio": settings.true_long_min_long_ratio,
            "true_long_min_raw_ratio": settings.true_long_min_raw_ratio,
            "true_long_min_support_count": settings.true_long_min_support_count,
            "true_long_best_long_lens_csv": settings.true_long_best_long_lens_csv,
            "true_long_source": settings.true_long_source,
        },
    }
    config_payload["baseline_results"] = baseline_results_path
    logger.save_config(config_payload)

    print("=" * 80, flush=True)
    print("Starting LCAL + official bounded repair experiment", flush=True)
    print(f"experiment_name       = {cfg.logging.experiment_name}", flush=True)
    print(f"output_dir            = {cfg.logging.output_dir}", flush=True)
    print(f"dataset_subset        = {cfg.data.dataset_subset}", flush=True)
    print(f"split                 = {cfg.data.split}", flush=True)
    print(f"max_samples           = {cfg.data.max_samples}", flush=True)
    print(f"model_path            = {cfg.model.model_path}", flush=True)
    print(f"short_policy          = {settings.lcal.short_safe_policy}", flush=True)
    print(f"correction_rule       = {settings.lcal.correction_selection_rule}", flush=True)
    print(f"base_grid             = {settings.lcal.base_probe_lengths_csv}", flush=True)
    print(f"base_alpha            = {settings.lcal.base_alpha}", flush=True)
    print(f"long_alpha            = {settings.lcal.long_alpha}", flush=True)
    print(f"official_eval_s3_len  <= {settings.official_eval_max_s3_len}", flush=True)
    print(f"repair_s3_len         <= {settings.repair_max_s3_len}", flush=True)
    print(f"repair_official_len   = {settings.repair_min_official_len}..{settings.repair_max_official_len}", flush=True)
    print(f"repair_delta          = {settings.repair_min_delta}..{settings.repair_max_delta}", flush=True)
    print(f"suspicion_s3_len      <= {settings.suspicion_max_s3_len}", flush=True)
    print(f"suspicion_official_len >= {settings.suspicion_min_official_len}", flush=True)
    print(f"suspicion_max_len     = {settings.suspicion_max_official_len}", flush=True)
    print(f"suspicion_min_delta   = {settings.suspicion_min_delta}", flush=True)
    print(f"mid_rescue_s3_len     <= {settings.mid_rescue_max_s3_len}", flush=True)
    print(f"mid_rescue_official   = {settings.mid_rescue_min_official_len}..{settings.mid_rescue_max_official_len}", flush=True)
    print(f"mid_rescue_delta      = {settings.mid_rescue_min_delta}..{settings.mid_rescue_max_delta}", flush=True)
    print(f"mid_rescue_ratio_min  = {settings.mid_rescue_min_long_ratio}", flush=True)
    print(f"mid_rescue_source     = {settings.mid_rescue_source}", flush=True)
    print(f"mid_rescue_support    >= {settings.mid_rescue_min_support_count}", flush=True)
    print(f"mid_rescue_best_lens  = {settings.mid_rescue_best_long_lens_csv}", flush=True)
    print(f"mid_rescue_veto       = s3<={settings.mid_rescue_veto_s3_le}, official>={settings.mid_rescue_veto_official_len}", flush=True)
    print(f"true_long_s3_len      <= {settings.true_long_max_s3_len}", flush=True)
    print(f"true_long_official    = {settings.true_long_min_official_len}..{settings.true_long_max_official_len}", flush=True)
    print(f"true_long_delta       >= {settings.true_long_min_delta}", flush=True)
    print(f"true_long_ratio_min   = {settings.true_long_min_long_ratio}", flush=True)
    print(f"true_long_raw_min     = {settings.true_long_min_raw_ratio}", flush=True)
    print(f"true_long_support     >= {settings.true_long_min_support_count}", flush=True)
    print(f"true_long_best_lens   = {settings.true_long_best_long_lens_csv}", flush=True)
    print(f"true_long_source      = {settings.true_long_source}", flush=True)
    print(f"official_initial_len  = {settings.official.initial_length}", flush=True)
    print(f"official_dstep        = {settings.official.dstep}", flush=True)
    print(f"official_use_bias     = {settings.official.use_bias}", flush=True)
    print(f"lcas_policy           = {getattr(cfg.decode, 'lcas_policy', None)}", flush=True)
    print(f"total_steps           = {cfg.decode.total_steps}", flush=True)
    print(f"seed                  = {cfg.decode.seed}", flush=True)
    print(f"baseline_results      = {baseline_results_path}", flush=True)
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
        annotate_result(result)
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
            "official_cal": result.get("official_cal"),
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
            f"s3_len={m['s3_selected_length']} | "
            f"official_len={m['official_selected_length']} | "
            f"final_len={m['selected_mask_length']} | "
            f"oracle_len={m['oracle_mask_length']} | "
            f"diff={m['selected_minus_oracle_length']} | "
            f"repair={m['official_repair_triggered']} | "
            f"repair_reason={m['official_repair_reason']} | "
            f"final_source={m['final_source']} | "
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
            print(f"WARNING: baseline results not found: {baseline_results_path}", flush=True)
    summary = summarize_bounded_repair_results(results, baseline_rows, baseline_results_path)
    logger.save_json("summary.json", summary)

    print("-" * 80, flush=True)
    print("LCAL + official bounded repair experiment finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)
    return {"results": results, "summary": summary, "run_dir": logger.run_dir}


def main() -> None:
    args = parse_args()
    lcal_settings = rescue.LcalV3Settings(
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
    )
    official_settings = official_cal.OfficialCalSettings(
        initial_length=args.official_initial_length,
        span=args.official_span,
        max_length=args.official_max_length,
        dstep=args.official_dstep,
        use_bias=not args.official_no_bias,
        bias_params=official_cal.parse_bias_params(args.official_bias_params),
    )
    settings = BoundedRepairSettings(
        lcal=lcal_settings,
        official=official_settings,
        official_eval_max_s3_len=args.official_eval_max_s3_len,
        repair_max_s3_len=args.repair_max_s3_len,
        repair_min_official_len=args.repair_min_official_len,
        repair_max_official_len=args.repair_max_official_len,
        repair_min_delta=args.repair_min_delta,
        repair_max_delta=args.repair_max_delta,
        suspicion_max_s3_len=args.suspicion_max_s3_len,
        suspicion_min_official_len=args.suspicion_min_official_len,
        suspicion_max_official_len=args.suspicion_max_official_len,
        suspicion_min_delta=args.suspicion_min_delta,
        mid_rescue_max_s3_len=args.mid_rescue_max_s3_len,
        mid_rescue_min_official_len=args.mid_rescue_min_official_len,
        mid_rescue_max_official_len=args.mid_rescue_max_official_len,
        mid_rescue_min_delta=args.mid_rescue_min_delta,
        mid_rescue_max_delta=args.mid_rescue_max_delta,
        mid_rescue_min_long_ratio=args.mid_rescue_min_long_ratio,
        mid_rescue_source=args.mid_rescue_source,
        mid_rescue_min_support_count=args.mid_rescue_min_support_count,
        mid_rescue_best_long_lens_csv=args.mid_rescue_best_long_lens,
        mid_rescue_veto_s3_le=args.mid_rescue_veto_s3_le,
        mid_rescue_veto_official_len=args.mid_rescue_veto_official_len,
        true_long_max_s3_len=args.true_long_max_s3_len,
        true_long_min_official_len=args.true_long_min_official_len,
        true_long_max_official_len=args.true_long_max_official_len,
        true_long_min_delta=args.true_long_min_delta,
        true_long_min_long_ratio=args.true_long_min_long_ratio,
        true_long_min_raw_ratio=args.true_long_min_raw_ratio,
        true_long_min_support_count=args.true_long_min_support_count,
        true_long_best_long_lens_csv=args.true_long_best_long_lens,
        true_long_source=args.true_long_source,
    )

    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.data.split = args.split
    cfg.data.dataset_subset = args.dataset_subset
    cfg.data.max_samples = args.max_samples
    cfg.decode.mask_length_source = "lcal_official_bounded_repair"
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

    print("===== LCAL Official Bounded Repair Summary =====")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")
    print(f"\nRun directory: {output['run_dir']}")


if __name__ == "__main__":
    main()
