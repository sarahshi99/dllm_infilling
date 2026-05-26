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
DEFAULT_LONG_GRID = "13,14,15,16,20,24,28,32,40"
DEFAULT_LONG_TRIGGER_EXTRA = "16,20,24"
DEFAULT_RATIO_TRIGGER_THRESHOLD = 0.97
DEFAULT_LONG_SCORE_FLOOR = 0.55

_ACTIVE_SETTINGS: Optional["LcalV3Settings"] = None
_LCAL_META_BY_TASK_ID: Dict[str, Dict[str, Any]] = {}


@dataclass
class LcalV3Settings:
    base_probe_lengths_csv: str
    base_alpha: float
    long_probe_lengths_csv: str
    long_alpha: float
    long_trigger_min_length: int
    long_trigger_extra_lengths_csv: str
    ratio_trigger_threshold: float
    long_score_floor: float
    tie_break: str
    score_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LCAL-v3: CAL-lite base length selection with long-aware correction and LCAS-v3 stopping."
    )
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-probe-lengths", type=str, default=DEFAULT_COMPACT_GRID)
    parser.add_argument("--base-alpha", type=float, default=0.06)
    parser.add_argument("--long-probe-lengths", type=str, default=DEFAULT_LONG_GRID)
    parser.add_argument("--long-alpha", type=float, default=0.10)
    parser.add_argument("--long-trigger-min-length", type=int, default=13)
    parser.add_argument("--long-trigger-extra-lengths", type=str, default=DEFAULT_LONG_TRIGGER_EXTRA)
    parser.add_argument("--ratio-trigger-threshold", type=float, default=DEFAULT_RATIO_TRIGGER_THRESHOLD)
    parser.add_argument("--long-score-floor", type=float, default=DEFAULT_LONG_SCORE_FLOOR)
    parser.add_argument("--tie-break", type=str, default="shorter", choices=["shorter", "longer"])
    parser.add_argument("--score-mode", type=str, default="length_power", choices=["raw", "length_power"])

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
    return probe_cfg


def _candidate_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("score", candidate.get("adjusted_score", candidate.get("raw_score"))))


def _best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if _ACTIVE_SETTINGS is None:
        raise RuntimeError("LCAL-v3 settings were not initialized")
    reverse_len_sign = -1 if _ACTIVE_SETTINGS.tie_break == "shorter" else 1
    return max(
        candidates,
        key=lambda item: (
            _candidate_score(item),
            reverse_len_sign * int(item["mask_length"]),
        ),
    )


def _base_best_info(base_selection: Dict[str, Any]) -> Dict[str, Any]:
    candidates = list(base_selection["candidate_scores"])
    best = _best_candidate(candidates)
    long_candidates = [item for item in candidates if int(item["mask_length"]) >= 13]
    best_long = _best_candidate(long_candidates)
    best_score = _candidate_score(best)
    best_long_score = _candidate_score(best_long)
    return {
        "best_len": int(best["mask_length"]),
        "best_score": best_score,
        "best_long_len": int(best_long["mask_length"]),
        "best_long_score": best_long_score,
        "long_ratio": best_long_score / best_score if best_score else None,
    }


def _long_trigger_reasons(best_info: Dict[str, Any], settings: LcalV3Settings) -> Tuple[List[str], bool]:
    reasons: List[str] = []
    if int(best_info["best_len"]) >= int(settings.long_trigger_min_length):
        reasons.append(f"best_len>={int(settings.long_trigger_min_length)}")

    long_score_floor_passed = float(best_info["best_long_score"]) >= float(settings.long_score_floor)
    long_ratio = best_info["long_ratio"]
    ratio_passed = (
        long_ratio is not None
        and float(long_ratio) >= float(settings.ratio_trigger_threshold)
        and long_score_floor_passed
    )
    if ratio_passed:
        reasons.append(
            "long_ratio>="
            f"{float(settings.ratio_trigger_threshold):.3f}"
            "_and_best_long_score>="
            f"{float(settings.long_score_floor):.3f}"
        )
    return reasons, long_score_floor_passed


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

    trigger_reasons, long_score_floor_passed = _long_trigger_reasons(best_info, settings)
    long_triggered = bool(trigger_reasons)

    long_selection: Optional[Dict[str, Any]] = None
    final_source = "base"
    final_selection = base_selection
    total_probe_sec = float(base_selection["length_probe_sec"])

    if long_triggered:
        long_cfg = _clone_probe_cfg(
            cfg=cfg,
            probe_lengths_csv=settings.long_probe_lengths_csv,
            alpha=settings.long_alpha,
            settings=settings,
        )
        long_selection = select_mask_length_cal_lite(task, tokenizer, model, long_cfg)
        long_selected = int(long_selection["selected_mask_length"])
        if long_selected >= base_selected:
            final_selection = long_selection
            final_source = "long_correction"
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
        "version": "lcal_v3_t2_ratio",
        "final_source": final_source,
        "long_triggered": long_triggered,
        "long_trigger_reasons": trigger_reasons,
        "trigger_reason": "+".join(trigger_reasons) if trigger_reasons else "none",
        "best_score": float(best_info["best_score"]),
        "best_len": int(best_info["best_len"]),
        "best_long_score": float(best_info["best_long_score"]),
        "best_long_len": int(best_info["best_long_len"]),
        "long_ratio": None if best_info["long_ratio"] is None else float(best_info["long_ratio"]),
        "long_score_floor_passed": bool(long_score_floor_passed),
        "ratio_trigger_threshold": float(settings.ratio_trigger_threshold),
        "long_score_floor": float(settings.long_score_floor),
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
        "long_probe_lengths": parse_probe_lengths(settings.long_probe_lengths_csv),
        "long_alpha": float(settings.long_alpha),
        "long_selected_mask_length": None,
        "long_selected_score": None,
        "long_selected_raw_score": None,
        "long_selected_adjusted_score": None,
        "long_length_probe_sec": 0.0,
        "long_candidate_scores": None,
        "long_selected_length": None,
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
            }
        )

    output = {
        "oracle_mask_length": None if oracle_mask_length is None else int(oracle_mask_length),
        "mask_length_source": "lcal_v3_t2_ratio",
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
            "long_score_floor_passed": lcal_meta["long_score_floor_passed"],
            "trigger_reason": lcal_meta["trigger_reason"],
            "base_selected_length": lcal_meta["base_selected_length"],
            "long_selected_length": lcal_meta["long_selected_length"],
            "final_selected_length": lcal_meta["final_selected_length"],
            "final_source": lcal_meta["final_source"],
            "lcal_v3_final_source": lcal_meta["final_source"],
            "lcal_v3_long_triggered": lcal_meta["long_triggered"],
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
            "lcal_v3_final_source_histogram": _hist(m.get("lcal_v3_final_source") for m in metrics),
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
    config_payload["decode"]["mask_length_source"] = "lcal_v3_t2_ratio"
    config_payload["decode"]["lcas_policy"] = getattr(cfg.decode, "lcas_policy", None)
    config_payload["lcal_v3"] = asdict(settings)
    config_payload["baseline_results"] = baseline_results_path
    logger.save_config(config_payload)

    print("=" * 80, flush=True)
    print("Starting LCAL-v3 T2 ratio-trigger experiment", flush=True)
    print(f"experiment_name = {cfg.logging.experiment_name}", flush=True)
    print(f"output_dir       = {cfg.logging.output_dir}", flush=True)
    print(f"dataset_subset   = {cfg.data.dataset_subset}", flush=True)
    print(f"split            = {cfg.data.split}", flush=True)
    print(f"max_samples      = {cfg.data.max_samples}", flush=True)
    print(f"model_path       = {cfg.model.model_path}", flush=True)
    print(f"base_grid        = {settings.base_probe_lengths_csv}", flush=True)
    print(f"base_alpha       = {settings.base_alpha}", flush=True)
    print(f"long_grid        = {settings.long_probe_lengths_csv}", flush=True)
    print(f"long_alpha       = {settings.long_alpha}", flush=True)
    print(f"trigger_min_len  = {settings.long_trigger_min_length}", flush=True)
    print(f"ratio_threshold  = {settings.ratio_trigger_threshold}", flush=True)
    print(f"long_score_floor = {settings.long_score_floor}", flush=True)
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
            f"trigger_reason={m['trigger_reason']} | "
            f"ratio={m['long_ratio']:.4f} | "
            f"stopped={m['stopped']} | "
            f"stop_step={m['stop_step']} | "
            f"effective_steps={m['effective_steps']} | "
            f"stop_reason={m['stop_reason']}",
            flush=True,
        )

    baseline_rows = load_jsonl(baseline_results_path)
    summary = summarize_lcal_v3_results(
        results=results,
        baseline_rows=baseline_rows,
        baseline_path=baseline_results_path,
    )
    logger.save_json("summary.json", summary)

    print("-" * 80, flush=True)
    print("LCAL-v3 T2 ratio-trigger experiment finished", flush=True)
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
        long_probe_lengths_csv=args.long_probe_lengths,
        long_alpha=args.long_alpha,
        long_trigger_min_length=args.long_trigger_min_length,
        long_trigger_extra_lengths_csv=args.long_trigger_extra_lengths,
        ratio_trigger_threshold=args.ratio_trigger_threshold,
        long_score_floor=args.long_score_floor,
        tie_break=args.tie_break,
        score_mode=args.score_mode,
    )

    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.data.split = args.split
    cfg.data.dataset_subset = args.dataset_subset
    cfg.data.max_samples = args.max_samples

    cfg.decode.mask_length_source = "lcal_v3_t2_ratio"
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

    output = run_lcal_v3_experiment(
        cfg=cfg,
        settings=settings,
        baseline_results_path=args.baseline_results,
    )

    print("===== LCAL-v3 Summary =====")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")
    print(f"\nRun directory: {output['run_dir']}")


if __name__ == "__main__":
    main()
