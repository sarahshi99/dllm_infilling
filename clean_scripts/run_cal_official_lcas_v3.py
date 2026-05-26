#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask, compute_oracle_mask_length, load_humaneval_infilling
from expvision_dllm_clean.decode import _length_diff
from expvision_dllm_clean.evaluation import summarize_results
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import load_model_and_tokenizer, resolve_mask_token_id, set_global_seed
from expvision_dllm_clean.runner_lcas_v3 import annotate_lcas_v3_result, patch_decode_lcas_policy, summarize_lcas_v3_results

import expvision_dllm_clean.decode_lcas as decode_lcas


DEFAULT_BIAS_PARAMS = "1.0,1.77,0.56,0.06,0.24"
_ACTIVE_SETTINGS: Optional["OfficialCalSettings"] = None
_CAL_META_BY_TASK_ID: Dict[str, Dict[str, Any]] = {}


@dataclass
class OfficialCalSettings:
    initial_length: int = 8
    span: int = 1
    max_length: int = 64
    dstep: int = 4
    use_bias: bool = True
    bias_params: Tuple[float, float, float, float, float] = (1.0, 1.77, 0.56, 0.06, 0.24)
    tie_break: str = "first"  # Official code updates only on strictly greater confidence.


def parse_bias_params(text: str) -> Tuple[float, float, float, float, float]:
    parts = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(parts) != 5:
        raise ValueError("--bias-params must contain five comma-separated floats: a,b,c,d,e")
    return tuple(parts)  # type: ignore[return-value]


def length_bias(length: int, params: Tuple[float, float, float, float, float]) -> float:
    a, b, c, d, e = params
    return float(a * math.exp(-b * length) + c * math.exp(-d * length) + e)


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


def probe_official_cal_score(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
    settings: OfficialCalSettings,
    mask_length: int,
) -> Dict[str, Any]:
    prepared = _build_probe_input(task, tokenizer, cfg, mask_length)
    device = _resolve_model_device(model)
    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(x_t, dtype=torch.long, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(x_t, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    probs = torch.softmax(logits, dim=-1)
    x0_t = torch.argmax(logits, dim=-1)
    x0_p_t = torch.gather(probs, dim=-1, index=x0_t.unsqueeze(-1)).squeeze(-1)
    mask_conf = x0_p_t[:, prepared["middle_start"] : prepared["middle_end"]]
    raw_phi = float(mask_conf.mean().item())
    bias = length_bias(mask_length, settings.bias_params)
    calibrated_phi = raw_phi / bias if settings.use_bias else raw_phi
    probe_sec = time.perf_counter() - start

    top2_values, _ = torch.topk(probs[:, prepared["middle_start"] : prepared["middle_end"], :], k=2, dim=-1)
    mean_top2_gap = float((top2_values[..., 0] - top2_values[..., 1]).mean().item())

    return {
        "mask_length": int(mask_length),
        "score": float(calibrated_phi),
        "calibrated_confidence": float(calibrated_phi),
        "raw_score": float(raw_phi),
        "raw_phi": float(raw_phi),
        "bias": float(bias),
        "mean_top1_prob": float(raw_phi),
        "mean_top2_gap": mean_top2_gap,
        "score_mode": "official_cal_bias" if settings.use_bias else "official_cal_raw",
        "probe_sec": float(probe_sec),
    }


def select_mask_length_official_cal(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    if _ACTIVE_SETTINGS is None:
        raise RuntimeError("Official CAL settings have not been initialized")
    settings = _ACTIVE_SETTINGS

    if settings.initial_length < 1:
        raise ValueError("initial_length must be >= 1")
    if settings.span < 1:
        raise ValueError("span must be >= 1")
    if settings.max_length < settings.initial_length:
        raise ValueError("max_length must be >= initial_length")
    if settings.dstep < 0:
        raise ValueError("dstep must be >= 0")

    candidate_scores: List[Dict[str, Any]] = []
    seen_lengths: set[int] = set()
    total_probe_sec = 0.0
    search_events: List[Dict[str, Any]] = []

    def get_conf_at_len(length: int, direction: str) -> Optional[Dict[str, Any]]:
        nonlocal total_probe_sec
        if length < 1 or length > settings.max_length or length in seen_lengths:
            return None
        seen_lengths.add(length)
        candidate = probe_official_cal_score(task, tokenizer, model, cfg, settings, length)
        candidate["search_direction"] = direction
        candidate_scores.append(candidate)
        total_probe_sec += float(candidate["probe_sec"])
        return candidate

    first = get_conf_at_len(settings.initial_length, "initial")
    if first is None:
        raise RuntimeError("failed to probe initial length")

    best = first
    search_events.append(
        {
            "length": int(first["mask_length"]),
            "score": float(first["score"]),
            "raw_phi": float(first["raw_phi"]),
            "bias": float(first["bias"]),
            "direction": "initial",
            "is_new_best": True,
            "consecutive_decreases": 0,
        }
    )

    curr_l = settings.initial_length + settings.span
    consecutive_decreases = 0
    while curr_l <= settings.max_length:
        candidate = get_conf_at_len(curr_l, "up")
        if candidate is None:
            curr_l += settings.span
            continue
        is_new_best = float(candidate["score"]) > float(best["score"])
        if is_new_best:
            best = candidate
            consecutive_decreases = 0
        else:
            consecutive_decreases += 1
        search_events.append(
            {
                "length": int(candidate["mask_length"]),
                "score": float(candidate["score"]),
                "raw_phi": float(candidate["raw_phi"]),
                "bias": float(candidate["bias"]),
                "direction": "up",
                "is_new_best": is_new_best,
                "consecutive_decreases": consecutive_decreases,
            }
        )
        if consecutive_decreases >= settings.dstep:
            break
        curr_l += settings.span

    curr_l = settings.initial_length - settings.span
    consecutive_decreases = 0
    while curr_l >= 1:
        candidate = get_conf_at_len(curr_l, "down")
        if candidate is None:
            curr_l -= settings.span
            continue
        is_new_best = float(candidate["score"]) > float(best["score"])
        if is_new_best:
            best = candidate
            consecutive_decreases = 0
        else:
            consecutive_decreases += 1
        search_events.append(
            {
                "length": int(candidate["mask_length"]),
                "score": float(candidate["score"]),
                "raw_phi": float(candidate["raw_phi"]),
                "bias": float(candidate["bias"]),
                "direction": "down",
                "is_new_best": is_new_best,
                "consecutive_decreases": consecutive_decreases,
            }
        )
        if consecutive_decreases >= settings.dstep:
            break
        curr_l -= settings.span

    selected = int(best["mask_length"])
    probed_lengths = [int(item["mask_length"]) for item in candidate_scores]
    oracle_length = compute_oracle_mask_length(task, tokenizer, add_special_tokens=False)
    diff = _length_diff(selected, None if oracle_length is None else int(oracle_length))

    meta = {
        "version": "official_cal_lcas_v3",
        "paper": "Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly",
        "formula": "Phi_norm(L)=Phi(L)/B(L), B(L)=a*exp(-bL)+c*exp(-dL)+e",
        "initial_length": settings.initial_length,
        "span": settings.span,
        "max_length": settings.max_length,
        "dstep": settings.dstep,
        "use_bias": settings.use_bias,
        "bias_params": list(settings.bias_params),
        "search_steps": len(candidate_scores),
        "probed_lengths": probed_lengths,
        "search_events": search_events,
        "selected_mask_length": selected,
        "selected_score": float(best["score"]),
        "selected_raw_score": float(best["raw_score"]),
        "selected_bias": float(best["bias"]),
        "selected_calibrated_confidence": float(best["calibrated_confidence"]),
    }
    _CAL_META_BY_TASK_ID[task.task_id] = meta

    return {
        "selected_mask_length": selected,
        "selected_score": float(best["score"]),
        "selected_raw_score": float(best["raw_score"]),
        "selected_adjusted_score": float(best["score"]),
        "candidate_scores": sorted(candidate_scores, key=lambda item: int(item["mask_length"])),
        "length_probe_sec": total_probe_sec,
        "probe_lengths": probed_lengths,
        "tie_break": settings.tie_break,
        "score_mode": "official_cal_bias" if settings.use_bias else "official_cal_raw",
        "length_alpha": None,
        **diff,
    }


def resolve_mask_length_official_cal(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> Dict[str, Any]:
    oracle_mask_length = compute_oracle_mask_length(task, tokenizer, add_special_tokens=False)
    selection = select_mask_length_official_cal(task, tokenizer, model, cfg)
    selected = int(selection["selected_mask_length"])
    return {
        "oracle_mask_length": None if oracle_mask_length is None else int(oracle_mask_length),
        "mask_length_source": "official_cal",
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
        **_length_diff(selected, None if oracle_mask_length is None else int(oracle_mask_length)),
    }


def patch_decode_for_official_cal(settings: OfficialCalSettings) -> None:
    global _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = settings
    _CAL_META_BY_TASK_ID.clear()
    patch_decode_lcas_policy()
    decode_lcas.resolve_mask_length = resolve_mask_length_official_cal


def _avg(values: Iterable[float | int | None]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return None if not filtered else sum(filtered) / len(filtered)


def _rate(values: Iterable[bool]) -> Optional[float]:
    values = list(values)
    return None if not values else sum(1 for value in values if value) / len(values)


def _hist(values: Iterable[Any]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _oracle_bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    if length <= 8:
        return "<=8"
    if length <= 12:
        return "9-12"
    if length <= 16:
        return "13-16"
    if length <= 24:
        return "17-24"
    return "25+"


def summarize_official_cal_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize_lcas_v3_results(results)
    metrics = [item["metrics"] for item in results]
    oracle_buckets = sorted({_oracle_bucket(m.get("oracle_mask_length")) for m in metrics})
    true_long = [m for m in metrics if m.get("oracle_mask_length") is not None and int(m["oracle_mask_length"]) >= 17]
    short = [m for m in metrics if m.get("oracle_mask_length") is not None and int(m["oracle_mask_length"]) <= 8]
    metas = [item.get("official_cal", {}) for item in results]

    summary.update(
        {
            "mask_length_source": "official_cal",
            "oracle_bucket_histogram": _hist(_oracle_bucket(m.get("oracle_mask_length")) for m in metrics),
            "oracle_bucket_pass_rate": {
                bucket: _rate(bool(m.get("passed", False)) for m in metrics if _oracle_bucket(m.get("oracle_mask_length")) == bucket)
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
            "under_select_rate_17plus": _rate(
                int(m.get("selected_mask_length")) < int(m.get("oracle_mask_length")) for m in true_long
            ),
            "under_select_by3_rate_17plus": _rate(
                int(m.get("oracle_mask_length")) - int(m.get("selected_mask_length")) >= 3 for m in true_long
            ),
            "over_select_rate_le8": _rate(
                int(m.get("selected_mask_length")) > int(m.get("oracle_mask_length")) for m in short
            ),
            "over_select_by3_rate_le8": _rate(
                int(m.get("selected_mask_length")) - int(m.get("oracle_mask_length")) >= 3 for m in short
            ),
            "avg_cal_search_steps": _avg(meta.get("search_steps") for meta in metas),
            "selected_length_bucket_histogram": _hist(_oracle_bucket(m.get("selected_mask_length")) for m in metrics),
        }
    )

    summary["required_metrics"] = {
        "overall_pass_rate": summary.get("pass_rate"),
        "<=8_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("<=8"),
        "9-12_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("9-12"),
        "13-16_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("13-16"),
        "17-24_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("17-24"),
        "25+_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("25+"),
        "avg_selected_minus_oracle_length": summary.get("avg_selected_minus_oracle_length"),
        "under_select_rate_17plus": summary.get("under_select_rate_17plus"),
        "under_select_by3_rate_17plus": summary.get("under_select_by3_rate_17plus"),
        "over_select_rate_le8": summary.get("over_select_rate_le8"),
        "over_select_by3_rate_le8": summary.get("over_select_by3_rate_le8"),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official CAL length discovery with LCAS-v3 stopping."
    )
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--initial-length", type=int, default=8)
    parser.add_argument("--span", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dstep", type=int, default=4)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--bias-params", type=str, default=DEFAULT_BIAS_PARAMS)

    parser.add_argument("--lcas-policy", type=str, default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    parser.add_argument("--output-dir", type=str, default="outputs_clean")
    parser.add_argument("--experiment-name", type=str, default="official_cal_lcas_v3_clean")
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--save-full-text-per-step", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = OfficialCalSettings(
        initial_length=args.initial_length,
        span=args.span,
        max_length=args.max_length,
        dstep=args.dstep,
        use_bias=not args.no_bias,
        bias_params=parse_bias_params(args.bias_params),
    )
    patch_decode_for_official_cal(settings)

    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.data.split = args.split
    cfg.data.dataset_subset = args.dataset_subset
    cfg.data.max_samples = args.max_samples
    cfg.decode.mask_length_source = "official_cal"
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed
    cfg.decode.lcas_policy = args.lcas_policy
    cfg.decode.save_step_traces = args.save_step_traces
    cfg.decode.save_full_text_per_step = args.save_full_text_per_step
    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name

    set_global_seed(cfg.decode.seed)
    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)
    config_payload = cfg.to_dict()
    config_payload["official_cal"] = {
        **asdict(settings),
        "bias_params": list(settings.bias_params),
        "source_paper": "Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly",
        "source_code": "https://github.com/NiuHechang/Calibrated_Adaptive_Length",
    }
    logger.save_config(config_payload)

    print("=" * 80, flush=True)
    print("Starting official CAL + LCAS-v3 experiment", flush=True)
    print(f"experiment_name = {cfg.logging.experiment_name}", flush=True)
    print(f"output_dir       = {cfg.logging.output_dir}", flush=True)
    print(f"dataset_subset   = {cfg.data.dataset_subset}", flush=True)
    print(f"split            = {cfg.data.split}", flush=True)
    print(f"max_samples      = {cfg.data.max_samples}", flush=True)
    print(f"model_path       = {cfg.model.model_path}", flush=True)
    print(f"initial_length   = {settings.initial_length}", flush=True)
    print(f"span             = {settings.span}", flush=True)
    print(f"max_length       = {settings.max_length}", flush=True)
    print(f"dstep            = {settings.dstep}", flush=True)
    print(f"use_bias         = {settings.use_bias}", flush=True)
    print(f"bias_params      = {settings.bias_params}", flush=True)
    print(f"lcas_policy      = {cfg.decode.lcas_policy}", flush=True)
    print(f"total_steps      = {cfg.decode.total_steps}", flush=True)
    print(f"seed             = {cfg.decode.seed}", flush=True)
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
        annotate_lcas_v3_result(result)
        result["dataset_subset"] = cfg.data.dataset_subset
        result["official_cal"] = copy.deepcopy(_CAL_META_BY_TASK_ID.get(task.task_id, {}))
        results.append(result)

        result["length_probe"]["official_cal"] = copy.deepcopy(result["official_cal"])
        m = result["metrics"]
        selected_raw_score = result["official_cal"].get("selected_raw_score")
        selected_adjusted_score = result["official_cal"].get("selected_calibrated_confidence")
        result["length_probe"]["selected_raw_score"] = selected_raw_score
        result["length_probe"]["selected_adjusted_score"] = selected_adjusted_score
        result["length_probe"]["score_mode"] = "official_cal_bias" if settings.use_bias else "official_cal_raw"
        result["length_probe"]["length_alpha"] = None
        m["selected_raw_score"] = selected_raw_score
        m["selected_adjusted_score"] = selected_adjusted_score
        m["score_mode"] = "official_cal_bias" if settings.use_bias else "official_cal_raw"
        m["official_cal_search_steps"] = result["official_cal"].get("search_steps")
        m["official_cal_selected_bias"] = result["official_cal"].get("selected_bias")
        m["official_cal_selected_calibrated_confidence"] = result["official_cal"].get("selected_calibrated_confidence")

        payload = {
            "task_id": result["task_id"],
            "dataset_subset": cfg.data.dataset_subset,
            "metrics": result["metrics"],
            "verification": result["verification"],
            "length_probe": result["length_probe"],
            "stopping": result["stopping"],
            "official_cal": result["official_cal"],
            "code": result["code"],
            "diagnostics": result["diagnostics"],
        }
        logger.log_result(payload)

        for trace in result["step_traces"]:
            logger.log_trace(trace)

        print(
            f"[{idx}/{total_tasks}] task_id={task.task_id} | "
            f"{'PASS' if m['passed'] else 'FAIL'} | "
            f"total_sec={m['total_sec']:.3f} | "
            f"total_sec_including_probe={m['total_sec_including_probe']:.3f} | "
            f"probe_steps={m['official_cal_search_steps']} | "
            f"mask_len={m['mask_length']} | "
            f"oracle_mask_len={m['oracle_mask_length']} | "
            f"diff={m['selected_minus_oracle_length']} | "
            f"score={m['selected_score']:.4f} | "
            f"bucket={m['lcas_v3_bucket']} | "
            f"stopped={m['stopped']} | "
            f"stop_step={m['stop_step']} | "
            f"effective_steps={m['effective_steps']} | "
            f"stop_reason={m['stop_reason']}",
            flush=True,
        )

    summary = summarize_official_cal_results(results)
    logger.save_json("summary.json", summary)

    print("-" * 80, flush=True)
    print("Official CAL + LCAS-v3 experiment finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
