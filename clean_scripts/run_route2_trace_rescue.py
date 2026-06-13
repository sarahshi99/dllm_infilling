#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analysis.trace_feature_audit_v2 import CandidateRule, compute_shape_features
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask, compute_oracle_mask_length, load_humaneval_infilling
from expvision_dllm_clean.evaluation import summarize_results
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.runner_lcas_v3 import annotate_lcas_v3_result

import expvision_dllm_clean.decode_lcas as decode_lcas
import run_lcal_official_bounded_repair_a6000_v2 as midcons_runner


JsonDict = Dict[str, Any]


POLICIES: Dict[str, List[List[Any]]] = {
    "broad_plateau": [
        ["top1_last", "<=", 0.667969],
        ["max_remaining_plateau_steps", ">=", 16.0],
    ],
    "precision_top1_conf": [
        ["top1_median", "<=", 0.464844],
        ["confidence_max", "<=", 0.84375],
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Route 2 trace-gated long rescue.")
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lcas-policy", type=str, default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    parser.add_argument("--route2-policy", type=str, required=True, choices=sorted(POLICIES))
    parser.add_argument("--rescue-length", type=int, default=24)
    parser.add_argument("--baseline-results", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs_clean")
    parser.add_argument("--experiment-name", type=str, default="route2_trace_rescue")
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--save-full-text-per-step", action="store_true")
    return parser.parse_args()


def default_midcons_settings() -> midcons_runner.BoundedRepairSettings:
    lcal_settings = midcons_runner.rescue.LcalV3Settings(
        base_probe_lengths_csv="3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24",
        base_alpha=0.06,
        weak_probe_lengths_csv="13,14,15,16",
        strong_probe_lengths_csv="13,14,15,16,20,24,28,32,40",
        long_alpha=0.10,
        strong_min_len=13,
        weak_min_base_len=8,
        weak_max_base_len=12,
        ratio_trigger_threshold=0.97,
        long_score_floor=0.55,
        raw_ratio_threshold=0.97,
        support_count_threshold=2,
        cap_base_le8=14,
        cap_base_9_12=16,
        short_safe_policy="s3",
        correction_selection_rule="shortest_supported",
        shortest_supported_ratio=0.985,
        tie_break="shorter",
        score_mode="length_power",
    )
    official_settings = midcons_runner.official_cal.OfficialCalSettings(
        initial_length=8,
        span=1,
        max_length=64,
        dstep=4,
        use_bias=True,
        bias_params=midcons_runner.official_cal.parse_bias_params(midcons_runner.DEFAULT_BIAS_PARAMS),
    )
    return midcons_runner.BoundedRepairSettings(
        lcal=lcal_settings,
        official=official_settings,
        official_eval_max_s3_len=12,
        repair_max_s3_len=5,
        repair_min_official_len=6,
        repair_max_official_len=9,
        repair_min_delta=1,
        repair_max_delta=8,
        suspicion_max_s3_len=5,
        suspicion_min_official_len=16,
        suspicion_max_official_len=64,
        suspicion_min_delta=1,
        mid_rescue_max_s3_len=12,
        mid_rescue_min_official_len=11,
        mid_rescue_max_official_len=13,
        mid_rescue_min_delta=3,
        mid_rescue_max_delta=7,
        mid_rescue_min_long_ratio=0.8,
        mid_rescue_source="base",
    )


def make_cfg(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.data.split = args.split
    cfg.data.dataset_subset = args.dataset_subset
    cfg.data.max_samples = args.max_samples
    cfg.decode.mask_length_source = "lcal_official_bounded_repair"
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed
    cfg.decode.save_step_traces = True
    cfg.decode.save_full_text_per_step = bool(args.save_full_text_per_step)
    cfg.decode.cal_lite_probe_lengths_csv = "3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24"
    cfg.decode.cal_lite_tie_break = "shorter"
    cfg.decode.cal_lite_score_mode = "length_power"
    cfg.decode.cal_lite_length_alpha = 0.06
    cfg.decode.lcas_policy = args.lcas_policy
    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name
    return cfg


def _length_diff(selected_length: Optional[int], oracle_length: Optional[int]) -> Dict[str, Optional[int]]:
    if selected_length is None or oracle_length is None:
        return {"selected_minus_oracle_length": None, "abs_selected_minus_oracle_length": None}
    diff = int(selected_length) - int(oracle_length)
    return {"selected_minus_oracle_length": diff, "abs_selected_minus_oracle_length": abs(diff)}


def fixed_length_resolver(mask_length: int, source_name: str):
    def resolve(task: CodeTask, tokenizer, model, cfg: ExperimentConfig) -> JsonDict:
        oracle_mask_length = compute_oracle_mask_length(task, tokenizer, add_special_tokens=False)
        selected = int(mask_length)
        return {
            "oracle_mask_length": None if oracle_mask_length is None else int(oracle_mask_length),
            "mask_length_source": source_name,
            "mask_length": selected,
            "selected_mask_length": selected,
            "selected_score": None,
            "selected_raw_score": None,
            "selected_adjusted_score": None,
            "candidate_scores": None,
            "length_probe_sec": 0.0,
            "probe_lengths": [selected],
            "tie_break": "fixed",
            "score_mode": "fixed",
            "length_alpha": None,
            **_length_diff(selected, None if oracle_mask_length is None else int(oracle_mask_length)),
        }

    return resolve


def trace_record_from_result(result: Mapping[str, Any]) -> JsonDict:
    features = compute_shape_features(result.get("step_traces") or [])
    features["selected_len"] = (result.get("metrics") or {}).get("selected_mask_length")
    return {"features": features, "labels": {}}


def route2_rule(policy_name: str) -> CandidateRule:
    return CandidateRule(name=policy_name, family="route2_trace_rescue", clauses=POLICIES[policy_name])


def oracle_bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    if int(length) <= 8:
        return "<=8"
    if int(length) <= 12:
        return "9-12"
    if int(length) <= 16:
        return "13-16"
    if int(length) <= 24:
        return "17-24"
    return "25+"


def load_jsonl(path: Optional[str]) -> List[JsonDict]:
    if not path:
        return []
    rows: List[JsonDict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def pairwise_vs_baseline(results: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    baseline = {str(row.get("task_id")): row for row in baseline_rows}
    counts = Counter()
    bucket_counts: Dict[str, Counter[str]] = {}
    for result in results:
        task_id = str(result.get("task_id"))
        base = baseline.get(task_id)
        if base is None:
            continue
        passed = bool((result.get("metrics") or {}).get("passed"))
        base_passed = bool((base.get("metrics") or {}).get("passed"))
        if passed and not base_passed:
            key = "win"
        elif (not passed) and base_passed:
            key = "loss"
        elif passed and base_passed:
            key = "tie_pass"
        else:
            key = "tie_fail"
        counts[key] += 1
        bucket = oracle_bucket((result.get("metrics") or {}).get("oracle_mask_length"))
        bucket_counts.setdefault(bucket, Counter())[key] += 1
    return {
        "counts": {key: counts.get(key, 0) for key in ["win", "loss", "tie_pass", "tie_fail"]},
        "by_oracle_bucket": {
            bucket: {key: counter.get(key, 0) for key in ["win", "loss", "tie_pass", "tie_fail"]}
            for bucket, counter in sorted(bucket_counts.items())
        },
    }


def rate(values: Iterable[bool]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def summarize_route2(results: List[JsonDict], baseline_rows: Sequence[Mapping[str, Any]], baseline_path: Optional[str]) -> JsonDict:
    summary = summarize_results(results)
    metrics = [item["metrics"] for item in results]
    triggered = [m for m in metrics if bool(m.get("route2_trace_rescue_triggered"))]
    triggered_oracle = [m for m in triggered if m.get("oracle_mask_length") is not None]
    buckets = sorted({oracle_bucket(m.get("oracle_mask_length")) for m in metrics})
    summary.update(
        {
            "mask_length_source": "route2_trace_rescue",
            "route2_trigger_count": len(triggered),
            "route2_trigger_rate": len(triggered) / len(metrics) if metrics else None,
            "route2_trigger_pass_rate": rate(bool(m.get("passed")) for m in triggered),
            "route2_trigger_true_long_precision": rate(
                int(m.get("oracle_mask_length")) >= 17 for m in triggered_oracle
            ),
            "route2_trigger_oracle_bucket_histogram": dict(
                sorted(Counter(oracle_bucket(m.get("oracle_mask_length")) for m in triggered).items())
            ),
            "route2_policy_histogram": dict(sorted(Counter(m.get("route2_policy") for m in metrics).items())),
            "route2_final_source_histogram": dict(
                sorted(Counter(m.get("route2_final_source") for m in metrics).items())
            ),
            "oracle_bucket_pass_rate": {
                bucket: rate(bool(m.get("passed")) for m in metrics if oracle_bucket(m.get("oracle_mask_length")) == bucket)
                for bucket in buckets
            },
            "baseline_results": baseline_path,
            "pairwise_vs_baseline": pairwise_vs_baseline(results, baseline_rows),
        }
    )
    return summary


def add_route2_metrics(
    final_result: JsonDict,
    primary_result: Mapping[str, Any],
    *,
    route2_policy_name: str,
    route2_triggered: bool,
    rescue_length: Optional[int],
    trace_features: Mapping[str, Any],
    rescue_result: Optional[Mapping[str, Any]],
) -> None:
    final_metrics = final_result["metrics"]
    primary_metrics = primary_result["metrics"]
    rescue_metrics = (rescue_result or {}).get("metrics") or {}
    primary_total = float(primary_metrics.get("total_sec_including_probe") or 0.0)
    rescue_total = float(rescue_metrics.get("total_sec_including_probe") or 0.0)
    combined_total = primary_total + (rescue_total if route2_triggered else 0.0)

    if route2_triggered:
        final_metrics["decode_sec"] = float(primary_metrics.get("decode_sec") or 0.0) + float(
            rescue_metrics.get("decode_sec") or 0.0
        )
        final_metrics["verification_sec"] = float(primary_metrics.get("verification_sec") or 0.0) + float(
            rescue_metrics.get("verification_sec") or 0.0
        )
        final_metrics["total_sec"] = float(primary_metrics.get("total_sec") or 0.0) + float(
            rescue_metrics.get("total_sec") or 0.0
        )
        final_metrics["length_probe_sec"] = float(primary_metrics.get("length_probe_sec") or 0.0) + float(
            rescue_metrics.get("length_probe_sec") or 0.0
        )
        final_metrics["total_sec_including_probe"] = combined_total

    final_metrics.update(
        {
            "mask_length_source": "route2_trace_rescue",
            "route2_policy": route2_policy_name,
            "route2_trace_rescue_triggered": bool(route2_triggered),
            "route2_final_source": "rescue" if route2_triggered else "primary",
            "route2_primary_passed": bool(primary_metrics.get("passed")),
            "route2_primary_selected_mask_length": primary_metrics.get("selected_mask_length"),
            "route2_primary_final_source": primary_metrics.get("final_source"),
            "route2_primary_stop_reason": primary_metrics.get("stop_reason"),
            "route2_rescue_length": rescue_length,
            "route2_primary_total_sec_including_probe": primary_total,
            "route2_rescue_total_sec_including_probe": rescue_total if route2_triggered else 0.0,
            "route2_combined_total_sec_including_probe": combined_total,
            "route2_trace_top1_last": trace_features.get("top1_last"),
            "route2_trace_top1_median": trace_features.get("top1_median"),
            "route2_trace_confidence_max": trace_features.get("confidence_max"),
            "route2_trace_max_remaining_plateau_steps": trace_features.get("max_remaining_plateau_steps"),
        }
    )
    final_result["route2_trace_rescue"] = {
        "policy": route2_policy_name,
        "clauses": POLICIES[route2_policy_name],
        "triggered": bool(route2_triggered),
        "final_source": "rescue" if route2_triggered else "primary",
        "rescue_length": rescue_length,
        "primary_passed": bool(primary_metrics.get("passed")),
        "primary_selected_mask_length": primary_metrics.get("selected_mask_length"),
        "primary_metrics": {
            key: primary_metrics.get(key)
            for key in [
                "passed",
                "selected_mask_length",
                "oracle_mask_length",
                "final_source",
                "stop_reason",
                "total_sec_including_probe",
            ]
        },
        "rescue_metrics": {
            key: rescue_metrics.get(key)
            for key in [
                "passed",
                "selected_mask_length",
                "oracle_mask_length",
                "stop_reason",
                "total_sec_including_probe",
            ]
        }
        if route2_triggered
        else None,
        "trace_features": dict(trace_features),
    }


def result_payload(result: Mapping[str, Any], dataset_subset: str) -> JsonDict:
    return {
        "task_id": result["task_id"],
        "dataset_subset": dataset_subset,
        "metrics": result["metrics"],
        "verification": result["verification"],
        "length_probe": result["length_probe"],
        "stopping": result["stopping"],
        "lcal_v3": result.get("lcal_v3"),
        "official_cal": result.get("official_cal"),
        "route2_trace_rescue": result.get("route2_trace_rescue"),
        "code": result["code"],
        "diagnostics": result["diagnostics"],
    }


def run_primary(task: CodeTask, tokenizer, model, cfg: ExperimentConfig, settings: midcons_runner.BoundedRepairSettings) -> JsonDict:
    midcons_runner.patch_decode(settings)
    result = decode_lcas.run_decode_with_lcas(task, tokenizer, model, cfg)
    midcons_runner.annotate_result(result)
    annotate_lcas_v3_result(result)
    return result


def run_fixed_rescue(task: CodeTask, tokenizer, model, cfg: ExperimentConfig, rescue_length: int) -> JsonDict:
    decode_lcas.resolve_mask_length = fixed_length_resolver(
        int(rescue_length),
        source_name="route2_trace_rescue_fixed",
    )
    result = decode_lcas.run_decode_with_lcas(task, tokenizer, model, cfg)
    annotate_lcas_v3_result(result)
    return result


def run_experiment(
    cfg: ExperimentConfig,
    settings: midcons_runner.BoundedRepairSettings,
    *,
    route2_policy_name: str,
    rescue_length: int,
    baseline_results_path: Optional[str],
) -> JsonDict:
    set_global_seed(cfg.decode.seed)
    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)
    rule = route2_rule(route2_policy_name)
    config_payload = cfg.to_dict()
    config_payload["route2_trace_rescue"] = {
        "policy": route2_policy_name,
        "clauses": POLICIES[route2_policy_name],
        "rescue_length": int(rescue_length),
        "primary_method": "lcal_official_bounded_repair_midcons",
        "selection_boundary": "trace_features_only_no_verifier_choice",
    }
    config_payload["lcal_official_bounded_repair"] = {
        "lcal": asdict(settings.lcal),
        "official": {**asdict(settings.official), "bias_params": list(settings.official.bias_params)},
        "repair": {
            key: value
            for key, value in asdict(settings).items()
            if key not in {"lcal", "official"}
        },
    }
    config_payload["baseline_results"] = baseline_results_path
    logger.save_config(config_payload)

    print("=" * 80, flush=True)
    print("Starting Route 2 trace-gated long rescue", flush=True)
    print(f"experiment_name = {cfg.logging.experiment_name}", flush=True)
    print(f"route2_policy   = {route2_policy_name}", flush=True)
    print(f"clauses         = {POLICIES[route2_policy_name]}", flush=True)
    print(f"rescue_length   = {rescue_length}", flush=True)
    print(f"model_path      = {cfg.model.model_path}", flush=True)
    print(f"max_samples     = {cfg.data.max_samples}", flush=True)
    print(f"baseline        = {baseline_results_path}", flush=True)
    print(f"run_dir         = {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)

    tokenizer, model = load_model_and_tokenizer(cfg.model)
    tasks = load_humaneval_infilling(
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        dataset_subset=cfg.data.dataset_subset,
    )
    results: List[JsonDict] = []
    total_tasks = len(tasks)
    print(f"Loaded {total_tasks} tasks", flush=True)
    print("-" * 80, flush=True)

    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{total_tasks}] task_id={task.task_id} | primary start", flush=True)
        primary = run_primary(task, tokenizer, model, cfg, settings)
        record = trace_record_from_result(primary)
        trace_features = record["features"]
        triggered = rule.matches(record)
        final_result = primary
        rescue: Optional[JsonDict] = None
        actual_rescue_length: Optional[int] = None

        if triggered:
            primary_selected = int(primary["metrics"]["selected_mask_length"])
            actual_rescue_length = max(primary_selected, int(rescue_length))
            print(
                f"[{idx}/{total_tasks}] task_id={task.task_id} | route2 triggered | "
                f"primary_len={primary_selected} | rescue_len={actual_rescue_length}",
                flush=True,
            )
            rescue = run_fixed_rescue(task, tokenizer, model, cfg, actual_rescue_length)
            final_result = rescue

        final_result = copy.deepcopy(final_result)
        add_route2_metrics(
            final_result,
            primary,
            route2_policy_name=route2_policy_name,
            route2_triggered=triggered,
            rescue_length=actual_rescue_length,
            trace_features=trace_features,
            rescue_result=rescue,
        )
        final_result["dataset_subset"] = cfg.data.dataset_subset
        results.append(final_result)
        logger.log_result(result_payload(final_result, cfg.data.dataset_subset))

        for trace in primary.get("step_traces") or []:
            trace_payload = dict(trace)
            trace_payload["route2_trace_phase"] = "primary"
            logger.log_trace(trace_payload)
        if rescue is not None:
            for trace in rescue.get("step_traces") or []:
                trace_payload = dict(trace)
                trace_payload["route2_trace_phase"] = "rescue"
                trace_payload["route2_rescue_length"] = actual_rescue_length
                logger.log_trace(trace_payload)

        m = final_result["metrics"]
        print(
            f"[{idx}/{total_tasks}] task_id={task.task_id} | "
            f"{'PASS' if m['passed'] else 'FAIL'} | "
            f"triggered={m['route2_trace_rescue_triggered']} | "
            f"final_source={m['route2_final_source']} | "
            f"primary_pass={m['route2_primary_passed']} | "
            f"final_len={m['selected_mask_length']} | "
            f"oracle_len={m['oracle_mask_length']} | "
            f"combined_sec={m['route2_combined_total_sec_including_probe']:.3f}",
            flush=True,
        )

    baseline_rows = load_jsonl(baseline_results_path)
    summary = summarize_route2(results, baseline_rows, baseline_results_path)
    logger.save_json("summary.json", summary)

    print("-" * 80, flush=True)
    print("Route 2 trace-gated long rescue finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)
    return {"results": results, "summary": summary, "run_dir": logger.run_dir}


def main() -> None:
    args = parse_args()
    cfg = make_cfg(args)
    settings = default_midcons_settings()
    output = run_experiment(
        cfg,
        settings,
        route2_policy_name=args.route2_policy,
        rescue_length=args.rescue_length,
        baseline_results_path=args.baseline_results,
    )
    print("===== Route 2 Trace Rescue Summary =====")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")
    print(f"\nRun directory: {output['run_dir']}")


if __name__ == "__main__":
    main()
