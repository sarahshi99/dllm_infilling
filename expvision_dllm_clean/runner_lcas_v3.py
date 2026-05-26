from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from .config import ExperimentConfig
from .dataset import load_humaneval_infilling
from .evaluation import summarize_results
from .logging import JsonlLogger
from .modeling import load_model_and_tokenizer, set_global_seed
from .stopping import GlobalGapStoppingConfig

import expvision_dllm_clean.decode_lcas as decode_lcas


def choose_lcas_v3_config(selected_mask_length: int, policy_name: str) -> GlobalGapStoppingConfig:
    """
    LCAS-v3 policies.

    Bucket definition:
      very_short: selected_length <= 8
      transition: 9 <= selected_length <= 12
      medium: 13 <= selected_length <= 20
      long: selected_length > 20

    Current action:
      v3a:
        very_short uses earlier but still safe stopping.
        all non-very-short buckets use safe_conservative.

      v3b:
        very_short uses relaxed_plus stopping.
        all non-very-short buckets use safe_conservative.

    This keeps long samples explicitly separable in logs, while avoiding
    candidate-grid or length-bias changes in the current experiment.
    """
    if policy_name == "lcas_v3a":
        if selected_mask_length <= 8:
            return GlobalGapStoppingConfig(
                min_stop_step=32,
                gap_threshold=0.50,
                top1_threshold=0.75,
                max_remaining_mask_ratio=0.50,
            )
        return GlobalGapStoppingConfig(
            min_stop_step=40,
            gap_threshold=0.50,
            top1_threshold=0.75,
            max_remaining_mask_ratio=0.375,
        )

    if policy_name == "lcas_v3b":
        if selected_mask_length <= 8:
            return GlobalGapStoppingConfig(
                min_stop_step=28,
                gap_threshold=0.40,
                top1_threshold=0.60,
                max_remaining_mask_ratio=0.625,
            )
        return GlobalGapStoppingConfig(
            min_stop_step=40,
            gap_threshold=0.50,
            top1_threshold=0.75,
            max_remaining_mask_ratio=0.375,
        )

    raise ValueError(f"Unknown LCAS-v3 policy: {policy_name}")


def lcas_v3_bucket(selected_mask_length: int) -> str:
    if selected_mask_length <= 8:
        return "very_short_selected_<=8"
    if selected_mask_length <= 12:
        return "transition_selected_9_12"
    if selected_mask_length <= 20:
        return "medium_selected_13_20"
    return "long_selected_>20"


def patch_decode_lcas_policy() -> None:
    decode_lcas.choose_lcas_config = choose_lcas_v3_config


def annotate_lcas_v3_result(result: Dict[str, Any]) -> None:
    selected_length = int(result["metrics"]["selected_mask_length"])
    bucket = lcas_v3_bucket(selected_length)

    result["metrics"]["lcas_bucket"] = bucket
    result["metrics"]["lcas_v3_bucket"] = bucket

    result["stopping"]["bucket"] = bucket
    result["stopping"]["v3_bucket"] = bucket

    if result.get("stopping_trace"):
        for item in result["stopping_trace"]:
            item["lcas_bucket"] = bucket
            item["lcas_v3_bucket"] = bucket


def _avg(values: Iterable[float | int | None]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


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


def summarize_lcas_v3_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_summary = summarize_results(results)
    metrics = [item["metrics"] for item in results]
    buckets = sorted(set(m.get("lcas_v3_bucket") for m in metrics))

    base_summary.update(
        {
            "stop_rate": _rate(bool(m.get("stopped", False)) for m in metrics),
            "avg_effective_steps": _avg(m.get("effective_steps") for m in metrics),
            "avg_stop_step_stopped_only": _avg(
                m.get("stop_step") for m in metrics if bool(m.get("stopped", False))
            ),
            "avg_remaining_masks_at_stop_or_final": _avg(
                m.get("remaining_masks_at_stop_or_final") for m in metrics
            ),
            "avg_remaining_mask_ratio_at_stop_or_final": _avg(
                m.get("remaining_mask_ratio_at_stop_or_final") for m in metrics
            ),
            "avg_mean_gap_at_stop_or_final": _avg(m.get("mean_gap_at_stop_or_final") for m in metrics),
            "avg_mean_top1_at_stop_or_final": _avg(m.get("mean_top1_at_stop_or_final") for m in metrics),
            "stop_reason_histogram": _hist(m.get("stop_reason") for m in metrics),
            "lcas_v3_bucket_histogram": _hist(m.get("lcas_v3_bucket") for m in metrics),
            "lcas_v3_bucket_pass_rate": {
                bucket: _rate(m.get("passed", False) for m in metrics if m.get("lcas_v3_bucket") == bucket)
                for bucket in buckets
            },
            "lcas_v3_bucket_stop_rate": {
                bucket: _rate(bool(m.get("stopped", False)) for m in metrics if m.get("lcas_v3_bucket") == bucket)
                for bucket in buckets
            },
            "lcas_v3_bucket_avg_effective_steps": {
                bucket: _avg(m.get("effective_steps") for m in metrics if m.get("lcas_v3_bucket") == bucket)
                for bucket in buckets
            },
        }
    )
    return base_summary


def run_lcas_v3_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    patch_decode_lcas_policy()

    set_global_seed(cfg.decode.seed)
    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)
    logger.save_config(cfg)

    print("=" * 80, flush=True)
    print("Starting LCAS-v3 experiment", flush=True)
    print(f"experiment_name = {cfg.logging.experiment_name}", flush=True)
    print(f"output_dir       = {cfg.logging.output_dir}", flush=True)
    print(f"dataset_subset   = {cfg.data.dataset_subset}", flush=True)
    print(f"split            = {cfg.data.split}", flush=True)
    print(f"max_samples      = {cfg.data.max_samples}", flush=True)
    print(f"model_path       = {cfg.model.model_path}", flush=True)
    print(f"mask_length_src  = {cfg.decode.mask_length_source}", flush=True)
    print(f"probe_lengths    = {getattr(cfg.decode, 'cal_lite_probe_lengths_csv', None)}", flush=True)
    print(f"score_mode       = {getattr(cfg.decode, 'cal_lite_score_mode', None)}", flush=True)
    print(f"length_alpha     = {getattr(cfg.decode, 'cal_lite_length_alpha', None)}", flush=True)
    print(f"lcas_policy      = {getattr(cfg.decode, 'lcas_policy', None)}", flush=True)
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
        results.append(result)

        payload = {
            "task_id": result["task_id"],
            "dataset_subset": cfg.data.dataset_subset,
            "metrics": result["metrics"],
            "verification": result["verification"],
            "length_probe": result["length_probe"],
            "stopping": result["stopping"],
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
            f"mask_len={m['mask_length']} | "
            f"oracle_mask_len={m['oracle_mask_length']} | "
            f"diff={m['selected_minus_oracle_length']} | "
            f"bucket={m['lcas_v3_bucket']} | "
            f"stopped={m['stopped']} | "
            f"stop_step={m['stop_step']} | "
            f"effective_steps={m['effective_steps']} | "
            f"stop_reason={m['stop_reason']}",
            flush=True,
        )

    summary = summarize_lcas_v3_results(results)
    logger.save_json("summary.json", summary)

    print("-" * 80, flush=True)
    print("LCAS-v3 experiment finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)

    return {
        "results": results,
        "summary": summary,
        "run_dir": logger.run_dir,
    }