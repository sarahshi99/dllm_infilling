#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import load_humaneval_infilling
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.runner_lcas_v3 import (
    annotate_lcas_v3_result,
    patch_decode_lcas_policy,
    summarize_lcas_v3_results,
)

import expvision_dllm_clean.decode_lcas as decode_lcas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume a CAL-lite LCAS-v3 run from an existing results.jsonl file. "
            "Completed task_ids are carried into a new run directory and skipped."
        )
    )
    parser.add_argument("--resume-results", type=str, required=True)
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Defaults to config.json next to --resume-results when present.",
    )

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--dataset-subset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--probe-lengths", type=str, default=None)
    parser.add_argument("--tie-break", type=str, default=None, choices=["shorter", "longer"])
    parser.add_argument("--score-mode", type=str, default=None, choices=["raw", "length_power"])
    parser.add_argument("--length-alpha", type=float, default=None)
    parser.add_argument("--lcas-policy", type=str, default=None, choices=["lcas_v3a", "lcas_v3b"])

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--save-step-traces", action="store_true", default=None)
    parser.add_argument("--save-full-text-per-step", action="store_true", default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_default_config_path(results_path: Path, explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        return Path(explicit_path)
    candidate = results_path.parent / "config.json"
    return candidate if candidate.exists() else None


def apply_saved_config(cfg: ExperimentConfig, saved: Dict[str, Any]) -> None:
    for section_name in ("model", "data", "decode", "logging"):
        section_payload = saved.get(section_name)
        if not isinstance(section_payload, dict):
            continue
        section = getattr(cfg, section_name)
        for key, value in section_payload.items():
            setattr(section, key, value)


def load_existing_results(path: Path) -> List[Dict[str, Any]]:
    by_task_id: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            task_id = row.get("task_id")
            if not task_id:
                raise ValueError(f"Missing task_id in {path} line {line_no}")
            by_task_id[str(task_id)] = row
    return list(by_task_id.values())


def infer_lcas_policy(existing_results: List[Dict[str, Any]]) -> str:
    for row in existing_results:
        metrics = row.get("metrics") or {}
        policy = metrics.get("lcas_policy")
        if policy:
            return str(policy)
        stopping = row.get("stopping") or {}
        policy = stopping.get("policy")
        if policy:
            return str(policy)
    return "lcas_v3a"


def build_config(args: argparse.Namespace, saved_config: Optional[Dict[str, Any]], existing_results: List[Dict[str, Any]]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if saved_config:
        apply_saved_config(cfg, saved_config)
    else:
        cfg.decode.mask_length_source = "cal_lite"

    if args.model_path is not None:
        cfg.model.model_path = args.model_path
    if args.split is not None:
        cfg.data.split = args.split
    if args.dataset_subset is not None:
        cfg.data.dataset_subset = args.dataset_subset
    if args.max_samples is not None:
        cfg.data.max_samples = args.max_samples
    if args.total_steps is not None:
        cfg.decode.total_steps = args.total_steps
    if args.seed is not None:
        cfg.decode.seed = args.seed

    if args.probe_lengths is not None:
        cfg.decode.cal_lite_probe_lengths_csv = args.probe_lengths
    if args.tie_break is not None:
        cfg.decode.cal_lite_tie_break = args.tie_break
    if args.score_mode is not None:
        cfg.decode.cal_lite_score_mode = args.score_mode
    if args.length_alpha is not None:
        cfg.decode.cal_lite_length_alpha = args.length_alpha

    policy = args.lcas_policy or getattr(cfg.decode, "lcas_policy", None) or infer_lcas_policy(existing_results)
    cfg.decode.lcas_policy = policy

    if args.save_step_traces is not None:
        cfg.decode.save_step_traces = bool(args.save_step_traces)
    if args.save_full_text_per_step is not None:
        cfg.decode.save_full_text_per_step = bool(args.save_full_text_per_step)

    if args.output_dir is not None:
        cfg.logging.output_dir = args.output_dir
    if args.experiment_name is not None:
        cfg.logging.experiment_name = args.experiment_name
    else:
        base_name = cfg.logging.experiment_name or "cal_lite_lcas_v3_clean"
        cfg.logging.experiment_name = base_name if base_name.endswith("_resume") else f"{base_name}_resume"

    return cfg


def config_payload(cfg: ExperimentConfig, args: argparse.Namespace, existing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = cfg.to_dict()
    payload.setdefault("decode", {})["lcas_policy"] = getattr(cfg.decode, "lcas_policy", None)
    payload["resume"] = {
        "resume_results": args.resume_results,
        "carried_over_results": len(existing_results),
        "note": "Completed task_ids are logged into this run and skipped during decoding.",
    }
    return payload


def should_print_progress(idx: int, total: int, progress_every: int) -> bool:
    every = max(1, progress_every)
    return idx == 1 or idx == total or idx % every == 0


def result_payload(result: Dict[str, Any], dataset_subset: str) -> Dict[str, Any]:
    return {
        "task_id": result["task_id"],
        "dataset_subset": dataset_subset,
        "metrics": result["metrics"],
        "verification": result["verification"],
        "length_probe": result["length_probe"],
        "stopping": result["stopping"],
        "code": result["code"],
        "diagnostics": result["diagnostics"],
    }


def main() -> None:
    args = parse_args()
    resume_results_path = Path(args.resume_results)
    if not resume_results_path.exists():
        raise FileNotFoundError(f"Resume results not found: {resume_results_path}")

    config_path = maybe_default_config_path(resume_results_path, args.config_path)
    saved_config = load_json(config_path) if config_path else None
    existing_results = load_existing_results(resume_results_path)
    existing_by_task_id = {str(row["task_id"]): row for row in existing_results}

    cfg = build_config(args, saved_config, existing_results)
    patch_decode_lcas_policy()
    set_global_seed(cfg.decode.seed)

    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)
    logger.save_config(config_payload(cfg, args, existing_results))

    print("=" * 80, flush=True)
    print("Resuming LCAS-v3 experiment", flush=True)
    print(f"experiment_name     = {cfg.logging.experiment_name}", flush=True)
    print(f"output_dir          = {cfg.logging.output_dir}", flush=True)
    print(f"resume_results      = {resume_results_path}", flush=True)
    print(f"resume_config       = {config_path}", flush=True)
    print(f"carried_result_rows = {len(existing_results)}", flush=True)
    print(f"dataset_subset      = {cfg.data.dataset_subset}", flush=True)
    print(f"split               = {cfg.data.split}", flush=True)
    print(f"max_samples         = {cfg.data.max_samples}", flush=True)
    print(f"model_path          = {cfg.model.model_path}", flush=True)
    print(f"probe_lengths       = {getattr(cfg.decode, 'cal_lite_probe_lengths_csv', None)}", flush=True)
    print(f"length_alpha        = {getattr(cfg.decode, 'cal_lite_length_alpha', None)}", flush=True)
    print(f"lcas_policy         = {getattr(cfg.decode, 'lcas_policy', None)}", flush=True)
    print(f"total_steps         = {cfg.decode.total_steps}", flush=True)
    print(f"seed                = {cfg.decode.seed}", flush=True)
    print("=" * 80, flush=True)

    tasks = load_humaneval_infilling(
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        dataset_subset=cfg.data.dataset_subset,
    )
    task_ids = {task.task_id for task in tasks}
    extra_result_ids = sorted(set(existing_by_task_id) - task_ids)
    missing_tasks = [task for task in tasks if task.task_id not in existing_by_task_id]

    print(f"Loaded {len(tasks)} tasks", flush=True)
    print(f"Already complete in current task set: {len(tasks) - len(missing_tasks)}", flush=True)
    print(f"Remaining to decode: {len(missing_tasks)}", flush=True)
    if extra_result_ids:
        print(f"Ignoring {len(extra_result_ids)} carried rows outside current task set", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("-" * 80, flush=True)

    tokenizer = None
    model = None
    if missing_tasks:
        tokenizer, model = load_model_and_tokenizer(cfg.model)

    results: List[Dict[str, Any]] = []
    carried_count = 0
    decoded_count = 0
    total_tasks = len(tasks)

    for idx, task in enumerate(tasks, start=1):
        existing = existing_by_task_id.get(task.task_id)
        if existing is not None:
            existing.setdefault("dataset_subset", cfg.data.dataset_subset)
            logger.log_result(existing)
            results.append(existing)
            carried_count += 1
            if should_print_progress(idx, total_tasks, args.progress_every):
                print(f"[{idx}/{total_tasks}] task_id={task.task_id} | SKIP carried", flush=True)
            continue

        assert tokenizer is not None and model is not None
        print(f"[{idx}/{total_tasks}] task_id={task.task_id} | start", flush=True)
        result = decode_lcas.run_decode_with_lcas(task, tokenizer, model, cfg)
        annotate_lcas_v3_result(result)
        result["dataset_subset"] = cfg.data.dataset_subset

        payload = result_payload(result, cfg.data.dataset_subset)
        logger.log_result(payload)
        for trace in result["step_traces"]:
            logger.log_trace(trace)

        results.append(payload)
        decoded_count += 1
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
    logger.save_json(
        "resume_status.json",
        {
            "resume_results": str(resume_results_path),
            "resume_config": str(config_path) if config_path else None,
            "carried_over": carried_count,
            "decoded": decoded_count,
            "total_tasks": total_tasks,
            "extra_result_ids_ignored": extra_result_ids,
        },
    )

    print("-" * 80, flush=True)
    print("LCAS-v3 resume finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"carried_over: {carried_count}", flush=True)
    print(f"decoded: {decoded_count}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
