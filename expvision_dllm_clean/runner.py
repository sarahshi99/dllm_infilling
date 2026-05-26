from __future__ import annotations

from typing import Any, Dict, List

from .config import ExperimentConfig
from .dataset import load_humaneval_infilling
from .decode import run_vanilla_decode
from .evaluation import summarize_results
from .logging import JsonlLogger
from .modeling import load_model_and_tokenizer, set_global_seed


def _format_bool_flag(value: bool) -> str:
    return "PASS" if value else "FAIL"


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    set_global_seed(cfg.decode.seed)
    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)
    logger.save_config(cfg)

    print("=" * 80, flush=True)
    print("Starting experiment", flush=True)
    print(f"experiment_name = {cfg.logging.experiment_name}", flush=True)
    print(f"output_dir       = {cfg.logging.output_dir}", flush=True)
    print(f"dataset_subset   = {cfg.data.dataset_subset}", flush=True)
    print(f"split            = {cfg.data.split}", flush=True)
    print(f"max_samples      = {cfg.data.max_samples}", flush=True)
    print(f"model_path       = {cfg.model.model_path}", flush=True)
    print(f"mask_length_src  = {cfg.decode.mask_length_source}", flush=True)
    print(f"fixed_mask_len   = {cfg.decode.fixed_mask_length}", flush=True)
    print(f"probe_lengths    = {cfg.decode.cal_lite_probe_lengths_csv}", flush=True)
    print(f"tie_break        = {cfg.decode.cal_lite_tie_break}", flush=True)
    print(f"score_mode       = {cfg.decode.cal_lite_score_mode}", flush=True)
    print(f"length_alpha     = {cfg.decode.cal_lite_length_alpha}", flush=True)
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

        result = run_vanilla_decode(task, tokenizer, model, cfg)
        result["dataset_subset"] = cfg.data.dataset_subset
        results.append(result)

        payload = {
            "task_id": result["task_id"],
            "dataset_subset": cfg.data.dataset_subset,
            "metrics": result["metrics"],
            "verification": result["verification"],
            "length_probe": result["length_probe"],
            "code": result["code"],
            "diagnostics": result["diagnostics"],
        }
        logger.log_result(payload)

        for trace in result["step_traces"]:
            logger.log_trace(trace)

        passed = bool(result["metrics"]["passed"])
        total_sec = float(result["metrics"]["total_sec"])
        total_sec_including_probe = float(result["metrics"]["total_sec_including_probe"])
        mask_length = result["metrics"].get("mask_length")
        oracle_mask_length = result["metrics"].get("oracle_mask_length")
        selected_minus_oracle = result["metrics"].get("selected_minus_oracle_length")
        abs_selected_minus_oracle = result["metrics"].get("abs_selected_minus_oracle_length")
        probe_sec = float(result["metrics"].get("length_probe_sec", 0.0))
        selected_raw_score = result["metrics"].get("selected_raw_score")
        selected_adjusted_score = result["metrics"].get("selected_adjusted_score")

        print(
            f"[{idx}/{total_tasks}] task_id={task.task_id} | "
            f"{_format_bool_flag(passed)} | "
            f"total_sec={total_sec:.3f} | "
            f"total_sec_including_probe={total_sec_including_probe:.3f} | "
            f"probe_sec={probe_sec:.3f} | "
            f"mask_len={mask_length} | "
            f"oracle_mask_len={oracle_mask_length} | "
            f"diff={selected_minus_oracle} | "
            f"abs_diff={abs_selected_minus_oracle} | "
            f"raw_score={selected_raw_score} | "
            f"adjusted_score={selected_adjusted_score}",
            flush=True,
        )

    summary = summarize_results(results)
    logger.save_json("summary.json", summary)

    print("-" * 80, flush=True)
    print("Experiment finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)

    return {
        "results": results,
        "summary": summary,
        "run_dir": logger.run_dir,
    }