#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.join(ROOT, "clean_scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import expvision_dllm_clean.decode_lcas as decode_lcas
from clean_scripts import run_cal_official_lcas_v3 as official_cal
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import load_humaneval_infilling
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.runner_lcas_v3 import annotate_lcas_v3_result
from experiments.action_ceiling.action_ceiling_matrix import current_branch, current_commit, environment_info, git_capture
from analysis.trace_long_rescue_features import load_jsonl


JsonDict = Dict[str, Any]


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        pass
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items() if str(key) != "task"}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def read_config(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_settings(config_payload: Mapping[str, Any]) -> official_cal.OfficialCalSettings:
    official = config_payload.get("official_cal") or {}
    return official_cal.OfficialCalSettings(
        initial_length=int(official.get("initial_length", 8)),
        span=int(official.get("span", 1)),
        max_length=int(official.get("max_length", 64)),
        dstep=int(official.get("dstep", 4)),
        use_bias=bool(official.get("use_bias", True)),
        bias_params=tuple(float(v) for v in official.get("bias_params", official_cal.DEFAULT_BIAS_PARAMS.split(","))),
    )


def build_cfg(config_payload: Mapping[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = str((config_payload.get("model") or {}).get("model_path", "GSAI-ML/LLaDA-8B-Base"))
    cfg.data.split = str((config_payload.get("data") or {}).get("split", "test"))
    cfg.data.dataset_subset = str((config_payload.get("data") or {}).get("dataset_subset", "HumanEval-SingleLineInfilling"))
    cfg.data.max_samples = None
    decode = config_payload.get("decode") or {}
    cfg.decode.mask_length_source = "official_cal"
    cfg.decode.total_steps = int(decode.get("total_steps", 64))
    cfg.decode.seed = int(decode.get("seed", 42))
    cfg.decode.lcas_policy = str(decode.get("lcas_policy", "lcas_v3b"))
    cfg.decode.save_step_traces = False
    cfg.decode.save_full_text_per_step = False
    return cfg


def result_payload(result: Mapping[str, Any], dataset_subset: str) -> JsonDict:
    return {
        "task_id": result["task_id"],
        "dataset_subset": dataset_subset,
        "metrics": result["metrics"],
        "verification": result["verification"],
        "length_probe": result["length_probe"],
        "stopping": result["stopping"],
        "official_cal": result["official_cal"],
        "code": result["code"],
        "diagnostics": result["diagnostics"],
    }


def run_missing_tasks(
    *,
    tasks: Sequence[Any],
    tokenizer: Any,
    model: Any,
    cfg: ExperimentConfig,
    logger: JsonlLogger,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    total = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        print(f"[resume {idx}/{total}] task_id={task.task_id} | start", flush=True)
        result = decode_lcas.run_decode_with_lcas(task, tokenizer, model, cfg)
        annotate_lcas_v3_result(result)
        result["dataset_subset"] = cfg.data.dataset_subset
        result["official_cal"] = copy.deepcopy(official_cal._CAL_META_BY_TASK_ID.get(task.task_id, {}))
        result["length_probe"]["official_cal"] = copy.deepcopy(result["official_cal"])
        m = result["metrics"]
        m["official_cal_search_steps"] = result["official_cal"].get("search_steps")
        m["official_cal_selected_bias"] = result["official_cal"].get("selected_bias")
        m["official_cal_selected_calibrated_confidence"] = result["official_cal"].get("selected_calibrated_confidence")
        m["selected_raw_score"] = result["official_cal"].get("selected_raw_score")
        m["selected_adjusted_score"] = result["official_cal"].get("selected_calibrated_confidence")
        m["score_mode"] = "official_cal_bias" if official_cal._ACTIVE_SETTINGS and official_cal._ACTIVE_SETTINGS.use_bias else "official_cal_raw"

        payload = result_payload(result, cfg.data.dataset_subset)
        logger.log_result(payload)
        rows.append(payload)
        print(
            f"[resume {idx}/{total}] task_id={task.task_id} | "
            f"{'PASS' if m['passed'] else 'FAIL'} | "
            f"total_sec_including_probe={m['total_sec_including_probe']:.3f} | "
            f"probe_steps={m['official_cal_search_steps']} | "
            f"mask_len={m['mask_length']} | oracle_mask_len={m['oracle_mask_length']} | "
            f"diff={m['selected_minus_oracle_length']} | stop_reason={m['stop_reason']}",
            flush=True,
        )
    return rows


def manifest_payload(args: argparse.Namespace, status: str, started_at: str, **extra: Any) -> JsonDict:
    command = shlex.join([sys.executable, *sys.argv])
    env_prefix = {
        key: os.environ[key]
        for key in ("CUDA_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        if key in os.environ
    }
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env_prefix.items()))
    return {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": command,
        "repro_command": f"{env_text} {command}".strip(),
        "env_command_prefix": env_prefix,
        "started_at": started_at,
        "execution_status": status,
        "partial_run_dir": args.partial_run_dir,
        "environment": environment_info(args.model_path),
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
        **extra,
    }


def execute(args: argparse.Namespace) -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_output_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload(args, "running", started_at), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        partial_dir = Path(args.partial_run_dir)
        partial_rows = list(load_jsonl(str(partial_dir / "results.jsonl")))
        partial_task_ids = [str(row["task_id"]) for row in partial_rows]
        duplicates = sorted({task_id for task_id in partial_task_ids if partial_task_ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate task ids in partial run: {duplicates[:10]}")

        config_payload = read_config(partial_dir / "config.json")
        settings = build_settings(config_payload)
        cfg = build_cfg(config_payload)
        cfg.model.model_path = args.model_path
        official_cal.patch_decode_for_official_cal(settings)
        set_global_seed(cfg.decode.seed)

        all_tasks = load_humaneval_infilling(split=cfg.data.split, dataset_subset=cfg.data.dataset_subset)
        all_task_ids = [task.task_id for task in all_tasks]
        missing_tasks = [task for task in all_tasks if task.task_id not in set(partial_task_ids)]
        if len(partial_rows) + len(missing_tasks) != len(all_tasks):
            raise RuntimeError(
                f"coverage mismatch before resume: partial={len(partial_rows)} missing={len(missing_tasks)} total={len(all_tasks)}"
            )
        if not missing_tasks:
            raise RuntimeError("partial run is already complete; no missing tasks to resume")

        logger = JsonlLogger(str(output_dir), args.supplement_experiment_name)
        resume_config = {
            **config_payload,
            "resume": {
                "partial_run_dir": str(partial_dir),
                "partial_count": len(partial_rows),
                "missing_count": len(missing_tasks),
                "missing_task_ids": [task.task_id for task in missing_tasks],
                "resume_script": "experiments/action_ceiling/cal_official_lcas_v3_resume.py",
            },
        }
        logger.save_config(resume_config)

        print("=" * 80, flush=True)
        print("Starting official CAL resume", flush=True)
        print(f"partial_run_dir = {partial_dir}", flush=True)
        print(f"supplement_dir  = {logger.run_dir}", flush=True)
        print(f"partial_count   = {len(partial_rows)}", flush=True)
        print(f"missing_count   = {len(missing_tasks)}", flush=True)
        print("=" * 80, flush=True)

        tokenizer, model = load_model_and_tokenizer(cfg.model)
        supplement_rows = run_missing_tasks(
            tasks=missing_tasks,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            logger=logger,
        )

        combined_by_task = {str(row["task_id"]): row for row in partial_rows}
        combined_by_task.update({str(row["task_id"]): row for row in supplement_rows})
        missing_after = [task_id for task_id in all_task_ids if task_id not in combined_by_task]
        if missing_after:
            raise RuntimeError(f"missing after resume: {missing_after[:10]}")
        combined_rows = [combined_by_task[task_id] for task_id in all_task_ids]
        if len(combined_rows) != len(all_tasks):
            raise RuntimeError(f"combined row count mismatch: {len(combined_rows)} != {len(all_tasks)}")

        merged_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_dir = output_dir / f"{args.merged_experiment_name}_{merged_stamp}"
        merged_dir.mkdir(parents=False, exist_ok=False)
        merged_config = {
            **resume_config,
            "resume": {
                **resume_config["resume"],
                "supplement_run_dir": logger.run_dir,
                "supplement_count": len(supplement_rows),
                "merged_run_dir": str(merged_dir),
                "merged_count": len(combined_rows),
            },
        }
        (merged_dir / "config.json").write_text(
            json.dumps(json_safe(merged_config), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_jsonl(merged_dir / "results.jsonl", combined_rows)
        summary = official_cal.summarize_official_cal_results([dict(row) for row in combined_rows])
        (merged_dir / "summary.json").write_text(
            json.dumps(json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        manifest = manifest_payload(
            args,
            "completed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            partial_count=len(partial_rows),
            supplement_count=len(supplement_rows),
            merged_count=len(combined_rows),
            total_expected=len(all_tasks),
            supplement_run_dir=logger.run_dir,
            merged_run_dir=str(merged_dir),
            summary_path=str(merged_dir / "summary.json"),
        )
        manifest_path.write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"merged_run_dir": str(merged_dir), "merged_count": len(combined_rows)}, indent=2, sort_keys=True))
    except BaseException:
        manifest = manifest_payload(
            args,
            "failed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            failure_traceback=traceback.format_exc(),
        )
        manifest_path.write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume an interrupted local official CAL + LCAS-v3 full run.")
    parser.add_argument("--partial-run-dir", required=True)
    parser.add_argument("--output-dir", default="/home/shx/projects/dllm_infilling/outputs_clean")
    parser.add_argument("--manifest-output-dir", required=True)
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--supplement-experiment-name", default="full_official_cal_lcas_v3b_accel_gpus10_20260702_supplement")
    parser.add_argument("--merged-experiment-name", default="full_official_cal_lcas_v3b_accel_gpus10_20260702_resumed_full")
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
