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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.join(ROOT, "clean_scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import expvision_dllm_clean.decode_lcas as decode_lcas
from experiments.action_ceiling.action_ceiling_matrix import (
    DEFAULT_BASELINE,
    DEFAULT_ROUTE2,
    current_branch,
    current_commit,
    environment_info,
    git_capture,
    rows_by_task,
    verification_error_type,
    write_csv,
    write_jsonl,
)
from analysis.trace_long_rescue_features import load_jsonl
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import load_humaneval_infilling
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.runner_lcas_v3 import annotate_lcas_v3_result
from clean_scripts import run_cal_official_lcas_v3 as official_cal


JsonDict = Dict[str, Any]

FIXED_CASE_IDS = [
    "SingleLineInfilling/HumanEval/0/L0",
    "SingleLineInfilling/HumanEval/0/L1",
    "SingleLineInfilling/HumanEval/60/L0",
    "SingleLineInfilling/HumanEval/116/L0",
    "SingleLineInfilling/HumanEval/85/L0",
    "SingleLineInfilling/HumanEval/113/L3",
    "SingleLineInfilling/HumanEval/108/L6",
    "SingleLineInfilling/HumanEval/104/L2",
    "SingleLineInfilling/HumanEval/34/L0",
    "SingleLineInfilling/HumanEval/90/L1",
]

VERDICTS = {
    "protocol_matched_cal",
    "cal_lite_not_official",
    "protocol_mismatch_repairable",
    "protocol_mismatch_blocked",
}


def make_output_dir(base_dir: str, timestamp: Optional[str]) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"cal_same_protocol_sanity_{stamp}"
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def compact_result(result: Mapping[str, Any], baseline_by_task: Mapping[str, Mapping[str, Any]], route2_by_task: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    task_id = str(result["task_id"])
    metrics = result.get("metrics") or {}
    baseline = baseline_by_task.get(task_id, {})
    route2 = route2_by_task.get(task_id, {})
    return {
        "task_id": task_id,
        "passed": metrics.get("passed"),
        "error_type": verification_error_type(result),
        "selected_mask_length": metrics.get("selected_mask_length"),
        "oracle_mask_length": metrics.get("oracle_mask_length"),
        "selected_minus_oracle_length": metrics.get("selected_minus_oracle_length"),
        "total_sec_including_probe": metrics.get("total_sec_including_probe"),
        "official_cal_search_steps": metrics.get("official_cal_search_steps"),
        "score_mode": metrics.get("score_mode"),
        "stop_reason": metrics.get("stop_reason"),
        "effective_steps": metrics.get("effective_steps"),
        "control_passed": (baseline.get("metrics") or {}).get("passed"),
        "route2_passed": (route2.get("metrics") or {}).get("passed"),
        "control_selected_mask_length": (baseline.get("metrics") or {}).get("selected_mask_length"),
        "route2_selected_mask_length": (route2.get("metrics") or {}).get("selected_mask_length"),
    }


def protocol_checks(args: argparse.Namespace, config_payload: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> JsonDict:
    metrics = [row.get("metrics") or {} for row in results]
    official_meta = [row.get("official_cal") or {} for row in results]
    checks = {
        "checkpoint_same": args.model_path == "GSAI-ML/LLaDA-8B-Base",
        "dataset_subset_same": args.dataset_subset == "HumanEval-SingleLineInfilling",
        "split_same": args.split == "test",
        "prompt_format_same": bool(config_payload["decode"]["add_special_tokens_to_prefix"]) and not bool(config_payload["decode"]["add_special_tokens_to_suffix"]),
        "mask_initialization_same": True,
        "denoising_budget_comparable": int(args.total_steps) == 64,
        "evaluation_harness_same": all("tier3_unit_tests" in (row.get("verification") or {}) for row in results),
        "seed_protocol_explicit": int(args.seed) == 42,
        "no_oracle_length_used_for_generation": all((row.get("length_probe") or {}).get("tie_break") == "first" for row in results),
        "no_test_result_used_for_generation": True,
        "output_schema_paired_compare": all("task_id" in row and "metrics" in row and "verification" in row for row in results),
        "official_cal_metadata_present": all(meta.get("version") == "official_cal_lcas_v3" for meta in official_meta),
        "fixed_case_filter_wrapper": True,
    }
    mismatches = [key for key, value in checks.items() if not value]
    if mismatches:
        verdict = "protocol_mismatch_repairable"
    elif not checks["official_cal_metadata_present"]:
        verdict = "cal_lite_not_official"
    else:
        verdict = "protocol_matched_cal"
    return {"checks": checks, "mismatches": mismatches, "verdict": verdict}


def render_protocol_comparison(protocol: Mapping[str, Any], results: Sequence[Mapping[str, Any]], run_dir: Path) -> str:
    lines = [
        "# CAL Same-Protocol Sanity",
        "",
        f"verdict: `{protocol.get('verdict')}`",
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "|---|---:|",
    ]
    for key, value in (protocol.get("checks") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This sanity uses the local `run_cal_official_lcas_v3.py` implementation and a wrapper-level fixed 10-case filter.",
            "- It is a same-repository protocol sanity, not an external official-code reproduction claim.",
            "- If verdict is `protocol_matched_cal`, a full run may be compared under the exact same local protocol; otherwise no misleading full comparison should be launched.",
            "",
            "## Artifacts",
            "",
            f"- run dir: `{run_dir}`",
            f"- results: `{run_dir / 'results.csv'}`",
            f"- summary: `{run_dir / 'summary.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_run_manifest(args: argparse.Namespace, output_dir: Path, status: str, started_at: str, **extra: Any) -> JsonDict:
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
        "output_dir": str(output_dir),
        "environment": environment_info(args.model_path),
        "fixed_case_ids": list(FIXED_CASE_IDS),
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
        **extra,
    }


def execute(args: argparse.Namespace) -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    output_dir = make_output_dir(args.output_dir, args.timestamp)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(build_run_manifest(args, output_dir, "running", started_at), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        settings = official_cal.OfficialCalSettings(
            initial_length=args.initial_length,
            span=args.span,
            max_length=args.max_length,
            dstep=args.dstep,
            use_bias=not args.no_bias,
            bias_params=official_cal.parse_bias_params(args.bias_params),
        )
        official_cal.patch_decode_for_official_cal(settings)
        cfg = ExperimentConfig()
        cfg.model.model_path = args.model_path
        cfg.data.split = args.split
        cfg.data.dataset_subset = args.dataset_subset
        cfg.decode.mask_length_source = "official_cal"
        cfg.decode.total_steps = args.total_steps
        cfg.decode.seed = args.seed
        cfg.decode.lcas_policy = args.lcas_policy
        cfg.decode.save_step_traces = False
        cfg.decode.save_full_text_per_step = False
        set_global_seed(cfg.decode.seed)

        tasks = load_humaneval_infilling(split=args.split, dataset_subset=args.dataset_subset)
        by_task = {task.task_id: task for task in tasks}
        missing = [task_id for task_id in FIXED_CASE_IDS if task_id not in by_task]
        if missing:
            raise ValueError(f"fixed sanity task ids missing: {missing}")
        selected_tasks = [by_task[task_id] for task_id in FIXED_CASE_IDS]
        case_manifest = [{"task_id": task.task_id, "order": idx} for idx, task in enumerate(selected_tasks)]
        write_csv(output_dir / "case_manifest.csv", case_manifest)

        tokenizer, model = load_model_and_tokenizer(cfg.model)
        results: List[Dict[str, Any]] = []
        for idx, task in enumerate(selected_tasks, start=1):
            print(f"[{idx}/{len(selected_tasks)}] CAL sanity task_id={task.task_id}", flush=True)
            result = decode_lcas.run_decode_with_lcas(task, tokenizer, model, cfg)
            annotate_lcas_v3_result(result)
            result["dataset_subset"] = cfg.data.dataset_subset
            result["official_cal"] = copy.deepcopy(official_cal._CAL_META_BY_TASK_ID.get(task.task_id, {}))
            result["length_probe"]["official_cal"] = copy.deepcopy(result["official_cal"])
            metrics = result["metrics"]
            metrics["official_cal_search_steps"] = result["official_cal"].get("search_steps")
            metrics["official_cal_selected_bias"] = result["official_cal"].get("selected_bias")
            metrics["official_cal_selected_calibrated_confidence"] = result["official_cal"].get("selected_calibrated_confidence")
            metrics["selected_raw_score"] = result["official_cal"].get("selected_raw_score")
            metrics["selected_adjusted_score"] = result["official_cal"].get("selected_calibrated_confidence")
            metrics["score_mode"] = "official_cal_bias" if settings.use_bias else "official_cal_raw"
            results.append(result)

        baseline_by_task = rows_by_task(load_jsonl(args.baseline_results), name="baseline")
        route2_by_task = rows_by_task(load_jsonl(args.route2_results), name="route2")
        config_payload = cfg.to_dict()
        config_payload["official_cal"] = {**settings.__dict__, "bias_params": list(settings.bias_params)}
        protocol = protocol_checks(args, config_payload, results)
        if protocol["verdict"] not in VERDICTS:
            raise ValueError(f"invalid CAL sanity verdict: {protocol['verdict']}")
        compact_rows = [compact_result(row, baseline_by_task, route2_by_task) for row in results]
        summary = official_cal.summarize_official_cal_results(results)
        summary["protocol"] = protocol
        summary["pass_count"] = sum(1 for row in results if (row.get("metrics") or {}).get("passed"))
        summary["case_count"] = len(results)
        summary["error_type_histogram"] = dict(Counter(str(row.get("error_type")) for row in compact_rows))
        write_jsonl(output_dir / "results.jsonl", results)
        write_csv(output_dir / "results.csv", compact_rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "protocol_comparison.md").write_text(render_protocol_comparison(protocol, results, output_dir), encoding="utf-8")
        (output_dir / "verdict.md").write_text(f"# CAL Sanity Verdict\n\n`{protocol['verdict']}`\n", encoding="utf-8")
        manifest = build_run_manifest(
            args,
            output_dir,
            "completed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            verdict=protocol["verdict"],
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "verdict": protocol["verdict"]}, ensure_ascii=False, indent=2))
    except BaseException:
        manifest = build_run_manifest(
            args,
            output_dir,
            "failed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            failure_traceback=traceback.format_exc(),
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed 10-case CAL same-protocol sanity.")
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE)
    parser.add_argument("--route2-results", default=DEFAULT_ROUTE2)
    parser.add_argument("--output-dir", default="analysis_outputs")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-subset", default="HumanEval-SingleLineInfilling")
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-length", type=int, default=8)
    parser.add_argument("--span", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dstep", type=int, default=4)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--bias-params", default=official_cal.DEFAULT_BIAS_PARAMS)
    parser.add_argument("--lcas-policy", default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    return parser.parse_args()


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
