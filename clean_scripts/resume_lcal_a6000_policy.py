#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import run_lcal_official_bounded_repair as legacy_runner
import run_lcal_official_bounded_repair_a6000_v2 as v2_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an interrupted A6000 LCAL official-bounded-repair policy run in-place. "
            "Completed task_ids in results.jsonl are skipped; missing tasks are appended."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Existing run directory with config.json/results.jsonl.")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def select_runner_module(config: Dict[str, Any]) -> ModuleType:
    repair = ((config.get("lcal_official_bounded_repair") or {}).get("repair") or {})
    v2_only_keys = {
        "mid_rescue_min_support_count",
        "mid_rescue_best_long_lens_csv",
        "mid_rescue_veto_s3_le",
        "mid_rescue_veto_official_len",
        "true_long_max_s3_len",
        "true_long_min_official_len",
        "true_long_min_long_ratio",
        "true_long_min_raw_ratio",
        "true_long_min_support_count",
        "true_long_best_long_lens_csv",
    }
    if any(key in repair for key in v2_only_keys):
        return v2_runner
    return legacy_runner


def apply_saved_config(cfg: Any, config: Dict[str, Any]) -> None:
    for section_name in ("model", "data", "decode", "logging"):
        section_payload = config.get(section_name)
        if not isinstance(section_payload, dict):
            continue
        section = getattr(cfg, section_name)
        for key, value in section_payload.items():
            setattr(section, key, value)


def _bias_params(payload: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    params = payload.get("bias_params", [1.0, 1.77, 0.56, 0.06, 0.24])
    if isinstance(params, str):
        return legacy_runner.official_cal.parse_bias_params(params)
    values = tuple(float(item) for item in params)
    if len(values) != 5:
        raise ValueError("official.bias_params must contain five values")
    return values  # type: ignore[return-value]


def build_cfg_settings_from_config(config: Dict[str, Any], runner: ModuleType) -> Tuple[Any, Any, Optional[str]]:
    cfg = runner.ExperimentConfig()
    apply_saved_config(cfg, config)
    cfg.decode.mask_length_source = "lcal_official_bounded_repair"

    block = config.get("lcal_official_bounded_repair") or {}
    lcal_payload = block.get("lcal") or {}
    official_payload = block.get("official") or {}
    repair_payload = block.get("repair") or {}

    lcal_settings = runner.rescue.LcalV3Settings(
        base_probe_lengths_csv=lcal_payload["base_probe_lengths_csv"],
        base_alpha=float(lcal_payload["base_alpha"]),
        weak_probe_lengths_csv=lcal_payload["weak_probe_lengths_csv"],
        strong_probe_lengths_csv=lcal_payload["strong_probe_lengths_csv"],
        long_alpha=float(lcal_payload["long_alpha"]),
        strong_min_len=int(lcal_payload["strong_min_len"]),
        weak_min_base_len=int(lcal_payload["weak_min_base_len"]),
        weak_max_base_len=int(lcal_payload["weak_max_base_len"]),
        ratio_trigger_threshold=float(lcal_payload["ratio_trigger_threshold"]),
        long_score_floor=float(lcal_payload["long_score_floor"]),
        raw_ratio_threshold=float(lcal_payload["raw_ratio_threshold"]),
        support_count_threshold=int(lcal_payload["support_count_threshold"]),
        cap_base_le8=int(lcal_payload["cap_base_le8"]),
        cap_base_9_12=int(lcal_payload["cap_base_9_12"]),
        short_safe_policy=str(lcal_payload["short_safe_policy"]),
        correction_selection_rule=str(lcal_payload["correction_selection_rule"]),
        shortest_supported_ratio=float(lcal_payload["shortest_supported_ratio"]),
        tie_break=str(lcal_payload["tie_break"]),
        score_mode=str(lcal_payload["score_mode"]),
    )
    official_settings = runner.official_cal.OfficialCalSettings(
        initial_length=int(official_payload["initial_length"]),
        span=int(official_payload["span"]),
        max_length=int(official_payload["max_length"]),
        dstep=int(official_payload["dstep"]),
        use_bias=bool(official_payload["use_bias"]),
        bias_params=_bias_params(official_payload),
        tie_break=str(official_payload.get("tie_break", "first")),
    )

    repair_keys = runner.BoundedRepairSettings.__dataclass_fields__.keys()
    repair_kwargs = {key: repair_payload[key] for key in repair_keys if key in repair_payload}
    settings = runner.BoundedRepairSettings(
        lcal=lcal_settings,
        official=official_settings,
        **repair_kwargs,
    )
    return cfg, settings, config.get("baseline_results")


def result_payload(result: Dict[str, Any], dataset_subset: str) -> Dict[str, Any]:
    return {
        "task_id": result["task_id"],
        "dataset_subset": dataset_subset,
        "metrics": result["metrics"],
        "verification": result["verification"],
        "length_probe": result["length_probe"],
        "stopping": result["stopping"],
        "lcal_v3": result.get("lcal_v3"),
        "official_cal": result.get("official_cal"),
        "code": result["code"],
        "diagnostics": result["diagnostics"],
    }


def should_print_progress(index: int, total: int, progress_every: int) -> bool:
    return index == 1 or index == total or (progress_every > 0 and index % progress_every == 0)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"
    trace_path = run_dir / "step_traces.jsonl"

    config = load_json(config_path)
    runner = select_runner_module(config)
    cfg, settings, baseline_results_path = build_cfg_settings_from_config(config, runner)
    existing_results = load_existing_results(results_path)
    existing_by_task_id = {str(row["task_id"]): row for row in existing_results}

    runner.patch_decode(settings)
    runner.set_global_seed(cfg.decode.seed)

    tasks = runner.load_humaneval_infilling(
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        dataset_subset=cfg.data.dataset_subset,
    )
    missing_tasks = [task for task in tasks if task.task_id not in existing_by_task_id]

    print("=" * 80, flush=True)
    print("Resuming A6000 LCAL official bounded repair policy", flush=True)
    print(f"run_dir            = {run_dir}", flush=True)
    print(f"runner             = {runner.__name__}", flush=True)
    print(f"carried_result_rows= {len(existing_results)}", flush=True)
    print(f"total_tasks        = {len(tasks)}", flush=True)
    print(f"remaining_to_decode= {len(missing_tasks)}", flush=True)
    print(f"model_path         = {cfg.model.model_path}", flush=True)
    print(f"baseline_results   = {baseline_results_path}", flush=True)
    print("=" * 80, flush=True)

    tokenizer = None
    model = None
    if missing_tasks:
        tokenizer, model = runner.load_model_and_tokenizer(cfg.model)

    results: List[Dict[str, Any]] = []
    carried_count = 0
    decoded_count = 0
    total_tasks = len(tasks)

    for idx, task in enumerate(tasks, start=1):
        existing = existing_by_task_id.get(task.task_id)
        if existing is not None:
            results.append(existing)
            carried_count += 1
            if should_print_progress(idx, total_tasks, args.progress_every):
                print(f"[{idx}/{total_tasks}] task_id={task.task_id} | SKIP carried", flush=True)
            continue

        assert tokenizer is not None and model is not None
        print(f"[{idx}/{total_tasks}] task_id={task.task_id} | start", flush=True)
        result = runner.decode_lcas.run_decode_with_lcas(task, tokenizer, model, cfg)
        runner.annotate_result(result)
        runner.annotate_lcas_v3_result(result)
        result["dataset_subset"] = cfg.data.dataset_subset
        payload = result_payload(result, cfg.data.dataset_subset)
        append_jsonl(results_path, payload)
        for trace in result.get("step_traces", []):
            append_jsonl(trace_path, trace)

        existing_by_task_id[task.task_id] = payload
        results.append(payload)
        decoded_count += 1

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

    baseline_rows = runner.load_jsonl(baseline_results_path) if baseline_results_path else []
    summary = runner.summarize_bounded_repair_results(results, baseline_rows, baseline_results_path)
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "resume_status.json",
        {
            "run_dir": str(run_dir),
            "runner": runner.__name__,
            "carried_over": carried_count,
            "decoded": decoded_count,
            "total_tasks": total_tasks,
            "baseline_results": baseline_results_path,
        },
    )

    print("-" * 80, flush=True)
    print("A6000 LCAL policy resume finished", flush=True)
    print(f"carried_over: {carried_count}", flush=True)
    print(f"decoded: {decoded_count}", flush=True)
    print(f"num_samples: {summary.get('num_samples')}", flush=True)
    print(f"pass_rate: {summary.get('pass_rate')}", flush=True)
    print(f"summary: {run_dir / 'summary.json'}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
