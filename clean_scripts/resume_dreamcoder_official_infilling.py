#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import load_humaneval_infilling
from expvision_dllm_clean.modeling import get_torch_dtype, set_global_seed

import run_dreamcoder_official_infilling as dream_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume an interrupted Dream-Coder official infilling run.")
    parser.add_argument("--run-dir", required=True, help="Existing run directory containing config.json/results.jsonl.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_result(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_cfg(config: Dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = config["model"]["model_path"]
    cfg.model.torch_dtype = config["model"]["torch_dtype"]
    cfg.model.device_map = config["model"]["device_map"]
    cfg.model.trust_remote_code = config["model"].get("trust_remote_code", True)

    cfg.data.split = config["data"]["split"]
    cfg.data.dataset_subset = config["data"]["dataset_subset"]
    cfg.data.max_samples = config["data"].get("max_samples")

    decode = config["decode"]
    cfg.decode.fixed_mask_length = int(decode["fixed_mask_length"])
    cfg.decode.mask_length_source = decode["mask_length_source"]
    cfg.decode.total_steps = int(decode["total_steps"])
    cfg.decode.seed = int(decode["seed"])
    cfg.decode.cal_lite_probe_lengths_csv = decode["cal_lite_probe_lengths_csv"]
    cfg.decode.cal_lite_tie_break = decode["cal_lite_tie_break"]
    cfg.decode.cal_lite_score_mode = decode["cal_lite_score_mode"]
    cfg.decode.cal_lite_length_alpha = float(decode["cal_lite_length_alpha"])

    cfg.logging.output_dir = config["logging"]["output_dir"]
    cfg.logging.experiment_name = config["logging"]["experiment_name"]
    return cfg


def build_run_args(config: Dict[str, Any]) -> SimpleNamespace:
    dream = config["dreamcoder_official_infilling"]
    cfg = build_cfg(config)
    return SimpleNamespace(
        model_path=cfg.model.model_path,
        split=cfg.data.split,
        dataset_subset=cfg.data.dataset_subset,
        max_samples=cfg.data.max_samples,
        seed=cfg.decode.seed,
        mask_length_source=cfg.decode.mask_length_source,
        fixed_mask_length=cfg.decode.fixed_mask_length,
        probe_lengths=cfg.decode.cal_lite_probe_lengths_csv,
        tie_break=cfg.decode.cal_lite_tie_break,
        score_mode=cfg.decode.cal_lite_score_mode,
        length_alpha=cfg.decode.cal_lite_length_alpha,
        dream_steps=int(dream["dream_steps"]),
        temperature=float(dream["temperature"]),
        top_p=dream.get("top_p"),
        top_k=dream.get("top_k"),
        alg=dream["alg"],
        alg_temp=dream.get("alg_temp"),
        eos_penalty=float(dream["eos_penalty"]),
        right_pad_new_tokens=int(dream.get("right_pad_new_tokens", 1)),
        no_force_right_pad_eos=not bool(dream.get("force_right_pad_eos", True)),
        no_bos=False,
        no_eos=False,
        torch_dtype=cfg.model.torch_dtype,
        device_map=cfg.model.device_map,
        output_dir=cfg.logging.output_dir,
        experiment_name=cfg.logging.experiment_name,
    )


def main() -> None:
    cli = parse_args()
    run_dir = Path(cli.run_dir)
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    config = load_json(config_path)
    cfg = build_cfg(config)
    args = build_run_args(config)
    set_global_seed(cfg.decode.seed)

    existing = load_results(results_path)
    completed = {row["task_id"] for row in existing}
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(cfg.model.torch_dtype),
        device_map=cfg.model.device_map,
    )
    model.eval()

    tasks = load_humaneval_infilling(
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        dataset_subset=cfg.data.dataset_subset,
    )
    pending = [task for task in tasks if task.task_id not in completed]
    print(f"Total tasks: {len(tasks)} | pending: {len(pending)}", flush=True)

    for idx, task in enumerate(pending, start=1):
        print(f"[resume {idx}/{len(pending)}] task_id={task.task_id} | start", flush=True)
        result = dream_run.run_task(task, tokenizer, model, cfg, args)
        append_result(results_path, result)
        m = result["metrics"]
        print(
            f"[resume {idx}/{len(pending)}] task_id={task.task_id} | "
            f"{'PASS' if m['passed'] else 'FAIL'} | "
            f"mask_len={m['mask_length']} | oracle_mask_len={m['oracle_mask_length']} | "
            f"diff={m['selected_minus_oracle_length']}",
            flush=True,
        )

    all_results = load_results(results_path)
    summary = dream_run.summarize(all_results, args)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Final rows: {len(all_results)}", flush=True)
    print(f"pass_rate: {summary.get('pass_rate')}", flush=True)
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
