#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm.config import ExperimentConfig
from expvision_dllm.data_loader import load_humaneval_infilling
from expvision_dllm.decode_vanilla import run_vanilla_decode
from expvision_dllm.evaluate import summarize_results
from expvision_dllm.logger import JsonlLogger
from expvision_dllm.model_utils import load_model_and_tokenizer, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P0 vanilla LLaDA infilling baseline.")
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-mask-tokens", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="p0_vanilla")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.decode.num_mask_tokens = args.num_mask_tokens
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed
    cfg.decode.max_samples = args.max_samples
    cfg.decode.dataset_subset = args.dataset_subset
    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name

    set_global_seed(cfg.decode.seed)
    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)
    logger.save_config(cfg)

    print("⏳ 正在加载模型与 tokenizer ...")
    tokenizer, model = load_model_and_tokenizer(cfg.model)
    print("✅ 模型加载完成。")

    tasks = load_humaneval_infilling(
        split=args.split,
        max_samples=args.max_samples,
        dataset_subset=args.dataset_subset,
    )
    print(f"📦 已加载 {len(tasks)} 个样本。subset={args.dataset_subset}")

    results = []
    for idx, task in enumerate(tasks, start=1):
        print(f"\n[{idx}/{len(tasks)}] 运行任务: {task.task_id}")
        result = run_vanilla_decode(task, tokenizer, model, cfg)
        result["dataset_subset"] = args.dataset_subset
        results.append(result)

        payload = {
            "task_id": result["task_id"],
            "dataset_subset": args.dataset_subset,
            "metrics": result["metrics"],
            "verification": result["verification"],
            "code": result["code"],
        }
        logger.log_result(payload)
        for trace in result["step_traces"]:
            logger.log_trace(trace)

        print(
            f"   PASS={result['metrics']['passed']} | "
            f"decode={result['metrics']['decode_sec']:.2f}s | total={result['metrics']['total_sec']:.2f}s"
        )

    summary = summarize_results(results)
    logger.save_json("summary.json", summary)
    print("\n===== P0 Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n结果目录: {logger.run_dir}")


if __name__ == "__main__":
    main()
