#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm.config import ExperimentConfig
from expvision_dllm.data_loader import load_humaneval_infilling
from expvision_dllm.decode_gseries import run_g3_restart_decode, run_gseries_decode
from expvision_dllm.evaluate import summarize_results
from expvision_dllm.logger import JsonlLogger
from expvision_dllm.model_utils import load_model_and_tokenizer, set_global_seed


VALID_G_MODES = {"G0", "G1", "G2", "G2P", "G3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G-series token-only control experiments.")
    parser.add_argument("--mode", type=str, required=True, choices=sorted(VALID_G_MODES))
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-mask-tokens", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="g_series")
    parser.add_argument("--probe-start-ratio", type=float, default=0.75)
    parser.add_argument("--probe-stride", type=int, default=4)
    parser.add_argument("--g3-num-restarts", type=int, default=4)
    parser.add_argument("--g3-selection-rule", type=str, default="proxy", choices=["proxy", "tier3"])
    parser.add_argument("--g3-seed-stride", type=int, default=1)
    parser.add_argument("--g3-top-k", type=int, default=5)
    parser.add_argument("--g3-temperature", type=float, default=1.0)
    parser.add_argument("--g3-sample-start-ratio", type=float, default=0.0)
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.decode.num_mask_tokens = args.num_mask_tokens
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed
    cfg.decode.max_samples = args.max_samples
    cfg.decode.dataset_subset = args.dataset_subset
    cfg.decode.g_control_mode = args.mode
    cfg.decode.inference_tier3_probe_start_ratio = args.probe_start_ratio
    cfg.decode.inference_tier3_probe_stride = args.probe_stride
    cfg.decode.g3_num_restarts = args.g3_num_restarts
    cfg.decode.g3_selection_rule = args.g3_selection_rule
    cfg.decode.g3_seed_stride = args.g3_seed_stride
    cfg.decode.g3_top_k = args.g3_top_k
    cfg.decode.g3_temperature = args.g3_temperature
    cfg.decode.g3_sample_start_ratio = args.g3_sample_start_ratio
    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name

    # 旧逻辑不存在 G 系列模式控制。
    # 新增：按模式显式设置 probe / early-stop 行为，避免脚本侧隐式漂移。
    if args.mode == "G0":
        cfg.decode.enable_inference_tier3_probe = False
        cfg.decode.early_stop_on_tier3_pass = False
    elif args.mode == "G1":
        cfg.decode.enable_inference_tier3_probe = True
        cfg.decode.early_stop_on_tier3_pass = True
    elif args.mode == "G2":
        cfg.decode.enable_inference_tier3_probe = False
        cfg.decode.early_stop_on_tier3_pass = False
    elif args.mode == "G2P":
        cfg.decode.enable_inference_tier3_probe = True
        cfg.decode.early_stop_on_tier3_pass = False
    elif args.mode == "G3":
        cfg.decode.enable_inference_tier3_probe = False
        cfg.decode.early_stop_on_tier3_pass = False
    else:
        raise ValueError(f"Unsupported G mode: {args.mode}")

    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)

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
    print(f"📦 已加载 {len(tasks)} 个样本。mode={args.mode} subset={args.dataset_subset}")

    results = []
    for idx, task in enumerate(tasks, start=1):
        print(f"\n[{idx}/{len(tasks)}] 运行任务: {task.task_id}")
        if args.mode == "G3":
            result = run_g3_restart_decode(task, tokenizer, model, cfg)
        else:
            result = run_gseries_decode(task, tokenizer, model, cfg)
        result["dataset_subset"] = args.dataset_subset
        results.append(result)

        payload = {
            "task_id": result["task_id"],
            "dataset_subset": args.dataset_subset,
            "metrics": result["metrics"],
            "verification": result["verification"],
            "diagnostics": result.get("diagnostics"),
            "code": result["code"],
        }
        logger.log_result(payload)
        for trace in result["step_traces"]:
            logger.log_trace(trace)

        print(
            f"   PASS={result['metrics']['passed']} | "
            f"mode={result['metrics'].get('g_control_mode')} | "
            f"early_stop={result['metrics']['early_stop']} | "
            f"first_pass_step={result['metrics']['first_pass_step']} | "
            f"infer_v_calls={result['metrics'].get('inference_verifier_calls')} | "
            f"total={result['metrics']['total_sec']:.2f}s"
        )

    summary = summarize_results(results)
    logger.save_json("summary.json", summary)
    print("\n===== G-Series Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n结果目录: {logger.run_dir}")


if __name__ == "__main__":
    main()
