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
from expvision_dllm.decode_s_series import run_s_series_decode
from expvision_dllm.evaluate import summarize_results
from expvision_dllm.logger import JsonlLogger
from expvision_dllm.model_utils import load_model_and_tokenizer, set_global_seed

# =========================
# 旧实现（保留对照，不再直接使用）
# =========================
# VALID_S_MODES = {"S0", "S1", "S2"}
#
# 删除原因：
# 当前需要在原有 S0/S1/S2 之外扩展 S3/S4/S5，并统一记录 selector 决策日志。
VALID_S_MODES = {"S0", "S1", "S2", "S3", "S4", "S5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S-series single-track snapshot selection experiments.")
    parser.add_argument("--mode", type=str, required=True, choices=sorted(VALID_S_MODES))
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-mask-tokens", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="s_series")

    # =========================
    # 旧实现（保留对照，不再直接使用）
    # =========================
    # parser.add_argument("--snapshot-start-ratio", type=float, default=0.75)
    # parser.add_argument("--snapshot-stride", type=int, default=4)
    #
    # 删除原因：
    # 当前采样窗口已经冻结成显式 sampling policy，不再直接从 ratio/stride 推导。
    parser.add_argument("--sampling-policy-version", type=str, default="late_uniform_5points_v1")
    parser.add_argument("--consistency-window", type=int, default=2)
    parser.add_argument("--recent-window", type=int, default=3)
    parser.add_argument("--low-conf-threshold", type=float, default=0.70)
    parser.add_argument("--feature-schema-version", type=str, default="snapshot_schema_v2")
    parser.add_argument("--selector-version", type=str, default="selector_v2")
    parser.add_argument("--verifier-version", type=str, default="verifier_v1")
    parser.add_argument("--baseline-chain-version", type=str, default="s_chain_v2")
    parser.add_argument("--capture-snapshot-tier3-summary", action="store_true")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.decode.num_mask_tokens = args.num_mask_tokens
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed
    cfg.decode.max_samples = args.max_samples
    cfg.decode.dataset_subset = args.dataset_subset
    cfg.decode.dataset_split = args.split
    cfg.decode.s_control_mode = args.mode
    cfg.decode.s_sampling_policy_version = args.sampling_policy_version
    cfg.decode.s_consistency_window = args.consistency_window
    cfg.decode.s_recent_window = args.recent_window
    cfg.decode.s_low_conf_threshold = args.low_conf_threshold
    cfg.decode.s_feature_schema_version = args.feature_schema_version
    cfg.decode.s_selector_version = args.selector_version
    cfg.decode.verifier_version = args.verifier_version
    cfg.decode.baseline_chain_version = args.baseline_chain_version
    cfg.decode.s_capture_snapshot_tier3_summary = bool(args.capture_snapshot_tier3_summary)
    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name
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
    print(
        f"📦 已加载 {len(tasks)} 个样本。mode={args.mode} subset={args.dataset_subset} "
        f"sampling_policy={args.sampling_policy_version} recent_window={args.recent_window}"
    )

    results = []
    for idx, task in enumerate(tasks, start=1):
        print(f"\n[{idx}/{len(tasks)}] 运行任务: {task.task_id}")
        result = run_s_series_decode(task, tokenizer, model, cfg, run_id=logger.run_id)
        result["dataset_subset"] = args.dataset_subset
        results.append(result)

        payload = {
            "task_id": result["task_id"],
            "run_metadata": result["run_metadata"],
            "dataset_subset": args.dataset_subset,
            "metrics": result["metrics"],
            "verification": result["verification"],
            "diagnostics": result.get("diagnostics"),
            "code": result["code"],
        }
        logger.log_result(payload)
        logger.log_snapshot_records(result.get("snapshot_records", []))
        logger.log_selector_decision(result.get("selector_decision", {}))
        logger.log_trajectory_summary(result.get("trajectory_summary", {}))
        for trace in result["step_traces"]:
            logger.log_trace(trace)

        print(
            f"   PASS={result['metrics']['passed']} | "
            f"mode={result['metrics'].get('s_control_mode')} | "
            f"selected_step={result['metrics'].get('selected_snapshot_step')} | "
            f"snapshots={result['metrics'].get('snapshot_count')} | "
            f"total={result['metrics']['total_sec']:.2f}s"
        )

    summary = summarize_results(results)
    logger.save_json("summary.json", summary)
    print("\n===== S-Series Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n结果目录: {logger.run_dir}")


if __name__ == "__main__":
    main()