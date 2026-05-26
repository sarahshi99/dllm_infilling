#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm.config import ExperimentConfig
from expvision_dllm.data_loader import load_humaneval_infilling
from expvision_dllm.decode_vanilla import _segment_decode, build_reconstruction_diagnostics, prepare_model_inputs
from expvision_dllm.model_utils import load_model_and_tokenizer, set_global_seed
from expvision_dllm.policies import linear_target_masks, select_low_confidence_mask_positions
from expvision_dllm.snapshot_pipeline import (
    BASELINE_CHAIN_VERSION,
    FEATURE_SCHEMA_VERSION,
    SAMPLING_POLICY_VERSION,
    VERIFIER_CACHE_VERSION,
    append_jsonl,
    build_snapshot_record,
    ensure_dir,
    make_run_id,
    resolve_sampling_steps,
    should_capture_step,
    write_json,
)
from expvision_dllm.verifier import run_verifier_stack


# 新增脚本原因：
# 1. 实验组织方式切换到 B 方案后，需要先统一导出整套 snapshot 数据集；
# 2. 旧版 run_s_series.py 是“每个 mode 重跑整条模型链”，不适合现在的离线 replay 方案；
# 3. 因此单独新增 dump 脚本，职责只做一件事：生成固定的 snapshot 数据集，并可选地缓存 Tier3。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump full SingleLine snapshot dataset for offline selector replay.")
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-mask-tokens", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="snapshot_dump")
    parser.add_argument("--sampling-policy-version", type=str, default=SAMPLING_POLICY_VERSION)
    parser.add_argument("--low-conf-threshold", type=float, default=0.70)
    parser.add_argument("--cache-tier3", action="store_true")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.decode.num_mask_tokens = args.num_mask_tokens
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed
    cfg.decode.max_samples = args.max_samples
    cfg.decode.dataset_subset = args.dataset_subset
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    set_global_seed(cfg.decode.seed)

    run_id = make_run_id(args.experiment_name)
    run_dir = os.path.join(args.output_dir, run_id)
    ensure_dir(run_dir)

    snapshot_path = os.path.join(run_dir, "snapshot_records.jsonl")
    summary_path = os.path.join(run_dir, "trajectory_summaries.jsonl")
    trace_path = os.path.join(run_dir, "step_traces.jsonl")

    config_payload = {
        "model_path": args.model_path,
        "split": args.split,
        "dataset_subset": args.dataset_subset,
        "max_samples": args.max_samples,
        "num_mask_tokens": args.num_mask_tokens,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "sampling_policy_version": args.sampling_policy_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "verifier_cache_version": VERIFIER_CACHE_VERSION,
        "baseline_chain_version": BASELINE_CHAIN_VERSION,
        "low_conf_threshold": args.low_conf_threshold,
        "cache_tier3": bool(args.cache_tier3),
    }
    write_json(os.path.join(run_dir, "config.json"), config_payload)

    print("⏳ 正在加载模型与 tokenizer ...")
    tokenizer, model = load_model_and_tokenizer(cfg.model)
    print("✅ 模型加载完成。")

    tasks = load_humaneval_infilling(
        split=args.split,
        max_samples=args.max_samples,
        dataset_subset=args.dataset_subset,
    )
    print(f"📦 已加载 {len(tasks)} 个样本。sampling_policy={args.sampling_policy_version} cache_tier3={args.cache_tier3}")

    sampling_steps = resolve_sampling_steps(args.total_steps, args.sampling_policy_version)
    sampling_step_set = set(sampling_steps)

    total_decode_sec = 0.0
    total_verifier_sec = 0.0
    total_snapshot_count = 0

    for idx, task in enumerate(tasks, start=1):
        print(f"\n[{idx}/{len(tasks)}] 导出任务: {task.task_id}")
        prepared = prepare_model_inputs(task, tokenizer, cfg)
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device

        x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
        mask_token_id = prepared["mask_token_id"]
        middle_start = prepared["middle_start"]
        middle_end = prepared["middle_end"]

        task_step_traces: List[Dict] = []
        task_snapshot_records: List[Dict] = []

        decode_start = time.perf_counter()
        for step in range(args.total_steps):
            current_mask_idx = (x_t == mask_token_id)
            with torch.no_grad():
                outputs = model(x_t)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)

            filled_tensor = x_t.clone()
            filled_tensor[current_mask_idx] = preds[current_mask_idx]

            target_masks = linear_target_masks(args.num_mask_tokens, args.total_steps, step)
            low_conf_indices = select_low_confidence_mask_positions(
                max_probs=max_probs[:, middle_start:middle_end],
                current_mask_idx=current_mask_idx[:, middle_start:middle_end],
                target_masks=target_masks,
            ) if target_masks > 0 else []

            next_tensor = filled_tensor.clone()
            if target_masks > 0 and low_conf_indices:
                next_tensor[0, [middle_start + pos for pos in low_conf_indices]] = mask_token_id
            x_t = next_tensor

            segments = _segment_decode(
                tokenizer=tokenizer,
                prefix_ids=prepared["prefix_ids"],
                middle_ids=filled_tensor[0, middle_start:middle_end].tolist(),
                suffix_ids=prepared["suffix_ids"],
            )

            detached_middle_conf = max_probs[0, middle_start:middle_end].detach().float().cpu()
            mean_conf = float(detached_middle_conf.mean().item()) if detached_middle_conf.numel() > 0 else 0.0
            min_conf = float(detached_middle_conf.min().item()) if detached_middle_conf.numel() > 0 else 0.0
            low_conf_count = int((detached_middle_conf < float(args.low_conf_threshold)).sum().item()) if detached_middle_conf.numel() > 0 else 0

            snapshot_payload = None
            if should_capture_step(step, sampling_step_set):
                verifier_start = time.perf_counter()
                verification = run_verifier_stack(
                    task=task,
                    full_code=segments["full_text"],
                    completion_without_suffix=segments["middle_text"],
                    timeout=cfg.verifier.timeout,
                )
                total_verifier_sec += time.perf_counter() - verifier_start

                tier1 = verification.get("tier1_parse_compile")
                tier2 = verification.get("tier2_smoke_exec")
                tier3 = verification.get("tier3_unit_tests") if args.cache_tier3 else None

                # 旧做法（保留说明，不再直接使用）：
                # snapshot 只缓存 Tier1/Tier2 proxy，不缓存 Tier3。
                #
                # 删除原因：
                # 1. 当前已切换到离线 replay 方案；
                # 2. 同一份 snapshot 的 Tier3 若已验证稳定，就应一次性缓存，避免每个 selector 反复重复跑。
                snapshot_record = build_snapshot_record(
                    run_id=run_id,
                    task=task,
                    dataset_subset=args.dataset_subset,
                    split=args.split,
                    dump_max_samples=args.max_samples,
                    seed=args.seed,
                    total_steps=args.total_steps,
                    step=step,
                    full_text=segments["full_text"],
                    middle_text=segments["middle_text"],
                    mean_confidence=mean_conf,
                    min_confidence=min_conf,
                    low_conf_count=low_conf_count,
                    tier1_result=tier1.to_dict() if tier1 is not None else {"passed": False},
                    tier2_result=tier2.to_dict() if tier2 is not None else {"passed": False},
                    tier3_result=tier3.to_dict() if tier3 is not None else None,
                )
                snapshot_payload = snapshot_record.to_dict()
                task_snapshot_records.append(snapshot_payload)
                append_jsonl(snapshot_path, snapshot_payload)
                total_snapshot_count += 1

            task_step_traces.append(
                {
                    "run_id": run_id,
                    "task_id": task.task_id,
                    "step": step,
                    "sampling_policy_version": args.sampling_policy_version,
                    "snapshot_captured": snapshot_payload,
                    "target_masks": target_masks,
                    "remaining_masks": int((x_t[0, middle_start:middle_end] == mask_token_id).sum().item()),
                    "low_confidence_indices": low_conf_indices,
                    "reconstruction_diagnostics": build_reconstruction_diagnostics(task, segments),
                }
            )

        decode_sec = time.perf_counter() - decode_start
        total_decode_sec += decode_sec

        first_tier1_step = next((int(item["step"]) for item in task_snapshot_records if bool(item.get("tier1_pass", False))), None)
        first_tier2_step = next((int(item["step"]) for item in task_snapshot_records if bool(item.get("tier2_pass", False))), None)

        if args.cache_tier3:
            first_tier3_step = next((int(item["step"]) for item in task_snapshot_records if bool(item.get("tier3_pass", False))), None)
            final_tier3_pass = bool(task_snapshot_records[-1].get("tier3_pass", False)) if task_snapshot_records else None
            backslide_happened = first_tier3_step is not None and not final_tier3_pass
            backslide_from_step = None
            if first_tier3_step is not None:
                for item in task_snapshot_records:
                    if int(item["step"]) > int(first_tier3_step) and not bool(item.get("tier3_pass", False)):
                        backslide_from_step = int(item["step"])
                        break
            oracle_first_pass_step = first_tier3_step
            tier3_snapshot_summary_available = True
        else:
            first_tier3_step = None
            final_tier3_pass = None
            backslide_happened = None
            backslide_from_step = None
            oracle_first_pass_step = None
            tier3_snapshot_summary_available = False

        trajectory_summary = {
            "run_id": run_id,
            "task_id": task.task_id,
            "dataset_subset": args.dataset_subset,
            "split": args.split,
            "dump_max_samples": args.max_samples,
            "seed": args.seed,
            "total_steps": args.total_steps,
            "sampling_policy_version": args.sampling_policy_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "verifier_cache_version": VERIFIER_CACHE_VERSION,
            "baseline_chain_version": BASELINE_CHAIN_VERSION,
            "first_tier1_step": first_tier1_step,
            "first_tier2_step": first_tier2_step,
            "first_tier3_step": first_tier3_step,
            "final_tier3_pass": final_tier3_pass,
            "backslide_happened": backslide_happened,
            "backslide_from_step": backslide_from_step,
            "oracle_first_pass_step": oracle_first_pass_step,
            "num_distinct_late_candidates": len({str(item["normalized_code_hash"]) for item in task_snapshot_records}),
            "tier3_snapshot_summary_available": tier3_snapshot_summary_available,
            "decode_sec": decode_sec,
        }
        append_jsonl(summary_path, trajectory_summary)

        for trace in task_step_traces:
            append_jsonl(trace_path, trace)

        print(f"   snapshots={len(task_snapshot_records)} | decode={decode_sec:.2f}s")

    manifest = {
        **config_payload,
        "run_id": run_id,
        "num_tasks": len(tasks),
        "sampling_steps": sampling_steps,
        "snapshot_count": total_snapshot_count,
        "total_decode_sec": total_decode_sec,
        "total_verifier_sec": total_verifier_sec,
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)

    print("\n===== Snapshot Dump Finished =====")
    print(f"run_dir: {run_dir}")
    print(f"num_tasks: {len(tasks)}")
    print(f"snapshot_count: {total_snapshot_count}")
    print(f"total_decode_sec: {total_decode_sec:.2f}")
    print(f"total_verifier_sec: {total_verifier_sec:.2f}")


if __name__ == "__main__":
    main()