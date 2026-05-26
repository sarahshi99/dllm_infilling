#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm.data_loader import load_humaneval_infilling
from expvision_dllm.snapshot_pipeline import (
    append_jsonl,
    ensure_dir,
    group_snapshot_records,
    rerun_tier3_for_snapshot,
    write_json,
)


# 新增脚本原因：
# 1. 当前准备把 per-snapshot Tier3 缓存当作正式分析输入；
# 2. 在这么做之前，需要先验证“同一份 snapshot 重复跑 Tier3 时是否稳定”；
# 3. 该脚本只做这件事，不参与 selector replay，也不生成新 snapshots。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Tier3 stability on cached snapshot dataset.")
    parser.add_argument("--dump-dir", type=str, required=True)
    parser.add_argument("--dataset-subset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--max-snapshots-per-task", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(__import__("json").loads(line))
    return rows


def main() -> None:
    args = parse_args()
    dump_dir = args.dump_dir
    snapshot_path = os.path.join(dump_dir, "snapshot_records.jsonl")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"snapshot_records.jsonl not found in {dump_dir}")

    if args.output_dir is None:
        output_dir = os.path.join(dump_dir, "tier3_stability_check")
    else:
        output_dir = args.output_dir
    ensure_dir(output_dir)

    snapshot_records = load_jsonl(snapshot_path)
    grouped = group_snapshot_records(snapshot_records)
    tasks = load_humaneval_infilling(split=args.split, max_samples=None, dataset_subset=args.dataset_subset)
    task_map = {task.task_id: task for task in tasks}

    random.seed(args.seed)
    all_task_ids = sorted(grouped.keys())
    selected_task_ids = all_task_ids[:]
    random.shuffle(selected_task_ids)
    selected_task_ids = selected_task_ids[: min(args.max_tasks, len(selected_task_ids))]

    detail_path = os.path.join(output_dir, "tier3_stability_details.jsonl")
    checked = 0
    mismatch_count = 0
    missing_cache_count = 0

    for task_id in selected_task_ids:
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        records = grouped[task_id]
        candidate_records = records[:]
        if args.max_snapshots_per_task is not None:
            candidate_records = candidate_records[: min(args.max_snapshots_per_task, len(candidate_records))]

        for record in candidate_records:
            cached_pass = record.get("tier3_pass")
            if cached_pass is None:
                missing_cache_count += 1

            rerun_results: List[Dict[str, Any]] = []
            rerun_passes: List[bool] = []
            for _ in range(max(1, args.repeats)):
                rerun = rerun_tier3_for_snapshot(task, record, timeout=args.timeout)
                rerun_results.append(rerun)
                rerun_passes.append(bool(rerun.get("passed", False)))

            stable_across_reruns = all(item == rerun_passes[0] for item in rerun_passes)
            matches_cache = (cached_pass is None) or all(item == bool(cached_pass) for item in rerun_passes)
            mismatch = (not stable_across_reruns) or (not matches_cache)

            if mismatch:
                mismatch_count += 1
            checked += 1

            append_jsonl(
                detail_path,
                {
                    "task_id": task_id,
                    "step": int(record["step"]),
                    "cached_tier3_pass": cached_pass,
                    "rerun_passes": rerun_passes,
                    "stable_across_reruns": stable_across_reruns,
                    "matches_cache": matches_cache,
                    "mismatch": mismatch,
                    "middle_text": record["middle_text"],
                },
            )

    summary = {
        "dump_dir": dump_dir,
        "dataset_subset": args.dataset_subset,
        "split": args.split,
        "seed": args.seed,
        "max_tasks": args.max_tasks,
        "max_snapshots_per_task": args.max_snapshots_per_task,
        "repeats": args.repeats,
        "timeout": args.timeout,
        "checked_snapshots": checked,
        "mismatch_count": mismatch_count,
        "missing_cache_count": missing_cache_count,
        "mismatch_rate": (mismatch_count / checked) if checked else None,
    }
    write_json(os.path.join(output_dir, "summary.json"), summary)

    print("===== Tier3 Stability Check =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()