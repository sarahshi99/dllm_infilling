#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm.snapshot_pipeline import (
    SELECTOR_FAMILY_VERSION,
    append_jsonl,
    compare_selector_outcomes,
    ensure_dir,
    group_snapshot_records,
    replay_selector,
    summarize_replay_results,
    write_json,
)


# 新增脚本原因：
# 1. 统一 dump 完 snapshot 之后，不应该再为每个 selector 重跑模型；
# 2. 当前脚本只负责离线 replay：从固定 snapshots 中应用 S0/S1/S2/S3/S4/S5；
# 3. 这样所有 selector 真正共享同一条轨迹，比较口径才最干净。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay offline selectors on cached snapshot dataset.")
    parser.add_argument("--dump-dir", type=str, required=True)
    parser.add_argument("--selectors", type=str, default="S0,S1,S2,S3,S4,S5")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--recent-window", type=int, default=3)
    parser.add_argument("--consistency-window", type=int, default=2)
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
    if args.output_dir is None:
        replay_dir = os.path.join(dump_dir, "offline_replay")
    else:
        replay_dir = args.output_dir
    ensure_dir(replay_dir)

    manifest_path = os.path.join(dump_dir, "manifest.json")
    snapshot_path = os.path.join(dump_dir, "snapshot_records.jsonl")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"snapshot_records.jsonl not found in {dump_dir}")

    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    snapshot_records = load_jsonl(snapshot_path)
    grouped = group_snapshot_records(snapshot_records)
    selectors = [item.strip().upper() for item in args.selectors.split(",") if item.strip()]

    all_selector_results: Dict[str, List[Dict[str, Any]]] = {}

    for selector_name in selectors:
        result_path = os.path.join(replay_dir, f"results_{selector_name}.jsonl")
        decision_path = os.path.join(replay_dir, f"selector_decisions_{selector_name}.jsonl")
        feature_path = os.path.join(replay_dir, f"snapshot_features_{selector_name}.jsonl")
        summary_path = os.path.join(replay_dir, f"trajectory_summaries_{selector_name}.jsonl")
        results_for_selector: List[Dict[str, Any]] = []

        print(f"\n===== Replaying {selector_name} =====")
        for idx, (task_id, records) in enumerate(sorted(grouped.items()), start=1):
            selected_snapshot, decision, feature_records, trajectory_summary = replay_selector(
                run_id=str(manifest["run_id"]),
                task_id=task_id,
                records=records,
                selector_name=selector_name,
                recent_window=args.recent_window,
                consistency_window=args.consistency_window,
            )
            payload = {
                "run_id": manifest["run_id"],
                "task_id": task_id,
                "selector_name": selector_name,
                "selector_version": SELECTOR_FAMILY_VERSION,
                "selected_step": int(selected_snapshot["step"]),
                "selected_snapshot": selected_snapshot,
                "passed": selected_snapshot.get("tier3_pass"),
                "tier3_cached_available": selected_snapshot.get("tier3_pass") is not None,
                "sampling_policy_version": manifest["sampling_policy_version"],
                "feature_schema_version": manifest["feature_schema_version"],
                "verifier_cache_version": manifest["verifier_cache_version"],
                "baseline_chain_version": manifest["baseline_chain_version"],
            }
            append_jsonl(result_path, payload)
            append_jsonl(decision_path, decision.to_dict())
            append_jsonl(summary_path, trajectory_summary.to_dict())
            for feature in feature_records:
                append_jsonl(feature_path, feature)
            results_for_selector.append(payload)

            if idx % 25 == 0 or idx == len(grouped):
                print(f"[{idx}/{len(grouped)}] {selector_name} replay done")

        selector_summary = summarize_replay_results(results_for_selector)
        selector_summary.update(
            {
                "selector_name": selector_name,
                "selector_version": SELECTOR_FAMILY_VERSION,
                "dump_run_id": manifest["run_id"],
                "sampling_policy_version": manifest["sampling_policy_version"],
                "feature_schema_version": manifest["feature_schema_version"],
                "verifier_cache_version": manifest["verifier_cache_version"],
                "baseline_chain_version": manifest["baseline_chain_version"],
                "recent_window": args.recent_window,
                "consistency_window": args.consistency_window,
            }
        )
        write_json(os.path.join(replay_dir, f"summary_{selector_name}.json"), selector_summary)
        all_selector_results[selector_name] = results_for_selector

    # 新增：默认同时输出相对 S0 的 win/loss/tie 比较，便于直接做方法线总结。
    if "S0" in all_selector_results:
        compare_payload: Dict[str, Any] = {"base_selector": "S0", "comparisons": {}}
        for selector_name, selector_results in all_selector_results.items():
            if selector_name == "S0":
                continue
            compare_payload["comparisons"][selector_name] = compare_selector_outcomes(
                all_selector_results["S0"],
                selector_results,
            )
        write_json(os.path.join(replay_dir, "compare_against_S0.json"), compare_payload)

    write_json(
        os.path.join(replay_dir, "replay_manifest.json"),
        {
            "dump_dir": dump_dir,
            "dump_run_id": manifest["run_id"],
            "selectors": selectors,
            "recent_window": args.recent_window,
            "consistency_window": args.consistency_window,
            "selector_version": SELECTOR_FAMILY_VERSION,
        },
    )
    print(f"\n离线 replay 完成。输出目录: {replay_dir}")


if __name__ == "__main__":
    main()