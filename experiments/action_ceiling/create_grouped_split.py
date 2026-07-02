#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.action_ceiling.action_ceiling_matrix import current_branch, current_commit, parse_task_id_group, write_csv
from expvision_dllm_clean.dataset import load_humaneval_infilling


JsonDict = Dict[str, Any]


def stable_score(group: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def assign_splits(groups: Sequence[str], seed: int) -> Dict[str, str]:
    ordered = sorted(groups, key=lambda group: (stable_score(group, seed), group))
    n = len(ordered)
    train_end = round(n * 0.60)
    calibration_end = train_end + round(n * 0.15)
    validation_end = calibration_end + round(n * 0.15)
    assignment: Dict[str, str] = {}
    for idx, group in enumerate(ordered):
        if idx < train_end:
            split = "train"
        elif idx < calibration_end:
            split = "calibration"
        elif idx < validation_end:
            split = "validation"
        else:
            split = "test"
        assignment[group] = split
    return assignment


def verify_no_overlap(split_tasks: Mapping[str, Sequence[str]]) -> JsonDict:
    seen: Dict[str, str] = {}
    overlaps: List[JsonDict] = []
    for split, groups in split_tasks.items():
        for group in groups:
            if group in seen:
                overlaps.append({"task_group": group, "first_split": seen[group], "second_split": split})
            seen[group] = split
    return {"passed": not overlaps, "overlaps": overlaps}


def render_protocol(split_dir: Path, manifest: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Grouped Split Protocol",
            "",
            "更新时间：2026-07-02 CST",
            "",
            "## 目的",
            "",
            "后续 risk-controlled controller 的拟合、阈值选择、校准和最终评测必须按原始 `HumanEval/<id>` 分组，避免同一原始题目的不同 infill location 跨 split 泄漏。",
            "",
            "## 规则",
            "",
            "- 分组单位：`HumanEval/<id>`。",
            "- 固定 seed：`{}`。".format(manifest.get("seed")),
            "- split：train / calibration / validation / test。",
            "- 同一 group 的所有 row 必须只出现在一个 split。",
            "- 当前 oracle action-ceiling / generation-ceiling 结果只能作为 diagnostic ceiling，不能称为 held-out method result。",
            "",
            "## 输出",
            "",
            f"- split manifest: `{split_dir / 'split_manifest.json'}`",
            f"- train tasks: `{split_dir / 'train_tasks.json'}`",
            f"- calibration tasks: `{split_dir / 'calibration_tasks.json'}`",
            f"- validation tasks: `{split_dir / 'validation_tasks.json'}`",
            f"- test tasks: `{split_dir / 'test_tasks.json'}`",
            f"- row assignment: `{split_dir / 'row_split_assignment.csv'}`",
            "",
            "## 使用约束",
            "",
            "- controller feature/threshold/rule selection 只能使用 train。",
            "- risk calibration 使用 calibration。",
            "- model selection / ablation selection 使用 validation。",
            "- 最终论文数字只在 test 上报告，并且不得用本轮 oracle canvas results 调整 test 规则。",
            "",
        ]
    )


def execute(args: argparse.Namespace) -> None:
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"grouped_split_{stamp}"
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    tasks = load_humaneval_infilling(split=args.split, dataset_subset=args.dataset_subset)
    by_group: Dict[str, List[str]] = defaultdict(list)
    for task in tasks:
        by_group[parse_task_id_group(task.task_id)].append(task.task_id)
    assignment = assign_splits(sorted(by_group), args.seed)
    row_assignment = [
        {
            "task_id": task.task_id,
            "task_group": parse_task_id_group(task.task_id),
            "split": assignment[parse_task_id_group(task.task_id)],
        }
        for task in tasks
    ]
    split_tasks = {
        split: sorted(group for group, assigned in assignment.items() if assigned == split)
        for split in ["train", "calibration", "validation", "test"]
    }
    split_row_counts = Counter(row["split"] for row in row_assignment)
    verification = verify_no_overlap(split_tasks)
    manifest: JsonDict = {
        "updated_at_cst": "2026-07-02",
        "branch": current_branch(),
        "commit": current_commit(),
        "seed": args.seed,
        "dataset_subset": args.dataset_subset,
        "source_split": args.split,
        "group_unit": "HumanEval/<id>",
        "task_group_count": len(by_group),
        "row_count": len(tasks),
        "split_group_counts": {split: len(groups) for split, groups in split_tasks.items()},
        "split_row_counts": dict(sorted(split_row_counts.items())),
        "verification": verification,
        "non_goal": "Current oracle action-ceiling results are diagnostic and must not be reported as held-out method results.",
    }
    for split, groups in split_tasks.items():
        (output_dir / f"{split}_tasks.json").write_text(
            json.dumps(groups, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    write_csv(output_dir / "row_split_assignment.csv", row_assignment)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    docs_path = Path(args.protocol_doc)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(render_protocol(output_dir, manifest), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "verification": verification}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a HumanEval task-grouped split manifest.")
    parser.add_argument("--output-dir", default="analysis_outputs")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-subset", default="HumanEval-SingleLineInfilling")
    parser.add_argument("--protocol-doc", default="docs/paper_agent/grouped_split_protocol.zh.md")
    return parser.parse_args()


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
