#!/usr/bin/env python3
"""Review actual sampled HumanEval overlap candidates without modifying source data."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.markov_data_identity import (
    assert_keys,
    code_key,
    human_full_source,
    human_index,
    human_matches,
    response_code,
)


def load_arrow(path: Path) -> dict[int, dict[str, Any]]:
    rows = pa.ipc.open_stream(path).read_all().to_pylist()
    return {int(row["seq_id"]): row for row in rows}


def load_humaneval(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            task = "/".join(str(row["task_id"]).split("/")[:3])
            grouped.setdefault(task, []).append(row)
    return grouped


def review(
    candidate: dict[str, Any],
    source: dict[str, Any],
    humans: list[dict[str, Any]],
    index: dict[str, Any],
) -> dict[str, Any]:
    entry = str(candidate["entry_point"])
    source_code = str(source["code"])
    source_response = response_code(source)
    source_keys = {code_key(source_code, entry), code_key(source_response, entry)} - {None}
    source_assertions = set().union(*(assert_keys(test, entry) for test in source.get("testcase") or []))
    human_assertions = set()
    selected_human = humans[0]
    for human in humans:
        human_entry = str(human.get("entry_point") or "")
        human_key = code_key(human_full_source(human), human_entry)
        human_assertions |= assert_keys(str(human.get("test") or ""), human_entry)
        if human_key in source_keys:
            selected_human = human
    task_ids = sorted({match["task_id"] for match in candidate["matches"]})
    recomputed = [match for match in human_matches(source, index) if match["task_id"] in task_ids]
    expected_pairs = {(match["task_id"], match["reason"]) for match in candidate["matches"]}
    recomputed_pairs = {(match["task_id"], match["reason"]) for match in recomputed}
    if recomputed_pairs != expected_pairs:
        raise ValueError(
            f"Candidate artifact mismatch for {candidate['record_id']}: "
            f"expected={sorted(expected_pairs)} recomputed={sorted(recomputed_pairs)}"
        )
    ast_match = any("ast_no_doc" in match["reason"] for match in recomputed)
    shared_assertions = len(source_assertions & human_assertions)
    reasons = [str(match["reason"]) for match in candidate["matches"]]
    if ast_match:
        conclusion = "违反既定基准排除规则：实际训练记录与对应 HumanEval 题的去文档字符串代码 AST 相同。"
        status = "confirmed_training_overlap_requires_pause"
    else:
        conclusion = "仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。"
        status = "assert_only_candidate_not_confirmed_code_overlap"
    return {
        "record_id": candidate["record_id"],
        "source_seq_id": int(candidate["source_seq_id"]),
        "problem_group_id": candidate["problem_group_id"],
        "entry_point": entry,
        "human_task_ids": task_ids,
        "match_reasons": sorted(set(reasons)),
        "sampled_train_transitions": int(candidate["sampled_transitions"]["train"]),
        "source_and_humaneval_code_ast_match": ast_match,
        "shared_normalized_assert_count": shared_assertions,
        "review_status": status,
        "conclusion": conclusion,
        "source_question": str(source["instruction"]),
        "source_code": source_code,
        "source_tests": list(source.get("testcase") or []),
        "humaneval_prompt": str(selected_human.get("prompt") or ""),
        "humaneval_solution": str(selected_human.get("canonical_solution") or ""),
        "humaneval_tests": str(selected_human.get("test") or ""),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 实际训练成员与 HumanEval 候选复核",
        "",
        "此报告逐条记录实际进入 v1 full training bank 的候选。原题意、代码和测试只保留在同目录的服务器 JSON，不提交为普通 Git 原始样本。",
        "",
        "| HumanEval | entry point | 实际训练 transition | 匹配 | 结论 |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        task = ", ".join(row["human_task_ids"])
        evidence = "代码 AST 相同" if row["source_and_humaneval_code_ast_match"] else f"仅 {row['shared_normalized_assert_count']} 条规范化断言"
        lines.append(
            f"| {task} | `{row['entry_point']}` | {row['sampled_train_transitions']} | {evidence} | {row['conclusion']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir)
    audit = json.loads((result_dir / "actual_bank_membership_audit.json").read_text())
    candidates = [
        row for row in audit["humaneval_sampled_candidates"] if row["sampled_transitions"].get("train")
    ]
    source = load_arrow(Path(args.source_arrow))
    human = load_humaneval(Path(args.humaneval))
    human_rows = [row for rows in human.values() for row in rows]
    index = human_index(human_rows)
    rows = []
    for candidate in candidates:
        source_row = source[int(candidate["source_seq_id"])]
        task = str(candidate["matches"][0]["task_id"])
        rows.append(review(candidate, source_row, human[task], index))
    rows.sort(key=lambda row: (row["review_status"], row["human_task_ids"], row["record_id"]))
    output = {
        "status": "completed_manual_evidence_review",
        "actual_training_candidate_records": len(rows),
        "confirmed_code_ast_training_overlap_records": sum(
            row["source_and_humaneval_code_ast_match"] for row in rows
        ),
        "humaneval_variant_consistency": index["variant_consistency"],
        "full_humaneval_evaluation_permitted": False,
        "records": rows,
    }
    (result_dir / "actual_bank_membership_review.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(result_dir / "actual_bank_membership_review.md", rows)
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--source-arrow", required=True)
    parser.add_argument("--humaneval", required=True)
    run(parser.parse_args())
