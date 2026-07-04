#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl, metric, oracle_bucket
from experiments.action_ceiling.action_ceiling_matrix import current_branch, current_commit, git_capture, parse_task_id_group, write_csv


JsonDict = Dict[str, Any]

PRIMARY_RESULTS = "/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl"
ARXIV_URL = "https://arxiv.org/abs/2602.07546"
PAPER_TITLE = "Improving Variable-Length Generation in Diffusion Language Models via Length Regularization"


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv_rows(path: Path) -> List[JsonDict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def make_output_dir(base_dir: str, timestamp: str | None) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"lrdllm_same_protocol_sanity_{stamp}"
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def select_cases(primary_rows: Sequence[Mapping[str, Any]], split_assignment: Mapping[str, str]) -> List[JsonDict]:
    candidates: List[JsonDict] = []
    for row in primary_rows:
        task_id = str(row["task_id"])
        split = split_assignment.get(task_id)
        if split not in {"train", "calibration", "validation"}:
            continue
        oracle_len = _to_int(metric(row, "oracle_mask_length"))
        passed = _to_bool(metric(row, "passed", False))
        if oracle_len <= 8:
            length_class = "short"
        elif oracle_len <= 16:
            length_class = "medium"
        else:
            length_class = "true_long"
        candidates.append(
            {
                "task_id": task_id,
                "task_group": parse_task_id_group(task_id),
                "split": split,
                "oracle_length": oracle_len,
                "oracle_bucket": oracle_bucket(oracle_len),
                "length_class": length_class,
                "primary_selected_length": _to_int(metric(row, "selected_mask_length", metric(row, "mask_length"))),
                "primary_passed": passed,
                "primary_error_type": None if passed else "primary_failed",
            }
        )
    selected: List[JsonDict] = []
    used = set()
    specs = [
        ("short_control_pass", "short", True, 2),
        ("short_control_fail", "short", False, 1),
        ("medium_control_pass", "medium", True, 2),
        ("medium_control_fail", "medium", False, 1),
        ("true_long_control_pass", "true_long", True, 1),
        ("true_long_control_fail", "true_long", False, 3),
    ]
    for label, length_class, passed, count in specs:
        matches = [
            row
            for row in candidates
            if row["task_id"] not in used and row["length_class"] == length_class and row["primary_passed"] is passed
        ]
        matches.sort(key=lambda row: (row["split"], row["task_id"]))
        for row in matches[:count]:
            row = dict(row)
            row["sanity_stratum"] = label
            selected.append(row)
            used.add(row["task_id"])
    if len(selected) < 10:
        for row in sorted(candidates, key=lambda item: (item["length_class"], item["primary_passed"], item["task_id"])):
            if row["task_id"] in used:
                continue
            row = dict(row)
            row["sanity_stratum"] = "fallback_fill"
            selected.append(row)
            used.add(row["task_id"])
            if len(selected) >= 10:
                break
    return selected[:10]


def protocol_comparison_rows() -> List[JsonDict]:
    return [
        {
            "criterion": "official_source",
            "local_status": "blocked",
            "evidence": "The repository contains literature anchors and the arXiv paper metadata, but no official LR-DLLM code or adapter.",
            "risk": "Cannot claim official reproduction.",
        },
        {
            "criterion": "checkpoint_backbone_compatibility",
            "local_status": "unknown",
            "evidence": "Paper reports several backbones including LLaDA-family anchors; no local Stage I/II implementation is available for GSAI-ML/LLaDA-8B-Base.",
            "risk": "A local approximation would mix method and implementation differences.",
        },
        {
            "criterion": "stage_i_length_regularization",
            "local_status": "not_implemented",
            "evidence": "No code path for LR-DLLM semantic compatibility / length-induced uncertainty correction was found in this repository.",
            "risk": "Cannot run Stage I same-protocol sanity.",
        },
        {
            "criterion": "stage_ii_dynamic_span_adjustment",
            "local_status": "not_implemented",
            "evidence": "No code path for LR-DLLM expansion/contraction policy or Stage II adapter was found.",
            "risk": "Cannot run Stage I+II full comparison.",
        },
        {
            "criterion": "prompt_dataset_evaluator",
            "local_status": "available_for_future_adapter",
            "evidence": "The local HumanEval-SingleLineInfilling dataset, prompt, checkpoint, and verifier are available through existing runners.",
            "risk": "Adapter is feasible only after official algorithm/code is available.",
        },
        {
            "criterion": "oracle_use",
            "local_status": "not_run",
            "evidence": "No LR-DLLM generation was executed; the sanity manifest intentionally contains no pass/fail output for LR-DLLM.",
            "risk": "Avoids misleading protocol-mismatched numbers.",
        },
    ]


def render_audit() -> str:
    return "\n".join(
        [
            "# LR-DLLM Protocol Audit",
            "",
            f"- paper: {PAPER_TITLE}",
            f"- arXiv: `{ARXIV_URL}`",
            "- checked locally: repository search for LR-DLLM / Stage I / Stage II implementation and existing literature notes.",
            "- result: no executable official LR-DLLM implementation is present in this repository.",
            "",
            "## Audit Verdict",
            "",
            "`protocol_mismatch_blocked`",
            "",
            "原因：当前仓库有 LR-DLLM 的 literature anchor 和跨 backbone 结果记录，但没有 official source、Stage I length-regularization 实现、Stage II dynamic span adapter，也没有可确认与本地 prompt/evaluator/checkpoint 对齐的执行入口。因此本轮不能运行 full same-protocol LR-DLLM，也不能把任何本地 heuristic 称为 LR-DLLM。",
            "",
            "## Future Minimal Adapter",
            "",
            "1. 定位官方代码或作者发布的伪代码/配置。",
            "2. 将 Stage I 的 length-regularized score 接到本地 `HumanEval-SingleLineInfilling` prompt 和 LLaDA-8B checkpoint。",
            "3. 在同一 verifier、seed、dataset subset 下跑本 sanity manifest 的 10 cases。",
            "4. 只有 sanity 判定为 `protocol_matched_lrdllm` 或明确的 `local_stage1_adaptation` 后，才考虑 1033-case full run。",
            "",
        ]
    )


def render_protocol_comparison(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# LR-DLLM Same-Protocol Sanity",
        "",
        "verdict: `protocol_mismatch_blocked`",
        "",
        "This is a protocol compatibility sanity, not a generation run. It intentionally does not report LR-DLLM pass/fail because no official or local Stage I/II implementation is available.",
        "",
        "| Criterion | Local Status | Evidence | Risk |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{}` | `{}` | {} | {} |".format(
                row["criterion"],
                row["local_status"],
                row["evidence"],
                row["risk"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> None:
    output_dir = make_output_dir(args.output_dir, args.timestamp)
    split_assignment = {row["task_id"]: row["split"] for row in read_csv_rows(Path(args.split_dir) / "row_split_assignment.csv")}
    primary_rows = load_jsonl(args.primary_results)
    cases = select_cases(primary_rows, split_assignment)
    comparison = protocol_comparison_rows()
    results = [
        {
            **case,
            "lrdllm_stage": "not_run",
            "lrdllm_passed": "",
            "lrdllm_output_hash": "",
            "blocked_reason": "official/local LR-DLLM Stage I/II implementation unavailable",
        }
        for case in cases
    ]
    verdict = "protocol_mismatch_blocked"
    summary = {
        "mode": "lrdllm_same_protocol_sanity",
        "verdict": verdict,
        "case_count": len(cases),
        "case_strata": dict(Counter(row["sanity_stratum"] for row in cases)),
        "generation_executed": False,
        "full_run_status": "not_run_protocol_mismatch_blocked",
        "official_paper": {"title": PAPER_TITLE, "arxiv": ARXIV_URL},
        "branch": current_branch(),
        "commit": current_commit(),
    }
    write_csv(output_dir / "case_manifest.csv", cases)
    write_csv(output_dir / "protocol_comparison.csv", comparison)
    write_csv(output_dir / "results.csv", results)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "protocol_comparison.md").write_text(render_protocol_comparison(comparison), encoding="utf-8")
    (output_dir / "verdict.md").write_text(f"# LR-DLLM Verdict\n\n`{verdict}`\n\nFull run is blocked; no misleading local LR-DLLM number was produced.\n", encoding="utf-8")
    run_manifest = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "execution_status": "completed_protocol_blocked",
        "output_dir": str(output_dir),
        "verdict": verdict,
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    audit_path = Path(args.audit_doc)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(render_audit(), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "verdict": verdict}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LR-DLLM protocol audit and blocked 10-case sanity manifest.")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-dir", default="analysis_outputs")
    parser.add_argument("--split-dir", default="analysis_outputs/grouped_split_20260702_accel2")
    parser.add_argument("--primary-results", default=PRIMARY_RESULTS)
    parser.add_argument("--audit-doc", default="docs/paper_agent/lrdllm_protocol_audit.zh.md")
    return parser.parse_args()


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
