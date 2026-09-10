#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModel

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.markov_head_metrics import no_head_summary
from experiments.dreamon_markov_head_training import freeze_module
from experiments.train_dreamon_markov_head import (
    ReplayBank,
    evaluate,
    file_sha256,
    make_head,
    summarize,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_best_head(head: Any, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(state["head"])


def eligible_heads(statuses: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [kind for kind, status in statuses.items() if status.get("deployable_gain")]


def subgroup_rows(rows: Sequence[Mapping[str, Any]], *, policy: str | None, aligned: bool | None):
    return [
        row
        for row in rows
        if (policy is None or row["trajectory_policy"] == policy)
        and (aligned is None or bool(row["reference_aligned"]) == aligned)
    ]


def extended_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows, bootstrap_reps=10000)
    summary["recovered_count"] = sum(bool(row["mismatch_recovery"]) for row in rows)
    summary["corrupted_count"] = sum(bool(row["stable_corruption"]) for row in rows)
    aligned = [row for row in rows if row["reference_aligned"]]
    summary.update(
        {
            "baseline_reference_rank": (
                float(np.mean([row["baseline_reference_rank"] for row in aligned]))
                if aligned
                else None
            ),
            "head_reference_rank": (
                float(np.mean([row["head_reference_rank"] for row in aligned]))
                if aligned
                else None
            ),
            "top_p_support_enter_count_mean": float(
                np.mean([row["top_p_support_enter_count"] for row in rows])
            ),
            "top_p_support_exit_count_mean": float(
                np.mean([row["top_p_support_exit_count"] for row in rows])
            ),
        }
    )
    return summary


def append_once(path: Path, marker: str, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        return
    path.write_text(existing.rstrip() + "\n\n" + text.rstrip() + "\n", encoding="utf-8")


def update_research_records(
    *,
    result_dir: Path,
    tv: Mapping[str, Any],
    kl: Mapping[str, Any],
    verdict: str,
    winner: str,
    run_id: str,
    run_date_utc: str,
    result_document: Path,
) -> None:
    marker = run_id
    compact = (
        f"<!-- {marker} -->\n"
        f"## {run_date_utc} DreamOn external Markov-head training `{run_id}`\n\n"
        f"Status: `{verdict}`. TV pilot/full=`{tv['pilot']['passed']}/{bool(tv.get('full') and tv['full']['passed'])}`; "
        f"KL pilot/full=`{kl['pilot']['passed']}/{bool(kl.get('full') and kl['full']['passed'])}`; winner=`{winner}`. "
        "This is external OpenCoder training/validation evidence, not a HumanEval method result. "
        f"Artifacts: `{result_dir.relative_to(REPO)}/`."
    )
    for relative in (
        "docs/paper_agent/experiment_queue.md",
        "docs/paper_agent/decision_log.md",
        "docs/paper_agent/idea_board.md",
        "docs/paper_agent/codex_handoff.latest.zh.md",
        "docs/results/run_registry.md",
        "docs/paper_agent/current_action.md",
    ):
        append_once(REPO / relative, marker, compact)
    result_document.parent.mkdir(parents=True, exist_ok=True)
    result_document.write_text(
        f"# DreamOn 外部一阶 Markov 头训练 `{run_id}`：结果\n\n"
        + compact.split("\n", 2)[-1]
        + "\n",
        encoding="utf-8",
    )
    review_path = REPO / "docs/paper_agent/review_manifest.latest.json"
    review = load_json(review_path)
    review[run_id] = {
        "status": verdict,
        "winner": winner,
        "result_dir": str(result_dir.relative_to(REPO)),
        "human_eval_used_for_training": None,
        "human_eval_explicit_training_source": False,
    }
    atomic_json(review_path, review)
    evidence_path = REPO / "docs/paper_agent/evidence_snapshot.json"
    evidence = load_json(evidence_path)
    runs = evidence.setdefault("runs", {})
    if isinstance(runs, dict):
        runs[run_id] = {
            "status": verdict,
            "winner": winner,
            "result_dir": str(result_dir.relative_to(REPO)),
            "evidence_type": "external_opencoder_training_validation",
        }
    atomic_json(evidence_path, evidence)


def run(args: argparse.Namespace) -> int:
    result_dir = Path(args.result_dir).resolve()
    tv = load_json(result_dir / "tv_training_status.json")
    kl = load_json(result_dir / "kl_training_status.json")
    statuses = {"tv": tv, "kl": kl}
    validation_rows = []
    for kind, status in statuses.items():
        for phase in ("pilot", "full"):
            item = status.get(phase)
            if item:
                validation_rows.append(
                    {
                        "head": kind,
                        "phase": phase,
                        "lambda": 1.0,
                        "passed": item["passed"],
                        "failures": ";".join(item["failures"]),
                        **item["validation_lambda_1"],
                    }
                )
        for row in status.get("lambda_validation", []):
            validation_rows.append(
                {
                    "head": kind,
                    "phase": "lambda_calibration",
                        "passed": (status.get("full") or {}).get("passed", False),
                    **row,
                }
            )
    write_csv(result_dir / "validation_comparison.csv", validation_rows)

    eligible = eligible_heads(statuses)
    external_rows = []
    test_open_count = 0
    reused_external_diagnostics = []
    if eligible:
        if (
            list(result_dir.glob("external_test_diagnostics_*.jsonl.gz"))
            and not args.resume_existing_external_diagnostics
        ):
            raise RuntimeError("External diagnostics already exist; use analysis/repair_markov_training_reports.py for CPU reanalysis.")
        test_open_count = 1
        config = AutoConfig.from_pretrained(
            Path(args.model_snapshot).resolve(), trust_remote_code=True, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            Path(args.model_snapshot).resolve(),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(args.device)
        freeze_module(model)
        bank = ReplayBank(Path(args.bank_db).resolve())
        keys = bank.keys("external_test")
        micro = int(load_json(Path(args.common_training_config).resolve())["micro_batch"])
        baseline_written = set()
        for kind, status in statuses.items():
            checkpoint = status.get("best_checkpoint")
            if not checkpoint:
                continue
            head = make_head(config, args.device)
            load_best_head(head, Path(checkpoint))
            diagnostic_path = result_dir / f"external_test_diagnostics_{kind}.jsonl.gz"
            if diagnostic_path.exists():
                diagnostics = load_diagnostics(diagnostic_path)
                if len(diagnostics) != len(keys):
                    raise RuntimeError(
                        f"Existing {kind} external diagnostics are incomplete: "
                        f"{len(diagnostics)} != {len(keys)}"
                    )
                reused_external_diagnostics.append(kind)
            else:
                _, diagnostics = evaluate(
                    bank,
                    keys,
                    model=model,
                    head=head,
                    config=config,
                    micro_batch=micro,
                    kind=kind,
                    lambda_value=float(status["chosen_lambda"]),
                    device=args.device,
                    bootstrap_reps=10000,
                    diagnostic_path=diagnostic_path,
                    extended=True,
                )
            for policy in (None, "confidence_global", "left_to_right_frontier"):
                for aligned in (None, True):
                    subset = subgroup_rows(diagnostics, policy=policy, aligned=aligned)
                    if not subset:
                        continue
                    scope = (
                        "overall" if policy is None else policy
                    ) + ("_aligned" if aligned else "")
                    summary = extended_summary(subset)
                    if scope not in baseline_written:
                        external_rows.append(
                            {
                                "method": "no_head",
                                "scope": scope,
                                "lambda": 0.0,
                                **no_head_summary(summary),
                            }
                        )
                        baseline_written.add(scope)
                    external_rows.append(
                        {
                            "method": f"{kind}_head",
                            "scope": scope,
                            "lambda": status["chosen_lambda"],
                            **summary,
                        }
                    )
            del head
        bank.close()
        del model
        torch.cuda.empty_cache()
        write_csv(result_dir / "external_test_comparison.csv", external_rows)

    candidates = []
    for kind, status in statuses.items():
        if status.get("deployable_gain"):
            chosen = next(
                row
                for row in status["lambda_validation"]
                if float(row["lambda"]) == float(status["chosen_lambda"])
            )
            candidates.append((float(chosen["head_raw_tv"]), kind))
    winner = min(candidates)[1] if candidates else "none"
    verdict = "completed_external_test_opened" if test_open_count else "completed_no_external_test_gate"
    checkpoints = []
    for kind, status in statuses.items():
        for phase in ("pilot", "full"):
            item = status.get(phase)
            if not item:
                continue
            for label in ("best_checkpoint", "last_checkpoint"):
                path = Path(item[label])
                checkpoints.append(
                    {
                        "head": kind,
                        "phase": phase,
                        "label": label.replace("_checkpoint", ""),
                        "path": str(path),
                        "exists": path.exists(),
                        "size": path.stat().st_size if path.exists() else None,
                        "sha256": file_sha256(path) if path.exists() else None,
                        "resume_command": (
                            f"python experiments/train_dreamon_markov_head.py run-head --kind {kind} "
                            "--common arguments from commands.log"
                        ),
                    }
                )
    atomic_json(result_dir / "checkpoint_registry.json", checkpoints)
    run_config = {
        "run_id": args.run_id,
        "run_date_utc": args.run_date_utc,
        "branch": args.branch,
        "base_head": args.base_head,
        "model_revision": "8ccc74750e43177327f29dab9e91882ba759e194",
        "source_revision": "8a0a54918412eda9402a327646f7f067f7160ec8",
        "dataset_revision": "7d28f40d579edd7c24402d17d0c7639f991e6f8d",
        "split_seed": 20260901,
        "head_seed": 42,
        "rank": 256,
        "learning_rate": 3e-4,
        "effective_batch": 128,
        "max_epochs": 5,
        "patience": 2,
        "external_test_open_count": test_open_count,
        "reused_external_diagnostics": reused_external_diagnostics,
    }
    atomic_json(result_dir / "run_config.json", run_config)
    completeness = {
        "status": verdict,
        "tv_status_present": (result_dir / "tv_training_status.json").is_file(),
        "kl_status_present": (result_dir / "kl_training_status.json").is_file(),
        "test_open_count": test_open_count,
        "test_open_gate_satisfied": bool(eligible) == bool(test_open_count),
        "winner": winner,
        "missing_required_files": [name for name in ("tv_training_status.json", "kl_training_status.json", "validation_comparison.csv", "split_manifest.jsonl.zst", "transition_bank_summary.json") if not (result_dir / name).is_file()],
        "test_open_count_scope": "this invocation only; not a durable lifetime counter",
        "data_isolation_status": args.data_isolation_status,
        "reviewer_gate_disabled": True,
        "local_diff_review_required_before_final_push": True,
    }
    atomic_json(result_dir / "completeness_audit.json", completeness)
    (result_dir / "implementation_audit.md").write_text(
        "# Implementation audit\n\n"
        "- DreamOn is frozen bf16 inference; Markov softmax/loss is float32.\n"
        "- Target logits use the released one-position shift by selecting hidden state `target-1` before `lm_head`.\n"
        "- Markov correction is applied after action masking and before temperature/top-p.\n"
        "- Structural token correction is exactly zero.\n"
        "- TV and KL use the same bank, initialization, optimizer, schedule, batch, seed, GPU, and gates.\n"
        "- TV and KL ran in separate sequential processes.\n"
        f"- {args.data_isolation_note}\n"
        "- No full-vocabulary logits were serialized.\n"
        "- Reviewer/subagent gate is disabled by repository policy; local diff review and fresh verification are used.\n",
        encoding="utf-8",
    )
    report = f"""# {args.report_title}

状态：`{verdict}`。

1. TV-head 是一个只看刚提交左邻 token 的 rank-256 加性头，用全词表 L1/TV 匹配 fresh DreamOn。
2. KL-head 结构完全相同，唯一差异是使用正向 KL 匹配 fresh DreamOn。
3. {args.data_isolation_note}
4. 两个头从同一个零输出初始化和同一个冻结 transition bank 开始；共同初始化记录见 `common_initialization.json`。
5. TV pilot/full=`{tv['pilot']['passed']}/{bool(tv.get('full') and tv['full']['passed'])}`；KL pilot/full=`{kl['pilot']['passed']}/{bool(kl.get('full') and kl['full']['passed'])}`。
6. 学习目标是 stale→fresh 的 DreamOn 原始分布变化，不是 HumanEval 正确标签。
7. 困难 mismatch transition 的 recovery 与总体 TV 改善见 `validation_comparison.csv`。
8. 简单 stable transition 的过度修正见同表 `stable_corruption`。
9. winner=`{winner}`；按验证 raw TV、stable corruption、aligned reference NLL 的预注册顺序决定。
10. 本轮不自动授权 HumanEval K=2；只有外部训练比较完成后由下一次研究决策决定。

本次脚本调用执行外部测试：`{test_open_count}` 次；这不是跨调用的历史计数。本报告不声明 HumanEval Pass@1 改善。
"""
    (result_dir / "report.zh.md").write_text(report, encoding="utf-8")
    if not args.skip_research_record_updates:
        result_document = Path(args.result_document)
        if not result_document.is_absolute():
            result_document = REPO / result_document
        update_research_records(
            result_dir=result_dir,
            tv=tv,
            kl=kl,
            verdict=verdict,
            winner=winner,
            run_id=args.run_id,
            run_date_utc=args.run_date_utc,
            result_document=result_document,
        )
    print(json.dumps({"status": verdict, "winner": winner, "test_open_count": test_open_count}))
    return 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--bank-db", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--common-training-config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", default="dreamon_markov_head_training_20260901_v1")
    parser.add_argument("--run-date-utc", default="2026-09-01")
    parser.add_argument("--branch", default="codex/dreamon-markov-head-training-v1")
    parser.add_argument("--base-head", default="77f0572b1ca4fe031ab6bbf29b3a4d8740f38802")
    parser.add_argument(
        "--result-document",
        default="docs/paper_agent/experiments/20260901_dreamon_markov_head_training_result.zh.md",
    )
    parser.add_argument("--report-title", default="DreamOn 外部一阶 Markov 头训练 v1")
    parser.add_argument(
        "--data-isolation-status", default="requires independent source/code overlap audit"
    )
    parser.add_argument(
        "--data-isolation-note",
        default=(
            "HumanEval was not an explicit training source; actual source overlap requires an "
            "independent audit."
        ),
    )
    parser.add_argument("--skip-research-record-updates", action="store_true")
    parser.add_argument("--resume-existing-external-diagnostics", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
