#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl, metric, oracle_bucket
from experiments.action_ceiling.action_ceiling_matrix import (
    compile_passed,
    current_branch,
    current_commit,
    environment_info,
    git_capture,
    parse_task_id_group,
    reset_action_seed,
    sha256_text,
    stable_task_seed,
    verification_error_message,
    verification_error_type,
    write_csv,
    write_jsonl,
)
from experiments.action_ceiling.distinct_candidate_ceiling import decode_fixed_canvas
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import load_humaneval_infilling
from expvision_dllm_clean.modeling import load_model_and_tokenizer


JsonDict = Dict[str, Any]

ACTION_TARGETS = {
    "KEEP_PRIMARY": None,
    "EXPAND_16": 16,
    "EXPAND_24": 24,
    "EXPAND_32": 32,
    "EXPAND_48": 48,
}
ACTION_ORDER = tuple(ACTION_TARGETS)
BANK_SPLITS = ("train", "calibration", "validation")
FEATURE_VARIANTS = ("probe_only", "trace_only", "probe_trace_fused")
POLICY_VARIANTS = ("benefit_only", "benefit_plus_harm", "risk_calibrated_benefit_plus_harm")
FEATURE_SCHEMA_VERSION = "frozen_canvas_features_v1"
ACTION_SCHEMA_VERSION = "deployable_canvas_actions_v1"
PRIMARY_RESULTS = "/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl"
CONTROL_RESULTS = "/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529/results.jsonl"
MIDCONS_RESULTS = "/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626/results.jsonl"
ROUTE2_RESULTS = "/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl"
V6_RESULTS = "/home/shx/projects/dllm_infilling/outputs_clean/full_route2_v6_short_override_gpu2_20260620_124754/results.jsonl"
CAL_RESULTS = "/home/shx/projects/dllm_infilling/outputs_clean/full_official_cal_lcas_v3b_accel_gpus10_20260702_resumed_full_20260703_111615/results.jsonl"
ROUTE2_ERROR_DIR = "analysis_outputs/route2_error_analysis_20260617_165806"


def read_csv_rows(path: Path) -> List[JsonDict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def row_passed(row: Mapping[str, Any]) -> bool:
    if "passed" in row:
        return _to_bool(row.get("passed"))
    return _to_bool(row.get("action_passed"))


def _safe_div(num: float, den: float) -> Optional[float]:
    return None if den == 0 else num / den


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def _mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else sum(values) / len(values)


def ensure_output_dir(prefix: str, timestamp: Optional[str], *, resume_dir: Optional[str] = None) -> Path:
    if resume_dir:
        path = Path(resume_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("analysis_outputs") / f"{prefix}_{stamp}"
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing_path(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(Path(ROOT).parent)
        except ValueError:
            relative = None
        if relative is not None:
            rooted = Path(ROOT) / relative
            if rooted.exists():
                return str(rooted)
    rooted = Path(ROOT) / path
    if rooted.exists():
        return str(rooted)
    return path


def rows_by_task(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["task_id"]): row for row in rows}


def load_split_assignments(split_dir: Path) -> Dict[str, str]:
    return {str(row["task_id"]): str(row["split"]) for row in read_csv_rows(split_dir / "row_split_assignment.csv")}


def load_split_task_ids(split_dir: Path) -> Dict[str, List[str]]:
    return {
        split: json.loads((split_dir / f"{split}_tasks.json").read_text(encoding="utf-8"))
        for split in ("train", "calibration", "validation", "test")
    }


def render_frozen_protocol(split_dir: Path, lock_path: Path) -> str:
    return "\n".join(
        [
            "# Frozen Controller Protocol",
            "",
            "更新时间：2026-07-03 CST",
            "",
            "## Central Claim",
            "",
            "Unknown-length DLLM infilling exhibits two coupled but separable regimes: canvas inadequacy and rescue inadequacy. Missed true-long failures are substantially trigger/canvas-limited, whereas already-triggered failures remain rescue-limited under the current longer-trajectory and trace-remasking action family. The main deployable opportunity is therefore risk-controlled, non-oracle prediction of when and how much to expand.",
            "",
            "中文：unknown-length DLLM infilling 至少有 canvas inadequacy 与 rescue inadequacy 两个相互耦合但可分离的 regime。missed true-long failures 主要受 trigger/canvas 限制；已经触发的 failures 在当前 longer-trajectory 和 trace-remasking action family 下仍然 rescue-limited。因此当前可部署机会是风险受控、非 oracle 地预测何时扩展以及扩展到多长。",
            "",
            "## Split Lock",
            "",
            f"- grouped split: `{split_dir}`",
            f"- test lock: `{lock_path}`",
            "- train：拟合 controller 参数。",
            "- calibration：概率校准与 harm threshold / operating point。",
            "- validation：方法、ablation 与 operating point 选择。",
            "- test：只允许最终冻结方法评测一次。",
            "",
            "## 禁止项",
            "",
            "- 不得把 oracle action-ceiling 结果写成 held-out controller result。",
            "- 不得用 test pass/fail、error type、reference code、oracle length 或 action outcome 构造 inference feature。",
            "- validation gate 通过前，test lock 必须保持 `sealed` 且 `test_evaluation_count=0`。",
            "- test 运行后不得反向修改 feature、model、threshold 或 action set。",
            "- E/F/G remasking 只作为 diagnostic actions，不进入 deployable action set。",
            "",
            "## Deployable Action Set",
            "",
            "- `KEEP_PRIMARY`",
            "- `EXPAND_16`",
            "- `EXPAND_24`",
            "- `EXPAND_32`",
            "- `EXPAND_48`",
            "- actual canvas 定义：`max(primary_selected_length, target_length)`。",
            "",
            "## Validation Gate",
            "",
            "- calibration intervention harm 的 95% upper confidence bound 不超过 5%。",
            "- validation `<=8` bucket 不出现净 regression。",
            "- validation 总 Pass@1 不低于 V6 same-protocol baseline。",
            "- validation 至少满足：总 Pass@1 提高、long bucket 提高且 aggregate 不下降、同准确率下降低 compute，或 risk-coverage Pareto 改善。",
            "",
        ]
    )


def execute_freeze(args: argparse.Namespace) -> None:
    split_dir = Path(args.split_dir)
    output_dir = ensure_output_dir("frozen_controller", args.timestamp, resume_dir=args.output_dir)
    split_manifest = split_dir / "split_manifest.json"
    split_payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    split_tasks = load_split_task_ids(split_dir)
    test_rows = [row for row in read_csv_rows(split_dir / "row_split_assignment.csv") if row["split"] == "test"]
    lock = {
        "created_at_cst": datetime.now().isoformat(timespec="seconds"),
        "branch": current_branch(),
        "commit": current_commit(),
        "split_dir": str(split_dir),
        "split_manifest_sha256": sha256_file(split_manifest),
        "split_manifest": split_payload,
        "train_task_ids": split_tasks["train"],
        "calibration_task_ids": split_tasks["calibration"],
        "validation_task_ids": split_tasks["validation"],
        "test_task_ids": split_tasks["test"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "test_row_count": len(test_rows),
        "test_status": "sealed",
        "test_evaluation_count": 0,
        "rules": {
            "train": "fit model coefficients",
            "calibration": "risk threshold and probability calibration",
            "validation": "model and ablation selection",
            "test": "one frozen final evaluation only after validation gate",
        },
        "non_goal": "Oracle action-ceiling results are diagnostic and are not held-out controller test results.",
    }
    lock_path = output_dir / "test_lock.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    protocol_path = Path(args.protocol_doc)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(render_frozen_protocol(split_dir, lock_path), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "test_lock": str(lock_path), "test_status": "sealed"}, ensure_ascii=False, indent=2))


def action_schema() -> JsonDict:
    return {
        "version": ACTION_SCHEMA_VERSION,
        "actions": [
            {"action": action, "target_length": target, "definition": "actual_canvas=max(primary_selected_length,target_length)"}
            for action, target in ACTION_TARGETS.items()
        ],
        "seed": 0,
        "denoising": {
            "total_steps": 64,
            "early_commit_enabled": True,
            "decoder": "experiments.action_ceiling.distinct_candidate_ceiling.decode_fixed_canvas",
        },
        "remasking_actions_excluded": ["E_oracle_sufficient_no_early_commit", "F_oracle_sufficient_trace_remask", "G_oracle_sufficient_trace_span_remask"],
    }


def actual_canvas(primary_len: int, action: str) -> int:
    target = ACTION_TARGETS[action]
    return int(primary_len) if target is None else max(int(primary_len), int(target))


def result_to_bank_rows(
    *,
    task_id: str,
    task_group: str,
    split: str,
    primary_selected_length: int,
    oracle_length: Optional[int],
    action_rows: Sequence[Tuple[str, int]],
    result: Mapping[str, Any],
    task_seed: int,
    source_commit: str,
) -> List[JsonDict]:
    metrics = result.get("metrics") or {}
    generated_text = result.get("middle_text")
    generated_hash = sha256_text(generated_text)
    error_message = verification_error_message(result)
    passed = bool(metrics.get("passed"))
    compile_ok = compile_passed(result)
    error_type = verification_error_type(result)
    rows = []
    for action, canvas in action_rows:
        rows.append(
            {
                "task_id": task_id,
                "task_group": task_group,
                "split": split,
                "primary_selected_length": primary_selected_length,
                "oracle_length": oracle_length,
                "oracle_bucket": oracle_bucket(oracle_length),
                "action": action,
                "actual_canvas": canvas,
                "target_length": ACTION_TARGETS[action],
                "passed": passed,
                "compile_passed": compile_ok,
                "error_type": error_type,
                "error_message_short": None if error_message is None else str(error_message).replace("\n", " ")[:240],
                "generated_text_sha256": generated_hash,
                "code_sha256": sha256_text(result.get("code")),
                "inference_cost_sec": metrics.get("total_sec_including_probe"),
                "decode_steps": metrics.get("total_steps"),
                "actual_forward_steps": metrics.get("actual_forward_steps"),
                "effective_update_steps": metrics.get("effective_update_steps"),
                "stop_reason": metrics.get("stop_reason"),
                "early_commit_enabled": metrics.get("early_commit_enabled"),
                "early_commit_triggered": metrics.get("stop_reason") == "global_gap_early_commit",
                "seed": task_seed,
                "experimental_seed": 0,
                "action_equivalence_class": f"{task_id}|canvas={canvas}",
                "action_schema_version": ACTION_SCHEMA_VERSION,
                "source_commit": source_commit,
            }
        )
    return rows


def load_existing_bank_rows(path: Path) -> List[JsonDict]:
    if not path.exists():
        return []
    rows: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def summarize_bank(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_split = Counter(str(row.get("split")) for row in rows)
    by_action = Counter(str(row.get("action")) for row in rows)
    primary_by_task = {str(row["task_id"]): row for row in rows if row.get("action") == "KEEP_PRIMARY"}
    rows_by_task_action = {(str(row["task_id"]), str(row["action"])): row for row in rows}
    benefit = 0
    harm = 0
    for (task_id, action), row in rows_by_task_action.items():
        if action == "KEEP_PRIMARY":
            continue
        primary = primary_by_task.get(task_id)
        if not primary:
            continue
        if (not row_passed(primary)) and row_passed(row):
            benefit += 1
        if row_passed(primary) and (not row_passed(row)):
            harm += 1
    costs = [_to_float(row.get("inference_cost_sec")) for row in rows if row.get("inference_cost_sec") not in {None, ""}]
    return {
        "mode": "controller_action_bank",
        "row_count": len(rows),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "split_counts": dict(sorted(by_split.items())),
        "action_counts": dict(sorted(by_action.items())),
        "pass_count": sum(1 for row in rows if row_passed(row)),
        "benefit_labels_non_keep": benefit,
        "harm_labels_non_keep": harm,
        "cost": {"mean": _mean(costs), "p50": _percentile(costs, 50), "p95": _percentile(costs, 95)},
    }


def render_bank_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Controller Action Bank",
        "",
        f"- row_count: `{summary.get('row_count')}`",
        f"- task_count: `{summary.get('task_count')}`",
        f"- split_counts: `{summary.get('split_counts')}`",
        f"- action_counts: `{summary.get('action_counts')}`",
        f"- pass_count: `{summary.get('pass_count')}`",
        f"- benefit_labels_non_keep: `{summary.get('benefit_labels_non_keep')}`",
        f"- harm_labels_non_keep: `{summary.get('harm_labels_non_keep')}`",
        f"- cost: `{summary.get('cost')}`",
    ]
    coverage = summary.get("merge_coverage")
    if isinstance(coverage, Mapping):
        lines.extend(
            [
                "",
                "## Merge Coverage",
                "",
                f"- expected_action_rows: `{coverage.get('expected_action_rows')}`",
                f"- observed_action_rows: `{coverage.get('observed_action_rows')}`",
                f"- missing_action_rows: `{coverage.get('missing_action_rows')}`",
                f"- extra_action_rows: `{coverage.get('extra_action_rows')}`",
                f"- duplicate_source_action_rows: `{coverage.get('duplicate_action_rows')}`",
            ]
        )
    lines.extend(
        [
            "",
            "The bank is generated only for train/calibration/validation. Test rows remain sealed unless the validation gate authorizes one frozen test run.",
            "",
        ]
    )
    return "\n".join(lines)


def build_bank_manifest(args: argparse.Namespace, output_dir: Path, status: str, started_at: str, **extra: Any) -> JsonDict:
    command = shlex.join([sys.executable, *sys.argv])
    env_prefix = {
        key: os.environ[key]
        for key in ("CUDA_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        if key in os.environ
    }
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env_prefix.items()))
    return {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": command,
        "repro_command": f"{env_text} {command}".strip(),
        "env_command_prefix": env_prefix,
        "started_at": started_at,
        "execution_status": status,
        "output_dir": str(output_dir),
        "split_dir": args.split_dir,
        "splits": list(args.bank_splits.split(",")),
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "primary_results": args.primary_results,
        "environment": environment_info(args.model_path),
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
        **extra,
    }


def execute_bank(args: argparse.Namespace) -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    output_dir = ensure_output_dir("controller_action_bank", args.timestamp, resume_dir=args.output_dir)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(build_bank_manifest(args, output_dir, "running", started_at), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        split_assignment = load_split_assignments(Path(args.split_dir))
        wanted_splits = {item.strip() for item in args.bank_splits.split(",") if item.strip()}
        primary_rows = rows_by_task(load_jsonl(resolve_existing_path(args.primary_results)))
        tasks = load_humaneval_infilling(split=args.split, dataset_subset=args.dataset_subset)
        by_task = {task.task_id: task for task in tasks}
        task_ids = [
            task_id
            for task_id in sorted(split_assignment)
            if split_assignment[task_id] in wanted_splits and task_id in primary_rows
        ]
        if args.shard_count > 1:
            task_ids = [task_id for idx, task_id in enumerate(task_ids) if idx % args.shard_count == args.shard_index]
        if args.limit_cases is not None:
            task_ids = task_ids[: args.limit_cases]

        cfg = ExperimentConfig()
        cfg.model.model_path = args.model_path
        cfg.data.split = args.split
        cfg.data.dataset_subset = args.dataset_subset
        cfg.decode.lcas_policy = args.lcas_policy
        tokenizer, model = load_model_and_tokenizer(cfg.model)
        source_commit = current_commit()

        bank_path = output_dir / "action_bank.jsonl"
        existing = load_existing_bank_rows(bank_path)
        existing_by_action = {(str(row["task_id"]), str(row["action"])): row for row in existing}
        existing_by_canvas: Dict[Tuple[str, int], List[JsonDict]] = defaultdict(list)
        for row in existing:
            existing_by_canvas[(str(row["task_id"]), int(row["actual_canvas"]))].append(row)

        total_planned = len(task_ids) * len(ACTION_ORDER)
        for index, task_id in enumerate(task_ids, start=1):
            primary = primary_rows[task_id]
            metrics = primary.get("metrics") or {}
            primary_len = _to_int(metric(primary, "selected_mask_length", metric(primary, "mask_length")), 64)
            oracle_len = _to_int(metric(primary, "oracle_mask_length"))
            split = split_assignment[task_id]
            task_group = parse_task_id_group(task_id)
            pending_by_canvas: Dict[int, List[str]] = defaultdict(list)
            for action in ACTION_ORDER:
                if (task_id, action) in existing_by_action:
                    continue
                canvas = actual_canvas(int(primary_len), action)
                pending_by_canvas[canvas].append(action)
            for canvas, actions in sorted(pending_by_canvas.items()):
                reusable = existing_by_canvas.get((task_id, canvas), [])
                if reusable:
                    base = reusable[0]
                    cloned = []
                    for action in actions:
                        row = dict(base)
                        row.update(
                            {
                                "action": action,
                                "target_length": ACTION_TARGETS[action],
                                "action_equivalence_class": f"{task_id}|canvas={canvas}",
                            }
                        )
                        cloned.append(row)
                    append_jsonl(bank_path, cloned)
                    continue
                task_seed = stable_task_seed(42, task_id, 0)
                reset_action_seed(task_seed)
                print(
                    f"[{index}/{len(task_ids)}] task_id={task_id} split={split} canvas={canvas} actions={','.join(actions)}",
                    flush=True,
                )
                result = decode_fixed_canvas(
                    task=by_task[task_id],
                    tokenizer=tokenizer,
                    model=model,
                    cfg=cfg,
                    canvas_len=int(canvas),
                    total_steps=64,
                    early_commit_enabled=True,
                    phase_name="controller_action_bank_fixed_canvas",
                )
                rows = result_to_bank_rows(
                    task_id=task_id,
                    task_group=task_group,
                    split=split,
                    primary_selected_length=int(primary_len),
                    oracle_length=oracle_len,
                    action_rows=[(action, int(canvas)) for action in actions],
                    result=result,
                    task_seed=task_seed,
                    source_commit=source_commit,
                )
                append_jsonl(bank_path, rows)

        rows = load_existing_bank_rows(bank_path)
        write_csv(output_dir / "action_bank.csv", rows)
        (output_dir / "action_schema.json").write_text(json.dumps(action_schema(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        summary = summarize_bank(rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "report.md").write_text(render_bank_report(summary), encoding="utf-8")
        manifest = build_bank_manifest(
            args,
            output_dir,
            "completed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            planned_task_count=len(task_ids),
            planned_action_rows=total_planned,
            written_rows=len(rows),
            summary=summary,
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, ensure_ascii=False, indent=2))
    except BaseException:
        manifest = build_bank_manifest(
            args,
            output_dir,
            "failed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            failure_traceback=traceback.format_exc(),
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        raise


PROBE_FEATURES = [
    "selected_len",
    "best_len",
    "best_score",
    "best_long_len",
    "best_long_score",
    "long_ratio",
    "raw_long_ratio",
    "selected_score",
    "selected_raw_score",
    "long_minus_best_score",
    "long_len_minus_selected",
    "official_selected_length",
    "s3_selected_length",
]
TRACE_FEATURES = [
    "mean_final_confidence",
    "mean_gap_at_stop_or_final",
    "mean_top1_at_stop_or_final",
    "remaining_mask_ratio_at_stop_or_final",
    "remaining_masks_at_stop_or_final",
    "stop_step",
    "route2_trace_top1_last",
    "route2_trace_top1_median",
    "route2_trace_confidence_max",
    "route2_trace_max_remaining_plateau_steps",
]
STATIC_FEATURES = ["primary_selected_length", "candidate_actual_canvas", "canvas_delta_from_primary", "normalized_canvas_cost"]


def feature_schema() -> JsonDict:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "identifier_columns": ["task_id", "task_group", "split"],
        "label_columns_not_features": [
            "oracle_length",
            "oracle_bucket",
            "primary_passed",
            "action_passed",
            "benefit_label",
            "harm_label",
            "underallocation_label",
            "error_type",
        ],
        "feature_variants": {
            "probe_only": PROBE_FEATURES + STATIC_FEATURES,
            "trace_only": TRACE_FEATURES + STATIC_FEATURES,
            "probe_trace_fused": PROBE_FEATURES + TRACE_FEATURES + STATIC_FEATURES,
        },
        "forbidden_features": [
            "oracle_length",
            "reference_code",
            "unit_test_result",
            "pass_fail",
            "action_bank_outcome",
            "error_type",
            "HumanEval task id",
            "split name",
            "test-derived statistic",
        ],
    }


def extract_base_features(primary: Mapping[str, Any], route2: Optional[Mapping[str, Any]]) -> JsonDict:
    m = primary.get("metrics") or {}
    r = (route2 or {}).get("metrics") or {}
    selected = _to_float(metric(primary, "selected_mask_length", metric(primary, "mask_length")), 0.0)
    best_long_score = _to_float(m.get("best_long_score"), 0.0)
    best_score = _to_float(m.get("best_score"), 0.0)
    return {
        "selected_len": selected,
        "best_len": _to_float(m.get("best_len"), 0.0),
        "best_score": best_score,
        "best_long_len": _to_float(m.get("best_long_len"), 0.0),
        "best_long_score": best_long_score,
        "long_ratio": _to_float(m.get("long_ratio"), 0.0),
        "raw_long_ratio": _to_float(m.get("raw_long_ratio"), 0.0),
        "selected_score": _to_float(m.get("selected_score"), 0.0),
        "selected_raw_score": _to_float(m.get("selected_raw_score"), 0.0),
        "long_minus_best_score": best_long_score - best_score,
        "long_len_minus_selected": _to_float(m.get("best_long_len"), 0.0) - selected,
        "official_selected_length": _to_float(m.get("official_selected_length"), 0.0),
        "s3_selected_length": _to_float(m.get("s3_selected_length"), 0.0),
        "mean_final_confidence": _to_float(m.get("mean_final_confidence"), 0.0),
        "mean_gap_at_stop_or_final": _to_float(m.get("mean_gap_at_stop_or_final"), 0.0),
        "mean_top1_at_stop_or_final": _to_float(m.get("mean_top1_at_stop_or_final"), 0.0),
        "remaining_mask_ratio_at_stop_or_final": _to_float(m.get("remaining_mask_ratio_at_stop_or_final"), 0.0),
        "remaining_masks_at_stop_or_final": _to_float(m.get("remaining_masks_at_stop_or_final"), 0.0),
        "stop_step": _to_float(m.get("stop_step"), 0.0),
        "route2_trace_top1_last": _to_float(r.get("route2_trace_top1_last"), 0.0),
        "route2_trace_top1_median": _to_float(r.get("route2_trace_top1_median"), 0.0),
        "route2_trace_confidence_max": _to_float(r.get("route2_trace_confidence_max"), 0.0),
        "route2_trace_max_remaining_plateau_steps": _to_float(r.get("route2_trace_max_remaining_plateau_steps"), 0.0),
    }


def build_feature_rows(
    *,
    split_assignment: Mapping[str, str],
    primary_rows: Mapping[str, Mapping[str, Any]],
    route2_rows: Mapping[str, Mapping[str, Any]],
) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for task_id, split in sorted(split_assignment.items()):
        if task_id not in primary_rows:
            continue
        primary = primary_rows[task_id]
        m = primary.get("metrics") or {}
        base = extract_base_features(primary, route2_rows.get(task_id))
        rows.append(
            {
                "task_id": task_id,
                "task_group": parse_task_id_group(task_id),
                "split": split,
                "oracle_length": _to_int(m.get("oracle_mask_length")),
                "oracle_bucket": oracle_bucket(_to_int(m.get("oracle_mask_length"))),
                "primary_passed": bool(m.get("passed")),
                "primary_selected_length": _to_int(m.get("selected_mask_length"), _to_int(m.get("mask_length"), 0)),
                **base,
            }
        )
    return rows


class LogisticModel:
    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = list(feature_names)
        self.mean: List[float] = []
        self.std: List[float] = []
        self.weights: List[float] = []

    def fit(self, rows: Sequence[Mapping[str, Any]], labels: Sequence[int], *, epochs: int = 600, lr: float = 0.08, l2: float = 0.01) -> None:
        x = np.array([[float(row.get(name, 0.0) or 0.0) for name in self.feature_names] for row in rows], dtype=np.float64)
        y = np.array(labels, dtype=np.float64)
        if x.size == 0:
            self.mean = [0.0 for _ in self.feature_names]
            self.std = [1.0 for _ in self.feature_names]
            self.weights = [0.0 for _ in range(len(self.feature_names) + 1)]
            return
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        z = (x - mean) / std
        z = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
        w = np.zeros(z.shape[1], dtype=np.float64)
        pos = max(1.0, float(y.sum()))
        neg = max(1.0, float(len(y) - y.sum()))
        sample_weight = np.where(y > 0, neg / pos, 1.0)
        for _ in range(int(epochs)):
            logits = np.clip(z @ w, -40.0, 40.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (z.T @ ((p - y) * sample_weight)) / max(1, len(y))
            grad[1:] += l2 * w[1:]
            w -= lr * grad
        self.mean = [float(v) for v in mean]
        self.std = [float(v) for v in std]
        self.weights = [float(v) for v in w]

    def predict_one(self, row: Mapping[str, Any]) -> float:
        if not self.weights:
            return 0.0
        vals = [float(row.get(name, 0.0) or 0.0) for name in self.feature_names]
        z = [1.0] + [(v - m) / s for v, m, s in zip(vals, self.mean, self.std)]
        logit = max(-40.0, min(40.0, sum(v * w for v, w in zip(z, self.weights))))
        return float(1.0 / (1.0 + math.exp(-logit)))

    def to_json(self) -> JsonDict:
        return {"feature_names": self.feature_names, "mean": self.mean, "std": self.std, "weights": self.weights}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "LogisticModel":
        model = cls(payload.get("feature_names") or [])
        model.mean = [float(value) for value in (payload.get("mean") or [])]
        model.std = [float(value) for value in (payload.get("std") or [])]
        model.weights = [float(value) for value in (payload.get("weights") or [])]
        return model


def make_action_feature_rows(feature_by_task: Mapping[str, Mapping[str, Any]], bank_rows: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    primary_pass = {str(row["task_id"]): row_passed(row) for row in bank_rows if row.get("action") == "KEEP_PRIMARY"}
    rows = []
    for row in bank_rows:
        task_id = str(row["task_id"])
        base = feature_by_task[task_id]
        actual = _to_float(row.get("actual_canvas"), 0.0)
        primary_len = _to_float(row.get("primary_selected_length"), 0.0)
        oracle_len = _to_int(row.get("oracle_length"))
        action_passed = row_passed(row)
        p_pass = primary_pass.get(task_id, False)
        rows.append(
            {
                **{key: base.get(key) for key in ["task_id", "task_group", "split", "oracle_length", "oracle_bucket"]},
                **{key: base.get(key, 0.0) for key in PROBE_FEATURES + TRACE_FEATURES},
                "action": row.get("action"),
                "actual_canvas": int(actual),
                "primary_selected_length": int(primary_len),
                "candidate_actual_canvas": actual,
                "canvas_delta_from_primary": max(0.0, actual - primary_len),
                "normalized_canvas_cost": actual / 48.0,
                "primary_passed": p_pass,
                "action_passed": action_passed,
                "benefit_label": (not p_pass) and action_passed,
                "harm_label": p_pass and (not action_passed),
                "underallocation_label": False if oracle_len is None else actual < oracle_len,
                "generated_text_sha256": row.get("generated_text_sha256"),
                "inference_cost_sec": row.get("inference_cost_sec"),
            }
        )
    return rows


def binomial_upper_95(successes: int, n: int) -> Optional[float]:
    if n == 0:
        return None
    z = 1.96
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return min(1.0, (centre + margin) / denom)


def train_models(action_rows: Sequence[Mapping[str, Any]], feature_cols: Sequence[str]) -> Dict[str, LogisticModel]:
    train = [row for row in action_rows if row.get("split") == "train" and row.get("action") != "KEEP_PRIMARY"]
    models = {
        "benefit": LogisticModel(feature_cols),
        "harm": LogisticModel(feature_cols),
        "underallocation": LogisticModel(feature_cols),
    }
    models["benefit"].fit(train, [1 if row.get("benefit_label") else 0 for row in train])
    models["harm"].fit(train, [1 if row.get("harm_label") else 0 for row in train])
    models["underallocation"].fit(train, [1 if row.get("underallocation_label") else 0 for row in train])
    return models


def score_action(row: Mapping[str, Any], models: Mapping[str, LogisticModel], policy_variant: str) -> JsonDict:
    if row.get("action") == "KEEP_PRIMARY":
        return {"p_benefit": 0.0, "p_harm": 0.0, "p_underalloc": 0.0, "score": 0.0}
    p_benefit = models["benefit"].predict_one(row)
    p_harm = models["harm"].predict_one(row)
    p_under = models["underallocation"].predict_one(row)
    cost = _to_float(row.get("normalized_canvas_cost"), 0.0)
    if policy_variant == "benefit_only":
        score = p_benefit - 0.02 * cost
    elif policy_variant == "benefit_plus_harm":
        score = p_benefit - 0.02 * cost - 0.75 * p_harm + 0.05 * (1.0 - p_under)
    else:
        score = p_benefit - 0.02 * cost - 1.25 * p_harm + 0.05 * (1.0 - p_under)
    return {"p_benefit": p_benefit, "p_harm": p_harm, "p_underalloc": p_under, "score": score}


def select_actions(
    action_rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, LogisticModel],
    *,
    policy_variant: str,
    score_threshold: float,
) -> List[JsonDict]:
    by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in action_rows:
        by_task[str(row["task_id"])].append(row)
    selected: List[JsonDict] = []
    for task_id, rows in sorted(by_task.items()):
        keep = next(row for row in rows if row.get("action") == "KEEP_PRIMARY")
        candidates = []
        for row in rows:
            scored = score_action(row, models, policy_variant)
            candidates.append((scored["score"], str(row.get("action")), row, scored))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, _, best_row, best_scored = candidates[0]
        if best_row.get("action") == "KEEP_PRIMARY" or best_score < score_threshold:
            best_row = keep
            best_scored = {"p_benefit": 0.0, "p_harm": 0.0, "p_underalloc": 0.0, "score": 0.0}
        selected.append({**dict(best_row), **best_scored, "selected_action": best_row.get("action"), "intervened": best_row.get("action") != "KEEP_PRIMARY"})
    return selected


def paired_summary(selected: Sequence[Mapping[str, Any]], primary_by_task: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    wins = losses = ties_pass = ties_fail = 0
    harms = benefits = primary_pass_losses = 0
    for row in selected:
        task_id = str(row["task_id"])
        primary = primary_by_task[task_id]
        before = row_passed(primary)
        after = row_passed(row)
        if (not before) and after:
            wins += 1
            benefits += 1
        elif before and (not after):
            losses += 1
            harms += 1
            primary_pass_losses += 1
        elif after:
            ties_pass += 1
        else:
            ties_fail += 1
    return {
        "pass_count": wins + ties_pass,
        "total": len(selected),
        "pass_rate": _safe_div(wins + ties_pass, len(selected)),
        "wins_vs_primary": wins,
        "losses_vs_primary": losses,
        "ties_pass": ties_pass,
        "ties_fail": ties_fail,
        "benefits": benefits,
        "harms": harms,
        "primary_pass_losses": primary_pass_losses,
    }


def evaluate_selection(selected: Sequence[Mapping[str, Any]], primary_by_task: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    paired = paired_summary(selected, primary_by_task)
    def is_intervened(row: Mapping[str, Any]) -> bool:
        if "intervened" in row:
            return _to_bool(row.get("intervened"))
        return str(row.get("action")) != "KEEP_PRIMARY"

    intervened = [row for row in selected if is_intervened(row)]
    harms = sum(1 for row in intervened if row_passed(primary_by_task[str(row["task_id"])]) and not row_passed(row))
    benefits = sum(1 for row in intervened if (not row_passed(primary_by_task[str(row["task_id"])])) and row_passed(row))
    costs = [_to_float(row.get("inference_cost_sec")) for row in selected if row.get("inference_cost_sec") not in {None, ""}]
    by_bucket: Dict[str, JsonDict] = {}
    for bucket in sorted({str(row.get("oracle_bucket")) for row in selected}):
        bucket_rows = [row for row in selected if str(row.get("oracle_bucket")) == bucket]
        bucket_primary = {task_id: primary_by_task[task_id] for task_id in primary_by_task if any(str(row["task_id"]) == task_id for row in bucket_rows)}
        by_bucket[bucket] = paired_summary(bucket_rows, bucket_primary)
    return {
        **paired,
        "intervention_count": len(intervened),
        "intervention_coverage": _safe_div(len(intervened), len(selected)),
        "intervention_harm_count": harms,
        "intervention_harm_rate": _safe_div(harms, len(intervened)),
        "intervention_harm_upper95": binomial_upper_95(harms, len(intervened)),
        "intervention_benefit_count": benefits,
        "benefit_precision": _safe_div(benefits, len(intervened)),
        "selected_canvas_distribution": dict(sorted(Counter(str(row.get("actual_canvas")) for row in selected).items())),
        "mean_cost_sec": _mean(costs),
        "p50_cost_sec": _percentile(costs, 50),
        "p95_cost_sec": _percentile(costs, 95),
        "bucket_summary": by_bucket,
    }


def calibrate_threshold(
    action_rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, LogisticModel],
    *,
    policy_variant: str,
    primary_by_task: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    calibration_rows = [row for row in action_rows if row.get("split") == "calibration"]
    raw_selected = select_actions(calibration_rows, models, policy_variant=policy_variant, score_threshold=-999.0)
    scores = sorted({float(row.get("score") or 0.0) for row in raw_selected if row.get("selected_action") != "KEEP_PRIMARY"}, reverse=True)
    candidates = [999.0] + scores + [0.0, -999.0]
    best: Optional[JsonDict] = None
    curve = []
    for threshold in candidates:
        selected = select_actions(calibration_rows, models, policy_variant=policy_variant, score_threshold=float(threshold))
        metrics = evaluate_selection(selected, primary_by_task)
        entry = {"score_threshold": float(threshold), **metrics}
        curve.append(entry)
        upper = metrics.get("intervention_harm_upper95")
        if upper is not None and upper <= 0.05:
            if best is None or (metrics.get("intervention_count") or 0) > (best.get("intervention_count") or 0):
                best = entry
    if best is None:
        best = curve[0]
        best["calibration_failed_harm_budget"] = True
    else:
        best["calibration_failed_harm_budget"] = False
    return {"selected_operating_point": best, "risk_coverage_curve": curve}


def action_rows_by_task_action(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(row["task_id"]), str(row["action"])): row for row in rows}


def baseline_from_action(action_rows: Sequence[Mapping[str, Any]], split: str, action: str) -> List[Mapping[str, Any]]:
    return [row for row in action_rows if row.get("split") == split and row.get("action") == action]


def historical_baseline(rows_path: str, task_ids: Sequence[str]) -> List[JsonDict]:
    resolved_path = resolve_existing_path(rows_path)
    by_task = rows_by_task(load_jsonl(resolved_path))
    output = []
    for task_id in task_ids:
        row = by_task.get(task_id)
        if row is None:
            continue
        m = row.get("metrics") or {}
        output.append(
            {
                "task_id": task_id,
                "passed": bool(m.get("passed")),
                "actual_canvas": m.get("selected_mask_length", m.get("mask_length")),
                "inference_cost_sec": m.get("total_sec_including_probe", m.get("total_sec")),
                "oracle_bucket": oracle_bucket(_to_int(m.get("oracle_mask_length"))),
                "action": Path(resolved_path).parent.name,
            }
        )
    return output


def select_equal_coverage_simple_gate(action_rows: Sequence[Mapping[str, Any]], feature_by_task: Mapping[str, Mapping[str, Any]], split: str, intervention_count: int) -> List[Mapping[str, Any]]:
    by_task_action = action_rows_by_task_action(action_rows)
    task_ids = sorted({str(row["task_id"]) for row in action_rows if row.get("split") == split})
    scored = []
    for task_id in task_ids:
        feat = feature_by_task[task_id]
        score = _to_float(feat.get("long_ratio")) + 0.02 * _to_float(feat.get("best_long_len")) - 0.01 * _to_float(feat.get("selected_len"))
        scored.append((score, task_id))
    chosen = {task_id for _, task_id in sorted(scored, reverse=True)[:intervention_count]}
    selected = []
    for task_id in task_ids:
        action = "EXPAND_32" if task_id in chosen else "KEEP_PRIMARY"
        selected.append(by_task_action[(task_id, action)])
    return selected


def grouped_bootstrap_ci(selected: Sequence[Mapping[str, Any]], *, iters: int = 500, seed: int = 20260703) -> JsonDict:
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[str(row.get("task_group"))].append(row)
    group_keys = sorted(groups)
    if not group_keys:
        return {"low": None, "high": None}
    values = []
    for _ in range(iters):
        sampled = [group_keys[int(rng.integers(0, len(group_keys)))] for _ in group_keys]
        rows = [row for group in sampled for row in groups[group]]
        values.append(sum(1 for row in rows if row_passed(row)) / max(1, len(rows)))
    return {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}


def execute_controller(args: argparse.Namespace) -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    output_dir = ensure_output_dir("controller_validation", args.timestamp, resume_dir=args.output_dir)
    bank_rows = load_existing_bank_rows(Path(args.bank_dir) / "action_bank.jsonl")
    if not bank_rows:
        raise ValueError(f"no action bank rows found in {args.bank_dir}")
    split_assignment = load_split_assignments(Path(args.split_dir))
    primary_rows = rows_by_task(load_jsonl(resolve_existing_path(args.primary_results)))
    route2_rows = rows_by_task(load_jsonl(resolve_existing_path(args.route2_results)))
    controller_split_assignment = {
        task_id: split
        for task_id, split in split_assignment.items()
        if split in BANK_SPLITS
    }
    feature_rows = build_feature_rows(split_assignment=controller_split_assignment, primary_rows=primary_rows, route2_rows=route2_rows)
    feature_by_task = {str(row["task_id"]): row for row in feature_rows}
    action_feature_rows = make_action_feature_rows(feature_by_task, bank_rows)
    schema = feature_schema()
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    for split in BANK_SPLITS:
        split_rows = [row for row in feature_rows if row.get("split") == split]
        write_csv(output_dir / f"{split}_features.csv", split_rows)
    write_csv(output_dir / "action_training_table.csv", action_feature_rows)

    primary_by_task = {str(row["task_id"]): row for row in bank_rows if row.get("action") == "KEEP_PRIMARY"}
    validation_rows = [row for row in action_feature_rows if row.get("split") == "validation"]
    validation_task_ids = sorted({str(row["task_id"]) for row in validation_rows})
    calibration_task_ids = sorted({str(row["task_id"]) for row in action_feature_rows if row.get("split") == "calibration"})

    controller_results: List[JsonDict] = []
    model_payload: Dict[str, Any] = {}
    for feature_variant in FEATURE_VARIANTS:
        feature_cols = schema["feature_variants"][feature_variant]
        models = train_models(action_feature_rows, feature_cols)
        model_payload[feature_variant] = {name: model.to_json() for name, model in models.items()}
        for policy_variant in POLICY_VARIANTS:
            calibration = calibrate_threshold(
                action_feature_rows,
                models,
                policy_variant=policy_variant,
                primary_by_task=primary_by_task,
            )
            threshold = float(calibration["selected_operating_point"]["score_threshold"])
            selected = select_actions(validation_rows, models, policy_variant=policy_variant, score_threshold=threshold)
            metrics = evaluate_selection(selected, primary_by_task)
            short_bucket = (metrics.get("bucket_summary") or {}).get("<=8", {})
            long_bucket_wins = sum(
                int(((metrics.get("bucket_summary") or {}).get(bucket) or {}).get("wins_vs_primary") or 0)
                for bucket in ("17-24", "25+")
            )
            long_bucket_losses = sum(
                int(((metrics.get("bucket_summary") or {}).get(bucket) or {}).get("losses_vs_primary") or 0)
                for bucket in ("17-24", "25+")
            )
            controller_results.append(
                {
                    "model_family": "logistic",
                    "feature_variant": feature_variant,
                    "policy_variant": policy_variant,
                    "score_threshold": threshold,
                    "calibration_harm_upper95": calibration["selected_operating_point"].get("intervention_harm_upper95"),
                    "calibration_intervention_count": calibration["selected_operating_point"].get("intervention_count"),
                    "calibration_failed_harm_budget": calibration["selected_operating_point"].get("calibration_failed_harm_budget"),
                    "validation_short_bucket_wins": short_bucket.get("wins_vs_primary"),
                    "validation_short_bucket_losses": short_bucket.get("losses_vs_primary"),
                    "validation_long_bucket_wins": long_bucket_wins,
                    "validation_long_bucket_losses": long_bucket_losses,
                    **{f"validation_{key}": value for key, value in metrics.items() if key != "bucket_summary"},
                    "validation_grouped_bootstrap_ci": grouped_bootstrap_ci(selected),
                }
            )
            (output_dir / f"risk_curve_{feature_variant}_{policy_variant}.json").write_text(
                json.dumps(calibration["risk_coverage_curve"], ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    write_csv(output_dir / "controller_validation_results.csv", controller_results)
    (output_dir / "model_coefficients.json").write_text(json.dumps(model_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    baseline_rows: List[JsonDict] = []
    for name, path in [
        ("control", args.control_results),
        ("midcons", args.midcons_results),
        ("route2", args.route2_results),
        ("v6", args.v6_results),
        ("local_cal", args.cal_results),
    ]:
        selected = historical_baseline(path, validation_task_ids)
        primary = {str(row["task_id"]): {"passed": row_passed(primary_by_task[str(row["task_id"])])} for row in selected}
        metrics = evaluate_selection([{**row, "intervened": False} for row in selected], primary) if selected else {}
        baseline_rows.append({"baseline": name, **{f"validation_{key}": value for key, value in metrics.items() if key != "bucket_summary"}})
    for action in ("EXPAND_16", "EXPAND_24", "EXPAND_32", "EXPAND_48"):
        selected = baseline_from_action(action_feature_rows, "validation", action)
        metrics = evaluate_selection(selected, primary_by_task)
        baseline_rows.append({"baseline": f"always_{action.lower()}", **{f"validation_{key}": value for key, value in metrics.items() if key != "bucket_summary"}})
    oracle_selected = []
    by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        by_task[str(row["task_id"])].append(row)
    for task_id, rows in sorted(by_task.items()):
        passed = [row for row in rows if row_passed(row)]
        oracle_selected.append((passed or rows)[0])
    oracle_metrics = evaluate_selection(oracle_selected, primary_by_task)
    baseline_rows.append({"baseline": "oracle_action_bank_upper_bound", **{f"validation_{key}": value for key, value in oracle_metrics.items() if key != "bucket_summary"}})

    v6_row = next((row for row in baseline_rows if row["baseline"] == "v6"), {})
    v6_pass_rate = _to_float(v6_row.get("validation_pass_rate"), 0.0)
    v6_mean_cost = _to_float(v6_row.get("validation_mean_cost_sec"), 999.0)

    def validation_gate_flags(row: Mapping[str, Any]) -> JsonDict:
        controller_pass_rate = _to_float(row.get("validation_pass_rate"), 0.0)
        controller_mean_cost = _to_float(row.get("validation_mean_cost_sec"), 999.0)
        aggregate_not_below_v6 = controller_pass_rate >= v6_pass_rate
        total_pass_improved = controller_pass_rate > v6_pass_rate
        long_bucket_improved = int(row.get("validation_long_bucket_wins") or 0) > int(row.get("validation_long_bucket_losses") or 0)
        same_accuracy_lower_compute = aggregate_not_below_v6 and controller_mean_cost < v6_mean_cost
        calibration_upper = row.get("calibration_harm_upper95")
        calibration_passed = calibration_upper is not None and float(calibration_upper) <= 0.05
        risk_coverage_pareto_improved = (
            aggregate_not_below_v6
            and int(row.get("validation_intervention_count") or 0) > 0
            and int(row.get("validation_wins_vs_primary") or 0) >= int(row.get("validation_losses_vs_primary") or 0)
            and calibration_passed
        )
        short_ok = int(row.get("validation_short_bucket_losses") or 0) <= int(row.get("validation_short_bucket_wins") or 0)
        opportunity_ok = (
            total_pass_improved
            or (long_bucket_improved and aggregate_not_below_v6)
            or same_accuracy_lower_compute
            or risk_coverage_pareto_improved
        )
        return {
            "calibration_harm_budget_passed": calibration_passed,
            "validation_short_bucket_no_net_regression": short_ok,
            "validation_not_below_v6": aggregate_not_below_v6,
            "validation_total_pass_improved": total_pass_improved,
            "validation_long_bucket_improved_and_aggregate_not_down": long_bucket_improved and aggregate_not_below_v6,
            "validation_same_accuracy_lower_compute": same_accuracy_lower_compute,
            "validation_risk_coverage_pareto_improved": risk_coverage_pareto_improved,
            "gate_passed": calibration_passed and short_ok and aggregate_not_below_v6 and opportunity_ok,
        }

    passing_controllers = [row for row in controller_results if validation_gate_flags(row)["gate_passed"]]
    selection_pool = passing_controllers or controller_results
    selected_controller = max(
        selection_pool,
        key=lambda row: (
            _to_float(row.get("validation_pass_rate"), 0.0),
            -_to_float(row.get("calibration_harm_upper95"), 1.0),
            -_to_float(row.get("validation_mean_cost_sec"), 999.0),
        ),
    )
    best_controller = selected_controller
    controller_interventions = int(best_controller.get("validation_intervention_count") or 0)
    simple_selected = select_equal_coverage_simple_gate(action_feature_rows, feature_by_task, "validation", controller_interventions)
    simple_metrics = evaluate_selection(simple_selected, primary_by_task)
    baseline_rows.append({"baseline": "compute_matched_simple_gate_expand32", **{f"validation_{key}": value for key, value in simple_metrics.items() if key != "bucket_summary"}})
    write_csv(output_dir / "validation_baselines.csv", baseline_rows)

    selected_flags = validation_gate_flags(selected_controller)
    controller_pass_rate = _to_float(selected_controller.get("validation_pass_rate"), 0.0)
    controller_mean_cost = _to_float(selected_controller.get("validation_mean_cost_sec"), 999.0)
    validation_gate = {
        "selected_controller": {
            "model_family": selected_controller["model_family"],
            "feature_variant": selected_controller["feature_variant"],
            "policy_variant": selected_controller["policy_variant"],
            "score_threshold": selected_controller["score_threshold"],
        },
        "selection_rule": "best_pre_registered_gate_passing_controller" if passing_controllers else "best_validation_pass_rate_no_gate_passing_controller",
        "calibration_harm_upper95": selected_controller.get("calibration_harm_upper95"),
        "calibration_harm_budget_passed": selected_flags["calibration_harm_budget_passed"],
        "validation_short_bucket_no_net_regression": selected_flags["validation_short_bucket_no_net_regression"],
        "validation_not_below_v6": selected_flags["validation_not_below_v6"],
        "validation_v6_pass_rate": v6_pass_rate,
        "validation_v6_mean_cost_sec": v6_mean_cost,
        "validation_controller_pass_rate": selected_controller.get("validation_pass_rate"),
        "validation_controller_mean_cost_sec": selected_controller.get("validation_mean_cost_sec"),
        "validation_total_pass_improved": selected_flags["validation_total_pass_improved"],
        "validation_long_bucket_improved_and_aggregate_not_down": selected_flags["validation_long_bucket_improved_and_aggregate_not_down"],
        "validation_same_accuracy_lower_compute": selected_flags["validation_same_accuracy_lower_compute"],
        "validation_risk_coverage_pareto_improved": selected_flags["validation_risk_coverage_pareto_improved"],
    }
    validation_gate["gate_passed"] = selected_flags["gate_passed"]
    validation_gate["test_decision"] = "authorized_once" if validation_gate["gate_passed"] else "sealed"
    if not validation_gate["gate_passed"]:
        validation_gate["failure_message"] = "VALIDATION FAILURE — TEST REMAINS SEALED"

    summary = {
        "mode": "frozen_canvas_controller_validation",
        "branch": current_branch(),
        "commit": current_commit(),
        "action_bank_dir": args.bank_dir,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "controller_variants": len(controller_results),
        "validation_rows": len(validation_task_ids),
        "calibration_rows": len(calibration_task_ids),
        "validation_gate": validation_gate,
        "test_status": "sealed" if not validation_gate["gate_passed"] else "authorized_once_not_run_by_this_stage",
        "test_evaluation_count": 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(render_controller_report(summary, best_controller, baseline_rows), encoding="utf-8")
    run_manifest = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_at": started_at,
        "execution_status": "completed",
        "output_dir": str(output_dir),
        "bank_dir": args.bank_dir,
        "validation_gate": validation_gate,
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "gate_passed": validation_gate["gate_passed"]}, ensure_ascii=False, indent=2))


def render_controller_report(summary: Mapping[str, Any], best_controller: Mapping[str, Any], baselines: Sequence[Mapping[str, Any]]) -> str:
    gate = summary["validation_gate"]
    lines = [
        "# Frozen Canvas Controller Validation",
        "",
        f"final_validation_decision: `{gate.get('test_decision')}`",
        "",
        "## Selected Controller",
        "",
        f"- model: `logistic`",
        f"- feature_variant: `{best_controller.get('feature_variant')}`",
        f"- policy_variant: `{best_controller.get('policy_variant')}`",
        f"- score_threshold: `{best_controller.get('score_threshold')}`",
        f"- calibration_harm_upper95: `{best_controller.get('calibration_harm_upper95')}`",
        f"- validation_pass_rate: `{best_controller.get('validation_pass_rate')}`",
        f"- validation wins/losses vs primary: `{best_controller.get('validation_wins_vs_primary')}/{best_controller.get('validation_losses_vs_primary')}`",
        f"- validation_intervention_coverage: `{best_controller.get('validation_intervention_coverage')}`",
        "",
        "## Validation Gate",
        "",
        f"- calibration_harm_budget_passed: `{gate.get('calibration_harm_budget_passed')}`",
        f"- validation_short_bucket_no_net_regression: `{gate.get('validation_short_bucket_no_net_regression')}`",
        f"- validation_not_below_v6: `{gate.get('validation_not_below_v6')}`",
        f"- gate_passed: `{gate.get('gate_passed')}`",
    ]
    if gate.get("failure_message"):
        lines.append(f"- failure_message: `{gate.get('failure_message')}`")
    lines.extend(["", "## Validation Baselines", "", "| Baseline | Pass Rate | Wins | Losses | Coverage | Mean Cost |", "|---|---:|---:|---:|---:|---:|"])
    for row in baselines:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                row.get("baseline"),
                row.get("validation_pass_rate"),
                row.get("validation_wins_vs_primary"),
                row.get("validation_losses_vs_primary"),
                row.get("validation_intervention_coverage"),
                row.get("validation_mean_cost_sec"),
            )
        )
    lines.extend(
        [
            "",
            "Test remains sealed unless the validation gate is passed and the lock is explicitly advanced to one-time authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def execute_frozen_test(args: argparse.Namespace) -> None:
    if not args.validation_dir:
        raise ValueError("--validation-dir is required for --mode frozen-test")
    if not args.test_bank_dir:
        raise ValueError("--test-bank-dir is required for --mode frozen-test")
    if not args.test_lock:
        raise ValueError("--test-lock is required for --mode frozen-test")
    started_at = datetime.now().isoformat(timespec="seconds")
    output_dir = ensure_output_dir("frozen_controller_test", args.timestamp, resume_dir=args.output_dir)
    validation_dir = Path(args.validation_dir)
    validation_summary = json.loads((validation_dir / "summary.json").read_text(encoding="utf-8"))
    gate = validation_summary.get("validation_gate") or {}
    if not gate.get("gate_passed"):
        raise RuntimeError("validation gate did not pass; frozen test remains sealed")
    lock_path = Path(args.test_lock)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("test_status") not in {"sealed", "authorized_once"} or int(lock.get("test_evaluation_count") or 0) != 0:
        raise RuntimeError(f"test lock is not eligible for one-time evaluation: {lock.get('test_status')} count={lock.get('test_evaluation_count')}")

    chosen = gate["selected_controller"]
    feature_variant = str(chosen["feature_variant"])
    policy_variant = str(chosen["policy_variant"])
    threshold = float(chosen["score_threshold"])
    model_payload = json.loads((validation_dir / "model_coefficients.json").read_text(encoding="utf-8"))[feature_variant]
    models = {name: LogisticModel.from_json(payload) for name, payload in model_payload.items()}

    bank_rows = load_existing_bank_rows(Path(args.test_bank_dir) / "action_bank.jsonl")
    if not bank_rows:
        raise ValueError(f"no test bank rows found in {args.test_bank_dir}")
    split_assignment = load_split_assignments(Path(args.split_dir))
    primary_rows = rows_by_task(load_jsonl(args.primary_results))
    route2_rows = rows_by_task(load_jsonl(args.route2_results))
    feature_rows = build_feature_rows(split_assignment=split_assignment, primary_rows=primary_rows, route2_rows=route2_rows)
    feature_by_task = {str(row["task_id"]): row for row in feature_rows}
    test_action_rows = make_action_feature_rows(feature_by_task, bank_rows)
    test_action_rows = [row for row in test_action_rows if row.get("split") == "test"]
    primary_by_task = {str(row["task_id"]): row for row in bank_rows if row.get("action") == "KEEP_PRIMARY"}

    lock["test_status"] = "authorized_once"
    lock["authorized_at_cst"] = datetime.now().isoformat(timespec="seconds")
    lock["authorized_by_validation_dir"] = str(validation_dir)
    lock["frozen_controller"] = chosen
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    selected = select_actions(test_action_rows, models, policy_variant=policy_variant, score_threshold=threshold)
    metrics = evaluate_selection(selected, primary_by_task)
    summary = {
        "mode": "frozen_controller_test",
        "branch": current_branch(),
        "commit": current_commit(),
        "validation_dir": str(validation_dir),
        "test_bank_dir": args.test_bank_dir,
        "frozen_controller": chosen,
        "test_metrics": metrics,
        "test_row_count": len(selected),
        "test_evaluation_count": 1,
        "test_status": "closed",
    }
    write_csv(output_dir / "test_selected_actions.csv", selected)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(render_frozen_test_report(summary), encoding="utf-8")
    run_manifest = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_at": started_at,
        "execution_status": "completed",
        "output_dir": str(output_dir),
        "validation_dir": str(validation_dir),
        "test_bank_dir": args.test_bank_dir,
        "test_lock": str(lock_path),
        "test_metrics": metrics,
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    lock["test_status"] = "closed"
    lock["closed_at_cst"] = datetime.now().isoformat(timespec="seconds")
    lock["test_evaluation_count"] = 1
    lock["test_result_dir"] = str(output_dir)
    lock["test_metrics_summary"] = metrics
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "test_status": "closed", "pass_rate": metrics.get("pass_rate")}, ensure_ascii=False, indent=2))


def render_frozen_test_report(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("test_metrics") or {}
    lines = [
        "# Frozen Controller Test",
        "",
        "This is the one-time frozen test evaluation authorized by the validation gate.",
        "",
        f"- test_row_count: `{summary.get('test_row_count')}`",
        f"- pass_rate: `{metrics.get('pass_rate')}`",
        f"- wins/losses vs primary: `{metrics.get('wins_vs_primary')}/{metrics.get('losses_vs_primary')}`",
        f"- intervention_coverage: `{metrics.get('intervention_coverage')}`",
        f"- intervention_harm_rate: `{metrics.get('intervention_harm_rate')}`",
        f"- intervention_harm_upper95: `{metrics.get('intervention_harm_upper95')}`",
        f"- mean_cost_sec: `{metrics.get('mean_cost_sec')}`",
        "",
    ]
    return "\n".join(lines)


def execute_merge_bank(args: argparse.Namespace) -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    output_dir = ensure_output_dir("controller_action_bank", args.timestamp, resume_dir=args.output_dir)
    merge_dirs = [Path(item.strip()) for item in str(args.merge_dirs or "").split(",") if item.strip()]
    if not merge_dirs:
        raise ValueError("--merge-dirs is required for --mode merge-bank")
    rows: List[JsonDict] = []
    source_counts: Dict[str, int] = {}
    for directory in merge_dirs:
        shard_rows = load_existing_bank_rows(directory / "action_bank.jsonl")
        source_counts[str(directory)] = len(shard_rows)
        rows.extend(shard_rows)
    dedup: Dict[Tuple[str, str], JsonDict] = {}
    duplicates: List[JsonDict] = []
    for row in rows:
        key = (str(row["task_id"]), str(row["action"]))
        if key in dedup:
            duplicates.append({"task_id": key[0], "action": key[1]})
        dedup[key] = row
    merged = [dedup[key] for key in sorted(dedup)]
    split_assignment = load_split_assignments(Path(args.split_dir))
    wanted_splits = {item.strip() for item in args.bank_splits.split(",") if item.strip()}
    expected_task_ids = sorted(task_id for task_id, split in split_assignment.items() if split in wanted_splits)
    expected_keys = {(task_id, action) for task_id in expected_task_ids for action in ACTION_ORDER}
    observed_keys = set(dedup)
    coverage = {
        "expected_task_count": len(expected_task_ids),
        "expected_action_rows": len(expected_keys),
        "observed_task_count": len({task_id for task_id, _ in observed_keys}),
        "observed_action_rows": len(observed_keys),
        "missing_action_rows": len(expected_keys - observed_keys),
        "extra_action_rows": len(observed_keys - expected_keys),
        "duplicate_action_rows": len(duplicates),
        "missing_examples": [
            {"task_id": task_id, "action": action}
            for task_id, action in sorted(expected_keys - observed_keys)[:20]
        ],
        "duplicate_examples": duplicates[:20],
    }
    write_jsonl(output_dir / "action_bank.jsonl", merged)
    write_csv(output_dir / "action_bank.csv", merged)
    (output_dir / "action_schema.json").write_text(json.dumps(action_schema(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary = summarize_bank(merged)
    summary["merge_coverage"] = coverage
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(render_bank_report(summary), encoding="utf-8")
    manifest = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_at": started_at,
        "execution_status": "completed" if coverage["missing_action_rows"] == 0 and coverage["duplicate_action_rows"] == 0 else "completed_with_coverage_warning",
        "output_dir": str(output_dir),
        "merge_dirs": [str(path) for path in merge_dirs],
        "source_counts": source_counts,
        "coverage": coverage,
        "summary": summary,
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "coverage": coverage}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and validate a risk-controlled canvas controller.")
    parser.add_argument("--mode", choices=["freeze", "bank", "merge-bank", "controller", "frozen-test"], required=True)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split-dir", default="analysis_outputs/grouped_split_20260702_accel2")
    parser.add_argument("--protocol-doc", default="docs/paper_agent/frozen_controller_protocol.zh.md")
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-subset", default="HumanEval-SingleLineInfilling")
    parser.add_argument("--lcas-policy", default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    parser.add_argument("--bank-splits", default="train,calibration,validation")
    parser.add_argument("--primary-results", default=PRIMARY_RESULTS)
    parser.add_argument("--control-results", default=CONTROL_RESULTS)
    parser.add_argument("--midcons-results", default=MIDCONS_RESULTS)
    parser.add_argument("--route2-results", default=ROUTE2_RESULTS)
    parser.add_argument("--v6-results", default=V6_RESULTS)
    parser.add_argument("--cal-results", default=CAL_RESULTS)
    parser.add_argument("--bank-dir", default=None)
    parser.add_argument("--merge-dirs", default=None)
    parser.add_argument("--validation-dir", default=None)
    parser.add_argument("--test-bank-dir", default=None)
    parser.add_argument("--test-lock", default=None)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "freeze":
        execute_freeze(args)
    elif args.mode == "bank":
        execute_bank(args)
    elif args.mode == "merge-bank":
        execute_merge_bank(args)
    elif args.mode == "controller":
        if not args.bank_dir:
            raise SystemExit("--bank-dir is required for --mode controller")
        execute_controller(args)
    elif args.mode == "frozen-test":
        execute_frozen_test(args)


if __name__ == "__main__":
    main()
