#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import shlex
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl, metric, oracle_bucket


JsonDict = Dict[str, Any]

DEFAULT_BASELINE = "/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl"
DEFAULT_ROUTE2 = "/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl"
DEFAULT_OUTPUT_DIR = "analysis_outputs"

ACTION_IDS = ("A_primary", "B_route2_len32", "C_oracle_sufficient", "D_oracle_sufficient_steps96")
PILOT_VERDICTS = {
    "pilot_invalid_reproducibility_failure",
    "no_new_candidate_found",
    "canvas_ceiling_signal",
    "schedule_ceiling_signal",
    "mixed_ceiling_signal",
    "positive_control_only",
    "needs_more_cases",
}


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    deployability: str
    canvas_rule: str
    steps: int
    description: str


ACTION_SPECS: Tuple[ActionSpec, ...] = (
    ActionSpec(
        action_id="A_primary",
        deployability="deployable_current_primary",
        canvas_rule="use_midcons_selected_length",
        steps=64,
        description="Current midcons primary output from the baseline run.",
    ),
    ActionSpec(
        action_id="B_route2_len32",
        deployability="deployable_current_route2",
        canvas_rule="use_existing_route2_precision_len32_output",
        steps=64,
        description="Current Route2 precision_top1_conf rescue len32 output when triggered; otherwise primary.",
    ),
    ActionSpec(
        action_id="C_oracle_sufficient",
        deployability="offline_ceiling_only_oracle_canvas",
        canvas_rule="max(primary_len, route2_len, oracle_len)",
        steps=64,
        description="Offline ceiling action: use a canvas guaranteed not shorter than oracle/reference length.",
    ),
    ActionSpec(
        action_id="D_oracle_sufficient_steps96",
        deployability="offline_ceiling_only_oracle_canvas",
        canvas_rule="max(primary_len, route2_len, oracle_len)",
        steps=96,
        description="Same oracle-sufficient canvas as C, but with a fixed 96-step schedule.",
    ),
)


def _to_bool(value: Any) -> bool:
    return bool(value)


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def rows_by_task(rows: Iterable[Mapping[str, Any]], *, name: str) -> Dict[str, Mapping[str, Any]]:
    by_task: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id is None:
            raise ValueError(f"missing task_id in {name}")
        by_task[str(task_id)] = row
    return by_task


def pairwise(candidate_passed: bool, baseline_passed: bool) -> str:
    if candidate_passed and not baseline_passed:
        return "win"
    if (not candidate_passed) and baseline_passed:
        return "loss"
    if candidate_passed and baseline_passed:
        return "tie_pass"
    return "tie_fail"


def parse_task_id_group(task_id: str) -> str:
    parts = task_id.split("/")
    if "HumanEval" in parts:
        idx = parts.index("HumanEval")
        if idx + 1 < len(parts):
            return f"HumanEval/{parts[idx + 1]}"
    return "/".join(parts[:-1]) if "/" in task_id else task_id


def oracle_sufficient_length(
    *,
    primary_len: Optional[int],
    route2_len: Optional[int],
    oracle_len: Optional[int],
    max_length: int,
) -> Optional[int]:
    values = [value for value in (primary_len, route2_len, oracle_len) if value is not None]
    if not values:
        return None
    sufficient = max(values)
    if sufficient > max_length:
        return None
    return sufficient


def canvas_is_oracle_sufficient(canvas_len: Optional[int], oracle_len: Optional[int]) -> Optional[bool]:
    if canvas_len is None or oracle_len is None:
        return None
    return int(canvas_len) >= int(oracle_len)


def classify_case_pool(baseline_row: Mapping[str, Any], route2_row: Mapping[str, Any]) -> Optional[str]:
    oracle_len = _to_int(metric(baseline_row, "oracle_mask_length", metric(route2_row, "oracle_mask_length")))
    if oracle_len is None:
        return None
    baseline_passed = _to_bool(metric(baseline_row, "passed", False))
    route2_passed = _to_bool(metric(route2_row, "passed", False))
    route2_triggered = _to_bool(metric(route2_row, "route2_trace_rescue_triggered", False))
    true_long = oracle_len >= 17

    if (not baseline_passed) and route2_passed:
        return "positive_control_rescued"
    if true_long and (not baseline_passed) and route2_triggered and (not route2_passed):
        return "triggered_failed_long"
    if true_long and (not baseline_passed) and (not route2_triggered):
        return "missed_failed_long"
    return None


def build_case_manifest(
    baseline_rows: Sequence[Mapping[str, Any]],
    route2_rows: Sequence[Mapping[str, Any]],
    *,
    max_cases_per_pool: int,
    task_ids: Optional[Sequence[str]] = None,
    max_canvas_length: int = 64,
) -> List[JsonDict]:
    baseline = rows_by_task(baseline_rows, name="baseline")
    route2 = rows_by_task(route2_rows, name="route2")
    requested = list(task_ids or [])
    common_ids = [task_id for task_id in baseline if task_id in route2]
    if requested:
        missing = [task_id for task_id in requested if task_id not in baseline or task_id not in route2]
        if missing:
            raise ValueError(f"requested task ids missing from inputs: {missing}")
        common_ids = requested

    pools: Dict[str, List[JsonDict]] = defaultdict(list)
    rows_by_id: Dict[str, JsonDict] = {}
    iteration_ids = common_ids if requested else sorted(common_ids)
    for task_id in iteration_ids:
        baseline_row = baseline[task_id]
        route2_row = route2[task_id]
        pool = classify_case_pool(baseline_row, route2_row)
        if pool is None and not requested:
            continue
        if pool is None:
            pool = "manual_requested"

        oracle_len = _to_int(metric(baseline_row, "oracle_mask_length", metric(route2_row, "oracle_mask_length")))
        primary_len = _to_int(metric(baseline_row, "selected_mask_length"))
        route2_len = _to_int(metric(route2_row, "route2_rescue_length", metric(route2_row, "selected_mask_length")))
        sufficient_len = oracle_sufficient_length(
            primary_len=primary_len,
            route2_len=route2_len,
            oracle_len=oracle_len,
            max_length=max_canvas_length,
        )
        oracle_exceeds_max_canvas = oracle_len is not None and oracle_len > max_canvas_length
        baseline_passed = _to_bool(metric(baseline_row, "passed", False))
        route2_passed = _to_bool(metric(route2_row, "passed", False))
        route2_triggered = _to_bool(metric(route2_row, "route2_trace_rescue_triggered", False))
        case_row = {
            "task_id": task_id,
            "task_group": parse_task_id_group(task_id),
            "case_pool": pool,
            "oracle_length": oracle_len,
            "oracle_bucket": oracle_bucket(oracle_len),
            "primary_selected_length": primary_len,
            "route2_triggered": route2_triggered,
            "route2_rescue_length": route2_len,
            "oracle_sufficient_length": sufficient_len,
            "max_canvas_length": max_canvas_length,
            "oracle_exceeds_max_canvas": oracle_exceeds_max_canvas,
            "canvas_is_oracle_sufficient": canvas_is_oracle_sufficient(sufficient_len, oracle_len),
            "baseline_passed": baseline_passed,
            "route2_passed": route2_passed,
            "route2_pairwise_vs_primary": pairwise(route2_passed, baseline_passed),
            "route2_rescue_len_ge_oracle": (
                None if route2_len is None or oracle_len is None else route2_len >= oracle_len
            ),
        }
        pools[pool].append(case_row)
        rows_by_id[task_id] = case_row

    selected: List[JsonDict] = []
    if requested:
        return [rows_by_id[task_id] for task_id in common_ids if task_id in rows_by_id]

    for pool in ("positive_control_rescued", "triggered_failed_long", "missed_failed_long", "manual_requested"):
        candidates = sorted(
            pools.get(pool, []),
            key=lambda row: (
                0 if (_to_int(row.get("oracle_length")) or 0) >= 17 else 1,
                -(_to_int(row.get("oracle_length")) or 0),
                str(row.get("task_id")),
            ),
        )
        selected.extend(candidates[: max(0, max_cases_per_pool)])
    return selected


def build_action_manifest(case_manifest: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for case in case_manifest:
        for spec in ACTION_SPECS:
            if spec.action_id == "A_primary":
                planned_canvas = case.get("primary_selected_length")
            elif spec.action_id == "B_route2_len32":
                planned_canvas = case.get("route2_rescue_length") or case.get("primary_selected_length")
            else:
                planned_canvas = case.get("oracle_sufficient_length")
            oracle_len = _to_int(case.get("oracle_length"))
            rows.append(
                {
                    "task_id": case["task_id"],
                    "case_pool": case["case_pool"],
                    "oracle_bucket": case["oracle_bucket"],
                    "action_id": spec.action_id,
                    "deployability": spec.deployability,
                    "canvas_rule": spec.canvas_rule,
                    "planned_canvas_length": planned_canvas,
                    "oracle_exceeds_max_canvas": case.get("oracle_exceeds_max_canvas"),
                    "canvas_is_oracle_sufficient": canvas_is_oracle_sufficient(_to_int(planned_canvas), oracle_len),
                    "planned_total_steps": spec.steps,
                    "description": spec.description,
                }
            )
    return rows


def summarize_manifest(case_manifest: Sequence[Mapping[str, Any]], action_manifest: Sequence[Mapping[str, Any]]) -> JsonDict:
    case_pool_counts = Counter(str(row.get("case_pool")) for row in case_manifest)
    bucket_counts = Counter(str(row.get("oracle_bucket")) for row in case_manifest)
    action_counts = Counter(str(row.get("action_id")) for row in action_manifest)
    route2_triggered = sum(1 for row in case_manifest if row.get("route2_triggered"))
    route2_rescue_ge_oracle = sum(1 for row in case_manifest if row.get("route2_rescue_len_ge_oracle") is True)
    known_route2_rescue_lengths = sum(1 for row in case_manifest if row.get("route2_rescue_len_ge_oracle") is not None)
    planned_forward_units = sum(int(row.get("planned_total_steps") or 0) for row in action_manifest)
    return {
        "mode": "dry_run_manifest",
        "case_count": len(case_manifest),
        "action_count": len(action_manifest),
        "case_pool_counts": dict(sorted(case_pool_counts.items())),
        "oracle_bucket_counts": dict(sorted(bucket_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "route2_triggered_case_count": route2_triggered,
        "route2_rescue_len_ge_oracle_rate_in_manifest": _safe_rate(
            route2_rescue_ge_oracle,
            known_route2_rescue_lengths,
        ),
        "planned_decode_step_units": planned_forward_units,
        "oracle_used_for_actions": ["C_oracle_sufficient", "D_oracle_sufficient_steps96"],
        "deployable_actions": ["A_primary", "B_route2_len32"],
        "not_deployable_ceiling_actions": ["C_oracle_sufficient", "D_oracle_sufficient_steps96"],
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "value"], lineterminator="\n")
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"key": key, "value": json.dumps(value, ensure_ascii=False)})


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_report(summary: Mapping[str, Any], case_manifest: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# True-Long Action-Ceiling Matrix Dry Run",
        "",
        "Decision: `ready_for_small_pilot_after_researcher_approval`",
        "",
        "## Pre-Registration",
        "",
        "- hypothesis: true-long failures are jointly limited by canvas adequacy, rescue generation quality, selector quality, and trigger recall.",
        "- intervention: compare current primary, current Route2 rescue, oracle-sufficient canvas, and oracle-sufficient canvas with a fixed 96-step schedule.",
        "- control: current `midcons` primary output and current Route2 precision len32 output.",
        "- expected outcomes: C/D reveal whether correct candidates exist when canvas is sufficient; A/B show current deployable gap.",
        "- decision rule: if C/D do not create correct candidates in triggered and missed long pools, stop blind true-long length-control and pivot to backbone/generation limitation; if C/D create candidates but B misses them, focus selector/action selection; if missed pool has C/D wins, focus trigger recall.",
        "- stop condition: no full run until a pilot report shows positive ceiling or clear negative anatomy on the preselected small case set.",
        "- leakage guard: C/D use oracle length for offline ceiling only and must not be described as deployable.",
        "",
        "## Dry-Run Summary",
        "",
        f"- case_count: `{summary['case_count']}`",
        f"- action_count: `{summary['action_count']}`",
        f"- planned_decode_step_units: `{summary['planned_decode_step_units']}`",
        f"- case_pool_counts: `{summary['case_pool_counts']}`",
        f"- oracle_bucket_counts: `{summary['oracle_bucket_counts']}`",
        f"- route2_rescue_len_ge_oracle_rate_in_manifest: `{summary['route2_rescue_len_ge_oracle_rate_in_manifest']}`",
        "",
        "## Planned Actions",
        "",
        "| Action | Deployability | Canvas | Steps |",
        "|---|---|---|---:|",
    ]
    for spec in ACTION_SPECS:
        lines.append(f"| `{spec.action_id}` | `{spec.deployability}` | `{spec.canvas_rule}` | `{spec.steps}` |")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Pool | Task | Bucket | Oracle | Primary Len | Route2 Triggered | Route2 Len | Ceiling Len |",
            "|---|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in case_manifest:
        lines.append(
            "| {pool} | `{task}` | `{bucket}` | {oracle} | {primary} | `{triggered}` | {route2} | {ceiling} |".format(
                pool=row.get("case_pool"),
                task=row.get("task_id"),
                bucket=row.get("oracle_bucket"),
                oracle="" if row.get("oracle_length") is None else row.get("oracle_length"),
                primary="" if row.get("primary_selected_length") is None else row.get("primary_selected_length"),
                triggered=row.get("route2_triggered"),
                route2="" if row.get("route2_rescue_length") is None else row.get("route2_rescue_length"),
                ceiling="" if row.get("oracle_sufficient_length") is None else row.get("oracle_sufficient_length"),
            )
        )

    lines.extend(
        [
            "",
            "## Next Command",
            "",
            "Run `--execute-pilot` only for this manifest or an explicitly reviewed `--task-ids-csv` subset. Do not use this scaffold for a full 1033-row run.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_output_dir(base_dir: str, timestamp: Optional[str] = None) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"action_ceiling_{stamp}"
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def write_dry_run_outputs(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    case_manifest: Sequence[Mapping[str, Any]],
    action_manifest: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "case_manifest.csv", case_manifest)
    write_csv(output_dir / "action_manifest.csv", action_manifest)
    (output_dir / "report.md").write_text(render_report(summary, case_manifest), encoding="utf-8")


def parse_task_ids_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    task_ids = [item.strip() for item in value.split(",") if item.strip()]
    return task_ids or None


def build_dry_run(
    *,
    baseline_results: str,
    route2_results: str,
    max_cases_per_pool: int,
    task_ids: Optional[Sequence[str]],
    max_canvas_length: int,
) -> Tuple[List[JsonDict], List[JsonDict], JsonDict]:
    baseline_rows = load_jsonl(baseline_results)
    route2_rows = load_jsonl(route2_results)
    case_manifest = build_case_manifest(
        baseline_rows,
        route2_rows,
        max_cases_per_pool=max_cases_per_pool,
        task_ids=task_ids,
        max_canvas_length=max_canvas_length,
    )
    action_manifest = build_action_manifest(case_manifest)
    summary = summarize_manifest(case_manifest, action_manifest)
    return case_manifest, action_manifest, summary


def sha256_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_task_seed(global_seed: int, task_id: str, experimental_seed: int = 0) -> int:
    payload = f"{int(global_seed)}\n{int(experimental_seed)}\n{task_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31 - 1)


def reset_action_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def git_capture(*parts: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *parts],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"git_capture_failed: {type(exc).__name__}: {exc}"


def current_commit() -> str:
    return git_capture("rev-parse", "HEAD")


def current_branch() -> str:
    return git_capture("branch", "--show-current")


def resolve_checkpoint_path(model_path: str) -> Optional[str]:
    local_path = Path(model_path).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if getattr(repo, "repo_id", None) == model_path:
                repo_path = getattr(repo, "repo_path", None)
                return None if repo_path is None else str(repo_path)
    except Exception:
        return None
    return None


def environment_info(model_path: str) -> JsonDict:
    info: JsonDict = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_name": model_path,
        "checkpoint_resolved_local_path": resolve_checkpoint_path(model_path),
    }
    try:
        import torch

        info.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "gpu_models": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
            }
        )
    except Exception as exc:
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return info


def build_run_manifest(
    args: argparse.Namespace,
    case_manifest: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    execution_status: str,
    started_at: str,
    wall_clock_sec: Optional[float] = None,
    determinism_check: Optional[Mapping[str, Any]] = None,
    experimental_seeds: Optional[Sequence[int]] = None,
    verdict: Optional[str] = None,
    failure_traceback: Optional[str] = None,
) -> JsonDict:
    return {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": [sys.executable, *sys.argv],
        "timestamp": args.timestamp,
        "started_at": started_at,
        "environment": environment_info(args.model_path),
        "dataset": {
            "name": "loubnabnl/humaneval_infilling",
            "subset": args.dataset_subset,
            "split": args.split,
        },
        "task_ids": [str(row["task_id"]) for row in case_manifest],
        "max_pilot_cases": args.max_pilot_cases,
        "seed_protocol": {
            "global_seed": args.seed,
            "stable_seed_formula": "sha256(global_seed, experimental_seed, task_id) mod 2^31-1",
            "reset_before_each_action": ["python.random", "numpy", "torch", "torch.cuda"],
            "same_task_seed_for_actions": True,
            "experimental_seeds": list(experimental_seeds or []),
        },
        "action_definitions": [spec.__dict__ for spec in ACTION_SPECS],
        "historical_results": {
            "baseline_midcons": args.baseline_results,
            "route2": args.route2_results,
        },
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
        "output_dir": str(output_dir),
        "execution_status": execution_status,
        "wall_clock_sec": wall_clock_sec,
        "determinism_check": determinism_check,
        "verdict": verdict,
        "failure_traceback": failure_traceback,
    }


def write_run_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def verification_error_type(result: Mapping[str, Any]) -> Optional[str]:
    verification = result.get("verification") or {}
    for tier in ("tier1_parse_compile", "tier2_smoke_exec", "tier3_unit_tests"):
        tier_result = verification.get(tier) or {}
        if tier_result and not tier_result.get("passed", False):
            return tier_result.get("error_type") or tier
    return None


def verification_error_message(result: Mapping[str, Any]) -> Optional[str]:
    verification = result.get("verification") or {}
    for tier in ("tier1_parse_compile", "tier2_smoke_exec", "tier3_unit_tests"):
        tier_result = verification.get(tier) or {}
        if tier_result and not tier_result.get("passed", False):
            message = tier_result.get("error_message") or tier_result.get("traceback_text")
            if message is not None:
                return str(message)
    diagnostics = result.get("diagnostics") or {}
    return diagnostics.get("final_full_code_compile_error") or diagnostics.get("final_full_code_parse_error")


def compile_passed(result: Mapping[str, Any]) -> Optional[bool]:
    diagnostics = result.get("diagnostics") or {}
    if "final_full_code_compile_passed" in diagnostics:
        return bool(diagnostics["final_full_code_compile_passed"])
    tier1 = (result.get("verification") or {}).get("tier1_parse_compile")
    if tier1:
        return bool(tier1.get("passed"))
    return None


def historical_passed_for_action(
    action_id: str,
    task_id: str,
    baseline_by_task: Mapping[str, Mapping[str, Any]],
    route2_by_task: Mapping[str, Mapping[str, Any]],
) -> Optional[bool]:
    if action_id == "A_primary":
        row = baseline_by_task.get(task_id)
    elif action_id == "B_route2_len32":
        row = route2_by_task.get(task_id)
    else:
        row = None
    if row is None:
        return None
    return _to_bool(metric(row, "passed", False))


def collect_result_record(
    *,
    case: Mapping[str, Any],
    action_id: str,
    result: Mapping[str, Any],
    seed: int,
    experimental_seed: int,
    source_commit: str,
    action_was_executed: bool,
    execution_note: str,
    baseline_by_task: Mapping[str, Mapping[str, Any]],
    route2_by_task: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    task_id = str(case["task_id"])
    metrics = result.get("metrics") or {}
    generated_text = result.get("middle_text")
    if generated_text is None:
        generated_text = (result.get("diagnostics") or {}).get("decoded_middle_text")
    passed = bool(metrics.get("passed"))
    historical_passed = historical_passed_for_action(action_id, task_id, baseline_by_task, route2_by_task)
    error_message = verification_error_message(result)
    selected_len = _to_int(metrics.get("selected_mask_length", metrics.get("mask_length")))
    oracle_len = _to_int(metrics.get("oracle_mask_length", case.get("oracle_length")))
    return {
        "task_id": task_id,
        "task_group": case.get("task_group"),
        "case_pool": case.get("case_pool"),
        "action_id": action_id,
        "deployability": next(spec.deployability for spec in ACTION_SPECS if spec.action_id == action_id),
        "seed": int(seed),
        "experimental_seed": int(experimental_seed),
        "passed": passed,
        "generated_text": generated_text,
        "generated_text_sha256": sha256_text(generated_text),
        "generated_text_len": None if generated_text is None else len(generated_text),
        "code_sha256": sha256_text(result.get("code")),
        "selected_mask_length": selected_len,
        "actual_canvas_length": _to_int(metrics.get("mask_length", selected_len)),
        "oracle_mask_length": oracle_len,
        "canvas_is_oracle_sufficient": canvas_is_oracle_sufficient(selected_len, oracle_len),
        "oracle_exceeds_max_canvas": case.get("oracle_exceeds_max_canvas"),
        "decode_steps": _to_int(metrics.get("total_steps")),
        "effective_steps": _to_int(metrics.get("effective_steps")),
        "total_sec_including_probe": metrics.get("total_sec_including_probe"),
        "error_type": verification_error_type(result),
        "error_message_short": None if error_message is None else str(error_message).replace("\n", " ")[:240],
        "compile_passed": compile_passed(result),
        "historical_passed": historical_passed,
        "replay_matches_historical": None if historical_passed is None else passed == historical_passed,
        "action_was_executed": bool(action_was_executed),
        "execution_note": execution_note,
        "source_commit": source_commit,
        "mask_length_source": metrics.get("mask_length_source"),
        "stopped": metrics.get("stopped"),
        "stop_reason": metrics.get("stop_reason"),
        "route2_triggered_historical": case.get("route2_triggered"),
        "route2_rescue_length_historical": case.get("route2_rescue_length"),
        "verification": result.get("verification"),
        "diagnostics": result.get("diagnostics"),
    }


def invalid_action_record(
    *,
    case: Mapping[str, Any],
    action_id: str,
    seed: int,
    experimental_seed: int,
    source_commit: str,
    reason: str,
) -> JsonDict:
    oracle_len = _to_int(case.get("oracle_length"))
    return {
        "task_id": str(case["task_id"]),
        "task_group": case.get("task_group"),
        "case_pool": case.get("case_pool"),
        "action_id": action_id,
        "deployability": next(spec.deployability for spec in ACTION_SPECS if spec.action_id == action_id),
        "seed": int(seed),
        "experimental_seed": int(experimental_seed),
        "passed": None,
        "generated_text": None,
        "generated_text_sha256": None,
        "generated_text_len": None,
        "code_sha256": None,
        "selected_mask_length": None,
        "actual_canvas_length": None,
        "oracle_mask_length": oracle_len,
        "canvas_is_oracle_sufficient": False if oracle_len is not None else None,
        "oracle_exceeds_max_canvas": case.get("oracle_exceeds_max_canvas"),
        "decode_steps": None,
        "effective_steps": None,
        "total_sec_including_probe": None,
        "error_type": reason,
        "error_message_short": reason,
        "compile_passed": None,
        "historical_passed": None,
        "replay_matches_historical": None,
        "action_was_executed": False,
        "execution_note": reason,
        "source_commit": source_commit,
        "mask_length_source": None,
        "stopped": None,
        "stop_reason": None,
        "route2_triggered_historical": case.get("route2_triggered"),
        "route2_rescue_length_historical": case.get("route2_rescue_length"),
        "verification": None,
        "diagnostics": None,
    }


def compact_result_record(row: Mapping[str, Any]) -> JsonDict:
    excluded = {"generated_text", "verification", "diagnostics"}
    return {key: value for key, value in row.items() if key not in excluded}


def run_single_action(
    *,
    action_id: str,
    case: Mapping[str, Any],
    task: Any,
    tokenizer: Any,
    model: Any,
    cfg: Any,
    settings: Any,
    route2_runner: Any,
    seed: int,
    primary_result: Optional[Mapping[str, Any]],
) -> Tuple[Optional[JsonDict], bool, str]:
    reset_action_seed(seed)
    if action_id == "A_primary":
        action_cfg = copy.deepcopy(cfg)
        action_cfg.decode.total_steps = 64
        return route2_runner.run_primary(task, tokenizer, model, action_cfg, settings), True, "executed_primary"
    if action_id == "B_route2_len32":
        rescue_len = _to_int(case.get("route2_rescue_length"))
        if bool(case.get("route2_triggered")) and rescue_len is not None:
            action_cfg = copy.deepcopy(cfg)
            action_cfg.decode.total_steps = 64
            return (
                route2_runner.run_fixed_rescue(task, tokenizer, model, action_cfg, rescue_len),
                True,
                f"executed_historical_route2_len_{rescue_len}",
            )
        if primary_result is None:
            action_cfg = copy.deepcopy(cfg)
            action_cfg.decode.total_steps = 64
            primary_result = route2_runner.run_primary(task, tokenizer, model, action_cfg, settings)
        return dict(primary_result), False, "route2_not_triggered_historically_reused_primary"
    if action_id in {"C_oracle_sufficient", "D_oracle_sufficient_steps96"}:
        ceiling_len = _to_int(case.get("oracle_sufficient_length"))
        if ceiling_len is None:
            return None, False, "invalid_oracle_ceiling_canvas"
        action_cfg = copy.deepcopy(cfg)
        action_cfg.decode.total_steps = 96 if action_id == "D_oracle_sufficient_steps96" else 64
        return (
            route2_runner.run_fixed_rescue(task, tokenizer, model, action_cfg, ceiling_len),
            True,
            f"executed_oracle_sufficient_len_{ceiling_len}",
        )
    raise ValueError(f"unknown action_id: {action_id}")


def run_determinism_check(
    *,
    case: Mapping[str, Any],
    task: Any,
    tokenizer: Any,
    model: Any,
    cfg: Any,
    settings: Any,
    route2_runner: Any,
    global_seed: int,
) -> JsonDict:
    action_id = "A_primary"
    seed = stable_task_seed(global_seed, str(case["task_id"]), experimental_seed=0)
    first, _, _ = run_single_action(
        action_id=action_id,
        case=case,
        task=task,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        settings=settings,
        route2_runner=route2_runner,
        seed=seed,
        primary_result=None,
    )
    second, _, _ = run_single_action(
        action_id=action_id,
        case=case,
        task=task,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        settings=settings,
        route2_runner=route2_runner,
        seed=seed,
        primary_result=None,
    )
    first_hash = sha256_text((first or {}).get("middle_text"))
    second_hash = sha256_text((second or {}).get("middle_text"))
    first_passed = bool(((first or {}).get("metrics") or {}).get("passed"))
    second_passed = bool(((second or {}).get("metrics") or {}).get("passed"))
    deterministic = first_hash == second_hash and first_passed == second_passed
    return {
        "task_id": str(case["task_id"]),
        "action_id": action_id,
        "seed": seed,
        "first_generated_text_sha256": first_hash,
        "second_generated_text_sha256": second_hash,
        "first_passed": first_passed,
        "second_passed": second_passed,
        "deterministic": deterministic,
    }


def _p95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return ordered[index]


def action_row_by_key(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int, str], Mapping[str, Any]]:
    return {
        (str(row["task_id"]), int(row["experimental_seed"]), str(row["action_id"])): row
        for row in rows
    }


def choose_pilot_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_task_ids: Sequence[str],
    determinism_check: Mapping[str, Any],
) -> str:
    by_key = action_row_by_key(rows)
    missing = []
    seeds = sorted({int(row["experimental_seed"]) for row in rows})
    for task_id in expected_task_ids:
        for seed in seeds:
            for action_id in ACTION_IDS:
                if (task_id, seed, action_id) not in by_key:
                    missing.append((task_id, seed, action_id))
    if missing or not determinism_check:
        return "pilot_invalid_reproducibility_failure"

    positive_mismatch = [
        row
        for row in rows
        if row.get("case_pool") == "positive_control_rescued"
        and row.get("action_id") in {"A_primary", "B_route2_len32"}
        and row.get("replay_matches_historical") is False
    ]
    if positive_mismatch:
        return "pilot_invalid_reproducibility_failure"

    canvas_signal = False
    schedule_signal = False
    any_non_positive_ceiling_candidate = False
    any_ceiling_candidate = False
    for task_id in expected_task_ids:
        for seed in seeds:
            a = by_key.get((task_id, seed, "A_primary"))
            b = by_key.get((task_id, seed, "B_route2_len32"))
            c = by_key.get((task_id, seed, "C_oracle_sufficient"))
            d = by_key.get((task_id, seed, "D_oracle_sufficient_steps96"))
            if not all([a, b, c, d]):
                continue
            c_pass = c.get("passed") is True
            d_pass = d.get("passed") is True
            if c_pass or d_pass:
                any_ceiling_candidate = True
                if a.get("case_pool") != "positive_control_rescued":
                    any_non_positive_ceiling_candidate = True
            if c_pass and b.get("passed") is not True and a.get("case_pool") != "positive_control_rescued":
                canvas_signal = True
            if d_pass and c.get("passed") is not True and a.get("case_pool") != "positive_control_rescued":
                schedule_signal = True

    if canvas_signal and schedule_signal:
        return "mixed_ceiling_signal"
    if schedule_signal:
        return "schedule_ceiling_signal"
    if canvas_signal:
        return "canvas_ceiling_signal"
    if any_non_positive_ceiling_candidate:
        return "needs_more_cases"
    if any_ceiling_candidate:
        return "positive_control_only"
    return "no_new_candidate_found"


def summarize_pilot(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_manifest: Sequence[Mapping[str, Any]],
    determinism_check: Mapping[str, Any],
    experimental_seeds: Sequence[int],
) -> JsonDict:
    expected_task_ids = [str(row["task_id"]) for row in case_manifest]
    verdict = choose_pilot_verdict(rows, expected_task_ids=expected_task_ids, determinism_check=determinism_check)
    by_key = action_row_by_key(rows)
    paired_counts: Counter[str] = Counter()
    action_pass_counts: Dict[str, Counter[str]] = {action_id: Counter() for action_id in ACTION_IDS}
    task_action_table: Dict[str, Dict[str, List[Optional[bool]]]] = {
        task_id: {action_id: [] for action_id in ACTION_IDS} for task_id in expected_task_ids
    }
    for row in rows:
        action_pass_counts[str(row["action_id"])][str(row.get("passed"))] += 1
        task_action_table[str(row["task_id"])][str(row["action_id"])].append(row.get("passed"))
    for task_id in expected_task_ids:
        for seed in experimental_seeds:
            primary = by_key.get((task_id, int(seed), "A_primary"))
            if primary is None:
                continue
            primary_passed = primary.get("passed") is True
            for action_id in ACTION_IDS:
                if action_id == "A_primary":
                    continue
                row = by_key.get((task_id, int(seed), action_id))
                if row is None or row.get("passed") is None:
                    continue
                paired_counts[pairwise(row.get("passed") is True, primary_passed)] += 1

    replay_mismatches = [
        {
            "task_id": row.get("task_id"),
            "action_id": row.get("action_id"),
            "experimental_seed": row.get("experimental_seed"),
            "historical_passed": row.get("historical_passed"),
            "pilot_passed": row.get("passed"),
        }
        for row in rows
        if row.get("replay_matches_historical") is False
    ]
    ceiling_candidates = [
        {
            "task_id": row.get("task_id"),
            "case_pool": row.get("case_pool"),
            "action_id": row.get("action_id"),
            "experimental_seed": row.get("experimental_seed"),
        }
        for row in rows
        if row.get("action_id") in {"C_oracle_sufficient", "D_oracle_sufficient_steps96"}
        and row.get("passed") is True
    ]
    trigger_opportunities = [
        item for item in ceiling_candidates if item.get("case_pool") == "missed_failed_long"
    ]
    by_task_seed = defaultdict(dict)
    for row in rows:
        by_task_seed[(row.get("task_id"), row.get("experimental_seed"))][row.get("action_id")] = row
    canvas_effects = []
    steps96_effects = []
    negative_cases = []
    for (task_id, seed), action_rows in by_task_seed.items():
        b = action_rows.get("B_route2_len32")
        c = action_rows.get("C_oracle_sufficient")
        d = action_rows.get("D_oracle_sufficient_steps96")
        if c and b and c.get("passed") is True and b.get("passed") is not True:
            canvas_effects.append({"task_id": task_id, "experimental_seed": seed})
        if d and c and d.get("passed") is True and c.get("passed") is not True:
            steps96_effects.append({"task_id": task_id, "experimental_seed": seed})
        if c and d and c.get("passed") is not True and d.get("passed") is not True:
            negative_cases.append({"task_id": task_id, "experimental_seed": seed})
    costs = [
        float(row["total_sec_including_probe"])
        for row in rows
        if row.get("total_sec_including_probe") is not None
    ]
    return {
        "mode": "three_case_action_ceiling_pilot",
        "case_count": len(case_manifest),
        "row_count": len(rows),
        "task_ids": expected_task_ids,
        "experimental_seeds": list(experimental_seeds),
        "action_ids": list(ACTION_IDS),
        "determinism_check": determinism_check,
        "determinism_status": "deterministic" if determinism_check.get("deterministic") else "nondeterministic",
        "verdict": verdict,
        "action_pass_counts": {key: dict(value) for key, value in action_pass_counts.items()},
        "paired_vs_primary_counts": dict(paired_counts),
        "historical_replay_mismatches": replay_mismatches,
        "ceiling_candidates": ceiling_candidates,
        "canvas_effects": canvas_effects,
        "steps96_effects": steps96_effects,
        "trigger_opportunities": trigger_opportunities,
        "negative_cases_no_c_or_d_pass": negative_cases,
        "cost": {
            "sum_total_sec_including_probe": sum(costs) if costs else None,
            "avg_total_sec_including_probe": (sum(costs) / len(costs)) if costs else None,
            "p95_total_sec_including_probe": _p95(costs),
        },
        "task_action_pass_table": task_action_table,
    }


def _pass_mark(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "NA"


def render_pilot_report(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    case_manifest: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> str:
    lines = [
        "# True-Long Action-Ceiling 3-Case Pilot",
        "",
        f"verdict: `{summary['verdict']}`",
        "",
        "## Execution Sanity",
        "",
        f"- branch: `{manifest.get('branch')}`",
        f"- commit: `{manifest.get('commit')}`",
        f"- command: `{manifest.get('command')}`",
        f"- output_dir: `{manifest.get('output_dir')}`",
        f"- task_ids: `{summary.get('task_ids')}`",
        f"- experimental_seeds: `{summary.get('experimental_seeds')}`",
        f"- row_count: `{summary.get('row_count')}`",
        "",
        "## Determinism Result",
        "",
        f"- status: `{summary.get('determinism_status')}`",
        f"- check: `{summary.get('determinism_check')}`",
        "",
        "## Historical Replay Comparison",
        "",
        f"- replay_mismatches: `{summary.get('historical_replay_mismatches')}`",
        "",
        "## Per-Task A/B/C/D Table",
        "",
        "| Task | Pool | Seed | A primary | B Route2 | C oracle canvas | D oracle canvas steps96 |",
        "|---|---|---:|---|---|---|---|",
    ]
    by_key = action_row_by_key(rows)
    seeds = list(summary.get("experimental_seeds") or [])
    for case in case_manifest:
        task_id = str(case["task_id"])
        for seed in seeds:
            lines.append(
                "| `{task}` | `{pool}` | {seed} | {a} | {b} | {c} | {d} |".format(
                    task=task_id,
                    pool=case.get("case_pool"),
                    seed=seed,
                    a=_pass_mark((by_key.get((task_id, int(seed), "A_primary")) or {}).get("passed")),
                    b=_pass_mark((by_key.get((task_id, int(seed), "B_route2_len32")) or {}).get("passed")),
                    c=_pass_mark((by_key.get((task_id, int(seed), "C_oracle_sufficient")) or {}).get("passed")),
                    d=_pass_mark(
                        (by_key.get((task_id, int(seed), "D_oracle_sufficient_steps96")) or {}).get("passed")
                    ),
                )
            )
    lines.extend(
        [
            "",
            "## Candidate Existence",
            "",
            f"- ceiling_candidates: `{summary.get('ceiling_candidates')}`",
            "",
            "## Canvas Effect",
            "",
            f"- canvas_effects: `{summary.get('canvas_effects')}`",
            "",
            "## Steps96 Effect",
            "",
            f"- steps96_effects: `{summary.get('steps96_effects')}`",
            "",
            "## Trigger Opportunity",
            "",
            f"- trigger_opportunities: `{summary.get('trigger_opportunities')}`",
            "",
            "## Cost",
            "",
            f"- cost: `{summary.get('cost')}`",
            "",
            "## Negative Results",
            "",
            f"- no C/D pass cases: `{summary.get('negative_cases_no_c_or_d_pass')}`",
            "",
            "## Next-Step Verdict",
            "",
            f"- verdict: `{summary['verdict']}`",
            "- stop_rule: do not expand beyond this 3-case pilot without researcher review.",
        ]
    )
    return "\n".join(lines) + "\n"


def execute_pilot(args: argparse.Namespace, case_manifest: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    if not case_manifest:
        raise ValueError("cannot execute pilot without cases")
    if len(case_manifest) > args.max_pilot_cases:
        raise ValueError(
            f"pilot has {len(case_manifest)} cases, exceeds --max-pilot-cases={args.max_pilot_cases}; "
            "use a smaller explicit subset"
        )

    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    manifest_path = output_dir / "run_manifest.json"
    manifest = build_run_manifest(
        args,
        case_manifest,
        output_dir,
        execution_status="running",
        started_at=started_at,
    )
    write_run_manifest(manifest_path, manifest)

    try:
        from types import SimpleNamespace

        from clean_scripts import run_route2_trace_rescue as route2_runner
        from expvision_dllm_clean.dataset import load_humaneval_infilling
        from expvision_dllm_clean.modeling import load_model_and_tokenizer

        baseline_by_task = rows_by_task(load_jsonl(args.baseline_results), name="baseline")
        route2_by_task = rows_by_task(load_jsonl(args.route2_results), name="route2")
        source_commit = current_commit()

        route2_args = SimpleNamespace(
            model_path=args.model_path,
            split=args.split,
            dataset_subset=args.dataset_subset,
            max_samples=None,
            total_steps=args.total_steps,
            seed=args.seed,
            lcas_policy=args.lcas_policy,
            save_full_text_per_step=args.save_full_text_per_step,
            output_dir=str(output_dir),
            experiment_name="action_ceiling_pilot_internal",
        )
        cfg = route2_runner.make_cfg(route2_args)
        cfg.logging.output_dir = str(output_dir)
        cfg.logging.experiment_name = "action_ceiling_pilot_internal"
        cfg.decode.save_step_traces = bool(args.save_step_traces)
        cfg.decode.save_full_text_per_step = bool(args.save_full_text_per_step)
        reset_action_seed(args.seed)
        tokenizer, model = load_model_and_tokenizer(cfg.model)
        settings = route2_runner.default_midcons_settings()
        tasks = load_humaneval_infilling(split=args.split, dataset_subset=args.dataset_subset)
        by_task = {task.task_id: task for task in tasks}

        first_case = case_manifest[0]
        first_task_id = str(first_case["task_id"])
        if first_task_id not in by_task:
            raise ValueError(f"task id not found in dataset: {first_task_id}")
        determinism_check = run_determinism_check(
            case=first_case,
            task=by_task[first_task_id],
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            settings=settings,
            route2_runner=route2_runner,
            global_seed=args.seed,
        )
        experimental_seeds = [0] if determinism_check.get("deterministic") else [0, 1]

        pilot_rows: List[JsonDict] = []
        for case in case_manifest:
            task_id = str(case["task_id"])
            if task_id not in by_task:
                raise ValueError(f"task id not found in dataset: {task_id}")
            task = by_task[task_id]
            for experimental_seed in experimental_seeds:
                task_seed = stable_task_seed(args.seed, task_id, experimental_seed)
                primary_result: Optional[JsonDict] = None
                for action_id in ACTION_IDS:
                    result, action_was_executed, execution_note = run_single_action(
                        action_id=action_id,
                        case=case,
                        task=task,
                        tokenizer=tokenizer,
                        model=model,
                        cfg=cfg,
                        settings=settings,
                        route2_runner=route2_runner,
                        seed=task_seed,
                        primary_result=primary_result,
                    )
                    if result is None:
                        pilot_rows.append(
                            invalid_action_record(
                                case=case,
                                action_id=action_id,
                                seed=task_seed,
                                experimental_seed=experimental_seed,
                                source_commit=source_commit,
                                reason=execution_note,
                            )
                        )
                        continue
                    if action_id == "A_primary":
                        primary_result = result
                    pilot_rows.append(
                        collect_result_record(
                            case=case,
                            action_id=action_id,
                            result=result,
                            seed=task_seed,
                            experimental_seed=experimental_seed,
                            source_commit=source_commit,
                            action_was_executed=action_was_executed,
                            execution_note=execution_note,
                            baseline_by_task=baseline_by_task,
                            route2_by_task=route2_by_task,
                        )
                    )

        summary = summarize_pilot(
            pilot_rows,
            case_manifest=case_manifest,
            determinism_check=determinism_check,
            experimental_seeds=experimental_seeds,
        )
        if summary["verdict"] not in PILOT_VERDICTS:
            raise ValueError(f"invalid pilot verdict: {summary['verdict']}")

        wall_clock_sec = time.perf_counter() - start
        manifest = build_run_manifest(
            args,
            case_manifest,
            output_dir,
            execution_status="completed",
            started_at=started_at,
            wall_clock_sec=wall_clock_sec,
            determinism_check=determinism_check,
            experimental_seeds=experimental_seeds,
            verdict=str(summary["verdict"]),
        )
        write_jsonl(output_dir / "pilot_results.jsonl", pilot_rows)
        write_csv(output_dir / "pilot_results.csv", [compact_result_record(row) for row in pilot_rows])
        (output_dir / "pilot_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "pilot_report.md").write_text(
            render_pilot_report(summary, pilot_rows, case_manifest, manifest),
            encoding="utf-8",
        )
        write_run_manifest(manifest_path, manifest)
    except BaseException:
        manifest = build_run_manifest(
            args,
            case_manifest,
            output_dir,
            execution_status="failed",
            started_at=started_at,
            wall_clock_sec=time.perf_counter() - start,
            failure_traceback=traceback.format_exc(),
        )
        write_run_manifest(manifest_path, manifest)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run a small true-long action-ceiling matrix.")
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE)
    parser.add_argument("--route2-results", default=DEFAULT_ROUTE2)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--max-cases-per-pool", type=int, default=3)
    parser.add_argument("--task-ids-csv", default=None)
    parser.add_argument("--max-canvas-length", type=int, default=64)
    parser.add_argument("--execute-pilot", action="store_true")
    parser.add_argument("--max-pilot-cases", type=int, default=9)
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-subset", default="HumanEval-SingleLineInfilling")
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lcas-policy", default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--save-full-text-per-step", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_ids = parse_task_ids_csv(args.task_ids_csv)
    output_dir = make_output_dir(args.output_dir, args.timestamp)
    config = {
        "baseline_results": args.baseline_results,
        "route2_results": args.route2_results,
        "max_cases_per_pool": args.max_cases_per_pool,
        "task_ids": task_ids,
        "max_canvas_length": args.max_canvas_length,
        "execute_pilot": bool(args.execute_pilot),
        "model_path": args.model_path,
        "split": args.split,
        "dataset_subset": args.dataset_subset,
        "seed": args.seed,
        "lcas_policy": args.lcas_policy,
        "actions": [spec.__dict__ for spec in ACTION_SPECS],
    }
    case_manifest, action_manifest, summary = build_dry_run(
        baseline_results=args.baseline_results,
        route2_results=args.route2_results,
        max_cases_per_pool=args.max_cases_per_pool,
        task_ids=task_ids,
        max_canvas_length=args.max_canvas_length,
    )
    write_dry_run_outputs(
        output_dir,
        summary=summary,
        case_manifest=case_manifest,
        action_manifest=action_manifest,
        config=config,
    )
    if args.execute_pilot:
        execute_pilot(args, case_manifest, output_dir)
    print(f"Action-ceiling {'pilot' if args.execute_pilot else 'dry-run'} written to {output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
