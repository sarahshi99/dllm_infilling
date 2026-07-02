#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
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

ACTION_IDS = ("A_primary", "B_route2_len32", "C_oracle_sufficient", "D_oracle_sufficient_conservative")


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
        action_id="D_oracle_sufficient_conservative",
        deployability="offline_ceiling_only_oracle_canvas",
        canvas_rule="max(primary_len, route2_len, oracle_len)",
        steps=96,
        description="Same oracle-sufficient canvas as C, but with a fixed conservative 96-step schedule.",
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
    return min(max(values), max_length)


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
            rows.append(
                {
                    "task_id": case["task_id"],
                    "case_pool": case["case_pool"],
                    "oracle_bucket": case["oracle_bucket"],
                    "action_id": spec.action_id,
                    "deployability": spec.deployability,
                    "canvas_rule": spec.canvas_rule,
                    "planned_canvas_length": planned_canvas,
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
        "oracle_used_for_actions": ["C_oracle_sufficient", "D_oracle_sufficient_conservative"],
        "deployable_actions": ["A_primary", "B_route2_len32"],
        "not_deployable_ceiling_actions": ["C_oracle_sufficient", "D_oracle_sufficient_conservative"],
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


def render_report(summary: Mapping[str, Any], case_manifest: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# True-Long Action-Ceiling Matrix Dry Run",
        "",
        "Decision: `ready_for_small_pilot_after_researcher_approval`",
        "",
        "## Pre-Registration",
        "",
        "- hypothesis: true-long failures are jointly limited by canvas adequacy, rescue generation quality, selector quality, and trigger recall.",
        "- intervention: compare current primary, current Route2 rescue, oracle-sufficient canvas, and oracle-sufficient canvas with a fixed 96-step conservative schedule.",
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


def execute_pilot(args: argparse.Namespace, case_manifest: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    if not case_manifest:
        raise ValueError("cannot execute pilot without cases")
    if len(case_manifest) > args.max_pilot_cases:
        raise ValueError(
            f"pilot has {len(case_manifest)} cases, exceeds --max-pilot-cases={args.max_pilot_cases}; "
            "use a smaller explicit subset"
        )

    from types import SimpleNamespace

    from clean_scripts import run_route2_trace_rescue as route2_runner
    from expvision_dllm_clean.dataset import load_humaneval_infilling
    from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed

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
    set_global_seed(cfg.decode.seed)
    tokenizer, model = load_model_and_tokenizer(cfg.model)
    settings = route2_runner.default_midcons_settings()
    tasks = load_humaneval_infilling(split=args.split, dataset_subset=args.dataset_subset)
    by_task = {task.task_id: task for task in tasks}

    pilot_rows: List[JsonDict] = []
    for case in case_manifest:
        task_id = str(case["task_id"])
        if task_id not in by_task:
            raise ValueError(f"task id not found in dataset: {task_id}")
        task = by_task[task_id]
        primary = route2_runner.run_primary(task, tokenizer, model, cfg, settings)
        action_results: Dict[str, JsonDict] = {"A_primary": primary}

        b_len = _to_int(case.get("route2_rescue_length"))
        if b_len is not None:
            action_results["B_route2_len32"] = route2_runner.run_fixed_rescue(task, tokenizer, model, cfg, b_len)
        else:
            action_results["B_route2_len32"] = primary

        ceiling_len = _to_int(case.get("oracle_sufficient_length"))
        if ceiling_len is not None:
            c_cfg = copy.deepcopy(cfg)
            c_cfg.decode.total_steps = 64
            action_results["C_oracle_sufficient"] = route2_runner.run_fixed_rescue(task, tokenizer, model, c_cfg, ceiling_len)

            d_cfg = copy.deepcopy(cfg)
            d_cfg.decode.total_steps = 96
            action_results["D_oracle_sufficient_conservative"] = route2_runner.run_fixed_rescue(
                task,
                tokenizer,
                model,
                d_cfg,
                ceiling_len,
            )

        for action_id, result in action_results.items():
            metrics = result.get("metrics") or {}
            pilot_rows.append(
                {
                    "task_id": task_id,
                    "case_pool": case.get("case_pool"),
                    "action_id": action_id,
                    "passed": bool(metrics.get("passed")),
                    "selected_mask_length": metrics.get("selected_mask_length"),
                    "oracle_mask_length": metrics.get("oracle_mask_length"),
                    "total_sec_including_probe": metrics.get("total_sec_including_probe"),
                    "deployability": next(spec.deployability for spec in ACTION_SPECS if spec.action_id == action_id),
                }
            )

    write_csv(output_dir / "pilot_results.csv", pilot_rows)


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
