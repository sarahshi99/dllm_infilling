#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl
from experiments.action_ceiling.action_ceiling_matrix import (
    current_branch,
    current_commit,
    environment_info,
    git_capture,
    reset_action_seed,
    stable_task_seed,
    write_csv,
    write_jsonl,
)
from experiments.action_ceiling.distinct_candidate_ceiling import (
    compact_result_record,
    decode_fixed_canvas,
    result_record,
)
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import load_humaneval_infilling
from expvision_dllm_clean.modeling import load_model_and_tokenizer


JsonDict = Dict[str, Any]

C_ACTION = "C_oracle_sufficient"
REFINEMENT_ACTIONS = (
    "E_oracle_sufficient_no_early_commit",
    "F_oracle_sufficient_trace_remask",
    "G_oracle_sufficient_trace_span_remask",
)
HARD_POOLS = {"triggered_failed_long", "missed_failed_long"}


def read_csv_rows(path: Path) -> List[JsonDict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def make_output_dir(base_dir: str, timestamp: Optional[str]) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"oracle_canvas_attribution_{stamp}"
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _sha(value: Mapping[str, Any]) -> Optional[str]:
    digest = value.get("generated_text_sha256")
    return None if digest in {None, ""} else str(digest)


def load_existing_seed0_rows(screen_dir: Path) -> List[JsonDict]:
    rows = load_jsonl(str(screen_dir / "seed0_results.jsonl"))
    rows = [dict(row) for row in rows if row.get("action_id") in REFINEMENT_ACTIONS]
    expected = 95 * len(REFINEMENT_ACTIONS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} seed0 E/F/G rows, found {len(rows)} in {screen_dir}")
    return rows


def run_c_for_cases(
    *,
    cases: Sequence[Mapping[str, Any]],
    model_path: str,
    split: str,
    dataset_subset: str,
    lcas_policy: str,
    source_commit: str,
) -> List[JsonDict]:
    cfg = ExperimentConfig()
    cfg.model.model_path = model_path
    cfg.data.split = split
    cfg.data.dataset_subset = dataset_subset
    cfg.decode.lcas_policy = lcas_policy
    tokenizer, model = load_model_and_tokenizer(cfg.model)
    tasks = load_humaneval_infilling(split=split, dataset_subset=dataset_subset)
    by_task = {task.task_id: task for task in tasks}
    missing = [str(case["task_id"]) for case in cases if str(case["task_id"]) not in by_task]
    if missing:
        raise ValueError(f"manifest tasks missing from dataset: {missing[:10]}")

    rows: List[JsonDict] = []
    for index, case in enumerate(cases, start=1):
        task_id = str(case["task_id"])
        task_seed = stable_task_seed(42, task_id, 0)
        reset_action_seed(task_seed)
        canvas_len = _to_int(case.get("oracle_sufficient_length") or case.get("oracle_length"))
        if canvas_len is None:
            raise ValueError(f"missing oracle_sufficient_length for {task_id}")
        print(f"[{index}/{len(cases)}] action={C_ACTION} task_id={task_id} canvas={canvas_len}", flush=True)
        result = decode_fixed_canvas(
            task=by_task[task_id],
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            canvas_len=int(canvas_len),
            total_steps=64,
            early_commit_enabled=True,
            phase_name="stage1_c_oracle_sufficient",
        )
        rows.append(
            result_record(
                case=case,
                action_id=C_ACTION,
                result=result,
                experimental_seed=0,
                task_seed=task_seed,
                source_commit=source_commit,
                action_was_executed=True,
                execution_note="executed_oracle_sufficient_with_global_gap_early_commit",
            )
        )
    return rows


def rows_by_task_action(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    by_task: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_task[str(row["task_id"])][str(row["action_id"])] = row
    return by_task


def build_per_case_attribution(cases: Sequence[Mapping[str, Any]], c_rows: Sequence[Mapping[str, Any]], efg_rows: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    c_by_task = {str(row["task_id"]): row for row in c_rows}
    efg_by_task = rows_by_task_action(efg_rows)
    output: List[JsonDict] = []
    for case in cases:
        task_id = str(case["task_id"])
        c = c_by_task[task_id]
        c_passed = bool(c.get("passed"))
        c_hash = _sha(c)
        row: JsonDict = {
            "task_id": task_id,
            "task_group": case.get("task_group") or case.get("original_humaneval_task"),
            "case_pool": case.get("case_pool") or case.get("pool"),
            "historical_error_type": case.get("historical_error_type"),
            "oracle_length": _to_int(case.get("oracle_length")),
            "oracle_sufficient_length": _to_int(case.get("oracle_sufficient_length") or case.get("oracle_length")),
            "c_passed": c_passed,
            "c_compile_passed": c.get("compile_passed"),
            "c_error_type": c.get("error_type"),
            "c_hash": c_hash,
            "c_stop_reason": c.get("stop_reason"),
            "c_early_commit_triggered": c.get("early_commit_triggered"),
            "c_actual_forward_steps": c.get("actual_forward_steps"),
            "c_effective_update_steps": c.get("effective_update_steps"),
        }
        changed_hash_actions: List[str] = []
        correctness_changed_actions: List[str] = []
        compile_changed_actions: List[str] = []
        error_type_changed_actions: List[str] = []
        incremental_actions: List[str] = []
        passing_actions: List[str] = [C_ACTION] if c_passed else []
        hashes = {c_hash} if c_hash else set()
        for action in REFINEMENT_ACTIONS:
            action_row = efg_by_task[task_id][action]
            prefix = action.split("_", 1)[0].lower()
            action_passed = bool(action_row.get("passed"))
            action_hash = _sha(action_row)
            output_changed = action_hash != c_hash
            correctness_changed = action_passed != c_passed
            compile_changed = bool(action_row.get("compile_passed")) != bool(c.get("compile_passed"))
            error_type_changed = str(action_row.get("error_type")) != str(c.get("error_type"))
            if action_hash:
                hashes.add(action_hash)
            if output_changed:
                changed_hash_actions.append(action)
            if correctness_changed:
                correctness_changed_actions.append(action)
            if compile_changed:
                compile_changed_actions.append(action)
            if error_type_changed:
                error_type_changed_actions.append(action)
            if (not c_passed) and action_passed:
                incremental_actions.append(action)
            if action_passed:
                passing_actions.append(action)
            row.update(
                {
                    f"{prefix}_passed": action_passed,
                    f"{prefix}_compile_passed": action_row.get("compile_passed"),
                    f"{prefix}_error_type": action_row.get("error_type"),
                    f"{prefix}_hash": action_hash,
                    f"{prefix}_hash_changed_vs_c": output_changed,
                    f"{prefix}_correctness_changed_vs_c": correctness_changed,
                    f"{prefix}_compile_status_changed_vs_c": compile_changed,
                    f"{prefix}_error_type_changed_vs_c": error_type_changed,
                    f"{prefix}_actual_forward_steps": action_row.get("actual_forward_steps"),
                    f"{prefix}_effective_update_steps": action_row.get("effective_update_steps"),
                }
            )
        row.update(
            {
                "unique_candidate_hash_count_c_efg": len(hashes),
                "any_efg_passed": any(bool(efg_by_task[task_id][action].get("passed")) for action in REFINEMENT_ACTIONS),
                "any_refinement_incremental_over_c": bool(incremental_actions),
                "incremental_correct_actions_over_c": ";".join(incremental_actions),
                "passing_actions": ";".join(passing_actions),
                "hash_changed_actions_vs_c": ";".join(changed_hash_actions),
                "correctness_changed_actions_vs_c": ";".join(correctness_changed_actions),
                "compile_changed_actions_vs_c": ";".join(compile_changed_actions),
                "error_type_changed_actions_vs_c": ";".join(error_type_changed_actions),
            }
        )
        output.append(row)
    return output


def summarize_attribution(per_case: Sequence[Mapping[str, Any]]) -> JsonDict:
    def selected(pool: Optional[str] = None) -> List[Mapping[str, Any]]:
        if pool is None:
            return [row for row in per_case if row.get("case_pool") in HARD_POOLS]
        return [row for row in per_case if row.get("case_pool") == pool]

    def count(rows: Sequence[Mapping[str, Any]], key: str) -> int:
        return sum(1 for row in rows if _to_bool(row.get(key)))

    hard = selected()
    missed = selected("missed_failed_long")
    triggered = selected("triggered_failed_long")
    positive = selected("positive_control_rescued")
    action_summary: Dict[str, JsonDict] = {}
    for action in REFINEMENT_ACTIONS:
        prefix = action.split("_", 1)[0].lower()
        incremental = [row for row in per_case if (row.get("case_pool") in HARD_POOLS and (not _to_bool(row.get("c_passed"))) and _to_bool(row.get(f"{prefix}_passed")))]
        action_summary[action] = {
            "hard_passed": count(hard, f"{prefix}_passed"),
            "missed_failed_long_passed": count(missed, f"{prefix}_passed"),
            "triggered_failed_long_passed": count(triggered, f"{prefix}_passed"),
            "positive_control_passed": count(positive, f"{prefix}_passed"),
            "incremental_hard_correct_over_c": len(incremental),
            "c_fail_action_pass_cases": [str(row["task_id"]) for row in incremental],
            "hash_changed_vs_c_cases": sum(1 for row in per_case if _to_bool(row.get(f"{prefix}_hash_changed_vs_c"))),
            "correctness_changed_vs_c_cases": sum(1 for row in per_case if _to_bool(row.get(f"{prefix}_correctness_changed_vs_c"))),
            "compile_status_changed_vs_c_cases": sum(1 for row in per_case if _to_bool(row.get(f"{prefix}_compile_status_changed_vs_c"))),
            "error_type_changed_vs_c_cases": sum(1 for row in per_case if _to_bool(row.get(f"{prefix}_error_type_changed_vs_c"))),
        }

    any_efg_hard = [row for row in hard if _to_bool(row.get("any_efg_passed"))]
    c_hard = [row for row in hard if _to_bool(row.get("c_passed"))]
    incremental_hard = [row for row in hard if (not _to_bool(row.get("c_passed"))) and _to_bool(row.get("any_efg_passed"))]
    overlap_counter = Counter(str(row.get("passing_actions")) for row in per_case)
    return {
        "mode": "oracle_canvas_incremental_attribution",
        "case_count": len(per_case),
        "hard_case_count": len(hard),
        "positive_control_count": len(positive),
        "c_correct": {
            "hard": len(c_hard),
            "missed_failed_long": count(missed, "c_passed"),
            "triggered_failed_long": count(triggered, "c_passed"),
            "positive_control_rescued": count(positive, "c_passed"),
        },
        "efg_any_correct": {
            "hard": len(any_efg_hard),
            "missed_failed_long": sum(1 for row in missed if _to_bool(row.get("any_efg_passed"))),
            "triggered_failed_long": sum(1 for row in triggered if _to_bool(row.get("any_efg_passed"))),
            "positive_control_rescued": sum(1 for row in positive if _to_bool(row.get("any_efg_passed"))),
        },
        "hard_recovery_attribution": {
            "total_efg_hard_recoveries": len(any_efg_hard),
            "sufficient_canvas_effect_c_passed": len(c_hard),
            "refinement_incremental_effect_c_failed_but_efg_passed": len(incremental_hard),
            "c_failed_but_efg_passed_cases": [str(row["task_id"]) for row in incremental_hard],
            "c_passed_cases": [str(row["task_id"]) for row in c_hard],
        },
        "action_summary": action_summary,
        "correctness_overlap": dict(sorted(overlap_counter.items())),
        "candidate_hash_change_vs_correctness_change": {
            "hash_changed_any_refinement": sum(1 for row in per_case if bool(row.get("hash_changed_actions_vs_c"))),
            "correctness_changed_any_refinement": sum(1 for row in per_case if bool(row.get("correctness_changed_actions_vs_c"))),
            "hash_changed_without_correctness_change": sum(
                1
                for row in per_case
                if bool(row.get("hash_changed_actions_vs_c")) and not bool(row.get("correctness_changed_actions_vs_c"))
            ),
            "correctness_changed_without_hash_change": sum(
                1
                for row in per_case
                if bool(row.get("correctness_changed_actions_vs_c")) and not bool(row.get("hash_changed_actions_vs_c"))
            ),
        },
    }


def render_report(summary: Mapping[str, Any]) -> str:
    c = summary["c_correct"]
    efg = summary["efg_any_correct"]
    attr = summary["hard_recovery_attribution"]
    lines = [
        "# Oracle Canvas Incremental Attribution",
        "",
        "This audit separates the pass-level effect of `C_oracle_sufficient` from incremental effects of E/F/G diagnostic refinement actions.",
        "",
        "## Main Counts",
        "",
        f"- hard_case_count: `{summary.get('hard_case_count')}`",
        f"- positive_control_count: `{summary.get('positive_control_count')}`",
        f"- C hard recoveries: `{c.get('hard')}`",
        f"- C missed-failed-long recoveries: `{c.get('missed_failed_long')}`",
        f"- C triggered-failed-long recoveries: `{c.get('triggered_failed_long')}`",
        f"- any E/F/G hard recoveries: `{efg.get('hard')}`",
        f"- sufficient canvas effect among E/F/G hard recoveries: `{attr.get('sufficient_canvas_effect_c_passed')}`",
        f"- refinement incremental effect among E/F/G hard recoveries: `{attr.get('refinement_incremental_effect_c_failed_but_efg_passed')}`",
        "",
        "## Action Increment Over C",
        "",
        "| Action | Hard Pass | Missed Pass | Triggered Pass | Incremental Hard Over C | Hash Changed vs C | Correctness Changed vs C |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for action, info in (summary.get("action_summary") or {}).items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                action,
                info.get("hard_passed"),
                info.get("missed_failed_long_passed"),
                info.get("triggered_failed_long_passed"),
                info.get("incremental_hard_correct_over_c"),
                info.get("hash_changed_vs_c_cases"),
                info.get("correctness_changed_vs_c_cases"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `sufficient_canvas_effect_c_passed` counts cases where the oracle-sufficient C candidate already passes.",
            "- `refinement_incremental_effect_c_failed_but_efg_passed` counts hard cases where C fails but at least one of E/F/G passes.",
            "- Hash changes and correctness changes are reported separately; a changed candidate is not treated as a pass-level canvas effect.",
            "",
        ]
    )
    return "\n".join(lines)


def build_run_manifest(args: argparse.Namespace, output_dir: Path, status: str, started_at: str, **extra: Any) -> JsonDict:
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
        "timestamp": args.timestamp,
        "execution_status": status,
        "output_dir": str(output_dir),
        "source_screen_dir": str(args.screen_dir),
        "environment": environment_info(args.model_path),
        "dataset": {"subset": args.dataset_subset, "split": args.split},
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
        **extra,
    }


def execute(args: argparse.Namespace) -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    output_dir = make_output_dir(args.output_dir, args.timestamp)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(build_run_manifest(args, output_dir, "running", started_at), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        screen_dir = Path(args.screen_dir)
        cases = read_csv_rows(screen_dir / "case_manifest.csv")
        efg_rows = load_existing_seed0_rows(screen_dir)
        if args.reuse_c_results:
            c_rows = load_jsonl(args.reuse_c_results)
        else:
            c_rows = run_c_for_cases(
                cases=cases,
                model_path=args.model_path,
                split=args.split,
                dataset_subset=args.dataset_subset,
                lcas_policy=args.lcas_policy,
                source_commit=current_commit(),
            )
        if len(c_rows) != len(cases):
            raise ValueError(f"C row count mismatch: expected {len(cases)}, got {len(c_rows)}")
        c_ids = {str(row["task_id"]) for row in c_rows}
        case_ids = {str(row["task_id"]) for row in cases}
        if c_ids != case_ids:
            raise ValueError(f"C task id set mismatch: missing={sorted(case_ids - c_ids)[:10]}, extra={sorted(c_ids - case_ids)[:10]}")

        write_jsonl(output_dir / "c_results.jsonl", c_rows)
        write_csv(output_dir / "c_results.csv", [compact_result_record(row) for row in c_rows])
        per_case = build_per_case_attribution(cases, c_rows, efg_rows)
        summary = summarize_attribution(per_case)
        write_csv(output_dir / "per_case_attribution.csv", per_case)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
        manifest = build_run_manifest(
            args,
            output_dir,
            "completed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            c_rows=len(c_rows),
            efg_rows=len(efg_rows),
            summary=summary,
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "c_hard_recoveries": summary["c_correct"]["hard"]}, ensure_ascii=False, indent=2))
    except BaseException:
        manifest = build_run_manifest(
            args,
            output_dir,
            "failed",
            started_at,
            wall_clock_sec=time.perf_counter() - start,
            failure_traceback=traceback.format_exc(),
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute C oracle canvas vs E/F/G refinement gains on the 95-case screen.")
    parser.add_argument("--screen-dir", default="analysis_outputs/long_failure_generation_screen_20260702_accel95")
    parser.add_argument("--output-dir", default="analysis_outputs")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--reuse-c-results", default=None)
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-subset", default="HumanEval-SingleLineInfilling")
    parser.add_argument("--lcas-policy", default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    return parser.parse_args()


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
