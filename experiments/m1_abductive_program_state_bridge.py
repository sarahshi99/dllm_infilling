#!/usr/bin/env python3
"""Independent M1 Abductive Program-State Bridge on RandomSpanLight.

The runner keeps a new 148-case candidate bank separate from the safely
paused historical 5079-case MultiLine raw outputs.  It runs the eight visible
fixed-canvas candidates, an oracle diagnostic ceiling, then the equal-compute
generic and dependency-cone refinements.  A smoke may automatically advance
only to the 148-case RandomSpanLight population.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.deployable_visible_task import evaluate_completion_after_decode, visible_task
from experiments.m2_constraint_homotopy import (
    append_jsonl,
    build_manifest,
    choose_smoke_manifest,
    load_frozen_groups,
    read_jsonl,
    sha256_text,
    write_csv,
    write_json,
)
from experiments.method_population_schedule import (
    RANDOMSPANLIGHT_ALLOWED_CASES,
    SMOKE_CASES,
    population_schedule,
    require_randomspanlight_full,
)
from experiments.phase6_abductive_bridge_runner import (
    REFINEMENT_METHODS,
    expected_refinement_keys,
    run_population as run_refinement_population,
)
from experiments.phase6_remask import decode_fixed_canvas_state
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed


MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
SOURCE_CONFIG = "HumanEval-RandomSpanInfillingLight"
CANVASES = (16, 32, 64, 128)
SEEDS = (0, 1)
TOTAL_STEPS = 64
GRID_KIND = "deployable_grid"
ORACLE_KIND = "oracle_sufficient_diagnostic_ceiling"


def cfg_for(canvas_tokens: int, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = MODEL_PATH
    cfg.model.torch_dtype = "bfloat16"
    cfg.model.device_map = "auto"
    cfg.data.dataset_subset = SOURCE_CONFIG
    cfg.decode.mask_length_source = "fixed"
    cfg.decode.fixed_mask_length = int(canvas_tokens)
    cfg.decode.total_steps = TOTAL_STEPS
    cfg.decode.seed = int(seed)
    cfg.decode.save_step_traces = False
    cfg.decode.save_full_text_per_step = False
    return cfg


def ast_hash(code: str) -> str:
    try:
        return hashlib.sha256(ast.dump(ast.parse(code), include_attributes=False).encode("utf-8")).hexdigest()
    except SyntaxError:
        return ""


def stage1_key(row_key: str, candidate_kind: str, canvas_tokens: int, seed: int) -> str:
    return f"{row_key}|{candidate_kind}|canvas={int(canvas_tokens)}|seed={int(seed)}"


def grid_specs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_kind": GRID_KIND,
            "control_label": "fixed64_seed0" if (canvas, seed) == (64, 0) else "visible_grid_candidate",
            "deployable": True,
            "canvas_tokens": canvas,
            "seed": seed,
            "total_steps": TOTAL_STEPS,
        }
        for canvas in CANVASES
        for seed in SEEDS
    ]


def oracle_spec(reference_middle_tokens: int) -> dict[str, Any]:
    return {
        "candidate_kind": ORACLE_KIND,
        "control_label": "offline_oracle_ceiling_only",
        "deployable": False,
        "canvas_tokens": max(1, int(reference_middle_tokens)),
        "seed": 0,
        "total_steps": TOTAL_STEPS,
    }


def expected_grid_keys(manifest: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        stage1_key(str(item["row_key"]), GRID_KIND, int(spec["canvas_tokens"]), int(spec["seed"]))
        for item in manifest
        for spec in grid_specs()
    }


def expected_oracle_keys(manifest: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        stage1_key(str(item["row_key"]), ORACLE_KIND, int(item["reference_middle_tokens"]), 0)
        for item in manifest
    }


def stage1_audit(rows: Sequence[Mapping[str, Any]], expected: set[str]) -> dict[str, Any]:
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    errors = [str(row.get("candidate_key") or "") for row in rows if row.get("status") != "ok"]
    forward_bad = [
        str(row.get("candidate_key") or "")
        for row in rows
        if row.get("status") == "ok"
        and int((row.get("metrics") or {}).get("actual_forward_count") or 0) != TOTAL_STEPS
    ]
    return {
        "passed": observed == expected and all(value == 1 for value in counts.values()) and not errors and not forward_bad,
        "expected_count": len(expected),
        "observed_count": len(rows),
        "unique_count": len(observed),
        "missing_count": len(expected - observed),
        "extra_count": len(observed - expected),
        "duplicate_count": sum(value > 1 for value in counts.values()),
        "error_count": len(errors),
        "non64_forward_count": len(forward_bad),
    }


def stage1_candidate(
    *,
    manifest_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    spec: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
) -> dict[str, Any]:
    key = stage1_key(
        str(manifest_row["row_key"]),
        str(spec["candidate_kind"]),
        int(spec["canvas_tokens"]),
        int(spec["seed"]),
    )
    started = time.perf_counter()
    try:
        set_global_seed(int(spec["seed"]))
        task = visible_task(str(source_row["prompt"]), str(source_row["suffix"]), method="m1_stage1")
        result = decode_fixed_canvas_state(
            task=task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg_for(int(spec["canvas_tokens"]), int(spec["seed"])),
            canvas_tokens=int(spec["canvas_tokens"]),
            total_steps=int(spec["total_steps"]),
            phase_name="m1_randomspanlight_stage1",
            evaluate_after_decode=lambda prefix, middle, suffix: evaluate_completion_after_decode(
                prefix=prefix,
                suffix=suffix,
                middle=middle,
                source_row=source_row,
                method="m1_stage1",
            ),
        )
        middle, code = str(result.get("middle_text") or ""), str(result.get("code") or "")
        metrics = dict(result.get("metrics") or {})
        return {
            "candidate_key": key,
            "row_key": manifest_row["row_key"],
            "case_index": manifest_row["case_index"],
            "source_row_id": manifest_row["source_row_id"],
            "task_group": manifest_row["task_group"],
            "length_bucket": manifest_row["length_bucket"],
            "reference_middle_tokens": manifest_row["reference_middle_tokens"],
            "candidate_kind": spec["candidate_kind"],
            "control_label": spec["control_label"],
            "deployable": spec["deployable"],
            "canvas_tokens": spec["canvas_tokens"],
            "seed": spec["seed"],
            "total_steps": spec["total_steps"],
            "status": "ok",
            "passed": bool(metrics.get("passed", False)),
            "prefix_text": str(task.prefix),
            "middle_text": middle,
            "suffix_text": str(task.suffix),
            "code": code,
            "candidate_middle_tokens": len(result.get("middle_token_ids") or []),
            "candidate_middle_sha256": sha256_text(middle),
            "candidate_full_code_sha256": sha256_text(code),
            "candidate_full_ast_sha256": ast_hash(code),
            "middle_token_ids": result.get("middle_token_ids") or [],
            "final_token_confidences": result.get("final_token_confidences") or [],
            "metrics": metrics,
            "verification": result.get("verification") or {},
            "diagnostics": result.get("diagnostics") or {},
            "trajectory": result.get("trajectory") or {},
            "wall_sec": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "candidate_key": key,
            "row_key": manifest_row["row_key"],
            "case_index": manifest_row["case_index"],
            "source_row_id": manifest_row["source_row_id"],
            "task_group": manifest_row["task_group"],
            "length_bucket": manifest_row["length_bucket"],
            "reference_middle_tokens": manifest_row["reference_middle_tokens"],
            "candidate_kind": spec["candidate_kind"],
            "control_label": spec["control_label"],
            "deployable": spec["deployable"],
            "canvas_tokens": spec["canvas_tokens"],
            "seed": spec["seed"],
            "total_steps": spec["total_steps"],
            "status": "error",
            "passed": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:240],
            "failure_traceback": traceback.format_exc(),
            "metrics": {},
            "verification": {},
            "wall_sec": time.perf_counter() - started,
        }


def run_stage1_population(
    *,
    manifest: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    raw_path: Path,
    tokenizer: Any,
    model: Any,
) -> int:
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    counts = Counter(str(row.get("candidate_key") or "") for row in existing)
    duplicate = [key for key, count in counts.items() if count > 1]
    if duplicate:
        raise RuntimeError(f"M1 stage-one resume refuses duplicate keys: {duplicate[:5]}")
    completed = set(counts)
    written = 0
    for item in manifest:
        source = source_rows[int(item["source_row_id"])]
        for spec in [*grid_specs(), oracle_spec(int(item["reference_middle_tokens"]))]:
            key = stage1_key(str(item["row_key"]), str(spec["candidate_kind"]), int(spec["canvas_tokens"]), int(spec["seed"]))
            if key in completed:
                continue
            row = stage1_candidate(
                manifest_row=item,
                source_row=source,
                spec=spec,
                tokenizer=tokenizer,
                model=model,
            )
            append_jsonl(raw_path, row)
            completed.add(key)
            written += 1
    return written


def refinement_audit(
    rows: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]], method: str
) -> dict[str, Any]:
    expected = expected_refinement_keys(manifest, methods=(method,))
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    errors = [row for row in rows if row.get("status") != "ok"]
    budget_bad = [
        row
        for row in rows
        if row.get("status") == "ok"
        and int((row.get("metrics") or {}).get("standalone_actual_forward_count") or 0) != 576
    ]
    return {
        "passed": observed == expected and all(value == 1 for value in counts.values()) and not errors and not budget_bad,
        "expected_count": len(expected),
        "observed_count": len(rows),
        "missing_count": len(expected - observed),
        "extra_count": len(observed - expected),
        "duplicate_count": sum(value > 1 for value in counts.values()),
        "error_count": len(errors),
        "non576_standalone_forward_count": len(budget_bad),
    }


def compact_summary(
    *,
    compact_dir: Path,
    phase: str,
    selected_manifest: Sequence[Mapping[str, Any]],
    stage1_rows: Sequence[Mapping[str, Any]],
    generic_rows: Sequence[Mapping[str, Any]],
    m1_rows: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    resume_noops: int,
    peak_memory_bytes: int,
    wall_sec: float,
) -> dict[str, Any]:
    grid = [row for row in stage1_rows if row.get("candidate_kind") == GRID_KIND]
    oracle = [row for row in stage1_rows if row.get("candidate_kind") == ORACLE_KIND]
    audits = {
        "stage1_grid": stage1_audit(grid, expected_grid_keys(selected_manifest)),
        "oracle_ceiling": stage1_audit(oracle, expected_oracle_keys(selected_manifest)),
        REFINEMENT_METHODS[0]: refinement_audit(generic_rows, selected_manifest, REFINEMENT_METHODS[0]),
        REFINEMENT_METHODS[1]: refinement_audit(m1_rows, selected_manifest, REFINEMENT_METHODS[1]),
    }
    frozen = lock.get("test_status") == "sealed" and int(lock.get("test_evaluation_count", -1)) == 0
    technical_passed = bool(all(value["passed"] for value in audits.values()) and frozen and resume_noops == 0 and peak_memory_bytes > 0)
    compact_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        compact_dir / f"{phase}_artifacts.csv",
        [
            {
                "candidate_key": row.get("candidate_key"),
                "row_key": row.get("row_key"),
                "method": row.get("candidate_kind"),
                "status": row.get("status"),
                "passed": row.get("passed"),
                "actual_forward_count": (row.get("metrics") or {}).get("actual_forward_count"),
                "standalone_actual_forward_count": (row.get("metrics") or {}).get("standalone_actual_forward_count"),
                "token_budget": (row.get("metrics") or {}).get("token_budget"),
            }
            for row in [*stage1_rows, *generic_rows, *m1_rows]
        ],
    )
    summary = {
        "phase": phase,
        "technical_gate_passed": technical_passed,
        "selected_case_count": len(selected_manifest),
        "audits": audits,
        "frozen_test_status": lock.get("test_status"),
        "test_evaluation_count": lock.get("test_evaluation_count"),
        "resume_noop_writes": resume_noops,
        "peak_cuda_memory_bytes": peak_memory_bytes,
        "wall_sec": wall_sec,
        "deployable_input_boundary": "prefix_suffix_candidate_tokens_confidence_only; evaluator_constructed_after_decode",
        "standalone_cost_contract": "stage_one_grid=8x64=512; generic_or_m1_full=512+64=576",
    }
    write_json(compact_dir / f"{phase}_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> int:
    dataset_jsonl = Path(args.dataset_jsonl).resolve()
    stage1_dir = Path(args.stage1_output_dir).resolve()
    generic_dir = Path(args.generic_output_dir).resolve()
    m1_dir = Path(args.m1_output_dir).resolve()
    compact_dir = Path(args.compact_dir).resolve()
    if len({stage1_dir, generic_dir, m1_dir}) != 3:
        raise ValueError("M1 stage-one, generic, and dependency-cone outputs must be independent")
    stage1_raw = stage1_dir / "m1_stage1_candidate_bank_raw.jsonl"
    generic_raw = generic_dir / "m1_equal_compute_generic_raw.jsonl"
    m1_raw = m1_dir / "m1_dependency_cone_raw.jsonl"
    frozen_groups, lock = load_frozen_groups()
    source_rows = read_jsonl(dataset_jsonl)
    if len(source_rows) != 164:
        raise RuntimeError(f"M1 expected 164 RandomSpanLight source rows, found {len(source_rows)}")
    tokenizer, model = load_model_and_tokenizer(cfg_for(64, 0).model)
    manifest = build_manifest(source_rows, tokenizer, frozen_groups)
    if len(manifest) != RANDOMSPANLIGHT_ALLOWED_CASES or len({row["task_group"] for row in manifest}) != RANDOMSPANLIGHT_ALLOWED_CASES:
        raise RuntimeError("M1 requires exactly 148 non-frozen RandomSpanLight task groups")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for M1")
    selected = choose_smoke_manifest(manifest, int(args.smoke_cases))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    run_stage1_population(manifest=selected, source_rows=source_rows, raw_path=stage1_raw, tokenizer=tokenizer, model=model)
    smoke_stage1 = read_jsonl(stage1_raw)
    for method, path in ((REFINEMENT_METHODS[0], generic_raw), (REFINEMENT_METHODS[1], m1_raw)):
        run_refinement_population(
            method=method,
            manifest=selected,
            source_rows=source_rows,
            stage1_rows=smoke_stage1,
            raw_path=path,
            tokenizer=tokenizer,
            model=model,
            cfg_for=cfg_for,
            set_seed=set_global_seed,
            append_jsonl=append_jsonl,
        )
    smoke_noops = run_stage1_population(manifest=selected, source_rows=source_rows, raw_path=stage1_raw, tokenizer=tokenizer, model=model)
    for method, path in ((REFINEMENT_METHODS[0], generic_raw), (REFINEMENT_METHODS[1], m1_raw)):
        smoke_noops += run_refinement_population(
            method=method,
            manifest=selected,
            source_rows=source_rows,
            stage1_rows=read_jsonl(stage1_raw),
            raw_path=path,
            tokenizer=tokenizer,
            model=model,
            cfg_for=cfg_for,
            set_seed=set_global_seed,
            append_jsonl=append_jsonl,
        )
    smoke = compact_summary(
        compact_dir=compact_dir,
        phase="smoke",
        selected_manifest=selected,
        stage1_rows=read_jsonl(stage1_raw),
        generic_rows=read_jsonl(generic_raw),
        m1_rows=read_jsonl(m1_raw),
        lock=lock,
        resume_noops=smoke_noops,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        wall_sec=time.perf_counter() - started,
    )
    if not smoke["technical_gate_passed"]:
        write_json(compact_dir / "run_manifest.json", {"status": "smoke_failed", "smoke": smoke, "test_evaluation_count": 0})
        return 2

    full: dict[str, Any] | None = None
    if args.auto_full:
        require_randomspanlight_full(len(manifest))
        run_stage1_population(manifest=manifest, source_rows=source_rows, raw_path=stage1_raw, tokenizer=tokenizer, model=model)
        full_stage1 = read_jsonl(stage1_raw)
        for method, path in ((REFINEMENT_METHODS[0], generic_raw), (REFINEMENT_METHODS[1], m1_raw)):
            run_refinement_population(
                method=method,
                manifest=manifest,
                source_rows=source_rows,
                stage1_rows=full_stage1,
                raw_path=path,
                tokenizer=tokenizer,
                model=model,
                cfg_for=cfg_for,
                set_seed=set_global_seed,
                append_jsonl=append_jsonl,
            )
        full_noops = run_stage1_population(manifest=manifest, source_rows=source_rows, raw_path=stage1_raw, tokenizer=tokenizer, model=model)
        for method, path in ((REFINEMENT_METHODS[0], generic_raw), (REFINEMENT_METHODS[1], m1_raw)):
            full_noops += run_refinement_population(
                method=method,
                manifest=manifest,
                source_rows=source_rows,
                stage1_rows=read_jsonl(stage1_raw),
                raw_path=path,
                tokenizer=tokenizer,
                model=model,
                cfg_for=cfg_for,
                set_seed=set_global_seed,
                append_jsonl=append_jsonl,
            )
        full = compact_summary(
            compact_dir=compact_dir,
            phase="full",
            selected_manifest=manifest,
            stage1_rows=read_jsonl(stage1_raw),
            generic_rows=read_jsonl(generic_raw),
            m1_rows=read_jsonl(m1_raw),
            lock=lock,
            resume_noops=full_noops,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
            wall_sec=time.perf_counter() - started,
        )
    write_json(
        compact_dir / "run_manifest.json",
        {
            "status": "completed" if full is None or full["technical_gate_passed"] else "full_failed",
            "model": MODEL_PATH,
            "source_config": SOURCE_CONFIG,
            "stage1_raw": str(stage1_raw),
            "generic_raw": str(generic_raw),
            "m1_raw": str(m1_raw),
            "smoke": smoke,
            "full": full,
            "population_schedule": population_schedule(),
            "automatic_full_population": "randomspanlight_full",
            "frozen_test_status": "sealed",
            "test_evaluation_count": 0,
        },
    )
    return 0 if full is None or full["technical_gate_passed"] else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="M1 Abductive Program-State Bridge")
    root.add_argument("--dataset-jsonl", required=True)
    root.add_argument("--stage1-output-dir", required=True)
    root.add_argument("--generic-output-dir", required=True)
    root.add_argument("--m1-output-dir", required=True)
    root.add_argument("--compact-dir", required=True)
    root.add_argument("--smoke-cases", type=int, default=SMOKE_CASES)
    root.add_argument("--auto-full", action="store_true")
    return root


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
