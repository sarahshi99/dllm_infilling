#!/usr/bin/env python3
"""Executable stage-two refinement for the M1.1 MultiLine protocol.

The stage-one bank is immutable input.  This module selects only from a
strictly filtered state view, performs real 64-forward refinement, and writes
its own resumable/deduplicated raw result file.
"""

from __future__ import annotations

import hashlib
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.phase6_abductive_bridge_v1 import (
    extract_backward_obligations,
    select_v1_candidate,
)
from experiments.phase5_randomspanlight_candidate_bank import ast_hash, code_task
from experiments.phase6_remask import (
    decode_fixed_canvas_state,
    dependency_cone_token_indices,
    generic_low_confidence_indices,
)


REFINEMENT_METHODS = (
    "equal_compute_generic_remask",
    "m1_dependency_cone_remask",
)


def refinement_key(row_key: str, method: str) -> str:
    if method not in REFINEMENT_METHODS:
        raise ValueError(f"Unknown refinement method {method!r}")
    return f"{row_key}|{method}|refinement=64"


def expected_refinement_keys(manifest: Sequence[Mapping[str, Any]]) -> set[str]:
    return {refinement_key(str(item["row_key"]), method) for item in manifest for method in REFINEMENT_METHODS}


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fixed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("candidate_kind") != "deployable_grid":
            continue
        copied = dict(row)
        row_key = str(copied["row_key"])
        grid[row_key].append(copied)
        if int(copied.get("canvas_tokens") or -1) == 64 and int(copied.get("seed") or -1) == 0:
            fixed[row_key] = copied
    if not grid or any(len(items) != 8 for items in grid.values()):
        raise RuntimeError("M1 refinement requires exactly eight stage-one deployable candidates per row")
    if set(grid) != set(fixed):
        raise RuntimeError("M1 refinement requires a fixed64/seed0 stage-one candidate for every row")
    return grid, fixed


def _selection_inputs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Construct the only input view permitted to deployable M1 selection."""
    return [
        {
            "candidate_ordinal": ordinal,
            "middle_text": str(row.get("middle_text") or ""),
            "canvas_tokens": int(row["canvas_tokens"]),
            "seed": int(row["seed"]),
        }
        for ordinal, row in enumerate(sorted(rows, key=lambda item: (int(item["canvas_tokens"]), int(item["seed"]))))
    ]


def _selection_diagnostics(prefix: str, suffix: str, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    selected, diagnostics = select_v1_candidate(prefix, suffix, _selection_inputs(rows))
    ordinal = int(selected["candidate_ordinal"])
    diagnostic = diagnostics[ordinal]
    return {"candidate_ordinal": ordinal}, {
        "full_parse_passed": diagnostic.full_parse_passed,
        "boundary_violation_count": len(diagnostic.boundary_violations),
        "control_contradiction_count": len(diagnostic.control_contradictions),
        "unsatisfied_obligation_count": len(diagnostic.unsatisfied_obligations),
        "def_use_conflict_count": len(diagnostic.def_use_conflicts),
        "undefined_use_count": len(diagnostic.undefined_uses),
        "restored_dependency_count": len(diagnostic.restored_dependencies),
    }


def _result_row(
    *,
    method: str,
    manifest_row: Mapping[str, Any],
    selected_base: Mapping[str, Any],
    result: Mapping[str, Any],
    remask_indices: Sequence[int],
    selection_diagnostics: Mapping[str, Any],
    fallback_to_fixed64: bool,
    fallback_reason: str,
) -> dict[str, Any]:
    metrics = dict(result.get("metrics") or {})
    middle = str(result.get("middle_text") or "")
    code = str(result.get("code") or "")
    stage1_forward = int(selected_base.get("metrics", {}).get("actual_forward_count") or selected_base.get("total_steps") or 0)
    refinement_forward = int(metrics.get("actual_forward_count") or 0)
    stage1_wall = float(selected_base.get("metrics", {}).get("total_sec_including_probe") or selected_base.get("wall_sec") or 0.0)
    refinement_wall = float(metrics.get("total_sec_including_probe") or 0.0)
    canvas = int(selected_base["canvas_tokens"])
    return {
        "candidate_key": refinement_key(str(manifest_row["row_key"]), method),
        "row_key": manifest_row["row_key"],
        "candidate_kind": method,
        "control_label": method,
        "deployable": True,
        "case_index": manifest_row["case_index"],
        "source_row_id": manifest_row["source_row_id"],
        "task_group": manifest_row["task_group"],
        "length_bucket": manifest_row["length_bucket"],
        "reference_middle_tokens": manifest_row["reference_middle_tokens"],
        "status": "ok",
        "passed": bool(metrics.get("passed", False)),
        "canvas_tokens": canvas,
        "seed": int(selected_base["seed"]),
        "total_steps": stage1_forward + refinement_forward,
        "selected_base_candidate_key": selected_base["candidate_key"],
        "selected_base_middle_sha256": selected_base.get("candidate_middle_sha256", ""),
        "candidate_middle_tokens": len(result.get("middle_token_ids") or []),
        "candidate_middle_sha256": hashlib.sha256(middle.encode("utf-8")).hexdigest(),
        "candidate_full_code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "candidate_full_ast_sha256": ast_hash(code),
        "middle_token_ids": result.get("middle_token_ids") or [],
        "final_token_confidences": result.get("final_token_confidences") or [],
        "remasked_token_indices": [int(index) for index in remask_indices],
        "remasked_token_count": len(remask_indices),
        "fallback_to_fixed64": bool(fallback_to_fixed64),
        "fallback_reason": fallback_reason,
        "selection_diagnostics": dict(selection_diagnostics),
        "metrics": {
            **metrics,
            "stage1_forward_count": stage1_forward,
            "refinement_forward_count": refinement_forward,
            "actual_forward_count": stage1_forward + refinement_forward,
            "stage1_token_budget": canvas * stage1_forward,
            "refinement_token_budget": canvas * refinement_forward,
            "token_budget": canvas * (stage1_forward + refinement_forward),
            "stage1_wall_sec": stage1_wall,
            "refinement_wall_sec": refinement_wall,
            "total_sec_including_probe": stage1_wall + refinement_wall,
            "refinement_executed": not fallback_to_fixed64,
        },
        "verification": result.get("verification") or {},
        "diagnostics": result.get("diagnostics") or {},
        "middle_text": middle,
        "code": code,
    }


def _fallback_row(
    *,
    manifest_row: Mapping[str, Any],
    fixed64: Mapping[str, Any],
    selected_base: Mapping[str, Any],
    selection_diagnostics: Mapping[str, Any],
    reason: str,
    method: str = "m1_dependency_cone_remask",
) -> dict[str, Any]:
    if method not in REFINEMENT_METHODS:
        raise ValueError(f"Unknown fallback refinement method {method!r}")
    copied = dict(fixed64)
    copied.update(
        {
            "candidate_key": refinement_key(str(manifest_row["row_key"]), method),
            "candidate_kind": method,
            "control_label": method,
            "case_index": manifest_row["case_index"],
            "source_row_id": manifest_row["source_row_id"],
            "task_group": manifest_row["task_group"],
            "length_bucket": manifest_row["length_bucket"],
            "reference_middle_tokens": manifest_row["reference_middle_tokens"],
            "selected_base_candidate_key": selected_base["candidate_key"],
            "selected_base_middle_sha256": selected_base.get("candidate_middle_sha256", ""),
            "fallback_to_fixed64": True,
            "fallback_reason": reason,
            "remasked_token_indices": [],
            "remasked_token_count": 0,
            "middle_token_ids": fixed64.get("middle_token_ids") or [],
            "final_token_confidences": fixed64.get("final_token_confidences") or [],
            "selection_diagnostics": dict(selection_diagnostics),
        }
    )
    metrics = dict(copied.get("metrics") or {})
    forward = int(metrics.get("actual_forward_count") or copied.get("total_steps") or 0)
    canvas = int(copied["canvas_tokens"])
    wall = float(metrics.get("total_sec_including_probe") or copied.get("wall_sec") or 0.0)
    copied["metrics"] = {
        **metrics,
        "stage1_forward_count": forward,
        "refinement_forward_count": 0,
        "actual_forward_count": forward,
        "stage1_token_budget": canvas * forward,
        "refinement_token_budget": 0,
        "token_budget": canvas * forward,
        "stage1_wall_sec": wall,
        "refinement_wall_sec": 0.0,
        "total_sec_including_probe": wall,
        "refinement_executed": False,
    }
    return copied


def _error_row(
    *,
    method: str,
    manifest_row: Mapping[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "candidate_key": refinement_key(str(manifest_row["row_key"]), method),
        "row_key": manifest_row["row_key"],
        "candidate_kind": method,
        "control_label": method,
        "deployable": True,
        "case_index": manifest_row["case_index"],
        "source_row_id": manifest_row["source_row_id"],
        "task_group": manifest_row["task_group"],
        "length_bucket": manifest_row["length_bucket"],
        "reference_middle_tokens": manifest_row["reference_middle_tokens"],
        "status": "error",
        "passed": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:240],
        "failure_traceback": traceback.format_exc(),
        "metrics": {},
        "verification": {},
        "diagnostics": {},
    }


def build_refinement_rows_for_case(
    *,
    manifest_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    stage1_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    model: Any,
    cfg_for: Any,
    set_seed: Any,
) -> list[dict[str, Any]]:
    """Build generic and M1 rows.  Selection never sees task identity or outcomes."""
    context = next((row for row in stage1_rows if row.get("prefix_text") is not None and row.get("suffix_text") is not None), stage1_rows[0])
    prefix = str(context.get("prefix_text") or "")
    suffix = str(context.get("suffix_text") or "")
    if not prefix and not suffix:
        raise RuntimeError("Stage-one candidate rows lack the deployable infilling context")
    selected_input, selection_diagnostics = _selection_diagnostics(prefix, suffix, stage1_rows)
    ordered_stage1 = sorted(stage1_rows, key=lambda item: (int(item["canvas_tokens"]), int(item["seed"])))
    selected_base = dict(ordered_stage1[int(selected_input["candidate_ordinal"])])
    fixed64 = next(
        row for row in stage1_rows if int(row["canvas_tokens"]) == 64 and int(row["seed"]) == 0
    )
    if str(selected_base.get("status")) != "ok":
        selected_base = dict(fixed64)
        selection_diagnostics = {
            **selection_diagnostics,
            "selection_stage1_non_ok_fallback": True,
        }
    token_ids = selected_base.get("middle_token_ids")
    confidences = selected_base.get("final_token_confidences")
    canvas = int(selected_base["canvas_tokens"])
    if not isinstance(token_ids, list) or not isinstance(confidences, list) or len(token_ids) != canvas or len(confidences) != canvas:
        reason = "selected_stage1_state_unavailable"
        generic = _fallback_row(
            manifest_row=manifest_row,
            fixed64=fixed64,
            selected_base=selected_base,
            selection_diagnostics=selection_diagnostics,
            reason=reason,
            method="equal_compute_generic_remask",
        )
        m1 = _fallback_row(
            manifest_row=manifest_row,
            fixed64=fixed64,
            selected_base=selected_base,
            selection_diagnostics=selection_diagnostics,
            reason=reason,
        )
        return [generic, m1]

    outputs: list[dict[str, Any]] = []
    task = code_task({
        "task_id": "phase6_runtime_task",
        "prompt": prefix,
        "suffix": suffix,
        "test": str(source_row["test"]),
        "entry_point": str(source_row["entry_point"]),
    })
    try:
        generic_indices = generic_low_confidence_indices(confidences, canvas)
        set_seed(int(selected_base["seed"]))
        generic_result = decode_fixed_canvas_state(
            task=task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg_for(canvas, int(selected_base["seed"])),
            canvas_tokens=canvas,
            total_steps=64,
            phase_name="stage2_equal_compute_generic_remask",
            initial_middle_ids=token_ids,
            initial_mask_indices=generic_indices,
            schedule_length=len(generic_indices),
        )
        outputs.append(
            _result_row(
                method="equal_compute_generic_remask",
                manifest_row=manifest_row,
                selected_base=selected_base,
                result=generic_result,
                remask_indices=generic_indices,
                selection_diagnostics=selection_diagnostics,
                fallback_to_fixed64=False,
                fallback_reason="",
            )
        )
    except Exception as exc:
        outputs.append(_error_row(method="equal_compute_generic_remask", manifest_row=manifest_row, exc=exc))

    try:
        obligations = extract_backward_obligations(prefix, suffix)
        cone_indices = dependency_cone_token_indices(tokenizer, token_ids, obligations.dependency_names)
        if not obligations.dependency_names:
            outputs.append(
                _fallback_row(
                    manifest_row=manifest_row,
                    fixed64=fixed64,
                    selected_base=selected_base,
                    selection_diagnostics=selection_diagnostics,
                    reason="no_suffix_dependency_obligation",
                )
            )
        elif not cone_indices:
            outputs.append(
                _fallback_row(
                    manifest_row=manifest_row,
                    fixed64=fixed64,
                    selected_base=selected_base,
                    selection_diagnostics=selection_diagnostics,
                    reason="dependency_cone_not_mappable_to_candidate_tokens",
                )
            )
        else:
            set_seed(int(selected_base["seed"]))
            m1_result = decode_fixed_canvas_state(
                task=task,
                tokenizer=tokenizer,
                model=model,
                cfg=cfg_for(canvas, int(selected_base["seed"])),
                canvas_tokens=canvas,
                total_steps=64,
                phase_name="stage2_m1_dependency_cone_remask",
                initial_middle_ids=token_ids,
                initial_mask_indices=cone_indices,
                schedule_length=len(cone_indices),
            )
            outputs.append(
                _result_row(
                    method="m1_dependency_cone_remask",
                    manifest_row=manifest_row,
                    selected_base=selected_base,
                    result=m1_result,
                    remask_indices=cone_indices,
                    selection_diagnostics=selection_diagnostics,
                    fallback_to_fixed64=False,
                    fallback_reason="",
                )
            )
    except Exception as exc:
        outputs.append(_error_row(method="m1_dependency_cone_remask", manifest_row=manifest_row, exc=exc))
    return outputs


def run_population(
    *,
    manifest: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    stage1_rows: Sequence[Mapping[str, Any]],
    raw_path: Path,
    tokenizer: Any,
    model: Any,
    cfg_for: Any,
    set_seed: Any,
    append_jsonl: Any,
) -> int:
    grid, _ = _index_rows(stage1_rows)
    existing = []
    if raw_path.exists():
        import json

        with raw_path.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    counts = Counter(str(row.get("candidate_key") or "") for row in existing)
    duplicate = [key for key, count in counts.items() if count > 1]
    if duplicate:
        raise RuntimeError(f"Refusing M1 refinement resume with duplicate keys: {duplicate[:5]}")
    completed = set(counts)
    written = 0
    for manifest_row in manifest:
        row_key = str(manifest_row["row_key"])
        missing_methods = [method for method in REFINEMENT_METHODS if refinement_key(row_key, method) not in completed]
        if not missing_methods:
            continue
        source = source_rows[int(manifest_row["source_row_id"])]
        generated = build_refinement_rows_for_case(
            manifest_row=manifest_row,
            source_row=source,
            stage1_rows=grid[row_key],
            tokenizer=tokenizer,
            model=model,
            cfg_for=cfg_for,
            set_seed=set_seed,
        )
        for row in generated:
            key = str(row["candidate_key"])
            if key in completed:
                continue
            append_jsonl(raw_path, row)
            completed.add(key)
            written += 1
    return written
