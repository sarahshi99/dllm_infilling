#!/usr/bin/env python3
"""Executable, isolated stage-two refinements for the M1.1 protocol.

The immutable stage-one bank is the only generation input. Selection and
remasking use a deliberately small deployable state view; evaluator test code
is constructed only after the target plan is fixed.
"""

from __future__ import annotations

import hashlib
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from analysis.phase6_abductive_bridge_v1 import extract_backward_obligations, select_v1_candidate
from experiments.phase5_randomspanlight_candidate_bank import ast_hash, code_task
from experiments.phase6_remask import (
    DependencyConePlan,
    decode_fixed_canvas_state,
    dependency_cone_plan,
    generic_low_confidence_indices,
    remask_count_for_canvas,
)


REFINEMENT_METHODS = (
    "equal_compute_generic_remask",
    "m1_dependency_cone_remask",
)


def refinement_key(row_key: str, method: str) -> str:
    if method not in REFINEMENT_METHODS:
        raise ValueError(f"Unknown refinement method {method!r}")
    return f"{row_key}|{method}|refinement=64"


def expected_refinement_keys(
    manifest: Sequence[Mapping[str, Any]],
    methods: Sequence[str] = REFINEMENT_METHODS,
) -> set[str]:
    requested = tuple(methods)
    if not set(requested) <= set(REFINEMENT_METHODS):
        raise ValueError("expected_refinement_keys received an unknown method")
    return {refinement_key(str(item["row_key"]), method) for item in manifest for method in requested}


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fixed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("candidate_kind") != "deployable_grid":
            continue
        copied = dict(row)
        row_key = str(copied["row_key"])
        grid[row_key].append(copied)
        canvas = copied.get("canvas_tokens")
        seed = copied.get("seed")
        if canvas is not None and seed is not None and int(canvas) == 64 and int(seed) == 0:
            fixed[row_key] = copied
    if not grid or any(len(items) != 8 for items in grid.values()):
        raise RuntimeError("M1 refinement requires exactly eight stage-one deployable candidates per row")
    if set(grid) != set(fixed):
        raise RuntimeError("M1 refinement requires a fixed64/seed0 stage-one candidate for every row")
    return grid, fixed


def _selection_inputs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Construct the only view permitted to deployable M1 candidate selection."""
    return [
        {
            "candidate_ordinal": ordinal,
            "middle_text": str(row.get("middle_text") or ""),
            "canvas_tokens": int(row["canvas_tokens"]),
            "seed": int(row["seed"]),
        }
        for ordinal, row in enumerate(sorted(rows, key=lambda item: (int(item["canvas_tokens"]), int(item["seed"]))))
    ]


def _deployable_state(row: Mapping[str, Any], prefix: str, suffix: str) -> dict[str, Any]:
    """Extract token state without touching labels, references, IDs, or split data."""
    token_ids = row.get("middle_token_ids")
    confidences = row.get("final_token_confidences")
    return {
        "candidate_key": str(row.get("candidate_key") or ""),
        "middle_text": str(row.get("middle_text") or ""),
        "middle_token_ids": list(token_ids) if isinstance(token_ids, list) else [],
        "final_token_confidences": list(confidences) if isinstance(confidences, list) else [],
        "canvas_tokens": int(row["canvas_tokens"]),
        "seed": int(row["seed"]),
        "status": str(row.get("status") or ""),
        "prefix_text": prefix,
        "suffix_text": suffix,
    }


def _selection_diagnostics(prefix: str, suffix: str, rows: Sequence[Mapping[str, Any]]) -> tuple[int, dict[str, Any]]:
    selected, diagnostics = select_v1_candidate(prefix, suffix, _selection_inputs(rows))
    ordinal = int(selected["candidate_ordinal"])
    diagnostic = diagnostics[ordinal]
    contradiction_set = sorted(
        [
            *[f"boundary:{item}" for item in diagnostic.boundary_violations],
            *[f"control:{item}" for item in diagnostic.control_contradictions],
            *[f"obligation:{item}" for item in diagnostic.unsatisfied_obligations],
            *[f"def_use:{item}" for item in diagnostic.def_use_conflicts],
            *[f"undefined:{item}" for item in diagnostic.undefined_uses],
        ]
    )
    return ordinal, {
        "full_parse_passed": diagnostic.full_parse_passed,
        "boundary_violation_count": len(diagnostic.boundary_violations),
        "control_contradiction_count": len(diagnostic.control_contradictions),
        "unsatisfied_obligation_count": len(diagnostic.unsatisfied_obligations),
        "def_use_conflict_count": len(diagnostic.def_use_conflicts),
        "undefined_use_count": len(diagnostic.undefined_uses),
        "restored_dependency_count": len(diagnostic.restored_dependencies),
        "forward_definitions": list(diagnostic.forward_definitions),
        "contradiction_set": contradiction_set,
    }


def build_deployable_refinement_plan(
    stage1_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any], dict[str, Any], DependencyConePlan]:
    """Select one stage-one state and build its suffix-to-candidate cone.

    This function is intentionally free of manifest metadata and evaluator
    inputs. It is the unit under forbidden-input tests.
    """
    ordered = sorted(stage1_rows, key=lambda item: (int(item["canvas_tokens"]), int(item["seed"])))
    context = next(
        (row for row in ordered if row.get("prefix_text") is not None and row.get("suffix_text") is not None),
        ordered[0],
    )
    prefix = str(context.get("prefix_text") or "")
    suffix = str(context.get("suffix_text") or "")
    if not prefix and not suffix:
        raise RuntimeError("Stage-one candidate rows lack deployable infilling context")
    selected_ordinal, diagnostics = _selection_diagnostics(prefix, suffix, ordered)
    selected_row = ordered[selected_ordinal]
    fixed64_row = next(row for row in ordered if int(row["canvas_tokens"]) == 64 and int(row["seed"]) == 0)
    if str(selected_row.get("status") or "") != "ok":
        selected_row = fixed64_row
        diagnostics = {**diagnostics, "selection_stage1_non_ok_fallback": True}
    selected_state = _deployable_state(selected_row, prefix, suffix)
    fixed64_state = _deployable_state(fixed64_row, prefix, suffix)
    token_ids = selected_state["middle_token_ids"]
    canvas = int(selected_state["canvas_tokens"])
    if len(token_ids) != canvas:
        # A missing stage-one state is not repaired with an oracle or a new
        # generation. We use fixed64 only if its saved deployable state exists.
        selected_row = fixed64_row
        selected_state = fixed64_state
        canvas = int(selected_state["canvas_tokens"])
        diagnostics = {**diagnostics, "selection_state_unavailable_fixed64_fallback": True}
    token_ids = selected_state["middle_token_ids"]
    if len(token_ids) != canvas:
        raise RuntimeError("M1 refinement cannot recover a saved deployable token state")
    obligations = extract_backward_obligations(prefix, suffix)
    cone = dependency_cone_plan(
        tokenizer,
        token_ids,
        prefix=prefix,
        suffix=suffix,
        dependency_names=obligations.dependency_names,
    )
    return selected_state, selected_row, fixed64_row, diagnostics, cone


def _stage1_forward_count(row: Mapping[str, Any]) -> int:
    metrics = row.get("metrics") or {}
    return int(metrics.get("actual_forward_count") or row.get("total_steps") or 0)


def _stage1_wall_sec(row: Mapping[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    return float(metrics.get("total_sec_including_probe") or row.get("wall_sec") or 0.0)


def _result_row(
    *,
    method: str,
    manifest_row: Mapping[str, Any],
    selected_base: Mapping[str, Any],
    selected_state: Mapping[str, Any],
    result: Mapping[str, Any],
    remask_indices: Sequence[int],
    selection_diagnostics: Mapping[str, Any],
    cone: DependencyConePlan,
    fallback_to_fixed64: bool,
    fallback_reason: str,
    targeted_remask_executed: bool,
    null_refinement_executed: bool,
) -> dict[str, Any]:
    metrics = dict(result.get("metrics") or {})
    middle = str(result.get("middle_text") or "")
    code = str(result.get("code") or "")
    stage1_forward = _stage1_forward_count(selected_base)
    refinement_forward = int(metrics.get("actual_forward_count") or 0)
    stage1_wall = _stage1_wall_sec(selected_base)
    refinement_wall = float(metrics.get("total_sec_including_probe") or 0.0)
    canvas = int(selected_state["canvas_tokens"])
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
        "seed": int(selected_state["seed"]),
        "total_steps": stage1_forward + refinement_forward,
        "selected_base_candidate_key": selected_state["candidate_key"],
        "selected_base_middle_sha256": hashlib.sha256(str(selected_state["middle_text"]).encode("utf-8")).hexdigest(),
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
        "targeted_remask_executed": bool(targeted_remask_executed),
        "null_refinement_executed": bool(null_refinement_executed),
        "selection_diagnostics": dict(selection_diagnostics),
        "dependency_cone": cone.to_dict(),
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
            "refinement_executed": refinement_forward == 64,
            "targeted_remask_executed": bool(targeted_remask_executed),
            "null_refinement_executed": bool(null_refinement_executed),
        },
        "verification": result.get("verification") or {},
        "diagnostics": result.get("diagnostics") or {},
        "middle_text": middle,
        "code": code,
    }


def _error_row(*, method: str, manifest_row: Mapping[str, Any], exc: Exception) -> dict[str, Any]:
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


def _evaluator_task(prefix: str, suffix: str, source_row: Mapping[str, Any]) -> Any:
    """Construct evaluator-only state after deployable planning is complete."""
    return code_task(
        {
            "task_id": "phase6_runtime_task",
            "prompt": prefix,
            "suffix": suffix,
            "test": str(source_row["test"]),
            "entry_point": str(source_row["entry_point"]),
        }
    )


def build_refinement_row_for_case(
    *,
    method: str,
    manifest_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    stage1_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    model: Any,
    cfg_for: Callable[[int, int], Any],
    set_seed: Callable[[int], None],
    decode_fn: Callable[..., Mapping[str, Any]] = decode_fixed_canvas_state,
    evaluator_task_factory: Callable[[str, str, Mapping[str, Any]], Any] = _evaluator_task,
) -> dict[str, Any]:
    """Execute one isolated generic or M1 row with exactly 64 stage-two forwards."""
    if method not in REFINEMENT_METHODS:
        raise ValueError(f"Unknown refinement method {method!r}")
    try:
        selected_state, selected_base, fixed64_base, diagnostics, cone = build_deployable_refinement_plan(stage1_rows, tokenizer)
        canvas = int(selected_state["canvas_tokens"])
        token_ids = selected_state["middle_token_ids"]
        confidences = selected_state["final_token_confidences"]
        state_valid = len(token_ids) == canvas and len(confidences) == canvas
        fallback_to_fixed64 = False
        fallback_reason = ""
        targeted_remask_executed = False
        null_refinement_executed = False

        if not state_valid:
            # This should only be reachable after a corrupt raw file. A null
            # refinement keeps the method's 64-forward accounting explicit.
            selected_base = fixed64_base
            selected_state = _deployable_state(fixed64_base, str(selected_state["prefix_text"]), str(selected_state["suffix_text"]))
            canvas = int(selected_state["canvas_tokens"])
            token_ids = selected_state["middle_token_ids"]
            confidences = selected_state["final_token_confidences"]
            fallback_to_fixed64 = True
            fallback_reason = "selected_stage1_state_unavailable"

        if method == "equal_compute_generic_remask":
            if not cone.executable:
                # Keep the control on the same safe fixed64 state used by M1
                # whenever a candidate cone cannot be deployed.
                selected_base = fixed64_base
                selected_state = _deployable_state(fixed64_base, str(selected_state["prefix_text"]), str(selected_state["suffix_text"]))
                canvas = int(selected_state["canvas_tokens"])
                token_ids = selected_state["middle_token_ids"]
                confidences = selected_state["final_token_confidences"]
                fallback_to_fixed64 = True
                fallback_reason = f"matched_m1_{cone.reason or 'dependency_cone_unavailable'}"
            count = len(cone.token_indices) if cone.executable else remask_count_for_canvas(canvas)
            remask_indices = generic_low_confidence_indices(confidences, canvas, remask_count=count)
            phase_name = "stage2_equal_compute_generic_remask"
        elif cone.executable:
            remask_indices = list(cone.token_indices)
            phase_name = "stage2_m1_dependency_cone_remask"
            targeted_remask_executed = True
        else:
            # Safe fallback is a true no-op state refinement: preserve the
            # fixed64 candidate while still running the required 64 forwards.
            selected_base = fixed64_base
            selected_state = _deployable_state(fixed64_base, str(selected_state["prefix_text"]), str(selected_state["suffix_text"]))
            canvas = int(selected_state["canvas_tokens"])
            token_ids = selected_state["middle_token_ids"]
            remask_indices = []
            fallback_to_fixed64 = True
            fallback_reason = cone.reason or "dependency_cone_not_mappable_to_candidate_tokens"
            phase_name = "stage2_m1_fixed64_null_refinement"
            null_refinement_executed = True

        if len(token_ids) != canvas:
            raise RuntimeError("M1 refinement has no saved deployable state for fixed64 fallback")
        # Evaluator-only task creation is deliberately after target selection,
        # cone construction, and generic/M1 remask choice.
        task = evaluator_task_factory(str(selected_state["prefix_text"]), str(selected_state["suffix_text"]), source_row)
        set_seed(int(selected_state["seed"]))
        result = decode_fn(
            task=task,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg_for(canvas, int(selected_state["seed"])),
            canvas_tokens=canvas,
            total_steps=64,
            phase_name=phase_name,
            initial_middle_ids=token_ids,
            initial_mask_indices=remask_indices,
            schedule_length=max(1, len(remask_indices)),
        )
        return _result_row(
            method=method,
            manifest_row=manifest_row,
            selected_base=selected_base,
            selected_state=selected_state,
            result=result,
            remask_indices=remask_indices,
            selection_diagnostics=diagnostics,
            cone=cone,
            fallback_to_fixed64=fallback_to_fixed64,
            fallback_reason=fallback_reason,
            targeted_remask_executed=targeted_remask_executed,
            null_refinement_executed=null_refinement_executed,
        )
    except Exception as exc:
        return _error_row(method=method, manifest_row=manifest_row, exc=exc)


def run_population(
    *,
    method: str,
    manifest: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    stage1_rows: Sequence[Mapping[str, Any]],
    raw_path: Path,
    tokenizer: Any,
    model: Any,
    cfg_for: Callable[[int, int], Any],
    set_seed: Callable[[int], None],
    append_jsonl: Callable[[Path, Mapping[str, Any]], None],
) -> int:
    """Resume one method into its own raw directory; reject any duplicate key."""
    if method not in REFINEMENT_METHODS:
        raise ValueError(f"Unknown refinement method {method!r}")
    grid, _ = _index_rows(stage1_rows)
    existing: list[dict[str, Any]] = []
    if raw_path.exists():
        import json

        with raw_path.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    counts = Counter(str(row.get("candidate_key") or "") for row in existing)
    duplicate = [key for key, count in counts.items() if count > 1]
    if duplicate:
        raise RuntimeError(f"Refusing {method} resume with duplicate keys: {duplicate[:5]}")
    wrong_method = [row.get("candidate_kind") for row in existing if row.get("candidate_kind") != method]
    if wrong_method:
        raise RuntimeError(f"Refusing {method} resume with rows for another method")
    completed = set(counts)
    written = 0
    for manifest_row in manifest:
        row_key = str(manifest_row["row_key"])
        key = refinement_key(row_key, method)
        if key in completed:
            continue
        source = source_rows[int(manifest_row["source_row_id"])]
        row = build_refinement_row_for_case(
            method=method,
            manifest_row=manifest_row,
            source_row=source,
            stage1_rows=grid[row_key],
            tokenizer=tokenizer,
            model=model,
            cfg_for=cfg_for,
            set_seed=set_seed,
        )
        append_jsonl(raw_path, row)
        completed.add(key)
        written += 1
    return written
