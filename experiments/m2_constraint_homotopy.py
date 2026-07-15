#!/usr/bin/env python3
"""M2 Constraint-Homotopy V0 on non-frozen RandomSpanLight rows.

Gradual and abrupt schedules use identical fixed canvas, seed, and 64-forward
budgets. Their only distinction is how strongly inference-visible prefix/suffix
and current-candidate constraints influence which currently masked tokens stay
masked at each denoising step.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.phase6_abductive_bridge_v1 import extract_backward_obligations
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.decode import linear_target_masks, prepare_model_inputs
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.verifier import parse_compile_diagnostics
from experiments.deployable_visible_task import (
    evaluate_completion_after_decode,
    visible_task,
)
from analysis.phase6_abductive_bridge_v1 import analyze_candidate
from experiments.phase6_remask import decoded_token_ranges
from experiments.method_population_schedule import (
    RANDOMSPANLIGHT_ALLOWED_CASES,
    SMOKE_CASES,
    population_schedule,
    require_randomspanlight_full,
)


MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
SOURCE_CONFIG = "HumanEval-RandomSpanInfillingLight"
CANVAS_TOKENS = 64
TOTAL_STEPS = 64
SEED = 0
METHODS = ("m2_vanilla_fixed64", "m2_gradual_constraints", "m2_abrupt_constraints")
TEST_LOCK = REPO / "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"
GROUPED_TEST_TASKS = REPO / "analysis_outputs/grouped_split_20260702_accel2/test_tasks.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_group(task_id: str) -> str:
    match = re.search(r"HumanEval/(\d+)", task_id)
    if not match:
        raise ValueError(f"Cannot derive HumanEval group from {task_id!r}")
    return f"HumanEval/{match.group(1)}"


def length_bucket(reference_text: str, tokenizer: Any) -> str:
    count = len(tokenizer.encode(reference_text, add_special_tokens=False))
    if count <= 8:
        return "short"
    if count <= 16:
        return "medium"
    if count <= 24:
        return "long"
    return "extreme"


def load_frozen_groups() -> tuple[set[str], dict[str, Any]]:
    lock = json.loads(TEST_LOCK.read_text(encoding="utf-8"))
    if lock.get("test_status") != "sealed" or int(lock.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError("Frozen controller test must remain sealed with count zero")
    groups = {str(item) for item in lock.get("test_task_ids", [])}
    groups.update(str(item) for item in json.loads(GROUPED_TEST_TASKS.read_text(encoding="utf-8")))
    return groups, lock


def build_manifest(rows: Sequence[Mapping[str, Any]], tokenizer: Any, frozen_groups: set[str]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_row_id, row in enumerate(rows):
        group = task_group(str(row["task_id"]))
        if group in frozen_groups:
            continue
        if group in seen:
            raise RuntimeError(f"RandomSpanLight has duplicate allowed group {group}")
        seen.add(group)
        reference = str(row.get("canonical_solution") or "")
        reference_tokens = len(tokenizer.encode(reference, add_special_tokens=False))
        if reference_tokens <= 0:
            raise RuntimeError("Offline manifest has empty reference length")
        manifest.append(
            {
                "case_index": len(manifest),
                "source_row_id": source_row_id,
                "row_key": f"{SOURCE_CONFIG}:{source_row_id}:{sha256_text(str(row['task_id']))[:16]}",
                "task_group": group,
                "length_bucket": length_bucket(reference, tokenizer),
                "reference_middle_tokens": reference_tokens,
                "frozen_controller_test_row": False,
            }
        )
    return manifest


def choose_smoke_manifest(manifest: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(manifest):
        raise ValueError(f"Invalid smoke case count {count}")
    chosen: list[dict[str, Any]] = []
    buckets = ("short", "medium", "long", "extreme")
    per_bucket, remainder = divmod(count, len(buckets))
    for index, bucket in enumerate(buckets):
        candidates = sorted(
            (dict(row) for row in manifest if row["length_bucket"] == bucket),
            key=lambda item: (int(item["reference_middle_tokens"]), int(item["case_index"])),
        )
        chosen.extend(candidates[: per_bucket + (1 if index < remainder else 0)])
    if len(chosen) != count:
        raise RuntimeError("Could not form the requested stratified smoke manifest")
    return sorted(chosen, key=lambda item: int(item["case_index"]))


def cfg_for() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = MODEL_PATH
    cfg.model.torch_dtype = "bfloat16"
    cfg.model.device_map = "auto"
    cfg.data.dataset_subset = SOURCE_CONFIG
    cfg.decode.mask_length_source = "fixed"
    cfg.decode.fixed_mask_length = CANVAS_TOKENS
    cfg.decode.total_steps = TOTAL_STEPS
    cfg.decode.seed = SEED
    cfg.decode.save_step_traces = False
    cfg.decode.save_full_text_per_step = False
    return cfg


def method_key(row_key: str, method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"Unknown M2 method {method!r}")
    return f"{row_key}|{method}|canvas={CANVAS_TOKENS}|seed={SEED}|steps={TOTAL_STEPS}"


def expected_keys(manifest: Sequence[Mapping[str, Any]], method: str) -> set[str]:
    return {method_key(str(row["row_key"]), method) for row in manifest}


def homotopy_weight(method: str, step: int, total_steps: int = TOTAL_STEPS) -> float:
    if method == "m2_vanilla_fixed64":
        return 0.0
    if method == "m2_gradual_constraints":
        return min(1.0, max(0.0, (int(step) + 1) / int(total_steps)))
    if method == "m2_abrupt_constraints":
        return 0.0 if int(step) < int(total_steps) // 2 else 1.0
    raise ValueError(f"Unknown M2 method {method!r}")


def visible_constraint_signals(
    *,
    prefix: str,
    suffix: str,
    tokenizer: Any,
    candidate_token_ids: Sequence[int],
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Return separate protection and repair signals from visible state only."""
    candidate, ranges = decoded_token_ranges(tokenizer, candidate_token_ids)
    obligations = extract_backward_obligations(prefix, suffix)
    diagnostic = analyze_candidate(prefix, candidate, suffix)
    parse_passed = bool(diagnostic.full_parse_passed)
    protection_bonus: list[float] = []
    repair_priority: list[float] = []
    required = set(obligations.dependency_names)
    unsatisfied = set(diagnostic.unsatisfied_obligations)
    for left, right in ranges:
        piece = candidate[left:right] if left >= 0 and right > left else ""
        protect = 0.0
        repair = 0.0
        if not parse_passed and any(mark in piece for mark in ("\n", ":", "(", ")", "[", "]", "{", "}")):
            repair += 1.0
        mentioned_required = [name for name in required if re.search(rf"\b{re.escape(name)}\b", piece)]
        if mentioned_required:
            if not parse_passed or any(name in unsatisfied for name in mentioned_required):
                repair += 1.0
            else:
                protect += 1.0
        if obligations.control_requirements and any(mark in piece for mark in ("if", "for", "while", "try", "except", "else", "finally", ":")):
            if parse_passed:
                protect += 0.5
            else:
                repair += 0.5
        protection_bonus.append(protect)
        repair_priority.append(repair)
    return protection_bonus, repair_priority, {
        "full_parse_passed": parse_passed,
        "suffix_dependency_count": len(required),
        "suffix_control_requirement_count": len(obligations.control_requirements),
        "unsatisfied_obligation_count": len(unsatisfied),
    }


def visible_constraint_scores(
    *, prefix: str, suffix: str, tokenizer: Any, candidate_token_ids: Sequence[int]
) -> tuple[list[float], dict[str, Any]]:
    """Compatibility view: repair priority minus protection for diagnostics."""
    protection, repair, meta = visible_constraint_signals(
        prefix=prefix,
        suffix=suffix,
        tokenizer=tokenizer,
        candidate_token_ids=candidate_token_ids,
    )
    return [repair_value - protect_value for protect_value, repair_value in zip(protection, repair)], meta


def choose_remask_positions(
    *,
    confidences: Sequence[float],
    protection_bonus: Sequence[float],
    repair_priority: Sequence[float],
    masked_positions: Sequence[int],
    target_masks: int,
    weight: float,
) -> list[int]:
    """Protect visible obligations and prioritize visibly broken positions."""
    ranked = sorted(
        (
            float(confidences[index])
            + float(weight) * float(protection_bonus[index])
            - float(weight) * float(repair_priority[index]),
            int(index),
        )
        for index in masked_positions
    )
    return [index for _, index in ranked[: min(int(target_masks), len(ranked))]]


def decode_constraint_homotopy(
    *,
    method: str,
    task: Any,
    tokenizer: Any,
    model: Any,
    evaluate_after_decode: Callable[[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run exactly 64 forwards; evaluator execution occurs after denoising."""
    cfg = cfg_for()
    prepared = prepare_model_inputs(task, tokenizer, CANVAS_TOKENS, cfg)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    middle_start = int(prepared["middle_start"])
    middle_end = int(prepared["middle_end"])
    mask_token_id = int(prepared["mask_token_id"])
    started = time.perf_counter()
    trace: list[dict[str, Any]] = []
    final_confidences = [0.0] * CANVAS_TOKENS
    total_changes = 0
    effective_update_steps = 0

    for step in range(TOTAL_STEPS):
        before_middle = x_t[0, middle_start:middle_end].detach().clone()
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probabilities = torch.softmax(logits, dim=-1)
        max_probs, predictions = torch.max(probabilities, dim=-1)
        current_mask = x_t == mask_token_id
        middle_mask = current_mask[:, middle_start:middle_end]
        middle_probs = max_probs[:, middle_start:middle_end]
        proposed_ids = x_t[0, middle_start:middle_end].detach().clone()
        proposed_ids[middle_mask[0]] = predictions[0, middle_start:middle_end][middle_mask[0]]
        protection_bonus, repair_priority, constraint_meta = visible_constraint_signals(
            prefix=task.prefix,
            suffix=task.suffix,
            tokenizer=tokenizer,
            candidate_token_ids=[int(value) for value in proposed_ids.detach().cpu().tolist()],
        )
        target_masks = linear_target_masks(CANVAS_TOKENS, TOTAL_STEPS, step)
        weight = homotopy_weight(method, step)
        masked_positions = torch.nonzero(middle_mask[0], as_tuple=False).flatten().tolist()
        retain = choose_remask_positions(
            confidences=[float(value) for value in middle_probs[0].detach().cpu().tolist()],
            protection_bonus=protection_bonus,
            repair_priority=repair_priority,
            masked_positions=masked_positions,
            target_masks=target_masks,
            weight=weight,
        )
        x_t[current_mask] = predictions[current_mask]
        if retain:
            x_t[0, [middle_start + index for index in retain]] = mask_token_id
        after_middle = x_t[0, middle_start:middle_end].detach().clone()
        changes = int((after_middle != before_middle).sum().item())
        total_changes += changes
        effective_update_steps += int(changes > 0)
        final_confidences = [float(value) for value in middle_probs[0].detach().cpu().tolist()]
        trace.append(
            {
                "step": step,
                "target_masks": target_masks,
                "homotopy_weight": weight,
                "remaining_masks_after": int((after_middle == mask_token_id).sum().item()),
                "constraint": constraint_meta,
                "protection_bonus": protection_bonus,
                "repair_priority": repair_priority,
                "remasked_positions": retain,
            }
        )

    decode_sec = time.perf_counter() - started
    middle_ids = [int(value) for value in x_t[0, middle_start:middle_end].detach().cpu().tolist()]
    prefix = tokenizer.decode(prepared["prefix_ids"], skip_special_tokens=True)
    middle = tokenizer.decode(middle_ids, skip_special_tokens=True)
    suffix = tokenizer.decode(prepared["suffix_ids"], skip_special_tokens=True)
    code = prefix + middle + suffix
    evaluated = dict(evaluate_after_decode(prefix, middle, suffix))
    return {
        "code": code,
        "middle_text": middle,
        "middle_token_ids": middle_ids,
        "final_token_confidences": final_confidences,
        "metrics": {
            "passed": bool(evaluated.get("passed", False)),
            "decode_sec": decode_sec,
            "verification_sec": float(evaluated.get("verification_sec") or 0.0),
            "total_sec_including_probe": decode_sec + float(evaluated.get("verification_sec") or 0.0),
            "actual_forward_count": TOTAL_STEPS,
            "token_budget": CANVAS_TOKENS * TOTAL_STEPS,
            "effective_update_steps": effective_update_steps,
            "total_token_changes": total_changes,
            "canvas_tokens": CANVAS_TOKENS,
        },
        "diagnostics": {
            "final_full_code": parse_compile_diagnostics(code, compile_mode="exec"),
            "final_constraint": trace[-1]["constraint"],
        },
        "verification": evaluated.get("verification") or {},
        "trajectory": trace,
    }


def result_row(
    *,
    method: str,
    manifest_row: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    middle = str(result["middle_text"])
    code = str(result["code"])
    return {
        "candidate_key": method_key(str(manifest_row["row_key"]), method),
        "row_key": manifest_row["row_key"],
        "candidate_kind": method,
        "control_label": method,
        "deployable": True,
        "case_index": manifest_row["case_index"],
        "source_row_id": manifest_row["source_row_id"],
        "task_group": manifest_row["task_group"],
        "length_bucket": manifest_row["length_bucket"],
        "reference_middle_tokens": manifest_row["reference_middle_tokens"],
        "canvas_tokens": CANVAS_TOKENS,
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
        "status": "ok",
        "passed": bool((result.get("metrics") or {}).get("passed", False)),
        "candidate_middle_sha256": sha256_text(middle),
        "candidate_full_code_sha256": sha256_text(code),
        "middle_token_ids": result.get("middle_token_ids") or [],
        "final_token_confidences": result.get("final_token_confidences") or [],
        "metrics": result.get("metrics") or {},
        "diagnostics": result.get("diagnostics") or {},
        "verification": result.get("verification") or {},
        "middle_text": middle,
        "code": code,
    }


def error_row(method: str, manifest_row: Mapping[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "candidate_key": method_key(str(manifest_row["row_key"]), method),
        "row_key": manifest_row["row_key"],
        "candidate_kind": method,
        "control_label": method,
        "case_index": manifest_row["case_index"],
        "source_row_id": manifest_row["source_row_id"],
        "task_group": manifest_row["task_group"],
        "length_bucket": manifest_row["length_bucket"],
        "reference_middle_tokens": manifest_row["reference_middle_tokens"],
        "canvas_tokens": CANVAS_TOKENS,
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
        "status": "error",
        "passed": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:240],
        "failure_traceback": traceback.format_exc(),
        "metrics": {},
        "diagnostics": {},
        "verification": {},
    }


def run_method(
    *,
    method: str,
    manifest: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    raw_path: Path,
    tokenizer: Any,
    model: Any,
) -> int:
    if method not in METHODS:
        raise ValueError(f"Unknown M2 method {method!r}")
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    counts = Counter(str(row.get("candidate_key") or "") for row in existing)
    duplicate = [key for key, count in counts.items() if count > 1]
    if duplicate:
        raise RuntimeError(f"Refusing M2 resume with duplicate keys: {duplicate[:5]}")
    if any(row.get("candidate_kind") != method for row in existing):
        raise RuntimeError("M2 method output directory contains another method")
    completed = set(counts)
    written = 0
    for item in manifest:
        key = method_key(str(item["row_key"]), method)
        if key in completed:
            continue
        source = source_rows[int(item["source_row_id"])]
        try:
            task = visible_task(str(source["prompt"]), str(source["suffix"]), method="m2")
            set_global_seed(SEED)
            row = result_row(
                method=method,
                manifest_row=item,
                result=decode_constraint_homotopy(
                    method=method,
                    task=task,
                    tokenizer=tokenizer,
                    model=model,
                    evaluate_after_decode=lambda prefix, middle, suffix: evaluate_completion_after_decode(
                        prefix=prefix,
                        suffix=suffix,
                        middle=middle,
                        source_row=source,
                        method="m2",
                    ),
                ),
            )
        except Exception as exc:
            row = error_row(method, item, exc)
        append_jsonl(raw_path, row)
        completed.add(key)
        written += 1
    return written


def audit_rows(rows: Sequence[Mapping[str, Any]], expected: set[str]) -> dict[str, Any]:
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    errors = sorted(str(row.get("candidate_key") or "") for row in rows if row.get("status") != "ok")
    forward_bad = sorted(
        str(row.get("candidate_key") or "")
        for row in rows
        if row.get("status") == "ok" and int((row.get("metrics") or {}).get("actual_forward_count") or 0) != TOTAL_STEPS
    )
    return {
        "passed": not missing and not extra and not duplicates and not errors and not forward_bad,
        "expected_count": len(expected),
        "observed_count": len(rows),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "duplicate_count": len(duplicates),
        "error_count": len(errors),
        "non64_forward_count": len(forward_bad),
        "missing_keys": missing[:100],
        "extra_keys": extra[:100],
        "duplicate_keys": duplicates[:100],
        "error_keys": errors[:100],
        "non64_forward_keys": forward_bad[:100],
    }


def compact_summary(
    *,
    compact_dir: Path,
    phase: str,
    selected_manifest: Sequence[Mapping[str, Any]],
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    test_lock: Mapping[str, Any],
    resume_noops: int,
    peak_memory_bytes: int,
    wall_sec: float,
) -> dict[str, Any]:
    audits = {
        method: audit_rows(rows_by_method[method], expected_keys(selected_manifest, method))
        for method in METHODS
    }
    frozen_ok = test_lock.get("test_status") == "sealed" and int(test_lock.get("test_evaluation_count", -1)) == 0
    gate_passed = bool(all(audit["passed"] for audit in audits.values()) and frozen_ok and resume_noops == 0 and peak_memory_bytes > 0)
    rows = [row for method_rows in rows_by_method.values() for row in method_rows]
    compact_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        compact_dir / f"{phase}_results.csv",
        [
            {
                "candidate_key": row.get("candidate_key"),
                "row_key": row.get("row_key"),
                "task_group": row.get("task_group"),
                "length_bucket": row.get("length_bucket"),
                "method": row.get("candidate_kind"),
                "status": row.get("status"),
                "passed": row.get("passed"),
                "actual_forward_count": (row.get("metrics") or {}).get("actual_forward_count"),
                "token_budget": (row.get("metrics") or {}).get("token_budget"),
                "wall_sec": (row.get("metrics") or {}).get("total_sec_including_probe"),
            }
            for row in rows
        ],
    )
    summary = {
        "phase": phase,
        "technical_gate_passed": gate_passed,
        "selected_case_count": len(selected_manifest),
        "methods": audits,
        "frozen_test_status": test_lock.get("test_status"),
        "test_evaluation_count": test_lock.get("test_evaluation_count"),
        "resume_noop_writes": resume_noops,
        "peak_cuda_memory_bytes": peak_memory_bytes,
        "wall_sec": wall_sec,
        "deployable_input_boundary": "prefix_suffix_current_candidate_inference_visible_confidence_only",
    }
    write_json(compact_dir / f"{phase}_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> int:
    dataset_jsonl = Path(args.dataset_jsonl).resolve()
    vanilla_dir = Path(args.vanilla_output_dir).resolve()
    gradual_dir = Path(args.gradual_output_dir).resolve()
    abrupt_dir = Path(args.abrupt_output_dir).resolve()
    compact_dir = Path(args.compact_dir).resolve()
    if len({vanilla_dir, gradual_dir, abrupt_dir}) != 3:
        raise ValueError("M2 vanilla, gradual, and abrupt methods require independent output directories")
    raw_by_method = {
        METHODS[0]: vanilla_dir / "m2_vanilla_raw.jsonl",
        METHODS[1]: gradual_dir / "m2_gradual_raw.jsonl",
        METHODS[2]: abrupt_dir / "m2_abrupt_raw.jsonl",
    }
    frozen_groups, lock = load_frozen_groups()
    source_rows = read_jsonl(dataset_jsonl)
    if len(source_rows) != 164:
        raise RuntimeError(f"Expected 164 RandomSpanLight source rows, found {len(source_rows)}")
    tokenizer, model = load_model_and_tokenizer(cfg_for().model)
    manifest = build_manifest(source_rows, tokenizer, frozen_groups)
    if len(manifest) != RANDOMSPANLIGHT_ALLOWED_CASES or len({row["task_group"] for row in manifest}) != RANDOMSPANLIGHT_ALLOWED_CASES:
        raise RuntimeError("M2 requires exactly 148 allowed RandomSpanLight task groups")
    selected = choose_smoke_manifest(manifest, int(args.smoke_cases))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for M2")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for method in METHODS:
        run_method(method=method, manifest=selected, source_rows=source_rows, raw_path=raw_by_method[method], tokenizer=tokenizer, model=model)
    smoke_noops = sum(
        run_method(method=method, manifest=selected, source_rows=source_rows, raw_path=raw_by_method[method], tokenizer=tokenizer, model=model)
        for method in METHODS
    )
    smoke = compact_summary(
        compact_dir=compact_dir,
        phase="smoke",
        selected_manifest=selected,
        rows_by_method={method: read_jsonl(raw_by_method[method]) for method in METHODS},
        test_lock=lock,
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
        for method in METHODS:
            run_method(method=method, manifest=manifest, source_rows=source_rows, raw_path=raw_by_method[method], tokenizer=tokenizer, model=model)
        full_noops = sum(
            run_method(method=method, manifest=manifest, source_rows=source_rows, raw_path=raw_by_method[method], tokenizer=tokenizer, model=model)
            for method in METHODS
        )
        full = compact_summary(
            compact_dir=compact_dir,
            phase="full",
            selected_manifest=manifest,
            rows_by_method={method: read_jsonl(raw_by_method[method]) for method in METHODS},
            test_lock=lock,
            resume_noops=full_noops,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
            wall_sec=time.perf_counter() - started,
        )
    write_json(
        compact_dir / "run_manifest.json",
        {
            "status": "completed" if full is None or full["technical_gate_passed"] else "full_failed",
            "started_at_utc": utc_now(),
            "model": MODEL_PATH,
            "canvas_tokens": CANVAS_TOKENS,
            "total_steps": TOTAL_STEPS,
            "methods": list(METHODS),
            "raw_by_method": {method: str(path) for method, path in raw_by_method.items()},
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
    root = argparse.ArgumentParser(description="M2 Constraint-Homotopy V0")
    root.add_argument("--dataset-jsonl", required=True)
    root.add_argument("--vanilla-output-dir", required=True)
    root.add_argument("--gradual-output-dir", required=True)
    root.add_argument("--abrupt-output-dir", required=True)
    root.add_argument("--compact-dir", required=True)
    root.add_argument("--smoke-cases", type=int, default=SMOKE_CASES)
    root.add_argument("--auto-full", action="store_true")
    return root


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
