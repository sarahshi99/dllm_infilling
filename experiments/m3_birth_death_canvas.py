#!/usr/bin/env python3
"""M3 Birth-Death Canvas Diffusion V0.

The method starts with 16/32/64/128 canvas particles and spends exactly 256
model forwards per task. Birth/death decisions are made only from visible
confidence, syntax, and prefix/suffix compatibility; the uniform fixed-grid
control spends the same 4 x 64 forward budget without reallocation.
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

from analysis.phase6_abductive_bridge_v1 import analyze_candidate
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask
from expvision_dllm_clean.decode import linear_target_masks, prepare_model_inputs
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.verifier import run_verifier_stack
from experiments.m2_constraint_homotopy import (
    METHODS as M2_METHODS,
    append_jsonl,
    build_manifest,
    choose_smoke_manifest,
    load_frozen_groups,
    read_jsonl,
    sha256_text,
    write_csv,
    write_json,
)
from experiments.phase6_remask import decoded_token_ranges


del M2_METHODS
MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
SOURCE_CONFIG = "HumanEval-RandomSpanInfillingLight"
CANVASES = (16, 32, 64, 128)
STEPS_PER_PARTICLE = 64
TOTAL_FORWARDS = len(CANVASES) * STEPS_PER_PARTICLE
SEED = 0
METHODS = ("m3_uniform_fixed_grid", "m3_birth_death_canvas")
DEATH_ROUNDS = (15, 31, 47)


def cfg_for(canvas_tokens: int) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = MODEL_PATH
    cfg.model.torch_dtype = "bfloat16"
    cfg.model.device_map = "auto"
    cfg.data.dataset_subset = SOURCE_CONFIG
    cfg.decode.mask_length_source = "fixed"
    cfg.decode.fixed_mask_length = int(canvas_tokens)
    cfg.decode.total_steps = STEPS_PER_PARTICLE
    cfg.decode.seed = SEED
    cfg.decode.save_step_traces = False
    cfg.decode.save_full_text_per_step = False
    return cfg


def method_key(row_key: str, method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"Unknown M3 method {method!r}")
    return f"{row_key}|{method}|canvases=16,32,64,128|forwards={TOTAL_FORWARDS}|seed={SEED}"


def expected_keys(manifest: Sequence[Mapping[str, Any]], method: str) -> set[str]:
    return {method_key(str(row["row_key"]), method) for row in manifest}


def visible_task(prefix: str, suffix: str) -> CodeTask:
    """Model input contains no evaluator test/reference state."""
    return CodeTask(
        task_id="m3_visible_state",
        prefix=prefix,
        suffix=suffix,
        full_prompt=prefix + "<FILL_ME>" + suffix,
        test_code="",
        entry_point="",
        canonical_solution="",
        raw={},
    )


def evaluator_task(prefix: str, suffix: str, source_row: Mapping[str, Any]) -> CodeTask:
    return CodeTask(
        task_id="m3_runtime_evaluator_task",
        prefix=prefix,
        suffix=suffix,
        full_prompt=prefix + "<FILL_ME>" + suffix,
        test_code=str(source_row["test"]),
        entry_point=str(source_row["entry_point"]),
        canonical_solution="",
        raw={},
    )


def _device(model: Any) -> Any:
    device = getattr(model, "device", None)
    return device if device is not None else next(model.parameters()).device


def new_particle(
    *,
    canvas_tokens: int,
    task: CodeTask,
    tokenizer: Any,
    model: Any,
    born_round: int = 0,
    candidate_text: str = "",
) -> dict[str, Any]:
    cfg = cfg_for(canvas_tokens)
    prepared = prepare_model_inputs(task, tokenizer, canvas_tokens, cfg)
    device = _device(model)
    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    if candidate_text:
        ids = tokenizer.encode(candidate_text, add_special_tokens=False)[:canvas_tokens]
        middle_start, middle_end = int(prepared["middle_start"]), int(prepared["middle_end"])
        mask_token_id = int(prepared["mask_token_id"])
        filled = ids + [mask_token_id] * (canvas_tokens - len(ids))
        x_t[0, middle_start:middle_end] = torch.tensor(filled, dtype=torch.long, device=device)
    return {
        "canvas_tokens": int(canvas_tokens),
        "prepared": prepared,
        "x_t": x_t,
        "born_round": int(born_round),
        "age": 0,
        "lifetime_steps": STEPS_PER_PARTICLE - int(born_round),
        "last_confidences": [0.0] * int(canvas_tokens),
    }


def particle_middle_ids(particle: Mapping[str, Any]) -> list[int]:
    prepared = particle["prepared"]
    return [
        int(value)
        for value in particle["x_t"][0, int(prepared["middle_start"]): int(prepared["middle_end"])].detach().cpu().tolist()
    ]


def particle_text(particle: Mapping[str, Any], tokenizer: Any) -> str:
    return tokenizer.decode(particle_middle_ids(particle), skip_special_tokens=True)


def visible_particle_score(particle: Mapping[str, Any], prefix: str, suffix: str, tokenizer: Any) -> tuple[int, int, int, float, int]:
    middle = particle_text(particle, tokenizer)
    diagnostic = analyze_candidate(prefix, middle, suffix)
    confidences = particle.get("last_confidences") or []
    confidence = sum(float(value) for value in confidences) / len(confidences) if confidences else 0.0
    return (
        int(diagnostic.full_parse_passed),
        -len(diagnostic.unsatisfied_obligations),
        -len(diagnostic.control_contradictions) - len(diagnostic.def_use_conflicts),
        confidence,
        -int(particle["canvas_tokens"]),
    )


def step_particle(
    *,
    particle: dict[str, Any],
    prefix: str,
    suffix: str,
    tokenizer: Any,
    model: Any,
) -> None:
    prepared = particle["prepared"]
    x_t = particle["x_t"]
    middle_start, middle_end = int(prepared["middle_start"]), int(prepared["middle_end"])
    mask_token_id = int(prepared["mask_token_id"])
    with torch.no_grad():
        outputs = model(x_t)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    probabilities = torch.softmax(logits, dim=-1)
    max_probs, predictions = torch.max(probabilities, dim=-1)
    current_mask = x_t == mask_token_id
    middle_mask = current_mask[:, middle_start:middle_end]
    middle_probs = max_probs[:, middle_start:middle_end]
    proposed = x_t[0, middle_start:middle_end].detach().clone()
    proposed[middle_mask[0]] = predictions[0, middle_start:middle_end][middle_mask[0]]
    proposed_text = tokenizer.decode([int(value) for value in proposed.detach().cpu().tolist()], skip_special_tokens=True)
    try:
        syntax_ok = bool(ast.parse(prefix + proposed_text + suffix))
    except SyntaxError:
        syntax_ok = False
    diagnostic = analyze_candidate(prefix, proposed_text, suffix)
    penalty = 0.25 * (not syntax_ok) + 0.10 * len(diagnostic.unsatisfied_obligations) + 0.05 * len(diagnostic.control_contradictions)
    remaining = max(1, int(particle["lifetime_steps"]) - int(particle["age"]))
    target_masks = linear_target_masks(int(particle["canvas_tokens"]), int(particle["lifetime_steps"]), int(particle["age"]))
    masked_positions = torch.nonzero(middle_mask[0], as_tuple=False).flatten().tolist()
    ranked = sorted((float(middle_probs[0, index].item()) - penalty, int(index)) for index in masked_positions)
    retain = [index for _, index in ranked[: min(target_masks, len(ranked))]]
    x_t[current_mask] = predictions[current_mask]
    if retain:
        x_t[0, [middle_start + index for index in retain]] = mask_token_id
    particle["last_confidences"] = [float(value) for value in middle_probs[0].detach().cpu().tolist()]
    particle["age"] = int(particle["age"]) + 1
    particle["last_visible_diagnostic"] = {
        "syntax_ok": syntax_ok,
        "unsatisfied_obligation_count": len(diagnostic.unsatisfied_obligations),
        "control_contradiction_count": len(diagnostic.control_contradictions),
        "remaining_local_steps": remaining,
    }


def decode_population(
    *,
    method: str,
    prefix: str,
    suffix: str,
    tokenizer: Any,
    model: Any,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown M3 method {method!r}")
    task = visible_task(prefix, suffix)
    particles = [new_particle(canvas_tokens=canvas, task=task, tokenizer=tokenizer, model=model) for canvas in CANVASES]
    events: list[dict[str, Any]] = []
    forwards = 0
    for round_index in range(STEPS_PER_PARTICLE):
        for particle in particles:
            step_particle(particle=particle, prefix=prefix, suffix=suffix, tokenizer=tokenizer, model=model)
            forwards += 1
        if method == "m3_birth_death_canvas" and round_index in DEATH_ROUNDS:
            ranked = sorted(
                enumerate(particles),
                key=lambda item: visible_particle_score(item[1], prefix, suffix, tokenizer),
            )
            dead_index, dead = ranked[0]
            _, parent = ranked[-1]
            parent_score = visible_particle_score(parent, prefix, suffix, tokenizer)
            dead_score = visible_particle_score(dead, prefix, suffix, tokenizer)
            if parent_score > dead_score:
                parent_text = particle_text(parent, tokenizer)
                particles[dead_index] = new_particle(
                    canvas_tokens=int(parent["canvas_tokens"]),
                    task=task,
                    tokenizer=tokenizer,
                    model=model,
                    born_round=round_index + 1,
                    candidate_text=parent_text,
                )
                events.append(
                    {
                        "round": round_index,
                        "dead_canvas": int(dead["canvas_tokens"]),
                        "born_canvas": int(parent["canvas_tokens"]),
                        "parent_score": list(parent_score),
                        "dead_score": list(dead_score),
                    }
                )
    if forwards != TOTAL_FORWARDS:
        raise RuntimeError("M3 violated fixed total forward budget")
    selected = max(particles, key=lambda item: visible_particle_score(item, prefix, suffix, tokenizer))
    candidate = particle_text(selected, tokenizer)
    return {
        "middle_text": candidate,
        "middle_token_ids": particle_middle_ids(selected),
        "final_token_confidences": selected.get("last_confidences") or [],
        "selected_canvas_tokens": int(selected["canvas_tokens"]),
        "selected_visible_score": list(visible_particle_score(selected, prefix, suffix, tokenizer)),
        "particle_canvases_final": [int(item["canvas_tokens"]) for item in particles],
        "birth_death_events": events,
        "actual_forward_count": forwards,
    }


def result_row(method: str, item: Mapping[str, Any], source: Mapping[str, Any], decoded: Mapping[str, Any]) -> dict[str, Any]:
    prefix, suffix = str(source["prompt"]), str(source["suffix"])
    middle = str(decoded["middle_text"])
    task = evaluator_task(prefix, suffix, source)
    code = prefix + middle + suffix
    verification = run_verifier_stack(task=task, full_code=code, completion_without_suffix=middle)
    verification_sec = sum(value.duration_sec for value in verification.values())
    tier3 = verification.get("tier3_unit_tests")
    return {
        "candidate_key": method_key(str(item["row_key"]), method),
        "row_key": item["row_key"],
        "candidate_kind": method,
        "case_index": item["case_index"],
        "source_row_id": item["source_row_id"],
        "task_group": item["task_group"],
        "length_bucket": item["length_bucket"],
        "reference_middle_tokens": item["reference_middle_tokens"],
        "status": "ok",
        "passed": bool(tier3.passed) if tier3 else False,
        "canvas_tokens": int(decoded["selected_canvas_tokens"]),
        "seed": SEED,
        "total_steps": TOTAL_FORWARDS,
        "candidate_middle_sha256": sha256_text(middle),
        "candidate_full_code_sha256": sha256_text(code),
        "middle_text": middle,
        "code": code,
        "metrics": {
            "actual_forward_count": int(decoded["actual_forward_count"]),
            "token_budget_proxy": int(decoded["actual_forward_count"]) * int(decoded["selected_canvas_tokens"]),
            "verification_sec": verification_sec,
        },
        "selection": {
            "selected_visible_score": decoded["selected_visible_score"],
            "particle_canvases_final": decoded["particle_canvases_final"],
            "birth_death_events": decoded["birth_death_events"],
        },
        "verification": {name: value.to_dict() for name, value in verification.items()},
    }


def error_row(method: str, item: Mapping[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "candidate_key": method_key(str(item["row_key"]), method),
        "row_key": item["row_key"],
        "candidate_kind": method,
        "case_index": item["case_index"],
        "source_row_id": item["source_row_id"],
        "task_group": item["task_group"],
        "length_bucket": item["length_bucket"],
        "status": "error",
        "passed": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:240],
        "failure_traceback": traceback.format_exc(),
        "metrics": {},
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
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    counts = Counter(str(row.get("candidate_key") or "") for row in existing)
    if any(value > 1 for value in counts.values()):
        raise RuntimeError("M3 resume refuses duplicate candidate keys")
    if any(row.get("candidate_kind") != method for row in existing):
        raise RuntimeError("M3 method output directory contains another method")
    completed = set(counts)
    written = 0
    for item in manifest:
        key = method_key(str(item["row_key"]), method)
        if key in completed:
            continue
        source = source_rows[int(item["source_row_id"])]
        try:
            set_global_seed(SEED)
            start = time.perf_counter()
            decoded = decode_population(method=method, prefix=str(source["prompt"]), suffix=str(source["suffix"]), tokenizer=tokenizer, model=model)
            row = result_row(method, item, source, decoded)
            row["metrics"]["wall_sec"] = time.perf_counter() - start
        except Exception as exc:
            row = error_row(method, item, exc)
        append_jsonl(raw_path, row)
        completed.add(key)
        written += 1
    return written


def audit(rows: Sequence[Mapping[str, Any]], expected: set[str]) -> dict[str, Any]:
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    errors = [row for row in rows if row.get("status") != "ok"]
    nonbudget = [row for row in rows if row.get("status") == "ok" and int((row.get("metrics") or {}).get("actual_forward_count") or 0) != TOTAL_FORWARDS]
    return {
        "passed": observed == expected and all(value == 1 for value in counts.values()) and not errors and not nonbudget,
        "missing_count": len(expected - observed),
        "extra_count": len(observed - expected),
        "duplicate_count": sum(value > 1 for value in counts.values()),
        "error_count": len(errors),
        "nonbudget_count": len(nonbudget),
    }


def summarize(compact_dir: Path, phase: str, manifest: Sequence[Mapping[str, Any]], uniform_rows: Sequence[Mapping[str, Any]], birth_rows: Sequence[Mapping[str, Any]], lock: Mapping[str, Any], resume_noops: int) -> dict[str, Any]:
    left, right = audit(uniform_rows, expected_keys(manifest, METHODS[0])), audit(birth_rows, expected_keys(manifest, METHODS[1]))
    frozen = lock.get("test_status") == "sealed" and int(lock.get("test_evaluation_count", -1)) == 0
    gate = bool(left["passed"] and right["passed"] and frozen and resume_noops == 0)
    paired = {str(row["row_key"]): row for row in uniform_rows}
    wins = losses = 0
    for row in birth_rows:
        baseline = paired[str(row["row_key"])]
        wins += bool(row.get("passed")) and not bool(baseline.get("passed"))
        losses += not bool(row.get("passed")) and bool(baseline.get("passed"))
    compact_dir.mkdir(parents=True, exist_ok=True)
    write_csv(compact_dir / f"{phase}_summary.csv", [{"method": METHODS[0], **left}, {"method": METHODS[1], **right}])
    result = {
        "phase": phase,
        "technical_gate_passed": gate,
        "selected_case_count": len(manifest),
        "methods": {METHODS[0]: left, METHODS[1]: right},
        "paired_birth_death_vs_uniform": {"wins": wins, "losses": losses, "net": wins - losses},
        "fixed_total_forward_budget": TOTAL_FORWARDS,
        "frozen_test_status": lock.get("test_status"),
        "test_evaluation_count": lock.get("test_evaluation_count"),
    }
    write_json(compact_dir / f"{phase}_summary.json", result)
    return result


def run(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset_jsonl).resolve()
    uniform_dir, birth_dir, compact_dir = Path(args.uniform_output_dir).resolve(), Path(args.birth_death_output_dir).resolve(), Path(args.compact_dir).resolve()
    if uniform_dir == birth_dir:
        raise ValueError("M3 methods require independent output directories")
    frozen_groups, lock = load_frozen_groups()
    source_rows = read_jsonl(dataset)
    if len(source_rows) != 164:
        raise RuntimeError("M3 expects 164 RandomSpanLight source rows")
    tokenizer, model = load_model_and_tokenizer(cfg_for(64).model)
    manifest = build_manifest(source_rows, tokenizer, frozen_groups)
    if len(manifest) != 148:
        raise RuntimeError("M3 requires 148 allowed task groups")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for M3")
    selected = choose_smoke_manifest(manifest, int(args.smoke_cases))
    uniform_raw, birth_raw = uniform_dir / "m3_uniform_raw.jsonl", birth_dir / "m3_birth_death_raw.jsonl"
    run_method(method=METHODS[0], manifest=selected, source_rows=source_rows, raw_path=uniform_raw, tokenizer=tokenizer, model=model)
    run_method(method=METHODS[1], manifest=selected, source_rows=source_rows, raw_path=birth_raw, tokenizer=tokenizer, model=model)
    noops = run_method(method=METHODS[0], manifest=selected, source_rows=source_rows, raw_path=uniform_raw, tokenizer=tokenizer, model=model)
    noops += run_method(method=METHODS[1], manifest=selected, source_rows=source_rows, raw_path=birth_raw, tokenizer=tokenizer, model=model)
    smoke = summarize(compact_dir, "smoke", selected, read_jsonl(uniform_raw), read_jsonl(birth_raw), lock, noops)
    if not smoke["technical_gate_passed"]:
        write_json(compact_dir / "run_manifest.json", {"status": "smoke_failed", "smoke": smoke, "test_evaluation_count": 0})
        return 2
    full = None
    if args.auto_full:
        run_method(method=METHODS[0], manifest=manifest, source_rows=source_rows, raw_path=uniform_raw, tokenizer=tokenizer, model=model)
        run_method(method=METHODS[1], manifest=manifest, source_rows=source_rows, raw_path=birth_raw, tokenizer=tokenizer, model=model)
        noops = run_method(method=METHODS[0], manifest=manifest, source_rows=source_rows, raw_path=uniform_raw, tokenizer=tokenizer, model=model)
        noops += run_method(method=METHODS[1], manifest=manifest, source_rows=source_rows, raw_path=birth_raw, tokenizer=tokenizer, model=model)
        full = summarize(compact_dir, "full", manifest, read_jsonl(uniform_raw), read_jsonl(birth_raw), lock, noops)
    write_json(compact_dir / "run_manifest.json", {"status": "completed" if full is None or full["technical_gate_passed"] else "full_failed", "smoke": smoke, "full": full, "methods": list(METHODS), "total_forwards": TOTAL_FORWARDS, "test_evaluation_count": 0})
    return 0 if full is None or full["technical_gate_passed"] else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="M3 Birth-Death Canvas Diffusion V0")
    root.add_argument("--dataset-jsonl", required=True)
    root.add_argument("--uniform-output-dir", required=True)
    root.add_argument("--birth-death-output-dir", required=True)
    root.add_argument("--compact-dir", required=True)
    root.add_argument("--smoke-cases", type=int, default=12)
    root.add_argument("--auto-full", action="store_true")
    return root


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
