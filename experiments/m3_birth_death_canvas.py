#!/usr/bin/env python3
"""M3 Birth-Death Canvas Diffusion V0.

The method starts with 16/32/64/128 canvas particles and spends exactly 256
model forwards per task. Birth/death decisions are made only from visible
confidence, syntax, and prefix/suffix compatibility; the uniform fixed-grid
control spends the same 4 x 64 forward budget without reallocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.phase6_abductive_bridge_v1 import analyze_candidate
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.decode import linear_target_masks, prepare_model_inputs
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from experiments.deployable_visible_task import (
    evaluate_completion_after_decode,
    visible_task as make_visible_task,
)
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
from experiments.method_population_schedule import (
    RANDOMSPANLIGHT_ALLOWED_CASES,
    SMOKE_CASES,
    population_schedule,
    require_randomspanlight_full,
)


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


def remaining_lifetime_steps(born_round: int) -> int:
    if not 0 <= int(born_round) < STEPS_PER_PARTICLE:
        raise ValueError("M3 born_round must fall within the global denoising schedule")
    return STEPS_PER_PARTICLE - int(born_round)


def confidence_retained_positions(
    confidences: Sequence[float], masked_positions: Sequence[int], target_masks: int
) -> list[int]:
    """Use ordinary within-particle confidence for remasking.

    Syntax, visible obligations, and contradictions instead rank particles for
    birth/death and final selection. A scalar particle penalty cannot alter a
    token ordering, so it is deliberately absent here.
    """
    ranked = sorted((float(confidences[index]), int(index)) for index in masked_positions)
    return [index for _, index in ranked[: min(int(target_masks), len(ranked))]]


def birth_death_enabled(method: str, round_index: int) -> bool:
    return method == METHODS[1] and int(round_index) in DEATH_ROUNDS


def particle_round_token_forwards(particles: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(particle["canvas_tokens"]) for particle in particles)


def require_isolated_output_dirs(uniform_dir: Path, birth_dir: Path) -> None:
    if uniform_dir == birth_dir:
        raise ValueError("M3 methods require independent output directories")


def method_key(row_key: str, method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"Unknown M3 method {method!r}")
    return f"{row_key}|{method}|canvases=16,32,64,128|forwards={TOTAL_FORWARDS}|seed={SEED}"


def expected_keys(manifest: Sequence[Mapping[str, Any]], method: str) -> set[str]:
    return {method_key(str(row["row_key"]), method) for row in manifest}


def visible_task(prefix: str, suffix: str) -> Any:
    """Model input contains no evaluator test/reference state."""
    return make_visible_task(prefix, suffix, method="m3")


def _device(model: Any) -> Any:
    device = getattr(model, "device", None)
    return device if device is not None else next(model.parameters()).device


def new_particle(
    *,
    canvas_tokens: int,
    task: Any,
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
        "lifetime_steps": remaining_lifetime_steps(int(born_round)),
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
    target_masks = linear_target_masks(int(particle["canvas_tokens"]), int(particle["lifetime_steps"]), int(particle["age"]))
    masked_positions = torch.nonzero(middle_mask[0], as_tuple=False).flatten().tolist()
    confidences = [float(value) for value in middle_probs[0].detach().cpu().tolist()]
    retain = confidence_retained_positions(confidences, masked_positions, target_masks)
    x_t[current_mask] = predictions[current_mask]
    if retain:
        x_t[0, [middle_start + index for index in retain]] = mask_token_id
    particle["last_confidences"] = confidences
    particle["age"] = int(particle["age"]) + 1


def birth_death_reallocate(
    *,
    particles: list[dict[str, Any]],
    prefix: str,
    suffix: str,
    tokenizer: Any,
    model: Any,
    task: Any,
    round_index: int,
    score_fn: Callable[[Mapping[str, Any], str, str, Any], tuple[int, int, int, float, int]] = visible_particle_score,
    particle_factory: Callable[..., dict[str, Any]] = new_particle,
) -> dict[str, Any] | None:
    """Replace the visibly weakest particle with a visible copy of the best."""
    ranked = sorted(
        enumerate(particles),
        key=lambda item: score_fn(item[1], prefix, suffix, tokenizer),
    )
    dead_index, dead = ranked[0]
    _, parent = ranked[-1]
    parent_score = score_fn(parent, prefix, suffix, tokenizer)
    dead_score = score_fn(dead, prefix, suffix, tokenizer)
    if parent_score <= dead_score:
        return None
    parent_text = particle_text(parent, tokenizer)
    particles[dead_index] = particle_factory(
        canvas_tokens=int(parent["canvas_tokens"]),
        task=task,
        tokenizer=tokenizer,
        model=model,
        born_round=round_index + 1,
        candidate_text=parent_text,
    )
    return {
        "round": round_index,
        "dead_canvas": int(dead["canvas_tokens"]),
        "born_canvas": int(parent["canvas_tokens"]),
        "parent_score": list(parent_score),
        "dead_score": list(dead_score),
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
    token_forwards = 0
    for round_index in range(STEPS_PER_PARTICLE):
        token_forwards += particle_round_token_forwards(particles)
        for particle in particles:
            step_particle(particle=particle, model=model)
            forwards += 1
        if birth_death_enabled(method, round_index):
            event = birth_death_reallocate(
                particles=particles,
                prefix=prefix,
                suffix=suffix,
                tokenizer=tokenizer,
                model=model,
                task=task,
                round_index=round_index,
            )
            if event is not None:
                events.append(event)
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
        "actual_token_forward_budget": token_forwards,
    }


def result_row(method: str, item: Mapping[str, Any], source: Mapping[str, Any], decoded: Mapping[str, Any]) -> dict[str, Any]:
    prefix, suffix = str(source["prompt"]), str(source["suffix"])
    middle = str(decoded["middle_text"])
    code = prefix + middle + suffix
    evaluated = evaluate_completion_after_decode(
        prefix=prefix,
        suffix=suffix,
        middle=middle,
        source_row=source,
        method="m3",
    )
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
        "passed": bool(evaluated["passed"]),
        "canvas_tokens": int(decoded["selected_canvas_tokens"]),
        "seed": SEED,
        "total_steps": TOTAL_FORWARDS,
        "candidate_middle_sha256": sha256_text(middle),
        "candidate_full_code_sha256": sha256_text(code),
        "middle_text": middle,
        "code": code,
        "metrics": {
            "actual_forward_count": int(decoded["actual_forward_count"]),
            "standalone_actual_forward_count": int(decoded["actual_forward_count"]),
            "actual_token_forward_budget": int(decoded["actual_token_forward_budget"]),
            "token_budget": int(decoded["actual_token_forward_budget"]),
            "verification_sec": float(evaluated["verification_sec"]),
        },
        "selection": {
            "selected_visible_score": decoded["selected_visible_score"],
            "particle_canvases_final": decoded["particle_canvases_final"],
            "birth_death_events": decoded["birth_death_events"],
        },
        "verification": evaluated["verification"],
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


def summarize(
    compact_dir: Path,
    phase: str,
    manifest: Sequence[Mapping[str, Any]],
    uniform_rows: Sequence[Mapping[str, Any]],
    birth_rows: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    resume_noops: int,
    peak_memory_bytes: int,
    wall_sec: float,
) -> dict[str, Any]:
    left, right = audit(uniform_rows, expected_keys(manifest, METHODS[0])), audit(birth_rows, expected_keys(manifest, METHODS[1]))
    frozen = lock.get("test_status") == "sealed" and int(lock.get("test_evaluation_count", -1)) == 0
    gate = bool(left["passed"] and right["passed"] and frozen and resume_noops == 0)
    compact_dir.mkdir(parents=True, exist_ok=True)
    write_csv(compact_dir / f"{phase}_summary.csv", [{"method": METHODS[0], **left}, {"method": METHODS[1], **right}])
    result = {
        "phase": phase,
        "technical_gate_passed": gate,
        "selected_case_count": len(manifest),
        "methods": {METHODS[0]: left, METHODS[1]: right},
        "fixed_total_forward_budget": TOTAL_FORWARDS,
        "peak_cuda_memory_bytes": int(peak_memory_bytes),
        "wall_sec": float(wall_sec),
        "frozen_test_status": lock.get("test_status"),
        "test_evaluation_count": lock.get("test_evaluation_count"),
    }
    write_json(compact_dir / f"{phase}_summary.json", result)
    return result


def run(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset_jsonl).resolve()
    uniform_dir, birth_dir, compact_dir = Path(args.uniform_output_dir).resolve(), Path(args.birth_death_output_dir).resolve(), Path(args.compact_dir).resolve()
    require_isolated_output_dirs(uniform_dir, birth_dir)
    frozen_groups, lock = load_frozen_groups()
    source_rows = read_jsonl(dataset)
    if len(source_rows) != 164:
        raise RuntimeError("M3 expects 164 RandomSpanLight source rows")
    tokenizer, model = load_model_and_tokenizer(cfg_for(64).model)
    manifest = build_manifest(source_rows, tokenizer, frozen_groups)
    if len(manifest) != RANDOMSPANLIGHT_ALLOWED_CASES:
        raise RuntimeError("M3 requires 148 allowed task groups")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for M3")
    selected = choose_smoke_manifest(manifest, int(args.smoke_cases))
    uniform_raw, birth_raw = uniform_dir / "m3_uniform_raw.jsonl", birth_dir / "m3_birth_death_raw.jsonl"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    run_method(method=METHODS[0], manifest=selected, source_rows=source_rows, raw_path=uniform_raw, tokenizer=tokenizer, model=model)
    run_method(method=METHODS[1], manifest=selected, source_rows=source_rows, raw_path=birth_raw, tokenizer=tokenizer, model=model)
    noops = run_method(method=METHODS[0], manifest=selected, source_rows=source_rows, raw_path=uniform_raw, tokenizer=tokenizer, model=model)
    noops += run_method(method=METHODS[1], manifest=selected, source_rows=source_rows, raw_path=birth_raw, tokenizer=tokenizer, model=model)
    smoke = summarize(
        compact_dir,
        "smoke",
        selected,
        read_jsonl(uniform_raw),
        read_jsonl(birth_raw),
        lock,
        noops,
        int(torch.cuda.max_memory_allocated()),
        time.perf_counter() - started,
    )
    if not smoke["technical_gate_passed"]:
        write_json(compact_dir / "run_manifest.json", {"status": "smoke_failed", "smoke": smoke, "test_evaluation_count": 0})
        return 2
    full = None
    if args.auto_full:
        require_randomspanlight_full(len(manifest))
        run_method(method=METHODS[0], manifest=manifest, source_rows=source_rows, raw_path=uniform_raw, tokenizer=tokenizer, model=model)
        run_method(method=METHODS[1], manifest=manifest, source_rows=source_rows, raw_path=birth_raw, tokenizer=tokenizer, model=model)
        noops = run_method(method=METHODS[0], manifest=manifest, source_rows=source_rows, raw_path=uniform_raw, tokenizer=tokenizer, model=model)
        noops += run_method(method=METHODS[1], manifest=manifest, source_rows=source_rows, raw_path=birth_raw, tokenizer=tokenizer, model=model)
        full = summarize(
            compact_dir,
            "full",
            manifest,
            read_jsonl(uniform_raw),
            read_jsonl(birth_raw),
            lock,
            noops,
            int(torch.cuda.max_memory_allocated()),
            time.perf_counter() - started,
        )
    write_json(compact_dir / "run_manifest.json", {"status": "completed" if full is None or full["technical_gate_passed"] else "full_failed", "model": MODEL_PATH, "source_config": SOURCE_CONFIG, "smoke": smoke, "full": full, "methods": list(METHODS), "total_forwards": TOTAL_FORWARDS, "compute_claim": "equal_forward_only_not_equal_token", "population_schedule": population_schedule(), "automatic_full_population": "randomspanlight_full", "frozen_test_status": "sealed", "test_evaluation_count": 0})
    return 0 if full is None or full["technical_gate_passed"] else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="M3 Birth-Death Canvas Diffusion V0")
    root.add_argument("--dataset-jsonl", required=True)
    root.add_argument("--uniform-output-dir", required=True)
    root.add_argument("--birth-death-output-dir", required=True)
    root.add_argument("--compact-dir", required=True)
    root.add_argument("--smoke-cases", type=int, default=SMOKE_CASES)
    root.add_argument("--auto-full", action="store_true")
    return root


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
