#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXTERNAL_ROOT = Path("/home/shx/projects/dllm_infilling")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dreamon_slot_generator import (  # noqa: E402
    Method,
    SlotGeneratorConfig,
    TokenizerSpec,
    run_slot_generation,
    scan_newline_token_ids,
)


DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B"
    / "snapshots/8ccc74750e43177327f29dab9e91882ba759e194"
)
DEFAULT_PROTOCOL = REPO_ROOT / "repro_results/dreamon_progressive_v2_protocol/protocol.json"
DEFAULT_POPULATION = (
    REPO_ROOT
    / "repro_results/dreamon_progressive_v2_protocol/generation_population.jsonl"
)
DEFAULT_DATASET = (
    EXTERNAL_ROOT
    / "human-eval-infilling/data/HumanEval-MultiLineInfilling.jsonl.gz"
)
HISTORICAL_DIR = EXTERNAL_ROOT / "repro_results/dreamon_progressive_three_line_all642"
HUMAN_EVAL_ROOT = EXTERNAL_ROOT / "human-eval-infilling"

STAGE_SIZES = {"smoke": 5, "pilot": 30, "full": 642}
ALLOWED_GENERATION_FIELDS = {"task_id", "base_problem_id", "prompt", "suffix"}
FORBIDDEN_GENERATION_FIELDS = {
    "canonical_solution",
    "reference_middle",
    "reference_lines",
    "reference_middle_tokens",
    "reference_line_tokens",
    "test_result",
    "passed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def durable_append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def build_method_config(
    method: Method, stage: str, runner_commit: str
) -> dict[str, Any]:
    if stage not in STAGE_SIZES:
        raise ValueError(f"Unknown stage: {stage}")
    protocol_sha = sha256(DEFAULT_PROTOCOL) if DEFAULT_PROTOCOL.exists() else "missing"
    population_sha = sha256(DEFAULT_POPULATION) if DEFAULT_POPULATION.exists() else "missing"
    return {
        "protocol_name": "dreamon_progressive_v2_slots",
        "protocol_version": 1,
        "protocol_sha256": protocol_sha,
        "method": method.value,
        "stage": stage,
        "stage_size": STAGE_SIZES[stage],
        "runner_commit": runner_commit,
        "population_sha256": population_sha,
        "model_revision": "8ccc74750e43177327f29dab9e91882ba759e194",
        "dtype": "bfloat16",
        "batch_size": 1,
        "initial_masks_per_region": 4,
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": None,
        "algorithm": "entropy",
        "algorithm_temperature": 0.0,
        "number_transfer_tokens": 1,
        "max_context_tokens": 2048,
        "max_prompt_tokens": 2048,
        "max_total_forwards": 256,
        "max_global_middle_tokens": 64,
        "max_hard_slot_tokens": 32,
        "seed": 42,
    }


def load_generation_population(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    task_ids: list[str] = []
    for row in rows:
        forbidden = set(row) & FORBIDDEN_GENERATION_FIELDS
        if forbidden:
            raise RuntimeError(f"forbidden generation field(s): {sorted(forbidden)}")
        if set(row) != ALLOWED_GENERATION_FIELDS:
            raise RuntimeError(f"unexpected generation schema: {sorted(row)}")
        task_ids.append(row["task_id"])
    if len(task_ids) != 642 or len(set(task_ids)) != 642:
        raise RuntimeError("Generation population must contain 642 unique tasks")
    return rows


def prediction_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return row["task_id"], row["method"], row["config_hash"]


def classify_protocol_outcome(row: Mapping[str, Any]) -> str:
    status = row["status"]
    flags = list(row["protocol_flags"])
    unresolved = int(row["unresolved_mask_count"])
    if status == "completed" and not flags and unresolved == 0:
        return "completed"
    if (
        status == "protocol_error"
        and flags == ["forward_cap_with_unresolved_masks"]
        and unresolved > 0
        and row["completion"] == ""
    ):
        return "allowed_terminal_protocol_failure"
    return "protocol_violation"


def validate_resume_rows(
    rows: Sequence[Mapping[str, Any]],
    method: Method,
    config_hash: str,
    selected_task_ids: Sequence[str],
) -> set[str]:
    seen_keys: set[tuple[str, str, str]] = set()
    selected = set(selected_task_ids)
    completed: set[str] = set()
    for row in rows:
        key = prediction_key(row)
        if key in seen_keys:
            raise RuntimeError(f"duplicate prediction key: {key}")
        seen_keys.add(key)
        if row["method"] != method.value or row["config_hash"] != config_hash:
            raise RuntimeError("Existing row method/config mismatch")
        if row["task_id"] not in selected:
            raise RuntimeError("Existing row task is outside selected stage")
        completed.add(row["task_id"])
    return completed


def base_problem_id(task_id: str) -> str:
    match = re.search(r"HumanEval/(\d+)", task_id)
    return match.group(1) if match else task_id


def environment_payload(args: argparse.Namespace, config_hash: str) -> dict[str, Any]:
    import transformers

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    return {
        "recorded_at_utc": utc_now(),
        "runner_commit": git_head(),
        "config_hash": config_hash,
        "python": sys.version,
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "model_path": str(args.model_path.resolve()),
        "population_path": str(args.population.resolve()),
        "protocol_path": str(args.protocol.resolve()),
        "device": args.device,
    }


def tokenizer_spec_from_hf(tokenizer, protocol: Mapping[str, Any]) -> TokenizerSpec:
    newline_ids, metadata = scan_newline_token_ids(tokenizer)
    expected = protocol["tokenizer"]
    if metadata["newline_token_count"] != expected["newline_token_count"]:
        raise RuntimeError("Tokenizer newline-token count differs from frozen protocol")
    if metadata["newline_token_ids_sha256"] != expected["newline_token_ids_sha256"]:
        raise RuntimeError("Tokenizer newline-token hash differs from frozen protocol")
    literal_newline_ids = tuple(tokenizer.encode("\n", add_special_tokens=False))
    if list(literal_newline_ids) != expected["literal_newline_token_ids"]:
        raise RuntimeError("Literal newline tokenization differs from frozen protocol")
    return TokenizerSpec(
        bos_id=int(tokenizer.bos_token_id),
        eos_id=int(tokenizer.eos_token_id),
        pad_id=int(tokenizer.pad_token_id),
        mask_id=int(tokenizer.mask_token_id),
        expand_id=int(expected["expand_token_id"]),
        newline_token_ids=frozenset(newline_ids),
        literal_newline_ids=literal_newline_ids,
    )


def encode_problem(row: Mapping[str, str], tokenizer, spec: TokenizerSpec):
    prefix_ids = [spec.bos_id] + tokenizer.encode(
        row["prompt"], add_special_tokens=False
    )
    suffix_ids = tokenizer.encode(row["suffix"], add_special_tokens=False) + [spec.eos_id]
    return prefix_ids, suffix_ids


def error_result(method: Method, error: BaseException) -> dict[str, Any]:
    region_names = (
        ["HARD_SLOT_0", "HARD_SLOT_1", "HARD_SLOT_2"]
        if method == Method.V2_HARD
        else ["HARD_SLOT_0", "HARD_SLOT_1", "OPEN_TAIL"]
    )
    zero = {name: 0 for name in region_names}
    return {
        "completion": "",
        "completion_token_ids": [],
        "regions": {name: {"token_ids": [], "text": ""} for name in region_names},
        "region_lengths": {"initial": zero, "final": zero},
        "region_forward_counts": zero,
        "selected_update_counts": zero,
        "total_forwards": 0,
        "forward_sequence_lengths": [],
        "token_forwards": 0,
        "normal_update_counts": zero,
        "expand_counts": zero,
        "delete_counts": zero,
        "mask_noop_counts": zero,
        "blank_region_flags": {name: True for name in region_names},
        "tail_newline_count": 0,
        "completion_physical_line_count": 0,
        "hard_forbidden_newline_attempts": 0,
        "slot_expand_cap_hits": 0,
        "global_expand_cap_hits": 0,
        "unresolved_mask_count": 0,
        "protocol_flags": ["runtime_exception"],
        "context_token_count": 0,
        "final_real_sequence_length": 0,
        "wall_time_seconds": 0.0,
        "peak_cuda_memory_bytes": 0,
        "status": "runtime_error",
        "first_selected_region": None,
        "joint_updates_before_slot0_complete": 0,
        "active_region_history": [],
        "step_trace": None,
        "runtime_error": repr(error),
        "runtime_traceback": traceback.format_exc(),
    }


def update_progress(
    path: Path,
    method: Method,
    stage: str,
    rows: Sequence[Mapping[str, Any]],
    total: int,
    current_task_id: str | None,
    status: str,
) -> None:
    completed = len(rows)
    wall = [float(row["wall_time_seconds"]) for row in rows]
    forwards = [int(row["total_forwards"]) for row in rows]
    token_forwards = [int(row["token_forwards"]) for row in rows]
    average_wall = statistics.mean(wall) if wall else None
    eta = None if average_wall is None else average_wall * max(0, total - completed)
    atomic_write_json(
        path,
        {
            "updated_at_utc": utc_now(),
            "method": method.value,
            "stage": stage,
            "status": status,
            "completed": completed,
            "total": total,
            "current_task_id": current_task_id,
            "average_wall_time_seconds": average_wall,
            "average_forwards": statistics.mean(forwards) if forwards else None,
            "total_token_forwards": sum(token_forwards),
            "allowed_terminal_protocol_failure_count": sum(
                classify_protocol_outcome(row) == "allowed_terminal_protocol_failure"
                for row in rows
            ),
            "protocol_violation_count": sum(
                classify_protocol_outcome(row) == "protocol_violation" for row in rows
            ),
            "eta_seconds": eta,
        },
    )


def run_generation(args: argparse.Namespace) -> None:
    from transformers import AutoModel, AutoTokenizer

    protocol = read_json(args.protocol)
    population = load_generation_population(args.population)
    selected = population[: STAGE_SIZES[args.stage]]
    selected_task_ids = [row["task_id"] for row in selected]
    method = Method(args.method)
    runner_commit = git_head()
    method_config = build_method_config(method, args.stage, runner_commit)
    config_hash = stable_json_hash(method_config)
    config_record = {**method_config, "config_hash": config_hash}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists() and read_json(config_path) != config_record:
        raise RuntimeError("Existing output config differs from requested config")
    atomic_write_json(config_path, config_record)
    atomic_write_json(
        args.output_dir / "environment.json", environment_payload(args, config_hash)
    )
    (args.output_dir / "manifest.sha256").write_text(
        protocol["population"]["manifest_sha256"] + "\n", encoding="utf-8"
    )
    violations_path = args.output_dir / "protocol_violations.jsonl"
    violations_path.touch(exist_ok=True)

    predictions_path = args.output_dir / "predictions.jsonl"
    existing = read_jsonl(predictions_path)
    completed_ids = validate_resume_rows(existing, method, config_hash, selected_task_ids)
    remaining = [row for row in selected if row["task_id"] not in completed_ids]
    update_progress(
        args.output_dir / "progress.json",
        method,
        args.stage,
        existing,
        len(selected),
        None,
        "running" if remaining else "generation_complete",
    )
    print(
        f"method={method.value} stage={args.stage} selected={len(selected)} "
        f"completed={len(existing)} remaining={len(remaining)} config_hash={config_hash}",
        flush=True,
    )
    if not remaining:
        return

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    tokenizer_spec = tokenizer_spec_from_hf(hf_tokenizer, protocol)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    generator_config = SlotGeneratorConfig()
    positions = {row["task_id"]: index for index, row in enumerate(selected, start=1)}
    generated_rows = list(existing)

    for problem in remaining:
        task_id = problem["task_id"]
        print(f"[{positions[task_id]}/{len(selected)}] {task_id} start", flush=True)
        prefix_ids, suffix_ids = encode_problem(problem, hf_tokenizer, tokenizer_spec)
        try:
            result = run_slot_generation(
                model=model,
                tokenizer=hf_tokenizer,
                tokenizer_spec=tokenizer_spec,
                method=method,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                config=generator_config,
                save_trace=args.stage == "smoke",
            )
        except Exception as error:  # explicit row-level runtime failure, never silent
            result = error_result(method, error)
        row = {
            "task_id": task_id,
            "base_problem_id": problem["base_problem_id"],
            "method": method.value,
            "config_hash": config_hash,
            "runner_commit": runner_commit,
            "model_revision": method_config["model_revision"],
            **result,
        }
        durable_append_jsonl(predictions_path, row)
        generated_rows.append(row)
        if classify_protocol_outcome(row) != "completed":
            durable_append_jsonl(violations_path, row)
        update_progress(
            args.output_dir / "progress.json",
            method,
            args.stage,
            generated_rows,
            len(selected),
            task_id,
            "running",
        )
        print(
            f"[{positions[task_id]}/{len(selected)}] {task_id} done "
            f"status={row['status']} forwards={row['total_forwards']} "
            f"token_forwards={row['token_forwards']} wall={row['wall_time_seconds']:.3f}s",
            flush=True,
        )
        if positions[task_id] % 25 == 0 or positions[task_id] == len(selected):
            average_wall = statistics.mean(
                float(item["wall_time_seconds"]) for item in generated_rows
            )
            average_forwards = statistics.mean(
                int(item["total_forwards"]) for item in generated_rows
            )
            violations = sum(
                classify_protocol_outcome(item) == "protocol_violation"
                for item in generated_rows
            )
            terminal_failures = sum(
                classify_protocol_outcome(item)
                == "allowed_terminal_protocol_failure"
                for item in generated_rows
            )
            eta = average_wall * (len(selected) - len(generated_rows))
            print(
                f"progress completed={len(generated_rows)}/{len(selected)} "
                f"avg_wall={average_wall:.3f}s avg_forwards={average_forwards:.2f} "
                f"token_forwards={sum(int(item['token_forwards']) for item in generated_rows)} "
                f"terminal_failures={terminal_failures} violations={violations} "
                f"eta_seconds={eta:.1f}",
                flush=True,
            )

    update_progress(
        args.output_dir / "progress.json",
        method,
        args.stage,
        generated_rows,
        len(selected),
        None,
        "generation_complete",
    )


def score_completion(problem: Mapping[str, Any], prediction: Mapping[str, Any], timeout: float):
    if prediction["status"] != "completed":
        return {
            "passed": False,
            "result": prediction["status"],
            "compile_passed": False,
            "exact_match": False,
        }
    if str(HUMAN_EVAL_ROOT) not in sys.path:
        sys.path.insert(0, str(HUMAN_EVAL_ROOT))
    from human_eval_infilling.execution import check_correctness

    completion = prediction["completion"]
    result = check_correctness(dict(problem), completion, timeout=timeout, completion_id=0)
    full_code = problem["prompt"] + completion + problem["suffix"]
    try:
        ast.parse(full_code)
        compile(full_code, "<string>", "exec")
        compile_passed = True
    except Exception:
        compile_passed = False
    return {
        "passed": bool(result.get("passed", False)),
        "result": result.get("result"),
        "compile_passed": compile_passed,
        "exact_match": completion == problem["canonical_solution"],
    }


def load_scoring_problems(path: Path, task_ids: set[str]) -> dict[str, dict[str, Any]]:
    problems: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["task_id"] in task_ids:
                problems[row["task_id"]] = row
    if set(problems) != task_ids:
        raise RuntimeError("Scoring dataset does not cover every selected task")
    return problems


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def method_summary(scored_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in scored_rows]
    by_base: dict[str, list[float]] = defaultdict(list)
    for row in scored_rows:
        by_base[row["base_problem_id"]].append(float(row["score"]["passed"]))
    forward_values = [int(row["total_forwards"]) for row in scored_rows]
    token_forward_values = [int(row["token_forwards"]) for row in scored_rows]
    wall_values = [float(row["wall_time_seconds"]) for row in scored_rows]
    peak_values = [int(row["peak_cuda_memory_bytes"]) for row in scored_rows]
    return {
        "num_samples": len(scored_rows),
        "num_base_problems": len(by_base),
        "passed": sum(bool(score["passed"]) for score in scores),
        "pass_rate": mean(float(score["passed"]) for score in scores),
        "compile_passed": sum(bool(score["compile_passed"]) for score in scores),
        "compile_pass_rate": mean(float(score["compile_passed"]) for score in scores),
        "exact_matches": sum(bool(score["exact_match"]) for score in scores),
        "exact_match_rate": mean(float(score["exact_match"]) for score in scores),
        "task_macro_pass_rate": mean(mean(values) for values in by_base.values()),
        "completion_tokens": {
            "mean": mean(len(row["completion_token_ids"]) for row in scored_rows),
            "median": statistics.median(
                len(row["completion_token_ids"]) for row in scored_rows
            ),
        },
        "completion_physical_lines": {
            "mean": mean(row["completion_physical_line_count"] for row in scored_rows),
            "distribution": dict(
                sorted(Counter(row["completion_physical_line_count"] for row in scored_rows).items())
            ),
        },
        "blank_region_count": sum(
            sum(bool(value) for value in row["blank_region_flags"].values())
            for row in scored_rows
        ),
        "expand_count": sum(
            sum(row["expand_counts"].values()) for row in scored_rows
        ),
        "delete_count": sum(
            sum(row["delete_counts"].values()) for row in scored_rows
        ),
        "slot_expand_cap_hits": sum(row["slot_expand_cap_hits"] for row in scored_rows),
        "global_expand_cap_hits": sum(row["global_expand_cap_hits"] for row in scored_rows),
        "unresolved_mask_count": sum(row["unresolved_mask_count"] for row in scored_rows),
        "protocol_error_count": sum(row["status"] == "protocol_error" for row in scored_rows),
        "runtime_error_count": sum(row["status"] == "runtime_error" for row in scored_rows),
        "forwards": {
            "mean": mean(forward_values),
            "median": statistics.median(forward_values),
            "max": max(forward_values),
            "total": sum(forward_values),
        },
        "token_forwards": {
            "mean": mean(token_forward_values),
            "median": statistics.median(token_forward_values),
            "max": max(token_forward_values),
            "total": sum(token_forward_values),
        },
        "wall_time_seconds": {
            "mean": mean(wall_values),
            "median": statistics.median(wall_values),
            "max": max(wall_values),
            "total": sum(wall_values),
        },
        "peak_cuda_memory_bytes": {
            "mean": mean(peak_values),
            "max": max(peak_values),
        },
    }


def run_scoring(args: argparse.Namespace) -> None:
    method = Method(args.method)
    predictions = read_jsonl(args.output_dir / "predictions.jsonl")
    expected = STAGE_SIZES[args.stage]
    if len(predictions) != expected:
        raise RuntimeError(f"Expected {expected} predictions, found {len(predictions)}")
    config_record = read_json(args.output_dir / "config.json")
    validate_resume_rows(
        predictions,
        method,
        config_record["config_hash"],
        [row["task_id"] for row in predictions],
    )
    problems = load_scoring_problems(args.dataset, {row["task_id"] for row in predictions})
    scored_by_task: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            row["task_id"]: executor.submit(
                score_completion, problems[row["task_id"]], row, args.timeout
            )
            for row in predictions
        }
        for index, row in enumerate(predictions, start=1):
            scored_by_task[row["task_id"]] = {
                **row,
                "score": futures[row["task_id"]].result(),
            }
            if index % 100 == 0 or index == len(predictions):
                print(f"scored {index}/{len(predictions)}", flush=True)
    scored_rows = [scored_by_task[row["task_id"]] for row in predictions]
    atomic_write_jsonl(args.output_dir / "scored.jsonl", scored_rows)
    summary = method_summary(scored_rows)
    atomic_write_json(args.output_dir / "summary.json", summary)
    diagnostics = {
        "method": method.value,
        "stage": args.stage,
        "status_distribution": dict(sorted(Counter(row["status"] for row in scored_rows).items())),
        "protocol_flag_distribution": dict(
            sorted(Counter(flag for row in scored_rows for flag in row["protocol_flags"]).items())
        ),
        "first_selected_region_distribution": dict(
            sorted(Counter(row["first_selected_region"] for row in scored_rows).items(), key=lambda x: str(x[0]))
        ),
        "joint_updates_before_slot0_complete": sum(
            row["joint_updates_before_slot0_complete"] for row in scored_rows
        ),
        "tail_multiline_rows": sum(row["tail_newline_count"] >= 2 for row in scored_rows),
        "hard_forbidden_newline_attempts": sum(
            row["hard_forbidden_newline_attempts"] for row in scored_rows
        ),
        "region_final_length_distributions": {
            region: dict(
                sorted(
                    Counter(
                        row["region_lengths"]["final"][region] for row in scored_rows
                    ).items()
                )
            )
            for region in scored_rows[0]["region_lengths"]["final"]
        },
    }
    atomic_write_json(args.output_dir / "diagnostics.json", diagnostics)
    write_completeness_audit(args, predictions, scored_rows, config_record)
    update_progress(
        args.output_dir / "progress.json",
        method,
        args.stage,
        predictions,
        expected,
        None,
        "scored_complete",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def write_completeness_audit(
    args: argparse.Namespace,
    predictions: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
    config_record: Mapping[str, Any],
) -> None:
    prediction_ids = [row["task_id"] for row in predictions]
    scored_ids = [row["task_id"] for row in scored_rows]
    selected_ids = [
        row["task_id"]
        for row in load_generation_population(args.population)[: STAGE_SIZES[args.stage]]
    ]
    audit = {
        "method": args.method,
        "stage": args.stage,
        "expected_rows": STAGE_SIZES[args.stage],
        "prediction_rows": len(predictions),
        "prediction_unique_task_ids": len(set(prediction_ids)),
        "score_rows": len(scored_rows),
        "score_unique_task_ids": len(set(scored_ids)),
        "task_order_matches_population": prediction_ids == selected_ids,
        "prediction_score_task_order_match": prediction_ids == scored_ids,
        "config_hash": config_record["config_hash"],
        "all_prediction_config_hashes_match": all(
            row["config_hash"] == config_record["config_hash"] for row in predictions
        ),
        "all_prediction_methods_match": all(
            row["method"] == args.method for row in predictions
        ),
        "manifest_sha256": (args.output_dir / "manifest.sha256").read_text().strip(),
        "allowed_terminal_protocol_failure_rows": sum(
            classify_protocol_outcome(row) == "allowed_terminal_protocol_failure"
            for row in predictions
        ),
        "disallowed_protocol_violation_rows": sum(
            classify_protocol_outcome(row) == "protocol_violation"
            for row in predictions
        ),
        "silent_unresolved_acceptance_rows": sum(
            row["status"] == "completed" and int(row["unresolved_mask_count"]) > 0
            for row in predictions
        ),
        "posthoc_truncation_fields_present": any(
            any("discard" in key or "first_line" in key for key in row)
            for row in predictions
        ),
    }
    audit["complete"] = all(
        [
            audit["prediction_rows"] == audit["expected_rows"],
            audit["prediction_unique_task_ids"] == audit["expected_rows"],
            audit["score_rows"] == audit["expected_rows"],
            audit["score_unique_task_ids"] == audit["expected_rows"],
            audit["task_order_matches_population"],
            audit["prediction_score_task_order_match"],
            audit["all_prediction_config_hashes_match"],
            audit["all_prediction_methods_match"],
            audit["disallowed_protocol_violation_rows"] == 0,
            audit["silent_unresolved_acceptance_rows"] == 0,
            not audit["posthoc_truncation_fields_present"],
        ]
    )
    atomic_write_json(args.output_dir / "completeness_audit.json", audit)


def run_audit(args: argparse.Namespace) -> None:
    audit = read_json(args.output_dir / "completeness_audit.json")
    if not audit.get("complete"):
        raise RuntimeError(f"Completeness/protocol audit failed: {json.dumps(audit)}")
    violations = read_jsonl(args.output_dir / "protocol_violations.jsonl")
    expected_flagged = (
        audit["allowed_terminal_protocol_failure_rows"]
        + audit["disallowed_protocol_violation_rows"]
    )
    if len(violations) != expected_flagged:
        raise RuntimeError(
            f"Protocol failure/violation file has {len(violations)} rows, "
            f"expected {expected_flagged}"
        )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "score", "audit"], required=True)
    parser.add_argument("--method", choices=[method.value for method in Method], required=True)
    parser.add_argument("--stage", choices=list(STAGE_SIZES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "generate":
        run_generation(args)
    elif args.mode == "score":
        run_scoring(args)
    else:
        run_audit(args)


if __name__ == "__main__":
    main()
