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
    scan_newline_token_metadata,
)


DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B"
    / "snapshots/8ccc74750e43177327f29dab9e91882ba759e194"
)
DEFAULT_PROTOCOL = REPO_ROOT / "repro_results/dreamon_progressive_v2_protocol/protocol.json"
V2_HARD_V2_PROTOCOL = (
    REPO_ROOT
    / "repro_results/dreamon_progressive_v2_hard_v2_protocol/protocol.json"
)
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
TARGETED_STAGE_SIZES = {"cycle5": 5, "affected6": 6}
ALL_STAGE_SIZES = {**STAGE_SIZES, **TARGETED_STAGE_SIZES}
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


def stage_size(stage: str) -> int:
    try:
        return ALL_STAGE_SIZES[stage]
    except KeyError as error:
        raise ValueError(f"Unknown stage: {stage}") from error


def is_v3_method(method: Method) -> bool:
    return method in {
        Method.V3_HARD_BUDGETED,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
    }


def build_method_config(
    method: Method,
    stage: str,
    runner_commit: str,
    protocol_path: Path = DEFAULT_PROTOCOL,
    population_path: Path = DEFAULT_POPULATION,
) -> dict[str, Any]:
    selected_stage_size = stage_size(stage)
    protocol_sha = sha256(protocol_path) if protocol_path.exists() else "missing"
    population_sha = sha256(population_path) if population_path.exists() else "missing"
    is_boundary_v2 = method == Method.V2_HARD_V2_BOUNDARY
    is_v3_budgeted = is_v3_method(method)
    nonempty_guard = method == Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE
    if is_v3_budgeted:
        protocol_name = (
            "dreamon_progressive_v3_hard_budgeted_nonempty_oracle"
            if nonempty_guard
            else "dreamon_progressive_v3_hard_budgeted"
        )
    elif is_boundary_v2:
        protocol_name = "dreamon_progressive_v2_hard_boundary"
    else:
        protocol_name = "dreamon_progressive_v2_slots"
    return {
        "protocol_name": protocol_name,
        "protocol_version": 3 if is_v3_budgeted else (2 if is_boundary_v2 else 1),
        "protocol_sha256": protocol_sha,
        "method": method.value,
        "stage": stage,
        "stage_size": selected_stage_size,
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
        "newline_boundary_action": is_boundary_v2 or is_v3_budgeted,
        "region_local_eos_broadcast": is_boundary_v2 or is_v3_budgeted,
        "exact_transition_cycle_detection": is_boundary_v2 or is_v3_budgeted,
        "initial_expand_budget": 64 if is_v3_budgeted else None,
        "expand_budget_scope": "per_sample_global" if is_v3_budgeted else None,
        "expand_budget_reset_per_slot": False if is_v3_budgeted else None,
        "expand_budget_refund_on_delete": False if is_v3_budgeted else None,
        "expand_budget_logit_mask_at_zero": True if is_v3_budgeted else None,
        "nonempty_guard": nonempty_guard,
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


def load_targeted_generation_population(
    path: Path, expected_rows: int
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    task_ids: list[str] = []
    for row in rows:
        forbidden = set(row) & FORBIDDEN_GENERATION_FIELDS
        if forbidden:
            raise RuntimeError(f"forbidden generation field(s): {sorted(forbidden)}")
        if set(row) != ALLOWED_GENERATION_FIELDS:
            raise RuntimeError(f"unexpected generation schema: {sorted(row)}")
        task_ids.append(row["task_id"])
    if len(task_ids) != expected_rows or len(set(task_ids)) != expected_rows:
        raise RuntimeError(
            f"Targeted generation population expected {expected_rows} unique tasks"
        )
    return rows


def load_stage_population(path: Path, stage: str) -> list[dict[str, Any]]:
    if stage in TARGETED_STAGE_SIZES:
        return load_targeted_generation_population(path, stage_size(stage))
    return load_generation_population(path)[: stage_size(stage)]


def prediction_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return row["task_id"], row["method"], row["config_hash"]


def classify_protocol_outcome(row: Mapping[str, Any]) -> str:
    status = row["status"]
    flags = list(row["protocol_flags"])
    unresolved = int(row["unresolved_mask_count"])
    if status == "completed" and not flags and unresolved == 0:
        return "completed"
    if row.get("method") in {
        Method.V2_HARD_V2_BOUNDARY.value,
        Method.V3_HARD_BUDGETED.value,
        Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE.value,
    }:
        if (
            status == "protocol_error"
            and len(flags) == 1
            and flags[0]
            in {
                "forward_cap_with_unresolved_masks",
                "exact_deterministic_cycle",
                "nonempty_guard_no_valid_action",
            }
            and row["completion"] == ""
        ):
            return "terminal_method_failure"
        return "protocol_violation"
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


def environment_payload(
    args: argparse.Namespace,
    config_hash: str,
    tokenizer_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
        "tokenizer_newline_mapping": dict(tokenizer_metadata or {}),
    }


def tokenizer_spec_from_hf(
    tokenizer, protocol: Mapping[str, Any]
) -> tuple[TokenizerSpec, dict[str, Any]]:
    newline_map, metadata = scan_newline_token_metadata(tokenizer)
    newline_ids = sorted(newline_map)
    expected = protocol["tokenizer"]
    if metadata["newline_token_count"] != expected["newline_token_count"]:
        raise RuntimeError("Tokenizer newline-token count differs from frozen protocol")
    if metadata["newline_token_ids_sha256"] != expected["newline_token_ids_sha256"]:
        raise RuntimeError("Tokenizer newline-token hash differs from frozen protocol")
    if "newline_token_mapping_count" in expected:
        if (
            metadata["newline_token_mapping_count"]
            != expected["newline_token_mapping_count"]
        ):
            raise RuntimeError("Tokenizer newline mapping count differs from protocol")
        if (
            metadata["newline_token_mapping_sha256"]
            != expected["newline_token_mapping_sha256"]
        ):
            raise RuntimeError("Tokenizer newline mapping hash differs from protocol")
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
        newline_token_map=newline_map,
        literal_newline_ids=literal_newline_ids,
    ), metadata


def encode_problem(row: Mapping[str, str], tokenizer, spec: TokenizerSpec):
    prefix_ids = [spec.bos_id] + tokenizer.encode(
        row["prompt"], add_special_tokens=False
    )
    suffix_ids = tokenizer.encode(row["suffix"], add_special_tokens=False) + [spec.eos_id]
    return prefix_ids, suffix_ids


def error_result(method: Method, error: BaseException) -> dict[str, Any]:
    region_names = (
        ["HARD_SLOT_0", "HARD_SLOT_1", "HARD_SLOT_2"]
        if method
        in (
            Method.V2_HARD,
            Method.V2_HARD_V2_BOUNDARY,
            Method.V3_HARD_BUDGETED,
            Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
        )
        else ["HARD_SLOT_0", "HARD_SLOT_1", "OPEN_TAIL"]
    )
    zero = {name: 0 for name in region_names}
    return {
        "completion": "",
        "completion_token_ids": [],
        "region_token_ids": {name: [] for name in region_names},
        "region_text": {name: "" for name in region_names},
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
        "newline_boundary_events": 0,
        "newline_boundary_event_details": [],
        "mixed_newline_token_events": 0,
        "pure_newline_token_events": 0,
        "multiple_newline_token_events": 0,
        "boundary_retokenized_left_token_count": 0,
        "discarded_internal_right_text": [],
        "discarded_masks_after_boundary": 0,
        "discarded_resolved_token_ids_after_boundary": [],
        "discarded_resolved_segments_after_boundary": [],
        "discarded_resolved_tokens_after_boundary": 0,
        "boundary_events_with_left_masks_remaining": 0,
        "region_local_eos_events": 0,
        "region_local_eos_event_details": [],
        "region_local_eos_deleted_masks": 0,
        "max_masks_deleted_by_one_eos": 0,
        "cross_region_delete_attempts": 0,
        "exact_cycle_events": [],
        "initial_expand_budget": (
            64
            if method
            in {
                Method.V3_HARD_BUDGETED,
                Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
            }
            else None
        ),
        "remaining_expand_budget": (
            64 if is_v3_method(method) else None
        ),
        "expand_budget_consumed": (
            0 if is_v3_method(method) else None
        ),
        "expand_budget_conservation_passed": False,
        "budget_exhausted": False,
        "budget_exhausted_forward_index": None,
        "expand_logit_blocked_by_budget_count": 0,
        "budget_draining_loop_count": 0,
        "budget_draining_loop_events": [],
        "nonempty_guard_rejection_count": 0,
        "nonempty_guard_rejections_by_slot": zero,
        "nonempty_guard_rejections_by_action": {},
        "nonempty_guard_rejection_events": [],
        "nonempty_guard_no_valid_action_count": 0,
        "final_nonempty_invariant_passed": False,
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
            "terminal_method_failure_count": sum(
                classify_protocol_outcome(row) == "terminal_method_failure"
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
    selected = load_stage_population(args.population, args.stage)
    selected_task_ids = [row["task_id"] for row in selected]
    method = Method(args.method)
    if method == Method.V2_HARD_V2_BOUNDARY:
        if protocol.get("protocol_version") != 2:
            raise RuntimeError("V2-Hard-v2 requires protocol version 2")
        if protocol.get("method") != method.value:
            raise RuntimeError("Protocol method does not match V2-Hard-v2")
    if is_v3_method(method):
        if protocol.get("protocol_version") != 3:
            raise RuntimeError("V3 requires protocol version 3")
        if protocol.get("method") != method.value:
            raise RuntimeError("Protocol method does not match V3 method")
        expected_nonempty = method == Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE
        if bool(protocol.get("decoder_revision", {}).get("nonempty_guard", False)) != expected_nonempty:
            raise RuntimeError("Protocol nonempty guard does not match V3 method")
    runner_commit = git_head()
    method_config = build_method_config(
        method,
        args.stage,
        runner_commit,
        protocol_path=args.protocol,
        population_path=args.population,
    )
    config_hash = stable_json_hash(method_config)
    config_record = {**method_config, "config_hash": config_hash}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists() and read_json(config_path) != config_record:
        raise RuntimeError("Existing output config differs from requested config")
    atomic_write_json(config_path, config_record)
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
    tokenizer_spec, tokenizer_metadata = tokenizer_spec_from_hf(hf_tokenizer, protocol)
    atomic_write_json(
        args.output_dir / "tokenizer_metadata.json", tokenizer_metadata
    )
    atomic_write_json(
        args.output_dir / "environment.json",
        environment_payload(args, config_hash, tokenizer_metadata),
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    generator_config = SlotGeneratorConfig(
        initial_expand_budget=64 if is_v3_method(method) else None,
        nonempty_guard=method == Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE,
    )
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
                save_trace=args.stage in {"smoke", "cycle5", "pilot", "affected6"},
                task_id=task_id,
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
        if method == Method.V2_HARD_V2_BOUNDARY or is_v3_method(method):
            invariant_flags = {
                "prefix_tokens_mutated",
                "suffix_tokens_mutated",
                "locked_separator_tokens_mutated",
                "future_region_mutated_before_activation",
                "cross_region_delete_attempt",
                "parallel_state_lengths_differ",
                "pad_region_inside_real_sequence",
                "non_pad_region_in_right_padding",
            }
            if row["status"] == "runtime_error" or invariant_flags.intersection(
                row["protocol_flags"]
            ):
                atomic_write_json(
                    args.output_dir / "early_stop.json",
                    {
                        "reason": "runtime_or_invariant_damage",
                        "task_id": task_id,
                        "completed_rows": len(generated_rows),
                        "protocol_flags": row["protocol_flags"],
                        "recorded_at_utc": utc_now(),
                    },
                )
                update_progress(
                    args.output_dir / "progress.json",
                    method,
                    args.stage,
                    generated_rows,
                    len(selected),
                    task_id,
                    "paused_early_stop",
                )
                raise RuntimeError("V2-Hard-v2 stopped on runtime/invariant damage")
            if args.stage == "full" and method == Method.V2_HARD_V2_BOUNDARY:
                noncompleted = sum(
                    item["status"] != "completed" for item in generated_rows
                )
                stop_reason = None
                if len(generated_rows) <= 50 and noncompleted >= 5:
                    stop_reason = "first_50_noncompleted_at_least_5"
                elif (
                    len(generated_rows) >= 100
                    and noncompleted / len(generated_rows) > 0.05
                ):
                    stop_reason = "post_100_noncompleted_rate_above_5_percent"
                if stop_reason is not None:
                    atomic_write_json(
                        args.output_dir / "early_stop.json",
                        {
                            "reason": stop_reason,
                            "task_id": task_id,
                            "completed_rows": len(generated_rows),
                            "noncompleted_rows": noncompleted,
                            "noncompleted_rate": noncompleted / len(generated_rows),
                            "recorded_at_utc": utc_now(),
                        },
                    )
                    update_progress(
                        args.output_dir / "progress.json",
                        method,
                        args.stage,
                        generated_rows,
                        len(selected),
                        task_id,
                        "paused_early_stop",
                    )
                    raise RuntimeError(f"V2-Hard-v2 full early stop: {stop_reason}")
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
            method_failures = sum(
                classify_protocol_outcome(item) == "terminal_method_failure"
                for item in generated_rows
            )
            eta = average_wall * (len(selected) - len(generated_rows))
            print(
                f"progress completed={len(generated_rows)}/{len(selected)} "
                f"avg_wall={average_wall:.3f}s avg_forwards={average_forwards:.2f} "
                f"token_forwards={sum(int(item['token_forwards']) for item in generated_rows)} "
                f"terminal_failures={terminal_failures} method_failures={method_failures} "
                f"violations={violations} "
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


def _normalize_diagnostic_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def _edit_similarity(left: str, right: str) -> float:
    left = _normalize_diagnostic_text(left)
    right = _normalize_diagnostic_text(right)
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(left), len(right), 1)


def _token_lcs_similarity(left: str, right: str, tokenizer) -> float:
    left_ids = tokenizer.encode(left, add_special_tokens=False)
    right_ids = tokenizer.encode(right, add_special_tokens=False)
    if not left_ids and not right_ids:
        return 1.0
    previous = [0] * (len(right_ids) + 1)
    for left_id in left_ids:
        current = [0]
        for right_index, right_id in enumerate(right_ids, start=1):
            if left_id == right_id:
                current.append(previous[right_index - 1] + 1)
            else:
                current.append(max(current[-1], previous[right_index]))
        previous = current
    return previous[-1] / max(len(left_ids), len(right_ids), 1)


def _similarity_metrics(
    candidate: str | None, target: str, tokenizer
) -> dict[str, Any]:
    if candidate is None:
        return {
            "available": False,
            "exact_match": False,
            "candidate_is_target_prefix": False,
            "candidate_is_target_substring": False,
            "character_edit_similarity": None,
            "token_lcs_similarity": None,
        }
    normalized_candidate = _normalize_diagnostic_text(candidate)
    normalized_target = _normalize_diagnostic_text(target)
    return {
        "available": True,
        "exact_match": normalized_candidate == normalized_target,
        "candidate_is_target_prefix": bool(normalized_candidate)
        and normalized_target.startswith(normalized_candidate),
        "candidate_is_target_substring": bool(normalized_candidate)
        and normalized_candidate in normalized_target,
        "character_edit_similarity": _edit_similarity(
            normalized_candidate, normalized_target
        ),
        "token_lcs_similarity": _token_lcs_similarity(
            normalized_candidate, normalized_target, tokenizer
        ),
    }


def _group_score_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "passed": sum(bool(row["score"]["passed"]) for row in rows),
        "pass_rate": mean(float(row["score"]["passed"]) for row in rows),
        "compile_passed": sum(
            bool(row["score"]["compile_passed"]) for row in rows
        ),
        "compile_pass_rate": mean(
            float(row["score"]["compile_passed"]) for row in rows
        ),
    }


def write_discard_reference_diagnostics(
    output_dir: Path,
    scored_rows: Sequence[Mapping[str, Any]],
    problems: Mapping[str, Mapping[str, Any]],
    tokenizer,
) -> dict[str, Any]:
    event_rows: list[dict[str, Any]] = []
    rows_with_resolved_discard: set[str] = set()
    total_discarded_tokens = 0
    total_normal_generated_tokens = sum(
        sum(int(value) for value in row["normal_update_counts"].values())
        + int(row.get("boundary_retokenized_left_token_count", 0))
        for row in scored_rows
    )
    for row in scored_rows:
        problem = problems[row["task_id"]]
        reference_lines = _normalize_diagnostic_text(
            problem["canonical_solution"]
        ).split("\n")
        for event_index, event in enumerate(
            row.get("newline_boundary_event_details", [])
        ):
            slot_name = event["slot"]
            if slot_name not in ("HARD_SLOT_0", "HARD_SLOT_1"):
                continue
            slot_index = int(slot_name.rsplit("_", 1)[1])
            next_slot = f"HARD_SLOT_{slot_index + 1}"
            reference_next = (
                reference_lines[slot_index + 1]
                if slot_index + 1 < len(reference_lines)
                else ""
            )
            actual_next = row["region_text"][next_slot]
            contiguous_segments = list(event.get("discarded_contiguous_segments", []))
            leading_segment = contiguous_segments[0] if contiguous_segments else ""
            complete_text = event.get("discarded_complete_text")
            resolved_segments = list(
                event.get("discarded_resolved_segments_after_boundary", [])
            )
            resolved_count = int(
                event.get("discarded_resolved_tokens_after_boundary", 0)
            )
            internal_token_count = len(
                tokenizer.encode(event.get("right_text", ""), add_special_tokens=False)
            )
            event_discarded_tokens = internal_token_count + resolved_count
            total_discarded_tokens += event_discarded_tokens
            if resolved_count:
                rows_with_resolved_discard.add(row["task_id"])
            event_rows.append(
                {
                    "task_id": row["task_id"],
                    "base_problem_id": row["base_problem_id"],
                    "event_index": event_index,
                    "slot": slot_name,
                    "boundary_proposal": {
                        "token_id": event["proposal_token_id"],
                        "decoded_text": event["decoded_text"],
                        "left_text": event["left_text"],
                        "right_text": event["right_text"],
                    },
                    "discarded_contiguous_segments": contiguous_segments,
                    "discarded_resolved_segments": resolved_segments,
                    "discarded_masks": event["discarded_masks_after_boundary"],
                    "discarded_token_count": event_discarded_tokens,
                    "reference_next_line": reference_next,
                    "final_next_slot_line": actual_next,
                    "complete_vs_reference": _similarity_metrics(
                        complete_text, reference_next, tokenizer
                    ),
                    "leading_segment_vs_reference": _similarity_metrics(
                        leading_segment, reference_next, tokenizer
                    ),
                    "complete_vs_final_next_slot": _similarity_metrics(
                        complete_text, actual_next, tokenizer
                    ),
                    "leading_segment_vs_final_next_slot": _similarity_metrics(
                        leading_segment, actual_next, tokenizer
                    ),
                    "oracle_mechanism_diagnostic": True,
                }
            )
    atomic_write_jsonl(output_dir / "discard_reference_events.jsonl", event_rows)
    resolved_cases = [
        event
        for event in event_rows
        if event["discarded_resolved_segments"]
    ][:10]
    atomic_write_jsonl(
        output_dir / "discard_reference_cases_top10.jsonl", resolved_cases
    )
    rows_with = [
        row for row in scored_rows if row["task_id"] in rows_with_resolved_discard
    ]
    rows_without = [
        row for row in scored_rows if row["task_id"] not in rows_with_resolved_discard
    ]
    complete_available = [
        event for event in event_rows if event["complete_vs_reference"]["available"]
    ]
    leading_nonempty = [
        event
        for event in event_rows
        if event["discarded_contiguous_segments"]
    ]
    discard_counts = [event["discarded_token_count"] for event in event_rows]
    summary = {
        "oracle_mechanism_diagnostic": True,
        "event_count": len(event_rows),
        "rows_with_boundary_event": len(
            {event["task_id"] for event in event_rows}
        ),
        "resolved_discard_events": sum(
            bool(event["discarded_resolved_segments"]) for event in event_rows
        ),
        "rows_with_resolved_discard": len(rows_with_resolved_discard),
        "row_fraction_with_resolved_discard": (
            len(rows_with_resolved_discard) / len(scored_rows)
        ),
        "discarded_tokens": {
            "total": total_discarded_tokens,
            "mean_per_event": mean(discard_counts),
            "median_per_event": (
                statistics.median(discard_counts) if discard_counts else None
            ),
            "fraction_of_all_normal_generated_tokens": (
                total_discarded_tokens / total_normal_generated_tokens
                if total_normal_generated_tokens
                else None
            ),
            "normal_generated_token_denominator": total_normal_generated_tokens,
        },
        "complete_text_reference": {
            "available_events": len(complete_available),
            "exact_match_rate": mean(
                float(event["complete_vs_reference"]["exact_match"])
                for event in complete_available
            ),
            "prefix_rate": mean(
                float(
                    event["complete_vs_reference"][
                        "candidate_is_target_prefix"
                    ]
                )
                for event in complete_available
            ),
            "substring_rate": mean(
                float(
                    event["complete_vs_reference"][
                        "candidate_is_target_substring"
                    ]
                )
                for event in complete_available
            ),
        },
        "leading_segment_reference": {
            "available_events": len(leading_nonempty),
            "prefix_rate": mean(
                float(
                    event["leading_segment_vs_reference"][
                        "candidate_is_target_prefix"
                    ]
                )
                for event in leading_nonempty
            ),
            "substring_rate": mean(
                float(
                    event["leading_segment_vs_reference"][
                        "candidate_is_target_substring"
                    ]
                )
                for event in leading_nonempty
            ),
        },
        "score_groups": {
            "with_resolved_discard": _group_score_summary(rows_with),
            "without_resolved_discard": _group_score_summary(rows_without),
        },
    }
    atomic_write_json(output_dir / "discard_reference_diagnostics.json", summary)
    return summary


def method_summary(scored_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in scored_rows]
    by_base: dict[str, list[float]] = defaultdict(list)
    for row in scored_rows:
        by_base[row["base_problem_id"]].append(float(row["score"]["passed"]))
    forward_values = [int(row["total_forwards"]) for row in scored_rows]
    token_forward_values = [int(row["token_forwards"]) for row in scored_rows]
    wall_values = [float(row["wall_time_seconds"]) for row in scored_rows]
    peak_values = [int(row["peak_cuda_memory_bytes"]) for row in scored_rows]
    completed_rows = [row for row in scored_rows if row["status"] == "completed"]
    completed_scores = [row["score"] for row in completed_rows]
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
        "completion_rate": len(completed_rows) / len(scored_rows),
        "completed_rows": len(completed_rows),
        "completed_only": {
            "passed": sum(bool(score["passed"]) for score in completed_scores),
            "pass_rate": mean(float(score["passed"]) for score in completed_scores),
            "compile_passed": sum(
                bool(score["compile_passed"]) for score in completed_scores
            ),
            "compile_pass_rate": mean(
                float(score["compile_passed"]) for score in completed_scores
            ),
            "exact_matches": sum(
                bool(score["exact_match"]) for score in completed_scores
            ),
            "exact_match_rate": mean(
                float(score["exact_match"]) for score in completed_scores
            ),
        },
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
        "blank_slot_rows": sum(
            any(bool(value) for value in row["blank_region_flags"].values())
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
        "exact_cycle_count": sum(
            "exact_deterministic_cycle" in row["protocol_flags"]
            for row in scored_rows
        ),
        "forward_cap_count": sum(
            "forward_cap_with_unresolved_masks" in row["protocol_flags"]
            for row in scored_rows
        ),
        "budget_exhausted_rows": sum(
            bool(row.get("budget_exhausted", False)) for row in scored_rows
        ),
        "budget_draining_loop_rows": sum(
            int(row.get("budget_draining_loop_count", 0)) > 0
            for row in scored_rows
        ),
        "budget_draining_loop_count": sum(
            int(row.get("budget_draining_loop_count", 0)) for row in scored_rows
        ),
        "expand_logit_blocked_by_budget_count": sum(
            int(row.get("expand_logit_blocked_by_budget_count", 0))
            for row in scored_rows
        ),
        "expand_budget_consumed": {
            "mean": mean(
                int(row["expand_budget_consumed"])
                for row in scored_rows
                if row.get("expand_budget_consumed") is not None
            ),
            "median": (
                statistics.median(
                    int(row["expand_budget_consumed"])
                    for row in scored_rows
                    if row.get("expand_budget_consumed") is not None
                )
                if any(
                    row.get("expand_budget_consumed") is not None
                    for row in scored_rows
                )
                else None
            ),
            "max": max(
                (
                    int(row["expand_budget_consumed"])
                    for row in scored_rows
                    if row.get("expand_budget_consumed") is not None
                ),
                default=None,
            ),
            "distribution": dict(
                sorted(
                    Counter(
                        int(row["expand_budget_consumed"])
                        for row in scored_rows
                        if row.get("expand_budget_consumed") is not None
                    ).items()
                )
            ),
        },
        "nonempty_guard_rejection_count": sum(
            int(row.get("nonempty_guard_rejection_count", 0))
            for row in scored_rows
        ),
        "nonempty_guard_triggered_rows": sum(
            int(row.get("nonempty_guard_rejection_count", 0)) > 0
            for row in scored_rows
        ),
        "nonempty_guard_no_valid_action_rows": sum(
            "nonempty_guard_no_valid_action" in row["protocol_flags"]
            for row in scored_rows
        ),
        "newline_boundary_events": sum(
            int(row.get("newline_boundary_events", 0)) for row in scored_rows
        ),
        "mixed_newline_token_events": sum(
            int(row.get("mixed_newline_token_events", 0)) for row in scored_rows
        ),
        "pure_newline_token_events": sum(
            int(row.get("pure_newline_token_events", 0)) for row in scored_rows
        ),
        "multiple_newline_token_events": sum(
            int(row.get("multiple_newline_token_events", 0)) for row in scored_rows
        ),
        "discarded_masks_after_boundary": sum(
            int(row.get("discarded_masks_after_boundary", 0)) for row in scored_rows
        ),
        "discarded_resolved_tokens_after_boundary": sum(
            int(row.get("discarded_resolved_tokens_after_boundary", 0))
            for row in scored_rows
        ),
        "region_local_eos_events": sum(
            int(row.get("region_local_eos_events", 0)) for row in scored_rows
        ),
        "region_local_eos_deleted_masks": sum(
            int(row.get("region_local_eos_deleted_masks", 0))
            for row in scored_rows
        ),
        "max_masks_deleted_by_one_eos": max(
            int(row.get("max_masks_deleted_by_one_eos", 0)) for row in scored_rows
        ),
        "cross_region_delete_attempts": sum(
            int(row.get("cross_region_delete_attempts", 0)) for row in scored_rows
        ),
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


def _observable_action_sequence(row: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            step["selected_position"],
            step["selected_region"],
            step["proposal_token_id"],
            step["action"],
        )
        for step in row.get("step_trace") or []
    ]


def nonempty_isolation_audit(
    candidate_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    control_by_task = {row["task_id"]: row for row in control_rows}
    missing_controls = [
        row["task_id"] for row in candidate_rows if row["task_id"] not in control_by_task
    ]
    inactive = [
        row
        for row in candidate_rows
        if int(row.get("nonempty_guard_rejection_count", 0)) == 0
    ]
    mismatches: list[dict[str, Any]] = []
    for row in inactive:
        control = control_by_task.get(row["task_id"])
        if control is None:
            continue
        checks = {
            "completion": row["completion"] == control["completion"],
            "score": row["score"] == control["score"],
            "action_sequence": _observable_action_sequence(row)
            == _observable_action_sequence(control),
            "total_forwards": row["total_forwards"] == control["total_forwards"],
        }
        if not all(checks.values()):
            mismatches.append({"task_id": row["task_id"], "checks": checks})
    return {
        "control_method": Method.V3_HARD_BUDGETED.value,
        "candidate_method": Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE.value,
        "candidate_rows": len(candidate_rows),
        "control_rows_available": len(control_rows),
        "missing_control_task_ids": missing_controls,
        "rows_without_guard_activation": len(inactive),
        "matching_inactive_rows": len(inactive) - len(mismatches),
        "mismatches": mismatches,
        "all_inactive_rows_match": not missing_controls and not mismatches,
    }


def run_scoring(args: argparse.Namespace) -> None:
    method = Method(args.method)
    predictions = read_jsonl(args.output_dir / "predictions.jsonl")
    expected = stage_size(args.stage)
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
    if method == Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE and args.stage in {
        "affected6",
        "pilot",
    }:
        control_rows = read_jsonl(
            REPO_ROOT
            / "repro_results/dreamon_progressive_v3_hard_budgeted_pilot30/scored.jsonl"
        )
        atomic_write_json(
            args.output_dir / "nonempty_isolation_audit.json",
            nonempty_isolation_audit(scored_rows, control_rows),
        )
    discard_reference_summary = None
    if method == Method.V2_HARD_V2_BOUNDARY or is_v3_method(method):
        from transformers import AutoTokenizer

        diagnostic_tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True
        )
        discard_reference_summary = write_discard_reference_diagnostics(
            args.output_dir, scored_rows, problems, diagnostic_tokenizer
        )
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
        "newline_boundary_events": sum(
            row.get("newline_boundary_events", 0) for row in scored_rows
        ),
        "mixed_newline_token_events": sum(
            row.get("mixed_newline_token_events", 0) for row in scored_rows
        ),
        "pure_newline_token_events": sum(
            row.get("pure_newline_token_events", 0) for row in scored_rows
        ),
        "multiple_newline_token_events": sum(
            row.get("multiple_newline_token_events", 0) for row in scored_rows
        ),
        "boundary_retokenized_left_token_count": sum(
            row.get("boundary_retokenized_left_token_count", 0)
            for row in scored_rows
        ),
        "discarded_masks_after_boundary": sum(
            row.get("discarded_masks_after_boundary", 0) for row in scored_rows
        ),
        "discarded_resolved_tokens_after_boundary": sum(
            row.get("discarded_resolved_tokens_after_boundary", 0)
            for row in scored_rows
        ),
        "boundary_events_with_left_masks_remaining": sum(
            row.get("boundary_events_with_left_masks_remaining", 0)
            for row in scored_rows
        ),
        "region_local_eos_events": sum(
            row.get("region_local_eos_events", 0) for row in scored_rows
        ),
        "region_local_eos_deleted_masks": sum(
            row.get("region_local_eos_deleted_masks", 0)
            for row in scored_rows
        ),
        "max_masks_deleted_by_one_eos": max(
            row.get("max_masks_deleted_by_one_eos", 0) for row in scored_rows
        ),
        "cross_region_delete_attempts": sum(
            row.get("cross_region_delete_attempts", 0) for row in scored_rows
        ),
        "exact_cycle_rows": sum(
            "exact_deterministic_cycle" in row["protocol_flags"]
            for row in scored_rows
        ),
        "budget_exhausted_rows": sum(
            bool(row.get("budget_exhausted", False)) for row in scored_rows
        ),
        "budget_draining_loop_rows": sum(
            int(row.get("budget_draining_loop_count", 0)) > 0
            for row in scored_rows
        ),
        "budget_draining_loop_count": sum(
            int(row.get("budget_draining_loop_count", 0)) for row in scored_rows
        ),
        "expand_logit_blocked_by_budget_count": sum(
            int(row.get("expand_logit_blocked_by_budget_count", 0))
            for row in scored_rows
        ),
        "expand_budget_conservation_failures": sum(
            not bool(row.get("expand_budget_conservation_passed", True))
            for row in scored_rows
        ),
        "nonempty_guard_rejection_count": sum(
            int(row.get("nonempty_guard_rejection_count", 0))
            for row in scored_rows
        ),
        "nonempty_guard_triggered_rows": sum(
            int(row.get("nonempty_guard_rejection_count", 0)) > 0
            for row in scored_rows
        ),
        "nonempty_guard_rejections_by_slot": dict(
            sorted(
                Counter(
                    event["slot"]
                    for row in scored_rows
                    for event in row.get("nonempty_guard_rejection_events", [])
                ).items()
            )
        ),
        "nonempty_guard_rejections_by_action": dict(
            sorted(
                Counter(
                    event["hypothetical_action"]
                    for row in scored_rows
                    for event in row.get("nonempty_guard_rejection_events", [])
                ).items()
            )
        ),
        "nonempty_guard_no_valid_action_rows": sum(
            "nonempty_guard_no_valid_action" in row["protocol_flags"]
            for row in scored_rows
        ),
        "slot_expand_distributions": {
            region: dict(
                sorted(Counter(row["expand_counts"][region] for row in scored_rows).items())
            )
            for region in scored_rows[0]["expand_counts"]
        },
        "discard_reference_summary": discard_reference_summary,
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
        for row in load_stage_population(args.population, args.stage)
    ]
    audit = {
        "method": args.method,
        "stage": args.stage,
        "expected_rows": stage_size(args.stage),
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
        "terminal_method_failure_rows": sum(
            classify_protocol_outcome(row) == "terminal_method_failure"
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
            any(
                key
                in {
                    "discarded_after_" + "first_line",
                    "accepted_first_line",
                    "first_line_only",
                    "posthoc_truncated_completion",
                }
                for key in row
            )
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
        + audit.get("terminal_method_failure_rows", 0)
        + audit["disallowed_protocol_violation_rows"]
    )
    if len(violations) != expected_flagged:
        raise RuntimeError(
            f"Protocol failure/violation file has {len(violations)} rows, "
            f"expected {expected_flagged}"
        )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


def _compressed_activation_history(row: Mapping[str, Any]) -> list[str]:
    compressed: list[str] = []
    for region in row.get("active_region_history", []):
        if not compressed or compressed[-1] != region:
            compressed.append(region)
    return compressed


def _v3_exact_cycle_budget_consistent(row: Mapping[str, Any]) -> bool:
    for event in row.get("exact_cycle_events", []):
        if event.get("classification") != "exact_deterministic_cycle":
            return False
        if event.get("first_pre_remaining_budget") != event.get(
            "repeat_pre_remaining_budget"
        ):
            return False
        if event.get("first_post_remaining_budget") != event.get(
            "repeat_post_remaining_budget"
        ):
            return False
    return True


def _run_v3_gate(args: argparse.Namespace) -> None:
    method = Method(args.method)
    predictions = read_jsonl(args.output_dir / "predictions.jsonl")
    scored = read_jsonl(args.output_dir / "scored.jsonl")
    summary = read_json(args.output_dir / "summary.json")
    audit = read_json(args.output_dir / "completeness_audit.json")
    expected = stage_size(args.stage)
    invariant_flags = {
        "prefix_tokens_mutated",
        "suffix_tokens_mutated",
        "locked_separator_tokens_mutated",
        "future_region_mutated_before_activation",
        "cross_region_delete_attempt",
        "parallel_state_lengths_differ",
        "pad_region_inside_real_sequence",
        "non_pad_region_in_right_padding",
        "expand_budget_conservation_failed",
        "invalid_initial_expand_budget",
        "invalid_remaining_expand_budget",
        "missing_remaining_expand_budget",
        "nonempty_guard_method_config_mismatch",
        "completed_nonempty_invariant_failed",
    }
    required_fields = {
        "newline_boundary_events",
        "region_local_eos_events",
        "cross_region_delete_attempts",
        "exact_cycle_events",
        "initial_expand_budget",
        "remaining_expand_budget",
        "expand_budget_consumed",
        "expand_budget_conservation_passed",
        "budget_exhausted",
        "budget_exhausted_forward_index",
        "expand_logit_blocked_by_budget_count",
        "budget_draining_loop_count",
        "budget_draining_loop_events",
    }
    if method == Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE:
        required_fields.update(
            {
                "nonempty_guard_rejection_count",
                "nonempty_guard_rejections_by_slot",
                "nonempty_guard_rejections_by_action",
                "nonempty_guard_rejection_events",
                "nonempty_guard_no_valid_action_count",
                "final_nonempty_invariant_passed",
            }
        )
    terminal_rows = [row for row in predictions if row["status"] != "completed"]
    engineering_checks = {
        "expected_prediction_rows": len(predictions) == expected,
        "expected_score_rows": len(scored) == expected,
        "completeness_audit": bool(audit.get("complete")),
        "all_required_diagnostic_fields": all(
            required_fields.issubset(row) for row in predictions
        ),
        "zero_runtime_errors": all(
            row["status"] != "runtime_error" for row in predictions
        ),
        "zero_invariant_errors": all(
            not invariant_flags.intersection(row["protocol_flags"])
            for row in predictions
        ),
        "zero_cross_region_delete_attempts": all(
            int(row["cross_region_delete_attempts"]) == 0 for row in predictions
        ),
        "zero_silent_unresolved_acceptance": all(
            row["status"] != "completed" or int(row["unresolved_mask_count"]) == 0
            for row in predictions
        ),
        "terminal_failures_explicit_and_empty": all(
            classify_protocol_outcome(row) == "terminal_method_failure"
            and row["completion"] == ""
            for row in terminal_rows
        ),
        "budget_conservation_all_rows": all(
            bool(row["expand_budget_conservation_passed"])
            for row in predictions
        ),
        "budget_bounds_all_rows": all(
            int(row["initial_expand_budget"]) == 64
            and 0 <= int(row["remaining_expand_budget"]) <= 64
            and int(row["expand_budget_consumed"])
            == 64 - int(row["remaining_expand_budget"])
            for row in predictions
        ),
        "no_premature_exact_cycle_classification": all(
            _v3_exact_cycle_budget_consistent(row) for row in predictions
        ),
        "budget_draining_events_nonterminal": all(
            all(
                event.get("classification") == "budget_draining_loop"
                and event.get("terminal") is False
                for event in row.get("budget_draining_loop_events", [])
            )
            for row in predictions
        ),
        "resume_dedup_config_audit": bool(audit.get("complete"))
        and bool(audit.get("all_prediction_config_hashes_match"))
        and bool(audit.get("task_order_matches_population")),
    }
    if args.stage in {"cycle5", "affected6"}:
        engineering_checks["full_step_trace_present"] = all(
            isinstance(row.get("step_trace"), list)
            and len(row["step_trace"]) == row["total_forwards"]
            for row in predictions
        )
    if method == Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE:
        isolation_path = args.output_dir / "nonempty_isolation_audit.json"
        engineering_checks.update(
            {
                "zero_completed_blank_slot": all(
                    row["status"] != "completed"
                    or not any(row["blank_region_flags"].values())
                    for row in predictions
                ),
                "final_nonempty_invariant_all_completed": all(
                    row["status"] != "completed"
                    or bool(row.get("final_nonempty_invariant_passed"))
                    for row in predictions
                ),
                "guard_rejections_are_unique_and_finite": all(
                    len(
                        {
                            (
                                event["forward_index"],
                                event["position"],
                                event["proposal_token_id"],
                            )
                            for event in row.get(
                                "nonempty_guard_rejection_events", []
                            )
                        }
                    )
                    == int(row.get("nonempty_guard_rejection_count", 0))
                    for row in predictions
                ),
                "inactive_rows_match_a": (
                    args.stage == "full"
                    or (
                        isolation_path.exists()
                        and read_json(isolation_path).get(
                            "all_inactive_rows_match"
                        )
                        is True
                    )
                ),
            }
        )
    engineering_gate_passed = all(engineering_checks.values())
    pass_at_least_18 = (
        int(summary["passed"]) >= 18 if args.stage == "pilot" else None
    )
    full_authorized = bool(
        args.stage == "pilot" and engineering_gate_passed and pass_at_least_18
    )
    payload = {
        "method": args.method,
        "stage": args.stage,
        "passed": engineering_gate_passed,
        "engineering_gate_passed": engineering_gate_passed,
        "pass_at_least_18": pass_at_least_18,
        "full_authorized": full_authorized,
        "engineering_checks": engineering_checks,
        "diagnostic_only_checks": {
            "completed_rows": summary["completed_rows"],
            "compile_passed": summary["compile_passed"],
            "passed": summary["passed"],
            "exact_matches": summary["exact_matches"],
            "blank_slot_rows": summary.get("blank_slot_rows", 0),
            "exact_cycles": summary.get("exact_cycle_count", 0),
            "forward_caps": summary.get("forward_cap_count", 0),
            "budget_exhausted_rows": summary.get("budget_exhausted_rows", 0),
            "budget_draining_loop_count": summary.get(
                "budget_draining_loop_count", 0
            ),
        },
        "recorded_at_utc": utc_now(),
    }
    atomic_write_json(args.output_dir / "gate.json", payload)
    update_progress(
        args.output_dir / "progress.json",
        method,
        args.stage,
        predictions,
        expected,
        None,
        "engineering_gate_passed"
        if engineering_gate_passed
        else "engineering_gate_failed",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if not engineering_gate_passed:
        failed = [
            name for name, value in engineering_checks.items() if not value
        ]
        raise RuntimeError(f"V3 {args.stage} engineering gate failed: {failed}")


def run_gate(args: argparse.Namespace) -> None:
    if is_v3_method(Method(args.method)):
        _run_v3_gate(args)
        return
    if args.method != Method.V2_HARD_V2_BOUNDARY.value:
        raise RuntimeError("Strict v2 gate is only defined for V2-Hard-v2")
    predictions = read_jsonl(args.output_dir / "predictions.jsonl")
    scored = read_jsonl(args.output_dir / "scored.jsonl")
    summary = read_json(args.output_dir / "summary.json")
    audit = read_json(args.output_dir / "completeness_audit.json")
    expected = stage_size(args.stage)
    invariant_flags = {
        "prefix_tokens_mutated",
        "suffix_tokens_mutated",
        "locked_separator_tokens_mutated",
        "future_region_mutated_before_activation",
        "cross_region_delete_attempt",
        "parallel_state_lengths_differ",
        "pad_region_inside_real_sequence",
        "non_pad_region_in_right_padding",
    }
    required_diagnostic_fields = {
        "newline_boundary_events",
        "mixed_newline_token_events",
        "pure_newline_token_events",
        "multiple_newline_token_events",
        "boundary_retokenized_left_token_count",
        "discarded_internal_right_text",
        "discarded_masks_after_boundary",
        "discarded_resolved_token_ids_after_boundary",
        "discarded_resolved_segments_after_boundary",
        "discarded_resolved_tokens_after_boundary",
        "boundary_events_with_left_masks_remaining",
        "region_local_eos_events",
        "region_local_eos_deleted_masks",
        "max_masks_deleted_by_one_eos",
        "cross_region_delete_attempts",
        "exact_cycle_events",
    }
    checks: dict[str, bool] = {
        "expected_prediction_rows": len(predictions) == expected,
        "expected_score_rows": len(scored) == expected,
        "completeness_audit": bool(audit.get("complete")),
        "all_required_diagnostic_fields": all(
            required_diagnostic_fields.issubset(row) for row in predictions
        ),
        "zero_unresolved_rows": all(
            int(row["unresolved_mask_count"]) == 0 for row in predictions
        ),
        "zero_runtime_errors": all(
            row["status"] != "runtime_error" for row in predictions
        ),
        "zero_protocol_errors": all(
            row["status"] != "protocol_error" for row in predictions
        ),
        "zero_exact_cycles": all(
            "exact_deterministic_cycle" not in row["protocol_flags"]
            for row in predictions
        ),
        "zero_invariant_errors": all(
            not invariant_flags.intersection(row["protocol_flags"])
            for row in predictions
        ),
        "zero_cross_region_delete_attempts": all(
            int(row["cross_region_delete_attempts"]) == 0 for row in predictions
        ),
    }
    if args.stage == "smoke":
        expected_activation = [
            "HARD_SLOT_0",
            "HARD_SLOT_1",
            "HARD_SLOT_2",
        ]
        checks.update(
            {
                "five_of_five_completed": sum(
                    row["status"] == "completed" for row in predictions
                )
                == 5,
                "sequential_activation_all_rows": all(
                    _compressed_activation_history(row) == expected_activation
                    for row in predictions
                ),
                "full_step_trace_present": all(
                    isinstance(row.get("step_trace"), list)
                    and len(row["step_trace"]) == row["total_forwards"]
                    for row in predictions
                ),
                "no_posthoc_truncation": not audit[
                    "posthoc_truncation_fields_present"
                ],
            }
        )
    elif args.stage == "pilot":
        discard_path = args.output_dir / "discard_reference_diagnostics.json"
        cases_path = args.output_dir / "discard_reference_cases_top10.jsonl"
        checks.update(
            {
                "thirty_of_thirty_completed": sum(
                    row["status"] == "completed" for row in predictions
                )
                == 30,
                "compile_at_least_24": int(summary["compile_passed"]) >= 24,
                "pass_at_least_10": int(summary["passed"]) >= 10,
                "discard_reference_diagnostic_complete": (
                    discard_path.exists()
                    and cases_path.exists()
                    and read_json(discard_path).get(
                        "oracle_mechanism_diagnostic"
                    )
                    is True
                ),
            }
        )
    else:
        checks.update(
            {
                "full_642_complete": len(predictions) == 642,
                "no_early_stop_file": not (
                    args.output_dir / "early_stop.json"
                ).exists(),
                "discard_reference_diagnostic_complete": (
                    args.output_dir / "discard_reference_diagnostics.json"
                ).exists(),
            }
        )
    passed = all(checks.values())
    payload = {
        "method": args.method,
        "stage": args.stage,
        "passed": passed,
        "checks": checks,
        "summary": {
            "predictions": len(predictions),
            "completed": sum(
                row["status"] == "completed" for row in predictions
            ),
            "passed": summary["passed"],
            "compile_passed": summary["compile_passed"],
            "exact_cycles": summary.get("exact_cycle_count", 0),
            "forward_caps": summary.get("forward_cap_count", 0),
        },
        "recorded_at_utc": utc_now(),
    }
    atomic_write_json(args.output_dir / "gate.json", payload)
    update_progress(
        args.output_dir / "progress.json",
        Method(args.method),
        args.stage,
        predictions,
        expected,
        None,
        "gate_passed" if passed else "gate_failed",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"V2-Hard-v2 {args.stage} gate failed: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["generate", "score", "audit", "gate"], required=True
    )
    parser.add_argument("--method", choices=[method.value for method in Method], required=True)
    parser.add_argument("--stage", choices=list(ALL_STAGE_SIZES), required=True)
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
    elif args.mode == "audit":
        run_audit(args)
    else:
        run_gate(args)


if __name__ == "__main__":
    main()
