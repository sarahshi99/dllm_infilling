#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import tokenize
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask
from expvision_dllm_clean.decode import prepare_model_inputs, run_vanilla_decode
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed
from expvision_dllm_clean.verifier import run_verifier_stack


MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
CANVAS_LENGTHS = (16, 32, 64, 128)
SEEDS = (0, 1)
TOTAL_STEPS = 64
SOURCE_CONFIG = "HumanEval-RandomSpanInfillingLight"
TEST_LOCK = REPO / "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"
GROUPED_TEST_TASKS = REPO / "analysis_outputs/grouped_split_20260702_accel2/test_tasks.json"
DYNAMIC_LOCAL_CALLS = {"locals", "globals", "vars", "eval", "exec", "getattr", "setattr", "delattr"}


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_group(task_id: str) -> str:
    match = re.search(r"HumanEval/(\d+)", task_id)
    if not match:
        raise ValueError(f"Cannot derive HumanEval task group from {task_id!r}")
    return f"HumanEval/{match.group(1)}"


def length_bucket(reference_middle_tokens: int) -> str:
    if reference_middle_tokens <= 8:
        return "short"
    if reference_middle_tokens <= 16:
        return "medium"
    if reference_middle_tokens <= 24:
        return "long"
    return "extreme"


def load_frozen_groups() -> tuple[set[str], dict[str, Any]]:
    if not TEST_LOCK.exists():
        raise FileNotFoundError(TEST_LOCK)
    lock = json.loads(TEST_LOCK.read_text(encoding="utf-8"))
    if lock.get("test_status") != "sealed" or int(lock.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError("Frozen controller test lock is not sealed with test_evaluation_count=0")
    groups = {str(item) for item in lock.get("test_task_ids", [])}
    if GROUPED_TEST_TASKS.exists():
        groups.update(str(item) for item in json.loads(GROUPED_TEST_TASKS.read_text(encoding="utf-8")))
    return groups, lock


def code_task(row: Mapping[str, Any]) -> CodeTask:
    prefix = str(row["prompt"])
    suffix = str(row["suffix"])
    return CodeTask(
        task_id=str(row["task_id"]),
        prefix=prefix,
        suffix=suffix,
        full_prompt=prefix + "<FILL_ME>" + suffix,
        test_code=str(row["test"]),
        entry_point=str(row["entry_point"]),
        canonical_solution=str(row.get("canonical_solution", "")),
        raw=dict(row),
    )


def build_candidate_specs(reference_middle_tokens: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for canvas in CANVAS_LENGTHS:
        for seed in SEEDS:
            specs.append(
                {
                    "candidate_kind": "deployable_grid",
                    "control_label": "fixed64_control" if (canvas, seed) == (64, 0) else "non_oracle_candidate",
                    "deployable": True,
                    "canvas_tokens": canvas,
                    "seed": seed,
                    "total_steps": TOTAL_STEPS,
                }
            )
    specs.append(
        {
            "candidate_kind": "oracle_sufficient_diagnostic_ceiling",
            "control_label": "diagnostic_ceiling_only",
            "deployable": False,
            "canvas_tokens": max(1, int(reference_middle_tokens)),
            "seed": 0,
            "total_steps": TOTAL_STEPS,
        }
    )
    return specs


def alpha_candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_kind": "alpha_renamed_auxiliary",
            "control_label": "metamorphic_auxiliary_only",
            "deployable": False,
            "canvas_tokens": canvas,
            "seed": seed,
            "total_steps": TOTAL_STEPS,
        }
        for canvas in CANVAS_LENGTHS
        for seed in SEEDS
    ]


def candidate_key(row_key: str, candidate_kind: str, canvas_tokens: int, seed: int) -> str:
    return f"{row_key}|{candidate_kind}|canvas={int(canvas_tokens)}|seed={int(seed)}"


def expected_candidate_keys(manifest: Sequence[Mapping[str, Any]], include_alpha: bool = False) -> set[str]:
    keys: set[str] = set()
    for row in manifest:
        for spec in build_candidate_specs(int(row["reference_middle_tokens"])):
            keys.add(candidate_key(str(row["row_key"]), str(spec["candidate_kind"]), int(spec["canvas_tokens"]), int(spec["seed"])))
        if include_alpha and bool(row.get("alpha_reference_verified", False)):
            for spec in alpha_candidate_specs():
                keys.add(candidate_key(str(row["row_key"]), str(spec["candidate_kind"]), int(spec["canvas_tokens"]), int(spec["seed"])))
    return keys


def build_manifest(rows: Sequence[Mapping[str, Any]], tokenizer: Any, frozen_groups: set[str]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    seen_task_groups: set[str] = set()
    for row_id, row in enumerate(rows):
        tid = str(row["task_id"])
        group = task_group(tid)
        if group in frozen_groups:
            continue
        if group in seen_task_groups:
            raise RuntimeError(f"RandomSpanLight expected one row per group, duplicate allowed group {group}")
        seen_task_groups.add(group)
        reference = str(row.get("canonical_solution", ""))
        reference_tokens = len(tokenizer.encode(reference, add_special_tokens=False))
        if reference_tokens <= 0:
            raise RuntimeError(f"Empty reference middle for {tid}")
        row_key = f"{SOURCE_CONFIG}:{row_id}:{sha256_text(tid)[:16]}"
        manifest.append(
            {
                "case_index": len(manifest),
                "source_row_id": row_id,
                "row_key": row_key,
                "source_config": SOURCE_CONFIG,
                "task_id": tid,
                "task_group": group,
                "reference_middle_tokens": reference_tokens,
                "length_bucket": length_bucket(reference_tokens),
                "prefix_tokens": len(tokenizer.encode(str(row["prompt"]), add_special_tokens=False)),
                "suffix_tokens": len(tokenizer.encode(str(row["suffix"]), add_special_tokens=False)),
                "frozen_controller_test_row": False,
                "alpha_structural_eligible": False,
                "alpha_reference_verified": False,
                "alpha_status": "not_checked",
            }
        )
    return manifest


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def _absolute_offset(offsets: Sequence[int], line: int, col: int) -> int:
    return offsets[line - 1] + col


def _apply_replacements(text: str, replacements: Sequence[tuple[int, int, str]], start: int, end: int) -> str:
    local = [(left - start, right - start, value) for left, right, value in replacements if start <= left < right <= end]
    out = text[start:end]
    for left, right, value in sorted(local, reverse=True):
        out = out[:left] + value + out[right:]
    return out


def build_alpha_renamed_task(row: Mapping[str, Any]) -> dict[str, Any] | None:
    prefix = str(row["prompt"])
    middle = str(row.get("canonical_solution", ""))
    suffix = str(row["suffix"])
    full = prefix + middle + suffix
    try:
        tree = ast.parse(full)
    except SyntaxError:
        return None
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == row["entry_point"]]
    if len(functions) != 1:
        return None
    root = functions[0]
    for node in ast.walk(root):
        if node is not root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return None
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DYNAMIC_LOCAL_CALLS:
            return None

    local_names: set[str] = set()
    args = list(root.args.posonlyargs) + list(root.args.args) + list(root.args.kwonlyargs)
    if root.args.vararg:
        args.append(root.args.vararg)
    if root.args.kwarg:
        args.append(root.args.kwarg)
    parameter_names = {arg.arg for arg in args}
    for node in ast.walk(root):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            local_names.add(node.id)
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ExceptHandler, ast.Match, ast.JoinedStr)):
            return None
    local_names.discard(str(row["entry_point"]))
    local_names.difference_update(parameter_names)
    if not local_names:
        return None

    existing_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    mapping: dict[str, str] = {}
    counter = 0
    for name in sorted(local_names):
        while f"phase5_local_{counter}" in existing_names:
            counter += 1
        mapping[name] = f"phase5_local_{counter}"
        counter += 1

    offsets = _line_offsets(full)
    replacements: set[tuple[int, int, str]] = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Name) and node.id in mapping:
            replacements.add(
                (
                    _absolute_offset(offsets, node.lineno, node.col_offset),
                    _absolute_offset(offsets, node.end_lineno, node.end_col_offset),
                    mapping[node.id],
                )
            )
    ordered = sorted(replacements)
    prefix_end = len(prefix)
    middle_end = prefix_end + len(middle)
    transformed = {
        **dict(row),
        "prompt": _apply_replacements(full, ordered, 0, prefix_end),
        "canonical_solution": _apply_replacements(full, ordered, prefix_end, middle_end),
        "suffix": _apply_replacements(full, ordered, middle_end, len(full)),
        "rename_map": mapping,
        "alpha_transform_id": sha256_text(json.dumps(mapping, sort_keys=True))[:16],
    }
    try:
        ast.parse(transformed["prompt"] + transformed["canonical_solution"] + transformed["suffix"])
    except SyntaxError:
        return None
    return transformed


def replace_name_tokens(text: str, mapping: Mapping[str, str]) -> tuple[str, bool]:
    if not mapping:
        return text, True
    try:
        tokens = []
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.NAME and token.string in mapping:
                token = tokenize.TokenInfo(token.type, mapping[token.string], token.start, token.end, token.line)
            tokens.append(token)
        return tokenize.untokenize(tokens), True
    except (tokenize.TokenError, IndentationError):
        pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in sorted(mapping, key=len, reverse=True)) + r")\b")
        return pattern.sub(lambda match: mapping[match.group(1)], text), False


def verify_alpha_manifest(
    manifest: list[dict[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    transformed_by_key: dict[str, dict[str, Any]] = {}
    for item in manifest:
        row = source_rows[int(item["source_row_id"])]
        transformed = build_alpha_renamed_task(row)
        if transformed is None:
            item["alpha_status"] = "strict_transform_rejected"
            continue
        item["alpha_structural_eligible"] = True
        task = code_task(transformed)
        verification = run_verifier_stack(
            task,
            full_code=task.prefix + str(transformed["canonical_solution"]) + task.suffix,
            completion_without_suffix=str(transformed["canonical_solution"]),
        )
        tier3 = verification.get("tier3_unit_tests")
        verified = bool(tier3 and tier3.passed)
        item["alpha_reference_verified"] = verified
        item["alpha_status"] = "reference_verified" if verified else "reference_verification_failed"
        item["alpha_transform_id"] = transformed["alpha_transform_id"]
        item["alpha_renamed_identifier_count"] = len(transformed["rename_map"])
        if verified:
            transformed_by_key[str(item["row_key"])] = transformed
    return transformed_by_key


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


def verification_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    verification = result.get("verification") or {}
    tier1 = verification.get("tier1_parse_compile") or {}
    tier2 = verification.get("tier2_smoke_exec") or {}
    tier3 = verification.get("tier3_unit_tests") or {}
    error = next((item for item in [tier3, tier2, tier1] if item and not bool(item.get("passed", False))), {})
    return {
        "compile_passed": bool(tier1.get("passed", False)),
        "tier2_smoke_exec_passed": bool(tier2.get("passed", False)),
        "tier3_unit_tests_passed": bool(tier3.get("passed", False)),
        "error_type": error.get("error_type") or "",
        "error_message": str(error.get("error_message") or "")[:240],
    }


def ast_hash(code: str) -> str:
    try:
        return sha256_text(ast.dump(ast.parse(code), include_attributes=False))
    except SyntaxError:
        return ""


def run_candidate(
    *,
    manifest_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    task: CodeTask,
    spec: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    alpha_inverse_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    key = candidate_key(str(manifest_row["row_key"]), str(spec["candidate_kind"]), int(spec["canvas_tokens"]), int(spec["seed"]))
    started = time.perf_counter()
    set_global_seed(int(spec["seed"]))
    cfg = cfg_for(int(spec["canvas_tokens"]), int(spec["seed"]))
    try:
        result = run_vanilla_decode(task, tokenizer, model, cfg)
        middle_text = str(result.get("middle_text", ""))
        full_code = str(result.get("code", ""))
        inverse_middle = ""
        inverse_full = ""
        inverse_tokenize_exact = True
        if alpha_inverse_map:
            inverse = {new: old for old, new in alpha_inverse_map.items()}
            inverse_middle, middle_exact = replace_name_tokens(middle_text, inverse)
            inverse_full, full_exact = replace_name_tokens(full_code, inverse)
            inverse_tokenize_exact = bool(middle_exact and full_exact)
        metrics = dict(result.get("metrics") or {})
        raw = {
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
            "prefix_text": task.prefix,
            "code": full_code,
            "middle_text": middle_text,
            "suffix_text": task.suffix,
            "reference_middle": str(source_row.get("canonical_solution", "")),
            "candidate_full_code_sha256": sha256_text(full_code),
            "candidate_middle_sha256": sha256_text(middle_text),
            "candidate_full_ast_sha256": ast_hash(full_code),
            "candidate_middle_tokens": len(tokenizer.encode(middle_text, add_special_tokens=False)),
            "inverse_renamed_full_code": inverse_full,
            "inverse_renamed_middle_text": inverse_middle,
            "inverse_renamed_full_code_sha256": sha256_text(inverse_full) if inverse_full else "",
            "inverse_renamed_middle_sha256": sha256_text(inverse_middle) if inverse_middle else "",
            "inverse_renamed_full_ast_sha256": ast_hash(inverse_full) if inverse_full else "",
            "inverse_rename_tokenize_exact": inverse_tokenize_exact,
            "metrics": metrics,
            "verification": result.get("verification") or {},
            "diagnostics": result.get("diagnostics") or {},
            "wall_sec": time.perf_counter() - started,
        }
        return raw
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
            "wall_sec": time.perf_counter() - started,
        }


def compact_candidate_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    metrics = raw.get("metrics") or {}
    verification = verification_fields(raw)
    return {
        "candidate_key": raw.get("candidate_key", ""),
        "row_key": raw.get("row_key", ""),
        "case_index": raw.get("case_index", ""),
        "task_group": raw.get("task_group", ""),
        "length_bucket": raw.get("length_bucket", ""),
        "candidate_kind": raw.get("candidate_kind", ""),
        "control_label": raw.get("control_label", ""),
        "deployable": raw.get("deployable", False),
        "canvas_tokens": raw.get("canvas_tokens", ""),
        "seed": raw.get("seed", ""),
        "total_steps": raw.get("total_steps", ""),
        "status": raw.get("status", ""),
        "passed": raw.get("passed", False),
        "reference_middle_tokens_offline_only": raw.get("reference_middle_tokens", ""),
        "candidate_middle_tokens": raw.get("candidate_middle_tokens", ""),
        "candidate_full_code_sha256": raw.get("candidate_full_code_sha256", ""),
        "candidate_middle_sha256": raw.get("candidate_middle_sha256", ""),
        "candidate_full_ast_sha256": raw.get("candidate_full_ast_sha256", ""),
        "inverse_renamed_full_code_sha256": raw.get("inverse_renamed_full_code_sha256", ""),
        "inverse_renamed_middle_sha256": raw.get("inverse_renamed_middle_sha256", ""),
        "inverse_renamed_full_ast_sha256": raw.get("inverse_renamed_full_ast_sha256", ""),
        "inverse_rename_tokenize_exact": raw.get("inverse_rename_tokenize_exact", ""),
        "mean_final_confidence": metrics.get("mean_final_confidence", ""),
        "decode_sec": metrics.get("decode_sec", ""),
        "verification_sec": metrics.get("verification_sec", ""),
        "total_sec_including_probe": metrics.get("total_sec_including_probe", raw.get("wall_sec", "")),
        "wall_sec": raw.get("wall_sec", ""),
        **verification,
    }


def audit_rows(rows: Sequence[Mapping[str, Any]], expected_keys: set[str]) -> dict[str, Any]:
    counts = Counter(str(row.get("candidate_key", "")) for row in rows)
    observed = set(counts)
    missing = sorted(expected_keys - observed)
    extra = sorted(observed - expected_keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    errors = [str(row.get("candidate_key", "")) for row in rows if row.get("status", "ok") != "ok"]
    return {
        "passed": not missing and not extra and not duplicates and not errors,
        "expected_count": len(expected_keys),
        "observed_count": len(rows),
        "unique_count": len(observed),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "duplicate_count": len(duplicates),
        "error_count": len(errors),
        "missing_keys": missing[:100],
        "extra_keys": extra[:100],
        "duplicate_keys": duplicates[:100],
        "error_keys": errors[:100],
    }


def choose_smoke_manifest(manifest: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(manifest):
        raise ValueError(f"Invalid smoke case count {count}")
    selected: list[dict[str, Any]] = []
    buckets = ["short", "medium", "long", "extreme"]
    per_bucket = count // len(buckets)
    remainder = count % len(buckets)
    for bucket_index, bucket in enumerate(buckets):
        candidates = sorted(
            (dict(row) for row in manifest if row["length_bucket"] == bucket),
            key=lambda row: (int(row["reference_middle_tokens"]), int(row["case_index"])),
        )
        take = per_bucket + (1 if bucket_index < remainder else 0)
        selected.extend(candidates[:take])
    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} smoke rows, expected {count}")
    return sorted(selected, key=lambda row: int(row["case_index"]))


def protocol_validation(tasks: Sequence[CodeTask], tokenizer: Any, model: Any) -> dict[str, Any]:
    cfg = cfg_for(128, 0)
    max_positions = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    input_lengths: list[int] = []
    correct_canvas = True
    for task in tasks:
        prepared = prepare_model_inputs(task, tokenizer, 128, cfg)
        input_lengths.append(len(prepared["input_ids"]))
        correct_canvas = correct_canvas and prepared["middle_end"] - prepared["middle_start"] == 128
    over_limit = sum(max_positions is not None and length > int(max_positions) for length in input_lengths)
    supported = bool(
        cfg.decode.mask_length_source == "fixed"
        and cfg.decode.fixed_mask_length == 128
        and cfg.decode.total_steps == TOTAL_STEPS
        and correct_canvas
        and over_limit == 0
    )
    return {
        "supported": supported,
        "canvas_tokens": 128,
        "total_steps": TOTAL_STEPS,
        "mask_length_source": cfg.decode.mask_length_source,
        "validated_task_count": len(tasks),
        "min_input_tokens": min(input_lengths),
        "max_input_tokens": max(input_lengths),
        "over_model_limit_count": over_limit,
        "max_position_embeddings": max_positions,
        "same_decoder": "expvision_dllm_clean.decode.run_vanilla_decode",
        "same_schedule": "linear_target_masks",
        "substitution_used": False,
    }


def run_population(
    *,
    selected_manifest: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    raw_path: Path,
    tokenizer: Any,
    model: Any,
) -> int:
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    counts = Counter(str(row.get("candidate_key", "")) for row in existing)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Refusing resume with duplicate candidate keys: {duplicates[:5]}")
    completed = set(counts)
    written = 0
    for item in selected_manifest:
        source = source_rows[int(item["source_row_id"])]
        task = code_task(source)
        specs = build_candidate_specs(int(item["reference_middle_tokens"]))
        for spec in specs:
            key = candidate_key(str(item["row_key"]), str(spec["candidate_kind"]), int(spec["canvas_tokens"]), int(spec["seed"]))
            if key in completed:
                continue
            row = run_candidate(
                manifest_row=item,
                source_row=source,
                task=task,
                spec=spec,
                tokenizer=tokenizer,
                model=model,
            )
            append_jsonl(raw_path, row)
            completed.add(key)
            written += 1
            print(json.dumps({"candidate_key": key, "status": row["status"], "written": written}, sort_keys=True), flush=True)

    return written


def run_alpha_population(
    *,
    manifest: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    transformed_by_key: Mapping[str, Mapping[str, Any]],
    raw_path: Path,
    tokenizer: Any,
    model: Any,
) -> int:
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    counts = Counter(str(row.get("candidate_key", "")) for row in existing)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Refusing alpha resume with duplicate candidate keys: {duplicates[:5]}")
    completed = set(counts)
    written = 0
    for item in manifest:
        transformed = transformed_by_key.get(str(item["row_key"]))
        if transformed is None:
            continue
        source = source_rows[int(item["source_row_id"])]
        alpha_task = code_task(transformed)
        for spec in alpha_candidate_specs():
            key = candidate_key(str(item["row_key"]), str(spec["candidate_kind"]), int(spec["canvas_tokens"]), int(spec["seed"]))
            if key in completed:
                continue
            row = run_candidate(
                manifest_row=item,
                source_row=source,
                task=alpha_task,
                spec=spec,
                tokenizer=tokenizer,
                model=model,
                alpha_inverse_map=transformed["rename_map"],
            )
            row["alpha_transform_id"] = transformed["alpha_transform_id"]
            append_jsonl(raw_path, row)
            completed.add(key)
            written += 1
            print(json.dumps({"candidate_key": key, "status": row["status"], "written": written}, sort_keys=True), flush=True)
    return written


def compact_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_index": row["case_index"],
        "row_key": row["row_key"],
        "source_row_id": row["source_row_id"],
        "task_group": row["task_group"],
        "source_config": row["source_config"],
        "reference_middle_tokens_offline_only": row["reference_middle_tokens"],
        "length_bucket_offline_only": row["length_bucket"],
        "prefix_tokens": row["prefix_tokens"],
        "suffix_tokens": row["suffix_tokens"],
        "frozen_controller_test_row": row["frozen_controller_test_row"],
        "alpha_structural_eligible": row["alpha_structural_eligible"],
        "alpha_reference_verified": row["alpha_reference_verified"],
        "alpha_status": row["alpha_status"],
        "alpha_transform_id": row.get("alpha_transform_id", ""),
        "alpha_renamed_identifier_count": row.get("alpha_renamed_identifier_count", ""),
    }


def create_compact_artifacts(
    *,
    compact_dir: Path,
    manifest: Sequence[Mapping[str, Any]],
    selected_manifest: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    phase_name: str,
    raw_dir: Path,
    dataset_jsonl: Path,
    started_at: str,
    wall_sec: float,
    peak_memory_bytes: int,
    test_lock: Mapping[str, Any],
    resume_noop_writes: int,
) -> dict[str, Any]:
    compact_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = {str(row["row_key"]) for row in selected_manifest}
    phase_rows = [
        row
        for row in raw_rows
        if str(row.get("row_key", "")) in selected_keys
        and row.get("candidate_kind") in {"deployable_grid", "oracle_sufficient_diagnostic_ceiling"}
    ]
    expected = expected_candidate_keys(selected_manifest, include_alpha=False)
    audit = audit_rows(phase_rows, expected)
    base_expected = expected
    base_rows = [row for row in phase_rows if row.get("candidate_kind") != "alpha_renamed_auxiliary"]
    base_audit = audit_rows(base_rows, base_expected)
    schema_required = {
        "candidate_key", "row_key", "candidate_kind", "canvas_tokens", "seed", "total_steps", "status", "passed", "verification"
    }
    schema_passed = all(schema_required <= set(row) for row in phase_rows)
    evaluator_passed = all(bool(row.get("verification")) for row in phase_rows if row.get("status") == "ok")
    canvas128_rows = [row for row in phase_rows if int(row.get("canvas_tokens", -1)) == 128]
    canvas128_generation_passed = bool(canvas128_rows) and all(row.get("status") == "ok" for row in canvas128_rows)
    resume_completed_keys = {str(row.get("candidate_key")) for row in phase_rows}
    resume_passed = (
        len(resume_completed_keys) == len(phase_rows)
        and resume_completed_keys == expected
        and int(resume_noop_writes) == 0
    )
    frozen_ok = (
        test_lock.get("test_status") == "sealed"
        and int(test_lock.get("test_evaluation_count", -1)) == 0
        and not any(bool(row.get("frozen_controller_test_row", False)) for row in selected_manifest)
    )
    gate_passed = bool(
        audit["passed"]
        and base_audit["passed"]
        and schema_passed
        and evaluator_passed
        and protocol.get("supported")
        and canvas128_generation_passed
        and resume_passed
        and frozen_ok
    )
    compact_rows = [compact_candidate_row(row) for row in phase_rows]
    write_csv(compact_dir / f"{phase_name}_candidate_results.csv", compact_rows)
    write_csv(compact_dir / "manifest.csv", [compact_manifest_row(row) for row in manifest])
    write_json(compact_dir / "protocol_validation.json", protocol)
    audit_payload = {
        "phase": phase_name,
        "passed": gate_passed,
        "schema_passed": schema_passed,
        "evaluator_executed_for_all_ok_rows": evaluator_passed,
        "canvas128_generation_passed": canvas128_generation_passed,
        "resume_passed": resume_passed,
        "resume_noop_writes": int(resume_noop_writes),
        "frozen_test_invariant_passed": frozen_ok,
        "all_rows": audit,
        "base_bank": base_audit,
    }
    write_json(compact_dir / f"{phase_name}_audit.json", audit_payload)
    pass_counts = Counter(str(row.get("candidate_kind")) for row in phase_rows if bool(row.get("passed", False)))
    kind_counts = Counter(str(row.get("candidate_kind")) for row in phase_rows)
    latencies = [float((row.get("metrics") or {}).get("total_sec_including_probe") or row.get("wall_sec") or 0.0) for row in phase_rows]
    decode_secs = [float((row.get("metrics") or {}).get("decode_sec") or 0.0) for row in phase_rows]
    verification_secs = [float((row.get("metrics") or {}).get("verification_sec") or 0.0) for row in phase_rows]
    summary = {
        "phase": phase_name,
        "verdict": f"{phase_name}_passed" if gate_passed else f"{phase_name}_failed",
        "gate_passed": gate_passed,
        "selected_case_count": len(selected_manifest),
        "full_manifest_case_count": len(manifest),
        "candidate_rows": len(phase_rows),
        "candidate_kind_counts": dict(kind_counts),
        "candidate_kind_pass_counts": dict(pass_counts),
        "base_expected_rows": len(base_expected),
        "denoising_steps_per_candidate": TOTAL_STEPS,
        "wall_sec": wall_sec,
        "summed_candidate_latency_sec": sum(latencies),
        "summed_gpu_decode_sec": sum(decode_secs),
        "summed_verification_sec": sum(verification_secs),
        "mean_candidate_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
        "peak_cuda_memory_bytes": peak_memory_bytes,
        "raw_output_dir_local_only": str(raw_dir),
        "dataset_jsonl_local_read_only": str(dataset_jsonl),
        "raw_code_committed": False,
        "frozen_test_status": test_lock.get("test_status"),
        "test_evaluation_count": test_lock.get("test_evaluation_count"),
        "audit": audit_payload,
    }
    write_json(compact_dir / f"{phase_name}_summary.json", summary)
    return summary


def create_alpha_compact_artifacts(
    *,
    compact_dir: Path,
    manifest: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    wall_sec: float,
    peak_memory_bytes: int,
    test_lock: Mapping[str, Any],
) -> dict[str, Any]:
    alpha_rows = [row for row in raw_rows if row.get("candidate_kind") == "alpha_renamed_auxiliary"]
    expected = expected_candidate_keys(manifest, include_alpha=True) - expected_candidate_keys(manifest, include_alpha=False)
    audit = audit_rows(alpha_rows, expected)
    frozen_ok = test_lock.get("test_status") == "sealed" and int(test_lock.get("test_evaluation_count", -1)) == 0
    summary = {
        "verdict": "alpha_auxiliary_passed" if audit["passed"] and frozen_ok else "alpha_auxiliary_failed",
        "gate_passed": bool(audit["passed"] and frozen_ok),
        "eligible_task_count": sum(bool(row.get("alpha_reference_verified", False)) for row in manifest),
        "candidate_rows": len(alpha_rows),
        "expected_candidate_rows": len(expected),
        "denoising_steps_per_candidate": TOTAL_STEPS,
        "wall_sec_since_run_start": wall_sec,
        "peak_cuda_memory_bytes": peak_memory_bytes,
        "audit": audit,
        "frozen_test_status": test_lock.get("test_status"),
        "test_evaluation_count": test_lock.get("test_evaluation_count"),
    }
    write_csv(compact_dir / "alpha_auxiliary_candidate_results.csv", [compact_candidate_row(row) for row in alpha_rows])
    write_csv(compact_dir / "manifest.csv", [compact_manifest_row(row) for row in manifest])
    write_json(compact_dir / "alpha_auxiliary_audit.json", {"passed": summary["gate_passed"], **audit})
    write_json(compact_dir / "alpha_auxiliary_summary.json", summary)
    return summary


def render_report(smoke: Mapping[str, Any], full: Mapping[str, Any] | None, alpha: Mapping[str, Any] | None) -> str:
    lines = [
        "# Phase 5 RandomSpanLight Candidate Bank",
        "",
        f"Smoke verdict: `{smoke['verdict']}`.",
        f"Smoke cases/candidates: `{smoke['selected_case_count']}` / `{smoke['candidate_rows']}`.",
        "Canvas 128 uses the same fixed-mask vanilla decoder and 64-step schedule; no substitute length is allowed.",
        "Raw generated code remains in ignored local output. Compact artifacts contain hashes and metrics only.",
        "Frozen controller test remains sealed with `test_evaluation_count=0`.",
    ]
    if full is not None:
        lines.extend(
            [
                "",
                f"Full verdict: `{full['verdict']}`.",
                f"Full cases/candidates: `{full['selected_case_count']}` / `{full['candidate_rows']}`.",
                f"Base bank expected rows: `{full['base_expected_rows']}`.",
                f"Mean candidate latency: `{full['mean_candidate_latency_sec']:.4f}` seconds.",
                f"Summed GPU decode time: `{full['summed_gpu_decode_sec']:.4f}` seconds.",
                f"Summed verification time: `{full['summed_verification_sec']:.4f}` seconds.",
                f"Peak CUDA memory: `{full['peak_cuda_memory_bytes']}` bytes.",
            ]
        )
    if alpha is not None:
        lines.extend(
            [
                "",
                f"Alpha auxiliary verdict: `{alpha['verdict']}`.",
                f"Verified alpha tasks/candidates: `{alpha['eligible_task_count']}` / `{alpha['candidate_rows']}`.",
                "These rows are auxiliary F2 mirrors, not deployable bank candidates.",
            ]
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    dataset_jsonl = Path(args.dataset_jsonl).resolve()
    raw_dir = Path(args.output_dir).resolve()
    compact_dir = Path(args.compact_dir).resolve()
    raw_path = raw_dir / "candidate_bank_raw.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    compact_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    start = time.perf_counter()
    frozen_groups, test_lock = load_frozen_groups()
    source_rows = read_jsonl(dataset_jsonl)
    if len(source_rows) != 164:
        raise RuntimeError(f"Expected 164 RandomSpanLight source rows, found {len(source_rows)}")

    tokenizer, model = load_model_and_tokenizer(cfg_for(64, 0).model)
    manifest = build_manifest(source_rows, tokenizer, frozen_groups)
    if len(manifest) != 148:
        raise RuntimeError(f"Expected exactly 148 allowed rows after frozen exclusion, found {len(manifest)}")
    if len({row["task_group"] for row in manifest}) != 148:
        raise RuntimeError("Allowed RandomSpanLight manifest must contain 148 unique HumanEval groups")
    smoke_manifest = choose_smoke_manifest(manifest, int(args.smoke_cases))
    protocol = protocol_validation(
        [code_task(source_rows[int(item["source_row_id"])]) for item in manifest],
        tokenizer,
        model,
    )
    if not protocol["supported"]:
        write_json(compact_dir / "blocker_canvas128.json", {"verdict": "blocked_canvas128_not_supported_same_protocol", **protocol})
        raise RuntimeError("Canvas 128 is not supported under the same decoding protocol; no substitution performed")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None  # type: ignore[assignment]

    run_population(
        selected_manifest=smoke_manifest,
        source_rows=source_rows,
        raw_path=raw_path,
        tokenizer=tokenizer,
        model=model,
    )
    smoke_resume_noop_writes = run_population(
        selected_manifest=smoke_manifest,
        source_rows=source_rows,
        raw_path=raw_path,
        tokenizer=tokenizer,
        model=model,
    )
    smoke_rows = read_jsonl(raw_path)
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch is not None and torch.cuda.is_available() else 0
    smoke_summary = create_compact_artifacts(
        compact_dir=compact_dir,
        manifest=manifest,
        selected_manifest=smoke_manifest,
        raw_rows=smoke_rows,
        protocol=protocol,
        phase_name="smoke",
        raw_dir=raw_dir,
        dataset_jsonl=dataset_jsonl,
        started_at=started_at,
        wall_sec=time.perf_counter() - start,
        peak_memory_bytes=peak_memory,
        test_lock=test_lock,
        resume_noop_writes=smoke_resume_noop_writes,
    )
    if not smoke_summary["gate_passed"]:
        (compact_dir / "report.md").write_text(render_report(smoke_summary, None, None), encoding="utf-8")
        write_json(
            compact_dir / "run_manifest.json",
            {
                "status": "smoke_failed",
                "started_at_utc": started_at,
                "ended_at_utc": utc_now(),
                "command": sys.argv,
                "model": MODEL_PATH,
                "dataset_jsonl": str(dataset_jsonl),
                "raw_output_dir": str(raw_dir),
                "compact_output_dir": str(compact_dir),
                "smoke": smoke_summary,
                "test_evaluation_count": 0,
            },
        )
        return 2

    full_summary: dict[str, Any] | None = None
    alpha_summary: dict[str, Any] | None = None
    if args.auto_full:
        run_population(
            selected_manifest=manifest,
            source_rows=source_rows,
            raw_path=raw_path,
            tokenizer=tokenizer,
            model=model,
        )
        full_resume_noop_writes = run_population(
            selected_manifest=manifest,
            source_rows=source_rows,
            raw_path=raw_path,
            tokenizer=tokenizer,
            model=model,
        )
        full_rows = read_jsonl(raw_path)
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch is not None and torch.cuda.is_available() else peak_memory
        full_summary = create_compact_artifacts(
            compact_dir=compact_dir,
            manifest=manifest,
            selected_manifest=manifest,
            raw_rows=full_rows,
            protocol=protocol,
            phase_name="full",
            raw_dir=raw_dir,
            dataset_jsonl=dataset_jsonl,
            started_at=started_at,
            wall_sec=time.perf_counter() - start,
            peak_memory_bytes=peak_memory,
            test_lock=test_lock,
            resume_noop_writes=full_resume_noop_writes,
        )
        if full_summary["gate_passed"]:
            transformed_by_key = verify_alpha_manifest(manifest, source_rows)
            run_alpha_population(
                manifest=manifest,
                source_rows=source_rows,
                transformed_by_key=transformed_by_key,
                raw_path=raw_path,
                tokenizer=tokenizer,
                model=model,
            )
            all_rows = read_jsonl(raw_path)
            peak_memory = int(torch.cuda.max_memory_allocated()) if torch is not None and torch.cuda.is_available() else peak_memory
            alpha_summary = create_alpha_compact_artifacts(
                compact_dir=compact_dir,
                manifest=manifest,
                raw_rows=all_rows,
                wall_sec=time.perf_counter() - start,
                peak_memory_bytes=peak_memory,
                test_lock=test_lock,
            )
    (compact_dir / "report.md").write_text(render_report(smoke_summary, full_summary, alpha_summary), encoding="utf-8")
    final_status = "completed"
    if full_summary is not None and not full_summary["gate_passed"]:
        final_status = "full_failed"
    elif alpha_summary is not None and not alpha_summary["gate_passed"]:
        final_status = "alpha_failed"
    write_json(
        compact_dir / "run_manifest.json",
        {
            "status": final_status,
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
            "wall_sec": time.perf_counter() - start,
            "command": sys.argv,
            "model": MODEL_PATH,
            "dataset_jsonl": str(dataset_jsonl),
            "raw_output_dir": str(raw_dir),
            "compact_output_dir": str(compact_dir),
            "source_row_count": len(source_rows),
            "allowed_case_count": len(manifest),
            "excluded_frozen_group_count": len(frozen_groups),
            "alpha_reference_verified_case_count": sum(bool(row["alpha_reference_verified"]) for row in manifest),
            "smoke": smoke_summary,
            "full": full_summary,
            "alpha_auxiliary": alpha_summary,
            "frozen_test_status": "sealed",
            "test_evaluation_count": 0,
        },
    )
    return 0 if final_status == "completed" else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Phase 5 RandomSpanLight shared candidate bank")
    sub = root.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--dataset-jsonl", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--compact-dir", required=True)
    run_parser.add_argument("--smoke-cases", type=int, default=12)
    run_parser.add_argument("--auto-full", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "run":
        return run(args)
    raise RuntimeError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
