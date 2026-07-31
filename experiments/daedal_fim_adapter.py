#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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

from experiments.p1_official_cal_adapter import (
    ForwardLedgerModel,
    _enabled_pinned_evaluator,
    append_jsonl,
    atomic_write_json,
    certify_population,
    evaluator_problem_map,
    gpu_snapshot,
    indexed_rows,
    progress_payload,
)
from experiments.p1_official_cal_source_audit import EXPECTED_CAL_COMMIT, EXPECTED_HUMANEVAL_COMMIT, git_head, read_jsonl, sha256
from expvision_dllm_clean.modeling import set_global_seed


MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
MODEL_CONFIG_SHA256 = "5f99fefe855fdb5100bb6cadb57bdb09fae723ad54811f95c00ecacf29d58a6a"
MODEL_INDEX_SHA256 = "28b4ec27206e42e7ade630450e6ce618bd197acf34d35120e8e86d2bb910a408"
DAEDAL_SOURCE_SHA256 = "9c89bc1fc31b35af3da107cce63e8caa945fe1652291b53fb274d8ec8d34372a"
DAEDAL_EVAL_SHA256 = "338d0409432f00fc16d497bb78099695e07644058c27edb2722da0d878bd79e9"
ARMS = ("daedal_dynamic", "daedal_fixed8_control")
BENCHMARKS = {
    "singleline": {"adapter_name": "single-line", "dataset_file": "HumanEval-SingleLineInfilling.jsonl.gz", "full_count": 838},
    "multiline": {"adapter_name": "multi-line", "dataset_file": "HumanEval-MultiLineInfilling.jsonl.gz", "full_count": 4990},
}


def arm_config(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown DAEDAL arm {arm!r}")
    return {
        "enable_stage1": False,
        "enable_stage2": arm == "daedal_dynamic",
        "initial_gen_length": 8,
        "max_gen_length": 128,
        "block_length": 32,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "high_conf_threshold": 0.90,
        "low_conf_threshold": 0.30,
        "expansion_factor": 2,
        "mask_id": 126336,
        "eos_token_id": 126081,
        "eos_confidence_threshold": 0.5,
        "expand_eos_confidence_threshold": 0.9,
        "eos_check_tokens": 8,
    }


def candidate_key(benchmark: str, source_row_id: int) -> str:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    return f"daedal_fim|dataset={benchmark}|source_row={int(source_row_id)}"


def load_pinned_runtime(cal_root: Path, evaluator_root: Path) -> tuple[Any, Any, dict[str, Any]]:
    if git_head(cal_root) != EXPECTED_CAL_COMMIT or git_head(evaluator_root) != EXPECTED_HUMANEVAL_COMMIT:
        raise RuntimeError("DAEDAL source or evaluator checkout is not pinned")
    algorithm_path = cal_root / "daedal_cal" / "daedal_cal.py"
    eval_path = cal_root / "daedal_cal" / "daedal_code_infilling.py"
    if sha256(algorithm_path) != DAEDAL_SOURCE_SHA256 or sha256(eval_path) != DAEDAL_EVAL_SHA256:
        raise RuntimeError("pinned DAEDAL source hash mismatch")
    if (cal_root / "LICENSE").exists() or (cal_root / "LICENSE.txt").exists():
        license_status = "license_file_present_unexpected_reaudit_required"
    else:
        license_status = "no LICENSE file; internal audit/run only"
    spec = importlib.util.spec_from_file_location("pinned_cal_authors_daedal", algorithm_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned DAEDAL algorithm")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generate = getattr(module, "generate", None)
    if not callable(generate):
        raise RuntimeError("pinned DAEDAL algorithm lacks generate")
    check_correctness, evaluator_info = _enabled_pinned_evaluator(evaluator_root)
    return generate, check_correctness, {
        **evaluator_info,
        "cal_revision": EXPECTED_CAL_COMMIT,
        "daedal_source_sha256": DAEDAL_SOURCE_SHA256,
        "daedal_eval_sha256": DAEDAL_EVAL_SHA256,
        "license": license_status,
        "adapter_boundary": "direct generate(prefix_ids=..., suffix_ids=...); upstream evaluation script uses incompatible prefix/suffix keyword names",
    }


def model_provenance(snapshot: Path) -> dict[str, Any]:
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError("LLaDA snapshot revision mismatch")
    if sha256(snapshot / "config.json") != MODEL_CONFIG_SHA256 or sha256(snapshot / "model.safetensors.index.json") != MODEL_INDEX_SHA256:
        raise RuntimeError("LLaDA cached checkpoint hash mismatch")
    return {
        "model_id": "GSAI-ML/LLaDA-8B-Base",
        "model_revision": MODEL_REVISION,
        "model_snapshot": str(snapshot),
        "config_sha256": MODEL_CONFIG_SHA256,
        "model_index_sha256": MODEL_INDEX_SHA256,
    }


def canonical_rows(path: Path, known_keys: set[str], arm: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path) if path.exists() else []
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    if any(row.get("status") != "ok" for row in rows) or any(not key or count > 1 for key, count in counts.items()):
        raise RuntimeError("DAEDAL canonical raw must be unique success-only rows")
    if not set(counts) <= known_keys or any(str(row.get("arm")) != arm for row in rows):
        raise RuntimeError("DAEDAL canonical raw contains unknown keys or wrong arm")
    return rows


def expected_keys(manifest: Sequence[Mapping[str, Any]], benchmark: str) -> set[str]:
    keys = {candidate_key(benchmark, int(row["source_row_id"])) for row in manifest}
    if len(keys) != len(manifest):
        raise RuntimeError("DAEDAL manifest source ids are duplicated")
    return keys


def extract_middle(output: torch.Tensor, prefix_length: int, suffix_length: int) -> torch.Tensor:
    return output[prefix_length : len(output) - suffix_length] if suffix_length else output[prefix_length:]


def decode_one(
    *,
    source: Mapping[str, Any],
    arm: str,
    tokenizer: Any,
    model: Any,
    generate: Any,
    check_correctness: Any,
    evaluator_problem: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    config = arm_config(arm)
    set_global_seed(seed)
    ledger = ForwardLedgerModel(model)
    prefix = str(source["prompt"])
    suffix = str(source["suffix"])
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    torch.cuda.reset_peak_memory_stats()
    decode_started = time.perf_counter()
    outputs, stage1_max, stage2_max = generate(
        ledger,
        tokenizer,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        **config,
    )
    decode_sec = time.perf_counter() - decode_started
    output = outputs[0]
    middle_ids = extract_middle(output, int(prefix_ids.shape[1]), int(suffix_ids.shape[1]))
    completion = tokenizer.decode(middle_ids, skip_special_tokens=True)
    remaining_masks = int((middle_ids == int(config["mask_id"])).sum().item())
    final_length = int(middle_ids.shape[0])
    evaluator_started = time.perf_counter()
    evaluation = check_correctness(evaluator_problem, completion, 3.0, 0)
    evaluator_sec = time.perf_counter() - evaluator_started
    total_forwards = len(ledger.forward_input_tokens)
    stage2_max_value = bool(stage2_max[0].item())
    termination = "stalled_with_remaining_masks" if remaining_masks else "max_length_reached" if stage2_max_value else "all_middle_positions_filled"
    return {
        "completion": completion,
        "passed": bool(evaluation["passed"]),
        "evaluator_result": evaluation["result"],
        "initial_gen_length": 8,
        "final_gen_length": final_length,
        "net_expansion_tokens": max(final_length - 8, 0),
        "net_contraction_tokens": 0,
        "inferred_expansion_events": max(final_length - 8, 0),
        "remaining_mask_tokens": remaining_masks,
        "stage1_reached_max": bool(stage1_max[0].item()),
        "stage2_reached_max": stage2_max_value,
        "termination_reason": termination,
        "search_forwards": 0,
        "formal_decode_forwards": total_forwards,
        "total_forwards": total_forwards,
        "search_token_forwards": 0,
        "formal_decode_token_forwards": sum(ledger.forward_input_tokens),
        "token_forwards": sum(ledger.forward_input_tokens),
        "decode_sec": decode_sec,
        "evaluator_sec": evaluator_sec,
        "wall_sec": decode_sec + evaluator_sec,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "seed": int(seed),
    }


def audit_rows(rows: Sequence[Mapping[str, Any]], expected: set[str], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("candidate_key")) in expected]
    counts = Counter(str(row.get("candidate_key") or "") for row in selected)
    observed = set(counts)
    accounting = [
        row for row in selected
        if int((row.get("metrics") or {}).get("total_forwards", -1))
        != int((row.get("metrics") or {}).get("search_forwards", 0)) + int((row.get("metrics") or {}).get("formal_decode_forwards", 0))
    ]
    stalled = [row for row in selected if int((row.get("metrics") or {}).get("remaining_mask_tokens", 0)) > 0]
    return {
        "passed": observed == expected and all(count == 1 for count in counts.values()) and not accounting and not stalled,
        "arm": arm,
        "expected_count": len(expected),
        "observed_count": len(selected),
        "missing_count": len(expected - observed),
        "extra_count": len(observed - expected),
        "duplicate_count": sum(count > 1 for count in counts.values()),
        "error_count": 0,
        "forward_accounting_error_count": len(accounting),
        "stalled_with_masks_count": len(stalled),
    }


def run(args: argparse.Namespace) -> int:
    benchmark = str(args.benchmark)
    contract = BENCHMARKS[benchmark]
    manifest_path = Path(args.manifest_jsonl).resolve()
    full_manifest_path = Path(args.full_manifest_jsonl).resolve()
    cal_root = Path(args.cal_root).resolve()
    evaluator_root = Path(args.evaluator_root).resolve()
    model_snapshot = Path(args.model_snapshot).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = read_jsonl(manifest_path)
    full_manifest = read_jsonl(full_manifest_path)
    expected_count = int(contract["full_count"]) if args.mode == "full" else 12
    if len(manifest) != expected_count or int(args.progress_every) <= 0:
        raise RuntimeError("DAEDAL manifest count or progress interval mismatch")
    population = certify_population(
        benchmark_name=str(contract["adapter_name"]),
        current_dataset=evaluator_root / "data" / str(contract["dataset_file"]),
        manifest_path=manifest_path,
        full_manifest_path=full_manifest_path,
    )
    model_info = model_provenance(model_snapshot)
    generate, check_correctness, runtime_info = load_pinned_runtime(cal_root, evaluator_root)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, trust_remote_code=True, local_files_only=True)
    tokenizer_info = {
        "bos_token_id": 126080,
        "eos_token_id": 126081,
        "mask_token_id": 126336,
        "mask_token": tokenizer.convert_ids_to_tokens(126336),
    }
    if tokenizer.eos_token_id != 126081 or tokenizer.convert_ids_to_tokens(126336) != "<|mdm_mask|>":
        raise RuntimeError("LLaDA tokenizer special ids do not match pinned DAEDAL defaults")
    if args.preflight_only:
        print(json.dumps({"population": population, "model": model_info, "runtime": runtime_info, "tokenizer": tokenizer_info}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    arm = str(args.arm)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{arm}_raw.jsonl"
    failure_path = output_dir / f"{arm}_failure_journal.jsonl"
    progress_path = output_dir / f"{arm}_{args.mode}_progress.json"
    run_manifest_path = output_dir / f"{arm}_{args.mode}_run_manifest.json"
    all_known = expected_keys(full_manifest, benchmark)
    expected = expected_keys(manifest, benchmark)
    existing = canonical_rows(raw_path, all_known, arm)
    completed = {str(row["candidate_key"]) for row in existing} & expected
    starting_completed = len(completed)
    started = time.perf_counter()
    failure_count = len(read_jsonl(failure_path)) if failure_path.exists() else 0
    initial = progress_payload(
        arm=arm,
        mode=str(args.mode),
        expected_count=len(expected),
        completed_count=len(completed),
        starting_completed_count=starting_completed,
        failure_journal_count=failure_count,
        started=started,
    )
    atomic_write_json(progress_path, initial)
    atomic_write_json(
        run_manifest_path,
        {
            **initial,
            **population,
            **model_info,
            **runtime_info,
            "benchmark": benchmark,
            "manifest": str(manifest_path),
            "full_manifest": str(full_manifest_path),
            "config": arm_config(arm),
            "label": "CAL authors’ DAEDAL FIM adaptation" if arm == "daedal_dynamic" else "Fixed8 control for CAL authors’ DAEDAL FIM adaptation",
            "frozen_test_status": "sealed",
            "test_evaluation_count": 0,
        },
    )
    if len(completed) == len(expected):
        audit = audit_rows(existing, expected, arm)
        final = {**initial, "status": "completed", "new_rows_written": 0, "final_audit": audit}
        atomic_write_json(progress_path, final)
        atomic_write_json(run_manifest_path, {**json.loads(run_manifest_path.read_text()), **final})
        return 0 if audit["passed"] and failure_count == 0 else 2
    model = AutoModel.from_pretrained(model_snapshot, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    source_rows = indexed_rows(evaluator_root / "data" / str(contract["dataset_file"]))
    evaluator_problems = evaluator_problem_map(evaluator_root, str(contract["adapter_name"]))
    manifest_by_key = {candidate_key(benchmark, int(item["source_row_id"])): item for item in manifest}
    for key in sorted(expected - completed, key=lambda value: int(manifest_by_key[value]["source_row_id"])):
        item = manifest_by_key[key]
        source_row_id = int(item["source_row_id"])
        source = source_rows[source_row_id]
        try:
            decoded = decode_one(
                source=source,
                arm=arm,
                tokenizer=tokenizer,
                model=model,
                generate=generate,
                check_correctness=check_correctness,
                evaluator_problem=evaluator_problems[str(source["task_id"])],
                seed=int(args.seed),
            )
            append_jsonl(
                raw_path,
                {
                    "candidate_key": key,
                    "arm": arm,
                    "benchmark": benchmark,
                    "source_row_id": source_row_id,
                    "task_id": str(source["task_id"]),
                    "task_group": str(item["task_group"]),
                    "status": "ok",
                    "passed": decoded.pop("passed"),
                    "evaluator_result": decoded.pop("evaluator_result"),
                    "completion": decoded.pop("completion"),
                    "metrics": decoded,
                },
            )
            completed.add(key)
        except Exception as exc:
            append_jsonl(
                failure_path,
                {
                    "candidate_key": key,
                    "arm": arm,
                    "benchmark": benchmark,
                    "source_row_id": source_row_id,
                    "task_id": str(source.get("task_id") or ""),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            failure_count += 1
            failed = progress_payload(
                arm=arm,
                mode=str(args.mode),
                expected_count=len(expected),
                completed_count=len(completed),
                starting_completed_count=starting_completed,
                failure_journal_count=failure_count,
                started=started,
            )
            failed["status"] = "failed"
            atomic_write_json(progress_path, failed)
            atomic_write_json(run_manifest_path, {**json.loads(run_manifest_path.read_text()), **failed})
            return 2
        if len(completed) % int(args.progress_every) == 0 or len(completed) == len(expected):
            atomic_write_json(
                progress_path,
                progress_payload(
                    arm=arm,
                    mode=str(args.mode),
                    expected_count=len(expected),
                    completed_count=len(completed),
                    starting_completed_count=starting_completed,
                    failure_journal_count=failure_count,
                    started=started,
                ),
            )
    rows = canonical_rows(raw_path, all_known, arm)
    audit = audit_rows(rows, expected, arm)
    final = progress_payload(
        arm=arm,
        mode=str(args.mode),
        expected_count=len(expected),
        completed_count=len(completed),
        starting_completed_count=starting_completed,
        failure_journal_count=failure_count,
        started=started,
    )
    final.update({"status": "completed" if audit["passed"] and failure_count == 0 else "audit_failed", "new_rows_written": len(completed) - starting_completed, "final_audit": audit})
    atomic_write_json(progress_path, final)
    atomic_write_json(run_manifest_path, {**json.loads(run_manifest_path.read_text()), **final})
    return 0 if audit["passed"] and failure_count == 0 else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Adapter for CAL authors’ DAEDAL FIM adaptation and same-decoder Fixed8 control.")
    result.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    result.add_argument("--manifest-jsonl", required=True)
    result.add_argument("--full-manifest-jsonl", required=True)
    result.add_argument("--cal-root", required=True)
    result.add_argument("--evaluator-root", required=True)
    result.add_argument("--model-snapshot", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--arm", choices=ARMS, required=True)
    result.add_argument("--mode", choices=("smoke", "full"), required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--progress-every", type=int, default=1)
    result.add_argument("--preflight-only", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
