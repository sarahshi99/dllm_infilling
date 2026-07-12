#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from expvision_dllm_clean.modeling import load_model_and_tokenizer
from expvision_dllm_clean.config import ExperimentConfig
from experiments.phase5_randomspanlight_candidate_bank import (
    code_task,
    compact_candidate_row,
    protocol_validation,
    run_candidate,
)


MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
SOURCE_CONFIG = "HumanEval-MultiLineInfilling"
CANVAS_LENGTHS = (16, 32, 64, 128)
SEEDS = (0, 1)
TOTAL_STEPS = 64
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
    lock = json.loads(TEST_LOCK.read_text(encoding="utf-8"))
    if lock.get("test_status") != "sealed" or int(lock.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError("Frozen controller test lock is not sealed with test_evaluation_count=0")
    groups = {str(item) for item in lock.get("test_task_ids", [])}
    groups.update(str(item) for item in json.loads(GROUPED_TEST_TASKS.read_text(encoding="utf-8")))
    return groups, lock


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


def build_candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_kind": "deployable_grid",
            "control_label": "fixed64_control" if (canvas, seed) == (64, 0) else "non_oracle_candidate",
            "deployable": True,
            "canvas_tokens": canvas,
            "seed": seed,
            "total_steps": TOTAL_STEPS,
        }
        for canvas in CANVAS_LENGTHS
        for seed in SEEDS
    ]


def normalize_candidate_row(row: Mapping[str, Any], task: Any) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("prefix_text", str(task.prefix))
    normalized.setdefault("middle_text", "")
    normalized.setdefault("suffix_text", str(task.suffix))
    normalized.setdefault("candidate_middle_tokens", 0)
    normalized.setdefault("metrics", {})
    normalized.setdefault("verification", {})
    normalized.setdefault("diagnostics", {})
    return normalized


def candidate_key(row_key: str, canvas_tokens: int, seed: int) -> str:
    return f"{row_key}|deployable_grid|canvas={int(canvas_tokens)}|seed={int(seed)}"


def expected_candidate_keys(manifest: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        candidate_key(str(row["row_key"]), int(spec["canvas_tokens"]), int(spec["seed"]))
        for row in manifest
        for spec in build_candidate_specs()
    }


def build_manifest_without_tokenizer(rows: Sequence[Mapping[str, Any]], frozen_groups: set[str]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for source_row_id, row in enumerate(rows):
        tid = str(row["task_id"])
        group = task_group(tid)
        if group in frozen_groups:
            continue
        manifest.append(
            {
                "case_index": len(manifest),
                "source_row_id": source_row_id,
                "row_key": f"{SOURCE_CONFIG}:{source_row_id}:{sha256_text(tid)[:16]}",
                "source_config": SOURCE_CONFIG,
                "task_id": tid,
                "task_group": group,
                "frozen_controller_test_row": False,
            }
        )
    return manifest


def build_manifest(rows: Sequence[Mapping[str, Any]], tokenizer: Any, frozen_groups: set[str]) -> list[dict[str, Any]]:
    manifest = build_manifest_without_tokenizer(rows, frozen_groups)
    for item in manifest:
        source = rows[int(item["source_row_id"])]
        reference_tokens = len(tokenizer.encode(str(source.get("canonical_solution", "")), add_special_tokens=False))
        if reference_tokens <= 0:
            raise RuntimeError(f"Empty reference middle for {item['task_id']}")
        item.update(
            {
                "reference_middle_tokens": reference_tokens,
                "length_bucket": length_bucket(reference_tokens),
                "prefix_tokens": len(tokenizer.encode(str(source["prompt"]), add_special_tokens=False)),
                "suffix_tokens": len(tokenizer.encode(str(source["suffix"]), add_special_tokens=False)),
            }
        )
    return manifest


def choose_smoke_manifest(manifest: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(manifest):
        raise ValueError(f"Invalid smoke count {count}")
    selected: list[dict[str, Any]] = []
    buckets = ["short", "medium", "long", "extreme"]
    per_bucket, remainder = divmod(count, len(buckets))
    for index, bucket in enumerate(buckets):
        rows = sorted(
            (dict(row) for row in manifest if row["length_bucket"] == bucket),
            key=lambda row: (int(row["reference_middle_tokens"]), int(row["case_index"])),
        )
        selected.extend(rows[: per_bucket + (1 if index < remainder else 0)])
    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} smoke rows, expected {count}")
    return sorted(selected, key=lambda row: int(row["case_index"]))


def run_population(
    manifest: Sequence[Mapping[str, Any]],
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
    specs = build_candidate_specs()
    for item in manifest:
        source = source_rows[int(item["source_row_id"])]
        task = code_task(source)
        for spec in specs:
            key = candidate_key(str(item["row_key"]), int(spec["canvas_tokens"]), int(spec["seed"]))
            if key in completed:
                continue
            row = normalize_candidate_row(run_candidate(
                manifest_row=item,
                source_row=source,
                task=task,
                spec=spec,
                tokenizer=tokenizer,
                model=model,
            ), task)
            append_jsonl(raw_path, row)
            completed.add(key)
            written += 1
            print(json.dumps({"candidate_key": key, "status": row["status"], "written": written}, sort_keys=True), flush=True)
    return written


def audit_rows(rows: Sequence[Mapping[str, Any]], expected: set[str]) -> dict[str, Any]:
    counts = Counter(str(row.get("candidate_key", "")) for row in rows)
    observed = set(counts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    errors = sorted(str(row.get("candidate_key", "")) for row in rows if row.get("status") != "ok")
    return {
        "integrity_passed": not missing and not extra and not duplicates,
        "expected_count": len(expected),
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
        "frozen_controller_test_row": False,
    }


def create_compact_artifacts(
    *,
    compact_dir: Path,
    full_manifest: Sequence[Mapping[str, Any]],
    selected_manifest: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    phase_name: str,
    raw_dir: Path,
    dataset_jsonl: Path,
    wall_sec: float,
    peak_memory_bytes: int,
    total_memory_bytes: int,
    test_lock: Mapping[str, Any],
    resume_noop_writes: int,
) -> dict[str, Any]:
    compact_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = {str(row["row_key"]) for row in selected_manifest}
    rows = [row for row in raw_rows if str(row.get("row_key", "")) in selected_keys and row.get("candidate_kind") == "deployable_grid"]
    expected = expected_candidate_keys(selected_manifest)
    audit = audit_rows(rows, expected)
    schema_required = {"candidate_key", "row_key", "candidate_kind", "canvas_tokens", "seed", "status", "passed", "verification"}
    schema_passed = all(schema_required <= set(row) for row in rows)
    evaluator_passed = all(bool(row.get("verification")) for row in rows if row.get("status") == "ok")
    frozen_ok = (
        test_lock.get("test_status") == "sealed"
        and int(test_lock.get("test_evaluation_count", -1)) == 0
        and not any(bool(row.get("frozen_controller_test_row")) for row in selected_manifest)
    )
    resume_passed = int(resume_noop_writes) == 0 and set(str(row.get("candidate_key")) for row in rows) == expected
    gpu_memory_healthy = peak_memory_bytes > 0 and total_memory_bytes > peak_memory_bytes
    technical_gate_passed = bool(
        audit["integrity_passed"]
        and schema_passed
        and evaluator_passed
        and protocol.get("supported")
        and frozen_ok
        and resume_passed
        and gpu_memory_healthy
        and any(row.get("status") == "ok" for row in rows)
    )
    compact_rows = [compact_candidate_row(row) for row in rows]
    write_csv(compact_dir / f"{phase_name}_candidate_results.csv", compact_rows)
    write_csv(compact_dir / "manifest.csv", [compact_manifest_row(row) for row in full_manifest])
    audit_payload = {
        "phase": phase_name,
        "technical_gate_passed": technical_gate_passed,
        "performance_gate_used": False,
        "schema_passed": schema_passed,
        "evaluator_executed_for_all_ok_rows": evaluator_passed,
        "frozen_test_invariant_passed": frozen_ok,
        "resume_passed": resume_passed,
        "resume_noop_writes": resume_noop_writes,
        "gpu_memory_healthy": gpu_memory_healthy,
        "peak_memory_bytes": peak_memory_bytes,
        "total_memory_bytes": total_memory_bytes,
        "rows": audit,
    }
    write_json(compact_dir / f"{phase_name}_audit.json", audit_payload)
    latencies = [float((row.get("metrics") or {}).get("total_sec_including_probe") or row.get("wall_sec") or 0.0) for row in rows]
    decode_secs = [float((row.get("metrics") or {}).get("decode_sec") or 0.0) for row in rows]
    verification_secs = [float((row.get("metrics") or {}).get("verification_sec") or 0.0) for row in rows]
    summary = {
        "phase": phase_name,
        "verdict": f"{phase_name}_technical_passed" if technical_gate_passed else f"{phase_name}_technical_failed",
        "technical_gate_passed": technical_gate_passed,
        "performance_gate_used": False,
        "selected_case_count": len(selected_manifest),
        "full_manifest_case_count": len(full_manifest),
        "candidate_rows": len(rows),
        "expected_candidate_rows": len(expected),
        "ok_candidate_rows": sum(row.get("status") == "ok" for row in rows),
        "error_candidate_rows": sum(row.get("status") != "ok" for row in rows),
        "pass_count": sum(bool(row.get("passed")) for row in rows),
        "denoising_steps_per_candidate": TOTAL_STEPS,
        "wall_sec": wall_sec,
        "summed_candidate_latency_sec": sum(latencies),
        "summed_gpu_decode_sec": sum(decode_secs),
        "summed_verification_sec": sum(verification_secs),
        "mean_candidate_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
        "peak_cuda_memory_bytes": peak_memory_bytes,
        "total_cuda_memory_bytes": total_memory_bytes,
        "raw_output_dir_local_only": str(raw_dir),
        "dataset_jsonl_local_read_only": str(dataset_jsonl),
        "raw_code_committed": False,
        "frozen_test_status": test_lock.get("test_status"),
        "test_evaluation_count": test_lock.get("test_evaluation_count"),
        "audit": audit_payload,
    }
    write_json(compact_dir / f"{phase_name}_summary.json", summary)
    return summary


def render_report(smoke: Mapping[str, Any], full: Mapping[str, Any] | None) -> str:
    lines = [
        "# Phase 6 MultiLine Shared Candidate Bank",
        "",
        f"Smoke verdict: `{smoke['verdict']}`.",
        f"Smoke cases/candidates/errors: `{smoke['selected_case_count']}` / `{smoke['candidate_rows']}` / `{smoke['error_candidate_rows']}`.",
        "Smoke is technical only; no performance threshold is used.",
        "Raw generated code remains local; compact artifacts contain hashes and metrics only.",
        "Frozen test remains sealed with `test_evaluation_count=0`.",
    ]
    if full is not None:
        lines.extend(
            [
                "",
                f"Full verdict: `{full['verdict']}`.",
                f"Full cases/candidates/errors: `{full['selected_case_count']}` / `{full['candidate_rows']}` / `{full['error_candidate_rows']}`.",
                f"Mean candidate latency: `{full['mean_candidate_latency_sec']:.4f}` seconds.",
                f"Summed GPU decode time: `{full['summed_gpu_decode_sec']:.4f}` seconds.",
                f"Summed verification time: `{full['summed_verification_sec']:.4f}` seconds.",
                f"Peak CUDA memory: `{full['peak_cuda_memory_bytes']}` bytes.",
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
    started = time.perf_counter()
    frozen_groups, test_lock = load_frozen_groups()
    source_rows = read_jsonl(dataset_jsonl)
    if len(source_rows) != 5815:
        raise RuntimeError(f"Expected 5815 MultiLine source rows, found {len(source_rows)}")

    tokenizer, model = load_model_and_tokenizer(cfg_for(64, 0).model)
    manifest = build_manifest(source_rows, tokenizer, frozen_groups)
    if len(manifest) != 5079:
        raise RuntimeError(f"Expected 5079 allowed rows, found {len(manifest)}")
    if any(row["task_group"] in frozen_groups for row in manifest):
        raise RuntimeError("Frozen row survived manifest exclusion")
    smoke_manifest = choose_smoke_manifest(manifest, int(args.smoke_cases))
    protocol = protocol_validation([code_task(source_rows[int(row["source_row_id"])]) for row in manifest], tokenizer, model)
    if not protocol["supported"]:
        write_json(compact_dir / "blocker_protocol.json", {"verdict": "blocked_protocol", **protocol})
        raise RuntimeError("Canvas 128 protocol validation failed")
    write_json(compact_dir / "protocol_validation.json", protocol)

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for Phase 6 GPU experiment")
        torch.cuda.reset_peak_memory_stats()
        total_memory = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    except Exception:
        raise

    run_population(smoke_manifest, source_rows, raw_path, tokenizer, model)
    smoke_noop = run_population(smoke_manifest, source_rows, raw_path, tokenizer, model)
    peak = int(torch.cuda.max_memory_allocated())
    smoke_summary = create_compact_artifacts(
        compact_dir=compact_dir,
        full_manifest=manifest,
        selected_manifest=smoke_manifest,
        raw_rows=read_jsonl(raw_path),
        protocol=protocol,
        phase_name="smoke",
        raw_dir=raw_dir,
        dataset_jsonl=dataset_jsonl,
        wall_sec=time.perf_counter() - started,
        peak_memory_bytes=peak,
        total_memory_bytes=total_memory,
        test_lock=test_lock,
        resume_noop_writes=smoke_noop,
    )
    if not smoke_summary["technical_gate_passed"]:
        (compact_dir / "report.md").write_text(render_report(smoke_summary, None), encoding="utf-8")
        write_json(
            compact_dir / "run_manifest.json",
            {
                "status": "smoke_technical_failed",
                "started_at_utc": started_at,
                "ended_at_utc": utc_now(),
                "command": sys.argv,
                "smoke": smoke_summary,
                "frozen_test_status": "sealed",
                "test_evaluation_count": 0,
            },
        )
        return 2

    full_summary: dict[str, Any] | None = None
    if args.auto_full:
        run_population(manifest, source_rows, raw_path, tokenizer, model)
        full_noop = run_population(manifest, source_rows, raw_path, tokenizer, model)
        peak = int(torch.cuda.max_memory_allocated())
        full_summary = create_compact_artifacts(
            compact_dir=compact_dir,
            full_manifest=manifest,
            selected_manifest=manifest,
            raw_rows=read_jsonl(raw_path),
            protocol=protocol,
            phase_name="full",
            raw_dir=raw_dir,
            dataset_jsonl=dataset_jsonl,
            wall_sec=time.perf_counter() - started,
            peak_memory_bytes=peak,
            total_memory_bytes=total_memory,
            test_lock=test_lock,
            resume_noop_writes=full_noop,
        )

    final_status = "completed" if full_summary is None or full_summary["technical_gate_passed"] else "full_integrity_failed"
    (compact_dir / "report.md").write_text(render_report(smoke_summary, full_summary), encoding="utf-8")
    write_json(
        compact_dir / "run_manifest.json",
        {
            "status": final_status,
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
            "wall_sec": time.perf_counter() - started,
            "command": sys.argv,
            "model": MODEL_PATH,
            "dataset_jsonl": str(dataset_jsonl),
            "source_row_count": len(source_rows),
            "allowed_case_count": len(manifest),
            "excluded_frozen_group_count": len(frozen_groups),
            "expected_full_candidate_rows": len(manifest) * 8,
            "smoke": smoke_summary,
            "full": full_summary,
            "frozen_test_status": "sealed",
            "test_evaluation_count": 0,
        },
    )
    return 0 if final_status == "completed" else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Phase 6 MultiLine shared candidate bank")
    sub = root.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--dataset-jsonl", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--compact-dir", required=True)
    run_parser.add_argument("--smoke-cases", type=int, default=8)
    run_parser.add_argument("--auto-full", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "run":
        return run(args)
    raise RuntimeError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
