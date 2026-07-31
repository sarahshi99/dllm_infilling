#!/usr/bin/env python3
"""Resumable adapter for pinned official LLaDA-CAL on the audited CAL-Rest set."""

from __future__ import annotations

import argparse
import hashlib
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

from experiments.p1_official_cal_source_audit import (
    OFFICIAL_FIXED32_CONFIG,
    PRIMARY_CAL_CONFIG,
    PROJECT_FIXED64_CONFIG,
    EXPECTED_CAL_COMMIT,
    EXPECTED_HUMANEVAL_COMMIT,
    git_head,
    read_jsonl,
    sha256,
)
from expvision_dllm_clean.modeling import set_global_seed


MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
ARMS = ("official_cal_primary", "official_fixed32", "project_fixed64_internal")
BENCHMARKS = {
    "multi-line": {
        "full_count": 4990,
        "cluster_count": 143,
        "dataset_filename": "HumanEval-MultiLineInfilling.jsonl.gz",
        "candidate_prefix": "cal_rest_source_row",
        "default_full_manifest_name": "cal_rest_common_manifest.jsonl",
    },
    "single-line": {
        "full_count": 838,
        "cluster_count": 143,
        "dataset_filename": "HumanEval-SingleLineInfilling.jsonl.gz",
        "candidate_prefix": "cal_singleline_rest_source_row",
        "default_full_manifest_name": None,
    },
}


class ForwardLedgerModel:
    """Transparent model proxy that records exact upstream decoder forwards."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.forward_input_tokens: list[int] = []

    @property
    def device(self) -> Any:
        return self._model.device

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        input_ids = args[0] if args else kwargs["input_ids"]
        self.forward_input_tokens.append(int(input_ids.shape[-1]))
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def arm_config(arm: str) -> dict[str, Any]:
    if arm == "official_cal_primary":
        return {**PRIMARY_CAL_CONFIG, "arm": arm}
    if arm == "official_fixed32":
        return dict(OFFICIAL_FIXED32_CONFIG)
    if arm == "project_fixed64_internal":
        return {
            "arm": arm,
            "initial_gen_length": 64,
            "span": 1,
            "dstep": -1,
            "max_gen_length": 64,
            "use_bias": False,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "oracle": False,
            "steps": 64,
            "block_length": 64,
            "comparison_boundary": PROJECT_FIXED64_CONFIG["arm"],
        }
    raise ValueError(f"unknown CAL arm {arm!r}")


def candidate_key(source_row_id: int, arm: str, benchmark_name: str = "multi-line") -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark_name!r}")
    prefix = BENCHMARKS[benchmark_name]["candidate_prefix"]
    return f"{prefix}={int(source_row_id)}|arm={arm}"


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a compact manifest atomically; raw results remain append-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_success_rows(path: Path) -> list[dict[str, Any]]:
    """Read a canonical raw file that is contractually success-only."""
    rows = read_jsonl(path) if path.exists() else []
    if any(row.get("status") != "ok" for row in rows):
        raise RuntimeError("official CAL canonical raw must contain status=ok rows only")
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    if any(not key or count > 1 for key, count in counts.items()):
        raise RuntimeError("official CAL resume refuses blank or duplicate candidate keys")
    return rows


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "name": torch.cuda.get_device_name(),
        "memory_free_bytes": int(free),
        "memory_total_bytes": int(total),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def progress_payload(
    *,
    arm: str,
    mode: str,
    expected_count: int,
    completed_count: int,
    starting_completed_count: int,
    failure_journal_count: int,
    started: float,
) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - started, 0.0)
    completed_since_start = max(completed_count - starting_completed_count, 0)
    rows_per_hour = (completed_since_start / elapsed * 3600.0) if elapsed and completed_since_start else None
    missing_count = expected_count - completed_count
    eta_seconds = (missing_count / rows_per_hour * 3600.0) if rows_per_hour and missing_count else 0.0 if not missing_count else None
    return {
        "status": "running",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": arm,
        "mode": mode,
        "expected_count": expected_count,
        "completed_count": completed_count,
        "existing_completed_count_at_start": starting_completed_count,
        "missing_count": missing_count,
        "canonical_error_count": 0,
        "failure_journal_count": failure_journal_count,
        "rows_per_hour": rows_per_hour,
        "eta_seconds": eta_seconds,
        "gpu": gpu_snapshot(),
    }


def enable_pinned_evaluator_source(source: str) -> str:
    """Restore exactly the single execution line documented by upstream."""
    marker = "#                     exec(check_program, exec_globals)"
    replacement = "                    exec(check_program, exec_globals)"
    if source.count(marker) != 1:
        raise RuntimeError("pinned evaluator enablement marker is absent or ambiguous")
    return source.replace(marker, replacement)


def _enabled_pinned_evaluator(humaneval_root: Path) -> tuple[Any, dict[str, Any]]:
    """Enable the one evaluator call the pinned upstream README requires users to restore.

    The pinned checkout is never modified.  The README documents that its
    execution call is deliberately commented pending an explicit local safety
    decision; without it the file is syntactically invalid.  This in-memory
    overlay restores exactly that one documented line and records both hashes.
    """
    source_path = humaneval_root / "human_eval_infilling" / "execution.py"
    source = source_path.read_text(encoding="utf-8")
    enabled_source = enable_pinned_evaluator_source(source)
    namespace: dict[str, Any] = {"__name__": "pinned_humaneval_infilling_execution_enabled", "__file__": str(source_path)}
    exec(compile(enabled_source, str(source_path), "exec"), namespace, namespace)
    check_correctness = namespace.get("check_correctness")
    if not callable(check_correctness):
        raise RuntimeError("enabled pinned evaluator did not define check_correctness")
    return check_correctness, {
        "evaluator_execution_mode": "upstream_documented_local_enablement_overlay",
        "evaluator_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "evaluator_enabled_source_sha256": hashlib.sha256(enabled_source.encode("utf-8")).hexdigest(),
        "evaluator_enablement": "restored only README-documented exec(check_program, exec_globals); pinned checkout unchanged",
    }


def load_official_runtime(official_cal_root: Path, humaneval_root: Path) -> tuple[Any, Any, dict[str, Any]]:
    if git_head(official_cal_root) != EXPECTED_CAL_COMMIT:
        raise RuntimeError("official CAL checkout is not pinned")
    if git_head(humaneval_root) != EXPECTED_HUMANEVAL_COMMIT:
        raise RuntimeError("HumanEval evaluator checkout is not pinned")
    for path in (str(official_cal_root), str(humaneval_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from llada_cal.llada_cal import generate  # type: ignore
    check_correctness, evaluator_provenance = _enabled_pinned_evaluator(humaneval_root)
    return generate, check_correctness, evaluator_provenance


def indexed_rows(dataset: Path) -> dict[int, dict[str, Any]]:
    return {index: row for index, row in enumerate(read_jsonl(dataset))}


def evaluator_problem_map(humaneval_root: Path, benchmark_name: str) -> dict[str, dict[str, Any]]:
    import gzip

    path = humaneval_root / "data" / str(BENCHMARKS[benchmark_name]["dataset_filename"])
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {str(row["task_id"]): row for row in (json.loads(line) for line in handle if line.strip())}


def decode_one(
    *,
    row: Mapping[str, Any],
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
    prefix = str(row["prompt"])
    suffix = str(row["suffix"])
    encoded_prefix = tokenizer(prefix, add_special_tokens=False, padding=True, return_tensors="pt")
    encoded_suffix = tokenizer(suffix, add_special_tokens=False, padding=True, return_tensors="pt")
    prefix_ids = encoded_prefix["input_ids"].to(model.device)
    suffix_ids = encoded_suffix["input_ids"].to(model.device)
    prefix_mask = encoded_prefix["attention_mask"].to(model.device)
    suffix_mask = encoded_suffix["attention_mask"].to(model.device)
    torch.cuda.reset_peak_memory_stats()
    decode_started = time.perf_counter()
    output, search_forwards = generate(
        ledger,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        attention_mask=prefix_mask,
        suffix_attention_mask=suffix_mask,
        steps=config["steps"],
        gen_length=config["initial_gen_length"],
        block_length=config["block_length"],
        temperature=config["temperature"],
        cfg_scale=config["cfg_scale"],
        span=config["span"],
        max_gen_length=config["max_gen_length"],
        dstep=config["dstep"],
        use_bias=config["use_bias"],
    )
    decode_sec = time.perf_counter() - decode_started
    full_ids = output[0]
    prefix_len, suffix_len = int(prefix_ids.shape[1]), int(suffix_ids.shape[1])
    middle_ids = full_ids[prefix_len : len(full_ids) - suffix_len] if suffix_len else full_ids[prefix_len:]
    completion = tokenizer.decode(middle_ids, skip_special_tokens=True)
    total_forwards = len(ledger.forward_input_tokens)
    formal_forwards = total_forwards - int(search_forwards)
    if formal_forwards < 0:
        raise RuntimeError("upstream CAL search forward count exceeds recorded forwards")
    evaluator_started = time.perf_counter()
    evaluation = check_correctness(evaluator_problem, completion, 3.0, 0)
    evaluator_sec = time.perf_counter() - evaluator_started
    return {
        "completion": completion,
        "initial_gen_length": int(config["initial_gen_length"]),
        "selected_length": len(middle_ids),
        "net_expansion_tokens": max(len(middle_ids) - int(config["initial_gen_length"]), 0),
        "net_contraction_tokens": max(int(config["initial_gen_length"]) - len(middle_ids), 0),
        "length_direction": "expanded" if len(middle_ids) > int(config["initial_gen_length"]) else "contracted" if len(middle_ids) < int(config["initial_gen_length"]) else "unchanged",
        "termination_reason": "upstream_event_level_reason_not_exposed",
        "search_forwards": int(search_forwards),
        "formal_decode_forwards": formal_forwards,
        "total_forwards": total_forwards,
        "search_token_budget": sum(ledger.forward_input_tokens[: int(search_forwards)]),
        "formal_decode_token_budget": sum(ledger.forward_input_tokens[int(search_forwards) :]),
        "token_budget": sum(ledger.forward_input_tokens),
        "wall_sec": decode_sec + evaluator_sec,
        "decode_sec": decode_sec,
        "evaluator_sec": evaluator_sec,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "passed": bool(evaluation["passed"]),
        "evaluator_result": evaluation["result"],
        "seed": int(seed),
        "determinism": "set_global_seed_before_each_case; official upstream has no seed CLI",
    }


def expected_keys(
    manifest: Sequence[Mapping[str, Any]], arm: str, benchmark_name: str = "multi-line"
) -> set[str]:
    return {candidate_key(int(row["source_row_id"]), arm, benchmark_name) for row in manifest}


def certify_population(
    *,
    benchmark_name: str,
    current_dataset: Path,
    manifest_path: Path,
    full_manifest_path: Path,
) -> dict[str, Any]:
    contract = BENCHMARKS[benchmark_name]
    full_manifest = read_jsonl(full_manifest_path)
    if len(full_manifest) != int(contract["full_count"]):
        raise RuntimeError(f"{benchmark_name} full manifest has the wrong row count")
    if len({str(row["task_group"]) for row in full_manifest}) != int(contract["cluster_count"]):
        raise RuntimeError(f"{benchmark_name} full manifest has the wrong cluster count")
    if benchmark_name == "multi-line":
        source_audit_path = manifest_path.parent / "source_audit.json"
        source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        if source_audit["population"]["project_nonfrozen_multiline_rest_common"] != 4990:
            raise RuntimeError("source audit does not certify corrected MultiLine population")
        return {
            "population_certificate": str(source_audit_path),
            "current_dataset_sha256": sha256(current_dataset),
            "full_manifest_sha256": sha256(full_manifest_path),
        }
    summary_path = full_manifest_path.parent / "cal_singleline_rest_nonfrozen_manifest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not (
        summary["cal_rest_intersection_nonfrozen_rows"] == 838
        and summary["cal_rest_intersection_nonfrozen_clusters"] == 143
        and summary["official_singleline_sha256"] == sha256(current_dataset)
        and summary["full_manifest_sha256"] == sha256(full_manifest_path)
        and summary["sealed_files_opened"] is False
    ):
        raise RuntimeError("SingleLine immutable population certificate mismatch")
    return {
        "population_certificate": str(summary_path),
        "current_dataset_sha256": sha256(current_dataset),
        "full_manifest_sha256": sha256(full_manifest_path),
    }


def audit_rows(rows: Sequence[Mapping[str, Any]], expected: set[str], arm: str) -> dict[str, Any]:
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    errors = [row for row in rows if row.get("status") != "ok"]
    wrong_arm = [row for row in rows if row.get("arm") != arm]
    accounting_bad = [
        row
        for row in rows
        if row.get("status") == "ok"
        and int((row.get("metrics") or {}).get("total_forwards") or -1)
        != int((row.get("metrics") or {}).get("search_forwards") or 0)
        + int((row.get("metrics") or {}).get("formal_decode_forwards") or 0)
    ]
    return {
        "passed": observed == expected and all(value == 1 for value in counts.values()) and not errors and not wrong_arm and not accounting_bad,
        "expected_count": len(expected),
        "observed_count": len(rows),
        "missing_count": len(expected - observed),
        "extra_count": len(observed - expected),
        "duplicate_count": sum(value > 1 for value in counts.values()),
        "error_count": len(errors),
        "wrong_arm_count": len(wrong_arm),
        "forward_accounting_error_count": len(accounting_bad),
    }


def failure_count(path: Path) -> int:
    return len(read_jsonl(path)) if path.exists() else 0


def run(args: argparse.Namespace) -> int:
    official_cal_root = Path(args.official_cal_root).resolve()
    humaneval_root = Path(args.humaneval_root).resolve()
    current_dataset = Path(args.current_dataset).resolve()
    manifest_path = Path(args.manifest_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    arm = str(args.arm)
    benchmark_name = str(args.benchmark_name)
    contract = BENCHMARKS[benchmark_name]
    if args.full_manifest_jsonl:
        full_manifest_path = Path(args.full_manifest_jsonl).resolve()
    else:
        default_name = contract["default_full_manifest_name"]
        if not default_name:
            raise RuntimeError("SingleLine requires --full-manifest-jsonl")
        full_manifest_path = manifest_path.parent / str(default_name)
    if int(args.progress_every) <= 0:
        raise ValueError("--progress-every must be positive")
    manifest = read_jsonl(manifest_path)
    mode = "full" if args.auto_full else "smoke"
    if args.auto_full:
        if len(manifest) != int(contract["full_count"]):
            raise RuntimeError(
                f"official CAL {benchmark_name} auto-full requires the audited {contract['full_count']}-row manifest"
            )
    elif len(manifest) != int(args.smoke_cases):
        raise RuntimeError("smoke manifest size does not match --smoke-cases")
    rows_by_source = indexed_rows(current_dataset)
    population_provenance = certify_population(
        benchmark_name=benchmark_name,
        current_dataset=current_dataset,
        manifest_path=manifest_path,
        full_manifest_path=full_manifest_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{arm}_raw.jsonl"
    failure_path = output_dir / f"{arm}_failure_journal.jsonl"
    progress_path = output_dir / f"{arm}_{mode}_progress.json"
    run_manifest_path = output_dir / f"{arm}_{mode}_run_manifest.json"
    existing = canonical_success_rows(raw_path)
    all_common_manifest = read_jsonl(full_manifest_path)
    all_known = expected_keys(all_common_manifest, arm, benchmark_name)
    existing_keys = {str(row["candidate_key"]) for row in existing}
    if not existing_keys <= all_known:
        raise RuntimeError("official CAL canonical raw contains keys outside the audited common population")
    expected = expected_keys(manifest, arm, benchmark_name)
    completed = existing_keys & expected
    starting_completed_count = len(completed)
    run_started = time.perf_counter()
    initial_progress = progress_payload(
        arm=arm,
        mode=mode,
        expected_count=len(expected),
        completed_count=len(completed),
        starting_completed_count=starting_completed_count,
        failure_journal_count=failure_count(failure_path),
        started=run_started,
    )
    atomic_write_json(progress_path, initial_progress)
    atomic_write_json(
        run_manifest_path,
        {
            **initial_progress,
            "status": "running",
            "raw_path": str(raw_path),
            "failure_journal_path": str(failure_path),
            "resume_contract": "skip only existing status=ok candidate_key values; canonical raw is append-only",
            "official_cal_commit": EXPECTED_CAL_COMMIT,
            "evaluator_commit": EXPECTED_HUMANEVAL_COMMIT,
            "seed": int(args.seed),
            "benchmark_name": benchmark_name,
            **population_provenance,
        },
    )
    try:
        generate, check_correctness, evaluator_provenance = load_official_runtime(official_cal_root, humaneval_root)
    except Exception as exc:
        preflight = {
            **initial_progress,
            "status": "preflight_failed",
            "preflight_error_type": type(exc).__name__,
            "preflight_error_message": str(exc)[:240],
        }
        atomic_write_json(progress_path, preflight)
        atomic_write_json(run_manifest_path, preflight)
        raise
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    evaluator_problems = evaluator_problem_map(humaneval_root, benchmark_name)
    for item in manifest:
        source_row_id = int(item["source_row_id"])
        key = candidate_key(source_row_id, arm, benchmark_name)
        if key in completed:
            continue
        source = rows_by_source[source_row_id]
        case_started = time.perf_counter()
        try:
            result = decode_one(
                row=source,
                arm=arm,
                tokenizer=tokenizer,
                model=model,
                generate=generate,
                check_correctness=check_correctness,
                evaluator_problem=evaluator_problems[str(source["task_id"])],
                seed=int(args.seed),
            )
            row = {
                "candidate_key": key,
                "arm": arm,
                "source_row_id": source_row_id,
                "task_id": str(source["task_id"]),
                "task_group": str(item["task_group"]),
                "benchmark_name": benchmark_name,
                "status": "ok",
                "passed": result.pop("passed"),
                "completion": result.pop("completion"),
                "evaluator_result": result.pop("evaluator_result"),
                "metrics": result,
                "official_cal_commit": EXPECTED_CAL_COMMIT,
                "evaluator_commit": EXPECTED_HUMANEVAL_COMMIT,
                "evaluator_provenance": evaluator_provenance,
                "config": arm_config(arm),
                "gpu": torch.cuda.get_device_name(),
            }
        except Exception as exc:
            failure = {
                "candidate_key": key,
                "arm": arm,
                "source_row_id": source_row_id,
                "task_id": str(source.get("task_id") or ""),
                "task_group": str(item.get("task_group") or ""),
                "benchmark_name": benchmark_name,
                "status": "error",
                "passed": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:240],
                "failure_traceback": traceback.format_exc(),
                "metrics": {"wall_sec": time.perf_counter() - case_started},
            }
            append_jsonl(failure_path, failure)
            failed_progress = progress_payload(
                arm=arm,
                mode=mode,
                expected_count=len(expected),
                completed_count=len(completed),
                starting_completed_count=starting_completed_count,
                failure_journal_count=failure_count(failure_path),
                started=run_started,
            )
            failed_progress["status"] = "failed_stop"
            atomic_write_json(progress_path, failed_progress)
            atomic_write_json(
                run_manifest_path,
                {**failed_progress, "status": "failed_stop", "failure_candidate_key": key},
            )
            raise RuntimeError(f"official CAL fail-stop at {key}; inspect failure journal") from exc
        if row.get("status") != "ok":
            raise RuntimeError("official CAL adapter attempted to append a non-ok canonical row")
        append_jsonl(raw_path, row)
        completed.add(key)
        newly_completed = len(completed) - starting_completed_count
        if newly_completed % int(args.progress_every) == 0 or len(completed) == len(expected):
            atomic_write_json(
                progress_path,
                progress_payload(
                    arm=arm,
                    mode=mode,
                    expected_count=len(expected),
                    completed_count=len(completed),
                    starting_completed_count=starting_completed_count,
                    failure_journal_count=failure_count(failure_path),
                    started=run_started,
                ),
            )
    rows = canonical_success_rows(raw_path)
    relevant_rows = [row for row in rows if str(row["candidate_key"]) in expected]
    audit = audit_rows(relevant_rows, expected, arm)
    audit["failure_journal_count"] = failure_count(failure_path)
    audit["canonical_raw_success_only"] = True
    audit["mode"] = mode
    atomic_write_json(output_dir / f"{arm}_{mode}_final_audit.json", audit)
    final_progress = progress_payload(
        arm=arm,
        mode=mode,
        expected_count=len(expected),
        completed_count=len(completed),
        starting_completed_count=starting_completed_count,
        failure_journal_count=failure_count(failure_path),
        started=run_started,
    )
    final_progress["status"] = "completed" if audit["passed"] and audit["failure_journal_count"] == 0 else "audit_failed"
    final_progress["new_rows_written"] = len(completed) - starting_completed_count
    atomic_write_json(progress_path, final_progress)
    atomic_write_json(
        run_manifest_path,
        {
            **final_progress,
            "benchmark_name": benchmark_name,
            "final_audit": audit,
            "evaluator_provenance": evaluator_provenance,
            **population_provenance,
        },
    )
    return 0 if audit["passed"] and audit["failure_journal_count"] == 0 else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--official-cal-root", required=True)
    root.add_argument("--humaneval-root", required=True)
    root.add_argument("--current-dataset", required=True)
    root.add_argument("--benchmark-name", choices=tuple(BENCHMARKS), default="multi-line")
    root.add_argument("--manifest-jsonl", required=True)
    root.add_argument("--full-manifest-jsonl")
    root.add_argument("--output-dir", required=True)
    root.add_argument("--arm", choices=ARMS, required=True)
    root.add_argument("--seed", type=int, default=42)
    root.add_argument("--smoke-cases", type=int, default=12)
    root.add_argument("--progress-every", type=int, default=25)
    root.add_argument("--auto-full", action="store_true")
    return root


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
