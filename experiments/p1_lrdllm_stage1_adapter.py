#!/usr/bin/env python3
"""Stage-I-only LR-DLLM length selector followed by one fixed-canvas LLaDA decode."""

from __future__ import annotations

import argparse
import gzip
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

from experiments.p1_official_cal_adapter import load_official_runtime
from experiments.p1_official_cal_source_audit import (
    EXPECTED_CAL_COMMIT,
    EXPECTED_HUMANEVAL_COMMIT,
    OFFICIAL_FIXED32_CONFIG,
    git_head,
    read_jsonl,
    sha256,
)
from expvision_dllm_clean.lrdllm_stage1 import (
    IMPLEMENTATION_LABEL,
    PAPER_ID,
    PAPER_VERSION,
    resolve_mask_token_id,
    select_length_lrdllm_stage1,
    tokenize_prefix_suffix,
)
from expvision_dllm_clean.modeling import set_global_seed


ARM = "local_lrdllm_stage1_fixed_decode"
DECODE_LABEL = "Stage-I-only selector + fixed-canvas decode"
MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
MAX_PROBE_LENGTH = 128
EXPECTED_PROBE_FORWARDS = 8
EXPECTED_MASK_TOKEN_ID = 126336
SMOKE_MANIFEST_SHA256 = "56559f3f83ba1ce84c9e03622c5ced6e2ed8d2fa08145a1a64dc0cae0885caa1"
FULL_MANIFEST_SHA256 = "52ef81385984a362fee8729c52cbfa7480265582cabb8250d3ffc27b0aa59af0"
FULL_COUNT = 838
FULL_CLUSTER_COUNT = 143
FORBIDDEN_MANIFEST_FIELDS = {
    "canonical_solution",
    "completion",
    "evaluator_outcome",
    "evaluator_result",
    "oracle_length",
    "passed",
    "reference_code",
    "test",
    "tests",
}


class ForwardLedgerModel:
    """Transparent proxy recording every actual model call and token-forward cost."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.forward_input_sequence_lengths: list[int] = []
        self.forward_token_counts: list[int] = []

    @property
    def device(self) -> Any:
        return self._model.device

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        input_ids = args[0] if args else kwargs["input_ids"]
        self.forward_input_sequence_lengths.append(int(input_ids.shape[-1]))
        self.forward_token_counts.append(int(input_ids.numel()))
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def candidate_key(source_row_id: int) -> str:
    return f"cal_singleline_rest_source_row={int(source_row_id)}|arm={ARM}"


def expected_keys(manifest: Sequence[Mapping[str, Any]]) -> set[str]:
    return {candidate_key(int(row["source_row_id"])) for row in manifest}


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_success_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path) if path.exists() else []
    if any(row.get("status") != "ok" for row in rows):
        raise RuntimeError("LR-DLLM Stage-I canonical raw must contain status=ok rows only")
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    if any(not key or count != 1 for key, count in counts.items()):
        raise RuntimeError("LR-DLLM Stage-I resume refuses blank or duplicate candidate keys")
    return rows


def fixed_decode_kwargs(selected_length: int) -> dict[str, Any]:
    length = int(selected_length)
    if length <= 0 or length > MAX_PROBE_LENGTH:
        raise ValueError("selected length is outside the frozen fixed-canvas domain")
    return {
        "steps": None,
        "gen_length": length,
        "block_length": None,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "span": 1,
        "max_gen_length": MAX_PROBE_LENGTH,
        "dstep": -1,
        "use_bias": False,
    }


def fixed32_equivalence_contract() -> dict[str, Any]:
    local = fixed_decode_kwargs(32)
    expected = {
        "steps": OFFICIAL_FIXED32_CONFIG["steps"],
        "gen_length": OFFICIAL_FIXED32_CONFIG["initial_gen_length"],
        "block_length": OFFICIAL_FIXED32_CONFIG["block_length"],
        "temperature": OFFICIAL_FIXED32_CONFIG["temperature"],
        "cfg_scale": OFFICIAL_FIXED32_CONFIG["cfg_scale"],
        "span": OFFICIAL_FIXED32_CONFIG["span"],
        "max_gen_length": OFFICIAL_FIXED32_CONFIG["max_gen_length"],
        "dstep": OFFICIAL_FIXED32_CONFIG["dstep"],
        "use_bias": OFFICIAL_FIXED32_CONFIG["use_bias"],
    }
    if local != expected:
        raise RuntimeError("fixed_decode_protocol_ambiguity: L=32 arguments differ from official_fixed32")
    return {
        "passed": True,
        "official_cal_commit": EXPECTED_CAL_COMMIT,
        "explicit_call_kwargs": local,
        "upstream_none_resolution": "steps=L and block_length=L inside pinned llada_cal.generate",
    }


def run_fixed_canvas_decode(
    *,
    generate: Any,
    model: Any,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    suffix_attention_mask: torch.Tensor,
    selected_length: int,
) -> tuple[torch.Tensor, int]:
    return generate(
        model,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        attention_mask=attention_mask,
        suffix_attention_mask=suffix_attention_mask,
        **fixed_decode_kwargs(selected_length),
    )


def _validate_manifest_fields(rows: Sequence[Mapping[str, Any]], *, name: str) -> None:
    if not rows:
        raise RuntimeError(f"{name} manifest is empty")
    forbidden = sorted(
        {field for row in rows for field in FORBIDDEN_MANIFEST_FIELDS if field in row}
    )
    if forbidden:
        raise RuntimeError(f"{name} manifest contains forbidden fields: {forbidden}")
    for field in ("candidate_key", "source_row_id", "task_id"):
        raw_values = [row.get(field) for row in rows]
        if any(value is None for value in raw_values):
            raise RuntimeError(f"{name} manifest has blank or duplicate {field}")
        values = [str(value) for value in raw_values]
        if any(value == "" for value in values) or len(values) != len(set(values)):
            raise RuntimeError(f"{name} manifest has blank or duplicate {field}")
    if any(not str(row.get("task_group") or "") for row in rows):
        raise RuntimeError(f"{name} manifest has blank task_group")
    if any(str(row.get("evaluator_commit")) != EXPECTED_HUMANEVAL_COMMIT for row in rows):
        raise RuntimeError(f"{name} manifest evaluator revision mismatch")


def preflight_manifests(
    *,
    manifest_path: Path,
    full_manifest_path: Path,
    current_dataset: Path,
    auto_full: bool,
    smoke_cases: int,
) -> dict[str, Any]:
    manifest = read_jsonl(manifest_path)
    full_manifest = read_jsonl(full_manifest_path)
    _validate_manifest_fields(manifest, name="selected")
    _validate_manifest_fields(full_manifest, name="full")
    selected_hash = sha256(manifest_path)
    full_hash = sha256(full_manifest_path)
    if full_hash != FULL_MANIFEST_SHA256:
        raise RuntimeError("full manifest SHA256 mismatch")
    if len(full_manifest) != FULL_COUNT:
        raise RuntimeError("full manifest row count mismatch")
    if len({str(row["task_group"]) for row in full_manifest}) != FULL_CLUSTER_COUNT:
        raise RuntimeError("full manifest cluster count mismatch")
    full_identities = {
        (int(row["source_row_id"]), str(row["task_id"]), str(row["candidate_key"]))
        for row in full_manifest
    }
    selected_identities = {
        (int(row["source_row_id"]), str(row["task_id"]), str(row["candidate_key"]))
        for row in manifest
    }
    if not selected_identities <= full_identities:
        raise RuntimeError("selected manifest is not an exact subset of the immutable full manifest")
    if auto_full:
        if selected_hash != FULL_MANIFEST_SHA256 or len(manifest) != FULL_COUNT:
            raise RuntimeError("full execution requires the exact immutable 838-row manifest")
        mode = "full"
    else:
        if selected_hash != SMOKE_MANIFEST_SHA256 or len(manifest) != int(smoke_cases):
            raise RuntimeError("technical smoke requires the exact immutable smoke12 manifest")
        mode = "smoke"

    summary_path = full_manifest_path.parent / "cal_singleline_rest_nonfrozen_manifest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frozen = summary.get("frozen_controller_test") or {}
    if not (
        summary.get("sealed_files_opened") is False
        and frozen.get("status") == "sealed"
        and int(frozen.get("test_evaluation_count", -1)) == 0
        and summary.get("full_manifest_sha256") == FULL_MANIFEST_SHA256
        and summary.get("smoke_manifest_sha256") == SMOKE_MANIFEST_SHA256
        and int(summary.get("cal_rest_intersection_nonfrozen_rows", -1)) == FULL_COUNT
        and int(summary.get("cal_rest_intersection_nonfrozen_clusters", -1)) == FULL_CLUSTER_COUNT
    ):
        raise RuntimeError("non-frozen population certificate or sealed/0 status mismatch")
    dataset_hash = sha256(current_dataset)
    if dataset_hash != str(summary.get("official_singleline_sha256")):
        raise RuntimeError("current SingleLine dataset SHA256 mismatch")
    if git_head(Path(str(current_dataset)).parents[1]) != EXPECTED_HUMANEVAL_COMMIT:
        raise RuntimeError("current SingleLine dataset is not inside the pinned evaluator checkout")

    return {
        "passed": True,
        "mode": mode,
        "manifest": manifest,
        "manifest_sha256": selected_hash,
        "manifest_row_count": len(manifest),
        "full_manifest_sha256": full_hash,
        "full_manifest_row_count": len(full_manifest),
        "full_manifest_cluster_count": FULL_CLUSTER_COUNT,
        "current_dataset_sha256": dataset_hash,
        "population_certificate": str(summary_path),
        "frozen_test_status": "sealed",
        "frozen_test_evaluation_count": 0,
        "sealed_files_opened": False,
        "dataset_access_contract": "opaque SHA256 plus JSON parsing only for manifest-selected source rows",
        "fixed32_equivalence": fixed32_equivalence_contract(),
    }


def load_selected_dataset_rows(path: Path, source_row_ids: set[int]) -> dict[int, dict[str, Any]]:
    """Parse only selected non-frozen rows; all other gzip lines stay opaque."""
    if not source_row_ids:
        return {}
    selected: dict[int, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in source_row_ids:
                selected[index] = json.loads(line)
                if len(selected) == len(source_row_ids):
                    break
    missing = sorted(source_row_ids - set(selected))
    if missing:
        raise RuntimeError(f"selected dataset source rows are missing: {missing[:5]}")
    return selected


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
        "arm": ARM,
        "implementation_label": IMPLEMENTATION_LABEL,
        "decode_label": DECODE_LABEL,
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


def failure_count(path: Path) -> int:
    return len(read_jsonl(path)) if path.exists() else 0


def decode_one(
    *,
    source: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    generate: Any,
    check_correctness: Any,
    seed: int,
) -> dict[str, Any]:
    set_global_seed(seed)
    ledger = ForwardLedgerModel(model)
    safe_task = {"prompt": str(source["prompt"]), "suffix": str(source["suffix"])}
    prepared = tokenize_prefix_suffix(safe_task, tokenizer, ledger.device)
    if resolve_mask_token_id(tokenizer) != EXPECTED_MASK_TOKEN_ID:
        raise RuntimeError("tokenizer mask id differs from pinned llada_cal.generate default")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    case_started = time.perf_counter()

    probe_started = time.perf_counter()
    stage1 = select_length_lrdllm_stage1(
        safe_task,
        tokenizer,
        ledger,
        {"max_probe_length": MAX_PROBE_LENGTH},
        prepared_inputs=prepared,
    )
    probe_sec = time.perf_counter() - probe_started
    probe_forwards = len(ledger.forward_token_counts)
    probe_token_forwards = sum(ledger.forward_token_counts)
    if not (
        probe_forwards == EXPECTED_PROBE_FORWARDS
        and probe_forwards == int(stage1["probe_forward_count"])
        and probe_token_forwards == int(stage1["probe_token_forwards"])
    ):
        raise RuntimeError("Stage I probe forward/token ledger mismatch")

    decode_started = time.perf_counter()
    output, upstream_search_forwards = run_fixed_canvas_decode(
        generate=generate,
        model=ledger,
        prefix_ids=prepared["prefix_ids"],
        suffix_ids=prepared["suffix_ids"],
        attention_mask=prepared["attention_mask"],
        suffix_attention_mask=prepared["suffix_attention_mask"],
        selected_length=int(stage1["selected_length"]),
    )
    decode_sec = time.perf_counter() - decode_started
    if int(upstream_search_forwards) != 0:
        raise RuntimeError("fixed decoder unexpectedly performed CAL length search")
    formal_decode_forwards = len(ledger.forward_token_counts) - probe_forwards
    formal_decode_token_forwards = sum(ledger.forward_token_counts[probe_forwards:])
    total_forwards = len(ledger.forward_token_counts)
    total_token_forwards = sum(ledger.forward_token_counts)
    if total_forwards != probe_forwards + formal_decode_forwards:
        raise RuntimeError("forward ledger does not conserve")
    if total_token_forwards != probe_token_forwards + formal_decode_token_forwards:
        raise RuntimeError("token-forward ledger does not conserve")

    full_ids = output[0]
    prefix_length = int(prepared["prefix_ids"].shape[1])
    suffix_length = int(prepared["suffix_ids"].shape[1])
    middle_ids = full_ids[prefix_length : len(full_ids) - suffix_length] if suffix_length else full_ids[prefix_length:]
    if len(middle_ids) != int(stage1["selected_length"]):
        raise RuntimeError("formal fixed canvas changed the Stage I selected length")
    completion = tokenizer.decode(middle_ids, skip_special_tokens=True)

    evaluator_started = time.perf_counter()
    evaluation = check_correctness(dict(source), completion, 3.0, 0)
    evaluator_sec = time.perf_counter() - evaluator_started
    return {
        "completion": completion,
        "passed": bool(evaluation["passed"]),
        "evaluator_result": evaluation["result"],
        "stage1": stage1,
        "metrics": {
            "selected_length": int(stage1["selected_length"]),
            "generated_length": len(middle_ids),
            "probe_forwards": probe_forwards,
            "search_forwards": probe_forwards,
            "formal_decode_forwards": formal_decode_forwards,
            "total_forwards": total_forwards,
            "probe_token_forwards": probe_token_forwards,
            "formal_decode_token_forwards": formal_decode_token_forwards,
            "total_token_forwards": total_token_forwards,
            "token_budget": total_token_forwards,
            "probe_sec": probe_sec,
            "decode_sec": decode_sec,
            "evaluator_sec": evaluator_sec,
            "wall_sec": time.perf_counter() - case_started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "fixed_decoder_upstream_search_forwards": int(upstream_search_forwards),
            "finite_check_passed": bool(stage1["finite_check_passed"]),
        },
        "fixed_decode_config": fixed_decode_kwargs(int(stage1["selected_length"])),
    }


def audit_rows(rows: Sequence[Mapping[str, Any]], expected: set[str]) -> dict[str, Any]:
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    errors = [row for row in rows if row.get("status") != "ok"]
    wrong_arm = [row for row in rows if row.get("arm") != ARM]
    probe_bad = [
        row for row in rows if int((row.get("metrics") or {}).get("probe_forwards", -1)) != EXPECTED_PROBE_FORWARDS
    ]
    finite_bad = [row for row in rows if (row.get("stage1") or {}).get("finite_check_passed") is not True]
    forward_bad = [
        row
        for row in rows
        if int((row.get("metrics") or {}).get("total_forwards", -1))
        != int((row.get("metrics") or {}).get("probe_forwards", 0))
        + int((row.get("metrics") or {}).get("formal_decode_forwards", 0))
    ]
    token_bad = [
        row
        for row in rows
        if int((row.get("metrics") or {}).get("total_token_forwards", -1))
        != int((row.get("metrics") or {}).get("probe_token_forwards", 0))
        + int((row.get("metrics") or {}).get("formal_decode_token_forwards", 0))
    ]
    lengths = Counter(int((row.get("stage1") or {}).get("selected_length", -1)) for row in rows)
    passed_count = sum(row.get("passed") is True for row in rows)
    audit_passed = (
        observed == expected
        and all(value == 1 for value in counts.values())
        and not errors
        and not wrong_arm
        and not probe_bad
        and not finite_bad
        and not forward_bad
        and not token_bad
    )
    return {
        "passed": audit_passed,
        "expected_count": len(expected),
        "observed_count": len(rows),
        "unique_count": len(observed),
        "missing_count": len(expected - observed),
        "extra_count": len(observed - expected),
        "duplicate_count": sum(value > 1 for value in counts.values()),
        "canonical_error_count": len(errors),
        "wrong_arm_count": len(wrong_arm),
        "probe_forward_error_count": len(probe_bad),
        "forward_accounting_error_count": len(forward_bad),
        "token_accounting_error_count": len(token_bad),
        "finite_diagnostic_error_count": len(finite_bad),
        "evaluator_completed_count": len(rows),
        "passed_count": passed_count,
        "technical_smoke_accuracy": passed_count / len(rows) if rows else None,
        "selected_length_distribution": {str(length): count for length, count in sorted(lengths.items())},
        "probe_forwards_total": sum(int((row.get("metrics") or {}).get("probe_forwards", 0)) for row in rows),
        "formal_decode_forwards_total": sum(int((row.get("metrics") or {}).get("formal_decode_forwards", 0)) for row in rows),
        "total_forwards": sum(int((row.get("metrics") or {}).get("total_forwards", 0)) for row in rows),
        "probe_token_forwards_total": sum(int((row.get("metrics") or {}).get("probe_token_forwards", 0)) for row in rows),
        "formal_decode_token_forwards_total": sum(int((row.get("metrics") or {}).get("formal_decode_token_forwards", 0)) for row in rows),
        "total_token_forwards": sum(int((row.get("metrics") or {}).get("total_token_forwards", 0)) for row in rows),
        "canonical_raw_success_only": not errors,
        "frozen_test_status": "sealed",
        "frozen_test_evaluation_count": 0,
    }


def write_analysis_summary(
    output_dir: Path,
    *,
    audit: Mapping[str, Any],
    preflight: Mapping[str, Any],
    mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": (
            "technical_smoke_complete"
            if mode == "smoke" and audit.get("passed")
            else "full_complete"
            if mode == "full" and audit.get("passed")
            else "audit_failed"
        ),
        "implementation_label": IMPLEMENTATION_LABEL,
        "decode_label": DECODE_LABEL,
        "paper_id": PAPER_ID,
        "paper_version": PAPER_VERSION,
        "mode": mode,
        "audit": dict(audit),
        "preflight": {key: value for key, value in preflight.items() if key != "manifest"},
        "scientific_boundary": "technical smoke only; Stage I selector plus fixed-canvas decode; no Stage II",
    }
    atomic_write_json(output_dir / "summary.json", payload)
    lines = [
        "# LR-DLLM Stage I-only Technical Smoke",
        "",
        f"- implementation：`{IMPLEMENTATION_LABEL}`",
        f"- decode：`{DECODE_LABEL}`",
        f"- status：`{payload['status']}`",
        f"- exact rows：`{audit.get('observed_count')}/{audit.get('expected_count')}`",
        f"- missing/duplicate/error/failure：`{audit.get('missing_count')}/{audit.get('duplicate_count')}/{audit.get('canonical_error_count')}/{audit.get('failure_journal_count', 0)}`",
        f"- selected lengths：`{json.dumps(audit.get('selected_length_distribution', {}), sort_keys=True)}`",
        f"- probe/formal/total forwards：`{audit.get('probe_forwards_total')}/{audit.get('formal_decode_forwards_total')}/{audit.get('total_forwards')}`",
        f"- probe/formal/total token-forwards：`{audit.get('probe_token_forwards_total')}/{audit.get('formal_decode_token_forwards_total')}/{audit.get('total_token_forwards')}`",
        "- boundary：technical smoke only；不代表 official/full LR-DLLM；Stage II 未实现。",
    ]
    (output_dir / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finalize(
    *,
    output_dir: Path,
    analysis_output_dir: Path | None,
    raw_path: Path,
    failure_path: Path,
    progress_path: Path,
    run_manifest_path: Path,
    expected: set[str],
    starting_completed_count: int,
    run_started: float,
    mode: str,
    preflight: Mapping[str, Any],
    evaluator_provenance: Mapping[str, Any] | None,
) -> int:
    rows = canonical_success_rows(raw_path)
    relevant = [row for row in rows if str(row["candidate_key"]) in expected]
    audit = audit_rows(relevant, expected)
    audit["failure_journal_count"] = failure_count(failure_path)
    final_progress = progress_payload(
        mode=mode,
        expected_count=len(expected),
        completed_count=len(relevant),
        starting_completed_count=starting_completed_count,
        failure_journal_count=audit["failure_journal_count"],
        started=run_started,
    )
    final_progress["new_rows_written"] = len(relevant) - starting_completed_count
    final_progress["resume_noop"] = final_progress["new_rows_written"] == 0 and len(relevant) == len(expected)
    final_progress["status"] = "completed" if audit["passed"] and audit["failure_journal_count"] == 0 else "audit_failed"
    atomic_write_json(output_dir / f"{ARM}_{mode}_final_audit.json", audit)
    atomic_write_json(progress_path, final_progress)
    atomic_write_json(
        run_manifest_path,
        {
            **final_progress,
            "final_audit": audit,
            "preflight": {key: value for key, value in preflight.items() if key != "manifest"},
            "evaluator_provenance": dict(evaluator_provenance or {}),
        },
    )
    if analysis_output_dir is not None:
        write_analysis_summary(analysis_output_dir, audit=audit, preflight=preflight, mode=mode)
    return 0 if audit["passed"] and audit["failure_journal_count"] == 0 else 2


def run(args: argparse.Namespace) -> int:
    official_cal_root = Path(args.official_cal_root).resolve()
    humaneval_root = Path(args.humaneval_root).resolve()
    current_dataset = Path(args.current_dataset).resolve()
    manifest_path = Path(args.manifest_jsonl).resolve()
    full_manifest_path = Path(args.full_manifest_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    analysis_output_dir = Path(args.analysis_output_dir).resolve() if args.analysis_output_dir else None
    if args.arm != ARM:
        raise RuntimeError("the Stage-I-only adapter has exactly one local arm")
    if int(args.max_probe_length) != MAX_PROBE_LENGTH:
        raise RuntimeError("primary protocol requires max_probe_length=128")
    if int(args.progress_every) <= 0:
        raise ValueError("--progress-every must be positive")
    if git_head(official_cal_root) != EXPECTED_CAL_COMMIT:
        raise RuntimeError("official CAL checkout is not pinned")
    if git_head(humaneval_root) != EXPECTED_HUMANEVAL_COMMIT:
        raise RuntimeError("HumanEval evaluator checkout is not pinned")
    preflight = preflight_manifests(
        manifest_path=manifest_path,
        full_manifest_path=full_manifest_path,
        current_dataset=current_dataset,
        auto_full=bool(args.auto_full),
        smoke_cases=int(args.smoke_cases),
    )
    if args.preflight_only:
        print(json.dumps({key: value for key, value in preflight.items() if key != "manifest"}, sort_keys=True))
        return 0

    manifest = list(preflight["manifest"])
    mode = str(preflight["mode"])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{ARM}_raw.jsonl"
    failure_path = output_dir / f"{ARM}_failure_journal.jsonl"
    progress_path = output_dir / f"{ARM}_{mode}_progress.json"
    run_manifest_path = output_dir / f"{ARM}_{mode}_run_manifest.json"
    existing = canonical_success_rows(raw_path)
    all_known = expected_keys(read_jsonl(full_manifest_path))
    existing_keys = {str(row["candidate_key"]) for row in existing}
    if not existing_keys <= all_known:
        raise RuntimeError("canonical raw contains keys outside the immutable non-frozen population")
    expected = expected_keys(manifest)
    completed = existing_keys & expected
    starting_completed_count = len(completed)
    run_started = time.perf_counter()
    initial_progress = progress_payload(
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
            "resume_contract": "skip only existing status=ok candidate keys; canonical raw is append-only",
            "official_cal_commit": EXPECTED_CAL_COMMIT,
            "evaluator_commit": EXPECTED_HUMANEVAL_COMMIT,
            "model": MODEL_PATH,
            "model_revision": MODEL_REVISION,
            "seed": int(args.seed),
            "preflight": {key: value for key, value in preflight.items() if key != "manifest"},
        },
    )
    if completed == expected:
        return _finalize(
            output_dir=output_dir,
            analysis_output_dir=analysis_output_dir,
            raw_path=raw_path,
            failure_path=failure_path,
            progress_path=progress_path,
            run_manifest_path=run_manifest_path,
            expected=expected,
            starting_completed_count=starting_completed_count,
            run_started=run_started,
            mode=mode,
            preflight=preflight,
            evaluator_provenance=None,
        )

    try:
        generate, check_correctness, evaluator_provenance = load_official_runtime(
            official_cal_root, humaneval_root
        )
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, revision=MODEL_REVISION, trust_remote_code=True
        )
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda").eval()
    except Exception as exc:
        failed = {
            **initial_progress,
            "status": "preflight_failed",
            "preflight_error_type": type(exc).__name__,
            "preflight_error_message": str(exc)[:240],
        }
        atomic_write_json(progress_path, failed)
        atomic_write_json(run_manifest_path, failed)
        raise

    source_ids = {int(row["source_row_id"]) for row in manifest if candidate_key(int(row["source_row_id"])) not in completed}
    selected_sources = load_selected_dataset_rows(current_dataset, source_ids)
    manifest_by_source = {int(row["source_row_id"]): row for row in manifest}
    for source_row_id in sorted(source_ids, key=lambda value: int(manifest_by_source[value].get("smoke_order", value))):
        item = manifest_by_source[source_row_id]
        key = candidate_key(source_row_id)
        source = selected_sources[source_row_id]
        if str(source.get("task_id")) != str(item["task_id"]):
            raise RuntimeError("selected source row task_id does not match immutable manifest")
        case_started = time.perf_counter()
        try:
            result = decode_one(
                source=source,
                tokenizer=tokenizer,
                model=model,
                generate=generate,
                check_correctness=check_correctness,
                seed=int(args.seed),
            )
            row = {
                "candidate_key": key,
                "arm": ARM,
                "implementation_label": IMPLEMENTATION_LABEL,
                "decode_label": DECODE_LABEL,
                "paper_id": PAPER_ID,
                "paper_version": PAPER_VERSION,
                "stage": "stage1_only",
                "source_row_id": source_row_id,
                "task_id": str(source["task_id"]),
                "task_group": str(item["task_group"]),
                "benchmark_name": "single-line",
                "population": "CAL-Rest_intersection_project_non-frozen",
                "status": "ok",
                "passed": result["passed"],
                "completion": result["completion"],
                "evaluator_result": result["evaluator_result"],
                "stage1": result["stage1"],
                "metrics": result["metrics"],
                "config": {
                    "max_probe_length": MAX_PROBE_LENGTH,
                    "fit_scope": result["stage1"]["fit_scope"],
                    "fixed_decode": result["fixed_decode_config"],
                    "seed": int(args.seed),
                },
                "official_cal_decoder_commit": EXPECTED_CAL_COMMIT,
                "evaluator_commit": EXPECTED_HUMANEVAL_COMMIT,
                "evaluator_provenance": evaluator_provenance,
                "model": MODEL_PATH,
                "model_revision": MODEL_REVISION,
                "gpu": torch.cuda.get_device_name(),
            }
        except Exception as exc:
            failure = {
                "candidate_key": key,
                "arm": ARM,
                "implementation_label": IMPLEMENTATION_LABEL,
                "source_row_id": source_row_id,
                "task_id": str(source.get("task_id") or ""),
                "task_group": str(item.get("task_group") or ""),
                "benchmark_name": "single-line",
                "status": "error",
                "passed": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:240],
                "failure_traceback": traceback.format_exc(),
                "metrics": {"wall_sec": time.perf_counter() - case_started},
            }
            append_jsonl(failure_path, failure)
            failed_progress = progress_payload(
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
            raise RuntimeError(f"LR-DLLM Stage-I fail-stop at {key}; inspect failure journal") from exc
        append_jsonl(raw_path, row)
        completed.add(key)
        newly_completed = len(completed) - starting_completed_count
        if newly_completed % int(args.progress_every) == 0 or len(completed) == len(expected):
            atomic_write_json(
                progress_path,
                progress_payload(
                    mode=mode,
                    expected_count=len(expected),
                    completed_count=len(completed),
                    starting_completed_count=starting_completed_count,
                    failure_journal_count=failure_count(failure_path),
                    started=run_started,
                ),
            )
    return _finalize(
        output_dir=output_dir,
        analysis_output_dir=analysis_output_dir,
        raw_path=raw_path,
        failure_path=failure_path,
        progress_path=progress_path,
        run_manifest_path=run_manifest_path,
        expected=expected,
        starting_completed_count=starting_completed_count,
        run_started=run_started,
        mode=mode,
        preflight=preflight,
        evaluator_provenance=evaluator_provenance,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--official-cal-root", required=True)
    result.add_argument("--humaneval-root", required=True)
    result.add_argument("--current-dataset", required=True)
    result.add_argument("--manifest-jsonl", required=True)
    result.add_argument("--full-manifest-jsonl", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--analysis-output-dir")
    result.add_argument("--arm", choices=(ARM,), default=ARM)
    result.add_argument("--max-probe-length", type=int, default=MAX_PROBE_LENGTH)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--smoke-cases", type=int, default=12)
    result.add_argument("--progress-every", type=int, default=1)
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--auto-full", action="store_true")
    return result


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
