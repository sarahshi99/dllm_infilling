#!/usr/bin/env python3
"""Stage-I-only LR-DLLM length selector followed by one fixed-canvas LLaDA decode."""

from __future__ import annotations

import argparse
import gzip
import json
import math
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
    candidate_key as official_cal_candidate_key,
    load_official_runtime,
)
from experiments.p1_official_cal_source_audit import (
    EXPECTED_CAL_COMMIT,
    EXPECTED_HUMANEVAL_COMMIT,
    OFFICIAL_FIXED32_CONFIG,
    git_head,
    read_jsonl,
    sha256,
)
from experiments.p1_lrdllm_stage1_audit import (
    EXPECTED_MASK_TOKEN_ID,
    MODEL_PATH,
    MODEL_REVISION,
    FullGateError,
    audit_evaluator_source,
    audit_fixed_decoder_source,
    audit_model_artifacts,
    current_code_commit,
    execution_gpu_ecc_gate,
    provenance_contract,
    require_no_failure_journal,
    runtime_environment,
    tracked_worktree_is_clean,
    validate_full_gate,
    validate_probe_only_gate,
)
from expvision_dllm_clean.lrdllm_stage1 import (
    FIT_SCOPE,
    IMPLEMENTATION_LABEL,
    PAPER_ID,
    PAPER_VERSION,
    STAGE,
    select_length_lrdllm_stage1,
    tokenize_prefix_suffix,
)
from expvision_dllm_clean.modeling import resolve_mask_token_id, set_global_seed


ARM = "local_lrdllm_stage1_fixed_decode"
DECODE_LABEL = "Stage-I-only selector + fixed-canvas decode"
MAX_PROBE_LENGTH = 128
EXPECTED_PROBE_FORWARDS = 8
EXPECTED_PROBE_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
SMOKE_MANIFEST_SHA256 = "56559f3f83ba1ce84c9e03622c5ced6e2ed8d2fa08145a1a64dc0cae0885caa1"
FULL_MANIFEST_SHA256 = "52ef81385984a362fee8729c52cbfa7480265582cabb8250d3ffc27b0aa59af0"
FULL_COUNT = 838
FULL_CLUSTER_COUNT = 143
MANIFEST_ALLOWED_FIELDS = {
    "candidate_key",
    "dataset",
    "evaluator_commit",
    "population",
    "seed",
    "smoke_order",
    "source_row_id",
    "task_group",
    "task_id",
    "technical_stratum",
}
MANIFEST_REQUIRED_FIELDS = {
    "candidate_key",
    "dataset",
    "evaluator_commit",
    "population",
    "seed",
    "source_row_id",
    "task_group",
    "task_id",
}
SEARCH_FORWARDS_ALIAS_SEMANTICS = (
    "compatibility alias for Stage-I length selection; not CAL hill-climbing search"
)


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


def candidate_key(source_row_id: int, mode: str = "smoke") -> str:
    base = f"cal_singleline_rest_source_row={int(source_row_id)}|arm={ARM}"
    return f"{base}|mode=probe_only" if mode == "probe_only" else base


def expected_keys(manifest: Sequence[Mapping[str, Any]], mode: str = "smoke") -> set[str]:
    return {candidate_key(int(row["source_row_id"]), mode) for row in manifest}


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
    unexpected = sorted({field for row in rows for field in row if field not in MANIFEST_ALLOWED_FIELDS})
    if unexpected:
        raise RuntimeError(f"{name} manifest contains fields outside the exact allowlist: {unexpected}")
    missing = sorted({field for row in rows for field in MANIFEST_REQUIRED_FIELDS if field not in row})
    if missing:
        raise RuntimeError(f"{name} manifest is missing required fields: {missing}")
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
    mode: str,
    smoke_cases: int,
    official_cal_root: Path,
    humaneval_root: Path,
    hf_home: Path,
    repo: Path = REPO,
) -> dict[str, Any]:
    if mode not in {"probe_only", "smoke", "full"}:
        raise ValueError(f"unsupported execution mode {mode!r}")
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
    source_manifest_row_count = len(manifest)
    if mode == "full":
        if selected_hash != FULL_MANIFEST_SHA256 or len(manifest) != FULL_COUNT:
            raise RuntimeError("full execution requires the exact immutable 838-row manifest")
    else:
        if selected_hash != SMOKE_MANIFEST_SHA256 or len(manifest) != int(smoke_cases):
            raise RuntimeError("probe-only/smoke requires the exact immutable smoke12 manifest")
        if mode == "probe_only":
            ordered = sorted(manifest, key=lambda row: int(row["smoke_order"]))
            if not ordered or int(ordered[0]["smoke_order"]) != 0:
                raise RuntimeError("probe-only requires smoke_order=0 from the frozen smoke manifest")
            manifest = ordered[:1]

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

    fixed_decoder_source_audit = audit_fixed_decoder_source(
        official_cal_root / "llada_cal" / "llada_cal.py"
    )
    model_provenance = audit_model_artifacts(hf_home=hf_home)
    evaluator_source_audit = audit_evaluator_source(humaneval_root)
    implementation_commit = current_code_commit(repo)

    return {
        "passed": True,
        "mode": mode,
        "manifest": manifest,
        "manifest_sha256": selected_hash,
        "manifest_row_count": len(manifest),
        "source_manifest_row_count": source_manifest_row_count,
        "smoke_manifest_sha256": SMOKE_MANIFEST_SHA256,
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
        "fixed_decoder_source_audit": fixed_decoder_source_audit,
        "model_provenance": model_provenance,
        "evaluator_source_audit": evaluator_source_audit,
        "runtime_environment": runtime_environment(),
        "implementation_commit": implementation_commit,
        "implementation_tracked_worktree_clean": tracked_worktree_is_clean(repo),
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
            "stage1_probe_forwards": probe_forwards,
            "length_selection_forwards": probe_forwards,
            "probe_forwards": probe_forwards,
            "search_forwards": probe_forwards,
            "search_forwards_semantics": SEARCH_FORWARDS_ALIAS_SEMANTICS,
            "formal_decode_forwards": formal_decode_forwards,
            "total_forwards": total_forwards,
            "stage1_probe_token_forwards": probe_token_forwards,
            "length_selection_token_forwards": probe_token_forwards,
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


def probe_only_one(
    *,
    source: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    seed: int,
) -> dict[str, Any]:
    """Run exactly the eight Stage-I probes; never decode or call an evaluator."""
    set_global_seed(seed)
    ledger = ForwardLedgerModel(model)
    safe_task = {"prompt": str(source["prompt"]), "suffix": str(source["suffix"])}
    prepared = tokenize_prefix_suffix(safe_task, tokenizer, ledger.device)
    if resolve_mask_token_id(tokenizer) != EXPECTED_MASK_TOKEN_ID:
        raise RuntimeError("tokenizer mask id differs from pinned llada_cal.generate default")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stage1 = select_length_lrdllm_stage1(
        safe_task,
        tokenizer,
        ledger,
        {"max_probe_length": MAX_PROBE_LENGTH},
        prepared_inputs=prepared,
    )
    wall_sec = time.perf_counter() - started
    stage1_probe_forwards = len(ledger.forward_token_counts)
    stage1_probe_token_forwards = sum(ledger.forward_token_counts)
    if not (
        stage1_probe_forwards == EXPECTED_PROBE_FORWARDS
        and stage1_probe_forwards == int(stage1["probe_forward_count"])
        and stage1_probe_token_forwards == int(stage1["probe_token_forwards"])
    ):
        raise RuntimeError("probe-only Stage I forward/token ledger mismatch")
    return {
        "stage1": stage1,
        "metrics": {
            "selected_length": int(stage1["selected_length"]),
            "stage1_probe_forwards": stage1_probe_forwards,
            "length_selection_forwards": stage1_probe_forwards,
            "probe_forwards": stage1_probe_forwards,
            "search_forwards": stage1_probe_forwards,
            "search_forwards_semantics": SEARCH_FORWARDS_ALIAS_SEMANTICS,
            "formal_decode_forwards": 0,
            "total_forwards": stage1_probe_forwards,
            "stage1_probe_token_forwards": stage1_probe_token_forwards,
            "length_selection_token_forwards": stage1_probe_token_forwards,
            "probe_token_forwards": stage1_probe_token_forwards,
            "formal_decode_token_forwards": 0,
            "total_token_forwards": stage1_probe_token_forwards,
            "token_budget": stage1_probe_token_forwards,
            "wall_sec": wall_sec,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0,
            "finite_check_passed": bool(stage1["finite_check_passed"]),
            "evaluator_called": False,
        },
    }


def _stage1_finite(stage1: Mapping[str, Any]) -> bool:
    values: list[float] = []
    for field in (
        "average_negative_entropy_by_length",
        "log_length_by_length",
        "cl_score_by_length",
    ):
        mapping = stage1.get(field)
        if not isinstance(mapping, Mapping):
            return False
        try:
            values.extend(float(value) for value in mapping.values())
        except (TypeError, ValueError):
            return False
    try:
        values.extend(
            float(stage1[field]) for field in ("k_hat", "intercept", "fit_r2")
        )
    except (KeyError, TypeError, ValueError):
        return False
    return stage1.get("finite_check_passed") is True and all(
        math.isfinite(value) for value in values
    )


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    expected: set[str],
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    errors = [row for row in rows if row.get("status") != "ok"]
    wrong_arm = [row for row in rows if row.get("arm") != ARM]
    probe_bad = [
        row
        for row in rows
        if not (
            int((row.get("metrics") or {}).get("stage1_probe_forwards", -1))
            == EXPECTED_PROBE_FORWARDS
            == int((row.get("metrics") or {}).get("length_selection_forwards", -1))
            == int((row.get("metrics") or {}).get("probe_forwards", -1))
            == int((row.get("metrics") or {}).get("search_forwards", -1))
            == int((row.get("stage1") or {}).get("probe_forward_count", -1))
        )
    ]
    finite_bad = [row for row in rows if not _stage1_finite(row.get("stage1") or {})]
    forward_bad = [
        row
        for row in rows
        if int((row.get("metrics") or {}).get("total_forwards", -1))
        != int((row.get("metrics") or {}).get("stage1_probe_forwards", 0))
        + int((row.get("metrics") or {}).get("formal_decode_forwards", 0))
    ]
    token_bad = [
        row
        for row in rows
        if int((row.get("metrics") or {}).get("total_token_forwards", -1))
        != int((row.get("metrics") or {}).get("stage1_probe_token_forwards", 0))
        + int((row.get("metrics") or {}).get("formal_decode_token_forwards", 0))
    ]
    identity_bad = [
        row
        for row in rows
        if not (
            row.get("paper_id") == PAPER_ID
            and row.get("paper_version") == PAPER_VERSION
            and row.get("stage") == STAGE
            and (row.get("stage1") or {}).get("paper_id") == PAPER_ID
            and (row.get("stage1") or {}).get("paper_version") == PAPER_VERSION
            and (row.get("stage1") or {}).get("stage") == STAGE
            and (row.get("stage1") or {}).get("fit_scope") == FIT_SCOPE
        )
    ]
    probe_grid_bad = [
        row
        for row in rows
        if list((row.get("stage1") or {}).get("probe_lengths") or [])
        != EXPECTED_PROBE_GRID
    ]
    selected_generated_bad = [
        row
        for row in rows
        if not (
            int((row.get("stage1") or {}).get("selected_length", -1))
            in EXPECTED_PROBE_GRID
            and int((row.get("metrics") or {}).get("selected_length", -1))
            == int((row.get("stage1") or {}).get("selected_length", -2))
            == int((row.get("metrics") or {}).get("generated_length", -3))
        )
    ]
    fixed_decode_bad = [
        row
        for row in rows
        if not (
            int((row.get("metrics") or {}).get("fixed_decoder_upstream_search_forwards", -1))
            == 0
            and int(((row.get("config") or {}).get("fixed_decode") or {}).get("dstep", 0))
            == -1
            and ((row.get("config") or {}).get("fixed_decode") or {}).get("use_bias")
            is False
        )
    ]
    alias_bad = [
        row
        for row in rows
        if (row.get("metrics") or {}).get("search_forwards_semantics")
        != SEARCH_FORWARDS_ALIAS_SEMANTICS
    ]
    provenance_bad = [
        row
        for row in rows
        if expected_provenance is not None
        and row.get("provenance") != dict(expected_provenance)
    ]
    semantic_bad_ids = {
        id(row)
        for group in (
            identity_bad,
            probe_grid_bad,
            selected_generated_bad,
            fixed_decode_bad,
            alias_bad,
        )
        for row in group
    }
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
        and not semantic_bad_ids
        and not provenance_bad
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
        "identity_error_count": len(identity_bad),
        "probe_grid_error_count": len(probe_grid_bad),
        "selected_generated_error_count": len(selected_generated_bad),
        "fixed_decode_error_count": len(fixed_decode_bad),
        "search_alias_semantics_error_count": len(alias_bad),
        "semantic_error_count": len(semantic_bad_ids),
        "provenance_error_count": len(provenance_bad),
        "rows_with_expected_probe_forwards": len(rows) - len(probe_bad),
        "evaluator_completed_count": len(rows),
        "passed_count": passed_count,
        "technical_smoke_accuracy": passed_count / len(rows) if rows else None,
        "selected_length_distribution": {str(length): count for length, count in sorted(lengths.items())},
        "stage1_probe_forwards_total": sum(int((row.get("metrics") or {}).get("stage1_probe_forwards", 0)) for row in rows),
        "length_selection_forwards_total": sum(int((row.get("metrics") or {}).get("length_selection_forwards", 0)) for row in rows),
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


def audit_probe_only_rows(
    rows: Sequence[Mapping[str, Any]],
    expected: set[str],
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    base = audit_rows([], set())
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    probe_bad = [
        row
        for row in rows
        if not (
            row.get("status") == "ok"
            and row.get("arm") == ARM
            and row.get("probe_only") is True
            and "completion" not in row
            and "passed" not in row
            and "evaluator_result" not in row
            and (row.get("metrics") or {}).get("evaluator_called") is False
            and int((row.get("metrics") or {}).get("stage1_probe_forwards", -1))
            == EXPECTED_PROBE_FORWARDS
            and int((row.get("metrics") or {}).get("formal_decode_forwards", -1)) == 0
            and int((row.get("metrics") or {}).get("total_forwards", -1))
            == EXPECTED_PROBE_FORWARDS
            and list((row.get("stage1") or {}).get("probe_lengths") or [])
            == EXPECTED_PROBE_GRID
            and int((row.get("stage1") or {}).get("selected_length", -1))
            in EXPECTED_PROBE_GRID
            and _stage1_finite(row.get("stage1") or {})
            and row.get("provenance") == dict(expected_provenance)
            and int((row.get("metrics") or {}).get("total_token_forwards", -1))
            == int((row.get("metrics") or {}).get("stage1_probe_token_forwards", -2))
        )
    ]
    passed = (
        observed == expected
        and len(rows) == 1
        and all(value == 1 for value in counts.values())
        and not probe_bad
    )
    base.update(
        {
            "passed": passed,
            "expected_count": len(expected),
            "observed_count": len(rows),
            "unique_count": len(observed),
            "missing_count": len(expected - observed),
            "extra_count": len(observed - expected),
            "duplicate_count": sum(value > 1 for value in counts.values()),
            "probe_forward_error_count": len(probe_bad),
            "finite_diagnostic_error_count": len(probe_bad),
            "semantic_error_count": len(probe_bad),
            "provenance_error_count": len(probe_bad),
            "evaluator_completed_count": 0,
            "formal_decode_forwards_total": 0,
            "stage1_probe_forwards_total": sum(
                int((row.get("metrics") or {}).get("stage1_probe_forwards", 0))
                for row in rows
            ),
            "total_forwards": sum(
                int((row.get("metrics") or {}).get("total_forwards", 0)) for row in rows
            ),
            "provenance": dict(expected_provenance),
        }
    )
    return base


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
            "probe_only_complete"
            if mode == "probe_only" and audit.get("passed")
            else "technical_smoke_complete"
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
        f"- Stage-I probe/formal/total forwards：`{audit.get('stage1_probe_forwards_total')}/{audit.get('formal_decode_forwards_total')}/{audit.get('total_forwards')}`",
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
    expected_provenance = provenance_contract(preflight)
    audit = (
        audit_probe_only_rows(relevant, expected, expected_provenance)
        if mode == "probe_only"
        else audit_rows(relevant, expected, expected_provenance)
    )
    audit["provenance"] = expected_provenance
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


def _mode(args: argparse.Namespace) -> str:
    if bool(args.probe_only) and bool(args.auto_full):
        raise RuntimeError("--probe-only and --auto-full are mutually exclusive")
    if args.probe_only:
        return "probe_only"
    return "full" if args.auto_full else "smoke"


def _output_paths(output_dir: Path, mode: str) -> dict[str, Path]:
    raw_prefix = f"{ARM}_probe_only" if mode == "probe_only" else ARM
    return {
        "raw": output_dir / f"{raw_prefix}_raw.jsonl",
        "failure": output_dir / f"{raw_prefix}_failure_journal.jsonl",
        "progress": output_dir / f"{ARM}_{mode}_progress.json",
        "run_manifest": output_dir / f"{ARM}_{mode}_run_manifest.json",
    }


def _require_usable_output_version(output_dir: Path, paths: Mapping[str, Path]) -> None:
    require_no_failure_journal(paths["failure"])
    if output_dir.exists() and any(output_dir.iterdir()) and not paths["raw"].is_file():
        raise RuntimeError(
            f"output directory conflict: {output_dir} is non-empty without this mode's canonical raw; use a new version"
        )


def validate_official_fixed32_control(
    *, output_dir: Path, full_manifest_path: Path
) -> dict[str, Any]:
    arm = "official_fixed32"
    failure_path = output_dir / f"{arm}_failure_journal.jsonl"
    require_no_failure_journal(failure_path)
    raw_path = output_dir / f"{arm}_raw.jsonl"
    audit_path = output_dir / f"{arm}_full_final_audit.json"
    if not raw_path.is_file() or not audit_path.is_file():
        raise FullGateError(
            "838 scientific run requires a completed same-key official_fixed32 control"
        )
    raw = read_jsonl(raw_path)
    full_manifest = read_jsonl(full_manifest_path)
    expected = {
        official_cal_candidate_key(int(row["source_row_id"]), arm, "single-line")
        for row in full_manifest
    }
    counts = Counter(str(row.get("candidate_key") or "") for row in raw)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    valid_rows = all(
        row.get("status") == "ok"
        and row.get("arm") == arm
        and row.get("config") == OFFICIAL_FIXED32_CONFIG
        and row.get("official_cal_commit") == EXPECTED_CAL_COMMIT
        and row.get("evaluator_commit") == EXPECTED_HUMANEVAL_COMMIT
        and int((row.get("metrics") or {}).get("selected_length", -1)) == 32
        for row in raw
    )
    if not (
        len(raw) == FULL_COUNT
        and set(counts) == expected
        and all(value == 1 for value in counts.values())
        and valid_rows
        and audit.get("passed") is True
        and int(audit.get("expected_count", -1)) == FULL_COUNT
        and int(audit.get("observed_count", -1)) == FULL_COUNT
        and int(audit.get("missing_count", -1)) == 0
        and int(audit.get("duplicate_count", -1)) == 0
        and int(audit.get("error_count", -1)) == 0
        and int(audit.get("failure_journal_count", -1)) == 0
    ):
        raise FullGateError(
            "official_fixed32 control is incomplete, corrupt, or not on the exact 838 keys"
        )
    return {
        "passed": True,
        "arm": arm,
        "row_count": FULL_COUNT,
        "full_manifest_sha256": FULL_MANIFEST_SHA256,
        "decoder_config": OFFICIAL_FIXED32_CONFIG,
        "raw_path": str(raw_path),
    }


def run(args: argparse.Namespace) -> int:
    official_cal_root = Path(args.official_cal_root).resolve()
    humaneval_root = Path(args.humaneval_root).resolve()
    current_dataset = Path(args.current_dataset).resolve()
    manifest_path = Path(args.manifest_jsonl).resolve()
    full_manifest_path = Path(args.full_manifest_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    analysis_output_dir = Path(args.analysis_output_dir).resolve() if args.analysis_output_dir else None
    hf_home = Path(args.hf_home).resolve()
    mode = _mode(args)
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
        mode=mode,
        smoke_cases=int(args.smoke_cases),
        official_cal_root=official_cal_root,
        humaneval_root=humaneval_root,
        hf_home=hf_home,
    )
    if args.preflight_only:
        print(json.dumps({key: value for key, value in preflight.items() if key != "manifest"}, sort_keys=True))
        return 0

    if not preflight["implementation_tracked_worktree_clean"]:
        raise RuntimeError("workspace gate failed: tracked worktree is not clean")
    if not (
        REPO / "docs/paper_agent/experiments/20260731_lrdllm_stage1_protocol_freeze.zh.md"
    ).is_file():
        raise RuntimeError("workspace gate failed: protocol freeze is missing")
    current_provenance = provenance_contract(preflight)
    if mode in {"smoke", "full"}:
        if not args.probe_output_dir:
            raise FullGateError("probe-only gate directory is required before smoke/full")
        validate_probe_only_gate(
            smoke_probe_output_dir=Path(args.probe_output_dir).resolve(),
            arm=ARM,
            current_provenance=current_provenance,
        )
    if mode == "full":
        if not args.smoke_output_dir:
            raise FullGateError("strict full gate requires --smoke-output-dir")
        validate_full_gate(
            smoke_output_dir=Path(args.smoke_output_dir).resolve(),
            arm=ARM,
            current_provenance=current_provenance,
        )
        if not args.official_fixed32_output_dir:
            raise FullGateError(
                "838 scientific run requires --official-fixed32-output-dir"
            )
        validate_official_fixed32_control(
            output_dir=Path(args.official_fixed32_output_dir).resolve(),
            full_manifest_path=full_manifest_path,
        )
    execution_gate = execution_gpu_ecc_gate()

    manifest = list(preflight["manifest"])
    paths = _output_paths(output_dir, mode)
    _require_usable_output_version(output_dir, paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = paths["raw"]
    failure_path = paths["failure"]
    progress_path = paths["progress"]
    run_manifest_path = paths["run_manifest"]
    existing = canonical_success_rows(raw_path)
    all_known = expected_keys(
        manifest if mode == "probe_only" else read_jsonl(full_manifest_path), mode
    )
    existing_keys = {str(row["candidate_key"]) for row in existing}
    if not existing_keys <= all_known:
        raise RuntimeError("canonical raw contains keys outside the immutable non-frozen population")
    expected = expected_keys(manifest, mode)
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
            "provenance": current_provenance,
            "execution_gate": execution_gate,
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
        from transformers import AutoModel, AutoTokenizer

        snapshot_path = Path(preflight["model_provenance"]["snapshot_path"])
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path, local_files_only=True, trust_remote_code=True
        )
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"
        if resolve_mask_token_id(tokenizer) != EXPECTED_MASK_TOKEN_ID:
            raise RuntimeError("resolved tokenizer mask token id mismatch before model load")
        model = AutoModel.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda").eval()
        if mode == "probe_only":
            generate = check_correctness = evaluator_provenance = None
        else:
            generate, check_correctness, evaluator_provenance = load_official_runtime(
                official_cal_root, humaneval_root
            )
            if not (
                evaluator_provenance["evaluator_source_sha256"]
                == preflight["evaluator_source_audit"]["execution_source_sha256"]
                and evaluator_provenance["evaluator_enabled_source_sha256"]
                == preflight["evaluator_source_audit"][
                    "enabled_execution_source_sha256"
                ]
            ):
                raise RuntimeError("loaded evaluator provenance differs from CPU preflight")
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

    source_ids = {
        int(row["source_row_id"])
        for row in manifest
        if candidate_key(int(row["source_row_id"]), mode) not in completed
    }
    selected_sources = load_selected_dataset_rows(current_dataset, source_ids)
    manifest_by_source = {int(row["source_row_id"]): row for row in manifest}
    for source_row_id in sorted(source_ids, key=lambda value: int(manifest_by_source[value].get("smoke_order", value))):
        item = manifest_by_source[source_row_id]
        key = candidate_key(source_row_id, mode)
        source = selected_sources[source_row_id]
        if str(source.get("task_id")) != str(item["task_id"]):
            raise RuntimeError("selected source row task_id does not match immutable manifest")
        case_started = time.perf_counter()
        try:
            common = {
                "candidate_key": key,
                "arm": ARM,
                "implementation_label": IMPLEMENTATION_LABEL,
                "decode_label": DECODE_LABEL,
                "paper_id": PAPER_ID,
                "paper_version": PAPER_VERSION,
                "stage": STAGE,
                "source_row_id": source_row_id,
                "task_id": str(source["task_id"]),
                "task_group": str(item["task_group"]),
                "benchmark_name": "single-line",
                "population": "CAL-Rest_intersection_project_non-frozen",
                "status": "ok",
                "official_cal_decoder_commit": EXPECTED_CAL_COMMIT,
                "evaluator_commit": EXPECTED_HUMANEVAL_COMMIT,
                "model": MODEL_PATH,
                "model_revision": MODEL_REVISION,
                "provenance": current_provenance,
                "fixed_decoder_source_audit": preflight["fixed_decoder_source_audit"],
                "model_artifact_sha256": preflight["model_provenance"]["artifact_sha256"],
                "runtime_environment": preflight["runtime_environment"],
                "gpu": torch.cuda.get_device_name(),
            }
            if mode == "probe_only":
                result = probe_only_one(
                    source=source,
                    tokenizer=tokenizer,
                    model=model,
                    seed=int(args.seed),
                )
                row = {
                    **common,
                    "probe_only": True,
                    "stage1": result["stage1"],
                    "metrics": result["metrics"],
                    "config": {
                        "max_probe_length": MAX_PROBE_LENGTH,
                        "fit_scope": result["stage1"]["fit_scope"],
                        "seed": int(args.seed),
                    },
                }
            else:
                result = decode_one(
                    source=source,
                    tokenizer=tokenizer,
                    model=model,
                    generate=generate,
                    check_correctness=check_correctness,
                    seed=int(args.seed),
                )
                row = {
                    **common,
                    "passed": result["passed"],
                    "completion": result["completion"],
                    "evaluator_result": result["evaluator_result"],
                    "evaluator_provenance": evaluator_provenance,
                    "stage1": result["stage1"],
                    "metrics": result["metrics"],
                    "config": {
                        "max_probe_length": MAX_PROBE_LENGTH,
                        "fit_scope": result["stage1"]["fit_scope"],
                        "fixed_decode": result["fixed_decode_config"],
                        "seed": int(args.seed),
                    },
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
                "probe_only": mode == "probe_only",
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
    result.add_argument(
        "--hf-home", default=os.environ.get("HF_HOME", "/home/shx/.cache/huggingface")
    )
    result.add_argument("--probe-output-dir")
    result.add_argument("--smoke-output-dir")
    result.add_argument("--official-fixed32-output-dir")
    result.add_argument("--arm", choices=(ARM,), default=ARM)
    result.add_argument("--max-probe-length", type=int, default=MAX_PROBE_LENGTH)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--smoke-cases", type=int, default=12)
    result.add_argument("--progress-every", type=int, default=1)
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--probe-only", action="store_true")
    result.add_argument("--auto-full", action="store_true")
    return result


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
