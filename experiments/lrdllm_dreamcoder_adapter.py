#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
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

from experiments.lrdllm_algorithm import (
    MAX_GEN,
    PROBE_GRID,
    ForwardTokenLedger,
    assert_selection_payload_safe,
    canonical_success_index,
    choose_local_length,
    derive_row_seed,
    local_neighbors,
    select_initial_length,
)
from analysis.build_lrdllm_common_manifests import row_fingerprint


MODEL_ID = "Dream-org/Dream-Coder-v0-Base-7B"
MODEL_REVISION = "2346ccd3be517d0d314152b988a3b9bafa7d6d63"
MODEL_CONFIG_SHA256 = "3c180c6d6d9b55a9d3a083b0460c27520ee3c695ff3724e2e87200849cff08c9"
MODEL_INDEX_SHA256 = "998a078123ffc97763690de7f2a677eb89168af5eaf8a5e12e6bc24d18e25bdb"
EVALUATOR_REVISION = "88062ff9859c875d04db115b698ed4b0f0395170"
ARMS = ("lrdllm_primary", "fixed64_common_protocol")
DATASETS = {
    "singleline": ("HumanEval-SingleLineInfilling", 927),
    "randomspan": ("HumanEval-RandomSpanInfilling", 1480),
    "multiline": ("HumanEval-MultiLineInfilling", 5079),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


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


def enable_pinned_evaluator_source(source: str) -> str:
    marker = "#                     exec(check_program, exec_globals)"
    replacement = "                    exec(check_program, exec_globals)"
    if source.count(marker) != 1:
        raise RuntimeError("pinned evaluator enablement marker is absent or ambiguous")
    return source.replace(marker, replacement)


def load_pinned_evaluator(evaluator_root: Path) -> tuple[Any, dict[str, Any]]:
    if git_head(evaluator_root) != EVALUATOR_REVISION:
        raise RuntimeError("HumanEval-Infilling evaluator checkout is not pinned")
    source_path = evaluator_root / "human_eval_infilling" / "execution.py"
    source = source_path.read_text(encoding="utf-8")
    enabled = enable_pinned_evaluator_source(source)
    namespace: dict[str, Any] = {"__name__": "pinned_humaneval_infilling_execution_enabled", "__file__": str(source_path)}
    exec(compile(enabled, str(source_path), "exec"), namespace, namespace)
    check_correctness = namespace.get("check_correctness")
    if not callable(check_correctness):
        raise RuntimeError("pinned evaluator overlay did not define check_correctness")
    return check_correctness, {
        "evaluator_revision": EVALUATOR_REVISION,
        "evaluator_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "evaluator_enabled_source_sha256": hashlib.sha256(enabled.encode("utf-8")).hexdigest(),
        "evaluator_mode": "upstream_documented_local_enablement_overlay",
    }


def load_dataset_rows(evaluator_root: Path, dataset_name: str) -> list[dict[str, Any]]:
    path = evaluator_root / "data" / f"{dataset_name}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selection_view(source: Mapping[str, Any]) -> dict[str, str]:
    view = {
        "task_id": str(source["task_id"]),
        "prefix": str(source["prompt"]),
        "suffix": str(source["suffix"]),
        "entry_point": str(source["entry_point"]),
    }
    assert_selection_payload_safe(view)
    return view


def verify_manifest_source_rows(manifest: Sequence[Mapping[str, Any]], source_rows: Sequence[Mapping[str, Any]]) -> None:
    for item in manifest:
        source = source_rows[int(item["source_row_id"])]
        safe = {
            "task_id": str(source["task_id"]),
            "prompt": str(source["prompt"]),
            "suffix": str(source["suffix"]),
            "entry_point": str(source["entry_point"]),
        }
        if str(item["task_id"]) != safe["task_id"] or str(item["safe_row_sha256"]) != row_fingerprint(safe):
            raise RuntimeError("manifest safe-row fingerprint mismatch")


def certify_population(
    *,
    dataset_name: str,
    manifest_path: Path,
    full_manifest_path: Path,
    summary_path: Path,
    evaluator_root: Path,
) -> dict[str, Any]:
    short_name = next(short for short, contract in DATASETS.items() if contract[0] == dataset_name)
    expected_count = DATASETS[short_name][1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_summary = summary["datasets"][dataset_name]
    if summary["sealed_files_opened"] is not False or summary["test_evaluation_count"] != 0:
        raise RuntimeError("manifest summary violates frozen seal")
    if dataset_summary["nonfrozen_rows"] != expected_count or dataset_summary["nonfrozen_clusters"] != 148:
        raise RuntimeError("manifest summary population mismatch")
    full_manifest = read_jsonl(full_manifest_path)
    if len(full_manifest) != expected_count or len({row["task_group"] for row in full_manifest}) != 148:
        raise RuntimeError("full manifest population mismatch")
    if dataset_summary["full_manifest_sha256"] != sha256(full_manifest_path):
        raise RuntimeError("full manifest hash mismatch")
    source_path = evaluator_root / "data" / f"{dataset_name}.jsonl.gz"
    if dataset_summary["source_sha256"] != sha256(source_path):
        raise RuntimeError("official dataset hash mismatch")
    manifest = read_jsonl(manifest_path)
    full_keys = {str(row["candidate_key"]) for row in full_manifest}
    if not manifest or not {str(row["candidate_key"]) for row in manifest} <= full_keys:
        raise RuntimeError("run manifest is not a non-empty subset of the immutable full manifest")
    return {
        "manifest_summary": str(summary_path),
        "manifest_summary_sha256": sha256(summary_path),
        "manifest_sha256": sha256(manifest_path),
        "full_manifest_sha256": sha256(full_manifest_path),
        "source_dataset_sha256": sha256(source_path),
        "population_rows": len(manifest),
        "population_clusters": len({row["task_group"] for row in manifest}),
    }


def model_provenance(snapshot: Path) -> dict[str, Any]:
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError("DreamCoder snapshot revision mismatch")
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if sha256(config_path) != MODEL_CONFIG_SHA256 or sha256(index_path) != MODEL_INDEX_SHA256:
        raise RuntimeError("DreamCoder cached checkpoint hash mismatch")
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot": str(snapshot),
        "config_sha256": MODEL_CONFIG_SHA256,
        "model_index_sha256": MODEL_INDEX_SHA256,
        "license": "not_present_in_cached_snapshot; external metadata verification pending",
    }


def candidate_keys(manifest: Sequence[Mapping[str, Any]]) -> set[str]:
    keys = {str(row["candidate_key"]) for row in manifest}
    if len(keys) != len(manifest) or "" in keys:
        raise RuntimeError("manifest candidate keys are blank or duplicated")
    return keys


def canonical_rows(path: Path, full_keys: set[str], arm: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    canonical_success_index(rows, full_keys)
    if any(str(row.get("arm")) != arm for row in rows):
        raise RuntimeError("canonical raw contains a wrong arm")
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
    rows_per_hour = completed_since_start / elapsed * 3600.0 if elapsed and completed_since_start else None
    missing = expected_count - completed_count
    eta_seconds = missing / rows_per_hour * 3600.0 if rows_per_hour and missing else 0.0 if not missing else None
    return {
        "status": "running",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": arm,
        "mode": mode,
        "expected_count": expected_count,
        "completed_count": completed_count,
        "existing_completed_count_at_start": starting_completed_count,
        "missing_count": missing,
        "failure_journal_count": failure_journal_count,
        "rows_per_hour": rows_per_hour,
        "eta_seconds": eta_seconds,
        "gpu": gpu_snapshot(),
    }


def ensure_modeling_rope_utils_available() -> None:
    import transformers.utils as transformers_utils

    if importlib.util.find_spec("transformers.modeling_flash_attention_utils") is None or os.environ.get("DLLM_DISABLE_FLASH_ATTN") == "1":
        def unavailable(*_args: Any, **_kwargs: Any) -> bool:
            return False

        transformers_utils.is_flash_attn_2_available = unavailable  # type: ignore[attr-defined]
        transformers_utils.is_flash_attn_greater_or_equal = unavailable  # type: ignore[attr-defined]
        transformers_utils.is_flash_attn_greater_or_equal_2_10 = unavailable  # type: ignore[attr-defined]
    try:
        import transformers.modeling_rope_utils  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    fallback = Path("/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages/transformers/modeling_rope_utils.py")
    spec = importlib.util.spec_from_file_location("transformers.modeling_rope_utils", fallback)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError("transformers.modeling_rope_utils is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules["transformers.modeling_rope_utils"] = module
    spec.loader.exec_module(module)


def resolve_special_ids(tokenizer: Any) -> dict[str, int]:
    def required(name: str, fallback_token: str) -> int:
        value = getattr(tokenizer, name, None)
        if value is None:
            value = tokenizer.convert_tokens_to_ids(fallback_token)
        if value is None:
            raise ValueError(f"tokenizer lacks {name}")
        return int(value)

    eos = required("eos_token_id", "<|endoftext|>")
    return {
        "bos": required("bos_token_id", "<|beginoftext|>"),
        "eos": eos,
        "pad": int(getattr(tokenizer, "pad_token_id", None) or eos),
        "mask": required("mask_token_id", "<|mask|>"),
    }


def build_canvas_ids(
    prefix_ids: Sequence[int],
    committed_ids: Sequence[int],
    suffix_ids: Sequence[int],
    remaining_length: int,
    special_ids: Mapping[str, int],
) -> tuple[list[int], int, int]:
    if not 1 <= remaining_length <= MAX_GEN:
        raise ValueError("remaining length is outside [1, 128]")
    left = [int(special_ids["bos"]), *map(int, prefix_ids), *map(int, committed_ids)]
    middle_start = len(left)
    values = [*left, *([int(special_ids["mask"])] * remaining_length), *map(int, suffix_ids), int(special_ids["eos"])]
    return values, middle_start, middle_start + remaining_length


def shifted_logits_for_dream(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=input_ids)
    logits = output.logits if hasattr(output, "logits") else output[0]
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def mean_negative_entropy_from_logits(logits: torch.Tensor) -> float:
    probabilities = torch.softmax(logits.float(), dim=-1)
    values = (probabilities * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))).sum(dim=-1)
    return float(values.mean().item())


def sample_top_p_torch(
    logits: torch.Tensor,
    *,
    generator: torch.Generator,
    forbidden_token_ids: set[int],
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> int:
    scores = logits.float().clone() / temperature
    valid_forbidden = [token_id for token_id in forbidden_token_ids if 0 <= token_id < scores.numel()]
    if valid_forbidden:
        scores[torch.tensor(valid_forbidden, device=scores.device)] = -torch.inf
    if not torch.isfinite(scores).any():
        raise RuntimeError("no finite non-special token remains for sampling")
    probabilities = torch.softmax(scores, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > top_p
    sorted_probs[remove] = 0
    sorted_probs /= sorted_probs.sum()
    sampled_rank = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    return int(sorted_indices[sampled_rank].item())


def model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    return torch.device(device) if device is not None else next(model.parameters()).device


def forward_canvas(
    *,
    model: Any,
    prefix_ids: Sequence[int],
    committed_ids: Sequence[int],
    suffix_ids: Sequence[int],
    remaining_length: int,
    special_ids: Mapping[str, int],
    ledger: ForwardTokenLedger,
    stage: str,
    return_left_logits: bool,
) -> tuple[float | None, torch.Tensor | None]:
    canvas, middle_start, middle_end = build_canvas_ids(prefix_ids, committed_ids, suffix_ids, remaining_length, special_ids)
    ledger.record_forward(stage, len(canvas))
    inputs = torch.tensor([canvas], dtype=torch.long, device=model_device(model))
    with torch.no_grad():
        logits = shifted_logits_for_dream(model, inputs)
    middle = logits[:, middle_start:middle_end, :]
    confidence = None if return_left_logits else mean_negative_entropy_from_logits(middle)
    left_logits = middle[0, 0].detach() if return_left_logits else None
    return confidence, left_logits


def decode_lrdllm(
    *,
    safe: Mapping[str, str],
    tokenizer: Any,
    model: Any,
    candidate_key: str,
    global_seed: int,
) -> dict[str, Any]:
    prefix_ids = tokenizer.encode(safe["prefix"], add_special_tokens=False)
    suffix_ids = tokenizer.encode(safe["suffix"], add_special_tokens=False)
    special = resolve_special_ids(tokenizer)
    forbidden = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    forbidden.update(special.values())
    ledger = ForwardTokenLedger()
    cache: dict[int, float] = {}

    def confidence(length: int, stage: str, committed: Sequence[int]) -> float:
        if length in cache:
            ledger.record_cache_hit(stage)
            return cache[length]
        value, _ = forward_canvas(
            model=model,
            prefix_ids=prefix_ids,
            committed_ids=committed,
            suffix_ids=suffix_ids,
            remaining_length=length,
            special_ids=special,
            ledger=ledger,
            stage=stage,
            return_left_logits=False,
        )
        if value is None:
            raise RuntimeError("confidence forward returned no score")
        cache[length] = value
        return value

    stage_i_raw = {length: confidence(length, "stage_i", ()) for length in PROBE_GRID}
    initial_length, fit, stage_i_adjusted = select_initial_length(stage_i_raw)
    remaining = initial_length
    committed: list[int] = []
    expansion_moves = 0
    contraction_moves = 0
    local_move_counts: list[int] = []
    stable_lengths: list[int] = []
    row_seed = derive_row_seed(global_seed, candidate_key)
    generator = torch.Generator(device=model_device(model)).manual_seed(row_seed)
    while remaining > 0 and len(committed) < MAX_GEN:
        moves = 0
        while True:
            scores = {length: confidence(length, "stage_ii", committed) for length in local_neighbors(remaining)}
            selected = choose_local_length(remaining, scores, fit.slope)
            if selected == remaining:
                break
            moves += 1
            if moves > MAX_GEN:
                raise RuntimeError("LR-DLLM local search exceeded 128 moves")
            if selected > remaining:
                expansion_moves += selected - remaining
            else:
                contraction_moves += remaining - selected
            remaining = selected
        local_move_counts.append(moves)
        stable_lengths.append(remaining)
        _, left_logits = forward_canvas(
            model=model,
            prefix_ids=prefix_ids,
            committed_ids=committed,
            suffix_ids=suffix_ids,
            remaining_length=remaining,
            special_ids=special,
            ledger=ledger,
            stage="commit",
            return_left_logits=True,
        )
        if left_logits is None:
            raise RuntimeError("commit forward returned no logits")
        committed.append(sample_top_p_torch(left_logits, generator=generator, forbidden_token_ids=forbidden))
        remaining -= 1
        cache.clear()
    termination = "remaining_length_zero" if remaining == 0 else "max_gen_reached"
    completion = tokenizer.decode(committed, skip_special_tokens=True)
    snapshot = ledger.snapshot()
    return {
        "completion": completion,
        "row_seed": row_seed,
        "initial_selected_length": initial_length,
        "final_generated_length": len(committed),
        "final_remaining_length": remaining,
        "stage_i_alpha": fit.alpha,
        "stage_i_slope": fit.slope,
        "stage_i_raw_confidence": stage_i_raw,
        "stage_i_adjusted_confidence": stage_i_adjusted,
        "expansion_moves": expansion_moves,
        "contraction_moves": contraction_moves,
        "local_search_events": len(local_move_counts),
        "local_move_counts": local_move_counts,
        "stable_length_trace": stable_lengths,
        "termination_reason": termination,
        **snapshot,
    }


def decode_fixed64(
    *,
    safe: Mapping[str, str],
    tokenizer: Any,
    model: Any,
    candidate_key: str,
    global_seed: int,
) -> dict[str, Any]:
    prefix_ids = tokenizer.encode(safe["prefix"], add_special_tokens=False)
    suffix_ids = tokenizer.encode(safe["suffix"], add_special_tokens=False)
    special = resolve_special_ids(tokenizer)
    forbidden = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    forbidden.update(special.values())
    ledger = ForwardTokenLedger()
    committed: list[int] = []
    remaining = 64
    row_seed = derive_row_seed(global_seed, candidate_key)
    generator = torch.Generator(device=model_device(model)).manual_seed(row_seed)
    while remaining > 0:
        _, left_logits = forward_canvas(
            model=model,
            prefix_ids=prefix_ids,
            committed_ids=committed,
            suffix_ids=suffix_ids,
            remaining_length=remaining,
            special_ids=special,
            ledger=ledger,
            stage="commit",
            return_left_logits=True,
        )
        if left_logits is None:
            raise RuntimeError("fixed64 commit forward returned no logits")
        committed.append(sample_top_p_torch(left_logits, generator=generator, forbidden_token_ids=forbidden))
        remaining -= 1
    return {
        "completion": tokenizer.decode(committed, skip_special_tokens=True),
        "row_seed": row_seed,
        "initial_selected_length": 64,
        "final_generated_length": 64,
        "final_remaining_length": 0,
        "expansion_moves": 0,
        "contraction_moves": 0,
        "local_search_events": 0,
        "local_move_counts": [],
        "stable_length_trace": [],
        "termination_reason": "fixed64_remaining_length_zero",
        **ledger.snapshot(),
    }


def audit_rows(rows: Sequence[Mapping[str, Any]], expected: set[str], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("candidate_key")) in expected]
    counts = Counter(str(row.get("candidate_key") or "") for row in selected)
    observed = set(counts)
    accounting_errors = [
        row
        for row in selected
        if int((row.get("metrics") or {}).get("total_forward_calls", -1))
        != int((row.get("metrics") or {}).get("search_forward_calls", 0))
        + int((row.get("metrics") or {}).get("decode_forward_calls", 0))
    ]
    return {
        "passed": observed == expected and all(count == 1 for count in counts.values()) and not accounting_errors,
        "arm": arm,
        "expected_count": len(expected),
        "observed_count": len(selected),
        "missing_count": len(expected - observed),
        "extra_count": len(observed - expected),
        "duplicate_count": sum(count > 1 for count in counts.values()),
        "error_count": 0,
        "forward_accounting_error_count": len(accounting_errors),
    }


def run(args: argparse.Namespace) -> int:
    dataset_name, full_count = DATASETS[str(args.dataset)]
    manifest_path = Path(args.manifest_jsonl).resolve()
    full_manifest_path = Path(args.full_manifest_jsonl).resolve()
    summary_path = Path(args.manifest_summary_json).resolve()
    evaluator_root = Path(args.evaluator_root).resolve()
    model_snapshot = Path(args.model_snapshot).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = read_jsonl(manifest_path)
    full_manifest = read_jsonl(full_manifest_path)
    if int(args.progress_every) <= 0:
        raise ValueError("--progress-every must be positive")
    mode = str(args.mode)
    expected_mode_count = {"technical-smoke": 12, "mechanism-smoke": 64, "full": full_count}[mode]
    if len(manifest) != expected_mode_count:
        raise RuntimeError(f"{mode} requires exactly {expected_mode_count} manifest rows")
    population = certify_population(
        dataset_name=dataset_name,
        manifest_path=manifest_path,
        full_manifest_path=full_manifest_path,
        summary_path=summary_path,
        evaluator_root=evaluator_root,
    )
    model_info = model_provenance(model_snapshot)
    source_rows = load_dataset_rows(evaluator_root, dataset_name)
    verify_manifest_source_rows(manifest, source_rows)
    if args.preflight_only:
        ensure_modeling_rope_utils_available()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_snapshot, trust_remote_code=True, local_files_only=True)
        print(
            json.dumps(
                {
                    "population": population,
                    "model": model_info,
                    "evaluator_revision": git_head(evaluator_root),
                    "tokenizer_special_ids": resolve_special_ids(tokenizer),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    arm = str(args.arm)
    raw_path = output_dir / f"{arm}_raw.jsonl"
    failure_path = output_dir / f"{arm}_failure_journal.jsonl"
    progress_path = output_dir / f"{arm}_{mode}_progress.json"
    run_manifest_path = output_dir / f"{arm}_{mode}_run_manifest.json"
    full_keys = candidate_keys(full_manifest)
    expected = candidate_keys(manifest)
    existing = canonical_rows(raw_path, full_keys, arm)
    completed = {str(row["candidate_key"]) for row in existing} & expected
    starting_completed = len(completed)
    started = time.perf_counter()
    failure_count = len(read_jsonl(failure_path))
    initial = progress_payload(
        arm=arm,
        mode=mode,
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
            "dataset": dataset_name,
            "manifest": str(manifest_path),
            "full_manifest": str(full_manifest_path),
            "temperature": 0.2,
            "top_p": 0.9,
            "max_gen": 128,
            "global_seed": int(args.seed),
            "implementation_label": "paper-guided, author-unverified reimplementation of LR-DLLM" if arm == "lrdllm_primary" else "Fixed64 common-protocol control for LR-DLLM",
            "commit_sampling_forward_policy": "independent forward; no double-counted search/decode call",
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
    check_correctness, evaluator_info = load_pinned_evaluator(evaluator_root)
    ensure_modeling_rope_utils_available()
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_snapshot,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    torch.cuda.reset_peak_memory_stats()
    manifest_by_key = {str(row["candidate_key"]): row for row in manifest}
    for key in sorted(expected - completed, key=lambda value: int(manifest_by_key[value]["source_row_id"])):
        item = manifest_by_key[key]
        source_row_id = int(item["source_row_id"])
        source = source_rows[source_row_id]
        safe = selection_view(source)
        if str(safe["task_id"]) != str(item["task_id"]):
            raise RuntimeError("manifest/source task id mismatch")
        row_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            decoded = (
                decode_lrdllm(safe=safe, tokenizer=tokenizer, model=model, candidate_key=key, global_seed=int(args.seed))
                if arm == "lrdllm_primary"
                else decode_fixed64(safe=safe, tokenizer=tokenizer, model=model, candidate_key=key, global_seed=int(args.seed))
            )
            completion = decoded.pop("completion")
            evaluator_started = time.perf_counter()
            evaluation = check_correctness(source, completion, 3.0, 0)
            evaluator_sec = time.perf_counter() - evaluator_started
            metrics = {
                **decoded,
                "decode_wall_sec": evaluator_started - row_started,
                "evaluator_sec": evaluator_sec,
                "wall_sec": time.perf_counter() - row_started,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
            append_jsonl(
                raw_path,
                {
                    "candidate_key": key,
                    "arm": arm,
                    "dataset": dataset_name,
                    "source_row_id": source_row_id,
                    "task_id": str(item["task_id"]),
                    "task_group": str(item["task_group"]),
                    "status": "ok",
                    "passed": bool(evaluation["passed"]),
                    "evaluator_result": evaluation["result"],
                    "completion": completion,
                    "metrics": metrics,
                },
            )
            completed.add(key)
        except Exception as exc:
            append_jsonl(
                failure_path,
                {
                    "candidate_key": key,
                    "arm": arm,
                    "dataset": dataset_name,
                    "source_row_id": source_row_id,
                    "task_id": str(item["task_id"]),
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
                mode=mode,
                expected_count=len(expected),
                completed_count=len(completed),
                starting_completed_count=starting_completed,
                failure_journal_count=failure_count,
                started=started,
            )
            failed["status"] = "failed"
            atomic_write_json(progress_path, failed)
            atomic_write_json(run_manifest_path, {**json.loads(run_manifest_path.read_text()), **failed, "evaluator": evaluator_info})
            return 2
        if len(completed) % int(args.progress_every) == 0 or len(completed) == len(expected):
            atomic_write_json(
                progress_path,
                progress_payload(
                    arm=arm,
                    mode=mode,
                    expected_count=len(expected),
                    completed_count=len(completed),
                    starting_completed_count=starting_completed,
                    failure_journal_count=failure_count,
                    started=started,
                ),
            )
    rows = canonical_rows(raw_path, full_keys, arm)
    audit = audit_rows(rows, expected, arm)
    final = progress_payload(
        arm=arm,
        mode=mode,
        expected_count=len(expected),
        completed_count=len(completed),
        starting_completed_count=starting_completed,
        failure_journal_count=failure_count,
        started=started,
    )
    final.update({"status": "completed" if audit["passed"] and failure_count == 0 else "audit_failed", "new_rows_written": len(completed) - starting_completed, "final_audit": audit})
    atomic_write_json(progress_path, final)
    atomic_write_json(run_manifest_path, {**json.loads(run_manifest_path.read_text()), **final, "evaluator": evaluator_info})
    return 0 if audit["passed"] and failure_count == 0 else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="DreamCoder adapter for the preregistered author-unverified LR-DLLM reimplementation and Fixed64 control.")
    result.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    result.add_argument("--manifest-jsonl", required=True)
    result.add_argument("--full-manifest-jsonl", required=True)
    result.add_argument("--manifest-summary-json", required=True)
    result.add_argument("--evaluator-root", required=True)
    result.add_argument("--model-snapshot", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--arm", choices=ARMS, required=True)
    result.add_argument("--mode", choices=("technical-smoke", "mechanism-smoke", "full"), required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--progress-every", type=int, default=1)
    result.add_argument("--preflight-only", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
