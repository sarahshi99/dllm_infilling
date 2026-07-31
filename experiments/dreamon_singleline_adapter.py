#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.lrdllm_dreamcoder_adapter import (
    append_jsonl,
    atomic_write_json,
    certify_population,
    ensure_modeling_rope_utils_available,
    git_head,
    gpu_snapshot,
    load_dataset_rows,
    load_pinned_evaluator,
    progress_payload,
    read_jsonl,
    selection_view,
    sha256,
    verify_manifest_source_rows,
)


SOURCE_REVISION = "8a0a54918412eda9402a327646f7f067f7160ec8"
SOURCE_GENERATOR_SHA256 = "7709f1ef5b465ae135c12a3dcd34faeae49e09e5cfb99b29606b4e925aa18858"
SOURCE_EVALUATE_SHA256 = "d8cf3ce89a23b182444d42f270d4b3bd030a9dc7e752e075b92919fb1a1da57c"
SOURCE_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
MODEL_INDEX_SHA256 = "998a078123ffc97763690de7f2a677eb89168af5eaf8a5e12e6bc24d18e25bdb"
MODEL_PROFILES = {
    "dreamon": {
        "id": "Dream-org/DreamOn-v0-7B",
        "revision": "8ccc74750e43177327f29dab9e91882ba759e194",
        "config_sha256": "c02a98999d7d22491e6cd43f9830fdfe241773cc0d0470f205d1850a3938e8d3",
        "license": "apache-2.0",
    },
    "dreamcoder_fixed": {
        "id": "Dream-org/Dream-Coder-v0-Base-7B",
        "revision": "2346ccd3be517d0d314152b988a3b9bafa7d6d63",
        "config_sha256": "3c180c6d6d9b55a9d3a083b0460c27520ee3c695ff3724e2e87200849cff08c9",
        "license": "not_present_in_cached_snapshot; external metadata verification pending",
    },
}
DATASETS = {
    "singleline": ("HumanEval-SingleLineInfilling", 927),
    "multiline": ("HumanEval-MultiLineInfilling", 5079),
}
CLUSTER_COUNT = 148
INITIAL_LENGTHS = (4, 8, 16, 32, 64)


def arm_name(min_gen_len: int, model_profile: str = "dreamon") -> str:
    if int(min_gen_len) not in INITIAL_LENGTHS:
        raise ValueError(f"DreamOn min_gen_len must be one of {INITIAL_LENGTHS}")
    if model_profile == "dreamon":
        return f"dreamon_dynamic_min{int(min_gen_len)}_max64"
    if model_profile == "dreamcoder_fixed":
        return f"dreamcoder_fixed{int(min_gen_len)}_dreamon_decoder"
    raise ValueError(f"unsupported model profile: {model_profile}")


def protocol_config(min_gen_len: int, model_profile: str = "dreamon") -> dict[str, Any]:
    arm_name(min_gen_len, model_profile)
    max_gen_len = 64 if model_profile == "dreamon" else int(min_gen_len)
    return {
        "dtype": "bf16",
        "device": "cuda",
        "max_tokens": 2048,
        "max_prompt_len": 2048,
        "min_gen_len": int(min_gen_len),
        "max_gen_len": max_gen_len,
        "batch_size": 1,
        "steps": 256,
        "eps": 1e-3,
        "pad_to_max_len": False,
        "pad_eos_to_right": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": None,
        "alg": "entropy",
        "alg_temp": 0.0,
        "delete_eos_token": True,
        "mask_expansion": True,
        "show_progress": False,
    }


def derive_row_seed(global_seed: int, candidate_key: str) -> int:
    digest = hashlib.sha256(f"{int(global_seed)}\0{candidate_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def candidate_key(
    model_profile: str,
    source_row_id: int,
    arm: str,
    dataset_profile: str = "singleline",
) -> str:
    if dataset_profile not in DATASETS:
        raise ValueError(f"unsupported dataset profile: {dataset_profile}")
    model_prefix = "dreamon" if model_profile == "dreamon" else "dreamcoder"
    prefix = f"{model_prefix}_{dataset_profile}"
    return f"{prefix}_source_row={int(source_row_id)}|arm={arm}"


def paired_seed_key(source_row_id: int, length: int, dataset_profile: str = "singleline") -> str:
    return candidate_key(
        "dreamon", source_row_id, arm_name(length, "dreamon"), dataset_profile
    )


def expected_keys(
    manifest: Sequence[Mapping[str, Any]],
    arm: str,
    model_profile: str = "dreamon",
    dataset_profile: str = "singleline",
) -> set[str]:
    keys = {
        candidate_key(model_profile, int(row["source_row_id"]), arm, dataset_profile)
        for row in manifest
    }
    if len(keys) != len(manifest) or "" in keys:
        raise RuntimeError("DreamOn manifest has duplicate or blank normalized keys")
    return keys


def call_with_generation_profile(
    callback: Callable[[], Any], target_code: Any
) -> tuple[Any, dict[str, int]]:
    captured: dict[str, int] = {}
    previous = sys.getprofile()

    def profiler(frame: Any, event: str, arg: Any) -> None:
        if frame.f_code is target_code and event == "return":
            for key in ("expand_budget", "num_generation_tokens"):
                if key in frame.f_locals:
                    captured[key] = int(frame.f_locals[key])
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(profiler)
    try:
        result = callback()
    finally:
        sys.setprofile(previous)
    if set(captured) != {"expand_budget", "num_generation_tokens"}:
        raise RuntimeError("DreamOn source counter profile did not capture generation locals")
    return result, captured


def generation_target_code(method: Any) -> Any:
    target = inspect.unwrap(method)
    code = getattr(target, "__code__", None)
    if code is None:
        raise RuntimeError("DreamOn source generation method has no inspectable Python code")
    return code


def movement_counts(
    *, initial_length: int, initial_expand_budget: int, captured: Mapping[str, int]
) -> tuple[int, int]:
    expansions = int(initial_expand_budget) - int(captured["expand_budget"])
    final_length = int(captured["num_generation_tokens"])
    contractions = int(initial_length) + expansions - final_length
    if expansions < 0 or contractions < 0:
        raise RuntimeError("DreamOn expansion/contraction counters are inconsistent")
    return expansions, contractions


class CountingModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.forward_calls = 0
        self.token_forwards = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        input_ids = args[0] if args else kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("DreamOn counting wrapper did not receive input_ids")
        self.forward_calls += 1
        self.token_forwards += int(input_ids.numel())
        return self.model(*args, **kwargs)


class OfficialHFTokenizerWrapper:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.mask_id = tokenizer.mask_token_id
        self.expand_id = 151667

    def encode(self, value: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return [self.bos_id] * add_bos + self.tokenizer.encode(value) + [self.eos_id] * add_eos

    def decode(self, tokens: Sequence[int], **kwargs: Any) -> str:
        return self.tokenizer.decode(tokens, **kwargs)


def source_provenance(source_root: Path) -> dict[str, Any]:
    if git_head(source_root) != SOURCE_REVISION:
        raise RuntimeError("DreamOn source checkout is not pinned")
    generator = source_root / "eval" / "generator.py"
    evaluate = source_root / "eval" / "evaluate.py"
    license_path = source_root / "LICENSE"
    if sha256(generator) != SOURCE_GENERATOR_SHA256:
        raise RuntimeError("DreamOn generator source hash mismatch")
    if sha256(evaluate) != SOURCE_EVALUATE_SHA256:
        raise RuntimeError("DreamOn evaluator source hash mismatch")
    if sha256(license_path) != SOURCE_LICENSE_SHA256:
        raise RuntimeError("DreamOn source license hash mismatch")
    generator_source = generator.read_text(encoding="utf-8")
    if generator_source.count("OmegaConf") != 1 or "from omegaconf import OmegaConf" not in generator_source:
        raise RuntimeError("DreamOn generator OmegaConf import-only compatibility assumption changed")
    return {
        "source_repository": "DreamLM/DreamOn",
        "source_revision": SOURCE_REVISION,
        "source_generator_sha256": SOURCE_GENERATOR_SHA256,
        "source_evaluate_sha256": SOURCE_EVALUATE_SHA256,
        "source_license": "Apache-2.0",
        "source_license_sha256": SOURCE_LICENSE_SHA256,
        "omegaconf_compatibility": "import-only shim; pinned generator never references OmegaConf after import",
    }


def model_provenance(snapshot: Path, model_profile: str = "dreamon") -> dict[str, Any]:
    profile = MODEL_PROFILES[model_profile]
    if snapshot.name != profile["revision"]:
        raise RuntimeError(f"{model_profile} cached checkpoint revision mismatch")
    if sha256(snapshot / "config.json") != profile["config_sha256"]:
        raise RuntimeError(f"{model_profile} checkpoint config hash mismatch")
    if sha256(snapshot / "model.safetensors.index.json") != MODEL_INDEX_SHA256:
        raise RuntimeError(f"{model_profile} checkpoint weight-index hash mismatch")
    if model_profile == "dreamon" and "license: apache-2.0" not in (
        snapshot / "README.md"
    ).read_text(encoding="utf-8").lower():
        raise RuntimeError("DreamOn checkpoint Apache-2.0 metadata is missing")
    return {
        "model_id": profile["id"],
        "model_revision": profile["revision"],
        "model_snapshot": str(snapshot),
        "config_sha256": profile["config_sha256"],
        "model_index_sha256": MODEL_INDEX_SHA256,
        "license": profile["license"],
    }


def load_source_runtime(source_root: Path) -> tuple[Any, Any]:
    if importlib.util.find_spec("omegaconf") is None and "omegaconf" not in sys.modules:
        compatibility = ModuleType("omegaconf")
        compatibility.OmegaConf = object()  # type: ignore[attr-defined]
        sys.modules["omegaconf"] = compatibility
    source_path = source_root / "eval" / "generator.py"
    spec = importlib.util.spec_from_file_location("dreamon_pinned_generator_8a0a549", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned DreamOn generator module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = getattr(module, "MDMGenerator", None)
    if generator is None:
        raise RuntimeError("pinned DreamOn generator module lacks MDMGenerator")
    target_code = generation_target_code(generator.batch_generate_with_expand_as_token)
    if Path(target_code.co_filename).resolve() != source_path.resolve():
        raise RuntimeError("DreamOn unwrapped generation code is not the pinned source file")
    return generator, OfficialHFTokenizerWrapper


def canonical_rows(path: Path, full_expected: set[str], arm: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    if "" in counts or any(count != 1 for count in counts.values()):
        raise RuntimeError("DreamOn canonical raw has blank or duplicate keys")
    if not set(counts) <= full_expected:
        raise RuntimeError("DreamOn canonical raw contains keys outside the immutable population")
    if any(str(row.get("arm")) != arm or str(row.get("status")) != "ok" for row in rows):
        raise RuntimeError("DreamOn canonical raw must contain only successful rows for one arm")
    return rows


def audit_rows(rows: Sequence[Mapping[str, Any]], expected: set[str], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("candidate_key")) in expected]
    counts = Counter(str(row.get("candidate_key") or "") for row in selected)
    observed = set(counts)
    accounting_errors = 0
    length_errors = 0
    for row in selected:
        metrics = row.get("metrics") or {}
        total = int(metrics.get("total_forward_calls", -1))
        search = int(metrics.get("search_forward_calls", -1))
        decode = int(metrics.get("decode_forward_calls", -1))
        if total != search + decode:
            accounting_errors += 1
        initial = int(metrics.get("initial_selected_length", -1))
        final = int(metrics.get("final_generated_length", -1))
        expansion = int(metrics.get("expansion_moves", -1))
        contraction = int(metrics.get("contraction_moves", -1))
        if final != initial + expansion - contraction:
            length_errors += 1
    return {
        "passed": (
            observed == expected
            and all(count == 1 for count in counts.values())
            and not accounting_errors
            and not length_errors
            and all(str(row.get("arm")) == arm for row in selected)
        ),
        "arm": arm,
        "expected_count": len(expected),
        "observed_count": len(selected),
        "missing_count": len(expected - observed),
        "extra_count": len(observed - expected),
        "duplicate_count": sum(count > 1 for count in counts.values()),
        "error_count": 0,
        "forward_accounting_error_count": accounting_errors,
        "length_accounting_error_count": length_errors,
    }


def decode_one(
    *,
    safe: Mapping[str, str],
    candidate_key: str,
    seed_key: str | None = None,
    global_seed: int,
    generator: Any,
    tokenizer: Any,
    counting_model: CountingModel,
) -> dict[str, Any]:
    row_seed = derive_row_seed(global_seed, seed_key or candidate_key)
    torch.manual_seed(row_seed)
    torch.cuda.manual_seed_all(row_seed)
    prefix_ids = tokenizer.encode(str(safe["prefix"]), add_bos=True, add_eos=False)
    prefix_len = len(prefix_ids)
    prompt = [
        *prefix_ids,
        *([int(tokenizer.mask_id)] * int(generator.min_gen_len)),
        *tokenizer.encode(str(safe["suffix"]), add_bos=False, add_eos=True),
    ]
    prompt = prompt[-int(generator.max_prompt_len) :]
    input_ids = torch.tensor([prompt], dtype=torch.long, device=str(generator.device))
    before_forwards = counting_model.forward_calls
    before_tokens = counting_model.token_forwards
    method = generator.batch_generate_with_expand_as_token
    result, captured = call_with_generation_profile(
        lambda: method(input_ids), generation_target_code(method)
    )
    response, response_length = result
    response_length = int(response_length)
    if response_length != int(captured["num_generation_tokens"]):
        raise RuntimeError("DreamOn returned length differs from captured source counter")
    expansions, contractions = movement_counts(
        initial_length=int(generator.min_gen_len),
        initial_expand_budget=int(generator.expand_budget),
        captured=captured,
    )
    completion_ids = response[0, prefix_len : prefix_len + response_length]
    remaining_masks = int((completion_ids == int(tokenizer.mask_id)).sum().item())
    completion = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)
    total_forwards = counting_model.forward_calls - before_forwards
    token_forwards = counting_model.token_forwards - before_tokens
    if total_forwards <= 0 or token_forwards <= 0:
        raise RuntimeError("DreamOn source generation produced no auditable model forwards")
    if total_forwards >= int(generator.steps):
        termination = "max_steps_reached"
    elif remaining_masks == 0:
        termination = "all_middle_positions_filled"
    else:
        termination = "source_loop_ended_with_remaining_masks"
    return {
        "completion": completion,
        "row_seed": row_seed,
        "initial_selected_length": int(generator.min_gen_len),
        "final_generated_length": response_length,
        "final_remaining_masks": remaining_masks,
        "expansion_moves": expansions,
        "contraction_moves": contractions,
        "termination_reason": termination,
        "search_forward_calls": 0,
        "decode_forward_calls": total_forwards,
        "total_forward_calls": total_forwards,
        "token_forwards": token_forwards,
        "source_counter_capture": "python_profile_return_locals",
    }


def run(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).resolve()
    evaluator_root = Path(args.evaluator_root).resolve()
    model_snapshot = Path(args.model_snapshot).resolve()
    manifest_path = Path(args.manifest_jsonl).resolve()
    full_manifest_path = Path(args.full_manifest_jsonl).resolve()
    summary_path = Path(args.manifest_summary_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    min_gen_len = int(args.min_gen_len)
    model_profile = str(args.model_profile)
    dataset_profile = str(args.dataset_profile)
    dataset_name, full_count = DATASETS[dataset_profile]
    arm = arm_name(min_gen_len, model_profile)
    config = protocol_config(min_gen_len, model_profile)
    mode = str(args.mode)
    expected_mode_count = 12 if mode == "smoke" else full_count
    manifest = read_jsonl(manifest_path)
    full_manifest = read_jsonl(full_manifest_path)
    if len(manifest) != expected_mode_count:
        raise RuntimeError(f"DreamOn {mode} requires exactly {expected_mode_count} manifest rows")
    if len(full_manifest) != full_count:
        raise RuntimeError(f"DreamOn {dataset_profile} full manifest must contain {full_count} rows")
    if int(args.progress_every) <= 0:
        raise ValueError("--progress-every must be positive")
    population = certify_population(
        dataset_name=dataset_name,
        manifest_path=manifest_path,
        full_manifest_path=full_manifest_path,
        summary_path=summary_path,
        evaluator_root=evaluator_root,
    )
    source_info = source_provenance(source_root)
    model_info = model_provenance(model_snapshot, model_profile)
    source_rows = load_dataset_rows(evaluator_root, dataset_name)
    verify_manifest_source_rows(manifest, source_rows)
    if args.preflight_only:
        _, tokenizer_wrapper = load_source_runtime(source_root)
        ensure_modeling_rope_utils_available()
        from transformers import AutoTokenizer

        tokenizer = tokenizer_wrapper(
            AutoTokenizer.from_pretrained(model_snapshot, trust_remote_code=True, local_files_only=True)
        )
        print(
            json.dumps(
                {
                    "population": population,
                    "source": source_info,
                    "model": model_info,
                    "protocol": config,
                    "tokenizer": {
                        "bos_id": tokenizer.bos_id,
                        "eos_id": tokenizer.eos_id,
                        "mask_id": tokenizer.mask_id,
                        "expand_id": tokenizer.expand_id,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{arm}_raw.jsonl"
    failure_path = output_dir / f"{arm}_failure_journal.jsonl"
    progress_path = output_dir / f"{arm}_{mode}_progress.json"
    run_manifest_path = output_dir / f"{arm}_{mode}_run_manifest.json"
    full_expected = expected_keys(full_manifest, arm, model_profile, dataset_profile)
    expected = expected_keys(manifest, arm, model_profile, dataset_profile)
    existing = canonical_rows(raw_path, full_expected, arm)
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
            **source_info,
            **model_info,
            "protocol": config,
            "dataset": dataset_name,
            "dataset_profile": dataset_profile,
            "manifest": str(manifest_path),
            "full_manifest": str(full_manifest_path),
            "implementation_label": (
                (
                    "DreamOn official-source single-H200 reproduction"
                    if dataset_profile == "singleline"
                    else "DreamOn official-source MultiLine reproduction via benchmark-only adapter"
                )
                if model_profile == "dreamon"
                else "DreamCoder Fixed4/8/16/32/64 under DreamOn sampling/decoder"
            ),
            "model_profile": model_profile,
            "topology_boundary": (
                (
                    "single H200; not exact official 8-GPU topology reproduction"
                    if dataset_profile == "singleline"
                    else "single H200; benchmark-only loader/manifest adapter; per-example source algorithm unchanged"
                )
                if model_profile == "dreamon"
                else "single-H200 common-protocol local fixed-canvas control"
            ),
            "seed_policy": (
                "SHA256(global_seed, candidate_key) deterministic per candidate"
                if model_profile == "dreamon"
                else "SHA256(global_seed, corresponding DreamOn dynamic candidate key)"
            ),
            "randomness_boundary": (
                "same temperature/top-p sampling distribution; not the unreleased unseeded official random stream"
                if model_profile == "dreamon"
                else "same row seed as the corresponding DreamOn dynamic arm; same sampling and source decoder"
            ),
            "fixed_control_semantics": (
                None
                if model_profile == "dreamon"
                else "min_gen_len=max_gen_len; source EOS contraction remains enabled"
            ),
            "benchmark_adapter_boundary": (
                None
                if dataset_profile == "singleline"
                else "only dataset population, manifest, candidate namespace, and evaluator input row change"
            ),
            "resume_contract": "append-only canonical raw; exact successful keys skipped",
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
    generator_class, tokenizer_wrapper = load_source_runtime(source_root)
    ensure_modeling_rope_utils_available()
    from transformers import AutoModel, AutoTokenizer

    tokenizer = tokenizer_wrapper(
        AutoTokenizer.from_pretrained(model_snapshot, trust_remote_code=True, local_files_only=True)
    )
    base_model = AutoModel.from_pretrained(
        model_snapshot,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    counting_model = CountingModel(base_model).eval()
    generator = generator_class(SimpleNamespace(**config), counting_model, tokenizer)
    manifest_by_source = {int(row["source_row_id"]): row for row in manifest}
    torch.cuda.reset_peak_memory_stats()
    for source_row_id in sorted(manifest_by_source):
        item = manifest_by_source[source_row_id]
        key = candidate_key(model_profile, source_row_id, arm, dataset_profile)
        if key in completed:
            continue
        source = source_rows[source_row_id]
        safe = selection_view(source)
        if str(safe["task_id"]) != str(item["task_id"]):
            raise RuntimeError(f"DreamOn {dataset_profile} manifest/source task id mismatch")
        row_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            decoded = decode_one(
                safe=safe,
                candidate_key=key,
                seed_key=(
                    None
                    if model_profile == "dreamon"
                    else paired_seed_key(source_row_id, min_gen_len, dataset_profile)
                ),
                global_seed=int(args.seed),
                generator=generator,
                tokenizer=tokenizer,
                counting_model=counting_model,
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
                    "config": config,
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
            failed["status"] = "failed_stop"
            atomic_write_json(progress_path, failed)
            atomic_write_json(
                run_manifest_path,
                {**json.loads(run_manifest_path.read_text()), **failed, "evaluator": evaluator_info},
            )
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
    rows = canonical_rows(raw_path, full_expected, arm)
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
    final.update(
        {
            "status": "completed" if audit["passed"] and failure_count == 0 else "audit_failed",
            "new_rows_written": len(completed) - starting_completed,
            "final_audit": audit,
        }
    )
    atomic_write_json(progress_path, final)
    atomic_write_json(
        run_manifest_path,
        {**json.loads(run_manifest_path.read_text()), **final, "evaluator": evaluator_info},
    )
    return 0 if audit["passed"] and failure_count == 0 else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--source-root", required=True)
    root.add_argument("--evaluator-root", required=True)
    root.add_argument("--model-snapshot", required=True)
    root.add_argument("--model-profile", choices=tuple(MODEL_PROFILES), default="dreamon")
    root.add_argument("--dataset-profile", choices=tuple(DATASETS), default="singleline")
    root.add_argument("--manifest-jsonl", required=True)
    root.add_argument("--full-manifest-jsonl", required=True)
    root.add_argument("--manifest-summary-json", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--min-gen-len", type=int, choices=INITIAL_LENGTHS, required=True)
    root.add_argument("--mode", choices=("smoke", "full"), required=True)
    root.add_argument("--seed", type=int, default=42)
    root.add_argument("--progress-every", type=int, default=1)
    root.add_argument("--preflight-only", action="store_true")
    return root


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
