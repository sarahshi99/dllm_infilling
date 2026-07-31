from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import transformers


MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
EXPECTED_MASK_TOKEN_ID = 126336
FIXED_DECODER_SOURCE_SHA256 = "b1d684040334ea1ffeacd92279cfa98e007ec5c9ce1a39840054e98420989225"
EVALUATOR_EXECUTION_SHA256 = "b96a5b82c45ed0a8a275b1988965b6e0d8fab872a2d4bdc0e26ab8b8fec6d7cb"
EVALUATOR_ENABLED_EXECUTION_SHA256 = "06c25200bd38da7938c70f1115c36215940868cf57fd089a9a0326d29efb2359"
EVALUATOR_DRIVER_SHA256 = "5c2ec64099ce03af215590bf60e20216ecf5fd64907bea70ab0e2da14db4d31c"
MODEL_ARTIFACT_SHA256 = {
    "config.json": "5f99fefe855fdb5100bb6cadb57bdb09fae723ad54811f95c00ecacf29d58a6a",
    "configuration_llada.py": "c947a64d8b735affa69d5adeb416c83ce607c8925270884e4471ae11068e2bfe",
    "generation_config.json": "00b3d1ffda2ca0a45633deb52ca4413dda046e7c94e9b05b73ed4a565b0677b3",
    "model-00001-of-00006.safetensors": "4c0652913997c26851c51d5881918a93b2d502c0eeabcf3fe409800441ebc385",
    "model-00002-of-00006.safetensors": "330b95434544dcdac54531ef01a14fd7dede828974d3ca2c440e443a011d0365",
    "model-00003-of-00006.safetensors": "e7847f941f2872c715af1700834056713aba45cfa66223d9e06000b1936e1abf",
    "model-00004-of-00006.safetensors": "9d5d75b011eae74467ec02db8875b582cce7b02eff143dab78b8530af4bf4dd2",
    "model-00005-of-00006.safetensors": "49e9018af8ae4852eefb2ffe7947df1931b7c4c1dcdd3648b36acd5c90eafd3d",
    "model-00006-of-00006.safetensors": "3a2c91cf7aac23ca84f48d4fdabf58ffd050e9f56b9c94b7962a3d3f5c35e3c6",
    "model.safetensors.index.json": "28b4ec27206e42e7ade630450e6ce618bd197acf34d35120e8e86d2bb910a408",
    "modeling_llada.py": "98bac7e53fef0bb7ca01e3716c11a7f710d183e10dbb9783b88db9dbba2e3766",
    "special_tokens_map.json": "981e83321bd28bee90ad62ce1bd0baa0fcef3540dd29ca35e2116ec5577fef71",
    "tokenizer.json": "ee1ef8e5f6d9493ac25480b7b7337ff5d2c1b946190afff18d12c86ca738ae00",
    "tokenizer_config.json": "6e9f41633217287fcf9a58890efb26e91e905bd6ae2234b534b65e0c36f4dd3c",
}
MODEL_ARTIFACT_SET_SHA256 = "923e056fe742858e8b817406ae942feccb53ace80116187070de81b29901719a"
EVALUATOR_ENABLEMENT_MARKER = "#                     exec(check_program, exec_globals)"
EVALUATOR_ENABLEMENT_REPLACEMENT = "                    exec(check_program, exec_globals)"


class FixedDecodeProtocolAmbiguity(RuntimeError):
    pass


class ModelArtifactProvenanceError(RuntimeError):
    pass


class FullGateError(RuntimeError):
    pass


class ExistingFailureJournalError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_mapping(values: Mapping[str, str]) -> str:
    payload = json.dumps(dict(values), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_code_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def tracked_worktree_is_clean(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def audit_fixed_decoder_source(source_path: Path) -> dict[str, Any]:
    actual_hash = sha256_file(source_path)
    if actual_hash != FIXED_DECODER_SOURCE_SHA256:
        raise FixedDecodeProtocolAmbiguity(
            "fixed_decode_protocol_ambiguity: pinned llada_cal.py SHA256 mismatch"
        )
    spec = importlib.util.spec_from_file_location("_lrdllm_fixed_decoder_source_audit", source_path)
    if spec is None or spec.loader is None:
        raise FixedDecodeProtocolAmbiguity(
            "fixed_decode_protocol_ambiguity: cannot load pinned llada_cal.py"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        transfer_observations: list[dict[str, int]] = []
        original_transfer = module.get_num_transfer_tokens

        def capture_transfer(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
            transfer_observations.append(
                {"block_mask_length": int(mask_index.shape[-1]), "steps_per_block": int(steps)}
            )
            return original_transfer(mask_index, steps)

        module.get_num_transfer_tokens = capture_transfer

        class ControlledModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self.forward_count = 0

            def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Any:
                self.forward_count += 1
                if attention_mask is None or attention_mask.shape != input_ids.shape:
                    raise RuntimeError("controlled fixed decoder audit received a bad attention mask")
                logits = torch.zeros((*input_ids.shape, 126464), dtype=torch.float32)
                logits[..., 0] = 1.0
                return SimpleNamespace(logits=logits)

        model = ControlledModel()
        prefix_ids = torch.tensor([[11]], dtype=torch.long)
        suffix_ids = torch.tensor([[12]], dtype=torch.long)
        prefix_mask = torch.ones_like(prefix_ids)
        suffix_mask = torch.ones_like(suffix_ids)
        selected_length = 4
        output, search_forwards = module.generate(
            model,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            attention_mask=prefix_mask,
            suffix_attention_mask=suffix_mask,
            steps=None,
            gen_length=selected_length,
            block_length=None,
            temperature=0.0,
            cfg_scale=0.0,
            span=1,
            max_gen_length=128,
            dstep=-1,
            use_bias=False,
        )
    except Exception as exc:
        if isinstance(exc, FixedDecodeProtocolAmbiguity):
            raise
        raise FixedDecodeProtocolAmbiguity(
            f"fixed_decode_protocol_ambiguity: controlled source execution failed: {exc}"
        ) from exc

    expected_sequence_length = 1 + selected_length + 1
    semantics_passed = (
        int(search_forwards) == 0
        and model.forward_count == selected_length
        and transfer_observations
        == [{"block_mask_length": selected_length, "steps_per_block": selected_length}]
        and tuple(output.shape) == (1, expected_sequence_length)
        and int(output.shape[1]) - 2 == selected_length
    )
    if not semantics_passed:
        raise FixedDecodeProtocolAmbiguity(
            "fixed_decode_protocol_ambiguity: controlled source semantics differ from frozen fixed path"
        )
    return {
        "status": "passed",
        "source_path": str(source_path),
        "source_sha256": actual_hash,
        "controlled_selected_length": selected_length,
        "steps_none_resolved_to": selected_length,
        "block_length_none_resolved_to": selected_length,
        "dstep_minus_one_search_forwards": int(search_forwards),
        "formal_decode_forwards": model.forward_count,
        "output_canvas_length": int(output.shape[1]) - 2,
        "transfer_observations": transfer_observations,
    }


def audit_model_artifacts(
    *,
    hf_home: Path,
    snapshot_path: Path | None = None,
    expected_revision: str = MODEL_REVISION,
    expected_hashes: Mapping[str, str] = MODEL_ARTIFACT_SHA256,
    expected_aggregate: str = MODEL_ARTIFACT_SET_SHA256,
) -> dict[str, Any]:
    if snapshot_path is None:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                MODEL_PATH,
                revision=expected_revision,
                cache_dir=str(hf_home / "hub"),
                local_files_only=True,
            )
        ).resolve()
    else:
        snapshot_path = snapshot_path.resolve()
    resolved_revision = snapshot_path.name
    if resolved_revision != expected_revision:
        raise ModelArtifactProvenanceError(
            f"resolved model commit mismatch: {resolved_revision} != {expected_revision}"
        )

    actual_hashes: dict[str, str] = {}
    artifact_sizes: dict[str, int] = {}
    for filename, expected_hash in expected_hashes.items():
        path = snapshot_path / filename
        if not path.is_file():
            raise ModelArtifactProvenanceError(f"required model artifact is missing: {filename}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ModelArtifactProvenanceError(
                f"model artifact SHA256 mismatch for {filename}: {actual_hash} != {expected_hash}"
            )
        actual_hashes[filename] = actual_hash
        artifact_sizes[filename] = int(path.stat().st_size)
    aggregate = hash_mapping(actual_hashes)
    if aggregate != expected_aggregate:
        raise ModelArtifactProvenanceError(
            f"model artifact set SHA256 mismatch: {aggregate} != {expected_aggregate}"
        )

    config = json.loads((snapshot_path / "config.json").read_text(encoding="utf-8"))
    if int(config.get("mask_token_id", -1)) != EXPECTED_MASK_TOKEN_ID:
        raise ModelArtifactProvenanceError("model config mask_token_id mismatch")
    index = json.loads(
        (snapshot_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    expected_shards = {
        filename for filename in expected_hashes if filename.endswith(".safetensors")
    }
    indexed_shards = set(index.get("weight_map", {}).values())
    if indexed_shards != expected_shards:
        raise ModelArtifactProvenanceError("weight index shard set mismatch")

    ref_path = snapshot_path.parents[1] / "refs" / "main"
    main_ref = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else None
    return {
        "status": "passed",
        "model_id": MODEL_PATH,
        "requested_revision": expected_revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot_path),
        "main_ref": main_ref,
        "artifact_sha256": actual_hashes,
        "artifact_sizes": artifact_sizes,
        "artifact_set_sha256": aggregate,
        "weight_index_shards": sorted(indexed_shards),
        "mask_token_id_from_config": int(config["mask_token_id"]),
    }


def audit_evaluator_source(humaneval_root: Path) -> dict[str, Any]:
    execution_path = humaneval_root / "human_eval_infilling" / "execution.py"
    driver_path = humaneval_root / "human_eval_infilling" / "evaluate_functional_correctness.py"
    source = execution_path.read_text(encoding="utf-8")
    if source.count(EVALUATOR_ENABLEMENT_MARKER) != 1:
        raise RuntimeError("pinned evaluator enablement marker is absent or ambiguous")
    enabled = source.replace(EVALUATOR_ENABLEMENT_MARKER, EVALUATOR_ENABLEMENT_REPLACEMENT)
    values = {
        "execution_source_sha256": sha256_file(execution_path),
        "enabled_execution_source_sha256": hashlib.sha256(enabled.encode("utf-8")).hexdigest(),
        "driver_source_sha256": sha256_file(driver_path),
    }
    expected = {
        "execution_source_sha256": EVALUATOR_EXECUTION_SHA256,
        "enabled_execution_source_sha256": EVALUATOR_ENABLED_EXECUTION_SHA256,
        "driver_source_sha256": EVALUATOR_DRIVER_SHA256,
    }
    if values != expected:
        raise RuntimeError("pinned evaluator source SHA256 mismatch")
    return {"status": "passed", **values}


def runtime_environment() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def provenance_contract(preflight: Mapping[str, Any]) -> dict[str, Any]:
    model = preflight["model_provenance"]
    source = preflight["fixed_decoder_source_audit"]
    evaluator = preflight["evaluator_source_audit"]
    return {
        "implementation_commit": str(preflight["implementation_commit"]),
        "fixed_decoder_source_sha256": str(source["source_sha256"]),
        "fixed_decoder_semantics": {
            "steps_none_resolved_to": int(source["steps_none_resolved_to"]),
            "block_length_none_resolved_to": int(
                source["block_length_none_resolved_to"]
            ),
            "dstep_minus_one_search_forwards": int(
                source["dstep_minus_one_search_forwards"]
            ),
            "output_canvas_length": int(source["output_canvas_length"]),
        },
        "model_revision": str(model["resolved_revision"]),
        "model_artifact_set_sha256": str(model["artifact_set_sha256"]),
        "model_artifact_sha256": dict(model["artifact_sha256"]),
        "mask_token_id": int(model["mask_token_id_from_config"]),
        "evaluator_execution_source_sha256": str(evaluator["execution_source_sha256"]),
        "evaluator_enabled_execution_source_sha256": str(
            evaluator["enabled_execution_source_sha256"]
        ),
        "evaluator_driver_source_sha256": str(evaluator["driver_source_sha256"]),
        "current_dataset_sha256": str(preflight["current_dataset_sha256"]),
        "smoke_manifest_sha256": str(preflight["smoke_manifest_sha256"]),
        "full_manifest_sha256": str(preflight["full_manifest_sha256"]),
        "runtime_environment": dict(preflight["runtime_environment"]),
    }


def require_no_failure_journal(path: Path) -> None:
    if path.exists():
        raise ExistingFailureJournalError(
            f"existing failure journal forbids reuse of this output version: {path}; use a new output directory"
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FullGateError(f"required gate record is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_probe_only_gate(
    *, smoke_probe_output_dir: Path, arm: str, current_provenance: Mapping[str, Any]
) -> dict[str, Any]:
    require_no_failure_journal(smoke_probe_output_dir / f"{arm}_probe_only_failure_journal.jsonl")
    audit = _read_json(smoke_probe_output_dir / f"{arm}_probe_only_final_audit.json")
    run_manifest = _read_json(smoke_probe_output_dir / f"{arm}_probe_only_run_manifest.json")
    if not (
        audit.get("passed") is True
        and int(audit.get("expected_count", -1)) == 1
        and int(audit.get("observed_count", -1)) == 1
        and int(audit.get("probe_forward_error_count", -1)) == 0
        and int(audit.get("finite_diagnostic_error_count", -1)) == 0
        and int(audit.get("evaluator_completed_count", -1)) == 0
        and int(audit.get("formal_decode_forwards_total", -1)) == 0
        and audit.get("provenance") == dict(current_provenance)
        and (run_manifest.get("final_audit") or {}).get("passed") is True
    ):
        raise FullGateError("probe-only gate is incomplete or incompatible with current provenance")
    return {"passed": True, "audit_path": str(smoke_probe_output_dir), "audit": audit}


def validate_full_gate(
    *, smoke_output_dir: Path, arm: str, current_provenance: Mapping[str, Any]
) -> dict[str, Any]:
    require_no_failure_journal(smoke_output_dir / f"{arm}_failure_journal.jsonl")
    audit = _read_json(smoke_output_dir / f"{arm}_smoke_final_audit.json")
    progress = _read_json(smoke_output_dir / f"{arm}_smoke_progress.json")
    run_manifest = _read_json(smoke_output_dir / f"{arm}_smoke_run_manifest.json")
    zero_fields = (
        "missing_count",
        "extra_count",
        "duplicate_count",
        "canonical_error_count",
        "wrong_arm_count",
        "probe_forward_error_count",
        "forward_accounting_error_count",
        "token_accounting_error_count",
        "finite_diagnostic_error_count",
        "semantic_error_count",
        "provenance_error_count",
        "failure_journal_count",
    )
    if not (
        audit.get("passed") is True
        and int(audit.get("expected_count", -1)) == 12
        and int(audit.get("observed_count", -1)) == 12
        and int(audit.get("unique_count", -1)) == 12
        and all(int(audit.get(field, -1)) == 0 for field in zero_fields)
        and int(audit.get("rows_with_expected_probe_forwards", -1)) == 12
        and int(audit.get("evaluator_completed_count", -1)) == 12
        and audit.get("provenance") == dict(current_provenance)
        and progress.get("status") == "completed"
        and progress.get("resume_noop") is True
        and int(progress.get("new_rows_written", -1)) == 0
        and (run_manifest.get("final_audit") or {}).get("passed") is True
        and (run_manifest.get("final_audit") or {}).get("provenance")
        == dict(current_provenance)
    ):
        raise FullGateError(
            "strict full gate failed: smoke audit/resume/provenance is incomplete or incompatible"
        )
    return {
        "passed": True,
        "smoke_output_dir": str(smoke_output_dir),
        "resume_noop": True,
        "new_rows_written": 0,
        "provenance": dict(current_provenance),
    }


def execution_gpu_ecc_gate() -> dict[str, Any]:
    processes = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if processes:
        raise RuntimeError("blocked_by_existing_gpu_process")
    uncorrectable = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    corrected = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=ecc.errors.corrected.volatile.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if uncorrectable != "0" or corrected != "0":
        raise RuntimeError(
            f"GPU ECC gate failed: corrected={corrected} uncorrectable={uncorrectable}"
        )
    return {
        "passed": True,
        "physical_gpu": 0,
        "existing_compute_process_count": 0,
        "volatile_corrected_ecc": 0,
        "volatile_uncorrectable_ecc": 0,
    }
