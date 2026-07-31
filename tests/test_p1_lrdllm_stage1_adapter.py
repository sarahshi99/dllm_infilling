from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.p1_lrdllm_stage1_adapter import (
    ARM,
    EXPECTED_PROBE_FORWARDS,
    EXPECTED_PROBE_GRID,
    FULL_COUNT,
    FULL_MANIFEST_SHA256,
    SEARCH_FORWARDS_ALIAS_SEMANTICS,
    SMOKE_MANIFEST_SHA256,
    ForwardLedgerModel,
    _validate_manifest_fields,
    audit_probe_only_rows,
    audit_rows,
    candidate_key,
    canonical_success_rows,
    expected_keys,
    fixed32_equivalence_contract,
    fixed_decode_kwargs,
    probe_only_one,
    run,
    run_fixed_canvas_decode,
    validate_official_fixed32_control,
)
from experiments.p1_lrdllm_stage1_audit import (
    FIXED_DECODER_SOURCE_SHA256,
    MODEL_ARTIFACT_SET_SHA256,
    MODEL_REVISION,
    ExistingFailureJournalError,
    FullGateError,
    ModelArtifactProvenanceError,
    audit_fixed_decoder_source,
    audit_model_artifacts,
    hash_mapping,
    require_no_failure_journal,
    validate_full_gate,
)
from experiments.p1_official_cal_source_audit import (
    EXPECTED_CAL_COMMIT,
    EXPECTED_HUMANEVAL_COMMIT,
    OFFICIAL_FIXED32_CONFIG,
    read_jsonl,
    sha256,
)
from expvision_dllm_clean.lrdllm_stage1 import FIT_SCOPE, PAPER_ID, PAPER_VERSION, STAGE


REPO = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = REPO / "analysis_outputs" / "baseline_manifests_20260731_v1"
PINNED_LLADA_CAL = Path(
    "/home/shx/.cache/dllm_infilling/calibrated_adaptive_length-741e8418/llada_cal/llada_cal.py"
)


def provenance() -> dict[str, str]:
    return {
        "implementation_commit": "code-commit",
        "fixed_decoder_source_sha256": FIXED_DECODER_SOURCE_SHA256,
        "model_revision": MODEL_REVISION,
        "model_artifact_set_sha256": MODEL_ARTIFACT_SET_SHA256,
        "evaluator_execution_source_sha256": "eval-source",
        "evaluator_enabled_execution_source_sha256": "eval-enabled",
        "evaluator_driver_source_sha256": "eval-driver",
        "current_dataset_sha256": "dataset",
        "smoke_manifest_sha256": SMOKE_MANIFEST_SHA256,
        "full_manifest_sha256": FULL_MANIFEST_SHA256,
    }


def valid_stage1(selected_length: int = 8) -> dict:
    entropy = {str(length): -1.0 + 0.01 * index for index, length in enumerate(EXPECTED_PROBE_GRID)}
    logs = {str(length): math.log(length) for length in EXPECTED_PROBE_GRID}
    cl = {str(length): -2.0 for length in EXPECTED_PROBE_GRID}
    cl[str(selected_length)] = -1.0
    return {
        "paper_id": PAPER_ID,
        "paper_version": PAPER_VERSION,
        "stage": STAGE,
        "fit_scope": FIT_SCOPE,
        "probe_lengths": list(EXPECTED_PROBE_GRID),
        "probe_forward_count": 8,
        "probe_token_forwards": 100,
        "average_negative_entropy_by_length": entropy,
        "log_length_by_length": logs,
        "k_hat": 0.1,
        "intercept": -1.0,
        "fit_r2": 0.9,
        "cl_score_by_length": cl,
        "selected_length": selected_length,
        "finite_check_passed": True,
    }


def valid_row(source_row_id: int = 4) -> dict:
    key = candidate_key(source_row_id)
    return {
        "candidate_key": key,
        "arm": ARM,
        "status": "ok",
        "passed": True,
        "paper_id": PAPER_ID,
        "paper_version": PAPER_VERSION,
        "stage": STAGE,
        "stage1": valid_stage1(),
        "provenance": provenance(),
        "config": {"fixed_decode": fixed_decode_kwargs(8)},
        "metrics": {
            "selected_length": 8,
            "generated_length": 8,
            "stage1_probe_forwards": 8,
            "length_selection_forwards": 8,
            "probe_forwards": 8,
            "search_forwards": 8,
            "search_forwards_semantics": SEARCH_FORWARDS_ALIAS_SEMANTICS,
            "formal_decode_forwards": 8,
            "total_forwards": 16,
            "stage1_probe_token_forwards": 100,
            "probe_token_forwards": 100,
            "formal_decode_token_forwards": 200,
            "total_token_forwards": 300,
            "fixed_decoder_upstream_search_forwards": 0,
        },
    }


class InnerModel:
    device = torch.device("cpu")

    def __call__(self, input_ids, **kwargs):
        return input_ids


class FakeTokenizer:
    mask_token_id = 126336

    def __call__(self, text, *, add_special_tokens, padding, return_tensors):
        values = [1, 2]
        return {
            "input_ids": torch.tensor([values], dtype=torch.long),
            "attention_mask": torch.ones((1, len(values)), dtype=torch.long),
        }

    def decode(self, *args, **kwargs):
        raise AssertionError("probe-only must not decode a completion")


class ProbeModel:
    device = torch.device("cpu")

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None):
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
        logits[..., 0] = float(input_ids.shape[-1]) / 10.0
        return SimpleNamespace(logits=logits)


class LrDllmStage1AdapterTest(unittest.TestCase):
    def test_candidate_key_resume_dedup_and_success_only_raw(self) -> None:
        manifest = [{"source_row_id": 7}]
        key = candidate_key(7)
        self.assertEqual(key, f"cal_singleline_rest_source_row=7|arm={ARM}")
        self.assertTrue(candidate_key(7, "probe_only").endswith("|mode=probe_only"))
        self.assertEqual(expected_keys(manifest), {key})
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            raw.write_text(json.dumps({"candidate_key": key, "status": "ok"}) + "\n", encoding="utf-8")
            self.assertEqual(len(canonical_success_rows(raw)), 1)
            raw.write_text(
                json.dumps({"candidate_key": key, "status": "ok"})
                + "\n"
                + json.dumps({"candidate_key": key, "status": "ok"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                canonical_success_rows(raw)

    def test_forward_and_token_ledger_conserve_actual_calls(self) -> None:
        ledger = ForwardLedgerModel(InnerModel())
        ledger(torch.ones((2, 3), dtype=torch.long))
        ledger(input_ids=torch.ones((1, 4), dtype=torch.long))
        self.assertEqual(ledger.forward_input_sequence_lengths, [3, 4])
        self.assertEqual(ledger.forward_token_counts, [6, 4])

    def test_l32_fixed_decode_mock_argument_capture_matches_official_fixed32(self) -> None:
        captured = {}

        def generate(model, **kwargs):
            captured.update(kwargs)
            total = kwargs["prefix_ids"].shape[1] + kwargs["gen_length"] + kwargs["suffix_ids"].shape[1]
            return torch.zeros((1, total), dtype=torch.long), 0

        prefix_ids = torch.tensor([[1, 2]], dtype=torch.long)
        suffix_ids = torch.tensor([[3]], dtype=torch.long)
        output, search_forwards = run_fixed_canvas_decode(
            generate=generate,
            model=object(),
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            suffix_attention_mask=torch.ones_like(suffix_ids),
            selected_length=32,
        )
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
        self.assertEqual({key: captured[key] for key in expected}, expected)
        self.assertEqual(search_forwards, 0)
        self.assertEqual(output.shape[1], 35)
        self.assertTrue(fixed32_equivalence_contract()["passed"])

    def test_real_pinned_source_semantic_audit(self) -> None:
        audit = audit_fixed_decoder_source(PINNED_LLADA_CAL)
        self.assertEqual(audit["source_sha256"], FIXED_DECODER_SOURCE_SHA256)
        self.assertEqual(audit["steps_none_resolved_to"], 4)
        self.assertEqual(audit["block_length_none_resolved_to"], 4)
        self.assertEqual(audit["dstep_minus_one_search_forwards"], 0)
        self.assertEqual(audit["output_canvas_length"], 4)

    def test_fixed_decoder_source_hash_mismatch_is_protocol_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            modified = Path(directory) / "llada_cal.py"
            modified.write_text(PINNED_LLADA_CAL.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fixed_decode_protocol_ambiguity"):
                audit_fixed_decoder_source(modified)

    def test_model_artifact_hash_mismatch_fails_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / MODEL_REVISION
            snapshot.mkdir()
            config = snapshot / "config.json"
            index = snapshot / "model.safetensors.index.json"
            config.write_text(json.dumps({"mask_token_id": 126336}), encoding="utf-8")
            index.write_text(json.dumps({"weight_map": {}}), encoding="utf-8")
            expected = {
                "config.json": hashlib.sha256(config.read_bytes()).hexdigest(),
                "model.safetensors.index.json": hashlib.sha256(index.read_bytes()).hexdigest(),
            }
            aggregate = hash_mapping(expected)
            config.write_text(json.dumps({"mask_token_id": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ModelArtifactProvenanceError, "SHA256 mismatch"):
                audit_model_artifacts(
                    hf_home=Path(directory),
                    snapshot_path=snapshot,
                    expected_hashes=expected,
                    expected_aggregate=aggregate,
                )

    def test_manifest_exact_allowlist_rejects_extra_field(self) -> None:
        row = read_jsonl(MANIFEST_ROOT / "cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl")[0]
        _validate_manifest_fields([row], name="valid")
        with self.assertRaisesRegex(RuntimeError, "exact allowlist"):
            _validate_manifest_fields([{**row, "canonical_solution": "leak"}], name="bad")

    def test_corrupted_raw_rows_each_fail_final_audit(self) -> None:
        mutations = {
            "paper_id": lambda row: row.__setitem__("paper_id", "wrong"),
            "paper_version": lambda row: row.__setitem__("paper_version", "v2"),
            "stage": lambda row: row.__setitem__("stage", "stage2"),
            "fit_scope": lambda row: row["stage1"].__setitem__("fit_scope", "upper_half"),
            "probe_grid": lambda row: row["stage1"].__setitem__("probe_lengths", [1, 2]),
            "probe_count": lambda row: row["metrics"].__setitem__("stage1_probe_forwards", 7),
            "selected_generated": lambda row: row["metrics"].__setitem__("generated_length", 16),
            "fixed_search": lambda row: row["metrics"].__setitem__("fixed_decoder_upstream_search_forwards", 1),
            "dstep": lambda row: row["config"]["fixed_decode"].__setitem__("dstep", 4),
            "use_bias": lambda row: row["config"]["fixed_decode"].__setitem__("use_bias", True),
            "entropy_nan": lambda row: row["stage1"]["average_negative_entropy_by_length"].__setitem__("1", float("nan")),
            "k_nan": lambda row: row["stage1"].__setitem__("k_hat", float("nan")),
            "intercept_inf": lambda row: row["stage1"].__setitem__("intercept", float("inf")),
            "r2_nan": lambda row: row["stage1"].__setitem__("fit_r2", float("nan")),
            "cl_nan": lambda row: row["stage1"]["cl_score_by_length"].__setitem__("8", float("nan")),
            "source_hash": lambda row: row["provenance"].__setitem__("fixed_decoder_source_sha256", "wrong"),
            "model_hash": lambda row: row["provenance"].__setitem__("model_artifact_set_sha256", "wrong"),
            "evaluator_hash": lambda row: row["provenance"].__setitem__("evaluator_execution_source_sha256", "wrong"),
            "forward_ledger": lambda row: row["metrics"].__setitem__("total_forwards", 17),
            "token_ledger": lambda row: row["metrics"].__setitem__("total_token_forwards", 301),
        }
        valid = valid_row()
        self.assertTrue(audit_rows([valid], {valid["candidate_key"]}, provenance())["passed"])
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                row = copy.deepcopy(valid)
                mutate(row)
                self.assertFalse(
                    audit_rows([row], {row["candidate_key"]}, provenance())["passed"]
                )

    def test_existing_failure_journal_requires_new_output_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "failure.jsonl"
            journal.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ExistingFailureJournalError, "new output directory"):
                require_no_failure_journal(journal)

    def test_probe_only_has_eight_forwards_no_decode_no_evaluator(self) -> None:
        result = probe_only_one(
            source={"prompt": "p", "suffix": "s"},
            tokenizer=FakeTokenizer(),
            model=ProbeModel(),
            seed=42,
        )
        self.assertEqual(result["metrics"]["stage1_probe_forwards"], 8)
        self.assertEqual(result["metrics"]["formal_decode_forwards"], 0)
        self.assertFalse(result["metrics"]["evaluator_called"])
        self.assertNotIn("completion", result)

    def test_probe_only_raw_audit_rejects_completion_or_evaluator(self) -> None:
        key = candidate_key(4, "probe_only")
        row = valid_row()
        row["candidate_key"] = key
        row["probe_only"] = True
        row.pop("passed")
        row.pop("config")
        row["metrics"]["formal_decode_forwards"] = 0
        row["metrics"]["total_forwards"] = 8
        row["metrics"]["formal_decode_token_forwards"] = 0
        row["metrics"]["total_token_forwards"] = 100
        row["metrics"]["evaluator_called"] = False
        self.assertTrue(audit_probe_only_rows([row], {key}, provenance())["passed"])
        row["completion"] = "forbidden"
        self.assertFalse(audit_probe_only_rows([row], {key}, provenance())["passed"])

    def test_strict_full_gate_requires_resume_noop_and_exact_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = {
                "passed": True,
                "expected_count": 12,
                "observed_count": 12,
                "unique_count": 12,
                "rows_with_expected_probe_forwards": 12,
                "evaluator_completed_count": 12,
                "provenance": provenance(),
            }
            for field in (
                "missing_count", "extra_count", "duplicate_count", "canonical_error_count",
                "wrong_arm_count", "probe_forward_error_count", "forward_accounting_error_count",
                "token_accounting_error_count", "finite_diagnostic_error_count", "semantic_error_count",
                "provenance_error_count", "failure_journal_count",
            ):
                audit[field] = 0
            (root / f"{ARM}_smoke_final_audit.json").write_text(json.dumps(audit), encoding="utf-8")
            progress = {"status": "completed", "resume_noop": True, "new_rows_written": 0}
            (root / f"{ARM}_smoke_progress.json").write_text(json.dumps(progress), encoding="utf-8")
            (root / f"{ARM}_smoke_run_manifest.json").write_text(
                json.dumps({"final_audit": audit}), encoding="utf-8"
            )
            self.assertTrue(
                validate_full_gate(
                    smoke_output_dir=root, arm=ARM, current_provenance=provenance()
                )["passed"]
            )
            progress["resume_noop"] = False
            (root / f"{ARM}_smoke_progress.json").write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaisesRegex(FullGateError, "strict full gate failed"):
                validate_full_gate(
                    smoke_output_dir=root, arm=ARM, current_provenance=provenance()
                )

    def test_full_environment_variable_cannot_bypass_missing_smoke_gate(self) -> None:
        args = argparse.Namespace(
            official_cal_root="/tmp/cal",
            humaneval_root="/tmp/eval",
            current_dataset="/tmp/eval/data/HumanEval-SingleLineInfilling.jsonl.gz",
            manifest_jsonl="/tmp/full.jsonl",
            full_manifest_jsonl="/tmp/full.jsonl",
            output_dir="/tmp/full-output",
            analysis_output_dir=None,
            hf_home="/tmp/hf",
            probe_output_dir="/tmp/probe",
            smoke_output_dir="/tmp/missing-smoke",
            arm=ARM,
            max_probe_length=128,
            seed=42,
            smoke_cases=12,
            progress_every=1,
            preflight_only=False,
            probe_only=False,
            auto_full=True,
        )
        preflight = {
            "passed": True,
            "mode": "full",
            "manifest": [{"source_row_id": index} for index in range(FULL_COUNT)],
            "implementation_tracked_worktree_clean": True,
            "implementation_commit": provenance()["implementation_commit"],
            "current_dataset_sha256": provenance()["current_dataset_sha256"],
            "smoke_manifest_sha256": SMOKE_MANIFEST_SHA256,
            "full_manifest_sha256": FULL_MANIFEST_SHA256,
            "fixed_decoder_source_audit": {
                "source_sha256": FIXED_DECODER_SOURCE_SHA256,
                "steps_none_resolved_to": 4,
                "block_length_none_resolved_to": 4,
                "dstep_minus_one_search_forwards": 0,
                "output_canvas_length": 4,
            },
            "model_provenance": {
                "resolved_revision": MODEL_REVISION,
                "artifact_set_sha256": MODEL_ARTIFACT_SET_SHA256,
                "artifact_sha256": {},
                "mask_token_id_from_config": 126336,
            },
            "evaluator_source_audit": {
                "execution_source_sha256": provenance()["evaluator_execution_source_sha256"],
                "enabled_execution_source_sha256": provenance()["evaluator_enabled_execution_source_sha256"],
                "driver_source_sha256": provenance()["evaluator_driver_source_sha256"],
            },
            "runtime_environment": {},
        }
        with (
            patch.dict(os.environ, {"ALLOW_LRDLLM_STAGE1_FULL": "1"}),
            patch("experiments.p1_lrdllm_stage1_adapter.git_head", side_effect=[EXPECTED_CAL_COMMIT, EXPECTED_HUMANEVAL_COMMIT]),
            patch("experiments.p1_lrdllm_stage1_adapter.preflight_manifests", return_value=preflight),
            patch("experiments.p1_lrdllm_stage1_adapter.validate_probe_only_gate", return_value={"passed": True}),
            patch("experiments.p1_lrdllm_stage1_adapter.validate_full_gate", side_effect=FullGateError("missing smoke")),
            patch("experiments.p1_lrdllm_stage1_adapter.execution_gpu_ecc_gate", side_effect=AssertionError("must not reach GPU gate")),
        ):
            with self.assertRaisesRegex(FullGateError, "missing smoke"):
                run(args)

    def test_full_requires_completed_same_key_official_fixed32_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FullGateError, "official_fixed32 control"):
                validate_official_fixed32_control(
                    output_dir=Path(directory),
                    full_manifest_path=MANIFEST_ROOT
                    / "cal_singleline_rest_nonfrozen_manifest.jsonl",
                )

    def test_immutable_smoke_and_full_manifest_hashes_and_counts(self) -> None:
        smoke = MANIFEST_ROOT / "cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl"
        full = MANIFEST_ROOT / "cal_singleline_rest_nonfrozen_manifest.jsonl"
        self.assertEqual(sha256(smoke), SMOKE_MANIFEST_SHA256)
        self.assertEqual(sha256(full), FULL_MANIFEST_SHA256)
        self.assertEqual(len(read_jsonl(smoke)), 12)
        self.assertEqual(len(read_jsonl(full)), FULL_COUNT)


if __name__ == "__main__":
    unittest.main()
