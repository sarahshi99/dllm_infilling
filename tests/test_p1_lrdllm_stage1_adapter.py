from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.p1_lrdllm_stage1_adapter import (
    ARM,
    EXPECTED_PROBE_FORWARDS,
    FULL_COUNT,
    FULL_MANIFEST_SHA256,
    SMOKE_MANIFEST_SHA256,
    ForwardLedgerModel,
    audit_rows,
    candidate_key,
    canonical_success_rows,
    expected_keys,
    fixed32_equivalence_contract,
    fixed_decode_kwargs,
    run,
    run_fixed_canvas_decode,
)
from experiments.p1_official_cal_source_audit import OFFICIAL_FIXED32_CONFIG, read_jsonl, sha256


REPO = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = REPO / "analysis_outputs" / "baseline_manifests_20260731_v1"


class InnerModel:
    device = torch.device("cpu")

    def __call__(self, input_ids, **kwargs):
        return input_ids


class LrDllmStage1AdapterTest(unittest.TestCase):
    def test_candidate_key_resume_dedup_and_success_only_raw(self) -> None:
        manifest = [{"source_row_id": 7}]
        key = candidate_key(7)
        self.assertEqual(key, f"cal_singleline_rest_source_row=7|arm={ARM}")
        self.assertEqual(expected_keys(manifest), {key})
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            raw.write_text(json.dumps({"candidate_key": key, "status": "ok"}) + "\n", encoding="utf-8")
            self.assertEqual(len(canonical_success_rows(raw)), 1)
            raw.write_text(
                "\n".join(
                    [
                        json.dumps({"candidate_key": key, "status": "ok"}),
                        json.dumps({"candidate_key": key, "status": "ok"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                canonical_success_rows(raw)
            raw.write_text(json.dumps({"candidate_key": key, "status": "error"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "status=ok"):
                canonical_success_rows(raw)

    def test_forward_and_token_ledger_conserve_actual_calls(self) -> None:
        ledger = ForwardLedgerModel(InnerModel())
        ledger(torch.ones((2, 3), dtype=torch.long))
        ledger(input_ids=torch.ones((1, 4), dtype=torch.long))
        self.assertEqual(ledger.forward_input_sequence_lengths, [3, 4])
        self.assertEqual(ledger.forward_token_counts, [6, 4])
        self.assertEqual(sum(ledger.forward_token_counts), 10)

    def test_l32_fixed_decode_mock_argument_capture_matches_official_fixed32(self) -> None:
        captured = {}

        def generate(model, **kwargs):
            captured.update(kwargs)
            length = int(kwargs["gen_length"])
            total = kwargs["prefix_ids"].shape[1] + length + kwargs["suffix_ids"].shape[1]
            return torch.zeros((1, total), dtype=torch.long), 0

        prefix_ids = torch.tensor([[1, 2]], dtype=torch.long)
        suffix_ids = torch.tensor([[3]], dtype=torch.long)
        prefix_mask = torch.ones_like(prefix_ids)
        suffix_mask = torch.ones_like(suffix_ids)
        output, search_forwards = run_fixed_canvas_decode(
            generate=generate,
            model=object(),
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            attention_mask=prefix_mask,
            suffix_attention_mask=suffix_mask,
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
        self.assertIs(captured["prefix_ids"], prefix_ids)
        self.assertIs(captured["suffix_ids"], suffix_ids)
        self.assertEqual(search_forwards, 0)
        self.assertEqual(output.shape[1], 35)
        self.assertTrue(fixed32_equivalence_contract()["passed"])

    def test_other_lengths_keep_same_fixed_template_and_only_change_canvas_source(self) -> None:
        fixed32 = fixed_decode_kwargs(32)
        fixed8 = fixed_decode_kwargs(8)
        changed = {key for key in fixed32 if fixed32[key] != fixed8[key]}
        self.assertEqual(changed, {"gen_length"})
        self.assertIsNone(fixed8["steps"])
        self.assertIsNone(fixed8["block_length"])
        self.assertEqual(fixed8["dstep"], -1)
        self.assertFalse(fixed8["use_bias"])

    def test_audit_requires_probe_eight_and_forward_token_conservation(self) -> None:
        key = candidate_key(4)
        row = {
            "candidate_key": key,
            "arm": ARM,
            "status": "ok",
            "passed": True,
            "stage1": {"selected_length": 8, "finite_check_passed": True},
            "metrics": {
                "probe_forwards": EXPECTED_PROBE_FORWARDS,
                "formal_decode_forwards": 8,
                "total_forwards": 16,
                "probe_token_forwards": 100,
                "formal_decode_token_forwards": 200,
                "total_token_forwards": 300,
            },
        }
        self.assertTrue(audit_rows([row], {key})["passed"])
        row["metrics"]["total_token_forwards"] = 301
        self.assertFalse(audit_rows([row], {key})["passed"])

    def test_immutable_smoke_and_full_manifest_hashes_and_counts(self) -> None:
        smoke = MANIFEST_ROOT / "cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl"
        full = MANIFEST_ROOT / "cal_singleline_rest_nonfrozen_manifest.jsonl"
        self.assertEqual(sha256(smoke), SMOKE_MANIFEST_SHA256)
        self.assertEqual(sha256(full), FULL_MANIFEST_SHA256)
        self.assertEqual(len(read_jsonl(smoke)), 12)
        self.assertEqual(len(read_jsonl(full)), FULL_COUNT)

    def test_preflight_only_full_mode_never_loads_runtime_or_executes_rows(self) -> None:
        manifest = [{"source_row_id": index} for index in range(FULL_COUNT)]
        args = argparse.Namespace(
            official_cal_root="/tmp/cal",
            humaneval_root="/tmp/eval",
            current_dataset="/tmp/eval/data/HumanEval-SingleLineInfilling.jsonl.gz",
            manifest_jsonl="/tmp/full.jsonl",
            full_manifest_jsonl="/tmp/full.jsonl",
            output_dir="/tmp/unused-output",
            analysis_output_dir=None,
            arm=ARM,
            max_probe_length=128,
            seed=42,
            smoke_cases=12,
            progress_every=1,
            preflight_only=True,
            auto_full=True,
        )
        preflight = {
            "passed": True,
            "mode": "full",
            "manifest": manifest,
            "manifest_sha256": FULL_MANIFEST_SHA256,
            "full_manifest_sha256": FULL_MANIFEST_SHA256,
            "full_manifest_row_count": FULL_COUNT,
            "frozen_test_status": "sealed",
            "frozen_test_evaluation_count": 0,
        }
        with (
            patch("experiments.p1_lrdllm_stage1_adapter.git_head", side_effect=["cal", "eval"]),
            patch("experiments.p1_lrdllm_stage1_adapter.EXPECTED_CAL_COMMIT", "cal"),
            patch("experiments.p1_lrdllm_stage1_adapter.EXPECTED_HUMANEVAL_COMMIT", "eval"),
            patch("experiments.p1_lrdllm_stage1_adapter.preflight_manifests", return_value=preflight),
            patch(
                "experiments.p1_lrdllm_stage1_adapter.load_official_runtime",
                side_effect=AssertionError("preflight-only must not load runtime"),
            ),
        ):
            self.assertEqual(run(args), 0)


if __name__ == "__main__":
    unittest.main()
