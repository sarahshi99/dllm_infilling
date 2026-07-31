from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.lrdllm_dreamcoder_adapter import (
    audit_rows,
    build_canvas_ids,
    enable_pinned_evaluator_source,
    mean_negative_entropy_from_logits,
    progress_payload,
    sample_top_p_torch,
    selection_view,
    verify_manifest_source_rows,
)


class LrDllmDreamCoderAdapterTest(unittest.TestCase):
    def test_selection_view_strips_evaluator_and_reference_fields(self) -> None:
        source = {
            "task_id": "SingleLineInfilling/HumanEval/0/L0",
            "prompt": "prefix",
            "suffix": "suffix",
            "entry_point": "f",
            "canonical_solution": "secret",
            "test": "secret tests",
        }
        self.assertEqual(set(selection_view(source)), {"task_id", "prefix", "suffix", "entry_point"})

    def test_canvas_is_bos_prefix_committed_masks_suffix_eos(self) -> None:
        canvas, start, end = build_canvas_ids([10], [11, 12], [13], 2, {"bos": 1, "eos": 2, "pad": 2, "mask": 3})
        self.assertEqual(canvas, [1, 10, 11, 12, 3, 3, 13, 2])
        self.assertEqual((start, end), (4, 6))

    def test_manifest_source_verification_uses_safe_fingerprint(self) -> None:
        source = {
            "task_id": "SingleLineInfilling/HumanEval/0/L0",
            "prompt": "prefix",
            "suffix": "suffix",
            "entry_point": "f",
            "canonical_solution": "secret",
            "test": "secret tests",
        }
        from analysis.build_lrdllm_common_manifests import row_fingerprint

        manifest = [{"source_row_id": 0, "task_id": source["task_id"], "safe_row_sha256": row_fingerprint(source)}]
        verify_manifest_source_rows(manifest, [source])
        manifest[0]["safe_row_sha256"] = "bad"
        with self.assertRaises(RuntimeError):
            verify_manifest_source_rows(manifest, [source])

    def test_negative_entropy_uses_full_distribution(self) -> None:
        logits = torch.tensor([[[0.0, 0.0], [100.0, -100.0]]])
        self.assertAlmostEqual(mean_negative_entropy_from_logits(logits), -0.5 * torch.log(torch.tensor(2.0)).item(), places=6)

    def test_torch_top_p_is_seeded_and_excludes_special(self) -> None:
        logits = torch.tensor([10.0, 9.0, 8.0])
        first = sample_top_p_torch(logits, generator=torch.Generator().manual_seed(7), forbidden_token_ids={0})
        second = sample_top_p_torch(logits, generator=torch.Generator().manual_seed(7), forbidden_token_ids={0})
        self.assertEqual(first, 1)
        self.assertEqual(first, second)

    def test_audit_requires_exact_keys_and_forward_partition(self) -> None:
        rows = [{"candidate_key": "a", "arm": "lrdllm_primary", "metrics": {"search_forward_calls": 2, "decode_forward_calls": 3, "total_forward_calls": 5}}]
        self.assertTrue(audit_rows(rows, {"a"}, "lrdllm_primary")["passed"])
        rows[0]["metrics"]["total_forward_calls"] = 4
        self.assertFalse(audit_rows(rows, {"a"}, "lrdllm_primary")["passed"])

    def test_evaluator_overlay_and_progress_are_deterministic(self) -> None:
        source = "before\n#                     exec(check_program, exec_globals)\nafter\n"
        self.assertNotIn("#                     exec", enable_pinned_evaluator_source(source))
        with tempfile.TemporaryDirectory() as _directory:
            progress = progress_payload(
                arm="lrdllm_primary",
                mode="technical-smoke",
                expected_count=12,
                completed_count=12,
                starting_completed_count=12,
                failure_journal_count=0,
                started=0.0,
            )
        self.assertEqual(progress["missing_count"], 0)
        self.assertEqual(progress["eta_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
