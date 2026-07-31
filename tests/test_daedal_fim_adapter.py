from __future__ import annotations

import unittest

import torch

from experiments.daedal_fim_adapter import (
    DAEDAL_EVAL_SHA256,
    DAEDAL_SOURCE_SHA256,
    arm_config,
    audit_rows,
    candidate_key,
    extract_middle,
)


class DaedalFimAdapterTest(unittest.TestCase):
    def test_dynamic_and_fixed8_share_decoder_config_except_adaptation(self) -> None:
        dynamic = arm_config("daedal_dynamic")
        fixed = arm_config("daedal_fixed8_control")
        differences = {key for key in dynamic if dynamic[key] != fixed[key]}
        self.assertEqual(differences, {"enable_stage2"})
        self.assertTrue(dynamic["enable_stage2"])
        self.assertFalse(fixed["enable_stage2"])
        self.assertFalse(dynamic["enable_stage1"])
        self.assertEqual(dynamic["initial_gen_length"], 8)

    def test_candidate_keys_are_arm_independent(self) -> None:
        self.assertEqual(candidate_key("singleline", 7), "daedal_fim|dataset=singleline|source_row=7")

    def test_suffix_zero_middle_extraction_is_not_empty(self) -> None:
        output = torch.tensor([1, 2, 3, 4])
        self.assertEqual(extract_middle(output, 2, 0).tolist(), [3, 4])
        self.assertEqual(extract_middle(output, 1, 1).tolist(), [2, 3])

    def test_audit_rejects_accounting_and_remaining_masks(self) -> None:
        row = {
            "candidate_key": "a",
            "arm": "daedal_dynamic",
            "metrics": {"search_forwards": 0, "formal_decode_forwards": 3, "total_forwards": 3, "remaining_mask_tokens": 0},
        }
        self.assertTrue(audit_rows([row], {"a"}, "daedal_dynamic")["passed"])
        row["metrics"]["remaining_mask_tokens"] = 1
        self.assertFalse(audit_rows([row], {"a"}, "daedal_dynamic")["passed"])

    def test_pinned_source_hashes_are_explicit(self) -> None:
        self.assertEqual(len(DAEDAL_SOURCE_SHA256), 64)
        self.assertEqual(len(DAEDAL_EVAL_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
