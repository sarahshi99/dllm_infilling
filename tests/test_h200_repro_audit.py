from __future__ import annotations

import unittest

from experiments.action_ceiling.h200_repro_audit import classify_verdict, compare_pair, generated_code_hash


def row(task_id: str, passed: bool, code: str, oracle_len: int = 8, selected_len: int = 8) -> dict:
    return {
        "task_id": task_id,
        "code": code,
        "metrics": {
            "passed": passed,
            "oracle_mask_length": oracle_len,
            "selected_mask_length": selected_len,
        },
        "verification": {
            "tier1_parse_compile": {"passed": passed, "error_type": None if passed else "SyntaxError"}
        },
    }


class H200ReproAuditTest(unittest.TestCase):
    def test_generated_code_hash_is_stable(self) -> None:
        first = generated_code_hash({"code": "def f():\n    return 1\n"})
        second = generated_code_hash({"code": "def f():\n    return 1\n"})
        self.assertEqual(first, second)
        self.assertIsNone(generated_code_hash({"code": None}))

    def test_compare_pair_counts_wins_losses_and_agreement(self) -> None:
        old_rows = [
            row("a", True, "same"),
            row("b", False, "old-b"),
            row("c", True, "old-c"),
            row("d", False, "same-d"),
        ]
        h200_rows = [
            row("a", True, "same"),
            row("b", True, "new-b"),
            row("c", False, "new-c"),
            row("d", False, "same-d"),
        ]
        comparison = compare_pair("toy", old_rows, h200_rows)
        self.assertEqual(comparison["h200_wins"], 1)
        self.assertEqual(comparison["h200_losses"], 1)
        self.assertEqual(comparison["tie_pass"], 1)
        self.assertEqual(comparison["tie_fail"], 1)
        self.assertEqual(comparison["pass_delta_h200_minus_old"], 0)
        self.assertAlmostEqual(comparison["outcome_agreement_rate"], 0.5)
        self.assertAlmostEqual(comparison["candidate_hash_agreement_rate"], 0.5)

    def test_classify_verdict_boundaries(self) -> None:
        confirmed = [{"run": "control", "pass_delta_h200_minus_old": 0, "outcome_agreement_rate": 1.0}]
        self.assertEqual(classify_verdict(confirmed), "h200_reproduction_confirmed")

        minor = [{"run": "control", "pass_delta_h200_minus_old": 1, "outcome_agreement_rate": 0.995}]
        self.assertEqual(classify_verdict(minor), "h200_minor_candidate_drift_same_claims")

        material = [{"run": "v6", "pass_delta_h200_minus_old": -2, "outcome_agreement_rate": 0.99}]
        self.assertEqual(classify_verdict(material), "h200_material_outcome_drift")


if __name__ == "__main__":
    unittest.main()
