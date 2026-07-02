from __future__ import annotations

import unittest

from analysis import proportional_length_widening_audit as audit


def candidate(length: int, score: float, raw: float | None = None) -> dict:
    return {
        "mask_length": length,
        "score": score,
        "adjusted_score": score,
        "raw_score": score if raw is None else raw,
    }


def row(task_id: str, *, selected: int, oracle: int, passed: bool, candidates: list[dict]) -> dict:
    return {
        "task_id": task_id,
        "metrics": {
            "selected_mask_length": selected,
            "oracle_mask_length": oracle,
            "passed": passed,
        },
        "lcal_v3": {
            "base_candidate_scores": candidates,
        },
    }


class ProportionalLengthWideningAuditTest(unittest.TestCase):
    def test_threshold_relaxes_with_length_ratio(self) -> None:
        policy = audit.WideningPolicy(
            base_threshold=0.97,
            slope=0.08,
            min_threshold=0.85,
            min_base_length=8,
            max_expansion=3.0,
            require_raw_confirm=False,
        )

        selected = audit.select_proportional_length(
            [
                candidate(10, 1.00),
                candidate(12, 0.95),
                candidate(20, 0.92),
            ],
            policy,
        )

        self.assertEqual(selected, 20)

    def test_raw_confirmation_blocks_low_raw_candidate(self) -> None:
        policy = audit.WideningPolicy(
            base_threshold=0.95,
            slope=0.08,
            min_threshold=0.85,
            min_base_length=8,
            max_expansion=3.0,
            require_raw_confirm=True,
        )

        selected = audit.select_proportional_length(
            [
                candidate(10, 1.00, raw=1.00),
                candidate(20, 0.94, raw=0.60),
            ],
            policy,
        )

        self.assertEqual(selected, 10)

    def test_row_decision_only_widens_current_selection(self) -> None:
        policy = audit.WideningPolicy(
            base_threshold=0.97,
            slope=0.02,
            min_threshold=0.90,
            min_base_length=8,
            max_expansion=2.0,
            require_raw_confirm=False,
        )

        decision = audit.row_decision(
            row(
                "already-long",
                selected=24,
                oracle=10,
                passed=True,
                candidates=[candidate(10, 1.0), candidate(12, 0.96)],
            ),
            policy,
        )

        self.assertEqual(decision["proportional_selected_length"], 10)
        self.assertEqual(decision["new_selected_length"], 24)
        self.assertFalse(decision["promoted"])

    def test_summary_reports_short_risk_and_failed_long_improvement(self) -> None:
        policy = audit.WideningPolicy(
            base_threshold=0.97,
            slope=0.08,
            min_threshold=0.85,
            min_base_length=8,
            max_expansion=3.0,
            require_raw_confirm=False,
        )
        rows = [
            row(
                "long-fail",
                selected=10,
                oracle=20,
                passed=False,
                candidates=[candidate(10, 1.0), candidate(20, 0.92)],
            ),
            row(
                "short-pass",
                selected=10,
                oracle=6,
                passed=True,
                candidates=[candidate(10, 1.0), candidate(20, 0.92)],
            ),
        ]

        decisions = [audit.row_decision(item, policy) for item in rows]
        summary = audit.summarize_decisions(policy, decisions)

        self.assertEqual(summary["promoted_count"], 2)
        self.assertEqual(summary["failed_true_long_improved_count"], 1)
        self.assertEqual(summary["short_promoted_count"], 1)
        self.assertEqual(summary["current_pass_promoted_count"], 1)

    def test_missing_candidates_keeps_current_length(self) -> None:
        policy = audit.WideningPolicy(
            base_threshold=0.97,
            slope=0.08,
            min_threshold=0.85,
            min_base_length=8,
            max_expansion=3.0,
            require_raw_confirm=False,
        )

        decision = audit.row_decision(
            {
                "task_id": "missing",
                "metrics": {"selected_mask_length": 8, "oracle_mask_length": 12, "passed": False},
            },
            policy,
        )

        self.assertEqual(decision["new_selected_length"], 8)
        self.assertFalse(decision["promoted"])


if __name__ == "__main__":
    unittest.main()
