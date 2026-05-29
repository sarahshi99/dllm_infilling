from __future__ import annotations

import unittest

from analysis import diagnose_long_underestimate_policy as diag


def row(
    *,
    task_id: str,
    passed: bool,
    oracle: int,
    selected: int,
    best_long: int,
    long_ratio: float,
    raw_ratio: float,
    support: int,
    source: str = "base",
) -> dict:
    return {
        "task_id": task_id,
        "metrics": {
            "passed": passed,
            "oracle_mask_length": oracle,
            "selected_mask_length": selected,
            "best_long_len": best_long,
            "long_ratio": long_ratio,
            "raw_long_ratio": raw_ratio,
            "support_count": support,
            "final_source": source,
        },
    }


class DiagnoseLongUnderestimatePolicyTest(unittest.TestCase):
    def test_row_triggers_when_long_curve_exceeds_selected_length(self) -> None:
        rule = diag.Rule(
            max_selected_len=8,
            min_best_long_len=16,
            min_gap=8,
            min_long_ratio=0.8,
            min_raw_long_ratio=0.75,
            min_support_count=0,
            allowed_sources=("base",),
        )

        self.assertTrue(
            diag.row_triggers(
                row(
                    task_id="long/fail",
                    passed=False,
                    oracle=20,
                    selected=6,
                    best_long=16,
                    long_ratio=0.85,
                    raw_ratio=0.80,
                    support=0,
                ),
                rule,
            )
        )
        self.assertFalse(
            diag.row_triggers(
                row(
                    task_id="short/pass",
                    passed=True,
                    oracle=6,
                    selected=6,
                    best_long=13,
                    long_ratio=0.90,
                    raw_ratio=0.85,
                    support=0,
                ),
                rule,
            )
        )

    def test_evaluate_rule_reports_precision_risk_and_recall(self) -> None:
        rows = [
            row(
                task_id="long/fail",
                passed=False,
                oracle=20,
                selected=6,
                best_long=16,
                long_ratio=0.85,
                raw_ratio=0.80,
                support=0,
            ),
            row(
                task_id="short/pass",
                passed=True,
                oracle=6,
                selected=6,
                best_long=16,
                long_ratio=0.90,
                raw_ratio=0.85,
                support=0,
            ),
            row(
                task_id="medium/pass",
                passed=True,
                oracle=12,
                selected=12,
                best_long=16,
                long_ratio=0.95,
                raw_ratio=0.90,
                support=1,
            ),
        ]
        rule = diag.Rule(
            max_selected_len=8,
            min_best_long_len=16,
            min_gap=8,
            min_long_ratio=0.8,
            min_raw_long_ratio=0.75,
            min_support_count=0,
            allowed_sources=("base",),
        )

        result = diag.evaluate_rule(rows, rule)

        self.assertEqual(result["trigger_count"], 2)
        self.assertEqual(result["true_long_count"], 1)
        self.assertEqual(result["failed_long_trigger_count"], 1)
        self.assertAlmostEqual(result["true_long_precision"], 0.5)
        self.assertAlmostEqual(result["short_risk_rate"], 0.5)
        self.assertAlmostEqual(result["failed_long_recall"], 1.0)
        self.assertAlmostEqual(result["current_pass_risk_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
