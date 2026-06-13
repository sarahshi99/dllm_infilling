from __future__ import annotations

import unittest

from clean_scripts.run_route2_trace_rescue import (
    pairwise_vs_baseline,
    route2_rule,
    summarize_route2,
    trace_record_from_result,
)


class Route2TraceRescueTest(unittest.TestCase):
    def test_broad_plateau_rule_matches_trace_features(self) -> None:
        result = {
            "metrics": {"selected_mask_length": 3},
            "step_traces": [
                {
                    "step": 0,
                    "remaining_masks_after_update": 3,
                    "mean_confidence": 0.2,
                    "stop_decision": {"mean_top1": 0.60},
                },
                {
                    "step": 1,
                    "remaining_masks_after_update": 3,
                    "mean_confidence": 0.2,
                    "stop_decision": {"mean_top1": 0.60},
                },
            ]
            + [
                {
                    "step": step,
                    "remaining_masks_after_update": 3,
                    "mean_confidence": 0.2,
                    "stop_decision": {"mean_top1": 0.60},
                }
                for step in range(2, 18)
            ],
        }

        record = trace_record_from_result(result)

        self.assertTrue(route2_rule("broad_plateau").matches(record))

    def test_pairwise_counts_wins_losses_and_ties(self) -> None:
        results = [
            {"task_id": "a", "metrics": {"passed": True, "oracle_mask_length": 20}},
            {"task_id": "b", "metrics": {"passed": False, "oracle_mask_length": 6}},
            {"task_id": "c", "metrics": {"passed": True, "oracle_mask_length": 10}},
            {"task_id": "d", "metrics": {"passed": False, "oracle_mask_length": 30}},
        ]
        baseline = [
            {"task_id": "a", "metrics": {"passed": False}},
            {"task_id": "b", "metrics": {"passed": True}},
            {"task_id": "c", "metrics": {"passed": True}},
            {"task_id": "d", "metrics": {"passed": False}},
        ]

        summary = pairwise_vs_baseline(results, baseline)

        self.assertEqual(summary["counts"], {"win": 1, "loss": 1, "tie_pass": 1, "tie_fail": 1})
        self.assertEqual(summary["by_oracle_bucket"]["17-24"]["win"], 1)
        self.assertEqual(summary["by_oracle_bucket"]["<=8"]["loss"], 1)

    def test_summarize_route2_counts_triggers_and_precision(self) -> None:
        def result(task_id: str, passed: bool, oracle: int, triggered: bool):
            return {
                "task_id": task_id,
                "metrics": {
                    "passed": passed,
                    "oracle_mask_length": oracle,
                    "selected_mask_length": 24 if triggered else 3,
                    "mask_length": 24 if triggered else 3,
                    "selected_minus_oracle_length": 0,
                    "abs_selected_minus_oracle_length": 0,
                    "route2_trace_rescue_triggered": triggered,
                    "route2_policy": "broad_plateau",
                    "route2_final_source": "rescue" if triggered else "primary",
                },
            }

        rows = [
            result("a", True, 20, True),
            result("b", False, 6, True),
            result("c", True, 10, False),
        ]

        summary = summarize_route2(rows, baseline_rows=[], baseline_path=None)

        self.assertEqual(summary["route2_trigger_count"], 2)
        self.assertAlmostEqual(summary["route2_trigger_true_long_precision"], 0.5)
        self.assertEqual(summary["route2_trigger_oracle_bucket_histogram"], {"17-24": 1, "<=8": 1})


if __name__ == "__main__":
    unittest.main()
