from __future__ import annotations

import unittest

from analysis.trace_long_rescue_features import (
    compute_gate_summary,
    extract_trace_features,
    join_results_and_traces,
)
from analysis.analyze_trace_long_rescue_routes import (
    route1_trace_only_trigger,
    route2_risk_controlled_trigger,
    route3_rerank_choice,
)


class TraceLongRescueFeaturesTest(unittest.TestCase):
    def test_join_results_and_traces_groups_by_task_id(self) -> None:
        rows = [
            {"task_id": "a", "metrics": {"passed": False, "oracle_mask_length": 18}},
            {"task_id": "b", "metrics": {"passed": True, "oracle_mask_length": 6}},
        ]
        traces = [
            {
                "task_id": "a",
                "step": 0,
                "remaining_masks_after_update": 18,
                "stop_decision": {"remaining_mask_ratio": 1.0},
            },
            {
                "task_id": "a",
                "step": 1,
                "remaining_masks_after_update": 12,
                "stop_decision": {"remaining_mask_ratio": 0.67},
            },
            {
                "task_id": "b",
                "step": 0,
                "remaining_masks_after_update": 2,
                "stop_decision": {"remaining_mask_ratio": 0.33},
            },
        ]

        joined = join_results_and_traces(rows, traces)

        self.assertEqual(len(joined), 2)
        self.assertEqual([item["step"] for item in joined["a"]["traces"]], [0, 1])
        self.assertEqual([item["step"] for item in joined["b"]["traces"]], [0])

    def test_extract_trace_features_uses_decode_dynamics(self) -> None:
        row = {
            "task_id": "a",
            "metrics": {
                "passed": False,
                "oracle_mask_length": 18,
                "selected_mask_length": 8,
                "remaining_masks_at_stop_or_final": 6,
                "remaining_mask_ratio_at_stop_or_final": 0.75,
                "mean_gap_at_stop_or_final": 0.12,
                "mean_top1_at_stop_or_final": 0.40,
            },
        }
        traces = [
            {
                "step": 0,
                "remaining_masks_before_update": 8,
                "remaining_masks_after_update": 8,
                "mean_confidence": 0.20,
                "stop_decision": {"reason": "too_many_remaining_masks"},
            },
            {
                "step": 1,
                "remaining_masks_before_update": 8,
                "remaining_masks_after_update": 6,
                "mean_confidence": 0.30,
                "stop_decision": {"reason": "mean_gap_below_threshold"},
            },
        ]

        features = extract_trace_features(row, traces)

        self.assertEqual(features["task_id"], "a")
        self.assertTrue(features["failed_long"])
        self.assertEqual(features["oracle_bucket"], "17-24")
        self.assertEqual(features["final_remaining_masks"], 6)
        self.assertAlmostEqual(features["final_remaining_mask_ratio"], 0.75)
        self.assertAlmostEqual(features["max_remaining_mask_ratio"], 1.0)
        self.assertEqual(features["last_stop_reason"], "mean_gap_below_threshold")

    def test_compute_gate_summary_counts_short_losses_and_long_gains(self) -> None:
        base_rows = {
            "short_loss": {"metrics": {"passed": True, "oracle_mask_length": 5}},
            "long_gain": {"metrics": {"passed": False, "oracle_mask_length": 20}},
            "neutral": {"metrics": {"passed": True, "oracle_mask_length": 10}},
        }
        candidate_passed = {
            "short_loss": False,
            "long_gain": True,
            "neutral": True,
        }

        summary = compute_gate_summary(base_rows, candidate_passed)

        self.assertEqual(summary["overall_delta"], 0)
        self.assertEqual(summary["short_net_loss"], 1)
        self.assertEqual(summary["long_net_gain"], 1)
        self.assertTrue(summary["gate_a_passed"])


class TraceLongRescueRoutesTest(unittest.TestCase):
    def test_route1_trace_only_trigger_requires_unstable_trace(self) -> None:
        features = {
            "selected_len": 8,
            "trace_steps": 4,
            "final_remaining_masks": 6,
            "final_remaining_mask_ratio": 0.75,
            "final_mean_gap": 0.10,
            "last_stop_reason": "mean_gap_below_threshold",
        }
        self.assertTrue(route1_trace_only_trigger(features))

        features["final_remaining_mask_ratio"] = 0.10
        features["final_remaining_masks"] = 1
        self.assertFalse(route1_trace_only_trigger(features))

    def test_route2_is_more_conservative_than_route1(self) -> None:
        features = {
            "selected_len": 8,
            "trace_steps": 4,
            "final_remaining_masks": 6,
            "final_remaining_mask_ratio": 0.75,
            "final_mean_gap": 0.10,
            "last_mean_confidence": 0.30,
            "last_stop_reason": "mean_gap_below_threshold",
        }
        self.assertTrue(route2_risk_controlled_trigger(features))

        features["selected_len"] = 13
        self.assertFalse(route2_risk_controlled_trigger(features))

    def test_route3_picks_highest_trace_quality_without_verifier(self) -> None:
        candidates = [
            {"length": 8, "trace_quality": 0.20, "length_penalty": 0.00},
            {"length": 20, "trace_quality": 0.55, "length_penalty": 0.05},
            {"length": 32, "trace_quality": 0.56, "length_penalty": 0.30},
        ]
        self.assertEqual(route3_rerank_choice(candidates)["length"], 20)


if __name__ == "__main__":
    unittest.main()
