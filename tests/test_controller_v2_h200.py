from __future__ import annotations

import unittest

from experiments.action_ceiling.controller_v2_h200 import (
    ACTION_ORDER,
    feature_schema_payload,
    monotonic_cumulative_probs,
    pairwise_training_rows,
    rank_distribution_from_cumulative,
    validation_gate,
)


class ControllerV2H200Test(unittest.TestCase):
    def test_feature_schema_excludes_forbidden_features_and_test_materialization(self) -> None:
        schema = feature_schema_payload()
        self.assertFalse(schema["test_features_materialized"])
        forbidden = set(schema["forbidden_features"])
        for columns in schema["feature_variants"].values():
            self.assertFalse(forbidden.intersection(columns))

    def test_ordinal_projection_is_monotonic_distribution(self) -> None:
        class Fixed:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict_one(self, row: dict) -> float:
                return self.value

        cumulative = monotonic_cumulative_probs([Fixed(0.8), Fixed(0.9), Fixed(0.4), Fixed(0.5), Fixed(0.1)], {})
        self.assertEqual(cumulative, [0.8, 0.8, 0.4, 0.4, 0.1])
        dist = rank_distribution_from_cumulative(cumulative)
        self.assertAlmostEqual(sum(dist), 1.0)
        self.assertTrue(all(value >= 0.0 for value in dist))

    def test_pairwise_training_rows_only_compare_within_task(self) -> None:
        rows = []
        for task_id in ["a", "b"]:
            for action, passed in zip(ACTION_ORDER, [False, True, False, False, False]):
                rows.append(
                    {
                        "task_id": task_id,
                        "split": "train",
                        "action": action,
                        "action_passed": passed,
                        "selected_len": 1.0,
                        "candidate_actual_canvas": 16.0,
                        "canvas_delta_from_primary": 1.0,
                        "normalized_canvas_cost": 0.5,
                        "action_rank": ACTION_ORDER.index(action),
                    }
                )
        pair_rows, labels = pairwise_training_rows(rows, "probe_only")
        self.assertEqual(len(pair_rows), len(labels))
        self.assertGreater(len(pair_rows), 0)
        for row in pair_rows:
            self.assertNotIn("task_id", row)
            self.assertNotIn("split", row)

    def test_validation_gate_requires_nonzero_and_population_harm_bound(self) -> None:
        v6 = {"validation_pass_count": 89, "validation_mean_cost_sec": 10.0}
        v1 = {"validation_intervention_count": 0}
        base = {
            "validation_pass_count": 91,
            "validation_wins_vs_primary": 3,
            "validation_losses_vs_primary": 1,
            "validation_intervention_count": 5,
            "validation_population_harm_upper95": 0.04,
            "validation_mean_cost_sec": 8.0,
            "validation_bucket_summary": {"<=8": {"wins": 0, "losses": 0}, "17-24": {"wins": 2, "losses": 0}},
        }
        passed, flags = validation_gate(base, v6, v1)
        self.assertTrue(passed)
        self.assertTrue(flags["nonzero_intervention"])
        failed, flags = validation_gate({**base, "validation_intervention_count": 0}, v6, v1)
        self.assertFalse(failed)
        self.assertFalse(flags["nonzero_intervention"])


if __name__ == "__main__":
    unittest.main()
