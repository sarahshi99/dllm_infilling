from __future__ import annotations

import unittest

from experiments.action_ceiling.frozen_canvas_controller import (
    ACTION_ORDER,
    LogisticModel,
    actual_canvas,
    binomial_upper_95,
    evaluate_selection,
    feature_schema,
    row_passed,
    select_actions,
)


class FrozenCanvasControllerTest(unittest.TestCase):
    def test_actual_canvas_is_max_primary_and_target(self) -> None:
        self.assertEqual(actual_canvas(20, "KEEP_PRIMARY"), 20)
        self.assertEqual(actual_canvas(20, "EXPAND_16"), 20)
        self.assertEqual(actual_canvas(20, "EXPAND_24"), 24)
        self.assertEqual(actual_canvas(3, "EXPAND_48"), 48)
        self.assertEqual(list(ACTION_ORDER), ["KEEP_PRIMARY", "EXPAND_16", "EXPAND_24", "EXPAND_32", "EXPAND_48"])

    def test_feature_schema_excludes_non_inference_visible_labels(self) -> None:
        schema = feature_schema()
        forbidden = set(schema["forbidden_features"])
        self.assertIn("oracle_length", forbidden)
        self.assertIn("unit_test_result", forbidden)
        self.assertIn("action_bank_outcome", forbidden)
        for variant, columns in schema["feature_variants"].items():
            self.assertNotIn("oracle_length", columns, variant)
            self.assertNotIn("primary_passed", columns, variant)
            self.assertNotIn("action_passed", columns, variant)

    def test_logistic_model_round_trip_is_stable(self) -> None:
        rows = [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        model = LogisticModel(["x"])
        model.fit(rows, [0, 0, 1, 1], epochs=20, lr=0.1)
        before = model.predict_one({"x": 2.5})
        restored = LogisticModel.from_json(model.to_json())
        after = restored.predict_one({"x": 2.5})
        self.assertAlmostEqual(before, after, places=12)

    def test_binomial_upper_bound_handles_zero_interventions(self) -> None:
        self.assertIsNone(binomial_upper_95(0, 0))
        self.assertGreater(binomial_upper_95(0, 5), 0.0)
        self.assertLessEqual(binomial_upper_95(0, 5), 1.0)

    def test_select_actions_respects_keep_reject_threshold(self) -> None:
        class FixedModel:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict_one(self, row: dict) -> float:
                return self.value

        rows = [
            {
                "task_id": "a",
                "task_group": "HumanEval/0",
                "action": "KEEP_PRIMARY",
                "actual_canvas": 8,
                "passed": True,
                "normalized_canvas_cost": 0.0,
            },
            {
                "task_id": "a",
                "task_group": "HumanEval/0",
                "action": "EXPAND_32",
                "actual_canvas": 32,
                "passed": False,
                "normalized_canvas_cost": 32 / 48,
            },
        ]
        models = {"benefit": FixedModel(0.2), "harm": FixedModel(0.0), "underallocation": FixedModel(0.0)}
        selected = select_actions(rows, models, policy_variant="benefit_only", score_threshold=999.0)
        self.assertEqual(selected[0]["selected_action"], "KEEP_PRIMARY")
        self.assertFalse(selected[0]["intervened"])

    def test_evaluate_selection_accepts_action_passed_rows(self) -> None:
        primary = {
            "a": {"task_id": "a", "passed": False, "oracle_bucket": "17-24"},
            "b": {"task_id": "b", "passed": True, "oracle_bucket": "<=8"},
        }
        selected = [
            {"task_id": "a", "action": "EXPAND_32", "action_passed": True, "oracle_bucket": "17-24"},
            {"task_id": "b", "action": "EXPAND_32", "action_passed": False, "oracle_bucket": "<=8"},
        ]
        self.assertTrue(row_passed(selected[0]))
        metrics = evaluate_selection(selected, primary)
        self.assertEqual(metrics["pass_count"], 1)
        self.assertEqual(metrics["wins_vs_primary"], 1)
        self.assertEqual(metrics["losses_vs_primary"], 1)
        self.assertEqual(metrics["intervention_count"], 2)


if __name__ == "__main__":
    unittest.main()
