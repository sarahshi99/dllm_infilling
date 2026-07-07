from __future__ import annotations

import unittest

from experiments.action_ceiling.controller_feasibility_audit import (
    ACTION_ORDER,
    auc_score,
    clopper_pearson_upper,
    feature_rows,
    minimum_passing_action,
    task_records,
    zero_harm_n_for_bound,
)


class ControllerFeasibilityAuditTest(unittest.TestCase):
    def test_minimum_passing_action_prefers_keep_and_lower_cost(self) -> None:
        actions = {
            "KEEP_PRIMARY": {"passed": True},
            "EXPAND_16": {"passed": True},
            "EXPAND_24": {"passed": True},
        }
        self.assertEqual(minimum_passing_action(actions), "KEEP")
        actions["KEEP_PRIMARY"] = {"passed": False}
        self.assertEqual(minimum_passing_action(actions), "EXPAND_16")

    def test_task_records_build_recoverability_and_harmability(self) -> None:
        rows = []
        for task_id, keep_pass, expand16_pass, expand24_pass in [
            ("recover", False, True, False),
            ("harm", True, False, True),
        ]:
            for action, passed in [
                ("KEEP_PRIMARY", keep_pass),
                ("EXPAND_16", expand16_pass),
                ("EXPAND_24", expand24_pass),
                ("EXPAND_32", False),
                ("EXPAND_48", False),
            ]:
                rows.append({"task_id": task_id, "task_group": "g", "split": "validation", "oracle_bucket": "17-24", "action": action, "passed": passed})
        records = {row["task_id"]: row for row in task_records(rows)}
        self.assertTrue(records["recover"]["recoverable"])
        self.assertFalse(records["recover"]["harmable"])
        self.assertEqual(records["recover"]["minimum_passing_action"], "EXPAND_16")
        self.assertFalse(records["harm"]["recoverable"])
        self.assertTrue(records["harm"]["harmable"])
        self.assertEqual(records["harm"]["minimum_passing_action"], "KEEP")

    def test_zero_harm_sample_complexity_and_cp_bound(self) -> None:
        self.assertEqual(zero_harm_n_for_bound(0.05), 59)
        self.assertLess(clopper_pearson_upper(0, 60), 0.05)
        self.assertGreater(clopper_pearson_upper(0, 10), 0.05)

    def test_auc_score_orders_perfect_ranking(self) -> None:
        self.assertEqual(auc_score([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9]), 1.0)

    def test_feature_rows_use_no_forbidden_label_features_in_base_columns(self) -> None:
        bank_rows = []
        for action, passed in zip(ACTION_ORDER, [False, True, False, False, False]):
            bank_rows.append(
                {
                    "task_id": "task/0",
                    "task_group": "g",
                    "split": "train",
                    "oracle_bucket": "17-24",
                    "oracle_length": 20,
                    "primary_selected_length": 8,
                    "actual_canvas": 16,
                    "action": action,
                    "passed": passed,
                    "inference_cost_sec": 1.0,
                }
            )
        primary = {
            "task_id": "task/0",
            "metrics": {"passed": False, "selected_mask_length": 8, "best_len": 8, "best_long_len": 24},
        }
        import tempfile
        import json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary.jsonl"
            path.write_text(json.dumps(primary) + "\n", encoding="utf-8")
            task_rows, action_rows = feature_rows(bank_rows, primary_results=str(path), route2_results=None)
        self.assertEqual(len(task_rows), 1)
        self.assertEqual(len(action_rows), 5)
        self.assertTrue(task_rows[0]["recoverable"])
        self.assertNotIn("error_type", task_rows[0])
        self.assertNotIn("reference_code", action_rows[0])


if __name__ == "__main__":
    unittest.main()
