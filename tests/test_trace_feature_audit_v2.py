from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.trace_feature_audit_v2 import (
    CandidateRule,
    build_feature_record,
    compute_shape_features,
    evaluate_rule,
    fold_id,
    generate_single_feature_rules,
    load_records,
    write_report_outputs,
)


class TraceFeatureAuditV2FeatureTest(unittest.TestCase):
    def test_compute_shape_features_captures_slope_area_and_plateau(self) -> None:
        traces = [
            {"step": 0, "remaining_masks_after_update": 8, "mean_confidence": 0.20},
            {"step": 1, "remaining_masks_after_update": 6, "mean_confidence": 0.25},
            {"step": 2, "remaining_masks_after_update": 6, "mean_confidence": 0.22},
            {"step": 3, "remaining_masks_after_update": 6, "mean_confidence": 0.18},
        ]

        features = compute_shape_features(traces)

        self.assertEqual(features["trace_steps"], 4)
        self.assertAlmostEqual(features["remaining_slope"], -2.0 / 3.0)
        self.assertAlmostEqual(features["remaining_auc_norm"], 26.0 / 32.0)
        self.assertEqual(features["late_remaining_plateau_steps"], 2)
        self.assertAlmostEqual(features["confidence_slope"], -0.02 / 3.0)

    def test_build_feature_record_separates_policy_features_from_labels(self) -> None:
        row = {
            "task_id": "task-a",
            "metrics": {
                "passed": False,
                "oracle_mask_length": 20,
                "selected_mask_length": 8,
                "long_score_max": 0.7,
            },
        }
        traces = [
            {
                "step": 0,
                "remaining_masks_after_update": 8,
                "mean_confidence": 0.20,
                "stop_decision": {"reason": "before_min_stop_step", "mean_gap": 0.30},
            },
            {
                "step": 1,
                "remaining_masks_after_update": 6,
                "mean_confidence": 0.19,
                "stop_decision": {"reason": "mean_gap_below_threshold", "mean_gap": 0.12},
            },
        ]

        record = build_feature_record(row, traces, source_name="midcons")

        self.assertEqual(record["task_id"], "task-a")
        self.assertTrue(record["labels"]["failed_long"])
        self.assertTrue(record["labels"]["true_long"])
        self.assertFalse(record["labels"]["short_risk"])
        self.assertNotIn("oracle_mask_length", record["features"])
        self.assertNotIn("passed", record["features"])
        self.assertEqual(record["features"]["selected_len"], 8)
        self.assertEqual(record["features"]["stop_reason"], "mean_gap_below_threshold")
        self.assertEqual(record["source"], "midcons")

    def test_fold_id_is_deterministic_and_bounded(self) -> None:
        first = fold_id("SingleLineInfilling/HumanEval/0/L0", folds=5)
        second = fold_id("SingleLineInfilling/HumanEval/0/L0", folds=5)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 5)


class TraceFeatureAuditV2RuleTest(unittest.TestCase):
    def test_evaluate_rule_counts_failed_long_short_and_current_pass_risk(self) -> None:
        records = [
            {
                "task_id": "long-fail",
                "features": {"score": 0.9},
                "labels": {
                    "failed_long": True,
                    "true_long": True,
                    "short_risk": False,
                    "current_pass_risk": False,
                },
            },
            {
                "task_id": "short-pass",
                "features": {"score": 0.8},
                "labels": {
                    "failed_long": False,
                    "true_long": False,
                    "short_risk": True,
                    "current_pass_risk": True,
                },
            },
            {
                "task_id": "safe",
                "features": {"score": 0.1},
                "labels": {
                    "failed_long": False,
                    "true_long": False,
                    "short_risk": False,
                    "current_pass_risk": False,
                },
            },
        ]
        rule = CandidateRule(name="score_ge_0.5", clauses=[["score", ">=", 0.5]])

        summary = evaluate_rule(rule, records)

        self.assertEqual(summary["trigger_count"], 2)
        self.assertEqual(summary["failed_long_count"], 1)
        self.assertEqual(summary["short_risk_count"], 1)
        self.assertEqual(summary["current_pass_risk_count"], 1)
        self.assertAlmostEqual(summary["true_long_precision"], 0.5)

    def test_generate_single_feature_rules_uses_numeric_thresholds(self) -> None:
        records = [
            {"features": {"score": 0.1}, "labels": {}},
            {"features": {"score": 0.5}, "labels": {}},
            {"features": {"score": 0.9}, "labels": {}},
        ]

        rules = generate_single_feature_rules(records, feature_names=["score"])

        self.assertTrue(any(rule.name.startswith("score_ge_") for rule in rules))
        self.assertTrue(any(rule.name.startswith("score_le_") for rule in rules))


class TraceFeatureAuditV2IoTest(unittest.TestCase):
    def test_load_records_joins_results_and_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            results = temp / "results.jsonl"
            traces = temp / "step_traces.jsonl"
            results.write_text(
                json.dumps(
                    {
                        "task_id": "task-a",
                        "metrics": {
                            "passed": False,
                            "oracle_mask_length": 20,
                            "selected_mask_length": 8,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            traces.write_text(
                json.dumps(
                    {
                        "task_id": "task-a",
                        "step": 0,
                        "remaining_masks_after_update": 8,
                        "mean_confidence": 0.2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_records(results, traces, source_name="previous")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["task_id"], "task-a")
        self.assertEqual(records[0]["features"]["trace_steps"], 1)

    def test_write_report_outputs_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = {
                "decision": "reject",
                "sources": {"midcons": {"rows": 1}},
                "top_candidates": [{"name": "score_ge_0.5", "trigger_count": 1}],
            }
            write_report_outputs(output_dir, summary, candidates=[{"name": "score_ge_0.5", "trigger_count": 1}])

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "candidates.csv").exists())
            self.assertTrue((output_dir / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
