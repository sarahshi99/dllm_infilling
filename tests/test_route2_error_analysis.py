from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.route2_error_analysis import (
    NUMERIC_FEATURES,
    bucket_oracle_length,
    classify_pairwise,
    classify_row,
    feature_row,
    rescue_length_sufficient,
    summarize_joined_rows,
    write_outputs,
)


def make_row(
    task_id: str,
    *,
    passed: bool,
    oracle: int,
    selected: int,
    triggered: bool = False,
    rescue_len: int | None = None,
    primary_selected: int | None = None,
    top1_median: float | None = None,
    confidence_max: float | None = None,
) -> dict:
    metrics = {
        "passed": passed,
        "oracle_mask_length": oracle,
        "selected_mask_length": selected,
        "route2_trace_rescue_triggered": triggered,
        "route2_rescue_length": rescue_len,
        "route2_primary_selected_mask_length": primary_selected if primary_selected is not None else selected,
        "route2_trace_top1_median": top1_median,
        "route2_trace_confidence_max": confidence_max,
    }
    return {
        "task_id": task_id,
        "metrics": metrics,
        "verification": {"passed": passed},
        "route2_trace_rescue": {"triggered": triggered} if triggered else None,
    }


class Route2ErrorAnalysisCoreTest(unittest.TestCase):
    def test_bucket_oracle_length(self) -> None:
        self.assertEqual(bucket_oracle_length(None), "unknown")
        self.assertEqual(bucket_oracle_length(8), "<=8")
        self.assertEqual(bucket_oracle_length(12), "9-12")
        self.assertEqual(bucket_oracle_length(16), "13-16")
        self.assertEqual(bucket_oracle_length(24), "17-24")
        self.assertEqual(bucket_oracle_length(25), "25+")

    def test_classify_pairwise(self) -> None:
        self.assertEqual(classify_pairwise(route_passed=True, baseline_passed=False), "win")
        self.assertEqual(classify_pairwise(route_passed=False, baseline_passed=True), "loss")
        self.assertEqual(classify_pairwise(route_passed=True, baseline_passed=True), "tie_pass")
        self.assertEqual(classify_pairwise(route_passed=False, baseline_passed=False), "tie_fail")

    def test_rescue_length_sufficient(self) -> None:
        self.assertTrue(rescue_length_sufficient(rescue_length=32, oracle_length=31))
        self.assertTrue(rescue_length_sufficient(rescue_length=32, oracle_length=32))
        self.assertFalse(rescue_length_sufficient(rescue_length=24, oracle_length=25))
        self.assertIsNone(rescue_length_sufficient(rescue_length=None, oracle_length=25))

    def test_classify_row_error_taxonomy(self) -> None:
        baseline_fail_long = make_row("a", passed=False, oracle=20, selected=8)
        route_trigger_fail = make_row("a", passed=False, oracle=20, selected=32, triggered=True, rescue_len=32)
        route_trigger_pass = make_row("a", passed=True, oracle=20, selected=32, triggered=True, rescue_len=32)
        route_missed = make_row("a", passed=False, oracle=20, selected=8, triggered=False)
        short_win = make_row("a", passed=True, oracle=8, selected=32, triggered=True, rescue_len=32)

        self.assertIn(
            "triggered_failed_long",
            classify_row(baseline_fail_long, route_trigger_fail)["classes"],
        )
        self.assertIn(
            "triggered_rescued_long",
            classify_row(baseline_fail_long, route_trigger_pass)["classes"],
        )
        self.assertIn("missed_failed_long", classify_row(baseline_fail_long, route_missed)["classes"])
        self.assertIn("short_or_medium_win", classify_row(baseline_fail_long, short_win)["classes"])

    def test_feature_row_excludes_policy_forbidden_labels(self) -> None:
        baseline = make_row("a", passed=False, oracle=20, selected=8)
        route = make_row(
            "a",
            passed=False,
            oracle=20,
            selected=32,
            triggered=True,
            rescue_len=32,
            top1_median=0.4,
            confidence_max=0.8,
        )

        row = feature_row(baseline, route)

        self.assertEqual(row["task_id"], "a")
        self.assertEqual(row["oracle_bucket"], "17-24")
        self.assertEqual(row["route2_trace_top1_median"], 0.4)
        self.assertNotIn("passed", row)
        self.assertNotIn("oracle_mask_length", row)

    def test_feature_contrast_candidates_exclude_oracle_derived_features(self) -> None:
        self.assertNotIn("route2_selected_minus_oracle_length", NUMERIC_FEATURES)
        self.assertNotIn("route2_abs_selected_minus_oracle_length", NUMERIC_FEATURES)


class Route2ErrorAnalysisSummaryTest(unittest.TestCase):
    def test_summarize_joined_rows_counts_known_classes(self) -> None:
        baseline = [
            make_row("win-long", passed=False, oracle=20, selected=3),
            make_row("miss-long", passed=False, oracle=25, selected=8),
            make_row("trigger-fail", passed=False, oracle=21, selected=3),
            make_row("short-win", passed=False, oracle=8, selected=3),
            make_row("tie-pass", passed=True, oracle=8, selected=8),
        ]
        route = [
            make_row("win-long", passed=True, oracle=20, selected=32, triggered=True, rescue_len=32),
            make_row("miss-long", passed=False, oracle=25, selected=8, triggered=False),
            make_row("trigger-fail", passed=False, oracle=21, selected=32, triggered=True, rescue_len=32),
            make_row("short-win", passed=True, oracle=8, selected=32, triggered=True, rescue_len=32),
            make_row("tie-pass", passed=True, oracle=8, selected=8, triggered=False),
        ]

        summary, taxonomy = summarize_joined_rows(baseline, route)

        self.assertEqual(summary["joined_rows"], 5)
        self.assertEqual(summary["pairwise_counts"], {"win": 2, "loss": 0, "tie_pass": 1, "tie_fail": 2})
        self.assertEqual(summary["class_counts"]["triggered_rescued_long"], 1)
        self.assertEqual(summary["class_counts"]["missed_failed_long"], 1)
        self.assertEqual(summary["class_counts"]["triggered_failed_long"], 1)
        self.assertEqual(summary["class_counts"]["short_or_medium_win"], 1)
        self.assertEqual(summary["triggered_failed_long_rescue_ge_oracle"], 1)
        self.assertEqual(len(taxonomy), 5)


class Route2ErrorAnalysisIoTest(unittest.TestCase):
    def test_write_outputs_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = {
                "joined_rows": 1,
                "pairwise_counts": {"win": 1, "loss": 0, "tie_pass": 0, "tie_fail": 0},
                "class_counts": {"route2_win": 1},
                "decision": "route2_polish_only",
            }
            taxonomy = [
                {
                    "task_id": "a",
                    "classes": "route2_win",
                    "pairwise": "win",
                    "oracle_bucket": "<=8",
                }
            ]
            write_outputs(output_dir, summary, taxonomy)

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "error_taxonomy.csv").exists())
            self.assertTrue((output_dir / "wins.csv").exists())
            self.assertTrue((output_dir / "report.md").exists())
            loaded = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["joined_rows"], 1)


if __name__ == "__main__":
    unittest.main()
