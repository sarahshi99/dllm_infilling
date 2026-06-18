from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.discovery_v4_signal_audit import (
    CandidateSlice,
    build_row_action_table,
    candidate_feature_columns,
    evaluate_slice,
    summarize_policy_deltas,
    weak_signal_votes,
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
    best_long_len: int | None = None,
    long_ratio: float | None = None,
    top1: float | None = None,
    stop_reason: str | None = None,
) -> dict:
    metrics = {
        "passed": passed,
        "oracle_mask_length": oracle,
        "selected_mask_length": selected,
        "best_long_len": best_long_len,
        "long_ratio": long_ratio,
        "mean_top1_at_stop_or_final": top1,
        "stop_reason": stop_reason,
        "route2_trace_rescue_triggered": triggered,
        "route2_rescue_length": rescue_len,
        "route2_primary_selected_mask_length": selected,
        "route2_trace_top1_last": top1,
        "route2_primary_stop_reason": stop_reason,
    }
    return {"task_id": task_id, "metrics": metrics, "verification": {"passed": passed}}


class DiscoveryV4RowActionTableTest(unittest.TestCase):
    def test_build_row_action_table_joins_multiple_policies(self) -> None:
        baseline = [
            make_row("missed", passed=False, oracle=20, selected=8, best_long_len=24, long_ratio=0.9),
            make_row("rescued", passed=False, oracle=21, selected=8),
            make_row("risk", passed=True, oracle=7, selected=7),
        ]
        len32 = [
            make_row("missed", passed=False, oracle=20, selected=8, triggered=False),
            make_row("rescued", passed=True, oracle=21, selected=32, triggered=True, rescue_len=32),
            make_row("risk", passed=True, oracle=7, selected=7, triggered=False),
        ]
        len24 = [
            make_row("missed", passed=False, oracle=20, selected=8, triggered=True, rescue_len=24),
            make_row("rescued", passed=False, oracle=21, selected=24, triggered=True, rescue_len=24),
            make_row("risk", passed=True, oracle=7, selected=7, triggered=False),
        ]

        rows = build_row_action_table(
            baseline,
            baseline_trace_rows=[],
            policies={"precision_len32": len32, "precision_len24": len24},
        )

        self.assertEqual(len(rows), 3)
        by_task = {row["task_id"]: row for row in rows}
        self.assertTrue(by_task["missed"]["missed_failed_long_precision_len32"])
        self.assertTrue(by_task["rescued"]["rescued_long_precision_len32"])
        self.assertEqual(by_task["rescued"]["precision_len32_pairwise"], "win")
        self.assertEqual(by_task["rescued"]["precision_len24_pairwise"], "tie_fail")

    def test_candidate_feature_columns_excludes_forbidden_labels(self) -> None:
        rows = build_row_action_table(
            [make_row("a", passed=False, oracle=20, selected=8, best_long_len=24)],
            baseline_trace_rows=[],
            policies={"precision_len32": [make_row("a", passed=False, oracle=20, selected=8)]},
        )

        columns = candidate_feature_columns(rows)

        joined = " ".join(columns)
        self.assertIn("baseline_selected_len", columns)
        self.assertNotIn("oracle_length", columns)
        self.assertNotIn("true_long", columns)
        self.assertNotIn("baseline_passed", columns)
        self.assertNotIn("precision_len32_passed", columns)
        self.assertNotIn("precision_len32_pairwise", columns)
        self.assertNotIn("missed_failed_long_precision_len32", columns)
        self.assertNotIn("triggered_rescue_failure_precision_len32", columns)
        self.assertNotIn("current_pass_risk", columns)
        self.assertNotIn("oracle", joined)
        self.assertNotIn("passed", joined)


class DiscoveryV4SliceTest(unittest.TestCase):
    def test_evaluate_slice_counts_targets_and_risks(self) -> None:
        rows = [
            {
                "task_id": "missed",
                "score": 0.9,
                "true_long": True,
                "short_risk": False,
                "current_pass_risk": False,
                "missed_failed_long_precision_len32": True,
                "triggered_rescue_failure_precision_len32": False,
                "precision_len32_pairwise": "tie_fail",
            },
            {
                "task_id": "short",
                "score": 0.8,
                "true_long": False,
                "short_risk": True,
                "current_pass_risk": True,
                "missed_failed_long_precision_len32": False,
                "triggered_rescue_failure_precision_len32": False,
                "precision_len32_pairwise": "tie_pass",
            },
            {
                "task_id": "safe",
                "score": 0.1,
                "true_long": False,
                "short_risk": False,
                "current_pass_risk": False,
                "missed_failed_long_precision_len32": False,
                "triggered_rescue_failure_precision_len32": False,
                "precision_len32_pairwise": "tie_fail",
            },
        ]
        rule = CandidateSlice(name="score_ge_0.5", family="single", clauses=[["score", ">=", 0.5]])

        summary = evaluate_slice(rule, rows)

        self.assertEqual(summary["trigger_count"], 2)
        self.assertEqual(summary["missed_failed_long_count"], 1)
        self.assertEqual(summary["short_risk_count"], 1)
        self.assertEqual(summary["current_pass_risk_count"], 1)
        self.assertAlmostEqual(summary["true_long_precision"], 0.5)

    def test_weak_signal_votes_are_inference_visible(self) -> None:
        row = {
            "baseline_selected_len": 8,
            "baseline_best_long_len": 24,
            "baseline_long_ratio": 0.91,
            "baseline_late_remaining_plateau_steps": 3,
            "baseline_top1_last": 0.99,
            "precision_len32_triggered": False,
            "oracle_length": 30,
            "baseline_passed": False,
        }

        votes = weak_signal_votes(row)

        self.assertGreaterEqual(votes["weak_vote_score"], 3)
        self.assertNotIn("oracle_length", votes)
        self.assertNotIn("baseline_passed", votes)


class DiscoveryV4IoTest(unittest.TestCase):
    def test_summarize_policy_deltas_and_write_outputs(self) -> None:
        rows = [
            {
                "task_id": "win",
                "oracle_bucket": "17-24",
                "precision_len32_pairwise": "win",
                "precision_len32_triggered": True,
            },
            {
                "task_id": "loss",
                "oracle_bucket": "<=8",
                "precision_len32_pairwise": "loss",
                "precision_len32_triggered": True,
            },
        ]

        deltas = summarize_policy_deltas(rows, policy_names=["precision_len32"])

        self.assertEqual(deltas[0]["policy"], "precision_len32")
        self.assertEqual(deltas[0]["wins"], 1)
        self.assertEqual(deltas[0]["losses"], 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = {
                "decision": "route2_polish_only",
                "row_count": 2,
                "policy_deltas": deltas,
                "top_slices": [],
            }
            write_outputs(
                output_dir,
                summary=summary,
                row_action_table=rows,
                slice_candidates=[],
                rule_candidates=[],
                trace_shape_candidates=[],
                calibration_residuals=[],
                weak_signal_rows=[],
                uplift_rows=[],
            )

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "row_action_table.csv").exists())
            self.assertTrue((output_dir / "policy_shortlist.md").exists())
            loaded = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["row_count"], 2)


if __name__ == "__main__":
    unittest.main()
