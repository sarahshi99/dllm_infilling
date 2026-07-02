from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.route2_v6_short_override_audit import (
    ANCHOR_ID,
    SHORT_ID,
    Rule,
    build_policy_features,
    build_triggered_table,
    candidate_by_id,
    candidate_pass,
    feature_columns,
    final_decision,
    score_rule,
    write_outputs,
)


def make_candidate(
    candidate_id: str,
    *,
    passed: bool,
    confidence: float,
    top1: float,
    text: str | None = None,
    compile_passed: bool = True,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "requested_min_length": 24 if candidate_id == SHORT_ID else 32,
        "steps": 64,
        "selected_mask_length": 24 if candidate_id == SHORT_ID else 32,
        "middle_text": text if text is not None else f"    return {candidate_id!r}\n",
        "trace_features": {
            "confidence_last": confidence,
            "top1_median": top1,
            "gap_median": confidence - 0.1,
            "final_remaining_ratio_by_selected": 0.0,
        },
        "syntax": {"parse_passed": True, "compile_passed": compile_passed},
        "offline_passed": passed,
        "offline_oracle_mask_length": 20,
    }


def make_v5_row(
    task_id: str,
    *,
    passed: bool,
    oracle: int,
    triggered: bool = True,
    short_passed: bool = False,
    anchor_passed: bool = False,
    short_confidence: float = 0.9,
    anchor_confidence: float = 0.7,
) -> dict:
    candidates = []
    if triggered:
        candidates = [
            make_candidate(SHORT_ID, passed=short_passed, confidence=short_confidence, top1=short_confidence),
            make_candidate(ANCHOR_ID, passed=anchor_passed, confidence=anchor_confidence, top1=anchor_confidence),
            make_candidate("len32_s96", passed=anchor_passed, confidence=anchor_confidence, top1=anchor_confidence),
        ]
    return {
        "task_id": task_id,
        "metrics": {"passed": passed, "oracle_mask_length": oracle},
        "route2_rescue_quality_v5": {
            "triggered": triggered,
            "selected_candidate_id": ANCHOR_ID if triggered else "primary",
            "candidates": candidates,
        },
    }


class Route2V6ParserTest(unittest.TestCase):
    def test_candidate_by_id_and_candidate_pass(self) -> None:
        row = make_v5_row("a", passed=True, oracle=20, short_passed=True)

        candidates = candidate_by_id(row)

        self.assertIn(SHORT_ID, candidates)
        self.assertTrue(candidate_pass(candidates[SHORT_ID]))
        self.assertFalse(candidate_pass(candidates[ANCHOR_ID]))

    def test_policy_features_exclude_forbidden_labels(self) -> None:
        candidates = candidate_by_id(make_v5_row("a", passed=True, oracle=20, short_passed=True))

        features = build_policy_features(candidates)

        self.assertIn("short_trace_confidence_last", features)
        self.assertIn("delta_short_minus_anchor_trace_confidence_last", features)
        forbidden = " ".join(features)
        self.assertNotIn("oracle", forbidden)
        self.assertNotIn("passed", forbidden)
        self.assertNotIn("offline", forbidden)
        self.assertNotIn("task", forbidden)

    def test_build_triggered_table_reproduces_summary(self) -> None:
        rows = [
            make_v5_row("a", passed=True, oracle=20, short_passed=True, anchor_passed=False),
            make_v5_row("b", passed=False, oracle=8, short_passed=False, anchor_passed=True),
            make_v5_row("c", passed=True, oracle=8, triggered=False),
        ]
        reference = [
            {"task_id": "a", "metrics": {"passed": False}},
            {"task_id": "b", "metrics": {"passed": False}},
            {"task_id": "c", "metrics": {"passed": True}},
        ]

        triggered, summary = build_triggered_table(rows, reference_rows=reference)

        self.assertEqual(summary["num_samples"], 3)
        self.assertEqual(summary["v5_pass_count"], 2)
        self.assertEqual(summary["triggered_count"], 2)
        self.assertEqual(summary["candidate_upper_bound_count"], 2)
        self.assertEqual(summary["pairwise_vs_reference"]["win"], 1)
        self.assertEqual(summary["pairwise_vs_reference"]["tie_pass"], 1)
        self.assertEqual(len(feature_columns(triggered)), len(set(feature_columns(triggered))))


class Route2V6RuleTest(unittest.TestCase):
    def test_score_rule_policy_candidate_when_short_captures_without_losses(self) -> None:
        triggered, _ = build_triggered_table(
            [
                make_v5_row(
                    "short-only",
                    passed=False,
                    oracle=15,
                    short_passed=True,
                    anchor_passed=False,
                    short_confidence=0.95,
                    anchor_confidence=0.5,
                ),
                make_v5_row(
                    "anchor-only",
                    passed=True,
                    oracle=21,
                    short_passed=False,
                    anchor_passed=True,
                    short_confidence=0.2,
                    anchor_confidence=0.9,
                ),
            ]
        )
        rule = Rule(
            name="short_conf_high",
            family="single",
            clauses=(("short_trace_confidence_last", ">=", 0.9),),
        )

        scored = score_rule(rule, triggered)

        self.assertEqual(scored["captured_short_only_wins"], 1)
        self.assertEqual(scored["anchor_losses"], 0)
        self.assertEqual(scored["short_risk_count"], 0)
        self.assertEqual(scored["decision"], "policy_candidate")

    def test_final_decision_prefers_policy_candidate(self) -> None:
        self.assertEqual(final_decision([{"decision": "policy_candidate"}]), "policy_candidate")
        self.assertEqual(final_decision([{"decision": "diagnostic_only"}]), "diagnostic_only")
        self.assertEqual(final_decision([{"decision": "reject"}]), "reject_selector_only")


class Route2V6IoTest(unittest.TestCase):
    def test_write_outputs(self) -> None:
        triggered, summary = build_triggered_table(
            [make_v5_row("a", passed=True, oracle=20, short_passed=True, anchor_passed=False)]
        )
        summary["decision"] = "diagnostic_only"
        summary["top_rules"] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_outputs(output_dir, summary, triggered, [])

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "triggered_candidate_table.csv").exists())
            self.assertTrue((output_dir / "rule_candidates.csv").exists())
            self.assertTrue((output_dir / "report.md").exists())
            loaded = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["triggered_count"], 1)


if __name__ == "__main__":
    unittest.main()
