from __future__ import annotations

import unittest

from analysis import analyze_probe_curve_long_signals as audit


def candidate(length: int, score: float, raw: float | None = None) -> dict:
    return {
        "mask_length": length,
        "score": score,
        "raw_score": score if raw is None else raw,
        "mean_top2_gap": score / 2.0,
    }


def row(
    task_id: str,
    *,
    passed: bool,
    oracle: int,
    selected: int,
    scores: list[dict],
) -> dict:
    return {
        "task_id": task_id,
        "metrics": {
            "passed": passed,
            "oracle_mask_length": oracle,
            "selected_mask_length": selected,
            "final_source": "base",
        },
        "lcal_v3": {
            "base_candidate_scores": scores,
        },
        "length_probe": {
            "base_candidate_scores": scores,
        },
    }


class AnalyzeProbeCurveLongSignalsTest(unittest.TestCase):
    def test_extract_features_summarizes_probe_curve_shape(self) -> None:
        item = row(
            "example",
            passed=False,
            oracle=20,
            selected=6,
            scores=[
                candidate(3, 0.4),
                candidate(6, 0.6),
                candidate(13, 0.7),
                candidate(20, 0.9),
            ],
        )

        features = audit.extract_features(item)

        self.assertEqual(features["selected_len"], 6.0)
        self.assertEqual(features["best_len"], 20.0)
        self.assertAlmostEqual(features["best_score"], 0.9)
        self.assertAlmostEqual(features["selected_to_best_gap"], 14.0)
        self.assertAlmostEqual(features["long_score_max"], 0.9)
        self.assertAlmostEqual(features["short_score_max"], 0.6)
        self.assertAlmostEqual(features["long_minus_short_score"], 0.3)

    def test_evaluate_feature_threshold_reports_precision_recall_and_risk(self) -> None:
        rows = [
            row(
                "long-fail",
                passed=False,
                oracle=20,
                selected=6,
                scores=[candidate(6, 0.4), candidate(20, 0.9)],
            ),
            row(
                "short-pass",
                passed=True,
                oracle=6,
                selected=6,
                scores=[candidate(6, 0.8), candidate(20, 0.7)],
            ),
            row(
                "medium-pass",
                passed=True,
                oracle=12,
                selected=12,
                scores=[candidate(12, 0.8), candidate(20, 0.6)],
            ),
        ]
        records = [audit.build_record(item) for item in rows]

        result = audit.evaluate_threshold(
            records,
            feature="long_minus_short_score",
            direction=">=",
            threshold=0.2,
        )

        self.assertEqual(result["trigger_count"], 1)
        self.assertEqual(result["true_long_count"], 1)
        self.assertEqual(result["failed_long_trigger_count"], 1)
        self.assertAlmostEqual(result["true_long_precision"], 1.0)
        self.assertAlmostEqual(result["failed_long_recall"], 1.0)
        self.assertAlmostEqual(result["short_risk_rate"], 0.0)
        self.assertAlmostEqual(result["current_pass_risk_rate"], 0.0)

    def test_sweep_features_returns_ranked_candidates(self) -> None:
        rows = [
            row(
                "long-fail",
                passed=False,
                oracle=20,
                selected=6,
                scores=[candidate(6, 0.4), candidate(20, 0.9)],
            ),
            row(
                "short-pass",
                passed=True,
                oracle=6,
                selected=6,
                scores=[candidate(6, 0.8), candidate(20, 0.7)],
            ),
        ]

        results = audit.sweep_feature_thresholds([audit.build_record(item) for item in rows])

        self.assertTrue(results)
        self.assertIn("feature", results[0])
        self.assertIn("score", results[0])


if __name__ == "__main__":
    unittest.main()
