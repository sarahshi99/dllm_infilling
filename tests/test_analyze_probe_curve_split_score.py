from __future__ import annotations

import unittest

from analysis import analyze_probe_curve_split_score as split_score


def record(
    task_id: str,
    *,
    failed_long: bool,
    true_long: bool,
    short: bool,
    passed: bool,
    features: dict[str, float],
) -> dict:
    return {
        "task_id": task_id,
        "failed_long": failed_long,
        "true_long": true_long,
        "short": short,
        "passed": passed,
        "features": features,
    }


class AnalyzeProbeCurveSplitScoreTest(unittest.TestCase):
    def test_stable_fold_is_deterministic_and_bounded(self) -> None:
        first = split_score.stable_fold("HumanEval/1", folds=5)
        second = split_score.stable_fold("HumanEval/1", folds=5)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 5)

    def test_fit_linear_signal_prefers_features_that_separate_failed_long(self) -> None:
        rows = [
            record(
                "pos-a",
                failed_long=True,
                true_long=True,
                short=False,
                passed=False,
                features={"long_low": 0.1, "noise": 0.5},
            ),
            record(
                "pos-b",
                failed_long=True,
                true_long=True,
                short=False,
                passed=False,
                features={"long_low": 0.2, "noise": 0.4},
            ),
            record(
                "risk-a",
                failed_long=False,
                true_long=False,
                short=True,
                passed=True,
                features={"long_low": 0.9, "noise": 0.5},
            ),
            record(
                "risk-b",
                failed_long=False,
                true_long=False,
                short=True,
                passed=True,
                features={"long_low": 0.8, "noise": 0.4},
            ),
        ]

        model = split_score.fit_linear_signal(rows, ["long_low", "noise"])
        scores = {row["task_id"]: split_score.score_record(row, model) for row in rows}

        self.assertLess(scores["risk-a"], scores["pos-a"])
        self.assertLess(scores["risk-b"], scores["pos-b"])
        self.assertLess(model["weights"]["long_low"], 0.0)

    def test_train_threshold_then_evaluate_heldout_uses_selected_threshold(self) -> None:
        train = [
            record(
                "pos-a",
                failed_long=True,
                true_long=True,
                short=False,
                passed=False,
                features={"risk": 2.0},
            ),
            record(
                "pos-b",
                failed_long=True,
                true_long=True,
                short=False,
                passed=False,
                features={"risk": 1.8},
            ),
            record(
                "risk-a",
                failed_long=False,
                true_long=False,
                short=True,
                passed=True,
                features={"risk": -1.0},
            ),
            record(
                "risk-b",
                failed_long=False,
                true_long=False,
                short=True,
                passed=True,
                features={"risk": -1.2},
            ),
        ]
        heldout = [
            record(
                "pos-heldout",
                failed_long=True,
                true_long=True,
                short=False,
                passed=False,
                features={"risk": 1.9},
            ),
            record(
                "short-heldout",
                failed_long=False,
                true_long=False,
                short=True,
                passed=True,
                features={"risk": -0.9},
            ),
        ]
        model = {
            "features": ["risk"],
            "means": {"risk": 0.0},
            "stdevs": {"risk": 1.0},
            "weights": {"risk": 1.0},
        }
        train_scores = [split_score.score_record(row, model) for row in train]

        threshold = split_score.select_threshold(
            train,
            train_scores,
            min_failed_long_triggers=2,
            max_short_risk=0.05,
        )
        heldout_scores = [split_score.score_record(row, model) for row in heldout]
        result = split_score.evaluate_scored_records(
            heldout,
            heldout_scores,
            threshold["threshold"],
        )

        self.assertEqual(result["trigger_count"], 1)
        self.assertEqual(result["failed_long_trigger_count"], 1)
        self.assertAlmostEqual(result["short_risk_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
