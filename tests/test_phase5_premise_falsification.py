from __future__ import annotations

import unittest

from analysis.phase5_premise_falsification import (
    FORBIDDEN_DEPLOYABLE_FEATURES,
    auc,
    bridge_features,
    combined_bridge_gate,
    grouped_bootstrap_metric,
    pairwise_ranking_accuracy,
    semantic_units,
    validate_feature_names,
)


class Phase5PremiseFalsificationTest(unittest.TestCase):
    def test_semantic_units_capture_middle_identifiers_and_control(self) -> None:
        prefix = "def f(xs):\n    total = 0\n"
        middle = "    for value in xs:\n        total += value\n"
        suffix = "    return total\n"
        units = semantic_units(prefix, middle, suffix)
        self.assertIn("control:For", units)
        self.assertIn("def:value", units)
        self.assertIn("use:xs", units)
        self.assertIn("def:total", units)

    def test_bridge_features_reward_suffix_required_definition(self) -> None:
        prefix = "def f(xs):\n    total = 0\n"
        good_middle = "    for value in xs:\n        total += value\n"
        bad_middle = "    pass\n"
        suffix = "    return total\n"
        good = bridge_features(prefix, good_middle, suffix)
        bad = bridge_features(prefix, bad_middle, suffix)
        self.assertGreaterEqual(good["combined_bridge_satisfaction"], bad["combined_bridge_satisfaction"])
        self.assertGreater(good["prefix_candidate_def_use_count"], bad["prefix_candidate_def_use_count"])

    def test_auc_and_pairwise_ranking(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(auc(labels, scores), 1.0)
        rows = [
            {"group": "a", "passed": False, "score": 0.1},
            {"group": "a", "passed": True, "score": 0.8},
            {"group": "b", "passed": False, "score": 0.2},
            {"group": "b", "passed": True, "score": 0.9},
        ]
        self.assertEqual(pairwise_ranking_accuracy(rows, "score"), 1.0)

    def test_grouped_bootstrap_is_deterministic_and_grouped(self) -> None:
        rows = [
            {"group": "a", "passed": False, "score": 0.1},
            {"group": "a", "passed": True, "score": 0.8},
            {"group": "b", "passed": False, "score": 0.2},
            {"group": "b", "passed": True, "score": 0.9},
        ]
        first = grouped_bootstrap_metric(rows, "group", lambda sample: pairwise_ranking_accuracy(sample, "score"), seed=7, replicates=50)
        second = grouped_bootstrap_metric(rows, "group", lambda sample: pairwise_ranking_accuracy(sample, "score"), seed=7, replicates=50)
        self.assertEqual(first, second)
        self.assertEqual(first["estimate"], 1.0)

    def test_combined_bridge_gate_requires_margin_and_positive_intervals(self) -> None:
        comparisons = {
            name: {"delta_auc": 0.02, "delta_ci_low": 0.001, "delta_ci_high": 0.04}
            for name in ["prefix_only", "suffix_only", "token_length", "ordinary_confidence"]
        }
        self.assertTrue(combined_bridge_gate(comparisons)["passed"])
        comparisons["suffix_only"] = {"delta_auc": 0.009, "delta_ci_low": 0.001, "delta_ci_high": 0.02}
        self.assertFalse(combined_bridge_gate(comparisons)["passed"])

    def test_forbidden_features_are_rejected(self) -> None:
        for forbidden in FORBIDDEN_DEPLOYABLE_FEATURES:
            with self.assertRaises(ValueError):
                validate_feature_names(["mean_final_confidence", forbidden])
        validate_feature_names(["mean_final_confidence", "combined_bridge_satisfaction"])


if __name__ == "__main__":
    unittest.main()
