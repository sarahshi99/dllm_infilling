from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.phase5_premise_falsification import (
    DEPLOYABLE_PROXY_SCORE_KEYS,
    FORBIDDEN_DEPLOYABLE_FEATURES,
    auc,
    bridge_features,
    corrected_deployable_gate,
    deployable_bridge_formula_spec,
    deployable_proxy_scores,
    forbidden_feature_audit,
    grouped_bootstrap_metric,
    pairwise_ranking_accuracy,
    semantic_units,
    validate_feature_names,
)
from analysis.phase5_semantic_bridge_v0 import (
    validate_deployable_score_key,
    validate_proxy_score_schema,
    write_killed_outputs,
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

    def test_corrected_gate_requires_within_task_deltas_and_short_safety(self) -> None:
        comparisons = {
            name: {"delta_primary": 0.02, "delta_ci_low": 0.001, "delta_ci_high": 0.04}
            for name in ["prefix_only", "suffix_only", "token_canvas", "ordinary_confidence"]
        }
        selection = {
            "vs_fixed64": {"net": 1, "short_net": 0},
            "vs_confidence": {"net": 1, "short_net": 0},
        }
        gate = corrected_deployable_gate(
            global_auc=0.9,
            within_task_accuracy=0.6,
            cross_canvas_accuracy=0.58,
            comparisons=comparisons,
            selection=selection,
        )
        self.assertTrue(gate["passed"])

        comparisons["suffix_only"] = {"delta_primary": 0.02, "delta_ci_low": -0.001, "delta_ci_high": 0.04}
        self.assertFalse(
            corrected_deployable_gate(
                global_auc=0.9,
                within_task_accuracy=0.6,
                cross_canvas_accuracy=0.58,
                comparisons=comparisons,
                selection=selection,
            )["passed"]
        )

    def test_global_auc_alone_cannot_authorize_v0(self) -> None:
        gate = corrected_deployable_gate(
            global_auc=1.0,
            within_task_accuracy=0.5,
            cross_canvas_accuracy=0.5,
            comparisons={},
            selection={"vs_fixed64": {"net": 10, "short_net": 0}, "vs_confidence": {"net": 10, "short_net": 0}},
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["global_auc_role"], "secondary_diagnostic_only")

    def test_deployable_formula_is_fixed_and_has_no_fitted_weights(self) -> None:
        spec = deployable_bridge_formula_spec()
        payload = json.dumps(spec, sort_keys=True)
        self.assertEqual(spec["mechanism_name"], "AST/def-use bridge proxy V0")
        self.assertEqual(spec["provenance"], "fixed_before_outcomes")
        self.assertNotIn("fitted", spec["parameter_source"])
        self.assertNotIn('"weights"', payload)
        self.assertNotIn("logistic", payload.lower())
        row = {
            "prefix_candidate_use_coverage": 1.0,
            "prefix_available_use_coverage": 0.5,
            "prefix_boundary_indent_match": 1.0,
            "prefix_candidate_def_use_count": 2.0,
            "suffix_required_recovery": 1.0,
            "suffix_boundary_indent_match": 1.0,
            "candidate_suffix_def_use_count": 1.0,
            "full_parse_passed": 1.0,
            "candidate_control_structure_count": 1.0,
            "ordinary_confidence": 0.8,
            "candidate_canvas_fill_ratio": 0.5,
        }
        scores = deployable_proxy_scores(row)
        self.assertEqual(set(scores), set(DEPLOYABLE_PROXY_SCORE_KEYS))
        self.assertGreater(scores["deployable_proxy_combined"], 0.0)

    def test_supervised_probe_scores_cannot_enter_v0(self) -> None:
        validate_deployable_score_key("deployable_proxy_combined")
        with self.assertRaises(ValueError):
            validate_deployable_score_key("supervised_probe_score_combined_bridge")
        with self.assertRaises(ValueError):
            validate_deployable_score_key("score_combined_bridge")
        with self.assertRaises(ValueError):
            validate_proxy_score_schema(
                [
                    {
                        "candidate_key": "c",
                        "row_key": "r",
                        "canvas_tokens": 64,
                        "seed": 0,
                        "deployable_proxy_combined": 0.5,
                        "supervised_probe_score_combined_bridge": 0.9,
                    }
                ]
            )

    def test_forbidden_audit_reports_indirect_label_use(self) -> None:
        audit = forbidden_feature_audit()
        self.assertTrue(audit["labels_used_for_offline_evaluation"])
        self.assertTrue(audit["supervised_probe_diagnostic"]["outcome_labels_used_for_fit"])
        self.assertFalse(audit["deployable_bridge_proxy"]["outcome_labels_used_for_fit"])
        self.assertFalse(audit["deployable_bridge_proxy"]["reference_used_for_fit"])
        self.assertFalse(audit["deployable_bridge_proxy"]["outcome_labels_used_for_selection"])
        self.assertFalse(audit["deployable_bridge_proxy"]["reference_used_for_selection"])
        self.assertTrue(audit["deployable_bridge_proxy"]["outcome_labels_used_for_offline_gate_evaluation"])

    def test_killed_report_is_written_when_corrected_gate_fails(self) -> None:
        gate = {"passed": False, "failed_conditions": ["cross_canvas_accuracy_above_chance"]}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_killed_outputs(output, gate)
            summary = json.loads((output / "summary.json").read_text())
            report = (output / "report.md").read_text()
            self.assertEqual(summary["verdict"], "killed_corrected_within_task_gate_failed")
            self.assertIn("AST/def-use bridge proxy V0", report)
            self.assertIn("cross_canvas_accuracy_above_chance", report)

    def test_forbidden_features_are_rejected(self) -> None:
        for forbidden in FORBIDDEN_DEPLOYABLE_FEATURES:
            with self.assertRaises(ValueError):
                validate_feature_names(["mean_final_confidence", forbidden])
        validate_feature_names(["mean_final_confidence", "combined_bridge_satisfaction"])


if __name__ == "__main__":
    unittest.main()
