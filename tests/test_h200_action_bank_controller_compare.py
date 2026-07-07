from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from experiments.action_ceiling.h200_action_bank_controller_compare import (
    add_labels,
    bank_scope_summary,
    compare_controllers,
)


class H200ActionBankControllerCompareTest(unittest.TestCase):
    def test_bank_scope_summary_tracks_outcome_and_label_agreement(self) -> None:
        old_rows = add_labels(
            [
                {"task_id": "a", "action": "KEEP_PRIMARY", "split": "validation", "passed": False},
                {"task_id": "a", "action": "EXPAND_16", "split": "validation", "passed": True, "generated_text_sha256": "old"},
                {"task_id": "b", "action": "KEEP_PRIMARY", "split": "validation", "passed": True},
                {"task_id": "b", "action": "EXPAND_16", "split": "validation", "passed": False},
            ]
        )
        h200_rows = add_labels(
            [
                {"task_id": "a", "action": "KEEP_PRIMARY", "split": "validation", "passed": False},
                {"task_id": "a", "action": "EXPAND_16", "split": "validation", "passed": True, "generated_text_sha256": "new"},
                {"task_id": "b", "action": "KEEP_PRIMARY", "split": "validation", "passed": True},
                {"task_id": "b", "action": "EXPAND_16", "split": "validation", "passed": True},
            ]
        )
        summary = bank_scope_summary("toy", old_rows, h200_rows)
        self.assertEqual(summary["h200_wins"], 1)
        self.assertEqual(summary["h200_losses"], 0)
        self.assertEqual(summary["old_benefit_count_non_keep"], 1)
        self.assertEqual(summary["h200_benefit_count_non_keep"], 1)
        self.assertEqual(summary["old_harm_count_non_keep"], 1)
        self.assertEqual(summary["h200_harm_count_non_keep"], 0)
        self.assertAlmostEqual(summary["outcome_agreement_rate"], 0.75)

    def test_compare_controllers_reports_selected_gate_and_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / "old"
            h200_dir = root / "h200"
            old_dir.mkdir()
            h200_dir.mkdir()
            for directory, passed, gate in [(old_dir, 90, False), (h200_dir, 89, False)]:
                (directory / "summary.json").write_text(
                    json.dumps(
                        {
                            "validation_gate": {
                                "selected_controller": {
                                    "model_family": "logistic",
                                    "feature_variant": "probe_only",
                                    "policy_variant": "benefit_only",
                                    "score_threshold": 999.0,
                                },
                                "gate_passed": gate,
                                "test_decision": "sealed",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                (directory / "controller_validation_results.csv").write_text(
                    "model_family,feature_variant,policy_variant,score_threshold,validation_pass_count,validation_total,validation_wins_vs_primary,validation_losses_vs_primary\n"
                    f"logistic,probe_only,benefit_only,999.0,{passed},127,0,0\n",
                    encoding="utf-8",
                )
                (directory / "validation_baselines.csv").write_text(
                    "baseline,validation_pass_count,validation_total,validation_wins_vs_primary,validation_losses_vs_primary\n"
                    f"v6,{passed},127,0,0\n",
                    encoding="utf-8",
                )
            rows = compare_controllers(old_dir, h200_dir)
            selected = rows[0]
            self.assertEqual(selected["entity"], "selected_controller")
            self.assertEqual(selected["pass_delta_h200_minus_old"], -1)
            self.assertEqual(selected["old_test_decision"], "sealed")
            self.assertEqual(selected["h200_test_decision"], "sealed")
            self.assertEqual(rows[1]["entity"], "baseline:v6")


if __name__ == "__main__":
    unittest.main()
