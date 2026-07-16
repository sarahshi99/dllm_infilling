from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.m2_constraint_homotopy import METHODS, activation_audit, write_activation_artifacts


def row(row_key: str, code_hash: str, middle_hash: str, updates: int, changes: int) -> dict[str, object]:
    return {
        "row_key": row_key,
        "candidate_full_code_sha256": code_hash,
        "candidate_middle_sha256": middle_hash,
        "metrics": {
            "effective_update_steps": updates,
            "total_token_changes": changes,
        },
    }


class M2ConstraintHomotopyAnalysisTest(unittest.TestCase):
    def test_activation_audit_uses_hashes_and_aggregate_dynamics_only(self) -> None:
        rows_by_method = {
            METHODS[0]: [row("a", "vanilla-a", "v-a", 3, 9), row("b", "same", "same", 4, 10)],
            METHODS[1]: [row("a", "gradual-a", "g-a", 5, 12), row("b", "same", "same", 6, 15)],
            METHODS[2]: [row("a", "abrupt-a", "a-a", 2, 6), row("b", "same", "same", 3, 7)],
        }
        audit = activation_audit(rows_by_method)
        pairs = {item["comparison"]: item for item in audit["pairwise_hash_differences"]}
        self.assertEqual(pairs["gradual_vs_vanilla"]["candidate_full_code_sha256_different_count"], 1)
        self.assertEqual(audit["dynamics"][0]["effective_update_steps"]["count"], 2)
        self.assertFalse(audit["raw_generated_code_included"])

    def test_written_activation_artifacts_do_not_include_raw_code(self) -> None:
        rows_by_method = {
            METHODS[0]: [row("a", "v", "v", 1, 2)],
            METHODS[1]: [row("a", "g", "g", 2, 3)],
            METHODS[2]: [row("a", "a", "a", 3, 4)],
        }
        summary = {
            "paired_effects": [
                {"method_a": METHODS[1], "method_b": METHODS[0], "task_macro_ci_low": -0.1, "task_macro_ci_high": 0.1},
                {"method_a": METHODS[2], "method_b": METHODS[0], "task_macro_ci_low": -0.1, "task_macro_ci_high": 0.1},
                {"method_a": METHODS[1], "method_b": METHODS[2], "task_macro_ci_low": -0.1, "task_macro_ci_high": 0.1},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_activation_artifacts(output_dir, rows_by_method, summary)
            payload = json.loads((output_dir / "activation_audit.json").read_text(encoding="utf-8"))
            self.assertNotIn("middle_text", json.dumps(payload))
            self.assertNotIn("code\":", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
