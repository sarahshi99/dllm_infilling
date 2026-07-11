from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase5_randomspanlight_candidate_bank import (
    build_alpha_renamed_task,
    build_candidate_specs,
    build_manifest,
    candidate_key,
    compact_candidate_row,
    expected_candidate_keys,
    audit_rows,
)


class Phase5CandidateBankTest(unittest.TestCase):
    def test_candidate_specs_have_eight_deployable_and_one_ceiling(self) -> None:
        specs = build_candidate_specs(reference_middle_tokens=23)
        self.assertEqual(len(specs), 9)
        deployable = [spec for spec in specs if spec["deployable"]]
        ceiling = [spec for spec in specs if not spec["deployable"]]
        self.assertEqual(len(deployable), 8)
        self.assertEqual(len(ceiling), 1)
        self.assertEqual(
            {(spec["canvas_tokens"], spec["seed"]) for spec in deployable},
            {(16, 0), (16, 1), (32, 0), (32, 1), (64, 0), (64, 1), (128, 0), (128, 1)},
        )
        fixed = [spec for spec in specs if spec["control_label"] == "fixed64_control"]
        self.assertEqual([(spec["canvas_tokens"], spec["seed"]) for spec in fixed], [(64, 0)])
        self.assertEqual(ceiling[0]["candidate_kind"], "oracle_sufficient_diagnostic_ceiling")
        self.assertEqual(ceiling[0]["canvas_tokens"], 23)

    def test_manifest_excludes_frozen_groups_and_has_stable_keys(self) -> None:
        rows = [
            {
                "task_id": f"RandomSpanInfillingLight/HumanEval/{idx}/1",
                "prompt": "def f(x):\n    y = x\n",
                "canonical_solution": "    return y\n",
                "suffix": "",
                "test": "def check(candidate):\n    assert candidate(1) == 1\n",
                "entry_point": "f",
            }
            for idx in range(4)
        ]

        class Tokenizer:
            @staticmethod
            def encode(text: str, add_special_tokens: bool = False) -> list[str]:
                del add_special_tokens
                return text.split()

        manifest = build_manifest(rows, Tokenizer(), {"HumanEval/1"})
        self.assertEqual(len(manifest), 3)
        self.assertNotIn("HumanEval/1", {row["task_group"] for row in manifest})
        self.assertEqual(len({row["row_key"] for row in manifest}), 3)

    def test_expected_key_audit_detects_missing_and_duplicates(self) -> None:
        manifest = [
            {"row_key": "r0", "reference_middle_tokens": 3},
            {"row_key": "r1", "reference_middle_tokens": 5},
        ]
        expected = expected_candidate_keys(manifest)
        rows = [{"candidate_key": key} for key in sorted(expected)]
        ok = audit_rows(rows, expected)
        self.assertTrue(ok["passed"])
        self.assertEqual(ok["missing_count"], 0)
        self.assertEqual(ok["duplicate_count"], 0)

        bad = audit_rows(rows[:-1] + [rows[0]], expected)
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["missing_count"], 1)
        self.assertEqual(bad["duplicate_count"], 1)

    def test_candidate_key_separates_grid_and_ceiling(self) -> None:
        grid = candidate_key("row", "deployable_grid", 64, 0)
        ceiling = candidate_key("row", "oracle_sufficient_diagnostic_ceiling", 64, 0)
        self.assertNotEqual(grid, ceiling)

    def test_alpha_renaming_changes_only_safe_local_identifiers(self) -> None:
        row = {
            "task_id": "RandomSpanInfillingLight/HumanEval/0/1",
            "entry_point": "f",
            "prompt": "def f(x):\n    total = x + 1\n",
            "canonical_solution": "    doubled = total * 2\n",
            "suffix": "    return doubled\n",
            "test": "def check(candidate):\n    assert candidate(2) == 6\n",
        }
        transformed = build_alpha_renamed_task(row)
        self.assertIsNotNone(transformed)
        assert transformed is not None
        self.assertEqual(transformed["entry_point"], "f")
        self.assertIn("phase5_local_", transformed["prompt"] + transformed["canonical_solution"] + transformed["suffix"])
        self.assertNotIn("phase5_local_", transformed["test"])
        namespace: dict[str, object] = {}
        exec(transformed["prompt"] + transformed["canonical_solution"] + transformed["suffix"] + transformed["test"], namespace)
        namespace["check"](namespace["f"])

    def test_alpha_renaming_rejects_dynamic_locals(self) -> None:
        row = {
            "task_id": "RandomSpanInfillingLight/HumanEval/0/1",
            "entry_point": "f",
            "prompt": "def f(x):\n    total = x + 1\n",
            "canonical_solution": "    seen = locals()\n",
            "suffix": "    return total\n",
            "test": "def check(candidate):\n    assert candidate(2) == 3\n",
        }
        self.assertIsNone(build_alpha_renamed_task(row))

    def test_compact_candidate_row_has_no_raw_code_or_reference(self) -> None:
        raw = {
            "candidate_key": "k",
            "row_key": "r",
            "task_group": "HumanEval/0",
            "candidate_kind": "deployable_grid",
            "control_label": "fixed64_control",
            "deployable": True,
            "canvas_tokens": 64,
            "seed": 0,
            "total_steps": 64,
            "status": "ok",
            "passed": True,
            "code": "secret raw code",
            "middle_text": "secret middle",
            "reference_middle": "secret reference",
            "candidate_full_code_sha256": "a",
            "candidate_middle_sha256": "b",
            "metrics": {"mean_final_confidence": 0.8, "total_sec_including_probe": 1.2},
            "verification": {},
        }
        compact = compact_candidate_row(raw)
        payload = json.dumps(compact)
        self.assertNotIn("secret", payload)
        self.assertNotIn("code", compact)
        self.assertNotIn("middle_text", compact)
        self.assertNotIn("reference_middle", compact)


if __name__ == "__main__":
    unittest.main()
