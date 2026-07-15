from __future__ import annotations

import unittest

from experiments.m1_abductive_program_state_bridge import (
    CANVASES,
    SEEDS,
    expected_grid_keys,
    expected_oracle_keys,
    grid_specs,
    parser,
    refinement_audit,
    stage1_audit,
    stage1_key,
)


class M1AbductiveProgramStateBridgeTest(unittest.TestCase):
    def test_stage_one_has_eight_deployable_candidates_and_a_separate_oracle(self) -> None:
        self.assertEqual(len(grid_specs()), 8)
        self.assertEqual({(row["canvas_tokens"], row["seed"]) for row in grid_specs()}, {(canvas, seed) for canvas in CANVASES for seed in SEEDS})
        manifest = [{"row_key": "a", "reference_middle_tokens": 9}]
        self.assertEqual(len(expected_grid_keys(manifest)), 8)
        self.assertEqual(len(expected_oracle_keys(manifest)), 1)

    def test_stage_one_and_refinement_budget_audits_are_strict(self) -> None:
        manifest = [{"row_key": "a", "reference_middle_tokens": 9}]
        grid_rows = [
            {"candidate_key": stage1_key("a", "deployable_grid", spec["canvas_tokens"], spec["seed"]), "status": "ok", "metrics": {"actual_forward_count": 64}}
            for spec in grid_specs()
        ]
        self.assertTrue(stage1_audit(grid_rows, expected_grid_keys(manifest))["passed"])
        refinement = [{"candidate_key": "a|equal_compute_generic_remask|refinement=64", "status": "ok", "metrics": {"standalone_actual_forward_count": 576}}]
        self.assertTrue(refinement_audit(refinement, manifest, "equal_compute_generic_remask")["passed"])

    def test_parser_only_exposes_randomspanlight_auto_full(self) -> None:
        args = parser().parse_args(
            [
                "--dataset-jsonl", "data.jsonl",
                "--stage1-output-dir", "stage1",
                "--generic-output-dir", "generic",
                "--m1-output-dir", "m1",
                "--compact-dir", "compact",
            ]
        )
        self.assertEqual(args.smoke_cases, 12)
        self.assertFalse(args.auto_full)


if __name__ == "__main__":
    unittest.main()
