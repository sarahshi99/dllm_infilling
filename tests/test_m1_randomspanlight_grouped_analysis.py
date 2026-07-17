from __future__ import annotations

import unittest

from analysis.m1_randomspanlight_grouped_analysis import (
    COMPARISONS,
    METHODS,
    refinement_record,
    stage1_selection_record,
)


def grid_row(canvas: int, seed: int, wall_sec: float) -> dict[str, object]:
    return {
        "candidate_key": f"row|canvas={canvas}|seed={seed}",
        "row_key": "row",
        "task_group": "HumanEval/0",
        "length_bucket": "medium",
        "canvas_tokens": canvas,
        "seed": seed,
        "status": "ok",
        "passed": False,
        "metrics": {"total_sec_including_probe": wall_sec, "actual_forward_count": 64},
    }


class M1RandomSpanLightGroupedAnalysisTest(unittest.TestCase):
    def test_primary_comparisons_are_compute_matched_before_outcomes(self) -> None:
        self.assertEqual(COMPARISONS[:3], (
            ("m1_dependency_cone_full", "equal_compute_generic_remask"),
            ("m1_score_only_abductive_selector", "ordinary_confidence_best_of_grid"),
            ("m1_dependency_cone_full", "m1_score_only_abductive_selector"),
        ))

    def test_standalone_costs_are_fixed_before_outcomes(self) -> None:
        grid = [
            grid_row(canvas, seed, float(index + 1))
            for index, (canvas, seed) in enumerate(
                ((16, 0), (16, 1), (32, 0), (32, 1), (64, 0), (64, 1), (128, 0), (128, 1))
            )
        ]
        fixed = stage1_selection_record(METHODS[0], grid[4], grid)
        selector = stage1_selection_record(METHODS[3], grid[-1], grid)

        self.assertEqual(fixed["metrics"]["standalone_actual_forward_count"], 64)
        self.assertEqual(fixed["metrics"]["standalone_token_budget"], 4096)
        self.assertEqual(fixed["metrics"]["total_sec_including_probe"], 5.0)
        self.assertEqual(selector["metrics"]["standalone_actual_forward_count"], 512)
        self.assertEqual(selector["metrics"]["standalone_token_budget"], 30720)
        self.assertEqual(selector["metrics"]["total_sec_including_probe"], 36.0)

    def test_refinement_requires_full_standalone_576_forwards(self) -> None:
        row = {
            **grid_row(64, 0, 10.0),
            "metrics": {
                "standalone_actual_forward_count": 576,
                "standalone_token_budget": 34816,
                "total_sec_including_probe": 10.0,
            },
        }
        record = refinement_record(METHODS[4], row)
        self.assertEqual(record["metrics"]["standalone_actual_forward_count"], 576)

        row["metrics"]["standalone_actual_forward_count"] = 64
        with self.assertRaisesRegex(RuntimeError, "576"):
            refinement_record(METHODS[4], row)


if __name__ == "__main__":
    unittest.main()
