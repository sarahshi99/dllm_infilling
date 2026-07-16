from __future__ import annotations

import unittest

from analysis.phase6_score_only_candidate_selection import METHODS, selection_record


def grid_row(canvas: int, seed: int, wall_sec: float) -> dict[str, object]:
    return {
        "candidate_key": f"row|canvas={canvas}|seed={seed}",
        "row_key": "row",
        "task_group": "HumanEval/0",
        "length_bucket": "medium",
        "canvas_tokens": canvas,
        "seed": seed,
        "passed": False,
        "metrics": {"total_sec_including_probe": wall_sec},
    }


class Phase6ScoreOnlyCostTest(unittest.TestCase):
    def test_fixed64_and_grid_selector_use_distinct_standalone_costs(self) -> None:
        rows = [
            grid_row(canvas, seed, float(index + 1))
            for index, (canvas, seed) in enumerate(
                ((16, 0), (16, 1), (32, 0), (32, 1), (64, 0), (64, 1), (128, 0), (128, 1))
            )
        ]
        fixed64 = rows[4]

        fixed = selection_record(METHODS[0], fixed64, rows)
        selector = selection_record(METHODS[1], rows[-1], rows)

        self.assertEqual(fixed["metrics"]["standalone_actual_forward_count"], 64)
        self.assertEqual(fixed["metrics"]["standalone_token_budget"], 64 * 64)
        self.assertEqual(fixed["metrics"]["total_sec_including_probe"], 5.0)
        self.assertEqual(selector["metrics"]["standalone_actual_forward_count"], 512)
        self.assertEqual(selector["metrics"]["standalone_token_budget"], 2 * (16 + 32 + 64 + 128) * 64)
        self.assertEqual(selector["metrics"]["selected_candidate_wall_sec"], 8.0)
        self.assertEqual(selector["metrics"]["total_sec_including_probe"], 36.0)


if __name__ == "__main__":
    unittest.main()
