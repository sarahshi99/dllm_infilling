from __future__ import annotations

import unittest
from types import SimpleNamespace

from clean_scripts.run_lcal_official_bounded_repair import (
    BoundedRepairSettings,
    apply_proportional_widening,
    correction_grid_has_usable_candidate,
    filter_tasks_by_ids,
    parse_task_ids_csv,
)
from clean_scripts.run_cal_lite_lcal_rescue_policy import LcalV3Settings
from clean_scripts.run_cal_official_lcas_v3 import OfficialCalSettings


def candidate(length: int, score: float, raw: float | None = None) -> dict:
    return {
        "mask_length": length,
        "score": score,
        "adjusted_score": score,
        "raw_score": score if raw is None else raw,
    }


def base_selection(candidates: list[dict], selected: int = 12) -> dict:
    selected_candidate = next(item for item in candidates if item["mask_length"] == selected)
    return {
        "selected_mask_length": selected,
        "selected_score": selected_candidate["score"],
        "selected_raw_score": selected_candidate["raw_score"],
        "selected_adjusted_score": selected_candidate["adjusted_score"],
        "candidate_scores": candidates,
        "length_probe_sec": 1.0,
        "probe_lengths": [item["mask_length"] for item in candidates],
        "tie_break": "shorter",
        "score_mode": "length_power",
        "length_alpha": 0.06,
    }


def settings(**overrides) -> BoundedRepairSettings:
    values = {
        "proportional_widening": True,
        "prop_base_threshold": 0.97,
        "prop_slope": 0.08,
        "prop_min_threshold": 0.85,
        "prop_min_base_length": 8,
        "prop_max_expansion": 3.0,
        "prop_require_raw_confirm": False,
    }
    values.update(overrides)
    return BoundedRepairSettings(
        lcal=LcalV3Settings(
            base_probe_lengths_csv="8,12,24",
            base_alpha=0.06,
            weak_probe_lengths_csv="13,14,15,16",
            strong_probe_lengths_csv="13,14,15,16,20,24",
            long_alpha=0.10,
            strong_min_len=13,
            weak_min_base_len=8,
            weak_max_base_len=12,
            ratio_trigger_threshold=0.985,
            long_score_floor=0.55,
            raw_ratio_threshold=0.97,
            support_count_threshold=2,
            cap_base_le8=14,
            cap_base_9_12=16,
            short_safe_policy="s3",
            correction_selection_rule="shortest_supported",
            shortest_supported_ratio=0.985,
            tie_break="shorter",
            score_mode="length_power",
        ),
        official=OfficialCalSettings(),
        **values,
    )


class LcalOfficialBoundedRepairProportionalTest(unittest.TestCase):
    def test_apply_proportional_widening_selects_long_near_best_candidate(self) -> None:
        selection = apply_proportional_widening(
            base_selection(
                [
                    candidate(12, 1.00),
                    candidate(16, 0.95),
                    candidate(24, 0.92),
                ],
                selected=12,
            ),
            settings(),
        )

        self.assertEqual(selection["selected_mask_length"], 24)
        self.assertTrue(selection["proportional_widening"]["promoted"])
        self.assertEqual(selection["proportional_widening"]["original_selected_length"], 12)
        self.assertEqual(selection["proportional_widening"]["proportional_selected_length"], 24)

    def test_raw_confirmation_blocks_low_raw_long_candidate(self) -> None:
        selection = apply_proportional_widening(
            base_selection(
                [
                    candidate(12, 1.00, raw=1.00),
                    candidate(24, 0.92, raw=0.20),
                ],
                selected=12,
            ),
            settings(prop_require_raw_confirm=True),
        )

        self.assertEqual(selection["selected_mask_length"], 12)
        self.assertFalse(selection["proportional_widening"]["promoted"])

    def test_min_base_length_keeps_short_best_candidate(self) -> None:
        selection = apply_proportional_widening(
            base_selection(
                [
                    candidate(8, 1.00),
                    candidate(16, 0.95),
                ],
                selected=8,
            ),
            settings(prop_min_base_length=12),
        )

        self.assertEqual(selection["selected_mask_length"], 8)
        self.assertFalse(selection["proportional_widening"]["promoted"])

    def test_task_id_filter_preserves_requested_order(self) -> None:
        tasks = [SimpleNamespace(task_id="a"), SimpleNamespace(task_id="b"), SimpleNamespace(task_id="c")]

        filtered = filter_tasks_by_ids(tasks, parse_task_ids_csv("c,a"))

        self.assertEqual([task.task_id for task in filtered], ["c", "a"])

    def test_correction_grid_guard_detects_grid_below_widened_base(self) -> None:
        self.assertFalse(correction_grid_has_usable_candidate("13,14,15,16,20,24,28,32,40", 48))
        self.assertTrue(correction_grid_has_usable_candidate("13,14,15,16,20,24,28,32,40,48", 48))


if __name__ == "__main__":
    unittest.main()
