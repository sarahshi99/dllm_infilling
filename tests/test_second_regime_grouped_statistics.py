from __future__ import annotations

import unittest

from analysis.second_regime_grouped_statistics import (
    group_label_swap_test,
    macro_accuracy,
    paired_task_outcomes,
    taxonomic_intersection,
)


def row(index: int, group: str, policy: str, passed: bool) -> dict[str, object]:
    return {
        "manifest_index": str(index),
        "task_group": group,
        "policy": policy,
        "passed": passed,
    }


class GroupedStatisticsTest(unittest.TestCase):
    def test_macro_weights_tasks_equally_not_spans(self) -> None:
        rows = [
            row(0, "HumanEval/0", "control_fixed64", True),
            row(1, "HumanEval/0", "control_fixed64", True),
            row(2, "HumanEval/0", "control_fixed64", True),
            row(3, "HumanEval/1", "control_fixed64", False),
        ]
        self.assertEqual(macro_accuracy(rows, "control_fixed64"), 0.5)

    def test_paired_outcomes_are_task_level(self) -> None:
        rows = [
            row(0, "HumanEval/0", "control_fixed64", False),
            row(0, "HumanEval/0", "best_deployable_cal_lite_alpha006", True),
            row(1, "HumanEval/1", "control_fixed64", True),
            row(1, "HumanEval/1", "best_deployable_cal_lite_alpha006", False),
        ]
        paired = paired_task_outcomes(rows, "best_deployable_cal_lite_alpha006", "control_fixed64")
        self.assertEqual([item["outcome"] for item in paired], ["win", "loss"])
        result = group_label_swap_test(paired, replicates=10, seed=1)
        self.assertEqual(result["cluster_count"], 2)

    def test_intersection_has_all_eight_cells(self) -> None:
        rows = []
        policies = ("control_fixed64", "best_deployable_cal_lite_alpha006", "oracle_sufficient_canvas")
        for index, flags in enumerate([(False, False, False), (True, True, True)]):
            for policy, passed in zip(policies, flags):
                rows.append(row(index, f"HumanEval/{index}", policy, passed))
        cells = taxonomic_intersection(rows)
        self.assertEqual(len(cells), 8)
        self.assertEqual(sum(int(cell["span_count"]) for cell in cells), 2)


if __name__ == "__main__":
    unittest.main()
