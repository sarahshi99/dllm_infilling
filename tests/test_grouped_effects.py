from __future__ import annotations

import unittest

from analysis.grouped_effects import method_summary, paired_effect


class GroupedEffectsTest(unittest.TestCase):
    def test_task_macro_is_equal_weight_and_pairing_is_group_aware(self) -> None:
        left = [
            {"row_key": "a0", "task_group": "A", "status": "ok", "passed": True},
            {"row_key": "a1", "task_group": "A", "status": "ok", "passed": False},
            {"row_key": "b0", "task_group": "B", "status": "ok", "passed": True},
        ]
        right = [
            {"row_key": "a0", "task_group": "A", "status": "ok", "passed": False},
            {"row_key": "a1", "task_group": "A", "status": "ok", "passed": False},
            {"row_key": "b0", "task_group": "B", "status": "ok", "passed": True},
        ]
        summary = method_summary(left, "left", bootstrap_replicates=20, seed=1)
        self.assertEqual(summary["task_group_count"], 2)
        self.assertAlmostEqual(summary["equal_weight_task_macro_accuracy"], 0.75)
        effect, tasks = paired_effect(left, right, method_a="left", method_b="right", bootstrap_replicates=20, seed=2)
        self.assertEqual(len(tasks), 2)
        self.assertEqual((effect["wins"], effect["losses"], effect["ties"]), (1, 0, 1))
        self.assertEqual((effect["span_help"], effect["span_harm"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
