from __future__ import annotations

import unittest

from experiments.p1_official_cal_source_audit import (
    choose_smoke,
    evaluation_hash,
    map_demo_to_multiline,
    seed42_demo_and_rest,
)


def row(index: int) -> dict[str, str]:
    return {
        "task_id": f"MultiLineInfilling/HumanEval/{index}/L0",
        "prompt": f"def f_{index}():\n",
        "suffix": "    return value\n",
        "canonical_solution": f"    value = {index}\n",
        "test": f"def check(candidate): assert candidate() == {index}\n",
        "entry_point": f"f_{index}",
    }


class OfficialCalSourceAuditTest(unittest.TestCase):
    def test_seed42_demo_maps_by_normalized_evaluation_fields_not_filtered_index(self) -> None:
        single = [row(index) for index in range(1033)]
        demo, rest = seed42_demo_and_rest(single)
        multiline = [row(index) for index in range(1033)]
        mapping = map_demo_to_multiline(demo, multiline)
        self.assertEqual(len(demo), 100)
        self.assertEqual(len(rest), 933)
        self.assertEqual(len(mapping), 100)
        self.assertEqual(len(set(mapping.values())), 100)
        self.assertNotEqual(list(mapping)[:3], [0, 1, 2])
        self.assertEqual(evaluation_hash(single[0]), evaluation_hash(multiline[0]))

    def test_smoke_is_base_task_deduplicated_and_length_stratified(self) -> None:
        rows = []
        for group in range(16):
            rows.append(
                {
                    "source_row_id": group * 2,
                    "task_id": f"MultiLineInfilling/HumanEval/{group}/L0",
                    "task_group": f"HumanEval/{group}",
                    "completion_utf8_bytes": group + 1,
                }
            )
            rows.append(
                {
                    "source_row_id": group * 2 + 1,
                    "task_id": f"MultiLineInfilling/HumanEval/{group}/L1",
                    "task_group": f"HumanEval/{group}",
                    "completion_utf8_bytes": group + 30,
                }
            )
        smoke = choose_smoke(rows, 12)
        self.assertEqual(len(smoke), 12)
        self.assertEqual(len({row["task_group"] for row in smoke}), 12)
        self.assertEqual({row["length_bucket"] for row in smoke}, {"q1_short", "q2_medium", "q3_long", "q4_extreme"})


if __name__ == "__main__":
    unittest.main()
