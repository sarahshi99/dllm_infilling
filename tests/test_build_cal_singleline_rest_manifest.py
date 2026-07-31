from __future__ import annotations

import unittest

from analysis.build_cal_singleline_rest_manifest import (
    build_common_manifest,
    choose_smoke,
    seed42_demo_and_rest,
)


def official_rows(count: int = 40) -> list[dict[str, str]]:
    return [
        {
            "task_id": f"SingleLineInfilling/HumanEval/{index // 2}/L{index % 2}",
            "canonical_solution": "x" * (index + 1),
        }
        for index in range(count)
    ]


class BuildCalSingleLineRestManifestTest(unittest.TestCase):
    def test_seed42_split_is_deterministic(self) -> None:
        rows = official_rows(1033)
        demo_a, rest_a = seed42_demo_and_rest(rows)
        demo_b, rest_b = seed42_demo_and_rest(rows)
        self.assertEqual([index for index, _ in demo_a], [index for index, _ in demo_b])
        self.assertEqual(len(demo_a), 100)
        self.assertEqual(len(rest_a), 933)
        self.assertEqual(rest_a, rest_b)

    def test_common_manifest_uses_only_allowed_rest_rows(self) -> None:
        rows = official_rows()
        demo, rest = seed42_demo_and_rest(rows, demo_count=4)
        rest_ids = {str(row["task_id"]) for _, row in rest}
        allowed = [
            {
                "task_id": task_id,
                "task_group": task_id.split("/L", 1)[0].replace("SingleLineInfilling/", ""),
                "frozen_controller_test_row": "False",
                "frozen_controller_test_exclusion_flag": "included_not_frozen_controller_test",
                "source_dataset": "HumanEval-SingleLineInfilling",
            }
            for task_id in sorted(rest_ids)[:-2]
        ]
        common = build_common_manifest(rows, allowed, demo_count=4)
        self.assertEqual(len(common), len(rest_ids) - 2)
        self.assertFalse({row["source_row_id"] for row in common} & {index for index, _ in demo})
        forbidden = {"canonical_solution", "oracle_length", "test", "passed", "completion"}
        self.assertFalse(any(forbidden & set(row) for row in common))

    def test_smoke_is_group_deduplicated(self) -> None:
        rows = []
        for group in range(16):
            for offset in range(2):
                rows.append(
                    {
                        "candidate_key": f"g{group}-{offset}",
                        "source_row_id": group * 2 + offset,
                        "task_id": f"SingleLineInfilling/HumanEval/{group}/L{offset}",
                        "task_group": f"HumanEval/{group}",
                        "dataset": "HumanEval-SingleLineInfilling",
                        "population": "CAL-Rest_intersection_project_non-frozen",
                        "evaluator_commit": "88062ff9859c875d04db115b698ed4b0f0395170",
                        "seed": 42,
                        "technical_length": group * 2 + offset + 1,
                    }
                )
        smoke = choose_smoke(rows, 12)
        self.assertEqual(len(smoke), 12)
        self.assertEqual(len({row["task_group"] for row in smoke}), 12)
        self.assertFalse(any("technical_length" in row for row in smoke))


if __name__ == "__main__":
    unittest.main()
