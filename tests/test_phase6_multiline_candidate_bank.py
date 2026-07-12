import unittest
import json
from pathlib import Path

from experiments.phase6_multiline_candidate_bank import (
    CANVAS_LENGTHS,
    SEEDS,
    build_candidate_specs,
    build_manifest_without_tokenizer,
    candidate_key,
    expected_candidate_keys,
)


class Phase6MultiLineCandidateBankTest(unittest.TestCase):
    def test_candidate_grid_is_eight_same_pool_candidates(self) -> None:
        specs = build_candidate_specs()
        self.assertEqual(len(specs), 8)
        self.assertEqual(
            {(row["canvas_tokens"], row["seed"]) for row in specs},
            {(canvas, seed) for canvas in CANVAS_LENGTHS for seed in SEEDS},
        )
        fixed = [row for row in specs if row["control_label"] == "fixed64_control"]
        self.assertEqual([(row["canvas_tokens"], row["seed"]) for row in fixed], [(64, 0)])

    def test_manifest_keeps_rows_not_unique_groups(self) -> None:
        rows = [
            {"task_id": "MultiLineInfilling/HumanEval/0/L0_L0", "prompt": "p", "suffix": "s", "canonical_solution": "m"},
            {"task_id": "MultiLineInfilling/HumanEval/0/L1_L0", "prompt": "p2", "suffix": "s2", "canonical_solution": "m2"},
            {"task_id": "MultiLineInfilling/HumanEval/1/L0_L0", "prompt": "p3", "suffix": "s3", "canonical_solution": "m3"},
        ]
        manifest = build_manifest_without_tokenizer(rows, {"HumanEval/1"})
        self.assertEqual(len(manifest), 2)
        self.assertEqual({row["task_group"] for row in manifest}, {"HumanEval/0"})
        self.assertNotEqual(manifest[0]["row_key"], manifest[1]["row_key"])

    def test_expected_keys_count_eight_per_row(self) -> None:
        manifest = [{"row_key": "a"}, {"row_key": "b"}]
        keys = expected_candidate_keys(manifest)
        self.assertEqual(len(keys), 16)
        self.assertIn(candidate_key("a", 128, 1), keys)

    def test_real_multiline_allowed_population_is_5079(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with (repo.parent.parent / "data/HumanEval-MultiLineInfilling.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle]
        frozen = set(json.loads((repo / "analysis_outputs/grouped_split_20260702_accel2/test_tasks.json").read_text()))
        manifest = build_manifest_without_tokenizer(rows, frozen)
        self.assertEqual(len(rows), 5815)
        self.assertEqual(len(manifest), 5079)
        self.assertEqual(len(expected_candidate_keys(manifest)), 40632)


if __name__ == "__main__":
    unittest.main()
