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
    expected_oracle_keys,
    normalize_candidate_row,
    resolve_output_dirs,
)
from experiments.phase6_abductive_bridge_runner import expected_refinement_keys


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
        manifest = [
            {"row_key": "a", "reference_middle_tokens": 3},
            {"row_key": "b", "reference_middle_tokens": 5},
        ]
        keys = expected_candidate_keys(manifest)
        self.assertEqual(len(keys), 16)
        self.assertIn(candidate_key("a", "deployable_grid", 128, 1), keys)
        self.assertEqual(len(expected_oracle_keys(manifest)), 2)
        self.assertEqual(len(expected_refinement_keys(manifest)), 4)

    def test_real_multiline_allowed_population_is_5079(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with (repo.parent.parent / "data/HumanEval-MultiLineInfilling.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle]
        frozen = set(json.loads((repo / "analysis_outputs/grouped_split_20260702_accel2/test_tasks.json").read_text()))
        manifest = build_manifest_without_tokenizer(rows, frozen)
        self.assertEqual(len(rows), 5815)
        self.assertEqual(len(manifest), 5079)
        self.assertEqual(len(expected_candidate_keys(manifest)), 40632)

    def test_default_smoke_is_twelve_cases(self) -> None:
        from experiments.phase6_multiline_candidate_bank import parser

        parsed = parser().parse_args(
            ["run", "--dataset-jsonl", "data.jsonl", "--output-dir", "raw", "--compact-dir", "compact"]
        )
        self.assertEqual(parsed.smoke_cases, 12)

    def test_stage_one_and_refinement_outputs_are_always_distinct(self) -> None:
        stage1, generic, m1 = resolve_output_dirs(Path("raw"), None, None)
        self.assertEqual(len({stage1, generic, m1}), 3)
        with self.assertRaises(ValueError):
            resolve_output_dirs(Path("raw"), "raw", "m1")

    def test_error_row_is_normalized_for_resume_and_analysis(self) -> None:
        task = type("Task", (), {"prefix": "def f():\n", "suffix": "    return x\n"})()
        row = normalize_candidate_row(
            {"candidate_key": "k", "status": "error", "passed": False},
            task,
        )
        self.assertEqual(row["middle_text"], "")
        self.assertEqual(row["prefix_text"], "def f():\n")
        self.assertEqual(row["suffix_text"], "    return x\n")
        self.assertEqual(row["candidate_middle_tokens"], 0)
        self.assertEqual(row["metrics"], {})
        self.assertEqual(row["verification"], {})


if __name__ == "__main__":
    unittest.main()
