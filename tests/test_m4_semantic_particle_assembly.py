import unittest

from experiments.m4_semantic_particle_assembly import (
    METHODS,
    assemble_fragments,
    candidate_fragments,
    candidate_view,
    offline_rows,
    parser,
)


PREFIX = "def f(xs):\n"
SUFFIX = "    return total\n"
MIDDLE = "    subtotal = sum(xs)\n    total = subtotal + 1\n"


class PoisonRow(dict):
    forbidden = {"passed", "reference_middle", "canonical_solution", "verification", "task_id", "task_group", "split_label", "oracle_length"}

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"forbidden assembly input read: {key}")
        return super().get(key, default)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"forbidden assembly input read: {key}")
        return super().__getitem__(key)


def candidate(seed: int = 0):
    return {
        "middle_text": MIDDLE,
        "canvas_tokens": 64,
        "seed": seed,
        "final_token_confidences": [0.8, 0.7],
        "prefix_text": PREFIX,
        "suffix_text": SUFFIX,
    }


class M4SemanticParticleAssemblyTest(unittest.TestCase):
    def test_extracts_statement_basic_block_and_def_use_fragments(self) -> None:
        kinds = {item["kind"] for item in candidate_fragments(candidate(), 0)}
        self.assertTrue({"statement", "basic_block", "def_use"} <= kinds)

    def test_assembly_uses_suffix_obligation_and_transitive_fragment(self) -> None:
        assembly = assemble_fragments([candidate(seed=index % 2) for index in range(8)])
        self.assertIn("total", assembly["suffix_obligations"])
        self.assertIn("total", assembly["resolved_obligations"])
        self.assertIn("total = subtotal + 1", assembly["assembly_text"])

    def test_candidate_view_rejects_forbidden_field_access_by_construction(self) -> None:
        row = PoisonRow({**candidate(), "passed": True, "canonical_solution": "POISON", "task_id": "POISON"})
        view = candidate_view(row)
        self.assertEqual(view["middle_text"], MIDDLE)
        self.assertEqual(view["canvas_tokens"], 64)

    def test_full_offline_assembly_check_is_structural_and_does_not_need_evaluator(self) -> None:
        manifest = [{"row_key": "r", "case_index": 0, "source_row_id": 0, "task_group": "HumanEval/0", "length_bucket": "short", "reference_middle_tokens": 4}]
        raw_candidates = [
            {**candidate(seed=index % 2), "candidate_kind": "deployable_grid"}
            for index in range(8)
        ]
        best, assembly = offline_rows(manifest, {"r": raw_candidates})
        self.assertEqual(len(best), 1)
        self.assertEqual(len(assembly), 1)
        self.assertIn("offline_structural_valid", best[0])
        self.assertNotIn("verification", best[0])

    def test_parser_defaults_to_twelve_case_smoke(self) -> None:
        args = parser().parse_args(
            [
                "--dataset-jsonl", "data.jsonl",
                "--candidate-bank-raw", "bank.jsonl",
                "--best-output-dir", "best",
                "--assembly-output-dir", "assembly",
                "--repair-output-dir", "repair",
                "--compact-dir", "compact",
            ]
        )
        self.assertEqual(args.smoke_cases, 12)
        self.assertEqual(len(METHODS), 3)


if __name__ == "__main__":
    unittest.main()
