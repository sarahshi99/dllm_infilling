import unittest

from experiments.m2_constraint_homotopy import (
    METHODS,
    TOTAL_STEPS,
    expected_keys,
    homotopy_weight,
    parser,
    visible_constraint_scores,
)


class Tokenizer:
    mapping = {0: "    total = sum(xs)\n", 1: "", 2: "    total = (sum(xs)\n"}

    def decode(self, ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        return "".join(self.mapping[int(item)] for item in ids)


class M2ConstraintHomotopyTest(unittest.TestCase):
    def test_gradual_and_abrupt_share_fixed_64_forward_budget_but_differ_in_schedule(self) -> None:
        self.assertEqual(TOTAL_STEPS, 64)
        self.assertLess(homotopy_weight(METHODS[0], 15), homotopy_weight(METHODS[0], 47))
        self.assertEqual(homotopy_weight(METHODS[1], 31), 0.0)
        self.assertEqual(homotopy_weight(METHODS[1], 32), 1.0)

    def test_constraint_view_uses_only_prefix_suffix_and_current_candidate(self) -> None:
        scores, meta = visible_constraint_scores(
            prefix="def f(xs):\n",
            suffix="    return total\n",
            tokenizer=Tokenizer(),
            candidate_token_ids=[0, 1],
        )
        self.assertEqual(len(scores), 2)
        self.assertTrue(meta["full_parse_passed"])
        self.assertEqual(meta["suffix_dependency_count"], 1)
        self.assertGreater(scores[0], 0.0)

    def test_syntax_incompatibility_is_inference_visible_constraint(self) -> None:
        scores, meta = visible_constraint_scores(
            prefix="def f(xs):\n",
            suffix="    return total\n",
            tokenizer=Tokenizer(),
            candidate_token_ids=[2],
        )
        self.assertFalse(meta["full_parse_passed"])
        self.assertGreater(scores[0], 0.0)

    def test_method_keys_cover_one_row_per_method(self) -> None:
        manifest = [{"row_key": "a"}, {"row_key": "b"}]
        self.assertEqual(len(expected_keys(manifest, METHODS[0])), 2)
        self.assertEqual(len(expected_keys(manifest, METHODS[1])), 2)

    def test_default_smoke_is_twelve_cases(self) -> None:
        args = parser().parse_args(
            [
                "--dataset-jsonl",
                "data.jsonl",
                "--gradual-output-dir",
                "gradual",
                "--abrupt-output-dir",
                "abrupt",
                "--compact-dir",
                "compact",
            ]
        )
        self.assertEqual(args.smoke_cases, 12)


if __name__ == "__main__":
    unittest.main()
