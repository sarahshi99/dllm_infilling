import tempfile
import unittest
from pathlib import Path

from experiments.m2_constraint_homotopy import (
    METHODS,
    TOTAL_STEPS,
    audit_rows,
    append_jsonl,
    expected_keys,
    homotopy_weight,
    parser,
    run_method,
    choose_remask_positions,
    visible_constraint_signals,
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
        self.assertEqual(homotopy_weight(METHODS[0], 47), 0.0)
        self.assertLess(homotopy_weight(METHODS[1], 15), homotopy_weight(METHODS[1], 47))
        self.assertEqual(homotopy_weight(METHODS[2], 31), 0.0)
        self.assertEqual(homotopy_weight(METHODS[2], 32), 1.0)

    def test_constraint_view_uses_only_prefix_suffix_and_current_candidate(self) -> None:
        protection, repair, meta = visible_constraint_signals(
            prefix="def f(xs):\n",
            suffix="    return total\n",
            tokenizer=Tokenizer(),
            candidate_token_ids=[0, 1],
        )
        self.assertEqual(len(protection), 2)
        self.assertTrue(meta["full_parse_passed"])
        self.assertEqual(meta["suffix_dependency_count"], 1)
        self.assertGreater(protection[0], 0.0)
        self.assertEqual(repair[0], 0.0)

    def test_syntax_incompatibility_is_inference_visible_constraint(self) -> None:
        protection, repair, meta = visible_constraint_signals(
            prefix="def f(xs):\n",
            suffix="    return total\n",
            tokenizer=Tokenizer(),
            candidate_token_ids=[2],
        )
        self.assertFalse(meta["full_parse_passed"])
        self.assertEqual(protection[0], 0.0)
        self.assertGreater(repair[0], 0.0)

    def test_constraint_schedule_changes_actual_remask_choice(self) -> None:
        vanilla = choose_remask_positions(
            confidences=[0.2, 0.2],
            protection_bonus=[3.0, 0.0],
            repair_priority=[0.0, 3.0],
            masked_positions=[0, 1],
            target_masks=1,
            weight=homotopy_weight(METHODS[0], 40),
        )
        constrained = choose_remask_positions(
            confidences=[0.2, 0.2],
            protection_bonus=[3.0, 0.0],
            repair_priority=[0.0, 3.0],
            masked_positions=[0, 1],
            target_masks=1,
            weight=homotopy_weight(METHODS[2], 40),
        )
        self.assertEqual(vanilla, [0])
        self.assertEqual(constrained, [1])

    def test_method_keys_cover_one_row_per_method(self) -> None:
        manifest = [{"row_key": "a"}, {"row_key": "b"}]
        self.assertEqual(len(expected_keys(manifest, METHODS[0])), 2)
        self.assertEqual(len(expected_keys(manifest, METHODS[1])), 2)

    def test_default_smoke_is_twelve_cases(self) -> None:
        args = parser().parse_args(
            [
                "--dataset-jsonl",
                "data.jsonl",
                "--vanilla-output-dir",
                "vanilla",
                "--gradual-output-dir",
                "gradual",
                "--abrupt-output-dir",
                "abrupt",
                "--compact-dir",
                "compact",
            ]
        )
        self.assertEqual(args.smoke_cases, 12)

    def test_forward_accounting_and_duplicate_audit_are_strict(self) -> None:
        key = next(iter(expected_keys([{"row_key": "a"}], METHODS[0])))
        audit = audit_rows([{"candidate_key": key, "status": "ok", "metrics": {"actual_forward_count": 64}}], {key})
        self.assertTrue(audit["passed"])
        duplicate = audit_rows(
            [
                {"candidate_key": key, "status": "ok", "metrics": {"actual_forward_count": 64}},
                {"candidate_key": key, "status": "ok", "metrics": {"actual_forward_count": 64}},
            ],
            {key},
        )
        self.assertEqual(duplicate["duplicate_count"], 1)
        self.assertFalse(duplicate["passed"])

    def test_resume_is_noop_and_duplicate_raw_is_refused_before_decode(self) -> None:
        manifest = [{"row_key": "a", "source_row_id": 0}]
        key = next(iter(expected_keys(manifest, METHODS[0])))
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "m2.jsonl"
            append_jsonl(raw, {"candidate_key": key, "candidate_kind": METHODS[0], "status": "ok"})
            self.assertEqual(
                run_method(
                    method=METHODS[0],
                    manifest=manifest,
                    source_rows=[],
                    raw_path=raw,
                    tokenizer=None,
                    model=None,
                ),
                0,
            )
            append_jsonl(raw, {"candidate_key": key, "candidate_kind": METHODS[0], "status": "ok"})
            with self.assertRaisesRegex(RuntimeError, "duplicate keys"):
                run_method(
                    method=METHODS[0],
                    manifest=manifest,
                    source_rows=[],
                    raw_path=raw,
                    tokenizer=None,
                    model=None,
                )


if __name__ == "__main__":
    unittest.main()
