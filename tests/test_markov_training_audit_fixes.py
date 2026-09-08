import unittest
import torch
from analysis.markov_head_metrics import no_head_summary
from experiments.dreamon_markov_head_training import (
    accumulated_markov_loss, markov_total_loss, deduplicate_and_split_records,
)
from experiments.markov_data_identity import connected_groups, human_index, human_matches


class AuditFixTests(unittest.TestCase):
    def test_baseline_cannot_inherit_head_improvements(self):
        before = {"baseline_raw_tv": .1, "head_raw_tv": .07,
                  "mismatch_recovery": .36, "stable_corruption": .004,
                  "cluster_tv_improvement": .03, "cluster_tv_improvement_ci95_low": .02,
                  "cluster_tv_improvement_ci95_high": .04, "recovered_count": 12,
                  "corrupted_count": 3, "transitions": 100}
        after = no_head_summary(before)
        self.assertEqual(after["head_raw_tv"], .1)
        for key in ("mismatch_recovery", "stable_corruption", "cluster_tv_improvement",
                    "cluster_tv_improvement_ci95_low", "cluster_tv_improvement_ci95_high",
                    "recovered_count", "corrupted_count"):
            self.assertEqual(after[key], 0)
        self.assertEqual(after["transitions"], 100)
        self.assertEqual(before["mismatch_recovery"], .36)

    def test_accumulated_loss_and_gradient_match_full_batch(self):
        torch.manual_seed(5)
        for kind in ("tv", "kl"):
            # Unequal aligned counts; an all-unaligned microbatch; final short batch.
            for mask in (torch.tensor([1, 0, 0, 0, 1, 1, 1], dtype=torch.bool),
                         torch.zeros(7, dtype=torch.bool)):
                fresh = torch.softmax(torch.randn(7, 11), -1)
                refs = torch.arange(7)
                whole = torch.randn(7, 11, requires_grad=True)
                split = whole.detach().clone().requires_grad_(True)
                full_loss = markov_total_loss(torch.softmax(whole, -1), fresh, whole, refs, mask, kind=kind)
                full_loss.backward()
                total = 0.0
                for a, b in ((0, 2), (2, 4), (4, 7)):
                    part = accumulated_markov_loss(
                        torch.softmax(split[a:b], -1), fresh[a:b], split[a:b], refs[a:b], mask[a:b],
                        kind=kind, total_rows=7, total_aligned=int(mask.sum()),
                    )
                    total += float(part.detach())
                    part.backward()
                self.assertAlmostEqual(float(full_loss.detach()), total, places=6)
                torch.testing.assert_close(whole.grad, split.grad, rtol=1e-5, atol=1e-7)

    def test_changed_tests_cannot_split_identical_problem(self):
        rows = [{"seq_id": i, "instruction": "Return x plus one.",
                 "code": "def f(x):\n    return x + 1", "output": "", "entry_point": "f",
                 "testcase": [f"assert f({i}) == {i+1}"]} for i in range(20)]
        records, audit = deduplicate_and_split_records(rows, human_eval_rows=[])
        self.assertEqual(len({r["problem_group_id"] for r in records}), 1)
        self.assertEqual(len({r["split"] for r in records}), 1)

    def test_transitive_identity_groups(self):
        rows = [dict(instruction="A", code="x=1"), dict(instruction="B", code="x=1"),
                dict(instruction="B", code="x=2"), dict(instruction="C", code="x=3")]
        groups = connected_groups(rows)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[3])

    def test_human_docstring_and_candidate_normalization(self):
        human = [{"task_id": "SingleLineInfilling/HumanEval/0/L0", "entry_point": "f",
                  "prompt": 'def f(x):\n    """Increment."""\n',
                  "canonical_solution": "    return x + 1\n", "suffix": "",
                  "test": "def check(candidate):\n    assert candidate(1) == 2"}]
        record = {"instruction": "Increment", "code": "def f(x):\n    return x+1",
                  "entry_point": "f", "testcase": ["assert f(1) == 2"]}
        reasons = {m["reason"] for m in human_matches(record, human_index(human))}
        self.assertIn("code_ast_no_doc_entry_normalized", reasons)
        self.assertIn("same_entry_normalized_assert_candidate", reasons)

    def test_actual_response_code_also_participates(self):
        rows = [dict(instruction="A", code="x=1", output="```python\nx=9\n```"),
                dict(instruction="B", code="x=2", output="```python\nx=9\n```")]
        self.assertEqual(len(set(connected_groups(rows))), 1)


if __name__ == "__main__":
    unittest.main()
