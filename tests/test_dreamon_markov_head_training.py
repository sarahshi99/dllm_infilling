from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.dreamon_markov_head_training import (
    MarkovHead,
    aligned_cross_entropy,
    atomic_save_checkpoint,
    backbone_state_digest,
    build_humaneval_fingerprints,
    deduplicate_and_split_records,
    deterministic_line_split,
    deterministic_split,
    distribution_loss,
    freeze_module,
    load_checkpoint,
    markov_total_loss,
    normalize_text,
    replay_target_logits,
)


class MarkovDataIsolationTest(unittest.TestCase):
    def test_normalization_and_grouped_split_are_deterministic(self) -> None:
        rows = [
            {
                "seq_id": 1,
                "instruction": "Do x.  \r\n",
                "code": "def f():\r\n    return 1  ",
                "output": "```python\ndef f():\n    return 1\n```",
                "entry_point": "f",
                "testcase": ["assert f() == 1"],
            },
            {
                "seq_id": 2,
                "instruction": "Do x.\n",
                "code": "def f():\n    return 1",
                "output": "```python\ndef f():\n    return 1\n```",
                "entry_point": "f",
                "testcase": ["assert f() == 1"],
            },
            {
                "seq_id": 3,
                "instruction": "Do x another way",
                "code": "def f():\n    return 2",
                "output": "```python\ndef f():\n    return 2\n```",
                "entry_point": "f",
                "testcase": ["assert f() == 1"],
            },
            {
                "seq_id": 4,
                "instruction": "Do y",
                "code": "def g():\n    return 3",
                "output": "```python\ndef g():\n    return 3\n```",
                "entry_point": "g",
                "testcase": ["assert g() == 3"],
            },
        ]
        clean, audit = deduplicate_and_split_records(
            rows, human_eval_rows=[], split_seed=20260901
        )
        self.assertEqual(normalize_text("a  \r\n b\t"), "a\n b")
        self.assertEqual(audit["normalized_duplicate_rows_removed"], 1)
        f_splits = {row["split"] for row in clean if row["entry_point"] == "f"}
        self.assertEqual(len(f_splits), 1)
        self.assertEqual(
            [row["record_id"] for row in clean],
            [row["record_id"] for row in deduplicate_and_split_records(
                rows, human_eval_rows=[], split_seed=20260901
            )[0]],
        )

    def test_humaneval_fingerprints_remove_code_or_test_overlap(self) -> None:
        human_eval = [
            {
                "task_id": "SingleLineInfilling/HumanEval/0/L0",
                "entry_point": "f",
                "prompt": "def f(x):\n",
                "canonical_solution": "    return x + 1\n",
                "suffix": "",
                "test": "def check(candidate):\n    assert candidate(1) == 2\n",
            }
        ]
        rows = [
            {
                "seq_id": 1,
                "instruction": "increment",
                "code": "def f(x):\n    return x + 1",
                "output": "",
                "entry_point": "f",
                "testcase": ["assert f(1) == 2"],
            },
            {
                "seq_id": 2,
                "instruction": "double",
                "code": "def h(x):\n    return x * 2",
                "output": "",
                "entry_point": "h",
                "testcase": ["assert h(2) == 4"],
            },
        ]
        fingerprints = build_humaneval_fingerprints(human_eval)
        self.assertEqual(fingerprints["base_task_count"], 1)
        clean, audit = deduplicate_and_split_records(
            rows, human_eval_rows=human_eval, split_seed=20260901
        )
        self.assertEqual([row["entry_point"] for row in clean], ["h"])
        self.assertEqual(audit["humaneval_decontaminated_rows"], 1)

    def test_hash_split_boundaries(self) -> None:
        observed = {
            deterministic_split(f"problem-{index}", seed=20260901)
            for index in range(200)
        }
        self.assertEqual(observed, {"train", "validation", "external_test"})

    def test_line_split_is_deterministic_and_has_two_tokens(self) -> None:
        code = "def f(x):\n    y = x + 1\n    z = y * 2\n    return z\n"
        left = deterministic_line_split(code, row_seed=123)
        right = deterministic_line_split(code, row_seed=123)
        self.assertEqual(left, right)
        self.assertGreaterEqual(len(left[1].split()), 2)
        self.assertNotIn("def ", left[1])


class MarkovHeadNumericsTest(unittest.TestCase):
    def test_zero_init_is_no_head_and_structural_rows_are_zero(self) -> None:
        head = MarkovHead(vocab_size=17, rank=4, structural_token_ids=(1, 2, 3), seed=42)
        previous = torch.tensor([0, 1, 4, 16])
        bias = head(previous)
        torch.testing.assert_close(bias, torch.zeros_like(bias), rtol=0, atol=0)
        with torch.no_grad():
            head.output.weight.fill_(0.25)
        bias = head(previous)
        self.assertTrue(torch.equal(bias[:, [1, 2, 3]], torch.zeros(4, 3)))
        self.assertEqual(head.parameter_count, 17 * 4 * 2)

    def test_tv_kl_and_aligned_ce_match_manual_values(self) -> None:
        fresh = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float32)
        q = torch.tensor([[0.5, 0.4, 0.1], [0.2, 0.2, 0.6]], dtype=torch.float32)
        tv_l1 = distribution_loss(q, fresh, kind="tv")
        expected_l1 = torch.tensor([(0.2 + 0.2), (0.1 + 0.1)]).mean()
        torch.testing.assert_close(tv_l1, expected_l1)
        kl = distribution_loss(q, fresh, kind="kl")
        expected_kl = (fresh * (fresh.log() - q.log())).sum(-1).mean()
        torch.testing.assert_close(kl, expected_kl)

        logits = q.log()
        targets = torch.tensor([0, 2])
        aligned = torch.tensor([True, False])
        ce = aligned_cross_entropy(logits, targets, aligned)
        torch.testing.assert_close(ce, -q[0, 0].log())
        zero = aligned_cross_entropy(logits, targets, torch.tensor([False, False]))
        torch.testing.assert_close(zero, torch.tensor(0.0))
        total = markov_total_loss(q, fresh, logits, targets, aligned, kind="tv")
        torch.testing.assert_close(total, 0.9 * expected_l1 + 0.1 * (-q[0, 0].log()))

    def test_replay_target_logits_match_released_shift(self) -> None:
        hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        lm_head = torch.nn.Linear(3, 7, bias=False)
        target_positions = torch.tensor([1, 4])
        selected = replay_target_logits(hidden, lm_head, target_positions)
        full = lm_head(hidden)
        shifted = torch.cat([full[:, :1], full[:, :-1]], dim=1)
        torch.testing.assert_close(selected, shifted[torch.arange(2), target_positions])


class MarkovTrainingSafetyTest(unittest.TestCase):
    def test_backbone_is_frozen_and_head_updates(self) -> None:
        torch.manual_seed(5)
        backbone = torch.nn.Sequential(torch.nn.Embedding(13, 6), torch.nn.Linear(6, 13))
        freeze_module(backbone)
        before = backbone_state_digest(backbone)
        head = MarkovHead(vocab_size=13, rank=3, structural_token_ids=(1, 2), seed=42)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.1, weight_decay=0)
        stale = torch.randn(8, 13)
        fresh = torch.softmax(torch.randn(8, 13), -1)
        previous = torch.arange(8) % 13
        q_logits = stale + head(previous)
        loss = distribution_loss(torch.softmax(q_logits.float(), -1), fresh, kind="tv")
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in head.parameters()))
        optimizer.step()
        self.assertEqual(before, backbone_state_digest(backbone))
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))

    def test_checkpoint_restores_optimizer_step_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            head = MarkovHead(vocab_size=11, rank=3, structural_token_ids=(1, 2), seed=42)
            optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            torch.manual_seed(1234)
            saved_rng = torch.random.get_rng_state().clone()
            atomic_save_checkpoint(
                path,
                head=head,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=9,
                epoch=2,
                best_metric=0.4,
                manifest_version="abc",
            )
            expected = torch.rand(4)
            clone = copy.deepcopy(head)
            optimizer2 = torch.optim.AdamW(clone.parameters(), lr=0.01)
            scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda _: 1.0)
            state = load_checkpoint(path, head=clone, optimizer=optimizer2, scheduler=scheduler2)
            self.assertEqual(state["global_step"], 9)
            self.assertEqual(state["epoch"], 2)
            self.assertEqual(state["manifest_version"], "abc")
            torch.testing.assert_close(torch.random.get_rng_state(), saved_rng)
            torch.testing.assert_close(torch.rand(4), expected)

    def test_synthetic_32_transition_overfit_decreases_without_nan(self) -> None:
        torch.manual_seed(7)
        vocab = 19
        previous = torch.arange(32) % vocab
        target = (previous * 3 + 1) % vocab
        fresh = torch.nn.functional.one_hot(target, vocab).float() * 0.98 + 0.02 / vocab
        stale_logits = torch.zeros(32, vocab)
        head = MarkovHead(vocab_size=vocab, rank=8, structural_token_ids=(1, 2), seed=42)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.2, weight_decay=0)
        losses = []
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            q_logits = stale_logits + head(previous)
            loss = distribution_loss(torch.softmax(q_logits.float(), -1), fresh, kind="kl")
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        self.assertLess(losses[-1], losses[0] * 0.35)


if __name__ == "__main__":
    unittest.main()
