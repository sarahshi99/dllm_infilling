from __future__ import annotations

import unittest

from analysis.finalize_dreamon_markov_head_training import eligible_heads, parser


class FinalizeVersionArgumentsTest(unittest.TestCase):
    def test_external_test_requires_nonzero_deployable_gain(self) -> None:
        statuses = {
            "tv": {"full": {"passed": True}, "chosen_lambda": 0.0, "deployable_gain": False},
            "kl": {"full": {"passed": True}, "chosen_lambda": 0.5, "deployable_gain": True},
        }
        self.assertEqual(eligible_heads(statuses), ["kl"])

    def test_v1_defaults_remain_backward_compatible(self) -> None:
        args = parser().parse_args(
            [
                "--result-dir", "result",
                "--bank-db", "bank.sqlite",
                "--model-snapshot", "model",
                "--common-training-config", "common.json",
            ]
        )
        self.assertEqual(args.run_id, "dreamon_markov_head_training_20260901_v1")
        self.assertEqual(args.run_date_utc, "2026-09-01")
        self.assertEqual(args.branch, "codex/dreamon-markov-head-training-v1")
        self.assertFalse(args.skip_research_record_updates)
        self.assertFalse(args.resume_existing_external_diagnostics)

    def test_v2_detached_runner_can_skip_cross_worktree_updates(self) -> None:
        args = parser().parse_args(
            [
                "--result-dir", "result",
                "--bank-db", "bank.sqlite",
                "--model-snapshot", "model",
                "--common-training-config", "common.json",
                "--run-id", "dreamon_markov_head_training_20260910_v2",
                "--run-date-utc", "2026-09-10",
                "--branch", "codex/dreamon-markov-k2-eval-v1",
                "--skip-research-record-updates",
                "--resume-existing-external-diagnostics",
            ]
        )
        self.assertEqual(args.run_id, "dreamon_markov_head_training_20260910_v2")
        self.assertTrue(args.skip_research_record_updates)
        self.assertTrue(args.resume_existing_external_diagnostics)


if __name__ == "__main__":
    unittest.main()
