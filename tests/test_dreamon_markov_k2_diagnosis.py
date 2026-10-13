from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from tests import test_dreamon_markov_k2 as fixtures


class DiagnosisDecoderTest(unittest.TestCase):
    decode = fixtures.DreamOnMarkovK2Test.decode
    def test_diagnostics_switch_preserves_generation_and_actions(self):
        for arm in "ABCDE":
            kwargs = dict(requested_k=1 if arm == "A" else 2,
                          markov_head=fixtures.ConditionalHead() if arm in "CD" else None,
                          markov_lambda=1.0 if arm in "CD" else 0.0,
                          right_recompute=arm == "E", capture_actions=True)
            on = self.decode(**kwargs, diagnostics=True)
            with patch("experiments.dreamon_singleline_order_parallel.prepare_decode_distribution",
                       side_effect=AssertionError("diagnostic computation on fast path")):
                off = self.decode(**kwargs, diagnostics=False)
            for key in ("completion_token_ids", "action_trace", "forward_count",
                        "base_forward_count", "recompute_forward_count", "outer_round_count"):
                self.assertEqual(on[key], off[key], (arm, key))
            self.assertEqual(off["step_trace"], [])
            self.assertEqual(off["replay_transitions"], [])

    def test_recompute_sees_only_left_and_uses_shift_once(self):
        class InspectModel(fixtures.FixedModel):
            def forward(self, x, *args):
                if self.calls == 1:
                    self.last_canvas = x.clone()
                result = super().forward(x, *args)
                if self.calls == 2:
                    # Target position 2 must use raw logits at position 1.
                    result.logits[:, 1, 6] = 40.0
                return result
        model = InspectModel()
        result = self.decode(model=model, right_recompute=True, diagnostics=False)
        self.assertEqual(model.last_canvas.tolist(), [[1, 5, 3, 2]])
        self.assertEqual(result["completion_token_ids"], [5, 6])
        self.assertEqual(result["forward_count"], 2)
        self.assertEqual(result["base_forward_count"], 1)
        self.assertEqual(result["outer_round_count"], 1)
        self.assertEqual(result["recompute_forward_count"], 1)

    def test_recompute_forward_and_sample_rng_are_isolated(self):
        class RandomRecomputeModel(fixtures.FixedModel):
            def forward(self, x, *args):
                if self.calls == 1:
                    torch.rand(7)
                return super().forward(x, *args)
        def stochastic(logits, **kwargs):
            probabilities = torch.softmax(logits.float(), -1)
            sampled = torch.multinomial(probabilities, 1).squeeze(-1)
            return probabilities.gather(-1, sampled[:, None]).squeeze(-1), sampled
        torch.manual_seed(84)
        self.decode(sample_tokens_fn=stochastic, diagnostics=False)
        baseline = torch.random.get_rng_state()
        torch.manual_seed(84)
        self.decode(model=RandomRecomputeModel(), sample_tokens_fn=stochastic,
                    right_recompute=True, diagnostics=False, markov_sampling_seed=567)
        self.assertTrue(torch.equal(baseline, torch.random.get_rng_state()))

    def test_recompute_structural_sample_only_commits_left(self):
        class StructuralModel(fixtures.FixedModel):
            def forward(self, x, *args):
                result = super().forward(x, *args)
                if self.calls == 2:
                    result.logits[..., 2] = 100.0
                return result
        result = self.decode(model=StructuralModel(), right_recompute=True,
                             diagnostics=False, capture_actions=True)
        self.assertEqual(result["completion_token_ids"], [5, 5])
        self.assertEqual(result["corrected_structural_stop_count"], 1)
        self.assertEqual(result["delete_count"], 0)
        self.assertEqual(result["action_trace"][0]["tokens"], [5])

    def test_recompute_keeps_other_future_positions_masked_and_expand_disabled(self):
        class InspectModel(fixtures.FixedModel):
            def forward(self, x, *args):
                result = super().forward(x, *args)
                if self.calls == 2:
                    self.fresh = x.clone()
                    result.logits[..., 4] = 1000.0
                return result
        model = InspectModel()
        result = self.decode(model=model, min_gen_len=3, max_gen_len=3,
                             right_recompute=True, diagnostics=False)
        self.assertEqual(model.fresh.tolist(), [[1, 5, 3, 3, 2]])
        self.assertEqual(result["expand_count"], 0)
        self.assertEqual(result["completion_token_ids"], [5, 5, 5])

    def test_recompute_does_not_trigger_on_original_structural_or_nonadjacent(self):
        result = self.decode(model=fixtures.FixedModel(normal_id=2), right_recompute=True,
                             diagnostics=False)
        self.assertEqual(result["recompute_forward_count"], 0)
        def nonadjacent(logits, **kwargs):
            confidence, predictions = fixtures.deterministic_sample(logits, **kwargs)
            return torch.tensor([.9, .1, .8][:len(confidence)]), predictions
        result = self.decode(min_gen_len=3, max_gen_len=3,
                             policy="confidence_global", sample_tokens_fn=nonadjacent,
                             right_recompute=True, diagnostics=False)
        self.assertEqual(result["recompute_forward_count"], 0)
        self.assertEqual(result["nonadjacent_fallback_count"], 1)

    def test_stochastic_diagnostics_equivalence_both_policies(self):
        from tests.test_dreamon_markov_premise import official_sample_tokens
        for policy in ("confidence_global", "left_to_right_frontier"):
            for arm in "ABCDE":
                kwargs = dict(policy=policy, requested_k=1 if arm == "A" else 2,
                              markov_head=fixtures.ConditionalHead() if arm in "CD" else None,
                              markov_lambda=1 if arm in "CD" else 0,
                              right_recompute=arm == "E", capture_actions=True,
                              sample_tokens_fn=official_sample_tokens, markov_sampling_seed=432)
                torch.manual_seed(789)
                on = self.decode(**kwargs, diagnostics=True)
                state_on = torch.random.get_rng_state()
                torch.manual_seed(789)
                off = self.decode(**kwargs, diagnostics=False)
                self.assertEqual(on["action_trace"], off["action_trace"])
                self.assertEqual(on["completion_token_ids"], off["completion_token_ids"])
                self.assertEqual(on["forward_count"], off["forward_count"])
                self.assertTrue(torch.equal(state_on, torch.random.get_rng_state()))

    def test_recompute_rejects_stacked_head(self):
        with self.assertRaises(ValueError):
            self.decode(right_recompute=True, markov_head=fixtures.ConditionalHead(), markov_lambda=1)


class DiagnosisPersistenceAndStatisticsTest(unittest.TestCase):
    def test_success_transaction_deduplicates_and_reopens(self):
        import tempfile
        from pathlib import Path
        import sqlite3
        from experiments.run_dreamon_markov_k2_diagnosis import open_store, save_case, load_cases
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_case.sqlite"
            db = open_store(path)
            row = dict(policy="global_confidence", arm="C", task_id="t", passed=False)
            save_case(db, row)
            with self.assertRaises(sqlite3.IntegrityError):
                save_case(db, row)
            db.close()
            db = open_store(path)
            self.assertEqual(len(load_cases(db)), 1)
            self.assertFalse(load_cases(db)[0]["passed"])
            self.assertGreaterEqual(load_cases(db)[0]["file_write_sec"], 0)
            db.close()

    def test_cluster_weighting_uses_actual_point_estimates(self):
        from analysis.analyze_dreamon_markov_k2_diagnosis import clustered_difference
        rows = []
        # One small helped cluster and one large harmed cluster: line != macro.
        for group, deltas in (("small", [1]), ("large", [-1, -1, -1])):
            for index, delta in enumerate(deltas):
                for arm, passed in (("D", delta == 1), ("C", delta == -1)):
                    rows.append(dict(arm=arm, task_id=f"{group}/{index}", base_task_id=group, passed=passed))
        result = clustered_difference(rows, "D", "C", replicates=1000)
        self.assertEqual(result["line_delta"], -.5)
        self.assertEqual(result["base_task_equal_delta"], 0)
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result, clustered_difference(rows, "D", "C", replicates=1000))

    def test_idle_gpu_required_even_when_utilization_is_zero(self):
        from experiments.run_dreamon_markov_k2_diagnosis import require_idle
        snapshot = dict(gpus="0, GPU-test, NVIDIA H200 NVL, 92000, 143771, 0, 35",
                        compute_processes="123, other-python, 92000")
        with self.assertRaises(RuntimeError):
            require_idle(snapshot)
        require_idle(snapshot, own_pid=123)

    def test_rotation_and_stable_ids_are_outcome_independent(self):
        from experiments.run_dreamon_markov_k2_diagnosis import rotated_arms, stable_smoke, ARMS
        rows = [dict(task_id=f"id{i:04d}", passed=i % 2) for i in range(1033)]
        first = [r["task_id"] for r in stable_smoke(rows)]
        self.assertEqual(len(set(first)), 16)
        self.assertEqual(first, [r["task_id"] for r in stable_smoke(list(reversed(rows)))])
        for index in range(10):
            self.assertEqual(set(rotated_arms(index)), set(ARMS))
            self.assertEqual(rotated_arms(index)[0], ARMS[index])

    def test_same_rank_zero_head_runs_same_sampling_path(self):
        import copy
        from experiments.dreamon_markov_head_training import MarkovHead
        trained = MarkovHead(vocab_size=8, rank=2, structural_token_ids=[2, 3, 4])
        with torch.no_grad():
            trained.output.weight.fill_(.1)
        zero = copy.deepcopy(trained)
        with torch.no_grad():
            zero.output.weight.zero_()
        self.assertIs(type(zero), type(trained))
        self.assertEqual(zero.rank, trained.rank)
        self.assertTrue(torch.equal(zero(torch.tensor([5])), torch.zeros(1, 8)))
        signatures = []
        for head in (zero, trained):
            calls = []
            hook = head.register_forward_hook(lambda *args: calls.append("head"))
            shapes = []
            def sample(logits, **kwargs):
                shapes.append(tuple(logits.shape))
                return fixtures.deterministic_sample(logits, **kwargs)
            helper = fixtures.DreamOnMarkovK2Test()
            result = helper.decode(markov_head=head, markov_lambda=1, sample_tokens_fn=sample,
                                   diagnostics=False)
            signatures.append((calls, shapes, result["markov_enabled_count"]))
            hook.remove()
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], (["head"], [(2, 8), (1, 8)], 1))


if __name__ == "__main__":
    unittest.main()
