"""Small-vocabulary checks against the actual pinned A correction/sampling code."""
import importlib.util
import os
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from ..contract import HeadConfig
from ..heads import ContextHead, MODEL_NAMES
from ..execution import execute
from ..test_cpu import cfg, fixture, propagation_head
from .adapter import KeywordHeadAdapter, config_from_a_schema


def load_a_adapter(path):
    spec = importlib.util.spec_from_file_location("context_heads_a_interop_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_view(h0):
    """Same tiny H0 E/W objects, with the old raw-head attribute interface."""
    result = nn.Module()
    result.embedding, result.output = h0.embedding, h0.output
    result.register_buffer("output_mask", h0.output_mask)
    return result


class BridgeCPU(unittest.TestCase):
    def test_all_seven_kwargs_dict_scores_gradients_and_counting(self):
        batch = fixture()
        fields = vars(batch)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                core = ContextHead(name, cfg())
                adapter = KeywordHeadAdapter(core)
                expected = core(batch)
                out = adapter(**fields)
                self.assertEqual(set(out), {"latent", "delta_logits"})
                torch.testing.assert_close(out["latent"], expected.latent, rtol=0, atol=0)
                torch.testing.assert_close(out["delta_logits"], expected.delta_logits, rtol=0, atol=0)
                self.assertEqual(adapter.calls, batch.query_indices.numel())
                out["delta_logits"].square().sum().backward()
                self.assertTrue(torch.isfinite(core.output.weight.grad).all())
                zero = KeywordHeadAdapter(core, zero_override=True)(**fields)
                self.assertEqual(zero["latent"].count_nonzero().item(), 0)
                self.assertEqual(zero["delta_logits"].count_nonzero().item(), 0)

    def test_missing_legacy_zero_is_explicit_and_never_candidate_initialization(self):
        core = ContextHead("Hctx-old-context-only", cfg())
        adapter = KeywordHeadAdapter(core)
        with self.assertRaises(AttributeError):
            _ = adapter.head
        old = legacy_view(ContextHead("H0-static", cfg(seed=55)))
        original_output = core.output.weight.detach().clone()
        adapter = KeywordHeadAdapter(core, legacy_zero_head=old)
        self.assertIs(adapter.head, old)
        self.assertFalse(hasattr(adapter.context_head, "embedding"))
        torch.testing.assert_close(core.output.weight, original_output, rtol=0, atol=0)
        self.assertNotEqual(core.output.weight.data_ptr(), old.output.weight.data_ptr())

    def test_provisional_is_rejected_without_counter_mutation(self):
        batch = fixture()
        batch = replace(batch, token_state=torch.full_like(batch.token_state, 2))
        for name in MODEL_NAMES:
            adapter = KeywordHeadAdapter(ContextHead(name, cfg()))
            with self.assertRaises(ValueError):
                adapter(**vars(batch))
            self.assertEqual(adapter.calls, 0)

    def test_schema_uses_padded_rows_as_special_verbatim(self):
        schema = dict(contract_id="dreamon-context-v1-20261015", schema_version=1,
                      H=7, V=11, rank=256, max_k=4, special_token_ids=[8, 9, 10])
        config = config_from_a_schema(schema, tokenizer_size=9)
        self.assertEqual(config, HeadConfig(11, 7, (8, 9, 10), tokenizer_size=9))
        head = ContextHead("H0-static", config)
        self.assertEqual(head.output_mask[-3:].tolist(), [0, 0, 0])
        with self.assertRaises(ValueError):
            config_from_a_schema({**schema, "schema_version": 2})


class ActualACPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("CONTEXT_HEADS_A_ADAPTER")
        if not source:
            raise unittest.SkipTest("set CONTEXT_HEADS_A_ADAPTER to pinned A adapter.py")
        cls.a = load_a_adapter(Path(source))

    def sampler(self, scores, u):
        return self.a.inverse_cdf(self.a.distribution(scores, temperature=.2, top_p=.9), u)

    def fixed_u(self, batch):
        # Jacobi iteration is intentionally absent from A's common key.
        return torch.tensor([[self.a.uniform(self.a.sample_key("B-CPU-interop", 0, int(p)))
                              for p in row] for row in batch.position_ids], dtype=torch.float64)

    def test_h0_matches_real_a_frozen_head_adapter(self):
        core = ContextHead("H0-static", cfg())
        old = legacy_view(core)
        new_adapter = KeywordHeadAdapter(core, legacy_zero_head=old)
        old_adapter = self.a.FrozenHeadAdapter(old)
        batch = fixture()
        nonanchors = batch.queries(batch.query_indices[:, 1:], batch.history_mask[:, 1:], batch.old_logits[:, 1:])
        fields = vars(nonanchors)
        new, previous = new_adapter(**fields), old_adapter(**fields)
        for key in ("latent", "delta_logits"):
            torch.testing.assert_close(new[key], previous[key], rtol=0, atol=0)
        self.assertEqual(new_adapter.calls, old_adapter.calls)

    def test_real_engine_correction_boundary_without_engine_or_model_execution(self):
        # Only Engine.correction is exercised. B does not construct/execute the
        # model-bearing Engine or decode any real case.
        old = legacy_view(ContextHead("H0-static", cfg(seed=55)))
        batch = fixture(b=1)
        one = batch.queries(batch.query_indices[:, 2:3], batch.history_mask[:, 2:3], batch.old_logits[:, 2:3])
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                core = ContextHead(name, cfg())
                wrapped = KeywordHeadAdapter(core, legacy_zero_head=old)
                engine = object.__new__(self.a.Engine)
                engine.head = wrapped
                engine.zero = self.a.FrozenHeadAdapter(wrapped.head, zero_override=True)
                actual = engine.correction(vars(one))
                torch.testing.assert_close(actual, core(one).delta_logits[0, 0], rtol=0, atol=0)
                self.assertEqual(engine.correction(vars(one), zero=True).count_nonzero().item(), 0)

    def test_actual_a_sampler_fixed_u_h3_h4_jacobi3_and_chain_resets(self):
        for name in ("H3-short-attention", "H4-causal-mixer"):
            for seed in (17, 43, 71):
                for pos, chains in (([0, 1, 2, 3], [0]*4), ([0, 1, 8, 9], [0, 0, 1, 1])):
                    with self.subTest(model=name, seed=seed, positions=pos):
                        head = ContextHead(name, cfg(seed=seed))
                        batch = fixture(positions=pos, chains=chains)
                        u = self.fixed_u(batch)
                        seq = execute(head, batch, batch.token_ids, u, self.sampler)
                        jac = execute(head, batch, batch.token_ids, u, self.sampler, mode="jacobi", rounds=3)
                        self.assertTrue(torch.equal(seq.token_ids, jac.token_ids))

    def test_actual_a_sampler_requires_three_rounds_for_nonzero_propagation(self):
        head, batch = propagation_head(), fixture(b=1)
        batch = replace(batch, old_logits=torch.zeros_like(batch.old_logits))
        initial, u = torch.zeros(1, 4, dtype=torch.long), self.fixed_u(batch)
        seq = execute(head, batch, initial, u, self.sampler)
        self.assertEqual(seq.token_ids.tolist(), [[0, 1, 0, 1]])
        for n in (1, 2, 3):
            jac = execute(head, batch, initial, u, self.sampler, mode="jacobi", rounds=n)
            self.assertEqual(torch.equal(seq.token_ids, jac.token_ids), n == 3)
            self.assertEqual(jac.projected_positions, 3*n)

    def test_actual_a_sampler_ties_and_padded_support_are_deterministic(self):
        scores = torch.zeros(1, 2, 11)
        scores[..., 9:] = -torch.inf
        u = torch.tensor([[.25, .75]], dtype=torch.float64)
        first, second = self.sampler(scores, u), self.sampler(scores, u)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue((first < 9).all())
        self.assertEqual(self.a.uniform(self.a.sample_key("B-CPU-interop", 0, 3)),
                         self.a.uniform(self.a.sample_key("B-CPU-interop", 0, 3)))


if __name__ == "__main__":
    unittest.main()
