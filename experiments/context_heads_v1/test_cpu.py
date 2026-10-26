"""Run: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest
experiments.context_heads_v1.test_cpu -v. No backbone or answer generation.
"""
import unittest
from dataclasses import replace

import torch
from torch import nn

from .contract import HeadBatch, HeadConfig, history_for, split_chains
from .heads import ContextHead, MODEL_NAMES
from .execution import execute, inverse_cdf_mock, greedy_mock, BarrierEvent
from .loss import teacher_kl

torch.set_num_threads(1)


def fixture(k=4, b=2, h=7, v=11, positions=None, chains=None):
    g = torch.Generator().manual_seed(61)
    pos = torch.arange(k).repeat(b, 1) if positions is None else torch.tensor(positions).repeat(b, 1)
    chain = torch.zeros(b, k, dtype=torch.long) if chains is None else torch.tensor(chains).repeat(b, 1)
    batch = HeadBatch(torch.randn(b, k, h, generator=g), pos, chain,
                      torch.ones(b, k, dtype=torch.bool), torch.arange(k).repeat(b, 1),
                      torch.randint(0, v-1, (b, k), generator=g), torch.ones(b, k, dtype=torch.long),
                      torch.zeros(b, k, k, dtype=torch.bool), torch.randn(b, k, v, generator=g) * .1)
    return replace(batch, history_mask=history_for(batch, batch.query_indices))


def cfg(**kwargs):
    return HeadConfig(11, 7, (10,), **kwargs)


def nontrivial(head):
    # H1 starts exactly at static; turn on its modulation to test causality.
    if head.name == "H1-context-prev":
        with torch.no_grad():
            head.gamma[-1].weight.fill_(.015)
            head.gamma[-1].bias.fill_(.02)
    return head


class HeadCPU(unittest.TestCase):
    def test_shapes_finite_gradients_frozen_inputs_and_latent_only(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                batch = fixture()
                batch.old_hidden.requires_grad_()
                batch.old_logits.requires_grad_()
                head = nontrivial(ContextHead(name, cfg()))
                out = head(batch)
                self.assertEqual(out.latent.shape, (2, 4, 256))
                self.assertEqual(out.delta_logits.shape, (2, 4, 11))
                self.assertGreater(out.delta_logits[:, 1:].abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(out.latent).all())
                loss = teacher_kl(batch.old_logits, out.delta_logits,
                                  torch.randn_like(batch.old_logits), batch.history_mask.any(-1))
                loss.backward()
                for param in head.parameters():
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())
                self.assertIsNone(batch.old_hidden.grad)
                self.assertIsNone(batch.old_logits.grad)
                self.assertIsNone(head(batch, project=False).delta_logits)

    def test_zero_override_anchor_and_special_delta_exact(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                head, batch = ContextHead(name, cfg()), fixture()
                with torch.no_grad():
                    for n, p in head.named_parameters():
                        if n.endswith("bias"):
                            p.fill_(3)
                for z in (head(batch).latent[:, 0], head(batch).delta_logits[:, 0],
                          head(batch).delta_logits[..., 10], head(batch, zero_override=True).latent,
                          head(batch, zero_override=True).delta_logits):
                    self.assertEqual(torch.count_nonzero(z).item(), 0)

    def test_h0_legacy_formula_key_mapping(self):
        head = ContextHead("H0-static", cfg())
        batch = fixture()
        # Synthetic nonzero old checkpoint: same old state keys and masked formula.
        state = {"head": {k: v.clone() for k, v in head.state_dict().items()}}
        mapped = ContextHead("H0-static", cfg(seed=9))
        mapped.load_legacy_static(state)
        expected = nn.functional.linear(head.embedding(batch.token_ids[:, :-1]), head.output.weight)
        expected *= head.output_mask
        torch.testing.assert_close(mapped(batch).delta_logits[:, 1:], expected, rtol=0, atol=0)
        author_state = {"markov_w1.weight": head.embedding.weight,
                        "markov_w2.weight": head.output.weight, "output_mask": head.output_mask}
        mapped.load_legacy_static(author_state)
        torch.testing.assert_close(mapped(batch).delta_logits, head(batch).delta_logits, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            mapped.load_legacy_static({"head": {**state["head"], "output_mask": torch.ones(11)}})

    def test_h1_gamma_one_is_static_and_old_h_prev_is_explicit(self):
        batch = fixture()
        h0 = ContextHead("H0-static", cfg())
        h1 = ContextHead("H1-context-prev", cfg())
        torch.testing.assert_close(h1(batch).delta_logits, h0(batch).delta_logits, rtol=0, atol=0)
        nontrivial(h1)
        changed = batch.old_hidden.clone()
        changed[:, 0] += 3
        self.assertFalse(torch.equal(h1(batch).latent[:, 1], h1(replace(batch, old_hidden=changed)).latent[:, 1]))

    def test_gated_matches_pinned_author_adapter_formula_with_bias(self):
        head, batch = ContextHead("Hgated-author-family", cfg()), fixture()
        prev = head.embedding(batch.token_ids[:, :-1])
        gate = torch.sigmoid(nn.functional.linear(torch.cat([batch.old_hidden[:, 1:], prev], -1),
                                                 head.gate.weight, head.gate.bias))
        expected = nn.functional.linear(prev * gate, head.output.weight) * head.output_mask
        torch.testing.assert_close(head(batch).delta_logits[:, 1:], expected, rtol=0, atol=0)

    def test_hctx_never_uses_new_tokens_or_has_embedding(self):
        head, batch = ContextHead("Hctx-old-context-only", cfg()), fixture()
        self.assertFalse(hasattr(head, "embedding"))
        changed = replace(batch, token_ids=(batch.token_ids + 3) % 10)
        torch.testing.assert_close(head(batch).latent, head(changed).latent, rtol=0, atol=0)
        # The actual readable prefix can shrink while fixed phase/positions stay fixed.
        mask = batch.history_mask.clone()
        mask[:, 3, 1:3] = False
        torch.testing.assert_close(head(batch).latent[:, 3], head(replace(batch, history_mask=mask)).latent[:, 3], rtol=0, atol=0)

    def test_future_tokens_and_query_specific_invisibility(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                head, batch = nontrivial(ContextHead(name, cfg())), fixture()
                changed = batch.token_ids.clone()
                changed[:, 1:] = (changed[:, 1:] + 1) % 10
                torch.testing.assert_close(head(batch).latent[:, :2], head(replace(batch, token_ids=changed)).latent[:, :2], rtol=0, atol=0)
                mask = batch.history_mask.clone()
                mask[:, 3, 1:3] = False
                batch = replace(batch, history_mask=mask)
                torch.testing.assert_close(head(batch).latent[:, 3], head(replace(batch, token_ids=changed)).latent[:, 3], rtol=0, atol=0)

    def test_padding_nan_hidden_and_invalid_token_have_no_contribution(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                head, batch = ContextHead(name, cfg()), fixture()
                valid, chains = batch.selected_mask.clone(), batch.chain_ids.clone()
                valid[:, 3], chains[:, 3] = False, -1
                batch = replace(batch, selected_mask=valid, chain_ids=chains)
                batch = replace(batch, history_mask=history_for(batch, batch.query_indices))
                changed_h, changed_t, changed_p = batch.old_hidden.clone(), batch.token_ids.clone(), batch.position_ids.clone()
                changed_h[:, 3], changed_t[:, 3], changed_p[:, 3] = torch.nan, 999999, 100000
                out = head(replace(batch, old_hidden=changed_h, token_ids=changed_t, position_ids=changed_p))
                torch.testing.assert_close(out.latent, head(batch).latent, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(out.latent[:, 3]).item(), 0)
                if out.attention_weights is not None:
                    self.assertEqual(torch.count_nonzero(out.attention_weights[..., 3]).item(), 0)

    def test_step_teacher_force_same_prefix_and_reset_round_chain(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                head, batch = ContextHead(name, cfg()), fixture()
                expected = head(batch).latent
                for j in range(4):
                    one = batch.queries(batch.query_indices[:, j:j+1], batch.history_mask[:, j:j+1])
                    torch.testing.assert_close(head.step(one).latent[:, 0], expected[:, j], rtol=1e-5, atol=1e-6)
                clear = head.reset(batch)
                self.assertEqual(torch.count_nonzero(clear.token_state).item(), 0)
                self.assertEqual(torch.count_nonzero(head(clear).latent).item(), 0)
                torch.testing.assert_close(head(batch).latent, expected, rtol=0, atol=0)
                multi = fixture(positions=[0, 1, 8, 9], chains=[0, 0, 1, 1])
                changed = multi.token_ids.clone()
                changed[:, :2] = (changed[:, :2] + 1) % 10
                torch.testing.assert_close(head(multi).latent[:, 2:], head(replace(multi, token_ids=changed)).latent[:, 2:], rtol=0, atol=0)
                reset = BarrierEvent(0, "expand").apply(head, multi)
                self.assertTrue(torch.equal(reset.token_state[:, 2:], multi.token_state[:, 2:]))
                torch.testing.assert_close(head(reset).latent[:, 2:], head(multi).latent[:, 2:], rtol=0, atol=0)

    def test_h3_single_record_weight_one_and_modes(self):
        for mode in ("learned-attention", "uniform-mean", "last-only"):
            with self.subTest(mode=mode):
                head, batch = ContextHead("H3-short-attention", cfg(attention_mode=mode)), fixture(k=2)
                w = head(batch).attention_weights
                torch.testing.assert_close(w[:, 1, :, 0], torch.ones(2, 4), rtol=0, atol=0)
                self.assertEqual(w[..., 1].count_nonzero().item(), 0)
                larger = head(fixture()).attention_weights
                torch.testing.assert_close(larger[:, 1:].sum(-1), torch.ones(2, 3, 4), rtol=1e-6, atol=1e-6)
                if mode == "uniform-mean":
                    torch.testing.assert_close(larger[:, 3, :, :3], torch.full((2, 4, 3), 1/3), rtol=0, atol=0)

    def test_gru_modes_and_k2_single_record(self):
        batch = fixture()
        full = ContextHead("H2-GRU", cfg())
        last = ContextHead("H2-GRU", cfg(memory_mode="last-only"))
        reset = ContextHead("H2-GRU", cfg(memory_mode="reset-every-step"))
        torch.testing.assert_close(last(batch).latent, reset(batch).latent, rtol=0, atol=0)
        for head in (last, reset):
            torch.testing.assert_close(full(batch).latent[:, 1], head(batch).latent[:, 1], rtol=0, atol=0)
        self.assertFalse(torch.equal(full(batch).latent[:, 3], last(batch).latent[:, 3]))

    def test_mask_validation_provisional_and_structure_barriers(self):
        batch = fixture()
        for name in MODEL_NAMES:
            head = ContextHead(name, cfg())
            bad_mask = batch.history_mask.clone()
            bad_mask[:, 1, 2] = True
            with self.assertRaises(ValueError):
                head(replace(batch, history_mask=bad_mask))
            provisional = replace(batch, token_state=torch.full_like(batch.token_state, 2))
            with self.assertRaises(ValueError):
                head(provisional)
            if name in ("H3-short-attention", "H4-causal-mixer"):
                head(provisional, allow_provisional=True)
            else:
                with self.assertRaises(ValueError):
                    head(provisional, allow_provisional=True)
            with self.assertRaises(ValueError):
                head(replace(batch, token_ids=torch.full_like(batch.token_ids, 10)))
            with self.assertRaises(ValueError):
                head(replace(batch, token_state=torch.zeros_like(batch.token_state)))
            with self.assertRaises(ValueError):
                head.step(batch)

    def test_empty_queries_no_history_singleton_long_and_unsorted(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                head, batch = ContextHead(name, cfg()), fixture()
                empty = batch.queries(batch.query_indices[:, :0], batch.history_mask[:, :0])
                self.assertEqual(head(empty).latent.shape, (2, 0, 256))
                self.assertEqual(head(fixture(k=1)).latent.count_nonzero().item(), 0)
                nohist = replace(batch, history_mask=torch.zeros_like(batch.history_mask))
                self.assertEqual(head(nohist).latent.count_nonzero().item(), 0)
                self.assertTrue(torch.isfinite(ContextHead(name, cfg(max_slots=8))(fixture(k=8)).latent).all())
                unsorted = fixture(positions=[2, 0, 3, 1])
                torch.testing.assert_close(head(unsorted).latent[:, 1], torch.zeros(2, 256), rtol=0, atol=0)

    def test_split_chains_gap_and_barrier_and_explicit_opt_in(self):
        pos = torch.tensor([[3, 0, 1, 4, 9]])
        valid = torch.ones_like(pos, dtype=torch.bool)
        chains = split_chains(pos, valid)
        self.assertEqual(chains.tolist(), [[1, 0, 0, 1, 2]])
        barrier = torch.tensor([[False, False, True, False, False]])
        self.assertEqual(split_chains(pos, valid, barrier).tolist(), [[1, 0, -1, 1, 2]])
        self.assertEqual(split_chains(pos, valid, allow_gaps=True).tolist(), [[0, 0, 0, 0, 0]])
        gap = fixture(positions=[0, 1, 4, 5])
        with self.assertRaises(ValueError):
            ContextHead("H0-static", cfg())(gap)
        ContextHead("H0-static", cfg(allow_gaps=True))(gap)

    def test_full_vocab_kl_direction_temperature_mask_and_softmax_special(self):
        old = torch.tensor([[[1., 0., -1.], [torch.nan]*3]], requires_grad=True)
        delta = torch.tensor([[[.1, -.2, 0.], [torch.nan]*3]], requires_grad=True)
        teacher = torch.tensor([[[0., 2., 1.], [torch.nan]*3]], requires_grad=True)
        loss = teacher_kl(old, delta, teacher, torch.tensor([[True, False]]))
        p = teacher[0, 0].softmax(-1)
        expected = (p * (p.log() - (old[0, 0] + delta[0, 0]).log_softmax(-1))).sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(delta.grad).all())
        # Special delta=0 still changes its normalized probability.
        self.assertNotEqual(old[0, 0].softmax(-1)[2].item(), (old[0, 0]+delta[0, 0]).softmax(-1)[2].item())
        empty_loss = teacher_kl(old, delta, teacher, torch.zeros(1, 2, dtype=torch.bool))
        self.assertEqual(empty_loss.item(), 0)

    def test_shared_initial_tables_seed_rng_and_meta(self):
        before = torch.random.get_rng_state()
        reference = ContextHead("H0-static", cfg())
        for name in MODEL_NAMES:
            head = ContextHead(name, cfg())
            torch.testing.assert_close(head.output.weight, reference.output.weight, rtol=0, atol=0)
            if hasattr(head, "embedding"):
                torch.testing.assert_close(head.embedding.weight, reference.embedding.weight, rtol=0, atol=0)
            with torch.device("meta"):
                meta = ContextHead(name, HeadConfig(152064, 3584))
            self.assertTrue(all(p.is_meta for p in meta.parameters()))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_sampler_adapter_example_uses_full_aligned_set(self):
        from .adapter_example import corrected_scores, cpu_mock_example
        self.assertEqual(cpu_mock_example(), (1, 4, 11))
        batch = fixture()
        for name in MODEL_NAMES:
            head = ContextHead(name, cfg())
            expected = batch.old_logits + head(batch).delta_logits
            torch.testing.assert_close(corrected_scores(head, vars(batch)), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            corrected_scores(head, vars(replace(batch, old_logits=None)))


def propagation_head(k=4):
    """Nonzero H4 which flips the preceding binary token: requires all 3 rounds."""
    head = ContextHead("H4-causal-mixer", cfg(max_slots=k))
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.embedding.weight[0, 0] = 1
        head.embedding.weight[1, 1] = 1
        head.output.weight[1, 0] = 20
        head.output.weight[0, 1] = 20
        head.fuse_record.weight[:, 512:] = torch.eye(256)
        for j in range(1, k):
            head.mixer[j, j-1] = 1
    return head


class ExecutorsCPU(unittest.TestCase):
    def test_long_chain_needs_length_minus_one_rounds(self):
        head, batch = propagation_head(k=8), fixture(k=8, b=1)
        batch = replace(batch, old_logits=torch.zeros_like(batch.old_logits))
        initial, u = torch.zeros(1, 8, dtype=torch.long), torch.full((1, 8), .5)
        seq = execute(head, batch, initial, u, greedy_mock)
        j3 = execute(head, batch, initial, u, greedy_mock, mode="jacobi", rounds=3)
        j7 = execute(head, batch, initial, u, greedy_mock, mode="jacobi", rounds=7)
        self.assertFalse(torch.equal(seq.token_ids, j3.token_ids))
        self.assertTrue(torch.equal(seq.token_ids, j7.token_ids))

    def test_padding_and_tokenizer_padding_support(self):
        head, batch = ContextHead("H4-causal-mixer", cfg(tokenizer_size=9)), fixture(b=1)
        valid, chain = batch.selected_mask.clone(), batch.chain_ids.clone()
        valid[:, 3], chain[:, 3] = False, -1
        batch = replace(batch, selected_mask=valid, chain_ids=chain)
        batch = replace(batch, history_mask=history_for(batch, batch.query_indices))
        initial = torch.tensor([[0, 0, 0, 9999]])
        u = torch.full((1, 4), .999)
        seq = execute(head, batch, initial, u, inverse_cdf_mock)
        self.assertEqual(seq.token_ids[0, 3].item(), 9999)
        self.assertTrue((seq.token_ids[valid] < 9).all())
        self.assertEqual(seq.projected_positions, 2)
        empty = replace(batch, selected_mask=torch.zeros_like(valid), chain_ids=torch.full_like(chain, -1),
                        history_mask=torch.zeros_like(batch.history_mask))
        result = execute(head, empty, initial, u, inverse_cdf_mock, mode="jacobi")
        self.assertEqual(result.projected_positions, 0)
        self.assertTrue(torch.equal(result.token_ids, initial))

    def test_h4_nonzero_one_two_round_counterexamples_three_exact(self):
        head, batch = propagation_head(), fixture(b=1)
        batch = replace(batch, old_logits=torch.zeros_like(batch.old_logits))
        initial, u = torch.zeros(1, 4, dtype=torch.long), torch.full((1, 4), .5)
        seq = execute(head, batch, initial, u, greedy_mock)
        self.assertEqual(seq.token_ids.tolist(), [[0, 1, 0, 1]])
        self.assertEqual(seq.positions_per_call, [1, 1, 1])
        expected = {1: [[0, 1, 1, 1]], 2: [[0, 1, 0, 0]], 3: [[0, 1, 0, 1]]}
        for n in (1, 2, 3):
            jac = execute(head, batch, initial, u, greedy_mock, mode="jacobi", rounds=n)
            self.assertEqual(jac.token_ids.tolist(), expected[n])
            self.assertEqual(jac.positions_per_call, [3]*n)
            self.assertEqual(jac.projected_positions, 3*n)
            self.assertEqual(jac.sampling_positions, 3*n)
            self.assertEqual(jac.token_state.tolist(), [[1, 2, 2, 2]])

    def test_dense_h3_h4_fixed_u_stochastic_three_round_equality(self):
        for name in ("H3-short-attention", "H4-causal-mixer"):
            for seed in (17, 43, 71):
                with self.subTest(model=name, seed=seed):
                    head, batch = ContextHead(name, cfg(seed=seed)), fixture()
                    g = torch.Generator().manual_seed(seed)
                    initial = torch.randint(0, 10, (2, 4), generator=g)
                    u = torch.rand(2, 4, generator=g).clamp(.001, .999)
                    seq = execute(head, batch, initial, u, inverse_cdf_mock)
                    jac = execute(head, batch, initial, u, inverse_cdf_mock, mode="jacobi", rounds=3)
                    self.assertTrue(torch.equal(seq.token_ids, jac.token_ids))
                    self.assertGreater(sum(p.abs().sum().item() for p in head.parameters()), 0)
                    # Verify actual token sensitivity, beyond a constant-output head.
                    changed = replace(batch, token_ids=(batch.token_ids+1) % 10)
                    self.assertFalse(torch.equal(head(batch).latent[:, 3], head(changed).latent[:, 3]))

    def test_redrawing_random_input_breaks_equivalence(self):
        head, batch = ContextHead("H4-causal-mixer", cfg()), fixture(b=1)
        initial, u = torch.zeros(1, 4, dtype=torch.long), torch.full((1, 4), .5)

        class BadRedraw:
            def __init__(self):
                self.calls = 0

            def __call__(self, scores, ignored_u):
                self.calls += 1
                draw = .001 if self.calls % 2 else .999
                return inverse_cdf_mock(scores, torch.full_like(ignored_u, draw))

        seq = execute(head, batch, initial, u, BadRedraw())
        jac = execute(head, batch, initial, u, BadRedraw(), mode="jacobi", rounds=3)
        self.assertFalse(torch.equal(seq.token_ids, jac.token_ids))

    def test_multichain_short_long_empty_and_barrier_support(self):
        head = ContextHead("H4-causal-mixer", cfg(max_slots=8))
        for positions, chains in (([0, 1, 8, 9], [0, 0, 1, 1]),
                                   ([0, 4, 8, 9], [0, 1, 2, 2]),
                                   ([0], [0]), (list(range(8)), [0]*8),
                                   ([2, 0, 3, 1], [0]*4)):
            with self.subTest(positions=positions):
                batch = fixture(k=len(positions), positions=positions, chains=chains)
                initial = batch.token_ids.clone()
                u = torch.full_like(initial, .4, dtype=torch.float)
                seq = execute(head, batch, initial, u, inverse_cdf_mock)
                jac = execute(head, batch, initial, u, inverse_cdf_mock, mode="jacobi", rounds=max(1, len(positions)-1))
                self.assertTrue(torch.equal(seq.token_ids, jac.token_ids))
        batch = fixture()
        invalid = batch.token_ids.clone()
        invalid[:, 2] = 10  # provisional EOS/expand is not ordinary memory
        with self.assertRaises(ValueError):
            execute(head, batch, invalid, torch.full((2, 4), .5), greedy_mock, mode="jacobi")
        with self.assertRaises(ValueError):
            execute(head, batch, batch.token_ids, torch.zeros(2, 4), greedy_mock)
        with self.assertRaises(ValueError):
            execute(head, batch, batch.token_ids, torch.full((2, 4), .5),
                    lambda scores, u: torch.full(u.shape, 10, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
