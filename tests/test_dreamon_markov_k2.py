from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from experiments.dreamon_singleline_order_parallel import (
    GLOBAL_CONFIDENCE,
    K2_SEQUENTIAL_ACTIONS,
    LEFT_TO_RIGHT_FRONTIER,
    decode_with_policy,
    k2_pair_decision,
)


class FakeTokenizer:
    mask_id = 3
    eos_id = 2
    expand_id = 4

    def decode(self, values: list[int], **_kwargs: object) -> str:
        return ",".join(map(str, values))


class FixedModel(torch.nn.Module):
    def __init__(self, *, normal_id: int = 5, eos_logit: float = -10.0) -> None:
        super().__init__()
        self.normal_id = normal_id
        self.eos_logit = eos_logit
        self.calls = 0

    def forward(self, x: torch.Tensor, *_args: object) -> SimpleNamespace:
        self.calls += 1
        logits = torch.full((*x.shape, 8), -20.0, device=x.device)
        logits[..., self.normal_id] = 10.0
        logits[..., 2] = self.eos_logit
        return SimpleNamespace(logits=logits)


class ConditionalHead(torch.nn.Module):
    def __init__(self, *, structural_second: bool = False) -> None:
        super().__init__()
        self.calls = 0
        self.structural_second = structural_second

    def forward(self, previous: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        result = torch.zeros((previous.numel(), 8), device=previous.device)
        if self.structural_second:
            result[:, 5] = -100.0  # Structural bias remains zero.
        else:
            result[:, 6] = 40.0
        return result


def deterministic_sample(
    logits: torch.Tensor, *, temperature: float, top_p: float | None, top_k: int | None, neg_entropy: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    del temperature, top_p, top_k, neg_entropy
    probabilities = torch.softmax(logits.float(), dim=-1)
    return probabilities.max(dim=-1).values, probabilities.argmax(dim=-1)


class DreamOnMarkovK2Test(unittest.TestCase):
    def decode(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "model": FixedModel(),
            "tokenizer": FakeTokenizer(),
            "sample_tokens_fn": deterministic_sample,
            "prefix_ids": [1],
            "suffix_ids": [2],
            "min_gen_len": 2,
            "max_gen_len": 2,
            "max_tokens": 16,
            "steps": 8,
            "expand_budget": 2,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": None,
            "requested_k": 2,
            "policy": LEFT_TO_RIGHT_FRONTIER,
            "device": "cpu",
            "k2_action_mode": K2_SEQUENTIAL_ACTIONS,
        }
        values.update(overrides)
        return decode_with_policy(**values)  # type: ignore[arg-type]

    def test_global_pair_keeps_original_top2_but_orders_adjacent_pair_spatially(self) -> None:
        decision = k2_pair_decision(
            selected_ranks=[1, 0],
            absolute_positions=[9, 10],
            policy=GLOBAL_CONFIDENCE,
        )
        self.assertEqual(decision.selected_ranks, [1, 0])
        self.assertEqual((decision.left_rank, decision.right_rank), (0, 1))
        self.assertTrue(decision.sequential_actions)

    def test_nonadjacent_global_top2_stays_parallel_fallback(self) -> None:
        decision = k2_pair_decision(
            selected_ranks=[0, 2],
            absolute_positions=[4, 5, 7],
            policy=GLOBAL_CONFIDENCE,
        )
        self.assertFalse(decision.sequential_actions)
        self.assertEqual((decision.left_rank, decision.right_rank), (None, None))

    def test_k1_bypasses_head_even_when_lambda_is_nonzero(self) -> None:
        head = ConditionalHead()
        result = self.decode(
            min_gen_len=1,
            max_gen_len=1,
            requested_k=1,
            markov_head=head,
            markov_lambda=1.0,
        )
        self.assertEqual(head.calls, 0)
        self.assertEqual(result["completion_token_ids"], [5])

    def test_adjacent_left_frontier_corrects_only_second_token_same_forward(self) -> None:
        model = FixedModel()
        head = ConditionalHead()
        result = self.decode(model=model, markov_head=head, markov_lambda=1.0)
        self.assertEqual(result["completion_token_ids"], [5, 6])
        self.assertEqual(result["forward_count"], 1)
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["markov_enabled_count"], 1)

    def test_corrected_structural_token_stops_round_without_executing_old_action(self) -> None:
        # Base second token is ordinary (5); correction suppresses it, leaving the
        # previously valid EOS candidate. The EOS itself must not be executed.
        model = FixedModel(eos_logit=9.0)
        head = ConditionalHead(structural_second=True)
        result = self.decode(model=model, markov_head=head, markov_lambda=1.0)
        self.assertEqual(result["completion_token_ids"], [5, 5])
        self.assertEqual(result["corrected_structural_stop_count"], 1)
        self.assertEqual(result["delete_count"], 0)

    def test_zero_lambda_does_not_call_head(self) -> None:
        head = ConditionalHead()
        result = self.decode(markov_head=head, markov_lambda=0.0)
        self.assertEqual(head.calls, 0)
        self.assertEqual(result["completion_token_ids"], [5, 5])

    def test_zero_lambda_matches_the_same_no_head_path_without_extra_sampling(self) -> None:
        calls = {"base": 0, "zero": 0}

        def counted_base(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            calls["base"] += 1
            return deterministic_sample(*args, **kwargs)  # type: ignore[arg-type]

        def counted_zero(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            calls["zero"] += 1
            return deterministic_sample(*args, **kwargs)  # type: ignore[arg-type]

        baseline = self.decode(sample_tokens_fn=counted_base)
        zero = self.decode(
            sample_tokens_fn=counted_zero,
            markov_head=ConditionalHead(),
            markov_lambda=0.0,
        )
        self.assertEqual(zero["completion_token_ids"], baseline["completion_token_ids"])
        self.assertEqual(zero["step_trace"], baseline["step_trace"])
        self.assertEqual(calls["zero"], calls["base"])

    def test_global_nonadjacent_pair_never_calls_markov_head(self) -> None:
        head = ConditionalHead()

        def nonadjacent_sample(
            logits: torch.Tensor, **kwargs: object
        ) -> tuple[torch.Tensor, torch.Tensor]:
            _, predictions = deterministic_sample(logits, **kwargs)  # type: ignore[arg-type]
            confidence = torch.tensor(
                [0.9, 0.1, 0.8][: logits.shape[0]], device=logits.device
            )
            return confidence, predictions

        result = self.decode(
            min_gen_len=3,
            max_gen_len=3,
            policy=GLOBAL_CONFIDENCE,
            sample_tokens_fn=nonadjacent_sample,
            markov_head=head,
            markov_lambda=1.0,
        )
        self.assertEqual(head.calls, 0)
        self.assertEqual(result["nonadjacent_fallback_count"], 1)

    def test_markov_second_sample_does_not_advance_base_rng_stream(self) -> None:
        def stochastic_sample(
            logits: torch.Tensor, **_kwargs: object
        ) -> tuple[torch.Tensor, torch.Tensor]:
            probabilities = torch.softmax(logits.float(), dim=-1)
            sampled = torch.multinomial(probabilities, 1).squeeze(-1)
            return probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1), sampled

        torch.manual_seed(1234)
        self.decode(sample_tokens_fn=stochastic_sample)
        baseline_state = torch.random.get_rng_state()
        torch.manual_seed(1234)
        self.decode(
            sample_tokens_fn=stochastic_sample,
            markov_head=ConditionalHead(),
            markov_lambda=1.0,
            markov_sampling_seed=9876,
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), baseline_state))


if __name__ == "__main__":
    unittest.main()
