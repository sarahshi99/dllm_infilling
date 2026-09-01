from __future__ import annotations

import copy
import unittest

import torch
from torch.distributions import Categorical

from analysis.analyze_dreamon_markov_premise import (
    audit_required_transition_fields,
    completeness_status,
)
from experiments.dreamon_singleline_order_parallel import (
    GLOBAL_CONFIDENCE,
    LEFT_TO_RIGHT_FRONTIER,
    decode_with_policy,
    prepare_decode_distribution,
)


def official_sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        logits = logits.masked_fill(
            logits < torch.topk(logits, top_k)[0][..., -1, None],
            torch.finfo(logits.dtype).min,
        )
    probabilities = torch.softmax(logits, dim=-1)
    if temperature > 0:
        sampled = Categorical(probs=probabilities).sample()
        confidence = torch.gather(probabilities, -1, sampled.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, sampled = probabilities.max(dim=-1)
    if neg_entropy:
        confidence = torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)
    return confidence, sampled


class TinyTokenizer:
    mask_id = 0
    eos_id = 1
    expand_id = 2

    def decode(self, tokens: list[int], **_: object) -> str:
        return ",".join(map(str, tokens))


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, structural_call: int | None = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.calls = 0
        self.structural_call = structural_call

    def forward(self, x: torch.Tensor, *_: object) -> object:
        logits = torch.full((*x.shape, self.vocab_size), -4.0)
        preferred = 2 if self.calls == self.structural_call else 3 + (self.calls % 3)
        logits[..., preferred] = 2.0
        logits[..., 7] = 1.0
        self.calls += 1
        return type("Output", (), {"logits": logits})()


def trajectory_view(decoded: dict[str, object]) -> dict[str, object]:
    return {
        "completion": decoded["completion"],
        "completion_token_ids": decoded["completion_token_ids"],
        "step_trace": [
            {
                key: row[key]
                for key in (
                    "selected_positions",
                    "selected_token_ids",
                    "executed_expand_count",
                    "executed_delete_count",
                    "canvas_state_after",
                )
            }
            for row in decoded["step_trace"]  # type: ignore[index]
        ],
    }


class DreamOnMarkovPremiseTest(unittest.TestCase):
    def test_actual_decode_distribution_matches_official_temperature_top_p(self) -> None:
        logits = torch.tensor(
            [[3.0, 2.5, 1.0, -0.5], [0.1, 0.0, -0.1, -0.2]], dtype=torch.bfloat16
        )
        _, official_tokens = official_sample_tokens(
            logits.clone(), temperature=0.2, top_p=0.9, neg_entropy=True
        )
        prepared = prepare_decode_distribution(
            logits, temperature=0.2, top_p=0.9, top_k=None
        )
        expected_logits = logits.clone() / 0.2
        sorted_logits, sorted_indices = torch.sort(expected_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > 0.9
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        mask = torch.zeros_like(expected_logits, dtype=torch.bool).scatter_(
            -1, sorted_indices, remove
        )
        expected_probabilities = torch.softmax(
            expected_logits.masked_fill(mask, torch.finfo(expected_logits.dtype).min), dim=-1
        )
        torch.testing.assert_close(prepared.probabilities, expected_probabilities)
        torch.manual_seed(123)
        expected = Categorical(probs=prepared.probabilities).sample()
        torch.manual_seed(123)
        _, actual = official_sample_tokens(
            logits.clone(), temperature=0.2, top_p=0.9, neg_entropy=True
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            prepared.retained_mask.tolist(),
            [[True, False, False, False], [True, True, True, True]],
        )
        self.assertEqual(official_tokens.shape, actual.shape)

    def test_distribution_diagnostics_do_not_consume_rng(self) -> None:
        logits = torch.randn(3, 11, dtype=torch.bfloat16)
        torch.manual_seed(9123)
        before = torch.random.get_rng_state().clone()
        prepare_decode_distribution(logits, temperature=0.2, top_p=0.9, top_k=None)
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))

    def test_reference_poison_cannot_change_trajectory(self) -> None:
        def run(reference_ids: list[int]) -> dict[str, object]:
            torch.manual_seed(77)
            return decode_with_policy(
                model=TinyModel(),
                tokenizer=TinyTokenizer(),
                sample_tokens_fn=official_sample_tokens,
                prefix_ids=[5],
                suffix_ids=[1],
                reference_ids=reference_ids,
                min_gen_len=4,
                max_gen_len=4,
                max_tokens=8,
                steps=8,
                expand_budget=0,
                temperature=0.2,
                top_p=0.9,
                top_k=None,
                requested_k=1,
                policy=GLOBAL_CONFIDENCE,
                device="cpu",
            )

        left = run([3, 4, 5, 3])
        right = run([7, 7, 7, 7])
        self.assertEqual(trajectory_view(left), trajectory_view(right))
        self.assertNotEqual(left["markov_transitions"], right["markov_transitions"])
        replay_fields = (
            "source_step_index",
            "trajectory_policy",
            "stale_input_ids",
            "fresh_input_ids",
            "previous_token_id",
            "target_position",
            "active_mask_count",
            "generation_stage",
            "canvas_length",
            "expand_action_masked",
        )
        self.assertEqual(
            [{field: row[field] for field in replay_fields} for row in left["replay_transitions"]],
            [{field: row[field] for field in replay_fields} for row in right["replay_transitions"]],
        )
        for row in left["replay_transitions"]:
            self.assertEqual(
                row["target_position"],
                row["stale_input_ids"].index(TinyTokenizer.mask_id, row["target_position"]),
            )
            self.assertEqual(
                row["fresh_input_ids"][row["target_position"]], TinyTokenizer.mask_id
            )
            self.assertEqual(
                row["fresh_input_ids"][row["target_position"] - 1],
                row["previous_token_id"],
            )

    def test_structure_action_aborts_pending_chain(self) -> None:
        torch.manual_seed(4)
        decoded = decode_with_policy(
            model=TinyModel(structural_call=1),
            tokenizer=TinyTokenizer(),
            sample_tokens_fn=official_sample_tokens,
            prefix_ids=[5],
            suffix_ids=[1],
            reference_ids=[3, 4, 5, 3],
            min_gen_len=4,
            max_gen_len=5,
            max_tokens=8,
            steps=3,
            expand_budget=1,
            temperature=0.0,
            top_p=0.9,
            top_k=None,
            requested_k=1,
            policy=LEFT_TO_RIGHT_FRONTIER,
            device="cpu",
        )
        aborted = [
            row
            for row in decoded["markov_transitions"]
            if not row["transition_eligible"]
        ]
        self.assertTrue(aborted)
        structural = [
            row
            for row in aborted
            if row["transition_ineligible_reason"] == "structural_or_coordinate_change"
        ]
        self.assertEqual({row["offset"] for row in structural}, {2, 3})

    def test_required_schema_and_hard_gate(self) -> None:
        decoded = decode_with_policy(
            model=TinyModel(),
            tokenizer=TinyTokenizer(),
            sample_tokens_fn=official_sample_tokens,
            prefix_ids=[5],
            suffix_ids=[1],
            reference_ids=[3, 4, 5, 3],
            min_gen_len=4,
            max_gen_len=4,
            max_tokens=8,
            steps=8,
            expand_budget=0,
            temperature=0.0,
            top_p=0.9,
            top_k=None,
            requested_k=1,
            policy=LEFT_TO_RIGHT_FRONTIER,
            device="cpu",
        )
        eligible = next(
            row for row in decoded["markov_transitions"] if row["transition_eligible"]
        )
        self.assertEqual(audit_required_transition_fields([eligible]), [])
        broken = copy.deepcopy(eligible)
        broken["fresh_global_rank"] = None
        self.assertIn("fresh_global_rank", audit_required_transition_fields([broken]))
        self.assertEqual(
            completeness_status(
                case_rows=2066,
                variant_counts={"C1": 1033, "L1": 1033},
                common_task_count=1033,
                task_group_count=164,
                duplicate_count=0,
                missing_count=0,
                error_count=0,
                pass_counts={"C1": 951, "L1": 942},
                required_field_errors=[],
                compressed_artifacts_valid=True,
            ),
            "completed",
        )
        self.assertEqual(
            completeness_status(
                case_rows=2065,
                variant_counts={"C1": 1033, "L1": 1032},
                common_task_count=1032,
                task_group_count=164,
                duplicate_count=0,
                missing_count=1,
                error_count=0,
                pass_counts={"C1": 951, "L1": 941},
                required_field_errors=[],
                compressed_artifacts_valid=True,
            ),
            "incomplete",
        )


if __name__ == "__main__":
    unittest.main()
