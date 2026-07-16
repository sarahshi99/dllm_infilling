import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.m3_birth_death_canvas import (
    CANVASES,
    DEATH_ROUNDS,
    METHODS,
    STEPS_PER_PARTICLE,
    TOTAL_FORWARDS,
    birth_death_enabled,
    birth_death_reallocate,
    confidence_retained_positions,
    expected_keys,
    particle_round_token_forwards,
    parser,
    remaining_lifetime_steps,
    require_isolated_output_dirs,
    run_method,
    visible_particle_score,
    visible_task,
)
from expvision_dllm_clean.decode import linear_target_masks
import experiments.m3_birth_death_canvas as m3


class Tokenizer:
    mapping = {0: "    total = sum(xs)\n", 1: ""}

    def decode(self, ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        return "".join(self.mapping[int(item)] for item in ids)


class M3BirthDeathCanvasTest(unittest.TestCase):
    def test_uniform_and_birth_death_use_identical_total_forward_budget(self) -> None:
        self.assertEqual(CANVASES, (16, 32, 64, 128))
        self.assertEqual(TOTAL_FORWARDS, len(CANVASES) * STEPS_PER_PARTICLE)
        self.assertEqual(TOTAL_FORWARDS, 256)

    def test_visible_particle_score_has_no_evaluator_state(self) -> None:
        particle = {
            "prepared": {"middle_start": 0, "middle_end": 2},
            "x_t": torch.tensor([[0, 1]]),
            "canvas_tokens": 16,
            "last_confidences": [0.8, 0.7],
        }
        score = visible_particle_score(particle, "def f(xs):\n", "    return total\n", Tokenizer())
        self.assertEqual(score[0], 1)
        self.assertGreater(score[3], 0.0)

    def test_visible_task_does_not_carry_test_or_reference(self) -> None:
        task = visible_task("def f(xs):\n", "    return total\n")
        self.assertEqual(task.test_code, "")
        self.assertIsNone(task.canonical_solution)

    def test_birth_death_event_is_deterministic_and_changes_population(self) -> None:
        particles = [
            {
                "canvas_tokens": 16,
                "score": (0, 0, 0, 0.1, -16),
                "prepared": {"middle_start": 0, "middle_end": 2},
                "x_t": torch.tensor([[0, 1]]),
            },
            {
                "canvas_tokens": 64,
                "score": (1, 0, 0, 0.9, -64),
                "prepared": {"middle_start": 0, "middle_end": 2},
                "x_t": torch.tensor([[0, 1]]),
            },
        ]

        def score(particle, prefix, suffix, tokenizer):
            del prefix, suffix, tokenizer
            return particle["score"]

        def factory(**kwargs):
            return {
                "canvas_tokens": kwargs["canvas_tokens"],
                "born_round": kwargs["born_round"],
                "copied_text": kwargs["candidate_text"],
                "score": (1, 0, 0, 0.9, -64),
            }

        event = birth_death_reallocate(
            particles=particles,
            prefix="def f(xs):\n",
            suffix="    return total\n",
            tokenizer=Tokenizer(),
            model=object(),
            task=object(),
            round_index=15,
            score_fn=score,
            particle_factory=factory,
        )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["dead_canvas"], 16)
        self.assertEqual(event["born_canvas"], 64)
        self.assertEqual(particles[0]["canvas_tokens"], 64)
        self.assertEqual(particles[0]["born_round"], 16)

    def test_method_key_population_and_smoke_default(self) -> None:
        self.assertEqual(len(expected_keys([{"row_key": "a"}], METHODS[0])), 1)
        args = parser().parse_args(
            [
                "--dataset-jsonl", "data.jsonl",
                "--uniform-output-dir", "uniform",
                "--birth-death-output-dir", "birth",
                "--compact-dir", "compact",
            ]
        )
        self.assertEqual(args.smoke_cases, 12)

    def test_removing_constant_particle_penalty_preserves_within_particle_ordering(self) -> None:
        confidences = [0.2, 0.1, 0.1, 0.4]
        masked = [0, 1, 2, 3]
        penalty = 0.85
        legacy = [index for _, index in sorted((confidence - penalty, index) for index, confidence in enumerate(confidences))[:3]]
        self.assertEqual(confidence_retained_positions(confidences, masked, 3), legacy)

    def test_newborn_schedule_ends_on_global_final_round(self) -> None:
        born_round = 16
        lifetime = remaining_lifetime_steps(born_round)
        self.assertEqual(lifetime, len(range(born_round, STEPS_PER_PARTICLE)))
        self.assertEqual(linear_target_masks(64, lifetime, lifetime - 1), 0)

    def test_uniform_has_no_birth_death_event_and_token_budget_uses_live_particles(self) -> None:
        self.assertFalse(any(birth_death_enabled(METHODS[0], round_index) for round_index in range(STEPS_PER_PARTICLE)))
        self.assertTrue(all(birth_death_enabled(METHODS[1], round_index) for round_index in DEATH_ROUNDS))
        self.assertEqual(particle_round_token_forwards([{"canvas_tokens": 16}, {"canvas_tokens": 32}, {"canvas_tokens": 64}, {"canvas_tokens": 128}]), 240)
        self.assertEqual(particle_round_token_forwards([{"canvas_tokens": 64}, {"canvas_tokens": 32}, {"canvas_tokens": 64}, {"canvas_tokens": 128}]), 288)

    def test_resume_dedup_and_output_directory_isolation(self) -> None:
        key = next(iter(expected_keys([{"row_key": "a"}], METHODS[0])))
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "uniform.jsonl"
            m3.append_jsonl(raw, {"candidate_key": key, "candidate_kind": METHODS[0], "status": "ok"})
            self.assertEqual(run_method(method=METHODS[0], manifest=[{"row_key": "a", "source_row_id": 0}], source_rows=[], raw_path=raw, tokenizer=None, model=None), 0)
            m3.append_jsonl(raw, {"candidate_key": key, "candidate_kind": METHODS[0], "status": "ok"})
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                run_method(method=METHODS[0], manifest=[{"row_key": "a", "source_row_id": 0}], source_rows=[], raw_path=raw, tokenizer=None, model=None)
            with self.assertRaisesRegex(ValueError, "independent"):
                require_isolated_output_dirs(raw.parent, raw.parent)

    def test_deployable_decode_does_not_receive_poisoned_evaluator_or_label_fields(self) -> None:
        class Poison:
            def __str__(self) -> str:
                raise AssertionError("forbidden field reached deployable decoding")

        source = {
            "prompt": "def f(x):\n",
            "suffix": "    return x\n",
            "canonical_solution": Poison(),
            "reference": Poison(),
            "test": Poison(),
            "entry_point": Poison(),
            "passed": Poison(),
            "task_id": Poison(),
            "task_group": Poison(),
            "split": Poison(),
            "oracle_length": Poison(),
        }
        calls: list[tuple[str, str]] = []

        def fake_decode(*, method, prefix, suffix, tokenizer, model):
            del method, tokenizer, model
            calls.append((prefix, suffix))
            return {"middle_text": "", "middle_token_ids": [], "final_token_confidences": [], "selected_canvas_tokens": 64, "selected_visible_score": [], "particle_canvases_final": [], "birth_death_events": [], "actual_forward_count": 256, "actual_token_forward_budget": 4096}

        def fake_result(method, item, source_row, decoded):
            del source_row, decoded
            return {"candidate_key": m3.method_key(item["row_key"], method), "candidate_kind": method, "status": "ok", "metrics": {}}

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(m3, "decode_population", side_effect=fake_decode), mock.patch.object(m3, "result_row", side_effect=fake_result), mock.patch.object(m3, "set_global_seed") as set_seed:
                written = run_method(method=METHODS[0], manifest=[{"row_key": "a", "source_row_id": 0, "case_index": 0}], source_rows=[source], raw_path=Path(directory) / "raw.jsonl", tokenizer=object(), model=object())
        self.assertEqual(written, 1)
        self.assertEqual(calls, [("def f(x):\n", "    return x\n")])
        set_seed.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
