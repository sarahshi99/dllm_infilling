import unittest

import torch

from experiments.m3_birth_death_canvas import (
    CANVASES,
    METHODS,
    STEPS_PER_PARTICLE,
    TOTAL_FORWARDS,
    birth_death_reallocate,
    expected_keys,
    parser,
    visible_particle_score,
    visible_task,
)


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


if __name__ == "__main__":
    unittest.main()
