import unittest

import torch

from experiments.m3_birth_death_canvas import (
    CANVASES,
    METHODS,
    STEPS_PER_PARTICLE,
    TOTAL_FORWARDS,
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
        self.assertEqual(task.canonical_solution, "")

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
