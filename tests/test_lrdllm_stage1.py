from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from expvision_dllm_clean.lrdllm_stage1 import (
    FIT_SCOPE,
    build_exponential_probe_lengths,
    compute_span_average_negative_entropy,
    fit_log_length_bias,
    select_best_corrected_length,
    select_length_lrdllm_stage1,
)


class FakeTokenizer:
    mask_token_id = 7

    def __call__(self, text, *, add_special_tokens, padding, return_tensors):
        if add_special_tokens or padding is not True or return_tensors != "pt":
            raise AssertionError("selector tokenization diverged from official adapter")
        values = [1 + (ord(character) % 5) for character in str(text)] or [1]
        return {
            "input_ids": torch.tensor([values], dtype=torch.long),
            "attention_mask": torch.ones((1, len(values)), dtype=torch.long),
        }


class CountingModel:
    def __init__(self, *, nan: bool = False) -> None:
        self.device = torch.device("cpu")
        self.forward_count = 0
        self.training = True
        self.nan = nan

    def eval(self):
        self.training = False
        return self

    def __call__(self, input_ids, attention_mask=None):
        self.forward_count += 1
        if attention_mask is None or attention_mask.shape != input_ids.shape:
            raise AssertionError("missing or malformed attention mask")
        batch, sequence = input_ids.shape
        logits = torch.zeros((batch, sequence, 4), dtype=torch.float32)
        logits[..., 0] = float(sequence) / 10.0
        if self.nan:
            logits[..., 0] = float("nan")
        return SimpleNamespace(logits=logits)


class GuardedTask(dict):
    forbidden = {
        "canonical_solution",
        "reference_completion",
        "oracle_length",
        "passed",
        "task_id",
        "test",
        "tests",
        "evaluator",
    }

    def __contains__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"forbidden task field inspected: {key}")
        return super().__contains__(key)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"forbidden task field read: {key}")
        return super().__getitem__(key)


class LrDllmStage1Test(unittest.TestCase):
    def test_max_length_128_has_exact_eight_exponential_lengths(self) -> None:
        self.assertEqual(build_exponential_probe_lengths(128), [1, 2, 4, 8, 16, 32, 64, 128])

    def test_non_power_of_two_limit_is_not_appended(self) -> None:
        self.assertEqual(build_exponential_probe_lengths(100), [1, 2, 4, 8, 16, 32, 64])

    def test_uniform_distribution_has_lower_negative_entropy_than_sharp_distribution(self) -> None:
        uniform = compute_span_average_negative_entropy(torch.zeros((1, 1, 4)), 0, 1)
        sharp = compute_span_average_negative_entropy(
            torch.tensor([[[20.0, -20.0, -20.0, -20.0]]]), 0, 1
        )
        self.assertLess(uniform, sharp)
        self.assertAlmostEqual(uniform, -math.log(4.0), places=6)

    def test_negative_entropy_counts_only_mask_span(self) -> None:
        logits = torch.tensor([[[20.0, -20.0], [0.0, 0.0], [20.0, -20.0]]])
        score = compute_span_average_negative_entropy(logits, 1, 2)
        self.assertAlmostEqual(score, -math.log(2.0), places=6)

    def test_synthetic_log_line_recovers_slope_and_intercept(self) -> None:
        lengths = [1, 2, 4, 8, 16, 32, 64, 128]
        intercept, slope = -3.25, 0.4
        scores = [intercept + slope * math.log(length) for length in lengths]
        fit = fit_log_length_bias(lengths, scores)
        self.assertAlmostEqual(fit["k_hat"], slope, places=12)
        self.assertAlmostEqual(fit["intercept"], intercept, places=12)
        self.assertAlmostEqual(fit["fit_r2"], 1.0, places=12)

    def test_corrected_score_uses_subtraction_not_addition(self) -> None:
        result = select_length_lrdllm_stage1(
            {"prompt": "ab", "suffix": "c"}, FakeTokenizer(), CountingModel(), {"max_probe_length": 8}
        )
        self.assertNotEqual(result["k_hat"], 0.0)
        for length in result["probe_lengths"]:
            raw = result["average_negative_entropy_by_length"][str(length)]
            log_length = result["log_length_by_length"][str(length)]
            expected = raw - result["k_hat"] * log_length
            wrong_sign = raw + result["k_hat"] * log_length
            self.assertAlmostEqual(result["cl_score_by_length"][str(length)], expected, places=12)
            if length > 1:
                self.assertNotAlmostEqual(expected, wrong_sign, places=8)

    def test_primary_fit_scope_contains_all_exponential_points(self) -> None:
        result = select_length_lrdllm_stage1(
            {"prompt": "p", "suffix": "s"}, FakeTokenizer(), CountingModel(), {"max_probe_length": 128}
        )
        self.assertEqual(result["fit_scope"], FIT_SCOPE)
        self.assertEqual(list(result["log_length_by_length"]), ["1", "2", "4", "8", "16", "32", "64", "128"])

    def test_negative_slope_is_not_clamped_or_absolute_valued(self) -> None:
        lengths = [1, 2, 4, 8]
        fit = fit_log_length_bias(lengths, [2.0 - 0.7 * math.log(length) for length in lengths])
        self.assertAlmostEqual(fit["k_hat"], -0.7, places=12)

    def test_exact_tie_break_is_deterministically_shorter(self) -> None:
        self.assertEqual(select_best_corrected_length([1, 2, 4], [-2.0, -2.0, -2.0]), 1)

    def test_each_probe_length_has_exactly_one_forward(self) -> None:
        model = CountingModel()
        result = select_length_lrdllm_stage1(
            {"prompt": "p", "suffix": "s"}, FakeTokenizer(), model, {"max_probe_length": 64}
        )
        self.assertEqual(model.forward_count, len(result["probe_lengths"]))
        self.assertFalse(model.training)

    def test_probe_forward_count_is_eight_at_128(self) -> None:
        model = CountingModel()
        result = select_length_lrdllm_stage1(
            {"prompt": "p", "suffix": "s"}, FakeTokenizer(), model, {"max_probe_length": 128}
        )
        self.assertEqual(result["probe_forward_count"], 8)
        self.assertEqual(model.forward_count, 8)

    def test_nan_and_inf_fail_stop(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            fit_log_length_bias([1, 2], [0.0, float("inf")])
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            select_length_lrdllm_stage1(
                {"prompt": "p", "suffix": "s"}, FakeTokenizer(), CountingModel(nan=True), {"max_probe_length": 2}
            )

    def test_selector_does_not_read_oracle_reference_passed_or_evaluator(self) -> None:
        task = GuardedTask(prompt="p", suffix="s")
        result = select_length_lrdllm_stage1(
            task, FakeTokenizer(), CountingModel(), {"max_probe_length": 4}
        )
        self.assertIn(result["selected_length"], [1, 2, 4])

    def test_same_input_repeats_same_selection(self) -> None:
        task = {"prompt": "repeat", "suffix": "same"}
        first = select_length_lrdllm_stage1(
            task, FakeTokenizer(), CountingModel(), {"max_probe_length": 16}
        )
        second = select_length_lrdllm_stage1(
            task, FakeTokenizer(), CountingModel(), {"max_probe_length": 16}
        )
        self.assertEqual(first["selected_length"], second["selected_length"])
        self.assertEqual(first["cl_score_by_length"], second["cl_score_by_length"])


if __name__ == "__main__":
    unittest.main()
