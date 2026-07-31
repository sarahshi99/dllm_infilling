from __future__ import annotations

import math
import random
import unittest

from experiments.lrdllm_algorithm import (
    MAX_GEN,
    PROBE_GRID,
    CommitState,
    ForwardTokenLedger,
    LocalSearchLimitError,
    adjusted_confidence,
    assert_selection_payload_safe,
    canonical_success_index,
    choose_local_length,
    commit_left_token,
    derive_row_seed,
    fit_log_length_confidence,
    local_neighbors,
    mean_negative_entropy,
    negative_entropy,
    run_local_search,
    sample_top_p,
    select_initial_length,
    validate_probe_grid,
)


class LrDllmAlgorithmTest(unittest.TestCase):
    def test_negative_entropy_is_meaned_over_masked_positions(self) -> None:
        self.assertAlmostEqual(negative_entropy([0.5, 0.5]), -math.log(2.0))
        self.assertAlmostEqual(negative_entropy([1.0, 0.0]), 0.0)
        self.assertAlmostEqual(mean_negative_entropy([[0.5, 0.5], [1.0, 0.0]]), -math.log(2.0) / 2.0)

    def test_primary_probe_grid_is_exact(self) -> None:
        self.assertEqual(validate_probe_grid(PROBE_GRID), PROBE_GRID)
        with self.assertRaises(ValueError):
            validate_probe_grid((2, 4, 8, 16, 32, 64, 128))
        unordered = {length: -1.0 for length in reversed(PROBE_GRID)}
        self.assertEqual(select_initial_length(unordered)[0], 1)

    def test_log_fit_and_stage_i_selection_use_all_probes(self) -> None:
        alpha = -2.0
        slope = 0.3
        confidence = {length: alpha + slope * math.log(length) for length in PROBE_GRID}
        confidence[8] += 0.25
        selected, fit, adjusted = select_initial_length(confidence)
        self.assertEqual(selected, 8)
        expected_fit = fit_log_length_confidence(PROBE_GRID, [confidence[length] for length in PROBE_GRID])
        self.assertAlmostEqual(fit.slope, expected_fit.slope)
        self.assertEqual(set(adjusted), set(PROBE_GRID))

    def test_stage_i_tie_breaks_shorter(self) -> None:
        confidence = {length: -1.0 + 0.2 * math.log(length) for length in PROBE_GRID}
        selected, fit, adjusted = select_initial_length(confidence)
        self.assertEqual(selected, 1)
        self.assertAlmostEqual(adjusted_confidence(confidence[128], 128, fit.slope), adjusted[1])

    def test_local_boundary_and_strict_improvement(self) -> None:
        self.assertEqual(local_neighbors(1), (1, 2))
        self.assertEqual(local_neighbors(MAX_GEN), (MAX_GEN - 1, MAX_GEN))
        self.assertEqual(choose_local_length(1, {1: -1.0, 2: -0.5}, slope=0.0), 2)
        self.assertEqual(choose_local_length(1, {1: -1.0, 2: -1.0}, slope=0.0), 1)

    def test_local_search_exercises_expansion_and_contraction(self) -> None:
        expansion = run_local_search(3, lambda length: -abs(length - 5), slope=0.0)
        contraction = run_local_search(5, lambda length: -abs(length - 2), slope=0.0)
        self.assertEqual(expansion.selected_length, 5)
        self.assertEqual(expansion.moves, (4, 5))
        self.assertEqual(contraction.selected_length, 2)
        self.assertEqual(contraction.moves, (4, 3, 2))
        self.assertGreater(expansion.cache_hits, 0)
        self.assertEqual(expansion.termination_reason, "strict_local_optimum")

    def test_local_search_guard_fails_instead_of_looping(self) -> None:
        with self.assertRaises(LocalSearchLimitError):
            run_local_search(1, lambda length: float(length), slope=0.0, max_moves=1)

    def test_top_p_sampling_excludes_special_tokens_and_is_seeded(self) -> None:
        logits = [10.0, 9.0, 8.0]
        first = sample_top_p(logits, temperature=0.2, top_p=0.9, rng=random.Random(7), forbidden_token_ids={0})
        second = sample_top_p(logits, temperature=0.2, top_p=0.9, rng=random.Random(7), forbidden_token_ids={0})
        self.assertEqual(first, 1)
        self.assertEqual(first, second)
        self.assertEqual(derive_row_seed(42, "row=1"), derive_row_seed(42, "row=1"))
        self.assertNotEqual(derive_row_seed(42, "row=1"), derive_row_seed(42, "row=2"))

    def test_left_token_commit_decrements_remaining_and_enforces_max_gen(self) -> None:
        state = commit_left_token(CommitState((), 2), 11)
        state = commit_left_token(state, 12)
        self.assertEqual(state, CommitState((11, 12), 0))
        with self.assertRaises(ValueError):
            commit_left_token(state, 13)
        with self.assertRaises(ValueError):
            commit_left_token(CommitState(tuple(range(MAX_GEN)), 1), 13)

    def test_forward_token_ledger_separates_search_decode_and_cache(self) -> None:
        ledger = ForwardTokenLedger()
        ledger.record_forward("stage_i", input_tokens=10)
        ledger.record_forward("stage_ii", input_tokens=12)
        ledger.record_forward("commit", input_tokens=12)
        ledger.record_cache_hit("stage_ii")
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["search_forward_calls"], 2)
        self.assertEqual(snapshot["decode_forward_calls"], 1)
        self.assertEqual(snapshot["total_forward_calls"], 3)
        self.assertEqual(snapshot["total_token_forwards"], 34)
        self.assertEqual(snapshot["cache_hits_by_stage"], {"stage_ii": 1})

    def test_resume_dedup_and_forbidden_input_guards(self) -> None:
        rows = [{"candidate_key": "row=1", "status": "ok"}]
        self.assertEqual(set(canonical_success_index(rows, {"row=1", "row=2"})), {"row=1"})
        with self.assertRaises(ValueError):
            canonical_success_index(rows * 2, {"row=1"})
        assert_selection_payload_safe({"task_id": "HumanEval/1", "prefix": "x", "suffix": "y"})
        with self.assertRaises(ValueError):
            assert_selection_payload_safe({"metadata": {"canonical_solution": "leak"}})


if __name__ == "__main__":
    unittest.main()
