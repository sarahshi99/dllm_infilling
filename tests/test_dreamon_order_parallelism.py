from __future__ import annotations

import unittest

import torch

from experiments.dreamon_singleline_order_parallel import (
    Proposal,
    apply_left_frontier_barrier,
    left_frontier_positions,
    select_positions,
    shift_logits_to_targets,
    validate_resume_rows,
)


class DreamOnOrderParallelismTest(unittest.TestCase):
    def test_confidence_global_selects_nonadjacent_topk(self) -> None:
        positions = [4, 7, 8, 12, 18]
        confidence = [0.10, 0.95, 0.20, 0.90, 0.80]
        self.assertEqual(
            select_positions(positions, confidence, requested_k=2, policy="confidence_global"),
            [7, 12],
        )
        self.assertEqual(
            select_positions(positions, confidence, requested_k=4, policy="confidence_global"),
            [7, 12, 18, 8],
        )

    def test_left_frontier_only_uses_contiguous_masks(self) -> None:
        positions = [4, 5, 6, 10, 11]
        confidence = [0.10, 0.95, 0.20, 0.90, 0.80]
        self.assertEqual(left_frontier_positions(positions, requested_k=4), [4, 5, 6])
        self.assertEqual(
            select_positions(positions, confidence, requested_k=4, policy="left_to_right_frontier"),
            [4, 5, 6],
        )

    def test_requested_k_exceeding_active_masks_is_safe(self) -> None:
        self.assertEqual(
            select_positions([3], [0.8], requested_k=4, policy="confidence_global"), [3]
        )
        self.assertEqual(left_frontier_positions([], requested_k=4), [])

    def test_frontier_does_not_cross_a_gap(self) -> None:
        self.assertEqual(left_frontier_positions([9, 10, 13, 14], requested_k=4), [9, 10])

    def test_shift_logits_maps_prediction_to_its_target_position(self) -> None:
        logits = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
        self.assertEqual(
            shift_logits_to_targets(logits).squeeze(-1).tolist(), [[10.0, 10.0, 20.0, 30.0]]
        )

    def test_left_frontier_barrier_executes_only_the_first_structural_action(self) -> None:
        proposals = [
            Proposal(position=4, token_id=101, token_type="normal"),
            Proposal(position=5, token_id=102, token_type="expand"),
            Proposal(position=6, token_id=103, token_type="normal"),
            Proposal(position=7, token_id=104, token_type="delete"),
        ]
        committed, action, discarded = apply_left_frontier_barrier(proposals)
        self.assertEqual([proposal.position for proposal in committed], [4])
        self.assertEqual(action.position if action else None, 5)
        self.assertEqual(discarded, 2)

    def test_resume_rejects_duplicate_or_wrong_variant(self) -> None:
        rows = [{"case_key": "a", "variant": "C1", "seed": 42}]
        self.assertEqual(validate_resume_rows(rows, variant="C1", seed=42), {"a"})
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            validate_resume_rows(rows * 2, variant="C1", seed=42)
        with self.assertRaisesRegex(RuntimeError, "variant/seed"):
            validate_resume_rows(rows, variant="L1", seed=42)


if __name__ == "__main__":
    unittest.main()
