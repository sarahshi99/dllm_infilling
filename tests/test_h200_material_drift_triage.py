from __future__ import annotations

import unittest

from experiments.action_ceiling.h200_material_drift_triage import (
    action_bank_flip_rows,
    action_scope_summary,
    core_flip_rows,
    core_summary_rows,
    enrich_action_labels,
)


def core_row(task_id: str, passed: bool, oracle_len: int = 10, selected_len: int = 10, triggered: bool = False) -> dict:
    return {
        "task_id": task_id,
        "code": f"def f():\n    return {int(passed)}\n",
        "metrics": {
            "passed": passed,
            "oracle_mask_length": oracle_len,
            "selected_mask_length": selected_len,
            "official_repair_triggered": triggered,
        },
        "verification": {
            "tier1_parse_compile": {"passed": passed, "error_type": None if passed else "AssertionError"}
        },
    }


class H200MaterialDriftTriageTest(unittest.TestCase):
    def test_core_flip_rows_suppresses_test_split(self) -> None:
        old_rows = [
            core_row("train_loss", True),
            core_row("test_loss", True),
            core_row("validation_win", False, oracle_len=18),
        ]
        h200_rows = [
            core_row("train_loss", False, selected_len=12, triggered=True),
            core_row("test_loss", False),
            core_row("validation_win", True, oracle_len=18),
        ]
        splits = {"train_loss": "train", "test_loss": "test", "validation_win": "validation"}
        rows = core_flip_rows("toy", old_rows, h200_rows, splits)
        task_ids = {row["task_id"] for row in rows}
        self.assertEqual(task_ids, {"train_loss", "validation_win"})
        self.assertEqual({row["split"] for row in rows}, {"train", "validation"})
        train_row = next(row for row in rows if row["task_id"] == "train_loss")
        self.assertEqual(train_row["direction"], "h200_loss")
        self.assertEqual(train_row["selected_length_delta_h200_minus_old"], 2)

    def test_core_summary_tracks_public_split_bucket_and_overall(self) -> None:
        old_rows = [
            core_row("a", True, oracle_len=7),
            core_row("b", False, oracle_len=20),
            core_row("c", True, oracle_len=20),
        ]
        h200_rows = [
            core_row("a", True, oracle_len=7),
            core_row("b", True, oracle_len=20, selected_len=24),
            core_row("c", False, oracle_len=20),
        ]
        overall, by_split_bucket = core_summary_rows("toy", old_rows, h200_rows, {"a": "train", "b": "train", "c": "test"})
        self.assertEqual(overall[0]["h200_wins"], 1)
        self.assertEqual(overall[0]["h200_losses"], 1)
        train_bucket = next(row for row in by_split_bucket if row["scope"] == "split:train|bucket:17-24")
        self.assertEqual(train_bucket["h200_wins"], 1)
        self.assertEqual(train_bucket["h200_losses"], 0)

    def test_action_summary_and_flips_track_label_drift(self) -> None:
        old_rows = enrich_action_labels(
            [
                {"task_id": "a", "action": "KEEP_PRIMARY", "split": "validation", "passed": False},
                {"task_id": "a", "action": "EXPAND_16", "split": "validation", "passed": True, "generated_text_sha256": "old"},
                {"task_id": "b", "action": "KEEP_PRIMARY", "split": "validation", "passed": True},
                {"task_id": "b", "action": "EXPAND_16", "split": "validation", "passed": False},
            ]
        )
        h200_rows = enrich_action_labels(
            [
                {"task_id": "a", "action": "KEEP_PRIMARY", "split": "validation", "passed": False},
                {"task_id": "a", "action": "EXPAND_16", "split": "validation", "passed": False, "generated_text_sha256": "new"},
                {"task_id": "b", "action": "KEEP_PRIMARY", "split": "validation", "passed": True},
                {"task_id": "b", "action": "EXPAND_16", "split": "validation", "passed": True},
            ]
        )
        summary = action_scope_summary("toy", old_rows, h200_rows)
        self.assertEqual(summary["h200_wins"], 1)
        self.assertEqual(summary["h200_losses"], 1)
        self.assertEqual(summary["old_benefit_count_non_keep"], 1)
        self.assertEqual(summary["h200_benefit_count_non_keep"], 0)
        self.assertEqual(summary["old_harm_count_non_keep"], 1)
        self.assertEqual(summary["h200_harm_count_non_keep"], 0)

        flips = action_bank_flip_rows(old_rows, h200_rows)
        self.assertEqual(len(flips), 2)
        self.assertEqual({row["action"] for row in flips}, {"EXPAND_16"})


if __name__ == "__main__":
    unittest.main()
