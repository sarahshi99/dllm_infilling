from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiments.build_dreamon_markov_transition_bank import ReservoirBank


def row(sample_key: str, problem: str, *, stage: str = "early") -> dict:
    return {
        "split": "train",
        "trajectory_policy": "global_confidence",
        "generation_stage": stage,
        "sample_key": sample_key,
        "problem_group_id": problem,
        "record_id": problem,
    }


class ReservoirBankResumeTest(unittest.TestCase):
    def _snapshot(self, bank: ReservoirBank) -> tuple[list[tuple], list[tuple], list[tuple]]:
        return (
            list(bank.connection.execute(
                "SELECT split,policy,stage,slot,sample_key,problem_group_id,record_id "
                "FROM samples ORDER BY split,policy,stage,slot"
            )),
            list(bank.connection.execute(
                "SELECT split,policy,stage,seen FROM counters ORDER BY split,policy,stage"
            )),
            list(bank.connection.execute(
                "SELECT split,next_record_index FROM progress ORDER BY split"
            )),
        )

    def test_checkpoint_rolls_back_partial_next_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.sqlite"
            bank = ReservoirBank(path, per_problem_cap=16, oversample=1.0)
            self.assertTrue(bank.add(row("sample-0", "problem-0"), capacity=10))
            bank.checkpoint("train", 1)
            self.assertTrue(bank.add(row("sample-1", "problem-1"), capacity=10))
            bank.connection.rollback()
            bank.connection.close()

            resumed = ReservoirBank(path, per_problem_cap=16, oversample=1.0)
            self.assertEqual(resumed.next_record_index("train"), 1)
            self.assertEqual(
                resumed.connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 1
            )
            self.assertEqual(
                resumed.connection.execute(
                    "SELECT seen FROM counters WHERE split='train' AND policy='global_confidence' AND stage='early'"
                ).fetchone()[0],
                1,
            )
            resumed.close()

    def test_interrupted_resume_matches_continuous_bank(self) -> None:
        rows = [row(f"sample-{index}", f"problem-{index}") for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            continuous = ReservoirBank(
                Path(directory) / "continuous.sqlite", per_problem_cap=16, oversample=1.0
            )
            for index, item in enumerate(rows):
                continuous.add(item, capacity=4)
                continuous.checkpoint("train", index + 1)
            continuous_snapshot = self._snapshot(continuous)
            continuous.close()

            path = Path(directory) / "resumed.sqlite"
            interrupted = ReservoirBank(path, per_problem_cap=16, oversample=1.0)
            for index, item in enumerate(rows[:3]):
                interrupted.add(item, capacity=4)
                interrupted.checkpoint("train", index + 1)
            interrupted.add(rows[3], capacity=4)
            interrupted.connection.rollback()
            interrupted.connection.close()

            resumed = ReservoirBank(path, per_problem_cap=16, oversample=1.0)
            self.assertEqual(resumed.next_record_index("train"), 3)
            for index, item in enumerate(rows[3:], start=3):
                resumed.add(item, capacity=4)
                resumed.checkpoint("train", index + 1)
            self.assertEqual(self._snapshot(resumed), continuous_snapshot)
            resumed.close()

    def test_rejects_legacy_partial_bank_without_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.sqlite"
            bank = ReservoirBank(path, per_problem_cap=16, oversample=1.0)
            bank.add(row("sample-0", "problem-0"), capacity=10)
            bank.connection.commit()
            bank.connection.close()
            connection = sqlite3.connect(path)
            connection.execute("DELETE FROM progress")
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(RuntimeError, "no exact-resume progress"):
                ReservoirBank(path, per_problem_cap=16, oversample=1.0)


if __name__ == "__main__":
    unittest.main()
