import gzip
import json
import tempfile
import unittest
from pathlib import Path

from experiments.p1_official_cal_source_audit import choose_smoke, task_group


class OfficialCalSourceAuditTest(unittest.TestCase):
    def test_task_group_and_smoke_manifest_are_deterministic(self) -> None:
        rows = [{"task_id": f"MultiLineInfilling/HumanEval/{index}/L0"} for index in range(12)]
        smoke = choose_smoke(rows, 4)
        self.assertEqual([row["source_row_id"] for row in smoke], [0, 4, 7, 11])
        self.assertEqual(task_group(smoke[0]["task_id"]), "HumanEval/0")
        self.assertEqual(len({row["task_id_sha256"] for row in smoke}), 4)

    def test_smoke_rejects_invalid_count(self) -> None:
        with self.assertRaises(ValueError):
            choose_smoke([{"task_id": "MultiLineInfilling/HumanEval/0/L0"}], 2)


if __name__ == "__main__":
    unittest.main()
