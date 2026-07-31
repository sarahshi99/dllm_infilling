from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.build_dreamon_multiline_smoke_manifest import select_smoke, validate_full, write_immutable


def row(index: int, group: str) -> dict[str, object]:
    return {
        "candidate_key": f"k{index}",
        "dataset": "HumanEval-MultiLineInfilling",
        "source_row_id": index,
        "task_id": f"MultiLineInfilling/{group}/L{index}",
        "task_group": group,
        "safe_row_sha256": f"sha{index}",
    }


class DreamOnMultiLineSmokeManifestTest(unittest.TestCase):
    def test_selects_one_row_from_twelve_distinct_groups(self) -> None:
        rows = [row(i, f"HumanEval/{i // 2}") for i in range(30)]
        selected = select_smoke(rows)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({item["task_group"] for item in selected}), 12)

    def test_full_validation_rejects_forbidden_fields(self) -> None:
        rows = [row(i, f"HumanEval/{i % 148}") for i in range(5079)]
        rows[0]["passed"] = True
        with self.assertRaises(RuntimeError):
            validate_full(rows)

    def test_immutable_write_rejects_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            write_immutable(path, (json.dumps(row(0, "HumanEval/0")) + "\n").encode())
            write_immutable(path, path.read_bytes())
            with self.assertRaises(FileExistsError):
                write_immutable(path, b"different\n")


if __name__ == "__main__":
    unittest.main()
