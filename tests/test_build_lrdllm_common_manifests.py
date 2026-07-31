from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis.build_lrdllm_common_manifests import (
    jsonl_bytes,
    manifest_rows,
    mechanism_rows,
    one_per_group,
    row_fingerprint,
    task_group,
    write_immutable,
)


class BuildLrDllmCommonManifestsTest(unittest.TestCase):
    def safe_row(self, source_row_id: int, group: int, context_size: int = 1) -> dict:
        return {
            "source_row_id": source_row_id,
            "task_id": f"SingleLineInfilling/HumanEval/{group}/L0",
            "task_group": f"HumanEval/{group}",
            "prompt": "p" * context_size,
            "suffix": "s",
            "entry_point": f"f{group}",
        }

    def test_task_group_and_fingerprint_use_safe_fields_only(self) -> None:
        row = self.safe_row(0, 7)
        self.assertEqual(task_group(row["task_id"]), "HumanEval/7")
        self.assertEqual(row_fingerprint(row), row_fingerprint({**row, "canonical_solution": "ignored", "test": "ignored"}))

    def test_manifest_candidate_keys_are_arm_independent(self) -> None:
        rows = [self.safe_row(0, 0), self.safe_row(1, 1), self.safe_row(2, 2)]
        manifest = manifest_rows("HumanEval-SingleLineInfilling", rows, {"HumanEval/0", "HumanEval/2"})
        self.assertEqual(len(manifest), 2)
        self.assertNotIn("arm=", manifest[0]["candidate_key"])
        self.assertNotIn("prompt", manifest[0])

    def test_smoke_and_mechanism_selection_are_cluster_unique(self) -> None:
        rows = [self.safe_row(index, index, context_size=index + 1) for index in range(80)]
        manifest = manifest_rows("HumanEval-SingleLineInfilling", rows, {f"HumanEval/{index}" for index in range(80)})
        self.assertEqual(len({row["task_group"] for row in one_per_group(manifest, 12)}), 12)
        mechanism = mechanism_rows(manifest, {index: row for index, row in enumerate(rows)})
        self.assertEqual(len(mechanism), 64)
        self.assertEqual(len({row["task_group"] for row in mechanism}), 64)
        self.assertEqual({row["context_length_bucket"] for row in mechanism}, {0, 1, 2, 3})

    def test_immutable_writer_allows_noop_and_rejects_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            content = jsonl_bytes([{"candidate_key": "a"}])
            write_immutable(path, content)
            write_immutable(path, content)
            with self.assertRaises(FileExistsError):
                write_immutable(path, jsonl_bytes([{"candidate_key": "b"}]))


if __name__ == "__main__":
    unittest.main()
