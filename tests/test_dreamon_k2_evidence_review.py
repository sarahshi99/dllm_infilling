from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from analysis.dreamon_k2_evidence import FIELDS, anonymize_gpu, source_rows, validate_rows
from analysis.review_dreamon_markov_k2_diagnosis import compare_table, paired_statistics


def recorded_case():
    row = {name: 0 for name in FIELDS}
    row.update(policy="global_confidence", arm="A", task_id="SingleLineInfilling/HumanEval/0/L0",
               base_task_id="0", line_id="L0", seed=42, row_seed=1, code_commit="fixture",
               output="x", output_chars=1, completion_token_ids=[7], output_tokens=1,
               passed=True, failure_reason="passed", generation_sec=.1, evaluator_sec=.2,
               file_write_sec=.01, forward_count=2, base_forward_count=1,
               recompute_forward_count=1, outer_round_count=1)
    return row


class EvidenceReviewTest(unittest.TestCase):
    def test_primary_keys_and_original_fields_are_required(self):
        row = recorded_case()
        validate_rows([row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows([row, copy.deepcopy(row)])
        missing = dict(row)
        del missing["failure_reason"]
        with self.assertRaisesRegex(ValueError, "missing recorded fields"):
            validate_rows([missing])

    def test_count_disagreement_is_not_repaired_from_other_columns(self):
        row = recorded_case()
        row["forward_count"] = 1
        with self.assertRaisesRegex(ValueError, "forward accounting"):
            validate_rows([row])
        row = recorded_case()
        row["output_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "output length"):
            validate_rows([row])

    def test_sqlite_and_jsonl_must_agree_without_writing_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = recorded_case()
            path = root / "per_case.sqlite"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE cases (policy TEXT, arm TEXT, task_id TEXT, payload TEXT)")
                db.execute("INSERT INTO cases VALUES (?,?,?,?)", (row["policy"], row["arm"], row["task_id"], json.dumps(row)))
            before = path.read_bytes()
            (root / "per_case.jsonl").write_text(json.dumps(row) + "\n")
            rows, provenance = source_rows(root)
            self.assertEqual(rows, [row])
            self.assertTrue(provenance["sqlite_jsonl_all_fields_equal"])
            self.assertEqual(before, path.read_bytes())
            row["passed"] = False
            (root / "per_case.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "disagree"):
                source_rows(root)
            self.assertEqual(before, path.read_bytes())

    def test_gpu_anonymization_preserves_load_and_removes_identity(self):
        snapshot = dict(time_utc="2026-10-13T01:00:00Z",
                        gpus="0, private-uuid, NVIDIA H200 NVL, 90000, 143771, 75, 40",
                        compute_processes="123456, /home/private-person/python, 10000\n654321, /private/own/python, 20000")
        result = anonymize_gpu(snapshot, own_pid=654321)
        self.assertEqual(result["other_task_count"], 1)
        self.assertEqual(result["own_task_count"], 1)
        self.assertEqual(result["other_memory_mib"], 10000)
        self.assertEqual(result["utilization_gpu_pct"], 75)
        encoded = json.dumps(result)
        for secret in ("private-person", "123456", "654321", "private-uuid", "/home", "/private"):
            self.assertNotIn(secret, encoded)

    def test_redaction_keeps_private_original_and_is_idempotent(self):
        from analysis.dreamon_k2_evidence import redact_manifest_gpu_snapshots
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = dict(code_commit="unchanged", initial_gpu_snapshot=dict(
                time_utc="2026-10-13T01:00:00Z", gpus="0, uuid, NVIDIA H200 NVL, 100, 1000, 1, 35",
                compute_processes="123456, /home/private-user/python, 100"))
            path = root / "run_manifest.json"
            path.write_text(json.dumps(original))
            self.assertEqual(redact_manifest_gpu_snapshots(root), ["run_manifest.json"])
            saved = json.loads(path.read_text())
            self.assertEqual(saved["code_commit"], "unchanged")
            self.assertEqual(saved["initial_gpu_snapshot"]["other_task_count"], 1)
            self.assertEqual(json.loads((root / ".private_original_manifests/run_manifest.json").read_text()), original)
            self.assertEqual(redact_manifest_gpu_snapshots(root), [])

    def test_cluster_weighting_not_independent_line_resampling(self):
        rows = []
        for group, deltas in (("small", [1]), ("large", [-1, -1, -1])):
            for index, delta in enumerate(deltas):
                for arm, passed in (("D", delta > 0), ("C", delta < 0)):
                    rows.append(dict(arm=arm, task_id=f"{group}/{index}", base_task_id=group, passed=passed))
        result = paired_statistics(rows, "D", "C", replicates=1000)
        self.assertEqual((result["help"], result["harm"], result["net"]), (1, 3, -2))
        self.assertEqual(result["line_delta"], -.5)
        self.assertEqual(result["base_task_equal_delta"], 0)
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result, paired_statistics(rows, "D", "C", replicates=1000))

    def test_comparison_uses_unrounded_ci_sign_and_exact_counts(self):
        for old, new in ((0.0, 1e-16), (-1e-16, 1e-16)):
            result = compare_table([dict(key="x", line_ci95_low=old)],
                                   [dict(key="x", line_ci95_low=new)], ("key",), "fixture")
            self.assertEqual(result[1]["status"], "DIFFERENT")
        result = compare_table([dict(key="x", passed=901)],
                               [dict(key="x", passed=902)], ("key",), "fixture")
        self.assertEqual(result[1]["status"], "DIFFERENT")


if __name__ == "__main__":
    unittest.main()
