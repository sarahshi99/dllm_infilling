from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from experiments.p1_official_cal_adapter import (
    ARMS,
    ForwardLedgerModel,
    arm_config,
    audit_rows,
    canonical_success_rows,
    candidate_key,
    expected_keys,
    progress_payload,
    atomic_write_json,
    enable_pinned_evaluator_source,
)


class OfficialCalAdapterTest(unittest.TestCase):
    def test_arm_contracts_keep_official_and_internal_controls_distinct(self) -> None:
        self.assertEqual(set(ARMS), {"official_cal_primary", "official_fixed32", "project_fixed64_internal"})
        self.assertTrue(arm_config("official_cal_primary")["use_bias"])
        self.assertEqual(arm_config("official_fixed32")["initial_gen_length"], 32)
        self.assertIn("not_same_compute", arm_config("project_fixed64_internal")["comparison_boundary"])

    def test_resume_audit_checks_exact_keys_and_forward_partition(self) -> None:
        manifest = [{"source_row_id": 3}]
        key = candidate_key(3, "official_cal_primary")
        rows = [{"candidate_key": key, "arm": "official_cal_primary", "status": "ok", "metrics": {"search_forwards": 5, "formal_decode_forwards": 32, "total_forwards": 37}}]
        self.assertTrue(audit_rows(rows, expected_keys(manifest, "official_cal_primary"), "official_cal_primary")["passed"])
        self.assertFalse(audit_rows(rows, expected_keys(manifest, "official_fixed32"), "official_fixed32")["passed"])

    def test_canonical_raw_is_success_only_and_progress_is_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            raw.write_text(json.dumps({"candidate_key": "a", "status": "ok"}) + "\n", encoding="utf-8")
            self.assertEqual(len(canonical_success_rows(raw)), 1)
            raw.write_text(json.dumps({"candidate_key": "a", "status": "error"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "status=ok"):
                canonical_success_rows(raw)
            manifest = root / "progress.json"
            atomic_write_json(manifest, progress_payload(arm="official_cal_primary", mode="smoke", expected_count=12, completed_count=3, starting_completed_count=0, failure_journal_count=0, started=0.0))
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["completed_count"], 3)

    def test_pinned_evaluator_enablement_is_one_documented_line(self) -> None:
        source = "before\n#                     exec(check_program, exec_globals)\nafter\n"
        enabled = enable_pinned_evaluator_source(source)
        self.assertEqual(enabled.count("exec(check_program, exec_globals)"), 1)
        self.assertNotIn("#                     exec", enabled)
        with self.assertRaisesRegex(RuntimeError, "absent or ambiguous"):
            enable_pinned_evaluator_source("no marker")


if __name__ == "__main__":
    unittest.main()
