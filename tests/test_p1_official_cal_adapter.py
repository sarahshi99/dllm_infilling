from __future__ import annotations

import unittest

from experiments.p1_official_cal_adapter import (
    ARMS,
    ForwardLedgerModel,
    arm_config,
    audit_rows,
    candidate_key,
    expected_keys,
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


if __name__ == "__main__":
    unittest.main()
