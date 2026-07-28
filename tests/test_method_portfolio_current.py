from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "paper_agent"


class MethodPortfolioCurrentTest(unittest.TestCase):
    def test_current_register_has_all_methods_and_fixed_statuses(self) -> None:
        payload = json.loads((DOCS / "method_portfolio.current.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["paper_primary_method"], None)
        self.assertEqual(payload["frozen_controller_test"], {"status": "sealed", "test_evaluation_count": 0})
        methods = payload["methods"]
        self.assertEqual(set(methods), {"M1", "M2", "M3", "M4"})
        self.assertIn("reviewed_not_promoted", methods["M1"]["current_status"])
        self.assertIn("safely_paused", methods["M1"]["current_status"])
        self.assertEqual(methods["M2"]["current_status"], "reviewed_not_promoted")
        self.assertEqual(methods["M3"]["current_status"], "reviewed_not_promoted_v0")
        self.assertEqual(methods["M4"]["current_status"], "multiline_core_148_manifest_frozen_repair_pending_launch_authorized")
        self.assertTrue(all("route" in method and "927" in method["route"] for method in methods.values()))

    def test_subordinate_current_docs_point_to_the_register(self) -> None:
        marker = "method_portfolio.current.json"
        for name in ("ccfa_master_roadmap.zh.md", "current_action.md", "experiment_queue.md", "idea_board.md"):
            self.assertIn(marker, (DOCS / name).read_text(encoding="utf-8"), name)

    def test_current_docs_carry_the_same_normalized_status_anchor(self) -> None:
        expected = (
            "M1=reviewed_not_promoted_v0; M2=reviewed_not_promoted; "
            "M3=reviewed_not_promoted_v0; "
            "M4=multiline_core_148_manifest_frozen_repair_pending_launch_authorized"
        )
        for name in ("ccfa_master_roadmap.zh.md", "current_action.md", "experiment_queue.md", "idea_board.md"):
            self.assertIn(expected, (DOCS / name).read_text(encoding="utf-8"), name)


if __name__ == "__main__":
    unittest.main()
