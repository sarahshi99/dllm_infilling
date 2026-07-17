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
        self.assertIn("safely_paused", methods["M1"]["current_status"])
        self.assertEqual(methods["M2"]["current_status"], "reviewed_not_promoted")
        self.assertIn("outcome_analysis_pending", methods["M3"]["current_status"])
        self.assertEqual(methods["M4"]["current_status"], "offline_148_complete_gpu_repair_pending")

    def test_subordinate_current_docs_point_to_the_register(self) -> None:
        marker = "method_portfolio.current.json"
        for name in ("ccfa_master_roadmap.zh.md", "current_action.md", "experiment_queue.md", "idea_board.md"):
            self.assertIn(marker, (DOCS / name).read_text(encoding="utf-8"), name)


if __name__ == "__main__":
    unittest.main()
