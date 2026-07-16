from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.m3_birth_death_canvas import METHODS, run_analysis


def row(row_key: str, group: str, passed: bool, token_budget: int) -> dict[str, object]:
    return {
        "row_key": row_key,
        "task_group": group,
        "length_bucket": "medium",
        "status": "ok",
        "passed": passed,
        "metrics": {
            "actual_forward_count": 256,
            "actual_token_forward_budget": token_budget,
            "wall_sec": 1.0,
        },
    }


class M3BirthDeathCanvasAnalysisTest(unittest.TestCase):
    def test_grouped_analyzer_uses_requested_cluster_bootstrap_budget(self) -> None:
        uniform = [row("a", "g0", False, 15360), row("b", "g1", True, 15360)]
        birth = [row("a", "g0", True, 16000), row("b", "g1", True, 17000)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uniform_raw, birth_raw = root / "uniform.jsonl", root / "birth.jsonl"
            uniform_raw.write_text("".join(json.dumps(item) + "\n" for item in uniform), encoding="utf-8")
            birth_raw.write_text("".join(json.dumps(item) + "\n" for item in birth), encoding="utf-8")
            summary = run_analysis(uniform_raw, birth_raw, root / "out", bootstrap_replicates=20)
        self.assertEqual(summary["bootstrap_replicates"], 20)
        self.assertEqual(summary["compute_claim"], "equal_forward_only; actual token-forward totals are descriptive and may differ")
        self.assertEqual(summary["paired_effects"][0]["task_group_count"], 2)


if __name__ == "__main__":
    unittest.main()
