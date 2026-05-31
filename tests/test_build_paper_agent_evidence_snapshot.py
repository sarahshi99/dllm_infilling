from __future__ import annotations

import unittest

from analysis import build_paper_agent_evidence_snapshot as snapshot


def row(task_id: str, *, passed: bool, oracle: int, selected: int, source: str = "base") -> dict:
    return {
        "task_id": task_id,
        "metrics": {
            "passed": passed,
            "oracle_mask_length": oracle,
            "selected_mask_length": selected,
            "final_source": source,
        },
    }


class BuildPaperAgentEvidenceSnapshotTest(unittest.TestCase):
    def test_summarize_run_reports_pass_rate_and_oracle_buckets(self) -> None:
        rows = [
            row("a", passed=True, oracle=6, selected=6),
            row("b", passed=False, oracle=14, selected=8),
            row("c", passed=True, oracle=20, selected=20),
        ]

        result = snapshot.summarize_run("candidate", "/tmp/run", rows)

        self.assertEqual(result["name"], "candidate")
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["pass_count"], 2)
        self.assertAlmostEqual(result["pass_rate"], 2 / 3)
        self.assertEqual(result["oracle_bucket_counts"], {"<=8": 1, "13-16": 1, "17-24": 1})
        self.assertAlmostEqual(result["oracle_bucket_pass_rates"]["<=8"], 1.0)
        self.assertAlmostEqual(result["oracle_bucket_pass_rates"]["13-16"], 0.0)
        self.assertAlmostEqual(result["oracle_bucket_pass_rates"]["17-24"], 1.0)

    def test_compare_runs_counts_pairwise_wins_losses_and_buckets(self) -> None:
        base_rows = [
            row("win", passed=False, oracle=14, selected=8),
            row("loss", passed=True, oracle=6, selected=6),
            row("tie", passed=True, oracle=20, selected=20),
        ]
        candidate_rows = [
            row("win", passed=True, oracle=14, selected=14, source="official_mid_rescue"),
            row("loss", passed=False, oracle=6, selected=13, source="official_mid_rescue"),
            row("tie", passed=True, oracle=20, selected=20),
        ]

        result = snapshot.compare_runs(base_rows, candidate_rows)

        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["net"], 0)
        self.assertEqual(result["wins_by_bucket"], {"13-16": 1})
        self.assertEqual(result["losses_by_bucket"], {"<=8": 1})
        self.assertEqual(result["win_rows"][0]["task_id"], "win")
        self.assertEqual(result["loss_rows"][0]["candidate_source"], "official_mid_rescue")

    def test_summarize_long_failures_reports_underselection_and_sources(self) -> None:
        rows = [
            row("long-under", passed=False, oracle=20, selected=6, source="base"),
            row("long-not-under", passed=False, oracle=20, selected=20, source="official"),
            row("long-pass", passed=True, oracle=20, selected=6, source="base"),
            row("short-fail", passed=False, oracle=6, selected=3, source="base"),
        ]

        result = snapshot.summarize_long_failures(rows)

        self.assertEqual(result["long_total"], 3)
        self.assertEqual(result["failed_long_total"], 2)
        self.assertEqual(result["underselected_failed_long"], 1)
        self.assertEqual(result["failed_long_base_source"], 1)
        self.assertEqual(result["failed_long_selected_length_histogram"], {"6": 1, "20": 1})


if __name__ == "__main__":
    unittest.main()
