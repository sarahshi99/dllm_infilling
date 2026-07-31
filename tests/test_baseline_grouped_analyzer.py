from __future__ import annotations

import unittest

from analysis.baseline_grouped_analyzer import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    normalize_rows,
    paired_comparison,
    summarize_method,
    validate_manifest,
)


def manifest() -> list[dict[str, object]]:
    return [
        {"candidate_key": "a0", "task_id": "task/a0", "task_group": "A"},
        {"candidate_key": "a1", "task_id": "task/a1", "task_group": "A"},
        {"candidate_key": "b0", "task_id": "task/b0", "task_group": "B"},
    ]


def raw_rows(passes: tuple[bool, bool, bool]) -> list[dict[str, object]]:
    rows = []
    for index, (item, passed) in enumerate(zip(manifest(), passes, strict=True)):
        rows.append(
            {
                "run_key": f"arm-specific-{index}",
                "task_id": item["task_id"],
                "task_group": item["task_group"],
                "status": "ok",
                "passed": passed,
                "metrics": {
                    "search_forwards": 1,
                    "formal_decode_forwards": 2,
                    "total_forwards": 3,
                    "token_budget": 30 + index,
                    "wall_sec": 0.5 + index,
                    "peak_memory_bytes": 100 + index,
                    "selected_length": 8 + index,
                    "expansion_count": int(index == 1),
                    "contraction_count": int(index == 2),
                    "termination_reason": "stable" if index < 2 else "max_length",
                },
            }
        )
    return rows


class BaselineGroupedAnalyzerTest(unittest.TestCase):
    def test_default_bootstrap_is_ten_thousand(self) -> None:
        self.assertEqual(DEFAULT_BOOTSTRAP_REPLICATES, 10_000)

    def test_summary_uses_row_pass_at_one_and_equal_weight_task_macro(self) -> None:
        rows = normalize_rows(raw_rows((True, False, True)), manifest())
        summary = summarize_method(
            rows,
            method="candidate",
            expected_rows=3,
            expected_clusters=2,
            bootstrap_replicates=200,
            seed=7,
        )
        self.assertAlmostEqual(summary["row_pass_at_1"], 2 / 3)
        self.assertAlmostEqual(summary["equal_weight_task_macro_pass_at_1"], 0.75)
        self.assertEqual(summary["cluster_count"], 2)
        self.assertEqual(summary["total_search_forward_calls"], 3)
        self.assertEqual(summary["total_decode_forward_calls"], 6)
        self.assertEqual(summary["total_forward_calls"], 9)
        self.assertEqual(summary["total_token_forwards"], 93)
        self.assertEqual(summary["dynamic"]["expansion_positive_rows"], 1)
        self.assertEqual(summary["dynamic"]["contraction_positive_rows"], 1)
        self.assertEqual(summary["dynamic"]["termination_reason_counts"]["stable"], 2)

    def test_pairing_reports_row_and_cluster_help_harm(self) -> None:
        left = normalize_rows(raw_rows((True, True, False)), manifest())
        right = normalize_rows(raw_rows((False, False, True)), manifest())
        effect = paired_comparison(
            left,
            right,
            method_a="left",
            method_b="right",
            expected_clusters=2,
            bootstrap_replicates=200,
            seed=11,
        )
        self.assertEqual((effect["row_help"], effect["row_harm"]), (2, 1))
        self.assertEqual((effect["cluster_help"], effect["cluster_harm"]), (1, 1))
        self.assertEqual(effect["paired_candidate_key_count"], 3)

    def test_pairing_rejects_different_candidate_keys(self) -> None:
        left = normalize_rows(raw_rows((True, False, True)), manifest())
        short_manifest = manifest()[:-1]
        right = normalize_rows(raw_rows((True, False, True))[:-1], short_manifest)
        with self.assertRaisesRegex(RuntimeError, "different candidate keys"):
            paired_comparison(
                left,
                right,
                method_a="left",
                method_b="right",
                expected_clusters=2,
                bootstrap_replicates=20,
                seed=13,
            )

    def test_forward_accounting_must_balance(self) -> None:
        raw = raw_rows((True, False, True))
        raw[0]["metrics"]["total_forwards"] = 4  # type: ignore[index]
        rows = normalize_rows(raw, manifest())
        with self.assertRaisesRegex(RuntimeError, "forward accounting"):
            summarize_method(
                rows,
                method="candidate",
                expected_rows=3,
                expected_clusters=2,
                bootstrap_replicates=20,
                seed=17,
            )

    def test_manifest_rejects_forbidden_fields(self) -> None:
        invalid = manifest()
        invalid[0]["oracle_length"] = 9
        with self.assertRaisesRegex(RuntimeError, "forbidden fields"):
            validate_manifest(invalid)


if __name__ == "__main__":
    unittest.main()
