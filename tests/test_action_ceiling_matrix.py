from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.action_ceiling.action_ceiling_matrix import (
    ACTION_IDS,
    build_action_manifest,
    build_case_manifest,
    canvas_is_oracle_sufficient,
    compact_result_record,
    summarize_pilot,
    oracle_sufficient_length,
    parse_task_id_group,
    summarize_manifest,
    write_csv,
    write_jsonl,
    write_dry_run_outputs,
)


def row(
    task_id: str,
    *,
    passed: bool,
    oracle: int,
    selected: int,
    triggered: bool = False,
    rescue_len: int | None = None,
) -> dict:
    metrics = {
        "passed": passed,
        "oracle_mask_length": oracle,
        "selected_mask_length": selected,
        "route2_trace_rescue_triggered": triggered,
        "route2_rescue_length": rescue_len,
    }
    return {"task_id": task_id, "metrics": metrics, "verification": {"passed": passed}}


class ActionCeilingManifestTest(unittest.TestCase):
    def test_oracle_sufficient_length_rejects_oracle_above_max_length(self) -> None:
        self.assertEqual(
            oracle_sufficient_length(primary_len=3, route2_len=32, oracle_len=40, max_length=64),
            40,
        )
        self.assertIsNone(
            oracle_sufficient_length(primary_len=3, route2_len=32, oracle_len=80, max_length=64),
        )
        self.assertIsNone(
            oracle_sufficient_length(primary_len=None, route2_len=None, oracle_len=None, max_length=64)
        )
        self.assertTrue(canvas_is_oracle_sufficient(40, 40))
        self.assertFalse(canvas_is_oracle_sufficient(32, 40))

    def test_parse_task_id_group_extracts_humaneval_id(self) -> None:
        self.assertEqual(parse_task_id_group("SingleLineInfilling/HumanEval/116/L0"), "HumanEval/116")
        self.assertEqual(parse_task_id_group("custom/task/location"), "custom/task")

    def test_build_case_manifest_selects_pre_registered_pools(self) -> None:
        baseline = [
            row("SingleLineInfilling/HumanEval/1/L0", passed=False, oracle=21, selected=3),
            row("SingleLineInfilling/HumanEval/2/L0", passed=False, oracle=25, selected=3),
            row("SingleLineInfilling/HumanEval/3/L0", passed=False, oracle=20, selected=16),
            row("SingleLineInfilling/HumanEval/4/L0", passed=True, oracle=8, selected=8),
        ]
        route2 = [
            row("SingleLineInfilling/HumanEval/1/L0", passed=True, oracle=21, selected=32, triggered=True, rescue_len=32),
            row("SingleLineInfilling/HumanEval/2/L0", passed=False, oracle=25, selected=32, triggered=True, rescue_len=32),
            row("SingleLineInfilling/HumanEval/3/L0", passed=False, oracle=20, selected=16, triggered=False),
            row("SingleLineInfilling/HumanEval/4/L0", passed=True, oracle=8, selected=8, triggered=False),
        ]

        manifest = build_case_manifest(baseline, route2, max_cases_per_pool=3, task_ids=None, max_canvas_length=64)
        pools = {item["task_id"]: item["case_pool"] for item in manifest}

        self.assertEqual(pools["SingleLineInfilling/HumanEval/1/L0"], "positive_control_rescued")
        self.assertEqual(pools["SingleLineInfilling/HumanEval/2/L0"], "triggered_failed_long")
        self.assertEqual(pools["SingleLineInfilling/HumanEval/3/L0"], "missed_failed_long")
        self.assertNotIn("SingleLineInfilling/HumanEval/4/L0", pools)
        self.assertEqual(
            next(item for item in manifest if item["task_id"].endswith("2/L0"))["oracle_sufficient_length"],
            32,
        )
        self.assertTrue(
            next(item for item in manifest if item["task_id"].endswith("2/L0"))["canvas_is_oracle_sufficient"]
        )

    def test_case_manifest_marks_oracle_above_max_canvas(self) -> None:
        baseline = [row("too-long", passed=False, oracle=80, selected=3)]
        route2 = [row("too-long", passed=False, oracle=80, selected=32, triggered=True, rescue_len=32)]

        manifest = build_case_manifest(
            baseline,
            route2,
            max_cases_per_pool=1,
            task_ids=["too-long"],
            max_canvas_length=64,
        )

        self.assertIsNone(manifest[0]["oracle_sufficient_length"])
        self.assertTrue(manifest[0]["oracle_exceeds_max_canvas"])
        self.assertIsNone(manifest[0]["canvas_is_oracle_sufficient"])

    def test_automatic_selection_prioritizes_true_long_positive_controls(self) -> None:
        baseline = [
            row("short-a", passed=False, oracle=8, selected=3),
            row("long-a", passed=False, oracle=21, selected=3),
            row("short-b", passed=False, oracle=7, selected=3),
        ]
        route2 = [
            row("short-a", passed=True, oracle=8, selected=32, triggered=True, rescue_len=32),
            row("long-a", passed=True, oracle=21, selected=32, triggered=True, rescue_len=32),
            row("short-b", passed=True, oracle=7, selected=32, triggered=True, rescue_len=32),
        ]

        manifest = build_case_manifest(baseline, route2, max_cases_per_pool=1, task_ids=None, max_canvas_length=64)

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["task_id"], "long-a")

    def test_manual_requested_keeps_non_pool_cases(self) -> None:
        baseline = [row("a", passed=True, oracle=8, selected=8), row("b", passed=True, oracle=8, selected=8)]
        route2 = [row("a", passed=True, oracle=8, selected=8), row("b", passed=True, oracle=8, selected=8)]

        manifest = build_case_manifest(baseline, route2, max_cases_per_pool=1, task_ids=["b", "a"], max_canvas_length=64)

        self.assertEqual([item["task_id"] for item in manifest], ["b", "a"])
        self.assertEqual(manifest[0]["case_pool"], "manual_requested")

    def test_action_manifest_has_four_actions_per_case(self) -> None:
        case_manifest = [
            {
                "task_id": "a",
                "case_pool": "triggered_failed_long",
                "oracle_bucket": "25+",
                "primary_selected_length": 3,
                "route2_rescue_length": 32,
                "oracle_sufficient_length": 40,
                "oracle_length": 40,
            }
        ]

        actions = build_action_manifest(case_manifest)

        self.assertEqual([row["action_id"] for row in actions], list(ACTION_IDS))
        self.assertEqual(actions[3]["action_id"], "D_oracle_sufficient_steps96")
        self.assertEqual(actions[2]["planned_canvas_length"], 40)
        self.assertEqual(actions[3]["planned_total_steps"], 96)
        self.assertTrue(actions[2]["canvas_is_oracle_sufficient"])

    def test_summary_and_outputs_are_machine_readable(self) -> None:
        case_manifest = [
            {
                "task_id": "a",
                "case_pool": "triggered_failed_long",
                "oracle_bucket": "25+",
                "route2_triggered": True,
                "route2_rescue_len_ge_oracle": True,
            }
        ]
        action_manifest = build_action_manifest(
            [
                {
                    **case_manifest[0],
                    "primary_selected_length": 3,
                    "route2_rescue_length": 32,
                    "oracle_sufficient_length": 40,
                    "oracle_length": 40,
                }
            ]
        )
        summary = summarize_manifest(case_manifest, action_manifest)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_dry_run_outputs(
                output_dir,
                summary=summary,
                case_manifest=case_manifest,
                action_manifest=action_manifest,
                config={"dry_run": True},
            )

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "case_manifest.csv").exists())
            self.assertTrue((output_dir / "action_manifest.csv").exists())
            self.assertTrue((output_dir / "report.md").exists())
            loaded = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["case_count"], 1)

    def test_summarize_pilot_reports_schedule_ceiling_signal(self) -> None:
        case_manifest = [
            {
                "task_id": "SingleLineInfilling/HumanEval/85/L0",
                "task_group": "HumanEval/85",
                "case_pool": "triggered_failed_long",
            }
        ]
        rows = [
            {
                "task_id": "SingleLineInfilling/HumanEval/85/L0",
                "case_pool": "triggered_failed_long",
                "action_id": "A_primary",
                "experimental_seed": 0,
                "passed": False,
                "replay_matches_historical": True,
            },
            {
                "task_id": "SingleLineInfilling/HumanEval/85/L0",
                "case_pool": "triggered_failed_long",
                "action_id": "B_route2_len32",
                "experimental_seed": 0,
                "passed": False,
                "replay_matches_historical": True,
            },
            {
                "task_id": "SingleLineInfilling/HumanEval/85/L0",
                "case_pool": "triggered_failed_long",
                "action_id": "C_oracle_sufficient",
                "experimental_seed": 0,
                "passed": False,
                "replay_matches_historical": None,
            },
            {
                "task_id": "SingleLineInfilling/HumanEval/85/L0",
                "case_pool": "triggered_failed_long",
                "action_id": "D_oracle_sufficient_steps96",
                "experimental_seed": 0,
                "passed": True,
                "replay_matches_historical": None,
            },
        ]

        summary = summarize_pilot(
            rows,
            case_manifest=case_manifest,
            determinism_check={"deterministic": True},
            experimental_seeds=[0],
        )

        self.assertEqual(summary["verdict"], "schedule_ceiling_signal")
        self.assertEqual(
            summary["steps96_effects"],
            [{"task_id": "SingleLineInfilling/HumanEval/85/L0", "experimental_seed": 0}],
        )

    def test_jsonl_keeps_diagnostics_but_csv_record_is_compact(self) -> None:
        row_data = {
            "task_id": "a",
            "action_id": "A_primary",
            "experimental_seed": 0,
            "passed": False,
            "generated_text": "completion",
            "verification": {"tier3_unit_tests": {"passed": False}},
            "diagnostics": {"decoded_middle_text": "completion"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_jsonl(output_dir / "pilot_results.jsonl", [row_data])
            write_csv(output_dir / "pilot_results.csv", [compact_result_record(row_data)])

            jsonl_row = json.loads((output_dir / "pilot_results.jsonl").read_text(encoding="utf-8"))
            csv_text = (output_dir / "pilot_results.csv").read_text(encoding="utf-8")

        self.assertIn("generated_text", jsonl_row)
        self.assertNotIn("generated_text", csv_text.splitlines()[0])
        self.assertNotIn("verification", csv_text.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
