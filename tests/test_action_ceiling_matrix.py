from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.action_ceiling.action_ceiling_matrix import (
    ACTION_IDS,
    build_action_equivalence,
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
from experiments.action_ceiling.distinct_candidate_ceiling import (
    choose_verdict as choose_distinct_candidate_verdict,
    cluster_hashes,
    has_multi_seed_diversity,
    remask_count_for_canvas,
    select_trace_span_remask,
    select_trace_remask_positions,
    smoke_gate,
    span_remask_width_for_canvas,
    stable_task_seed,
    summarize_diversity,
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
        self.assertEqual(summary["pass_level_canvas_effects"], [])
        self.assertEqual(summary["candidate_level_canvas_effects"], [])

    def test_candidate_hash_clustering_and_change_metrics_are_separate(self) -> None:
        rows = [
            {
                "task_id": "t",
                "action_id": "A_primary",
                "generated_text_sha256": "a",
                "passed": "False",
                "compile_passed": "False",
                "error_type": "SyntaxError",
                "decode_steps": "64",
                "effective_steps": "64",
                "stop_reason": "no_remaining_masks",
                "selected_mask_length": "3",
                "actual_canvas_length": "3",
            },
            {
                "task_id": "t",
                "action_id": "B_route2_len32",
                "generated_text_sha256": "b",
                "passed": "False",
                "compile_passed": "False",
                "error_type": "SyntaxError",
                "decode_steps": "64",
                "effective_steps": "41",
                "stop_reason": "global_gap_early_commit",
                "selected_mask_length": "32",
                "actual_canvas_length": "32",
            },
            {
                "task_id": "t",
                "action_id": "C_oracle_sufficient",
                "generated_text_sha256": "b",
                "passed": "False",
                "compile_passed": "False",
                "error_type": "SyntaxError",
                "decode_steps": "64",
                "effective_steps": "41",
                "stop_reason": "global_gap_early_commit",
                "selected_mask_length": "32",
                "actual_canvas_length": "32",
            },
            {
                "task_id": "t",
                "action_id": "D_oracle_sufficient_steps96",
                "generated_text_sha256": "b",
                "passed": "False",
                "compile_passed": "False",
                "error_type": "SyntaxError",
                "decode_steps": "96",
                "effective_steps": "41",
                "stop_reason": "global_gap_early_commit",
                "selected_mask_length": "32",
                "actual_canvas_length": "32",
            },
        ]

        audit = build_action_equivalence(rows)
        info = audit["tasks"]["t"]

        self.assertEqual(info["unique_candidate_hash_count"], 2)
        self.assertEqual(info["candidate_level_canvas_effects"], ["B_route2_len32", "C_oracle_sufficient", "D_oracle_sufficient_steps96"])
        self.assertEqual(info["pass_level_canvas_effects"], [])
        self.assertIn("B_route2_len32", info["output_changed_without_correctness_change"])
        self.assertIn(["C_oracle_sufficient", "D_oracle_sufficient_steps96"], info["config_different_but_output_equivalent"])
        self.assertFalse(info["correctness_changed"]["C_oracle_sufficient"])
        self.assertTrue(info["output_changed_vs_primary"]["C_oracle_sufficient"])

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

    def test_trace_remask_selects_real_positions_without_test_result(self) -> None:
        stage1 = {
            "trajectory": {
                "top1_history": [
                    [10, 20, 30, 40],
                    [10, 21, 31, 40],
                    [11, 21, 32, 40],
                ],
                "top1_prob_history": [
                    [0.9, 0.8, 0.7, 0.6],
                    [0.8, 0.7, 0.6, 0.5],
                    [0.7, 0.4, 0.9, 0.3],
                ],
            },
            "passed": True,
            "verification": {"tier3_unit_tests": {"passed": True}},
        }

        selected = select_trace_remask_positions(stage1, canvas_len=4)

        self.assertEqual(remask_count_for_canvas(4), 1)
        self.assertEqual(selected[0]["index"], 2)
        self.assertNotIn("passed", selected[0])
        self.assertNotIn("verification", selected[0])

    def test_trace_span_remask_uses_fixed_contiguous_span_without_test_result(self) -> None:
        stage1 = {
            "trajectory": {
                "top1_history": [
                    [10, 20, 30, 40, 50, 60],
                    [10, 21, 31, 40, 50, 60],
                    [11, 21, 32, 40, 50, 60],
                ],
                "top1_prob_history": [
                    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                    [0.7, 0.4, 0.9, 0.3, 0.2, 0.1],
                ],
            },
            "passed": True,
            "verification": {"tier3_unit_tests": {"passed": True}},
        }

        span = select_trace_span_remask(stage1, canvas_len=6)

        self.assertEqual(span_remask_width_for_canvas(6), 2)
        self.assertEqual(span["center_index"], 2)
        self.assertEqual(span["remasked_token_indices"], [1, 2])
        self.assertEqual(span["span_end_exclusive"] - span["span_start"], 2)
        self.assertFalse(span["remask_rule_uses_test_result"])
        self.assertNotIn("passed", span)
        self.assertNotIn("verification", span)

    def test_experimental_seed_order_does_not_change_stable_seed(self) -> None:
        task_id = "SingleLineInfilling/HumanEval/85/L0"
        forward = [stable_task_seed(42, task_id, seed) for seed in [0, 1, 2]]
        reverse = {seed: stable_task_seed(42, task_id, seed) for seed in [2, 1, 0]}

        self.assertEqual(forward, [reverse[0], reverse[1], reverse[2]])
        self.assertEqual(len(set(forward)), 3)

    def test_distinct_candidate_hash_clustering_and_pass_at3(self) -> None:
        rows = [
            {"task_id": "hard", "action_id": "C_oracle_sufficient", "experimental_seed": 0, "generated_text_sha256": "a", "passed": False, "compile_passed": True, "error_type": "UnitTestFailure", "generated_text_len": 10},
            {"task_id": "hard", "action_id": "E_oracle_sufficient_no_early_commit", "experimental_seed": 0, "generated_text_sha256": "b", "passed": False, "compile_passed": True, "error_type": "UnitTestFailure", "generated_text_len": 11},
            {"task_id": "hard", "action_id": "F_oracle_sufficient_trace_remask", "experimental_seed": 0, "generated_text_sha256": "c", "passed": True, "compile_passed": True, "error_type": None, "generated_text_len": 12},
        ]

        summary = summarize_diversity(rows)
        clusters = cluster_hashes(rows)

        self.assertEqual(summary["case_summary"]["hard"]["unique_candidate_hash_count"], 3)
        self.assertTrue(summary["case_summary"]["hard"]["candidate_existence_pass_at_3_by_seed"]["0"])
        self.assertEqual(len(clusters), 3)
        self.assertFalse(has_multi_seed_diversity(rows))

        multi_seed_rows = [
            {"task_id": "hard", "action_id": "C_oracle_sufficient", "experimental_seed": 0, "generated_text_sha256": "a", "passed": False},
            {"task_id": "hard", "action_id": "C_oracle_sufficient", "experimental_seed": 1, "generated_text_sha256": "b", "passed": False},
        ]
        self.assertTrue(has_multi_seed_diversity(multi_seed_rows))

    def test_action_distinctness_stop_rule(self) -> None:
        rows = [
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "action_id": "C_oracle_sufficient", "generated_text_sha256": "same", "passed": False},
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "action_id": "E_oracle_sufficient_no_early_commit", "generated_text_sha256": "same", "passed": False, "early_commit_enabled": False, "actual_forward_steps": 64},
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "action_id": "F_oracle_sufficient_trace_remask", "generated_text_sha256": "same", "passed": False, "remasked_token_count": 0, "refinement_executed": False},
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "action_id": "G_oracle_sufficient_trace_span_remask", "generated_text_sha256": "same", "passed": False, "remasked_token_count": 0, "remasked_span_width": 0, "span_refinement_executed": False},
        ]

        gate = smoke_gate(rows)
        verdict = choose_distinct_candidate_verdict(rows, gate)

        self.assertFalse(gate["passed"])
        self.assertEqual(verdict, "invalid_action_not_distinct")

    def test_g_action_distinctness_accepts_real_span_mechanism(self) -> None:
        rows = [
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "action_id": "C_oracle_sufficient", "generated_text_sha256": "same", "passed": False, "actual_forward_steps": 40},
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "action_id": "E_oracle_sufficient_no_early_commit", "generated_text_sha256": "same", "passed": False, "early_commit_enabled": False, "actual_forward_steps": 64},
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "action_id": "F_oracle_sufficient_trace_remask", "generated_text_sha256": "same", "passed": False, "remasked_token_count": 4, "refinement_executed": True},
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "action_id": "G_oracle_sufficient_trace_span_remask", "generated_text_sha256": "same", "passed": False, "remasked_token_count": 6, "remasked_span_width": 6, "span_refinement_executed": True, "effective_update_steps": 3},
        ]

        gate = smoke_gate(rows)

        self.assertTrue(gate["passed"])
        self.assertTrue(gate["g_mechanism_real_executed"])
        self.assertEqual(gate["g_remasked_span_width"], 6)

    def test_output_change_and_correctness_change_are_separate_in_phase1b(self) -> None:
        rows = [
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "case_pool": "triggered_failed_long", "action_id": "C_oracle_sufficient", "experimental_seed": 0, "generated_text_sha256": "a", "passed": False, "compile_passed": False, "error_type": "SyntaxError", "generated_text_len": 10},
            {"task_id": "SingleLineInfilling/HumanEval/85/L0", "case_pool": "triggered_failed_long", "action_id": "E_oracle_sufficient_no_early_commit", "experimental_seed": 0, "generated_text_sha256": "b", "passed": False, "compile_passed": False, "error_type": "SyntaxError", "generated_text_len": 12},
        ]

        summary = summarize_diversity(rows)

        self.assertEqual(summary["case_summary"]["SingleLineInfilling/HumanEval/85/L0"]["unique_candidate_hash_count"], 2)
        self.assertFalse(summary["case_summary"]["SingleLineInfilling/HumanEval/85/L0"]["candidate_existence_pass_at_3_any_seed"])


if __name__ == "__main__":
    unittest.main()
