from __future__ import annotations

import unittest
from types import SimpleNamespace

from clean_scripts.run_route2_rescue_quality_v5 import (
    CandidateSpec,
    add_route2_rescue_quality_v5_metadata,
    build_candidate_record,
    filter_tasks_by_ids,
    oracle_upper_bound_summary,
    pairwise_vs_references,
    parse_task_ids_csv,
    selector_metadata,
    select_candidate,
)


def make_result(
    task_id: str,
    *,
    passed: bool,
    selected: int,
    middle_text: str,
    confidence: float = 0.8,
    top1: float = 0.8,
    gap: float = 0.2,
    remaining: int = 0,
    parse_passed: bool = True,
    compile_passed: bool = True,
) -> dict:
    return {
        "task_id": task_id,
        "code": f"def f():\n{middle_text}\n",
        "middle_text": middle_text,
        "metrics": {
            "passed": passed,
            "oracle_mask_length": 999 if passed else 1,
            "selected_mask_length": selected,
            "mask_length": selected,
            "selected_minus_oracle_length": 0,
            "abs_selected_minus_oracle_length": 0,
            "decode_sec": 1.0,
            "verification_sec": 0.1,
            "total_sec": 1.1,
            "length_probe_sec": 0.0,
            "total_sec_including_probe": 1.1,
        },
        "verification": {"tier3_unit_tests": {"passed": passed}},
        "diagnostics": {
            "final_full_code_parse_passed": parse_passed,
            "final_full_code_compile_passed": compile_passed,
        },
        "length_probe": {},
        "stopping": {},
        "step_traces": [
            {
                "step": 0,
                "remaining_masks_after_update": remaining,
                "mean_confidence": confidence,
                "stop_decision": {"mean_top1": top1, "mean_gap": gap},
            }
        ],
    }


class Route2RescueQualityV5SelectorTest(unittest.TestCase):
    def test_consensus_confidence_selector_ignores_forbidden_outcome_fields(self) -> None:
        good_signal_bad_outcome = build_candidate_record(
            CandidateSpec("good_signal", min_length=24, steps=64),
            make_result(
                "t",
                passed=False,
                selected=24,
                middle_text="    return x + 1",
                confidence=0.95,
                top1=0.94,
                gap=0.55,
                remaining=0,
            ),
        )
        bad_signal_good_outcome = build_candidate_record(
            CandidateSpec("bad_signal", min_length=32, steps=64),
            make_result(
                "t",
                passed=True,
                selected=32,
                middle_text="    return x - 1",
                confidence=0.10,
                top1=0.10,
                gap=0.01,
                remaining=12,
            ),
        )

        selected, scores = select_candidate(
            [bad_signal_good_outcome, good_signal_bad_outcome],
            selector_name="consensus_confidence",
        )

        self.assertEqual(selected["candidate_id"], "good_signal")
        self.assertEqual(scores["selector"], "consensus_confidence")
        self.assertNotIn("passed", scores["used_policy_fields"])
        self.assertNotIn("oracle_mask_length", scores["used_policy_fields"])
        self.assertNotIn("unit_tests", " ".join(scores["used_policy_fields"]))

    def test_syntax_aware_selector_is_labeled_as_compiler_assisted(self) -> None:
        metadata = selector_metadata("syntax_aware")

        self.assertEqual(metadata["claim_boundary"], "compiler_assisted_inference_time")
        self.assertIn("parse_passed", metadata["used_policy_fields"])
        self.assertIn("compile_passed", metadata["used_policy_fields"])
        self.assertNotIn("passed", metadata["used_policy_fields"])

    def test_anchor_len32_selector_keeps_anchor_over_high_scoring_short_candidate(self) -> None:
        short = build_candidate_record(
            CandidateSpec("len24_s64", min_length=24, steps=64),
            make_result(
                "t",
                passed=False,
                selected=24,
                middle_text="    return short",
                confidence=0.99,
                top1=0.99,
                gap=0.80,
                remaining=0,
            ),
        )
        anchor = build_candidate_record(
            CandidateSpec("len32_s64", min_length=32, steps=64),
            make_result(
                "t",
                passed=True,
                selected=32,
                middle_text="    return anchor",
                confidence=0.30,
                top1=0.30,
                gap=0.05,
                remaining=4,
            ),
        )
        slow = build_candidate_record(
            CandidateSpec("len32_s96", min_length=32, steps=96),
            make_result(
                "t",
                passed=False,
                selected=32,
                middle_text="    return slow",
                confidence=0.29,
                top1=0.29,
                gap=0.05,
                remaining=4,
            ),
        )

        selected, scores = select_candidate(
            [short, anchor, slow],
            selector_name="anchor_len32_confidence",
        )

        self.assertEqual(selected["candidate_id"], "len32_s64")
        self.assertEqual(scores["selection_reason"], "anchor_default")
        self.assertIn("len24_s64", scores["diagnostic_only_candidate_ids"])
        self.assertNotIn("passed", scores["used_policy_fields"])

    def test_anchor_len32_selector_switches_to_slow_len32_on_margin(self) -> None:
        anchor = build_candidate_record(
            CandidateSpec("len32_s64", min_length=32, steps=64),
            make_result(
                "t",
                passed=True,
                selected=32,
                middle_text="    return anchor",
                confidence=0.30,
                top1=0.30,
                gap=0.05,
                remaining=4,
            ),
        )
        slow = build_candidate_record(
            CandidateSpec("len32_s96", min_length=32, steps=96),
            make_result(
                "t",
                passed=False,
                selected=32,
                middle_text="    return slow",
                confidence=0.95,
                top1=0.95,
                gap=0.40,
                remaining=0,
            ),
        )

        selected, scores = select_candidate(
            [anchor, slow],
            selector_name="anchor_len32_confidence",
            anchor_switch_margin=0.02,
        )

        self.assertEqual(selected["candidate_id"], "len32_s96")
        self.assertEqual(scores["selection_reason"], "non_anchor_len32_score_margin")
        self.assertEqual(scores["anchor_candidate_id"], "len32_s64")

    def test_anchor_len32_short_trace_override_selects_len24_only_on_strong_trace_delta(self) -> None:
        short = build_candidate_record(
            CandidateSpec("len24_s64", min_length=24, steps=64),
            make_result(
                "t",
                passed=True,
                selected=24,
                middle_text="    return short",
                confidence=0.95,
                top1=0.80,
                gap=0.70,
                remaining=0,
            ),
        )
        anchor = build_candidate_record(
            CandidateSpec("len32_s64", min_length=32, steps=64),
            make_result(
                "t",
                passed=False,
                selected=32,
                middle_text="    return anchor",
                confidence=0.95,
                top1=0.40,
                gap=0.30,
                remaining=0,
            ),
        )

        selected, scores = select_candidate(
            [short, anchor],
            selector_name="anchor_len32_short_trace_override",
            short_override_gap_margin=0.28,
            short_override_top1_margin=0.28,
        )

        self.assertEqual(selected["candidate_id"], "len24_s64")
        self.assertEqual(scores["selection_reason"], "short_trace_gap_top1_override")
        self.assertEqual(scores["claim_boundary"], "pure_inference_time_verifier_free_anchor_protected_short_trace_override")
        self.assertNotIn("passed", scores["used_policy_fields"])

    def test_anchor_len32_short_trace_override_keeps_anchor_without_strong_delta(self) -> None:
        short = build_candidate_record(
            CandidateSpec("len24_s64", min_length=24, steps=64),
            make_result(
                "t",
                passed=True,
                selected=24,
                middle_text="    return short",
                confidence=0.99,
                top1=0.50,
                gap=0.40,
                remaining=0,
            ),
        )
        anchor = build_candidate_record(
            CandidateSpec("len32_s64", min_length=32, steps=64),
            make_result(
                "t",
                passed=False,
                selected=32,
                middle_text="    return anchor",
                confidence=0.30,
                top1=0.40,
                gap=0.30,
                remaining=4,
            ),
        )

        selected, scores = select_candidate(
            [short, anchor],
            selector_name="anchor_len32_short_trace_override",
            short_override_gap_margin=0.28,
            short_override_top1_margin=0.28,
        )

        self.assertEqual(selected["candidate_id"], "len32_s64")
        self.assertEqual(scores["selection_reason"], "anchor_default")

    def test_oracle_upper_bound_is_offline_only(self) -> None:
        candidates = [
            build_candidate_record(
                CandidateSpec("a", min_length=24, steps=64),
                make_result("t", passed=False, selected=24, middle_text="    return 0"),
            ),
            build_candidate_record(
                CandidateSpec("b", min_length=32, steps=64),
                make_result("t", passed=True, selected=32, middle_text="    return 1"),
            ),
        ]

        summary = oracle_upper_bound_summary(candidates)

        self.assertTrue(summary["offline_only"])
        self.assertTrue(summary["any_candidate_passed"])
        self.assertEqual(summary["passing_candidate_ids"], ["b"])

    def test_selected_candidate_metadata_is_attached_to_final_result(self) -> None:
        primary = make_result("t", passed=False, selected=8, middle_text="    return 0")
        selected = build_candidate_record(
            CandidateSpec("len32_s64", min_length=32, steps=64),
            make_result("t", passed=True, selected=32, middle_text="    return 1"),
        )
        final = make_result("t", passed=True, selected=32, middle_text="    return 1")

        add_route2_rescue_quality_v5_metadata(
            final,
            primary,
            route2_policy_name="precision_top1_conf",
            route2_triggered=True,
            candidates=[selected],
            selected_candidate=selected,
            selector_name="consensus_confidence",
            selector_scores={"selected_score": 1.0, "candidate_scores": {"len32_s64": 1.0}},
        )

        metrics = final["metrics"]
        self.assertTrue(metrics["route2_rescue_quality_v5_triggered"])
        self.assertEqual(metrics["route2_rescue_quality_v5_selector"], "consensus_confidence")
        self.assertEqual(metrics["route2_rescue_quality_v5_selected_candidate_id"], "len32_s64")
        self.assertEqual(final["route2_rescue_quality_v5"]["selected_candidate_id"], "len32_s64")


class Route2RescueQualityV5SummaryTest(unittest.TestCase):
    def test_pairwise_vs_references_summarizes_multiple_references(self) -> None:
        results = [
            {"task_id": "win", "metrics": {"passed": True, "oracle_mask_length": 20}},
            {"task_id": "loss", "metrics": {"passed": False, "oracle_mask_length": 6}},
            {"task_id": "tie", "metrics": {"passed": True, "oracle_mask_length": 10}},
        ]
        references = {
            "midcons": [
                {"task_id": "win", "metrics": {"passed": False}},
                {"task_id": "loss", "metrics": {"passed": True}},
                {"task_id": "tie", "metrics": {"passed": True}},
            ],
            "route2_precision_len32": [
                {"task_id": "win", "metrics": {"passed": True}},
                {"task_id": "loss", "metrics": {"passed": False}},
                {"task_id": "tie", "metrics": {"passed": True}},
            ],
        }

        summary = pairwise_vs_references(results, references)

        self.assertEqual(summary["midcons"]["counts"], {"win": 1, "loss": 1, "tie_pass": 1, "tie_fail": 0})
        self.assertEqual(
            summary["route2_precision_len32"]["counts"],
            {"win": 0, "loss": 0, "tie_pass": 2, "tie_fail": 1},
        )


class Route2RescueQualityV5TaskSelectionTest(unittest.TestCase):
    def test_parse_task_ids_csv_strips_empty_items(self) -> None:
        self.assertEqual(parse_task_ids_csv(" a, ,b ,, c "), ["a", "b", "c"])
        self.assertIsNone(parse_task_ids_csv(None))

    def test_filter_tasks_by_ids_uses_requested_order_and_rejects_missing(self) -> None:
        tasks = [SimpleNamespace(task_id="a"), SimpleNamespace(task_id="b"), SimpleNamespace(task_id="c")]

        selected = filter_tasks_by_ids(tasks, ["c", "a"])

        self.assertEqual([task.task_id for task in selected], ["c", "a"])
        with self.assertRaises(ValueError):
            filter_tasks_by_ids(tasks, ["missing"])


if __name__ == "__main__":
    unittest.main()
