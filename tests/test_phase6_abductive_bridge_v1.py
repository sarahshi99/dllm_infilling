import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.phase6_abductive_bridge_v1 import (
    FORBIDDEN_SELECTION_FIELDS,
    analyze_candidate,
    extract_backward_obligations,
    select_v1_candidate,
    validate_selection_fields,
)
from experiments.phase6_abductive_bridge_runner import (
    build_deployable_refinement_plan,
    build_refinement_row_for_case,
    refinement_key,
    run_population,
)
from experiments.phase6_remask import (
    dependency_cone_plan,
    dependency_cone_token_indices,
    generic_low_confidence_indices,
)


PREFIX = "def f(xs):\n"
SUFFIX = "    return total\n"
CANDIDATE = "    subtotal = sum(xs)\n    total = subtotal + 1\n"


class Tokenizer:
    mapping = {0: "    subtotal = sum(xs)\n", 1: "    total = subtotal + 1\n", 99: ""}

    def decode(self, ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        return "".join(self.mapping[int(item)] for item in ids)


class PoisonMapping(dict):
    forbidden = {
        "reference_code",
        "reference_middle",
        "canonical_solution",
        "verification",
        "passed",
        "pass_fail",
        "oracle_length",
        "oracle_mask_length",
        "task_id",
        "task_group",
        "split",
        "split_label",
        "source_row_id",
        "case_index",
        "reference_middle_tokens",
        "length_bucket",
    }

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"forbidden deployable input read: {key}")
        return super().get(key, default)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"forbidden deployable input read: {key}")
        return super().__getitem__(key)


def stage1_rows(candidate: str = CANDIDATE, poison: bool = False):
    rows = []
    for canvas in (16, 32, 64, 128):
        for seed in (0, 1):
            row_type = PoisonMapping if poison else dict
            row = row_type(
                {
                    "candidate_key": f"bank:{canvas}:{seed}",
                    "candidate_kind": "deployable_grid",
                    "row_key": "row-a",
                    "canvas_tokens": canvas,
                    "seed": seed,
                    "status": "ok",
                    "prefix_text": PREFIX,
                    "middle_text": candidate,
                    "suffix_text": SUFFIX,
                    "middle_token_ids": [0, 1] + [99] * (canvas - 2),
                    "final_token_confidences": [0.9, 0.1] + [0.5] * (canvas - 2),
                    "metrics": {"actual_forward_count": 64, "total_sec_including_probe": 1.0},
                    "reference_code": "POISON",
                    "canonical_solution": "POISON",
                    "passed": True,
                    "task_group": "POISON",
                    "split_label": "POISON",
                    "reference_middle_tokens": 999,
                }
            )
            rows.append(row)
    return rows


def manifest_row():
    return {
        "row_key": "row-a",
        "case_index": 0,
        "source_row_id": 0,
        "task_group": "HumanEval/0",
        "length_bucket": "short",
        "reference_middle_tokens": 4,
    }


def fake_decode(calls):
    def run(**kwargs):
        calls.append(kwargs)
        canvas = int(kwargs["canvas_tokens"])
        middle_ids = list(kwargs["initial_middle_ids"])
        middle = kwargs["tokenizer"].decode(middle_ids)
        code = PREFIX + middle + SUFFIX
        return {
            "code": code,
            "middle_text": middle,
            "middle_token_ids": middle_ids,
            "final_token_confidences": [0.5] * canvas,
            "metrics": {
                "passed": False,
                "actual_forward_count": int(kwargs["total_steps"]),
                "total_sec_including_probe": 2.0,
            },
            "verification": {"tier3_unit_tests": {}},
            "diagnostics": {},
        }

    return run


class Phase6AbductiveBridgeV1Test(unittest.TestCase):
    def test_suffix_extracts_required_definition_and_boundary(self) -> None:
        obligations = extract_backward_obligations(
            "def f(xs):\n",
            "    total += x\n    return total\n",
        )
        self.assertIn("total", obligations.required_definitions)
        self.assertEqual(obligations.entry_indent, 4)
        self.assertTrue(obligations.requires_fallthrough)

    def test_candidate_restores_suffix_definition(self) -> None:
        prefix = "def f(xs):\n"
        suffix = "    return total\n"
        good = analyze_candidate(prefix, "    total = sum(xs)\n", suffix)
        bad = analyze_candidate(prefix, "    value = sum(xs)\n", suffix)
        self.assertEqual(good.unsatisfied_obligations, ())
        self.assertIn("total", bad.unsatisfied_obligations)
        self.assertLess(good.ranking_key, bad.ranking_key)

    def test_candidate_reports_undefined_use_and_control_contradiction(self) -> None:
        analysis = analyze_candidate(
            "def f(flag):\n",
            "    break\n    value = missing + 1\n",
            "    return value\n",
        )
        self.assertIn("missing", analysis.undefined_uses)
        self.assertIn("break_outside_loop", analysis.control_contradictions)

    def test_control_obligation_is_explicitly_unsatisfied(self) -> None:
        analysis = analyze_candidate(
            "def f(x):\n",
            "    value = x\n",
            "    break\n",
        )
        self.assertIn("control:active_loop", analysis.unsatisfied_obligations)

    def test_lexicographic_selection_prefers_satisfied_obligations(self) -> None:
        rows = [
            {"candidate_ordinal": 0, "middle_text": "    total = missing\n", "canvas_tokens": 16, "seed": 0},
            {"candidate_ordinal": 1, "middle_text": "    total = sum(xs)\n", "canvas_tokens": 32, "seed": 1},
        ]
        chosen, diagnostics = select_v1_candidate(PREFIX, SUFFIX, rows)
        self.assertEqual(chosen["candidate_ordinal"], 1)
        self.assertEqual(diagnostics[1].undefined_uses, ())

    def test_error_candidate_is_ranked_without_crashing(self) -> None:
        rows = [
            {"candidate_ordinal": 0, "middle_text": "", "canvas_tokens": 16, "seed": 0},
            {"candidate_ordinal": 1, "middle_text": "    total = sum(xs)\n", "canvas_tokens": 32, "seed": 0},
        ]
        chosen, _ = select_v1_candidate(PREFIX, SUFFIX, rows)
        self.assertEqual(chosen["candidate_ordinal"], 1)

    def test_forbidden_selection_fields_are_rejected(self) -> None:
        for field in FORBIDDEN_SELECTION_FIELDS:
            with self.assertRaises(ValueError):
                validate_selection_fields(["candidate_key", field])

    def test_generic_remask_uses_only_low_confidence_token_state(self) -> None:
        indices = generic_low_confidence_indices([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.95], 10)
        self.assertEqual(indices, [1])

    def test_dependency_cone_uses_transitive_statement_def_use_not_name_match(self) -> None:
        plan = dependency_cone_plan(
            Tokenizer(),
            [0, 1],
            prefix=PREFIX,
            suffix=SUFFIX,
            dependency_names=["total"],
        )
        self.assertTrue(plan.executable)
        self.assertEqual(plan.token_indices, (0, 1))
        self.assertEqual(plan.selected_statement_count, 2)
        self.assertEqual(
            dependency_cone_token_indices(
                Tokenizer(), [0, 1], ["total"], prefix=PREFIX, suffix=SUFFIX
            ),
            [0, 1],
        )

    def test_missing_suffix_definition_is_explicit_null_refinement(self) -> None:
        class MissingTokenizer(Tokenizer):
            mapping = {0: "    value = sum(xs)\n", 1: "", 99: ""}

        calls = []
        row = build_refinement_row_for_case(
            method="m1_dependency_cone_remask",
            manifest_row=manifest_row(),
            source_row=PoisonMapping(),
            stage1_rows=stage1_rows("    value = sum(xs)\n"),
            tokenizer=MissingTokenizer(),
            model=object(),
            cfg_for=lambda canvas, seed: (canvas, seed),
            set_seed=lambda seed: None,
            decode_fn=fake_decode(calls),
            evaluator_task_factory=lambda prefix, suffix, source: object(),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["total_steps"], 64)
        self.assertEqual(calls[0]["initial_mask_indices"], [])
        self.assertEqual(calls[0]["phase_name"], "stage2_m1_fixed64_null_refinement")
        self.assertTrue(row["fallback_to_fixed64"])
        self.assertTrue(row["null_refinement_executed"])
        self.assertEqual(row["metrics"]["refinement_forward_count"], 64)
        self.assertEqual(row["metrics"]["actual_forward_count"], 128)

    def test_synthetic_activation_runs_64_forwards_for_generic_and_m1(self) -> None:
        calls = []
        kwargs = {
            "manifest_row": manifest_row(),
            "source_row": PoisonMapping(),
            "stage1_rows": stage1_rows(),
            "tokenizer": Tokenizer(),
            "model": object(),
            "cfg_for": lambda canvas, seed: (canvas, seed),
            "set_seed": lambda seed: None,
            "decode_fn": fake_decode(calls),
            "evaluator_task_factory": lambda prefix, suffix, source: object(),
        }
        generic = build_refinement_row_for_case(method="equal_compute_generic_remask", **kwargs)
        m1 = build_refinement_row_for_case(method="m1_dependency_cone_remask", **kwargs)
        self.assertEqual([call["total_steps"] for call in calls], [64, 64])
        self.assertEqual(len(calls[0]["initial_mask_indices"]), len(calls[1]["initial_mask_indices"]))
        self.assertEqual(generic["metrics"]["refinement_forward_count"], 64)
        self.assertEqual(m1["metrics"]["refinement_forward_count"], 64)
        self.assertTrue(m1["targeted_remask_executed"])

    def test_seed_and_remask_plan_are_deterministic(self) -> None:
        calls = []
        seeds = []
        kwargs = {
            "method": "m1_dependency_cone_remask",
            "manifest_row": manifest_row(),
            "source_row": PoisonMapping(),
            "stage1_rows": stage1_rows(),
            "tokenizer": Tokenizer(),
            "model": object(),
            "cfg_for": lambda canvas, seed: (canvas, seed),
            "set_seed": seeds.append,
            "decode_fn": fake_decode(calls),
            "evaluator_task_factory": lambda prefix, suffix, source: object(),
        }
        first = build_refinement_row_for_case(**kwargs)
        second = build_refinement_row_for_case(**kwargs)
        self.assertEqual(seeds, [0, 0])
        self.assertEqual(first["remasked_token_indices"], second["remasked_token_indices"])
        self.assertEqual(first["selected_base_candidate_key"], second["selected_base_candidate_key"])

    def test_refinement_resume_deduplicates_and_rejects_duplicate_raw_keys(self) -> None:
        method = "equal_compute_generic_remask"
        manifest = [manifest_row()]
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "generic.jsonl"

            def append(path, row):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")

            generated = {
                "candidate_key": refinement_key("row-a", method),
                "candidate_kind": method,
                "row_key": "row-a",
                "status": "ok",
            }
            with patch(
                "experiments.phase6_abductive_bridge_runner.build_refinement_row_for_case",
                return_value=generated,
            ):
                self.assertEqual(
                    run_population(
                        method=method,
                        manifest=manifest,
                        source_rows=[{}],
                        stage1_rows=stage1_rows(),
                        raw_path=raw_path,
                        tokenizer=Tokenizer(),
                        model=object(),
                        cfg_for=lambda canvas, seed: (canvas, seed),
                        set_seed=lambda seed: None,
                        append_jsonl=append,
                    ),
                    1,
                )
                self.assertEqual(
                    run_population(
                        method=method,
                        manifest=manifest,
                        source_rows=[{}],
                        stage1_rows=stage1_rows(),
                        raw_path=raw_path,
                        tokenizer=Tokenizer(),
                        model=object(),
                        cfg_for=lambda canvas, seed: (canvas, seed),
                        set_seed=lambda seed: None,
                        append_jsonl=append,
                    ),
                    0,
                )
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(generated) + "\n")
            with self.assertRaises(RuntimeError):
                run_population(
                    method=method,
                    manifest=manifest,
                    source_rows=[{}],
                    stage1_rows=stage1_rows(),
                    raw_path=raw_path,
                    tokenizer=Tokenizer(),
                    model=object(),
                    cfg_for=lambda canvas, seed: (canvas, seed),
                    set_seed=lambda seed: None,
                    append_jsonl=append,
                )

    def test_forbidden_inputs_do_not_enter_deployable_selection_or_remask_plan(self) -> None:
        selected_state, _, _, _, cone = build_deployable_refinement_plan(stage1_rows(poison=True), Tokenizer())
        self.assertEqual(selected_state["canvas_tokens"], 16)
        self.assertTrue(cone.executable)


if __name__ == "__main__":
    unittest.main()
