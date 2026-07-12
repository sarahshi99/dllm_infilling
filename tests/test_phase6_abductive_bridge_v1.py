import unittest

from analysis.phase6_abductive_bridge_v1 import (
    FORBIDDEN_SELECTION_FIELDS,
    analyze_candidate,
    extract_backward_obligations,
    select_v1_candidate,
    validate_selection_fields,
)


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
        prefix = "def f(xs):\n"
        suffix = "    return total\n"
        rows = [
            {"candidate_key": "bad", "middle_text": "    total = missing\n", "canvas_tokens": 16, "seed": 0},
            {"candidate_key": "good", "middle_text": "    total = sum(xs)\n", "canvas_tokens": 32, "seed": 1},
        ]
        chosen, diagnostics = select_v1_candidate(prefix, suffix, rows)
        self.assertEqual(chosen["candidate_key"], "good")
        self.assertEqual(diagnostics["good"].undefined_uses, ())

    def test_error_candidate_is_ranked_without_crashing(self) -> None:
        prefix = "def f(xs):\n"
        suffix = "    return total\n"
        rows = [
            {"candidate_key": "error", "middle_text": "", "canvas_tokens": 16, "seed": 0},
            {"candidate_key": "good", "middle_text": "    total = sum(xs)\n", "canvas_tokens": 32, "seed": 0},
        ]
        chosen, _ = select_v1_candidate(prefix, suffix, rows)
        self.assertEqual(chosen["candidate_key"], "good")

    def test_forbidden_selection_fields_are_rejected(self) -> None:
        for field in FORBIDDEN_SELECTION_FIELDS:
            with self.assertRaises(ValueError):
                validate_selection_fields(["candidate_key", field])


if __name__ == "__main__":
    unittest.main()
