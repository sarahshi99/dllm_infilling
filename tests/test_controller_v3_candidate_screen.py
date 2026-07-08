import unittest

from experiments.action_ceiling import controller_v3_candidate_screen as v3


class ControllerV3CandidateScreenTest(unittest.TestCase):
    def test_feature_columns_exclude_forbidden_fields(self):
        for variant in ("probe_only", "probe_trace_fused"):
            cols = set(v3.feature_columns(variant, action=True))
            self.assertFalse(cols & v3.FORBIDDEN_FEATURES)

    def test_validate_test_lock_requires_sealed_zero(self):
        self.assertEqual(
            v3.validate_test_lock({"test_status": "sealed", "test_evaluation_count": 0}),
            {"test_status": "sealed", "test_evaluation_count": 0},
        )
        with self.assertRaises(ValueError):
            v3.validate_test_lock({"test_status": "authorized_once", "test_evaluation_count": 0})
        with self.assertRaises(ValueError):
            v3.validate_test_lock({"test_status": "sealed", "test_evaluation_count": 1})

    def test_gate_flags_require_net_two_for_frozen_test(self):
        base = {
            "interventions": 5,
            "pass_count": 90,
            "wins": 1,
            "losses": 0,
            "population_harm_upper95": 0.023,
            "short_bucket_losses": 0,
            "net": 1,
            "missed_long_wins": 0,
            "average_cost": 0.82,
        }
        flags = v3.gate_flags(base, v6_pass_count=89, v6_cost=1.69, family="targeted_missed_long")
        self.assertTrue(flags["exploratory_gate_passed"])
        self.assertFalse(flags["frozen_test_gate_passed"])

        stronger = dict(base, wins=2, net=2)
        flags = v3.gate_flags(stronger, v6_pass_count=89, v6_cost=1.69, family="targeted_missed_long")
        self.assertTrue(flags["frozen_test_gate_passed"])

    def test_family_c_cannot_authorize_frozen_test(self):
        row = {
            "interventions": 5,
            "pass_count": 91,
            "wins": 2,
            "losses": 0,
            "population_harm_upper95": 0.023,
            "short_bucket_losses": 0,
            "net": 2,
            "missed_long_wins": 0,
            "average_cost": 0.82,
        }
        flags = v3.gate_flags(row, v6_pass_count=89, v6_cost=1.69, family="oracle_win_distillation")
        self.assertTrue(flags["exploratory_gate_passed"])
        self.assertFalse(flags["frozen_test_gate_passed"])


if __name__ == "__main__":
    unittest.main()
