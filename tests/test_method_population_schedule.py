import unittest

from experiments.method_population_schedule import (
    MULTILINE_CORE_CASES,
    RANDOMSPANLIGHT_ALLOWED_CASES,
    SELECTED_METHOD_MULTILINE_CASES,
    SINGLELINE_DEVELOPMENT_CASES,
    SMOKE_CASES,
    population_schedule,
    require_randomspanlight_full,
)


class MethodPopulationScheduleTest(unittest.TestCase):
    def test_route_is_fixed_from_smoke_to_selected_only_multiline(self) -> None:
        self.assertEqual(
            population_schedule(),
            [
                {"stage": "technical_smoke", "case_count": SMOKE_CASES},
                {"stage": "randomspanlight_full", "case_count": RANDOMSPANLIGHT_ALLOWED_CASES},
                {"stage": "multiline_core", "case_count": MULTILINE_CORE_CASES},
                {"stage": "singleline_development", "case_count": SINGLELINE_DEVELOPMENT_CASES},
                {"stage": "selected_method_multiline", "case_count": SELECTED_METHOD_MULTILINE_CASES},
            ],
        )

    def test_automatic_full_is_limited_to_randomspanlight_148(self) -> None:
        require_randomspanlight_full(RANDOMSPANLIGHT_ALLOWED_CASES)
        with self.assertRaises(RuntimeError):
            require_randomspanlight_full(SELECTED_METHOD_MULTILINE_CASES)


if __name__ == "__main__":
    unittest.main()
