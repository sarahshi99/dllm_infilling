"""Fixed population route for independent candidate methods.

The route is a compute-budget guard, not an outcome gate.  Only the
RandomSpanLight 148-task tier may be reached automatically from a technical
smoke.  The 5079-span MultiLine tier requires an explicit selected-method
authorization after the intermediate development tiers.
"""

from __future__ import annotations

SMOKE_CASES = 12
RANDOMSPANLIGHT_ALLOWED_CASES = 148
MULTILINE_CORE_CASES = 296
SINGLELINE_DEVELOPMENT_CASES = 927
SELECTED_METHOD_MULTILINE_CASES = 5079

POPULATION_ROUTE = (
    ("technical_smoke", SMOKE_CASES),
    ("randomspanlight_full", RANDOMSPANLIGHT_ALLOWED_CASES),
    ("multiline_core", MULTILINE_CORE_CASES),
    ("singleline_development", SINGLELINE_DEVELOPMENT_CASES),
    ("selected_method_multiline", SELECTED_METHOD_MULTILINE_CASES),
)


def population_schedule() -> list[dict[str, int | str]]:
    return [{"stage": stage, "case_count": count} for stage, count in POPULATION_ROUTE]


def require_randomspanlight_full(case_count: int) -> None:
    if int(case_count) != RANDOMSPANLIGHT_ALLOWED_CASES:
        raise RuntimeError(
            f"automatic full is restricted to {RANDOMSPANLIGHT_ALLOWED_CASES} RandomSpanLight cases, got {case_count}"
        )


def selected_multiline_full_requested(*, auto_full: bool, selected_method_only_5079: bool) -> bool:
    """Reject the historical bare --auto-full route to 5079 MultiLine rows."""
    if bool(auto_full) != bool(selected_method_only_5079):
        raise ValueError(
            "5079 MultiLine execution requires both --auto-full and "
            "--selected-method-only-5079 after the intermediate tiers"
        )
    return bool(auto_full)
