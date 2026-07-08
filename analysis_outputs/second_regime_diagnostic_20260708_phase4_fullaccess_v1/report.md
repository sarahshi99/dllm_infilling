# Second-Regime Minimal Diagnostic

Verdict: `second_regime_minimal_diagnostic_completed`.

Regime: `synthetic_second_regime_minimal` (official MultiLine/RandomSpan JSONL files remain missing; this is not an official benchmark result).

Cases: `18`.
Short/medium/long counts: `{'short': 6, 'medium': 6, 'long': 6}`.
Control fresh pass: `18`.
Best deployable cal-lite fresh pass: `18`.
Oracle-sufficient fresh pass: `18`.
Canvas-limited fraction: `0.0`.
Rescue-limited fraction: `0.0`.
Sufficient-canvas recoverability: `0.0`.
Deployable policy gap: `0`.
Average cost sec: `{'control_fixed_fresh': 1.4054840178481147, 'best_deployable_cal_lite_fresh': 1.6589558525431332, 'oracle_sufficient_canvas_fresh': 1.3660979580276438}`.

The deployable comparison uses the existing cal-lite length policy freshly on the synthetic tasks. Historical V6 rows are not reused because the synthetic task ids have no matching baseline/reference rows.
