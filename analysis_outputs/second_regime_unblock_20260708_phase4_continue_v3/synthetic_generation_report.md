# Synthetic Second-Regime Minimal Construction

Verdict: `synthetic_second_regime_minimal_constructed`.

Cases: `18`.
Bucket histogram: `{'short': 6, 'medium': 6, 'long': 6}`.

Construction: reconstruct HumanEval solutions from existing SingleLine source fields, then cut synthetic 1-line, 3-line, and 6-line spans while preserving tests and entry point. This is explicitly synthetic and must not be reported as the official MultiLine or RandomSpan benchmark.
