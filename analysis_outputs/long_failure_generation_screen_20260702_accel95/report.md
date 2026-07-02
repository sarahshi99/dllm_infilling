# Long-Failure Generation Screen

final_verdict: `mixed_generation_signal`

## Coverage

- cases: `95` = hard `89` + positive controls `6`
- seed-0 generation actions: `285` / expected `285`
- conditional multi-seed actions: `180` over `30` selected cases, experimental seeds `[1, 2]`
- `experimental_seed` is the preregistered 0/1/2 seed; `seed` is the per-task derived RNG seed.

## Pool Summary

| Pool | Cases | New Correct | Unique Candidate Cases | Compile Repairs | Changed vs C | Unchanged By All | Mean Sec | P95 Sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `missed_failed_long` | 56 | 31 | 7 | 6 | 10 | 46 | 4.712 | 6.449 |
| `positive_control_rescued` | 6 | 6 | 1 | 4 | 1 | 5 | 3.801 | 6.482 |
| `triggered_failed_long` | 33 | 0 | 3 | 3 | 3 | 30 | 3.784 | 5.542 |

## Action Summary

| Action | Rows | Hard Correct | Compile Success | Changed vs C | Avg Remask Tokens | Avg Span Width | Mean Sec | P95 Sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E_oracle_sufficient_no_early_commit` | 95 | 31 | 73 | 13 | 0.00 | 0.00 | 4.042 | 6.208 |
| `F_oracle_sufficient_trace_remask` | 95 | 30 | 71 | 7 | 3.18 | 0.00 | 4.229 | 5.896 |
| `G_oracle_sufficient_trace_span_remask` | 95 | 30 | 71 | 7 | 5.79 | 5.79 | 4.725 | 6.668 |

## Hard-Case Correctness Overlap

- hard cases with any new correct candidate: `31/89`
- E correct hard cases: `31`
- F correct hard cases: `30`
- G correct hard cases: `30`
- hard cases with >1 candidate hash: `10/89`
- hard compile repairs: `9/89`
- hard all-actions-unchanged: `76/89`
- hard all-actions-distinct-but-wrong: `1/89`

| Passing action pattern on hard cases | Count |
|---|---:|
| `E+F` | 1 |
| `E+F+G` | 29 |
| `E+G` | 1 |
| `none` | 58 |

## Error-Type Transitions

| Pool | Historical Error | Action | New Error | Count |
|---|---|---|---|---:|
| `missed_failed_long` | `SyntaxError` | `E_oracle_sufficient_no_early_commit` | `None` | 3 |
| `missed_failed_long` | `SyntaxError` | `E_oracle_sufficient_no_early_commit` | `UnitTestFailure` | 3 |
| `missed_failed_long` | `SyntaxError` | `F_oracle_sufficient_trace_remask` | `None` | 3 |
| `missed_failed_long` | `SyntaxError` | `F_oracle_sufficient_trace_remask` | `UnitTestFailure` | 3 |
| `missed_failed_long` | `SyntaxError` | `G_oracle_sufficient_trace_span_remask` | `None` | 3 |
| `missed_failed_long` | `SyntaxError` | `G_oracle_sufficient_trace_span_remask` | `UnitTestFailure` | 3 |
| `missed_failed_long` | `UnitTestFailure` | `E_oracle_sufficient_no_early_commit` | `None` | 28 |
| `missed_failed_long` | `UnitTestFailure` | `E_oracle_sufficient_no_early_commit` | `SyntaxError` | 2 |
| `missed_failed_long` | `UnitTestFailure` | `F_oracle_sufficient_trace_remask` | `None` | 27 |
| `missed_failed_long` | `UnitTestFailure` | `F_oracle_sufficient_trace_remask` | `SyntaxError` | 3 |
| `missed_failed_long` | `UnitTestFailure` | `G_oracle_sufficient_trace_span_remask` | `None` | 27 |
| `missed_failed_long` | `UnitTestFailure` | `G_oracle_sufficient_trace_span_remask` | `SyntaxError` | 3 |
| `positive_control_rescued` | `SyntaxError` | `E_oracle_sufficient_no_early_commit` | `None` | 4 |
| `positive_control_rescued` | `SyntaxError` | `F_oracle_sufficient_trace_remask` | `None` | 4 |
| `positive_control_rescued` | `SyntaxError` | `G_oracle_sufficient_trace_span_remask` | `None` | 3 |
| `positive_control_rescued` | `SyntaxError` | `G_oracle_sufficient_trace_span_remask` | `UnitTestFailure` | 1 |
| `positive_control_rescued` | `UnitTestFailure` | `E_oracle_sufficient_no_early_commit` | `None` | 2 |
| `positive_control_rescued` | `UnitTestFailure` | `F_oracle_sufficient_trace_remask` | `None` | 2 |
| `positive_control_rescued` | `UnitTestFailure` | `G_oracle_sufficient_trace_span_remask` | `None` | 2 |
| `triggered_failed_long` | `SyntaxError` | `E_oracle_sufficient_no_early_commit` | `UnitTestFailure` | 3 |
| `triggered_failed_long` | `SyntaxError` | `F_oracle_sufficient_trace_remask` | `UnitTestFailure` | 2 |
| `triggered_failed_long` | `SyntaxError` | `G_oracle_sufficient_trace_span_remask` | `UnitTestFailure` | 2 |

## Conditional Multi-Seed Diagnostic

This section is conditional diagnostic only; it is not an unbiased benchmark Pass@3.

- selected cases: `30`
- actions: `180`
- hard selected cases with any correct candidate under seeds 1/2: `24`
- hard cases corrected only by conditional seeds 1/2, not seed 0: `0`

## Interpretation Boundary

- These are offline oracle-sufficient generation-ceiling diagnostics, not deployable controller Pass@1.
- The main positive signal is concentrated in missed failed-long cases; triggered failed-long still has zero new correct candidates under E/F/G seed 0.
- Conditional multi-seed was selected after seed-0 diagnostics and must not be treated as an unbiased benchmark metric.
