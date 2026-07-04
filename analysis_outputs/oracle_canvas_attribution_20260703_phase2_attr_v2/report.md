# Oracle Canvas Incremental Attribution

This audit separates the pass-level effect of `C_oracle_sufficient` from incremental effects of E/F/G diagnostic refinement actions.

## Main Counts

- hard_case_count: `89`
- positive_control_count: `6`
- C hard recoveries: `29`
- C missed-failed-long recoveries: `29`
- C triggered-failed-long recoveries: `0`
- any E/F/G hard recoveries: `31`
- sufficient canvas effect among E/F/G hard recoveries: `29`
- refinement incremental effect among E/F/G hard recoveries: `2`

## Action Increment Over C

| Action | Hard Pass | Missed Pass | Triggered Pass | Incremental Hard Over C | Hash Changed vs C | Correctness Changed vs C |
|---|---:|---:|---:|---:|---:|---:|
| `E_oracle_sufficient_no_early_commit` | 31 | 31 | 0 | 2 | 13 | 2 |
| `F_oracle_sufficient_trace_remask` | 30 | 30 | 0 | 1 | 7 | 1 |
| `G_oracle_sufficient_trace_span_remask` | 30 | 30 | 0 | 1 | 7 | 2 |

## Interpretation

- `sufficient_canvas_effect_c_passed` counts cases where the oracle-sufficient C candidate already passes.
- `refinement_incremental_effect_c_failed_but_efg_passed` counts hard cases where C fails but at least one of E/F/G passes.
- Hash changes and correctness changes are reported separately; a changed candidate is not treated as a pass-level canvas effect.
