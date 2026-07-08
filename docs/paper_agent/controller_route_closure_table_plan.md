# Controller Route-Closure Table Plan

Updated: 2026-07-08 UTC

Purpose: one paper table that closes the current controller route without opening frozen test or continuing Controller V4.

Status: superseded by real artifacts in `analysis_outputs/controller_route_closure_20260708_v1/` (`route_closure_table.md`, `route_closure_table.csv`, `summary.json`).

## Planned Table

| Controller | Oracle upper bound | Selected policy | Interventions | Validation wins/losses | Validation pass | Harm upper95 | Frozen-test decision | Evidence path |
|---|---:|---|---:|---:|---:|---:|---|---|
| V1 H200 replay | 106/127, 17/0 wins/losses | `probe_only + benefit_only`, threshold `999.0` | 0 | 0/0 | 89/127 | n/a, no nonzero calibration point | sealed | `analysis_outputs/controller_validation_h200_20260707_v1_replay/report.md` |
| V2 H200 validation | 106/127 inherited action-bank upper bound | best nonzero: `ordinal_only + probe_trace_fused`; selected gate policy: none | 41 for best nonzero | 5/4 for best nonzero | 90/127 for best nonzero | 7.06% population upper95 | sealed | `analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation/report.md` |
| V3 H200 candidate screen | 106/127 inherited action-bank upper bound | exploratory: Family A `targeted_missed_long`, `probe_only`, k=5 | 5 | 1/0 | 90/127 | 2.33% population upper95 | sealed; frozen gate failed | `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/report.md` |

## Interpretation Rule

- The table supports a route-closure claim, not a positive controller claim.
- Oracle action-bank headroom exists, but deployable validation policies either select zero interventions, have too much harm, or are too weak for frozen-test authorization.
- Frozen test remains sealed with `test_evaluation_count=0`.
- Do not add Controller V4 unless a new evidence source appears outside repeated validation tuning.
