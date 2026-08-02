# DreamOn V3-Hard Budgeted

Date: 2026-08-02 UTC
Decision: `iterate_and_execute`
Method: `v3_hard_budgeted`
Role: official-protocol-fidelity repair/control on the exact-three-line development/mechanism population; not held-out test.

## Protocol change

Relative to V2-Hard-v2, this method changes only one decoding mechanism: it restores DreamOn's official per-sample cumulative expansion budget. The budget starts at `64`, is shared across all three sequential hard slots, decreases only after a successfully committed expand, is never reset or refunded, and blocks the expand vocabulary logit before entropy/proposal/position selection when exhausted. The V2 boundary and region-local EOS semantics are unchanged. A canvas repeat with lower remaining budget is recorded as `budget_draining_loop`; only an identical full transition at the same budget is an exact deterministic cycle.

Pinned source: DreamOn commit `8a0a54918412eda9402a327646f7f067f7160ec8`, `eval/generator.py`, `MDMGenerator.__init__` and `batch_generate_with_expand_as_token`; official `max_gen_len=64` is recorded in the frozen protocol.

## Verification

- Fresh focused and legacy suite before post-hoc Full: `109 passed`.
- `py_compile`: passed.
- Shell syntax: passed.
- Source scan: no BoundaryShift, AST repair, retry, or fallback; the shared runner enables nonempty guard only for B, while A protocol/config keeps `nonempty_guard=false`.
- Protocol SHA256: `8a54392fbbd136f243a485baeccbee8016a33d29a381cfdacf45adde69860dea`.
- Cycle5 manifest SHA256: `9d4922c330937e0e685061d00006b58419e4bbcbe8b2a09f26add474ebe2d263`.

## Cycle5

All five old V2 exact-cycle rows completed. Every V3 trajectory matched an isolated V2 control rerun through the old stopping forward, and that control rerun matched the historical V2 terminal state. At the old stop points, remaining budgets were `54`, `46`, `38`, `58`, and `54`, proving the old detector stopped before the official cumulative budget was exhausted. All five later reached budget zero, selected a constrained non-expand action, and completed.

Result: `5/5` completed, `4/5` compiled, `3/5` passed, `0` exact cycles, `0` forward caps, `0` runtime/invariant errors. This is post-hoc mechanism evidence, not an independent performance estimate.

## Pilot30

| Metric | V3-Budgeted |
|---|---:|
| Completed | 30/30 |
| Pass@1 | 14/30 (46.67%) |
| Compile | 23/30 (76.67%) |
| Exact match | 7/30 (23.33%) |
| Task-macro Pass@1 | 49.07% |
| Blank-slot rows | 6/30 |
| Exact cycles / forward caps | 0 / 0 |

Paired against V2-Hard-v2: `3` wins, `0` losses, `11` both pass, `16` both fail, exact McNemar `p=0.25`; compile help/harm `4/0`. The 25 rows that neither hit the old cycle nor exhausted budget had identical completions and scores under A and V2, so the observed differences are isolated to the restored budget mechanism.

Against historical one-shot: `4` wins / `9` losses, net `-5`, McNemar `p=0.2668`, compile help/harm `0/7`. Against historical V1 progressive: `3` wins / `5` losses, net `-2`, McNemar `p=0.7266`, compile help/harm `2/4`. With only nine base-problem clusters, these pilot comparisons are development diagnostics and do not support a significance claim.

Five rows exhausted the budget; all five completed, four compiled, and three passed. Six completed rows still contain a blank hard slot, motivating the separately preregistered B oracle nonempty diagnostic. Residual errors are not explained by cumulative budget or empty-slot termination alone; no causal attribution to progressive conditioning is made without additional controls.

## Gate decision

Engineering gate passed. The only preregistered Full authorization criterion was overall `Pass@1 >= 18/30`; observed Pass@1 was `14/30`. Therefore A Full642 was not authorized in the original A/B execution, and B (`v3_hard_budgeted_nonempty_oracle`) proceeded independently under its own gate.

The user later explicitly requested A Full642 so that A and B could be compared on the complete frozen population. That later run is preserved as a post-hoc development/mechanism diagnostic; it does not retroactively change the failed Pilot30 gate or become a preregistered performance claim.

## Post-hoc Full642

| Metric | V3-Budgeted |
|---|---:|
| Completed | 642/642 |
| Pass@1 | 281/642 (43.77%) |
| Compile | 538/642 (83.80%) |
| Exact match | 97/642 (15.11%) |
| Task-macro Pass@1 | 41.17% |
| Blank-slot rows | 137/642 |
| Cycle / forward cap / runtime / protocol | 0 / 0 / 0 / 0 |

Against one-shot, A has 94 wins / 114 losses, McNemar `p=0.1876`, clustered 95% CI `[-8.74pp, +2.47pp]`, and task-macro delta `-2.78pp`. Against historical V1 progressive, A has 60 wins / 95 losses, McNemar `p=0.00613`, clustered 95% CI `[-9.68pp, -1.39pp]`, and task-macro delta `-5.29pp`.

B versus A is 15 wins / 16 losses, McNemar `p=1.0`, clustered 95% CI `[-2.60pp, +2.09pp]`. B removes all 137 A blank-slot rows but reduces compile from 538 to 471, raises exact match from 97 to 106, and changes overall Pass from 281 to 280. The other 505 rows match A exactly on completion, score, action counters, budget, forwards, and token-forwards.

A used 35,738 forwards, 9,126,663 token-forwards, and 2,702.23 seconds generation wall time. These data support A as the cleaner budget-fidelity control, but A remains below one-shot and V1 on this development/mechanism population.

## Artifacts

- Protocol: `repro_results/dreamon_progressive_v3_hard_budgeted_protocol/`
- Cycle5: `repro_results/dreamon_progressive_v3_hard_budgeted_cycle5/`
- Pilot30: `repro_results/dreamon_progressive_v3_hard_budgeted_pilot30/`
- Post-hoc Full642: `repro_results/dreamon_progressive_v3_hard_budgeted_all642/`
- A/B comparison: `repro_results/dreamon_progressive_v3_hard_budgeted_ab_comparison_all642/`
