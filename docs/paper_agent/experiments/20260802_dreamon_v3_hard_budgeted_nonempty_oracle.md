# DreamOn V3-Hard Budgeted + Nonempty Oracle

Date: 2026-08-02 UTC

Method: `v3_hard_budgeted_nonempty_oracle`. Role: oracle structural diagnostic; not deployable; not held-out.

The method is derived from committed A and changes only hard-slot termination legality. Candidates are simulated on a cloned canvas. A candidate is rejected only when it completes the active hard slot with whitespace-only decoded text, then selection is recomputed from the same cached logits without another model forward. No token insertion, retry, fallback, or reference/test access is used.

Verification: 121 tests passed. Protocol SHA256 `efdfe1166ce9c67bb1555fb48750b4712028c979d477e0877ff0d46ec804b977`; Affected6 SHA256 `cf17ba2d4685b4fb33db86d46b0408755846f4374f515fd59b2754698f3be9ae`.

Pilot30 completed 30/30 with 18 Pass, 25 compile, 8 exact, and zero blank/terminal failures. The guard triggered on six rows; B vs A was 4 wins / 0 losses, while the 24 inactive rows exactly matched A. The preregistered Full threshold was met exactly.

## Full642

| Metric | Result |
|---|---:|
| Completed | 642/642 |
| Pass@1 | 280/642 (43.61%) |
| Compile | 471/642 (73.36%) |
| Exact match | 106/642 (16.51%) |
| Task-macro Pass@1 | 42.46% |
| Blank / cycle / forward cap / runtime / protocol | 0 / 0 / 0 / 0 / 0 |

Paired versus one-shot: 94 wins / 115 losses, McNemar `p=0.1664`, clustered 95% CI `[-0.0858, 0.0237]`, task-macro delta `-1.49pp`. Versus historical V1: 66 wins / 102 losses, McNemar `p=0.00675`, clustered 95% CI `[-0.1017, -0.0103]`, task-macro delta `-4.00pp`. Versus V2-Hard-v1: 274 wins / 1 loss, but this is a multifactor comparison against a decoder/protocol failure diagnostic.

A Full was later run by explicit user authorization after A failed its original Pilot30 performance gate. This post-hoc mechanism comparison gives B versus A: 15 wins / 16 losses / 265 both pass / 346 both fail, McNemar `p=1.0`, row bootstrap 95% CI `[-1.87pp, +1.56pp]`, clustered 95% CI `[-2.60pp, +2.09pp]`. B compile help/harm is `6/73`; exact-match help/harm is `9/0`. B task-macro is 1.29pp above A despite row-level Pass being one lower.

Mechanism/cost: guard triggered on exactly the 137 rows where A produced at least one blank slot. The 505 inactive rows exactly match A. On the affected subgroup, A has 27 Pass / 120 compile while B has 26 Pass / 53 compile. B records 9009 candidate rejections and no no-valid-action terminal. Budget exhausted on 217 rows. B uses 44,067 forwards and 11,412,369 token-forwards, respectively 23.31% and 25.04% above A; generation wall time is 4,061.30 seconds and peak CUDA memory is 15,819,467,264 bytes.

Failures comprise 171 compile failures and 191 compiled functional failures. These are residual errors not explained by cumulative budget or empty-slot termination. Boundary diagnostics find resolved discard on 356/642 rows; 51/164 complete discarded texts exactly match the oracle next line, but BoundaryShift was not implemented.

Decision: `reframe`. Budget fidelity repairs premature cycle classification, but the oracle nonempty guard does not improve Full Pass@1, substantially harms compilation, and increases compute. It improves surface exact match without functional gain. Stop this experiment family pending independent review.

Artifacts: `repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_{protocol,affected6,pilot30,all642}/` and `repro_results/dreamon_progressive_v3_hard_budgeted_ab_comparison_all642/`.
