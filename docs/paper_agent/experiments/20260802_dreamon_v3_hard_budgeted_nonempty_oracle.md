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

Paired versus one-shot: 94 wins / 115 losses, McNemar `p=0.1664`, clustered 95% CI `[-0.0858, 0.0237]`, task-macro delta `-1.49pp`. Versus historical V1: 66 wins / 102 losses, McNemar `p=0.00675`, clustered 95% CI `[-0.1017, -0.0103]`, task-macro delta `-4.00pp`. Versus V2-Hard-v1: 274 wins / 1 loss, but this is a multifactor comparison against a decoder/protocol failure diagnostic. A Full comparison is unavailable because A did not pass its Full gate.

Mechanism/cost: guard triggered on 137 rows (26 Pass, 53 compile), with 9009 candidate rejections and no no-valid-action terminal. Budget exhausted on 217 rows (76 Pass, 122 compile). Mean forwards 68.64; total forwards 44,067; total token-forwards 11,412,369; generation wall time 4,061.30 seconds; peak CUDA memory 15,819,467,264 bytes.

Failures comprise 171 compile failures and 191 compiled functional failures. These are residual errors not explained by cumulative budget or empty-slot termination. Boundary diagnostics find resolved discard on 356/642 rows; 51/164 complete discarded texts exactly match the oracle next line, but BoundaryShift was not implemented.

Decision: `reframe`. Budget fidelity and nonempty termination repair real decoder failures, but the oracle Full remains below one-shot and historical V1. Stop this experiment family pending independent review.

Artifacts: `repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_{protocol,affected6,pilot30,all642}/`.
