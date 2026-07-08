# Dream-Coder Fresh Oracle-Sufficient Diagnostic

Verdict: `second_backbone_fresh_oracle_completed`.

Cases: `37`.
Primary/control pass: `11/37`.
Best simple length policy pass: `11/37`.
Oracle-sufficient canvas pass: `26/37`.
Missed-long oracle recoveries: `3`.
Triggered-long oracle recoveries: `4`.
Short-case regressions vs primary under oracle canvas: `0`.
Average oracle cost sec: `1.2704700535695017`.

Qualitative agreement: mixed: Dream-Coder subset does not cleanly match the LLaDA H200 missed-vs-triggered split

E/F/G actions are not applicable for Dream-Coder because the current repository has no Dream-Coder trace-remasking adapter.

## Stratum Results

| Stratum | Cases | Primary | Simple | Oracle | Canvas Recoverable | Rescue/Noncanvas Limited | Stable Pass | Oracle Regression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `missed_failed_long` | 12 | 0/12 | 0/12 | 3/12 | 3 | 9 | 0 | 0 |
| `triggered_failed_long` | 4 | 0/4 | 0/4 | 4/4 | 4 | 0 | 0 | 0 |
| `medium_near_long_underselection` | 8 | 1/8 | 1/8 | 6/8 | 5 | 2 | 1 | 0 |
| `short_primary_pass_harmable` | 6 | 5/6 | 5/6 | 6/6 | 1 | 0 | 5 | 0 |
| `positive_control_recoverable` | 7 | 5/7 | 5/7 | 7/7 | 2 | 0 | 5 | 0 |

Interpretation: this is second-backbone diagnostic evidence, not model-agnostic confirmation. Oracle-canvas recoverability is strong overall, but the expanded taxonomy remains model-dependent: Dream-Coder recovers all 4 triggered-failed-long proxy cases under oracle canvas, unlike the LLaDA H200 attribution where triggered-long C/E/F/G recoveries were 0.

Compact taxonomy outputs:

- `stratum_summary.csv`
- `stratum_taxonomy.csv`
