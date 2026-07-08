# Official Second-Regime Bounded Diagnostic

Verdict: `official_second_regime_nontrivial_failures_oracle_recovers_subset_hard_tail_needed`.

Regime: official `HumanEval-MultiLineInfilling`, `HumanEval-RandomSpanInfilling`, and `HumanEval-RandomSpanInfillingLight` from `loubnabnl/humaneval_infilling`.
Frozen controller test: `sealed_not_touched`; manifest frozen-test rows `0`.

## Overall

Cases: `120`.
Control fixed64 pass: `34/120`.
Best deployable cal-lite pass: `36/120`.
Oracle-sufficient canvas pass: `49/120`.
Oracle gain vs control cases: `26`.
Deployable gain vs control cases: `17`.
Deployable harm vs control cases: `15`.
Oracle harm vs control cases: `11`.
Hard-tail candidates flagged in taxonomy: `101`.

## By Stratum

| Source config | Bucket | Cases | Control | Deployable | Oracle | Oracle gain | Hard-tail |
|---|---|---:|---:|---:|---:|---:|---:|
| `HumanEval-MultiLineInfilling` | `extreme` | `10` | `0` | `0` | `0` | `0` | `10` |
| `HumanEval-MultiLineInfilling` | `long` | `10` | `5` | `3` | `9` | `4` | `8` |
| `HumanEval-MultiLineInfilling` | `medium` | `10` | `3` | `5` | `6` | `3` | `7` |
| `HumanEval-MultiLineInfilling` | `short` | `10` | `5` | `10` | `9` | `4` | `5` |
| `HumanEval-RandomSpanInfilling` | `extreme` | `10` | `0` | `0` | `0` | `0` | `10` |
| `HumanEval-RandomSpanInfilling` | `long` | `10` | `1` | `1` | `0` | `0` | `10` |
| `HumanEval-RandomSpanInfilling` | `medium` | `10` | `3` | `1` | `1` | `1` | `9` |
| `HumanEval-RandomSpanInfilling` | `short` | `10` | `5` | `2` | `2` | `0` | `8` |
| `HumanEval-RandomSpanInfillingLight` | `extreme` | `10` | `3` | `0` | `3` | `1` | `10` |
| `HumanEval-RandomSpanInfillingLight` | `long` | `10` | `1` | `4` | `6` | `6` | `10` |
| `HumanEval-RandomSpanInfillingLight` | `medium` | `10` | `3` | `4` | `5` | `4` | `9` |
| `HumanEval-RandomSpanInfillingLight` | `short` | `10` | `5` | `6` | `8` | `3` | `5` |

## Stop-Rule Interpretation

- If all policies are near ceiling, this first pass is weak stress and should stop without stronger claims.
- If control/deployable failures exist and oracle recovers a nonzero subset, do not immediately full-run; use `failure_taxonomy.csv` hard-tail flags to build the next bounded manifest.
- If oracle does not recover failures, official second-regime should be written as a scope boundary for the canvas-rescue claim.

## Compact Outputs

- `manifest.csv`
- `results.csv`
- `stratum_summary.csv`
- `failure_taxonomy.csv`
- `summary.json`
- `report.md`
