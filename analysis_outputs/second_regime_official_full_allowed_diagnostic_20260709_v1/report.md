# Official Second-Regime Full Allowed Diagnostic

Verdict: `official_second_regime_full_allowed_mixed_stress_evidence`.

This is the full allowed official second-regime diagnostic: all rows from `HumanEval-MultiLineInfilling`, `HumanEval-RandomSpanInfilling`, and `HumanEval-RandomSpanInfillingLight` are included except HumanEval task groups in the frozen-controller-test split.
Frozen test remains sealed. This is not a controller test, does not use synthetic stress, does not add policies, and does not tune cal-lite after seeing results.
The same three fixed policies are used: control fixed64, best deployable cal-lite, and oracle-sufficient canvas.

## Overall Pass Rates

Cases: `6707`.
Control fixed64: `2019/6707` (`30.10%`).
Best deployable cal-lite: `1464/6707` (`21.83%`).
Oracle-sufficient canvas: `3180/6707` (`47.41%`).
Oracle gain vs control: `1633`.
Deployable help vs control: `620`.
Deployable harm vs control: `1175`.
Oracle harm vs control: `472`.
Rescue/non-canvas-limited cases: `3055`.

## Answers

1. Full allowed official pass rates are listed above and in `summary.json`.
2. The 120-case first-pass estimate should be read as the unbiased diagnostic estimate; `comparison_vs_120_first_pass.csv` records how the full allowed population shifts that estimate.
3. Configs supporting canvas recoverability, ranked by oracle gain vs control:
   - `HumanEval-MultiLineInfilling`: oracle gain `1412/5079` (`27.80%`).
   - `HumanEval-RandomSpanInfilling`: oracle gain `172/1480` (`11.62%`).
   - `HumanEval-RandomSpanInfillingLight`: oracle gain `49/148` (`33.11%`).
4. Configs mainly rescue/non-canvas-limited, ranked by rescue/non-canvas rate:
   - `HumanEval-RandomSpanInfilling`: rescue/non-canvas `924/1480` (`62.43%`).
   - `HumanEval-RandomSpanInfillingLight`: rescue/non-canvas `61/148` (`41.22%`).
   - `HumanEval-MultiLineInfilling`: rescue/non-canvas `2070/5079` (`40.76%`).
5. Deployable cal-lite helps on `620` cases and harms control on `1175` cases; see `harm_summary.csv`.
6. Oracle canvas harms control on `472` cases; see `harm_summary.csv` and `taxonomy_summary.csv`.
7. The full official result is interpreted as `strengthens_mixed_diagnostic_claim` for the central diagnostic claim.
8. Allowed claims: official second-regime contains real canvas-recoverable cases and substantial rescue/non-canvas-limited and harm-risk cases; this strengthens a diagnostic mixed-stress paper. Forbidden claims: deployable controller success, frozen-test performance, synthetic benchmark evidence, or unbiased hard-tail benchmark performance.

## By Config

| Config | Cases | Control | Deployable | Oracle | Oracle gain | Rescue/non-canvas | Deployable harm | Oracle harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `HumanEval-MultiLineInfilling` | `5079` | `1597` | `1206` | `2761` | `1412` | `2070` | `904` | `248` |
| `HumanEval-RandomSpanInfilling` | `1480` | `384` | `227` | `346` | `172` | `924` | `244` | `210` |
| `HumanEval-RandomSpanInfillingLight` | `148` | `38` | `31` | `73` | `49` | `61` | `27` | `14` |

## Comparison Notes

- `comparison_vs_120_first_pass.csv` compares the unbiased 120-case first pass to this full allowed population.
- `comparison_vs_full104_hard_tail.csv` compares the fixed hard-tail stress/taxonomy subset to this full allowed population.
- Full allowed official diagnostic is broader than hard-tail, but it is still not a controller test and does not open frozen test.

## Stop Rule

Stop second-regime GPU work after this full allowed official diagnostic unless a concrete bug is found. Move next to paper evidence consolidation.

## Compact Outputs

- `manifest.csv`
- `results.csv`
- `config_summary.csv`
- `length_bucket_summary.csv`
- `taxonomy_summary.csv`
- `harm_summary.csv`
- `comparison_vs_120_first_pass.csv`
- `comparison_vs_full104_hard_tail.csv`
- `summary.json`
- `report.md`
