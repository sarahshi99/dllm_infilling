# Official Second-Regime Writeup

The full allowed official second-regime diagnostic is the strongest official-regime evidence now available. It includes all non-frozen rows from `HumanEval-MultiLineInfilling`, `HumanEval-RandomSpanInfilling`, and `HumanEval-RandomSpanInfillingLight`: `6707` cases and `0` frozen-controller-test rows.

Overall, control fixed64 passes `2019/6707` (`30.10%`), best deployable cal-lite passes `1464/6707` (`21.83%`), and oracle-sufficient canvas passes `3180/6707` (`47.41%`). Oracle canvas recovers `1633` control failures, but deployable cal-lite harms `1175` control-passing cases and oracle canvas harms `472` control-passing cases. `3055` cases remain rescue/non-canvas-limited.

## Config-Level Analysis

- `HumanEval-MultiLineInfilling`: control `1597/5079` (`31.44%`), deployable `1206/5079` (`23.74%`), oracle `2761/5079` (`54.36%`), oracle gain `1412`, rescue/non-canvas `2070`, deployable harm `904`, oracle harm `248`.
- `HumanEval-RandomSpanInfilling`: control `384/1480` (`25.95%`), deployable `227/1480` (`15.34%`), oracle `346/1480` (`23.38%`), oracle gain `172`, rescue/non-canvas `924`, deployable harm `244`, oracle harm `210`.
- `HumanEval-RandomSpanInfillingLight`: control `38/148` (`25.68%`), deployable `31/148` (`20.95%`), oracle `73/148` (`49.32%`), oracle gain `49`, rescue/non-canvas `61`, deployable harm `27`, oracle harm `14`.

MultiLine supplies most absolute canvas recoveries. RandomSpanLight is small but has strong oracle gain. RandomSpan is the clearest scope boundary: its rescue/non-canvas-limited rate is highest, so random spans should be discussed as stressing semantic/rescue adequacy beyond canvas length.

## Comparison To First-Pass And Hard-Tail

The 120-case first-pass remains the preregistered/unbiased official diagnostic estimate: control `34/120`, deployable `36/120`, oracle `49/120`, oracle gain `26`.
The full allowed population strengthens rather than reverses that estimate: control remains low (`30.10%`), oracle is materially higher (`47.41%`), and deployable cal-lite underperforms control.
The full104 hard-tail is a post-first-pass fixed stress/taxonomy set, not an unbiased benchmark aggregate: control `18/104`, deployable `20/104`, oracle `33/104`, genuine canvas-recoverable `26`, rescue/non-canvas `60`.

## Allowed Claims

- Official second-regime contains real canvas-recoverable failures.
- Full allowed official population strengthens a mixed diagnostic paper.
- RandomSpan and extreme-length cases expose rescue/non-canvas limitations and harm risk.
- Deployable cal-lite is a scope-boundary result, not a successful controller.

## Forbidden Claims

- Do not claim deployable controller success.
- Do not claim frozen-test performance.
- Do not treat hard-tail full104 as an unbiased official benchmark aggregate.
- Do not hide oracle/deployable harm cases.
