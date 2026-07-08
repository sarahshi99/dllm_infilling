# Official Second-Regime Hard-Tail Manifest

Verdict: `official_second_regime_hard_tail_manifest_built_cpu_only`.

Source diagnostic: `analysis_outputs/second_regime_official_diagnostic_20260708_v1`.
Cases: `104`.
Primary hard-tail candidates from taxonomy: `101`.
Oracle harm-risk extras retained: `3`.
Frozen-controller-test rows: `0`.

Selection rule: include every first-pass official row whose taxonomy is not `stable_all_pass`. This follows the stop rule by preparing a hard-tail manifest from failed/control-risk cases and does not run additional GPU.

## Taxonomy Counts

| Taxonomy | Cases |
|---|---:|
| `deployable_and_oracle_recover_control_failure` | `13` |
| `deployable_harm_vs_control` | `15` |
| `oracle_canvas_harm_vs_control` | `3` |
| `oracle_only_canvas_recoverable` | `13` |
| `rescue_limited_or_noncanvas_failure` | `60` |

## Source Counts

| Source config | Cases |
|---|---:|
| `HumanEval-MultiLineInfilling` | `30` |
| `HumanEval-RandomSpanInfilling` | `39` |
| `HumanEval-RandomSpanInfillingLight` | `35` |

## Bucket Counts

| Bucket | Cases |
|---|---:|
| `extreme` | `30` |
| `long` | `28` |
| `medium` | `27` |
| `short` | `19` |

No GPU diagnostic was run for this second manifest. The next decision is whether web/user wants a smaller, bounded hard-tail GPU run or only a paper-scope-boundary analysis.
