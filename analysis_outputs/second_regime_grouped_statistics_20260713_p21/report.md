# P2.1 Grouped Statistics: Official Full-Allowed Second Regime

Input integrity: `True`.

Primary estimand: equal-weight base-task macro accuracy. For each policy and base task, calculate span accuracy within task, then average equally over task groups. Span-micro rates are descriptive only; no row-independent significance test is reported.

## Overall

| Policy | Task-macro accuracy (95% cluster CI) | Span-micro accuracy | Mean sec/span |
|---|---:|---:|---:|
| `control_fixed64` | `0.3110` [`0.2897`, `0.3324`] | `0.3010` | `2.0000` |
| `cal_lite` | `0.2310` [`0.2161`, `0.2450`] | `0.2183` | `3.1976` |
| `oracle_ceiling` | `0.4202` [`0.3964`, `0.4435`] | `0.4741` | `3.1222` |

## Pairwise Cluster Tests

| Comparison | Macro delta | 95% cluster CI | Task wins/losses/ties | Label-swap p |
|---|---:|---:|---:|---:|
| `cal_lite - control_fixed64` | `-0.0800` | [`-0.0972`, `-0.0627`] | `38/97/13` | `0.0001` |
| `oracle_ceiling - control_fixed64` | `0.1092` | [`0.0881`, `0.1305`] | `106/28/14` | `0.0001` |
| `oracle_ceiling - cal_lite` | `0.1892` | [`0.1684`, `0.2099`] | `124/10/14` | `0.0001` |

## Scope

- The 148 base tasks are the inferential clusters; 6707 spans and 20121 policy rows remain descriptive accounting totals.
- `task_paired_outcomes.csv` contains one paired accuracy comparison per task group; `strata_cluster_ci.csv` reports both span and group counts for config, length, and error strata.
- `intersection_8cell.csv` is the control/CAL-lite/oracle outcome intersection. `accuracy_cost_frontier.csv` is paper-ready plotting data.
