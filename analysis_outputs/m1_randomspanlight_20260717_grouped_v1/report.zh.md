# M1 Abductive Program-State Bridge / RandomSpanLight

Primary estimand: equal-weight base-task macro accuracy with task_group as the cluster. Span-micro accuracy is descriptive only.

| Method | Task macro (95% cluster CI) | Span micro | Standalone forwards |
|---|---:|---:|---:|
| `fixed64_seed0` | `0.2568` [`0.1892`, `0.3311`] | `0.2568` | `64.0` |
| `ordinary_confidence_best_of_grid` | `0.2770` [`0.2095`, `0.3514`] | `0.2770` | `512.0` |
| `equal_compute_generic_remask` | `0.2432` [`0.1757`, `0.3108`] | `0.2432` | `576.0` |
| `m1_score_only_abductive_selector` | `0.2568` [`0.1892`, `0.3311`] | `0.2568` | `512.0` |
| `m1_dependency_cone_full` | `0.2365` [`0.1689`, `0.3041`] | `0.2365` | `576.0` |
| `oracle_ceiling_offline_only` | `0.4932` [`0.4122`, `0.5743`] | `0.4932` | `64.0` |

| Comparison | Task macro delta | wins/losses/ties | help/harm | label-swap p |
|---|---:|---:|---:|---:|
| `m1_dependency_cone_full - equal_compute_generic_remask` | `-0.0068` | `0/1/147` | `0/1` | `1.0000` |
| `m1_score_only_abductive_selector - ordinary_confidence_best_of_grid` | `-0.0203` | `13/16/119` | `13/16` | `0.7071` |
| `m1_dependency_cone_full - m1_score_only_abductive_selector` | `-0.0203` | `16/19/113` | `16/19` | `0.7384` |
| `ordinary_confidence_best_of_grid - fixed64_seed0` | `0.0203` | `19/16/113` | `19/16` | `0.7416` |
| `equal_compute_generic_remask - fixed64_seed0` | `-0.0135` | `2/4/142` | `2/4` | `0.6913` |
| `m1_score_only_abductive_selector - fixed64_seed0` | `0.0000` | `20/20/108` | `20/20` | `1.0000` |
| `m1_dependency_cone_full - fixed64_seed0` | `-0.0203` | `1/4/143` | `1/4` | `0.3759` |
| `oracle_ceiling_offline_only - fixed64_seed0` | `0.2365` | `49/14/85` | `49/14` | `0.0001` |
