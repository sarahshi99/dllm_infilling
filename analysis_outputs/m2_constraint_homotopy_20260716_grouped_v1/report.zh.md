# M2 Constraint-Homotopy V0

Primary estimand: equal-weight base-task macro accuracy with task_group as the cluster. Span-micro accuracy is descriptive only.

| Method | Task macro (95% cluster CI) | Span micro | Standalone forwards |
|---|---:|---:|---:|
| `m2_vanilla_fixed64` | `0.2568` [`0.1892`, `0.3311`] | `0.2568` | `64.0` |
| `m2_gradual_constraints` | `0.2703` [`0.2027`, `0.3446`] | `0.2703` | `64.0` |
| `m2_abrupt_constraints` | `0.2365` [`0.1689`, `0.3041`] | `0.2365` | `64.0` |

| Comparison | Task macro delta | wins/losses/ties | help/harm | label-swap p |
|---|---:|---:|---:|---:|
| `m2_gradual_constraints - m2_vanilla_fixed64` | `0.0135` | `9/7/132` | `9/7` | `0.8113` |
| `m2_abrupt_constraints - m2_vanilla_fixed64` | `-0.0203` | `3/6/139` | `3/6` | `0.5108` |
| `m2_gradual_constraints - m2_abrupt_constraints` | `0.0338` | `7/2/139` | `7/2` | `0.1783` |
