# M4 Semantic Particle Assembly V0

Primary estimand: equal-weight base-task macro accuracy with task_group as the cluster. Span-micro accuracy is descriptive only.

| Method | Task macro (95% cluster CI) | Span micro | Standalone forwards |
|---|---:|---:|---:|
| `m4_best_single_particle` | `0.2973` [`0.2230`, `0.3716`] | `0.2973` | `512.0` |
| `m4_assembly_without_repair` | `0.1689` [`0.1081`, `0.2297`] | `0.1689` | `512.0` |
| `m4_assembly_with_repair` | `0.1419` [`0.0878`, `0.1959`] | `0.1419` | `576.0` |

| Comparison | Task macro delta | wins/losses/ties | help/harm | label-swap p |
|---|---:|---:|---:|---:|
| `m4_assembly_without_repair - m4_best_single_particle` | `-0.1284` | `0/19/129` | `0/19` | `0.0001` |
| `m4_assembly_with_repair - m4_best_single_particle` | `-0.1554` | `11/34/103` | `11/34` | `0.0002` |
| `m4_assembly_with_repair - m4_assembly_without_repair` | `-0.0270` | `11/15/122` | `11/15` | `0.5610` |
