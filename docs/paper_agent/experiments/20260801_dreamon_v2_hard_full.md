# DreamOn V2-Hard Full Result

Timestamp: 2026-08-01 UTC

Method label: `oracle exact-three-line structural diagnostic`

Population: frozen 642-row / 115-base-problem development/mechanism population. This is not a held-out test and not an unknown-line-count aggregate.

## Completeness

- predictions: 642 unique rows
- scores: 642 unique rows
- manifest hash: `aab2ea784635e7827c5851bcd5a7ccdc3437fdb4be45f185aa012e504b92f7bc`
- config hash: `8b34ade47cb6e4ea303252a55529c740d2ff3eb930a55edf2a88bc548b0b1668`
- runner commit: `42054755bb7934d085560c176bdbae2b1f2cdae5`
- disallowed protocol violations: 0
- runtime errors: 0
- silent unresolved acceptance: 0
- post-hoc truncation: 0
- explicit 256-forward terminal failures: 421

## Result

| Metric | V2-Hard |
|---|---:|
| Pass@1 | `7/642 = 1.09%` |
| Compile | `46/642 = 7.17%` |
| Exact match | `3/642 = 0.47%` |
| Task-macro Pass@1 | `1.41%` |
| Mean forwards | `188.64` |
| Mean token-forwards | `51,129.05` |
| Mean wall time | `16.50 s` |
| Max per-process CUDA allocation | `15,850,694,656` bytes |

Paired against the historical one-shot baseline: 2 wins, 296 losses, 5 both pass, 339 both fail; exact McNemar `p=1.75e-85`.

Paired against historical V1 progressive: 2 wins, 311 losses, 5 both pass, 324 both fail; exact McNemar `p=5.89e-90`. This is a multifactor comparison because V1 also differs in canvas construction, activation, and post-hoc truncation.

## Interpretation

V2-Hard is a strong negative result. Future-mask visibility and strict local updates do eliminate V1's wrapper truncation mechanism, but the deterministic selected-position birth/death process frequently enters expand/delete cycles. `421/642` rows exhaust the 256-forward cap without a legal completion. The result weakens the hypothesis that exact-three-line hard slots alone provide a stable progressive DreamOn decoder.

It does not isolate whether open-tail capacity or sequential freezing is responsible. Those questions remain assigned to the already frozen `v2_opentail` and `joint_opentail` comparisons.

Raw result directory: `repro_results/dreamon_progressive_v2_hard_all642/`.
