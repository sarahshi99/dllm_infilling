# DreamOn V2-Hard Full Result

> Superseded-protocol interpretation: **V2-Hard-v1 decoder/protocol failure diagnostic; not a clean test of progressive-slot efficacy.** The raw 642-row result remains immutable and must not be overwritten. Protocol-v2 separately tests corrected newline-boundary, local-EOS, and exact-cycle decoding.

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

V2-Hard-v1 is a strong decoder/protocol failure diagnostic. Future-mask visibility and strict local updates eliminate V1's wrapper truncation mechanism, but the implementation blanket-banned newline proposals, treated EOS as a one-mask deletion, and lacked exact deterministic-transition cycle detection. The selected-position birth/death process consequently entered repeated expand/delete behavior and `421/642` rows exhausted the 256-forward cap. This result is not a clean test of progressive-slot efficacy and cannot isolate the value of slot conditioning.

The old V2-OpenTail-v1 partial run is abandoned rather than resumed. OpenTail and Joint comparisons are not authorized in the current protocol-v2 round.

Raw result directory: `repro_results/dreamon_progressive_v2_hard_all642/`.
