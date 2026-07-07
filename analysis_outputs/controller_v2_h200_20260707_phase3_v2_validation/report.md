# H200 Controller V2 Validation

controller_verdict: `risk_certification_limited_test_sealed`
validation_gate_passed: `False`
test_decision: `sealed`

## Variants

| Variant | Feature | Pass | Wins | Losses | Interventions | Pop Harm Upper95 | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| `ordinal_only` | `probe_only` | 89/127 | 0 | 0 | 0 | 0.023312410428213137 | `False` |
| `ordinal_only` | `probe_trace_fused` | 90/127 | 5 | 4 | 41 | 0.07062223955314123 | `False` |
| `pairwise_only` | `probe_only` | 89/127 | 0 | 0 | 0 | 0.023312410428213137 | `False` |
| `pairwise_only` | `probe_trace_fused` | 89/127 | 0 | 0 | 0 | 0.023312410428213137 | `False` |
| `pairwise_only` | `trace_only` | 89/127 | 0 | 0 | 0 | 0.023312410428213137 | `False` |
| `ordinal_pairwise_harm` | `probe_only` | 89/127 | 0 | 0 | 0 | 0.023312410428213137 | `False` |
| `ordinal_pairwise_harm` | `probe_trace_fused` | 89/127 | 0 | 0 | 0 | 0.023312410428213137 | `False` |

Frozen test remains sealed unless validation gate passes.
