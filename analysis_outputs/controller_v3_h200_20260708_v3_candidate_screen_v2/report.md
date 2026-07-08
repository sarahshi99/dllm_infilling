# H200 Controller V3 Candidate Screen

route_decision: `weak_validation_signal_test_sealed`
exploratory_gate_passed: `True`
frozen_test_gate_passed: `False`
test_status: `sealed`

## Family Best Top-k

| Family | Feature | k | Pass | Wins | Losses | Net | Pop Harm Upper95 | Oracle-win Hits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `targeted_missed_long` | `probe_trace_fused` | 15 | 91/127 | 3 | 1 | 2 | 0.036807182572575155 | 5 |
| `two_stage_rejector` | `probe_only` | 5 | 90/127 | 1 | 0 | 1 | 0.023312410428213137 | 2 |
| `oracle_win_distillation` | `probe_only` | 5 | 90/127 | 1 | 0 | 1 | 0.023312410428213137 | 2 |

## Feature Safety

- Forbidden feature check: passed.
- Test labels/features/results were not materialized.
- Family C is diagnostic and cannot alone authorize frozen test.
