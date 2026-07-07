# Frozen Canvas Controller Validation

final_validation_decision: `sealed`

## Selected Controller

- model: `logistic`
- feature_variant: `probe_only`
- policy_variant: `benefit_only`
- score_threshold: `999.0`
- calibration_harm_upper95: `None`
- validation_pass_rate: `0.7007874015748031`
- validation wins/losses vs primary: `0/0`
- validation_intervention_coverage: `0.0`

## Validation Gate

- calibration_harm_budget_passed: `False`
- validation_short_bucket_no_net_regression: `True`
- validation_not_below_v6: `True`
- gate_passed: `False`
- failure_message: `VALIDATION FAILURE — TEST REMAINS SEALED`

## Validation Baselines

| Baseline | Pass Rate | Wins | Losses | Coverage | Mean Cost |
|---|---:|---:|---:|---:|---:|
| `control` | 0.6929133858267716 | 0 | 1 | 0.0 | 1.261865018910039 |
| `midcons` | 0.7007874015748031 | 0 | 0 | 0.0 | 1.370363959315 |
| `route2` | 0.7007874015748031 | 0 | 0 | 0.0 | 1.482944338993118 |
| `v6` | 0.7007874015748031 | 0 | 0 | 0.0 | 1.6981044114946835 |
| `local_cal` | 0.7007874015748031 | 2 | 2 | 0.0 | 1.0821220675812426 |
| `always_expand_16` | 0.47244094488188976 | 9 | 38 | 1.0 | 1.1959576426380072 |
| `always_expand_24` | 0.4881889763779528 | 8 | 35 | 1.0 | 1.2301881997112742 |
| `always_expand_32` | 0.44881889763779526 | 9 | 41 | 1.0 | 1.2078909410116094 |
| `always_expand_48` | 0.41732283464566927 | 5 | 41 | 1.0 | 1.3301764765188766 |
| `oracle_action_bank_upper_bound` | 0.8346456692913385 | 17 | 0 | 0.13385826771653545 | 0.8836506943220244 |
| `compute_matched_simple_gate_expand32` | 0.7007874015748031 | 0 | 0 | 0.0 | 0.8260514874017695 |

Test remains sealed unless the validation gate is passed and the lock is explicitly advanced to one-time authorization.
