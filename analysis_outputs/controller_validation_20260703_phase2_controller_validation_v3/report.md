# Frozen Canvas Controller Validation

final_validation_decision: `sealed`

## Selected Controller

- model: `logistic`
- feature_variant: `probe_only`
- policy_variant: `benefit_only`
- score_threshold: `999.0`
- calibration_harm_upper95: `None`
- validation_pass_rate: `0.7086614173228346`
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
| `control` | 0.7007874015748031 | 0 | 1 | 0.0 | 5.567309581359305 |
| `midcons` | 0.7086614173228346 | 0 | 0 | 0.0 | 11.132317982900037 |
| `route2` | 0.7086614173228346 | 0 | 0 | 0.0 | 5.289454268335592 |
| `v6` | 0.7086614173228346 | 0 | 0 | 0.0 | 4.8113056458386145 |
| `local_cal` | 0.7086614173228346 | 2 | 2 | 0.0 | 3.1551778682041913 |
| `always_expand_16` | 0.48031496062992124 | 9 | 38 | 1.0 | 3.3445903549509812 |
| `always_expand_24` | 0.4566929133858268 | 7 | 39 | 1.0 | 3.516532654335033 |
| `always_expand_32` | 0.4409448818897638 | 6 | 40 | 1.0 | 3.506120371066701 |
| `always_expand_48` | 0.41732283464566927 | 6 | 43 | 1.0 | 3.8649170858855175 |
| `oracle_action_bank_upper_bound` | 0.8188976377952756 | 14 | 0 | 0.8661417322834646 | 3.2755823338369465 |
| `compute_matched_simple_gate_expand32` | 0.7086614173228346 | 0 | 0 | 0.0 | 2.2306961533952667 |

Test remains sealed unless the validation gate is passed and the lock is explicitly advanced to one-time authorization.
