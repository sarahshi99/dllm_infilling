# Probe-Curve Strict-Split Score Audit

Generated from existing local `results.jsonl`; no GPU job was launched.

## Split Discipline

- split_discipline: `deterministic_sha256_task_id_folds_train_thresholds_only`
- folds: `5`
- thresholds are selected on train folds only and evaluated on held-out folds.

## Labels And Features

- rows: `1033`
- feature_count: `24`
- true_long: `113`
- failed_long: `91`
- short: `598`
- passed: `795`

## Aggregate Held-Out Result

- strict_heldout_pass: `False`
- trigger_count: `63`
- true_long_precision: `47.62%`
- failed_long_recall: `32.97%`
- short_risk_rate: `22.22%`
- current_pass_risk_rate: `7.94%`

## Fold Results

| Fold | Train | Held-Out | Strict Train Pass | Threshold | Held-Out Triggers | Precision | Failed-Long Recall | Short Risk | Current-Pass Risk |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 824 | 209 | `False` | 32.1702 | 17 | 58.82% | 52.63% | 23.53% | 5.88% |
| 1 | 824 | 209 | `False` | 38.3904 | 17 | 29.41% | 31.25% | 17.65% | 11.76% |
| 2 | 835 | 198 | `True` | 53.713 | 1 | 0.00% | 0.00% | 100.00% | 100.00% |
| 3 | 827 | 206 | `False` | 32.212 | 19 | 47.37% | 56.25% | 21.05% | 0.00% |
| 4 | 822 | 211 | `False` | 35.6871 | 9 | 66.67% | 28.57% | 22.22% | 11.11% |

## Interpretation

This audit tests whether a simple multivariate probe-curve score has an offline safety signal under strict split discipline. A positive result requires the aggregate held-out score to satisfy the same safety posture used by the paper-agent plan: enough failed-long triggers, at least `60%` true-long precision, and at most `5%` short-risk. Passing this diagnostic would only justify a later smoke plan; it would not by itself justify a full GPU run.

This run does not pass that gate: held-out short-risk is `22.22%`, well above `5%`, and true-long precision is `47.62%`. The result is therefore a negative diagnostic result against launching a GPU smoke run from the current probe-curve fields alone.
