# Probe-Curve Strict-Split Score Audit

该 audit 基于已有本地 `results.jsonl` 生成；没有启动 GPU job。

## Split Discipline

- split_discipline：`deterministic_sha256_task_id_folds_train_thresholds_only`
- folds：`5`
- thresholds 只在 train folds 上选择，并在 held-out folds 上评估。

## Labels And Features

- rows：`1033`
- feature_count：`24`
- true_long：`113`
- failed_long：`91`
- short：`598`
- passed：`795`

## Aggregate Held-Out Result

- strict_heldout_pass：`False`
- trigger_count：`63`
- true_long_precision：`47.62%`
- failed_long_recall：`32.97%`
- short_risk_rate：`22.22%`
- current_pass_risk_rate：`7.94%`

## Fold Results

| Fold | Train | Held-Out | Strict Train Pass | Threshold | Held-Out Triggers | Precision | Failed-Long Recall | Short Risk | Current-Pass Risk |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 824 | 209 | `False` | 32.1702 | 17 | 58.82% | 52.63% | 23.53% | 5.88% |
| 1 | 824 | 209 | `False` | 38.3904 | 17 | 29.41% | 31.25% | 17.65% | 11.76% |
| 2 | 835 | 198 | `True` | 53.713 | 1 | 0.00% | 0.00% | 100.00% | 100.00% |
| 3 | 827 | 206 | `False` | 32.212 | 19 | 47.37% | 56.25% | 21.05% | 0.00% |
| 4 | 822 | 211 | `False` | 35.6871 | 9 | 66.67% | 28.57% | 22.22% | 11.11% |

## Interpretation

该 audit 检验一个简单 multivariate probe-curve score 在 strict split discipline 下是否具有 offline safety signal。Positive result 需要 aggregate held-out score 满足 paper-agent plan 中使用的同一 safety posture：保留足够 failed-long triggers、达到至少 `60%` true-long precision，并且 short-risk 不超过 `5%`。本次结果没有通过该 gate：short-risk 为 `22.22%`，远高于 `5%`，且 true-long precision 只有 `47.62%`。因此它是一个 negative diagnostic result；它不支持启动 GPU smoke run，但进一步支持当前判断：现有 probe-curve fields 的简单线性组合不足以安全解决 true-long under-selection。
