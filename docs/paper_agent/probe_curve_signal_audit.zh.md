# Probe-Curve Long-Signal Audit

本报告由已有本地 `results.jsonl` 生成；没有启动 GPU job。

## Availability

- rows：`1033`
- rows_with_probe_curve_features：`1033`
- rows_with_stopping_trace：`0`

## Labels

- true_long：`113`
- failed_long：`91`
- short：`598`
- passed：`795`

## Threshold Sweep

- evaluated_thresholds：`4106`
- strict_viable_thresholds：`0`

| Feature | Direction | Threshold | Triggers | Precision | Failed-Long Recall | Short Risk | Current-Pass Risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| `long_score_max` | `<=` | `0.229253` | 46 | 63.04% | 31.87% | 8.70% | 2.17% |
| `long_score_max` | `<=` | `0.237278` | 51 | 60.78% | 34.07% | 11.76% | 5.88% |
| `long_raw_max` | `<=` | `0.198242` | 52 | 59.62% | 34.07% | 11.54% | 5.77% |
| `long_raw_max` | `<=` | `0.192383` | 48 | 60.42% | 31.87% | 10.42% | 4.17% |
| `long_score_avg` | `<=` | `0.227948` | 50 | 60.00% | 32.97% | 12.00% | 6.00% |
| `long_raw_max` | `<=` | `0.188477` | 44 | 61.36% | 29.67% | 9.09% | 2.27% |
| `long_gap_max` | `<=` | `0.114746` | 56 | 58.93% | 36.26% | 16.07% | 10.71% |
| `long_raw_max` | `<=` | `0.203125` | 56 | 57.14% | 35.16% | 14.29% | 8.93% |
| `best_raw_score` | `<=` | `0.648438` | 85 | 48.24% | 45.05% | 22.35% | 10.59% |
| `medium_score_max` | `<=` | `0.314198` | 48 | 58.33% | 30.77% | 12.50% | 4.17% |
| `long_score_max` | `<=` | `0.244291` | 57 | 56.14% | 35.16% | 15.79% | 10.53% |
| `long_gap_max` | `<=` | `0.108887` | 52 | 59.62% | 34.07% | 17.31% | 11.54% |
| `short_score_avg` | `<=` | `0.511695` | 73 | 50.68% | 40.66% | 21.92% | 8.22% |
| `best_raw_score` | `<=` | `0.652344` | 86 | 47.67% | 45.05% | 23.26% | 10.47% |
| `medium_raw_max` | `<=` | `0.357422` | 70 | 51.43% | 39.56% | 20.00% | 10.00% |

## Interpretation

当前 full-run artifact 支持 probe-curve diagnostics，但不支持 trajectory diagnostics，因为 `rows_with_stopping_trace = 0`。单变量 probe-curve thresholds 没有通过 strict viability gate。最佳阈值接近 true-long precision gate，但 short-risk 超过 `5%`，因此不应直接进入 GPU full run。
