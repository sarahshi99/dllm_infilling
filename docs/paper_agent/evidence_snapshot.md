# Paper-Agent Evidence Snapshot

Generated from existing local `results.jsonl` files. Raw outputs are not copied here.

## Runs

| Run | Rows | Pass | Rate | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `a6000_control` | 1033 | 787 | 76.19% | 89.80% | 77.16% | 54.44% | 20.73% | 16.13% |
| `midcons` | 1033 | 795 | 76.96% | 89.97% | 78.45% | 58.89% | 20.73% | 16.13% |
| `mid_precision` | 1033 | 787 | 76.19% | 89.80% | 77.16% | 54.44% | 20.73% | 16.13% |
| `true_long` | 1033 | 787 | 76.19% | 89.80% | 77.16% | 54.44% | 20.73% | 16.13% |
| `combined` | 1033 | 787 | 76.19% | 89.80% | 77.16% | 54.44% | 20.73% | 16.13% |

## Pairwise Against A6000 Control

| Candidate | Common Rows | Wins | Losses | Net | Wins By Bucket | Losses By Bucket |
|---|---:|---:|---:|---:|---|---|
| `midcons` | 1033 | 8 | 0 | 8 | `{'<=8': 1, '9-12': 3, '13-16': 4}` | `{}` |
| `mid_precision` | 1033 | 0 | 0 | 0 | `{}` | `{}` |
| `true_long` | 1033 | 0 | 0 | 0 | `{}` | `{}` |
| `combined` | 1033 | 0 | 0 | 0 | `{}` | `{}` |

## Long-Failure Summary For `midcons`

- long_total: `113`
- failed_long_total: `91`
- underselected_failed_long: `90` (98.90%)
- failed_long_base_source: `71`
- failed_long_selected_length_histogram: `{'3': 39, '4': 5, '5': 2, '6': 8, '7': 6, '8': 3, '9': 7, '10': 1, '11': 1, '12': 4, '13': 5, '14': 3, '15': 2, '16': 3, '20': 1, '21': 1}`

## Long-Underestimate Sweep

- evaluated_rules: `16776`
- strict_viable_rules: `0`
- best_rule: `sel<=3|best>=13|gap>=4|ratio>=0.45|raw>=0.4|supp>=0|src=base`
- best_true_long_precision: `35.48%`
- best_failed_long_recall: `36.26%`
- best_short_risk_rate: `40.86%`
- best_current_pass_risk_rate: `27.96%`
