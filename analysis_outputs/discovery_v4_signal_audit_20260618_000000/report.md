# Discovery V4 Signal Audit

Decision: `route2_polish_only`

Decision reason: No stable low-risk V4 signal was found beyond Route2 polish.

## Overview

- Rows: `1033`
- True-long rows: `113`
- Baseline failed-long rows: `91`

## Policy Deltas

| Policy | Bucket | Win | Loss | Tie pass | Tie fail | Net | Triggers |
|---|---|---:|---:|---:|---:|---:|---:|
| precision_len32 | all | 6 | 0 | 795 | 232 | 6 | 57 |
| precision_len32 | 13-16 | 0 | 0 | 53 | 37 | 0 | 10 |
| precision_len32 | 17-24 | 2 | 0 | 17 | 63 | 2 | 24 |
| precision_len32 | 25+ | 0 | 0 | 5 | 26 | 0 | 11 |
| precision_len32 | 9-12 | 2 | 0 | 182 | 48 | 2 | 6 |
| precision_len32 | <=8 | 2 | 0 | 538 | 58 | 2 | 6 |
| precision_len24 | all | 5 | 0 | 795 | 233 | 5 | 57 |
| precision_len24 | 13-16 | 2 | 0 | 53 | 35 | 2 | 10 |
| precision_len24 | 17-24 | 1 | 0 | 17 | 64 | 1 | 24 |
| precision_len24 | 25+ | 0 | 0 | 5 | 26 | 0 | 11 |
| precision_len24 | 9-12 | 1 | 0 | 182 | 49 | 1 | 6 |
| precision_len24 | <=8 | 1 | 0 | 538 | 59 | 1 | 6 |
| broad_len24 | all | 7 | 1 | 794 | 231 | 6 | 73 |
| broad_len24 | 13-16 | 2 | 0 | 53 | 35 | 2 | 13 |
| broad_len24 | 17-24 | 2 | 0 | 17 | 63 | 2 | 27 |
| broad_len24 | 25+ | 0 | 0 | 5 | 26 | 0 | 12 |
| broad_len24 | 9-12 | 2 | 0 | 182 | 48 | 2 | 11 |
| broad_len24 | <=8 | 1 | 1 | 537 | 59 | 0 | 10 |

## Top Slices

| Candidate | Family | Decision | Stable folds | Triggers | Missed-long | Rescue-failure | Short risk | Current-pass risk | Precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `broad_len24_triggered_ge_1` | `single` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_rescue_len_ge_24` | `single` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_rescue_len_le_24` | `single` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_final_source_eq_rescue` | `single` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_triggered_ge_1_AND_broad_len24_rescue_len_ge_24` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_triggered_ge_1_AND_broad_len24_rescue_len_le_24` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_triggered_ge_1_AND_broad_len24_final_source_eq_rescue` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_triggered_ge_1_AND_broad_len24_selected_len_ge_24` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_triggered_ge_1_AND_broad_len24_selected_len_ge_19` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_rescue_len_ge_24_AND_broad_len24_final_source_eq_rescue` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_rescue_len_ge_24_AND_broad_len24_selected_len_ge_24` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_rescue_len_ge_24_AND_broad_len24_selected_len_ge_19` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_rescue_len_le_24_AND_broad_len24_final_source_eq_rescue` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_rescue_len_le_24_AND_broad_len24_selected_len_ge_24` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_rescue_len_le_24_AND_broad_len24_selected_len_ge_19` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_final_source_eq_rescue_AND_broad_len24_selected_len_ge_24` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_final_source_eq_rescue_AND_broad_len24_selected_len_ge_19` | `pairwise` | `reject` | `3` | `73` | `4` | `33` | `10` | `1` | `0.534` |
| `broad_len24_triggered_ge_1_AND_baseline_confidence_last_le_0p828125` | `pairwise` | `reject` | `3` | `67` | `3` | `32` | `9` | `0` | `0.552` |
| `broad_len24_rescue_len_ge_24_AND_baseline_confidence_last_le_0p828125` | `pairwise` | `reject` | `3` | `67` | `3` | `32` | `9` | `0` | `0.552` |
| `broad_len24_rescue_len_le_24_AND_baseline_confidence_last_le_0p828125` | `pairwise` | `reject` | `3` | `67` | `3` | `32` | `9` | `0` | `0.552` |
| `broad_len24_final_source_eq_rescue_AND_baseline_confidence_last_le_0p828125` | `pairwise` | `reject` | `3` | `67` | `3` | `32` | `9` | `0` | `0.552` |
| `broad_len24_selected_len_ge_24_AND_baseline_confidence_last_le_0p828125` | `pairwise` | `reject` | `3` | `67` | `3` | `32` | `9` | `0` | `0.552` |
| `broad_len24_selected_len_ge_24` | `single` | `reject` | `3` | `80` | `4` | `33` | `10` | `8` | `0.562` |
| `baseline_confidence_late_slope_ge_0p0390625` | `single` | `reject` | `3` | `8` | `3` | `0` | `2` | `2` | `0.375` |
| `baseline_confidence_late_slope_ge_0p0390625` | `trace_shape` | `reject` | `3` | `8` | `3` | `0` | `2` | `2` | `0.375` |
| `broad_len24_triggered_ge_1_AND_baseline_top1_last_le_0p605469` | `pairwise` | `reject` | `2` | `71` | `3` | `33` | `10` | `1` | `0.535` |
| `broad_len24_triggered_ge_1_AND_precision_len32_trace_top1_last_le_0p605469` | `pairwise` | `reject` | `2` | `71` | `3` | `33` | `10` | `1` | `0.535` |
| `broad_len24_triggered_ge_1_AND_precision_len24_trace_top1_last_le_0p605469` | `pairwise` | `reject` | `2` | `71` | `3` | `33` | `10` | `1` | `0.535` |
| `broad_len24_triggered_ge_1_AND_broad_len24_trace_top1_last_le_0p605469` | `pairwise` | `reject` | `2` | `71` | `3` | `33` | `10` | `1` | `0.535` |
| `broad_len24_rescue_len_ge_24_AND_baseline_top1_last_le_0p605469` | `pairwise` | `reject` | `2` | `71` | `3` | `33` | `10` | `1` | `0.535` |

## Interpretation Guardrails

- This is a CPU-only discovery audit, not a new pass-rate claim.
- Oracle/pass labels are used for offline accounting only.
- Learned or fitted discovery scores remain microscopes unless separately promoted to a learned-controller design.
- A GPU run requires a held-out low-risk candidate and a new action brief.
