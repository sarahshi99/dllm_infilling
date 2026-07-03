# CAL Full Same-Protocol Comparison

cal_full_status: `completed`
cal_run_dir: `/home/shx/projects/dllm_infilling/outputs_clean/full_official_cal_lcas_v3b_accel_gpus10_20260702_resumed_full_20260703_111615`

## Overall

| Run | Rows | Pass | Pass Rate |
|---|---:|---:|---:|
| `cal_full` | 1033 | 774 | 0.7493 |
| `control` | 1033 | 787 | 0.7619 |
| `midcons` | 1033 | 795 | 0.7696 |
| `route2_len32` | 1033 | 801 | 0.7754 |
| `v6_short_override` | 1033 | 802 | 0.7764 |

## Paired Against CAL

| Baseline | Common | CAL Wins | CAL Losses | Tie Pass | Tie Fail |
|---|---:|---:|---:|---:|---:|
| `control` | 1033 | 25 | 38 | 749 | 221 |
| `midcons` | 1033 | 17 | 38 | 757 | 221 |
| `route2_len32` | 1033 | 17 | 44 | 757 | 215 |
| `v6_short_override` | 1033 | 17 | 45 | 757 | 214 |

## Oracle Bucket Pass Rate

| Run | Bucket | Rows | Pass | Rate |
|---|---|---:|---:|---:|
| `cal_full` | `13-16` | 90 | 51 | 0.5667 |
| `cal_full` | `17-24` | 82 | 16 | 0.1951 |
| `cal_full` | `25+` | 31 | 3 | 0.0968 |
| `cal_full` | `9-12` | 232 | 188 | 0.8103 |
| `cal_full` | `<=8` | 598 | 516 | 0.8629 |
| `control` | `13-16` | 90 | 49 | 0.5444 |
| `control` | `17-24` | 82 | 17 | 0.2073 |
| `control` | `25+` | 31 | 5 | 0.1613 |
| `control` | `9-12` | 232 | 179 | 0.7716 |
| `control` | `<=8` | 598 | 537 | 0.8980 |
| `midcons` | `13-16` | 90 | 53 | 0.5889 |
| `midcons` | `17-24` | 82 | 17 | 0.2073 |
| `midcons` | `25+` | 31 | 5 | 0.1613 |
| `midcons` | `9-12` | 232 | 182 | 0.7845 |
| `midcons` | `<=8` | 598 | 538 | 0.8997 |
| `route2_len32` | `13-16` | 90 | 53 | 0.5889 |
| `route2_len32` | `17-24` | 82 | 19 | 0.2317 |
| `route2_len32` | `25+` | 31 | 5 | 0.1613 |
| `route2_len32` | `9-12` | 232 | 184 | 0.7931 |
| `route2_len32` | `<=8` | 598 | 540 | 0.9030 |
| `v6_short_override` | `13-16` | 90 | 53 | 0.5889 |
| `v6_short_override` | `17-24` | 82 | 20 | 0.2439 |
| `v6_short_override` | `25+` | 31 | 5 | 0.1613 |
| `v6_short_override` | `9-12` | 232 | 184 | 0.7931 |
| `v6_short_override` | `<=8` | 598 | 540 | 0.9030 |

## Interpretation Boundary

- This is a same-repository, same-protocol local CAL comparison after the 10-case sanity verdict.
- It is not an external official-code reproduction claim.
- Raw 1033-row JSONL outputs remain in `outputs_clean/`; this directory contains compact, auditable summaries only.
