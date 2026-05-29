## A6000 受控运行

生成时间：2026-05-29 11:48:13

| 运行 | 方法 | Pass | 相对 A6000 control 的变化 | <=8 | 9-12 | 13-16 | 17-24 | 25+ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `a6000_control` | union control：S3 short-safe + bounded repair + long suspicion | `787/1033` `76.19%` | baseline | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | A6000 baseline |
| `midcons` | union + conservative mid rescue，`off11..13`、`delta3..7`、`ratio>=0.8` | `795/1033` `76.96%` | `+8` (8W/0L) | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` | A6000 mid baseline |
| `mid_precision` | mid rescue + support/best-len/short-jump guards | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | candidate |
| `true_long` | true-long rescue，`off>=17`、`delta>=8`、`ratio>=0.85`、`support>=2` | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | candidate |
| `combined` | mid precision + true-long rescue | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | candidate |

### A6000 运行目录

- `a6000_control`：`outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529`
- `midcons`：`outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626`
- `mid_precision`：`outputs_clean/full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517`
- `true_long`：`outputs_clean/full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755`
- `combined`：`outputs_clean/full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756`
