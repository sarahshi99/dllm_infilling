# A6000 Mid/Long Rescue 报告

更新时间：2026-05-29 Asia/Shanghai

## 状态

四个 A6000 recovery runs 已在完整 `HumanEval-SingleLineInfilling` test split 上完成，每个 run 都有 `1033` 个任务。Raw outputs 仍保留在本地 `outputs_clean/`；紧凑 pairwise summaries 和 generated scoreboard 跟踪在 `analysis_outputs/a6000_midcons_longrescue/`。

## 命令

Recovery launcher 已提交在 `clean_scripts/resume_lcal_a6000_four_policies_offline.sh`：

```bash
bash clean_scripts/resume_lcal_a6000_four_policies_offline.sh
```

Pairwise analysis：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_lcal_pairwise.py \
  --base-results outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529/results.jsonl \
  --candidate-results <candidate-run-dir>/results.jsonl \
  --output-dir analysis_outputs/a6000_midcons_longrescue/<label>_vs_a6000_control
```

Scoreboard generation：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_a6000_scoreboard_section.py
```

## 主结果

| Run | Pass | 相对 A6000 control | <=8 | 9-12 | 13-16 | 17-24 | 25+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6000 control | `787/1033` `76.19%` | baseline | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` |
| midcons | `795/1033` `76.96%` | `+8` (8W/0L) | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` |
| mid_precision | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` |
| true_long | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` |
| combined | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` |

## 解释

`midcons` 是当前 A6000 最佳 checkpoint。它相对 A6000 control 提供 `+8` wins 且 `0` losses。收益集中在 medium lengths：

- `<=8`: +1 net，无 losses。
- `9-12`: +3 net，无 losses。
- `13-16`: +4 net，无 losses。
- `17-24` 和 `25+`: 不变。

八个 wins 的 final source 全部是 `official_mid_rescue`。这在同硬件条件下验证了 conservative medium-length rescue rule。

`mid_precision`、`true_long` 和 `combined` 都不是改进。这些 runs 中，额外 precision guards 移除了全部 mid-rescue triggers，而 true-long branch 触发 0 次。它们是有用的负面证据：当前基于 official-CAL 的 true-long trigger 太保守，不能作为可行 long-tail fix。

## 为什么 True-Long 没有帮助

True-long branch 要求同时满足：official length 至少 17、large delta、high adjusted/raw long ratio、support count、allowed best-long length 和 source gate。

`true_long` 诊断：

- `official_repair_true_long_len_passed`: 17 tasks。
- `official_repair_true_long_delta_passed`: 47 tasks。
- `official_repair_true_long_support_passed`: 2 tasks。
- all gates passed: 0 tasks。
- all gates except support passed: 4 tasks，但四个都是已经 passing 的 short/medium cases。

这意味着放松 support guard 主要会增加 false positives，而不是解决 long bucket。核心问题不同：在 `midcons` long failures (`oracle >= 17`) 中，`90/91` 是 under-selected，`71/91` 最终仍来自 `base`。Official-CAL 经常在 true-long failures 上选择极短长度（`1`、`2`、`6`、`7`、`9`），因此 official-CAL 不足以作为 long trigger。

## 输出路径

- A6000 control: `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529`
- midcons: `outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626`
- mid_precision: `outputs_clean/full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517`
- true_long: `outputs_clean/full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755`
- combined: `outputs_clean/full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756`

## 下一步决策

把 `midcons` 作为当前 short/medium lengths 的 A6000 checkpoint。对 long bucket，停止继续迭代当前 official-CAL true-long trigger，改为设计独立 long-underestimation detector，使用 long-curve evidence 和 failure signatures，同时不牺牲 `<=8` 与 `9-12`。

后续诊断：

- `analysis_outputs/long_underestimate_detector/a6000_midcons/sweep.md`
- `docs/results/long_underestimate_detector_report.md`

离线 sweep 没有从当前 result fields 中找到安全的 heuristic long-underestimation rule。这进一步支持转向 trajectory features、learned length classification 或 literature-style length regularization。
