# 已完成实验分析，2026-05-29

## 范围

本文整合当前 DLLM code infilling 阶段在 A6000 runs 完成后的工作。由于 GPU 当前需要留给其他用户，本阶段不启动新的 GPU 实验；目标是让已完成结果可审阅、可复现，并能支撑下一步研究决策。

## 使用的 Superpowers

- `receiving-code-review`：用于处理外部 review 报告的三个 P2 问题。
- `test-driven-development`：用于 review bugfix；仓库已有 missing pairwise comparison、singular bucket-rate key 和 GPU override behavior 的回归测试。
- `verification-before-completion`：用于在声明 review issue 已处理或实验已完成前做验证。
- `writing-plans`：此前用于 CCF-A roadmap；本文更新该 plan 的执行状态。

## Review 修复状态

| Review item | 当前行为 | 验证 |
|---|---|---|
| 缺失 pairwise comparison 被误标为 `baseline` | 非 control 行现在显示 `comparison missing`；只有真正 control 行显示 `baseline` | `tests/test_build_a6000_scoreboard_section.py` |
| Registry 忽略 singular `oracle_bucket_pass_rate` | Registry 会读取 `oracle_bucket_pass_rates`、`oracle_bucket_pass_rate` 和 `bucket_pass_rates` | `tests/test_build_run_registry.py` |
| wait script 检查一组 GPU，却硬编码在 `0,1` 上启动 | `wait_and_run_lcal_a6000_baselines.sh` 会通过 `CUDA_VISIBLE_DEVICES` 传递 `GPU_IDS` | `tests/test_a6000_launchers.py` |

验证命令：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest \
  tests/test_build_a6000_scoreboard_section.py \
  tests/test_a6000_launchers.py

PYTHONPATH=. /home/shx/miniconda3/envs/dllm_env/bin/python \
  tests/test_build_run_registry.py
```

## 已完成 A6000 Runs

五个 A6000 runs 均有 `1033` 行和 `summary.json`：

| Run | 原始输出目录 | Pass | 相对 A6000 control |
|---|---|---:|---:|
| `a6000_control` | `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529` | `787/1033 = 76.19%` | baseline |
| `midcons` | `outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626` | `795/1033 = 76.96%` | `+8` wins, `0` losses |
| `mid_precision` | `outputs_clean/full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517` | `787/1033 = 76.19%` | `0` |
| `true_long` | `outputs_clean/full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755` | `787/1033 = 76.19%` | `0` |
| `combined` | `outputs_clean/full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756` | `787/1033 = 76.19%` | `0` |

已跟踪的紧凑输出：

- `analysis_outputs/a6000_midcons_longrescue/a6000_scoreboard_section.md`
- `analysis_outputs/a6000_midcons_longrescue/*_vs_a6000_control/summary.json`
- `analysis_outputs/a6000_midcons_longrescue/*_vs_a6000_control/bucket_summary.csv`
- `analysis_outputs/a6000_midcons_longrescue/*_vs_a6000_control/pairwise.csv`
- `docs/results/a6000_midcons_longrescue_report.md`
- `docs/results/long_underestimate_detector_report.md`

## 主要发现

`midcons` 是当前 A6000 checkpoint。它是唯一改善同硬件 control 的 candidate，并且没有 pairwise losses。

Bucket-level effect：

| Oracle bucket | Control pass rate | `midcons` pass rate | Net change |
|---|---:|---:|---:|
| `<=8` | `89.80%` | `89.97%` | `+1` |
| `9-12` | `77.16%` | `78.45%` | `+3` |
| `13-16` | `54.44%` | `58.89%` | `+4` |
| `17-24` | `20.73%` | `20.73%` | `0` |
| `25+` | `16.13%` | `16.13%` | `0` |

八个 `midcons` wins 全部由 `official_mid_rescue` 恢复：

| Task | Oracle len | Control len | `midcons` len |
|---|---:|---:|---:|
| `SingleLineInfilling/HumanEval/139/L2` | 13 | 8 | 13 |
| `SingleLineInfilling/HumanEval/143/L5` | 12 | 9 | 12 |
| `SingleLineInfilling/HumanEval/37/L5` | 9 | 6 | 12 |
| `SingleLineInfilling/HumanEval/39/L3` | 8 | 6 | 13 |
| `SingleLineInfilling/HumanEval/73/L1` | 13 | 9 | 13 |
| `SingleLineInfilling/HumanEval/75/L12` | 11 | 4 | 11 |
| `SingleLineInfilling/HumanEval/80/L3` | 13 | 8 | 11 |
| `SingleLineInfilling/HumanEval/82/L1` | 13 | 6 | 13 |

解释：conservative medium rescue 有用，因为它修复了 under-selected short-to-medium completions；但它不解决 true long infilling。

## 负面证据

`mid_precision`、`true_long` 和 `combined` 不是改进。它们的价值在于诊断：

- 额外 mid precision guards 太保守，移除了有用的 mid rescue 行为；
- true-long rescue 在 safety gates 下触发 0 次；
- combined 继承了两个问题。

`true_long` summary 记录：

- `official_true_long_rescue_trigger_count = 0`；
- `official_long_suspicion_trigger_count = 11`；
- `official_long_suspicion_true_long_precision = 27.27%`；
- `official_repair_true_long_precision = 8.86%`。

这不足以支撑再跑同一家族 long-rescue GPU 实验。

## Long-Tail 诊断

对 A6000 `midcons` run：

- oracle length `>=17`：`113` 个任务；
- passed long tasks：`22`；
- failed long tasks：`91`；
- failed long tasks 中 under-selected：`90/91`；
- failed long tasks 中最终 source 仍为 `base`：`71/91`。

大多数 failed long samples 选择了非常短的长度。Failed long tasks 中最常见 selected lengths：

| Selected length | Count |
|---:|---:|
| 3 | 39 |
| 6 | 8 |
| 9 | 7 |
| 7 | 6 |
| 13 | 5 |
| 4 | 5 |
| 12 | 4 |

离线 long-underestimate sweep 评估了 `16776` 条规则。最佳规则只有 `35.48%` true-long precision、`36.26%` failed-long recall 和 `40.86%` short-risk。没有找到严格可行规则。

决策：停止把 GPU 花在当前 official-CAL true-long gate family 上。下一条可信 long-tail 路线需要更强信号，例如 denoising trajectory features、learned length classifier、DreamOn-style dynamic canvas control 或 LR-DLLM-style length regularization。

## 文献定位

当前结果不应称为 SOTA。

本地 `midcons` 在 LLaDA-8B-Base 上，于本地 `HumanEval-SingleLineInfilling` protocol 达到 `76.96%`。DreamOn 在 DreamCoder/DiffuCoder 上通过 dynamic canvas method 报告了更高的 HumanEval-Infilling single-line pass@1，但 model family、canvas protocol 和 evaluation setup 必须匹配后才能直接比较。LR-DLLM 也高度相关，因为它直接面向 unknown-length generation，但其报告设置尚未与当前本地 protocol 对齐。

当前 paper-positioning claim 应更窄：

> 同硬件证据表明，confidence-curve agreement 可以在不牺牲 short-bucket 的情况下恢复 medium-length DLLM code infilling；但 long-tail infilling 仍由 length underestimation 主导，需要更强的 length modeling 机制。

## 需求追踪

| 讨论中的需求 | 状态 | 查看位置 |
|---|---|---|
| 使用 Git worktrees | 已完成 | `git_workspace/.worktrees/a6000-midcons-longrescue`, `git_workspace/.worktrees/result-archive` |
| 保留 `outputs_clean/` | 已完成 | raw outputs 保留在本地并 symlink 到 worktrees |
| 不提交 raw heavy outputs | 已完成 | 只跟踪 compact reports 和 summaries |
| 写入协作/反驳假设规则 | 已完成 | `AGENTS.md` 与 `AGENTS.zh.md` |
| 解释并使用 spec files | 已完成 | `docs/superpowers/specs/*.md` 是 design contracts；`docs/superpowers/plans/*.md` 是 executable plans |
| 恢复/使用 git metadata | 已完成 | worktrees 挂在分支上并已 push；raw non-git root 仍分离 |
| 在 A6000 上 rerun midcons | 已完成 | 上表中的 `midcons` A6000 run |
| 分析已完成实验 | 已完成 | 本文和相关 reports |
| 文献比较 | 部分完成，已记录 source-checked anchors；不声称 SOTA | `docs/results/literature_sota_notes.md` |
| Cross-model validation | 历史 registry 已完成；新 runs 因当前 no-GPU 指令暂停 | `docs/results/model_generalization_registry.md` |

## 下一步工程计划

在用户明确说明显卡再次可用前，不启动新 GPU jobs。

可以继续安全推进的 CPU-only 工作：

1. 继续完善 result archive 解释和 paper tables。
2. 围绕 medium rescue 与 long-tail failure analysis 起草 method section。
3. 把 long-tail 诊断转化为 trajectory-feature 或 learned length-detector 实验 spec。

下一步 GPU 工作，仅在硬件可用后执行：

1. 对任何新 long detector 先运行 small smoke test。
2. 只在 matched prompt/canvas settings 下重跑 cross-model checks。
3. 只有 protocol fields 对齐后，才与 DreamOn/LR-DLLM 比较。
