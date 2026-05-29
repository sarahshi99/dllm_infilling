# A6000 Midcons Longrescue 实施计划

> **给 agentic worker：** 英文版是完整可执行规格，包含大量代码块和精确命令；本中文版本翻译研究意图、任务结构、关键文件、运行方式和验证逻辑。执行时保留英文版中的代码/命令原文。

**目标：** 构建可复现的 A6000 实验路径，重新运行 union control 与 `midcons`，并评估 mid-rescue precision 和 true-long rescue candidates。

**架构：** 以现有 LCAL official bounded repair runner 作为中心执行路径；新增小型 A6000 run scripts、聚焦的 decision-policy 参数和紧凑 pairwise analysis 工具，使每个 candidate 都与同一个 A6000 control 比较。

**技术栈：** Python 3、bash、JSONL results、现有 `expvision_dllm_clean` package、现有 `clean_scripts` runners。

---

## 文件结构

- 创建 `clean_scripts/run_lcal_a6000_baselines.sh`：运行 A6000 control 与 `midcons`。
- 创建 `analysis/analyze_lcal_pairwise.py`：按 pass/fail、oracle bucket、selected length transition 和 final source 比较两个 `results.jsonl`。
- 修改 `clean_scripts/run_lcal_official_bounded_repair.py` 或新增 A6000 v2 runner：暴露 mid-rescue precision guards 与 true-long policy flags。
- 创建 `clean_scripts/run_lcal_a6000_candidates.sh`：启动 `mid_precision`、`true_long` 和 `combined` candidates。
- 修改 `analysis_outputs/experiment_scoreboard.md`：记录 A6000 同硬件结果。

## Task 1：新增 A6000 Baseline Run Script

新增 bash launcher，默认使用 `GPU_IDS=0,1`、`TOKENIZERS_PARALLELISM=false`，并分别运行：

- A6000 union control；
- A6000 `midcons`。

验证：

- `bash -n clean_scripts/run_lcal_a6000_baselines.sh`
- 确认脚本不硬编码不可覆盖的 GPU。

## Task 2：新增 Pairwise LCAL Analysis Utility

新增 `analysis/analyze_lcal_pairwise.py`，读取 base/candidate `results.jsonl`，按 `task_id` 对齐，输出：

- `summary.json`
- `bucket_summary.json`
- `bucket_summary.csv`
- `pairwise.csv`

核心指标：

- common 样本数；
- candidate wins/losses/net；
- oracle bucket 维度 pass rate；
- selected length 与 final source 的变化。

验证：

- 构造小 JSONL fixture，确认 wins/losses 和 bucket summaries 正确。
- 在真实 A6000 outputs 上运行时，`common` 应为 `1033`。

## Task 3：新增 Mid-Rescue Precision Guards

在 runner 中加入更精确的 mid-rescue 条件，用于减少 false positives：

- 限制 official length 上界；
- 限制 delta；
- 引入 support/best-long/short-jump veto；
- 记录触发数、pass rate、oracle bucket histogram 和 direct loss 风险。

设计目的：保留 `midcons` 的 medium-length wins，同时减少 short/mid false positives。

## Task 4：新增 True-Long Rescue Policy Flags

增加与 mid rescue 分离的 true-long branch，目标是 long buckets：

- official length 高；
- delta 大；
- long-ratio 强；
- support count 足够；
- source gate 保护 short buckets。

该 branch 初期应被视为 candidate/diagnostic，不应在未验证前成为默认策略。

## Task 5：新增 Candidate Run Script

新增 A6000 candidates launcher，运行：

- `mid_precision`
- `true_long`
- `combined`

每个 run 都必须写入独立 experiment name 和 output directory，避免覆盖历史结果。

## Task 6：运行并比较 A6000 Full Experiments

执行顺序：

1. 运行 A6000 control。
2. 运行 `midcons`。
3. 运行 `mid_precision`、`true_long`、`combined`。
4. 用 `analysis/analyze_lcal_pairwise.py` 对每个 candidate 做同硬件 pairwise comparison。
5. 更新 `analysis_outputs/experiment_scoreboard.md` 和相关报告。

## A6000 受控运行结果

本计划已执行完成后的结果：

| Run | Pass | 相对 A6000 control | 结论 |
|---|---:|---:|---|
| `a6000_control` | `787/1033 = 76.19%` | baseline | 同硬件 control |
| `midcons` | `795/1033 = 76.96%` | `+8` wins, `0` losses | 当前 A6000 best |
| `mid_precision` | `787/1033 = 76.19%` | `0` | guards 过严 |
| `true_long` | `787/1033 = 76.19%` | `0` | true-long trigger 未触发 |
| `combined` | `787/1033 = 76.19%` | `0` | 继承上述问题 |

主要结论：`midcons` 是有效的 medium rescue；当前 official-CAL true-long 方向是负面证据，下一步应改用独立 long-underestimation detector。
