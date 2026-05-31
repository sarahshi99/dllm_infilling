# Paper Agent Dashboard

更新时间：2026-05-31 22:26 CST

## 当前研究目标

将当前 DLLM 代码 infilling 项目推进为有竞争力的 CCF-A 论文：把已有 LCAL/LCAS 经验进展转化为有原则的 length-control 贡献，并配套可复现实验证据。

## 当前 central claim

DLLM 代码 infilling 的 inference-time length control 可以通过区分 medium rescue 与 true-long detection，安全恢复 medium-length under-selection；但是 true-long infilling 仍主要受 length underestimation 支配，可能需要比当前 official-CAL gate family 更强的 length-modeling signal。

## 当前实验方案版本

`v3`：probe-curve-first long-length modeling 方案，并采用用户确认的 GPU allocation `2,3`。当前 A6000 checkpoint 是 `midcons`；下一步 GPU 工作不应继续放宽 official-CAL heuristic，也不应使用 single-feature probe threshold，而应先定义更强的 long-tail signal。

## 本次会话已完成

- 阅读了 `AGENTS.md`、近期计划、结果报告、run registry、文献记录，以及核心 clean runner/analysis 代码。
- 创建了 `paper-agent-overnight` 分支。
- 确认 GPU 0,1,2,3 已有其他 Python 任务占用；未启动重型 GPU 实验。
- 初始化了双语 `docs/paper_agent/` 研究跟踪结构。
- 新增经过测试的 evidence snapshot builder，并从本地 raw outputs 重新生成当前 A6000 evidence snapshot。
- 新增经过测试的 probe-curve signal audit；它没有找到 strict viable single-feature threshold，并确认当前 outputs 没有保存 stopping traces。
- 使用 `8` 个 focused tests、compile checks、audit regeneration、JSON assertions 和 `git diff --check` 验证了 probe-curve milestone。
- 已进入优雅暂停流程；新增 `docs/paper_agent/pause_checkpoint.current.md`，未启动新的研究或 GPU 实验。

## 最新结果摘要

- A6000 control：`787/1033 = 76.19%`。
- A6000 `midcons`：`795/1033 = 76.96%`，相对 same-hardware control 为 `+8` wins、`0` losses。
- Long buckets 仍未改善：`17-24 = 20.73%`，`25+ = 16.13%`。
- Offline long-underestimate sweep 没有从当前 result fields 中找到安全 heuristic rule。
- 最新 snapshot：`docs/paper_agent/evidence_snapshot.md`，由 `analysis/build_paper_agent_evidence_snapshot.py` 生成。
- Probe-curve audit：`4106` 个 thresholds，`0` 个 strict viable；最佳 short-risk 为 `8.70%`，高于 `5%` GPU gate。
- Verification：`tests/test_analyze_probe_curve_long_signals.py`、`tests/test_build_paper_agent_evidence_snapshot.py` 和 `tests/test_diagnose_long_underestimate_policy.py` 一起通过。

## 关键方案调整

- 将 `midcons` 视为真实的 short/medium checkpoint，而不是完整解决方案。
- 在定义更强 signal 前，停止把 GPU 投入当前 official-CAL true-long trigger family。
- 优先从当前 outputs 做 probe-curve multivariate/learned scoring；trajectory diagnostics 需要 trace-enabled smoke run。
- 将 trajectory features、learned length classification、DreamOn-style dynamic canvas control 或 LR-DLLM-style length regularization 提升为下一阶段 paper-level 方向。
- 将未来 GPU 实验限制在卡 `2,3`，并采用等待而非中断已有任务的方式。

## Workflow / Skill Status

| Workflow / Skill | Status | Evidence | Output files | Notes |
|---|---|---|---|---|
| gstack `/office-hours` | completed | `research_design.initial.*.md` 和 `research_design.current.*.md` 包含 research community/user、need、minimum publishable contribution；日志 `2026-05-31 12:39 CST` 记录初始化 paper-agent documents | `docs/paper_agent/research_design.initial.en.md`, `docs/paper_agent/research_design.initial.zh.md`, `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | 没有单独 CLI transcript；完成依据是产出的 research design docs |
| gstack `/plan-ceo-review` | completed | `research_design.current.en.md` 的 `CEO-Style Stress Review` 覆盖 novelty、importance、reviewer appeal、scope、central claim、weakest assumption 和 CCF-A realism | `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | 结论：当前还不是 CCF-A claim，但有可推进 foothold |
| gstack `/plan-eng-review` | completed | `experiment_plan.current.en.md` 的 `Engineering Review Summary`、datasets、baselines、metrics、ablations/kill criteria/compute budget/reproducibility sections | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | 当前版本为 `v3` |
| Superpowers `brainstorming` | not_started | no evidence found | none | 当前方向由 gstack-style research design 和 experiment plan 承接；下次若改变 spec，应先运行 |
| Superpowers `writing-plans` | completed | 当前上下文显示已读取该 skill；`experiment_plan.current.*.md` 和 history 给出可执行计划 | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | 没有另建 Superpowers plan 文件 |
| Superpowers `systematic-debugging` | not_needed_yet | no evidence found | none | 本次未遇到需要 root-cause debugging 的代码 bug；negative diagnostic result 已按实验结果记录 |
| Superpowers `verification-before-completion` | completed | 日志 `2026-05-31 21:47 CST` 和 `2026-05-31 22:11 CST` 记录 tests、compile、audit regeneration、JSON assertions、`git diff --check` | `docs/paper_agent/overnight_log.en.md`, `docs/paper_agent/overnight_log.zh.md`, `docs/paper_agent/paper_agent_dashboard.zh.md` | 关键验证：`Ran 8 tests` / `OK`，`thresholds=4106 strict_viable=0` |
| Superpowers `requesting-code-review` | blocked | 日志 `2026-05-31 21:47 CST` 记录 no independent Task/subagent reviewer tool visible；使用 local diff review + fresh tests 作为 fallback | `docs/paper_agent/overnight_log.en.md`, `docs/paper_agent/overnight_log.zh.md` | 独立 reviewer 未完成；风险已记录 |

## 当前最大风险

当前提升幅度较小且 heuristic 色彩较强；除非下一阶段产生有原则的 long-length control 或强 cross-model/protocol-matched validation，否则不足以支撑 CCF-A 贡献。

## 下一步计划

1. 在严格 split discipline 下设计 multivariate 或 learned probe-curve score。
2. 在任何 GPU smoke run 前，先运行 CPU-only held-out diagnostics。
3. 在提出任何 SOTA 或 competitive claim 前，重新检查 literature/protocol alignment。
4. 准备一个使用 GPU `2,3` 且只等待、不打断现有任务的 trace-enabled smoke plan。

## 需要用户决策的问题

目前没有。只有当下一阶段从 inference-time rescue 转向 training-time 或 fine-tuning-based length regularization 时，才需要用户决策。

Paused by user request; no further autonomous work should continue until explicitly resumed.
