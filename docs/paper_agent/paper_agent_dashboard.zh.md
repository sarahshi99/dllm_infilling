# Paper Agent Dashboard

更新时间：2026-05-31 12:36 CST

## 当前研究目标

将当前 DLLM 代码 infilling 项目推进为有竞争力的 CCF-A 论文：把已有 LCAL/LCAS 经验进展转化为有原则的 length-control 贡献，并配套可复现实验证据。

## 当前 central claim

DLLM 代码 infilling 的 inference-time length control 可以通过区分 medium rescue 与 true-long detection，安全恢复 medium-length under-selection；但是 true-long infilling 仍主要受 length underestimation 支配，可能需要比当前 official-CAL gate family 更强的 length-modeling signal。

## 当前实验方案版本

`v1`：diagnostic-first long-length modeling 方案。当前 A6000 checkpoint 是 `midcons`；下一步 GPU 工作不应继续放宽 official-CAL heuristic，而应先定义更强的 long-tail signal。

## 本次会话已完成

- 阅读了 `AGENTS.md`、近期计划、结果报告、run registry、文献记录，以及核心 clean runner/analysis 代码。
- 创建了 `paper-agent-overnight` 分支。
- 确认 GPU 0,1,2,3 已有其他 Python 任务占用；未启动重型 GPU 实验。
- 初始化了双语 `docs/paper_agent/` 研究跟踪结构。
- 新增经过测试的 evidence snapshot builder，并从本地 raw outputs 重新生成当前 A6000 evidence snapshot。

## 最新结果摘要

- A6000 control：`787/1033 = 76.19%`。
- A6000 `midcons`：`795/1033 = 76.96%`，相对 same-hardware control 为 `+8` wins、`0` losses。
- Long buckets 仍未改善：`17-24 = 20.73%`，`25+ = 16.13%`。
- Offline long-underestimate sweep 没有从当前 result fields 中找到安全 heuristic rule。
- 最新 snapshot：`docs/paper_agent/evidence_snapshot.md`，由 `analysis/build_paper_agent_evidence_snapshot.py` 生成。

## 关键方案调整

- 将 `midcons` 视为真实的 short/medium checkpoint，而不是完整解决方案。
- 在定义更强 signal 前，停止把 GPU 投入当前 official-CAL true-long trigger family。
- 将 trajectory features、learned length classification、DreamOn-style dynamic canvas control 或 LR-DLLM-style length regularization 提升为下一阶段 paper-level 方向。

## 当前最大风险

当前提升幅度较小且 heuristic 色彩较强；除非下一阶段产生有原则的 long-length control 或强 cross-model/protocol-matched validation，否则不足以支撑 CCF-A 贡献。

## 下一步计划

1. 验证、commit，并在 remote 安全时 push paper-agent milestone。
2. 准备只在不打断现有任务时使用 GPU 0,1,2,3 的排队或 smoke-only 实验方案。
3. 在提出任何 SOTA 或 competitive claim 前，重新检查 literature/protocol alignment。
4. 起草下一项 trajectory-based 或 learned long-length signal 的 diagnostic feature spec。

## 需要用户决策的问题

目前没有。只有当下一阶段从 inference-time rescue 转向 training-time 或 fine-tuning-based length regularization 时，才需要用户决策。
