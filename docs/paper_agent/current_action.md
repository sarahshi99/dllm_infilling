# Current Paper-Agent Action

更新时间：2026-07-12 UTC

权威路线：`docs/paper_agent/ccfa_master_roadmap.zh.md`

起点证据：`52f07457a7974fafa69d59914bbc412a2d83e305`

## Action Name

`FOUNDATION-01`: official baseline protocol audit + 6707-row grouped-statistics specification + external-benchmark feasibility。

## 为什么现在做

Phase 5 已完整结束，不存在 infrastructure blocker。四个正式方法尚未实现；M1-D0 fixed proxy 只有正向点估计并在严格 gate 下被 kill。当前更紧急的问题是论文证据地基仍不完整：official baseline 未对齐，6707 rows 只有 148 个独立 base-task groups，且没有非 HumanEval 外部评价。

本动作不改变 central claim，不打开 frozen test，不运行 GPU generation。它把 P1/P2/P4 变成下一轮 Codex 可直接执行的工作包，同时允许 M1--M4 的独立 premise diagnostics 另行并行准备。

## 立即执行的三个独立工作包

### `BASE-PROTOCOL-01`

按顺序审计 official CAL、rho-EOS、DreamOn；LR-DLLM 只复核 code/adapter availability。每个 official repository 固定 commit，并生成统一 matrix：training、model/checkpoint、dataset/split、prompt/FIM format、canvas/length rule、steps、forward count、seed、temperature、evaluator、oracle、wall-clock/GPU/memory，以及可比性等级。

输出：

- `docs/paper_agent/external_baseline_protocol_matrix.zh.md`
- `analysis_outputs/external_baseline_protocol_audit_<timestamp>/summary.json`
- 对每个 baseline 给出 `ready_for_smoke`、`adapter_required`、`training_stratum_only` 或 `blocked_missing_detail`。

不得把 local CAL/CAL-lite 改名为 official CAL，不得根据本地结果修改 official algorithm。

### `STAT-GROUP-01`

只读取 existing full-allowed compact results。统计单位固定为 `task_group=HumanEval/<id>`，当前 allowed group count 应为 `148`。生成：

- control vs cal-lite、control vs oracle 的 base-task grouped bootstrap CI；
- paired wins/losses 和 group-aware permutation test；
- config、length bucket、error type 的 cluster-aware CI；
- accuracy--cost frontier；
- control/cal-lite/oracle help-harm-recoverability intersection table 和 paper-ready figure data。

行级 `n=6707` 只能作为 descriptive span count，不得用于独立显著性。

输出：`analysis_outputs/second_regime_grouped_statistics_<timestamp>/`。

### `EXT-FEAS-01`

审计至少三个非 HumanEval 候选，优先选择具备 repository/file context、可执行 tests 或可靠 exact-match harness、公开许可和可冻结 split 的 benchmark。输出 dataset/evaluator/license/model-context/estimated-cost matrix，并推荐一个最小可执行者。不得把 HumanEval MultiLine 或换 backbone 当作非 HumanEval 外部评价。

输出：`docs/paper_agent/external_benchmark_feasibility.zh.md`。

## 并行规则

以上三个 CPU/read-only 工作包可以并行；并行指独立作业，不使用 subagent。它们不能互相改 protocol。M2/M3/M4 premise diagnostics 可以同时设计，但在各自 experiment brief 冻结前不写实现，也不得与 M1 fusion。

## 后续 GPU

本动作结束后，能适配的 official baselines 进入技术 smoke；smoke 只检查执行正确性，通过后按 full-first 原则运行完整 allowed population。探索实验不再用论文级性能 gate 阻止启动。任何 frozen test、正式方法 fusion 或大规模 training 仍需单独授权。

## 当前结论

- Operational status：`ready_for_foundation_work`。
- Scientific status：`diagnostic_mixed_not_submission_ready`。
- Frozen test：`sealed`，`test_evaluation_count=0`。
- Phase 5 historical result report：`docs/paper_agent/experiments/20260712_phase5_h200_candidate_bank_result.md`。
