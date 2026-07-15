# Runtime Status — 2026-07-15 UTC

权威分支是 `codex/ccfa-execution-sprint-v1`，基线为 `ce416c4670fbb118cc8ac70d2a3ef9315f4912d1`；`afd3c45` integration branch 已 superseded。本记录不包含、也没有读取 M1 的任何部分 pass rate、help/harm 或性能结论：`performance_inspected=false`。

## 历史 MultiLine 共享候选库

- 历史 PID `1195368` / tmux `phase6-multiline-bank` 已退出。
- raw 保持原位：`40,632 / 40,632`，unique=`40,632`，duplicate=`0`，`ok=40,632`、`error=0`。
- 决定：`completed_pending_score_only_analysis_do_not_restart`。不得重启、改写、删除或重建 raw。

## M1 5079-case MultiLine

- 用户 Ctrl+C 后，tmux `phase6-m1-multiline-v3` 不存在，历史 PID `1576214` 不存在；本 agent 没有发送信号。
- 状态：`safely_paused_resumable`，原因 `pre_outcome_compute_budget_reconciliation`，`performance_inspected_before_pause=false`。
- stage-one：`27,217 / 45,711`，unique=`27,217`，duplicate=`0`，`ok=27,217`、`error=0`，最后一行可解析；raw 仍在 `outputs_clean/m1_stage1_multiline_20260713_v3/`。
- generic refinement：`12 / 5,079`，unique=`12`，duplicate=`0`，`ok=12`、`error=0`，最后一行可解析；raw 未动。
- dependency-cone refinement：`12 / 5,079`，unique=`12`，duplicate=`0`，`ok=12`、`error=0`，最后一行可解析；raw 未动。
- 原 interrupted run 没有可用 final manifest，因此 launch snapshot SHA 未伪造；runner source history 包含 `1d9ef3f`。
- 将来仅当某一方法获选进入 5079 时，才可在**原三个目录**上恢复，并显式使用 `--auto-full --selected-method-only-5079` 与 existing-key dedup；绝不从头重跑。

冻结 controller test 继续 `sealed`，`test_evaluation_count=0`。下一项实际 GPU 工作是新独立输出目录中的 M1 RandomSpanLight `12 smoke → 148 full`。
