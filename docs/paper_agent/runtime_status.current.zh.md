# Runtime Status — 2026-07-17 UTC

审计时间：`2026-07-17T10:41:05Z`。Operational decision=`execute_and_monitor`；frozen controller test=`sealed`，`test_evaluation_count=0`。

## 已完成的 M1--M3 决策

- M1 RandomSpanLight 完整性为 stage-one=`1332/1332`、generic=`148/148`、dependency-cone=`148/148`，148 groups、0 missing/duplicate/error。预冻结 fair primary（full − generic，576 vs 576）=`-0.68pp`、help/harm=`0/1`，targeted cone=`17/148`；状态=`reviewed_not_promoted_v0`。历史 5079 M1 仍为 `safely_paused_resumable`，未恢复。
- M2 保持 `constraints_activated_no_reliable_grouped_advantage`，不晋级。
- M3 两臂各 `148/148`、0 missing/duplicate/error、每臂 256 forwards；birth-death − uniform=`-4.05pp`、help/harm=`8/14`。444 个 birth/death events 证明机制激活，但较低 token-forward 不满足 accuracy 或 efficiency promotion；状态=`reviewed_not_promoted_v0`。
- 因而当前没有 paper primary，M1/M2/M3 不进入 296/927/5079，且不做 outcome-driven retune/fusion。

## official CAL：smoke 已过，4,990 full 正在运行

- 标签固定为 **official CAL reproduction on the 4,990-case non-frozen common subset**；CAL=`741e8418`，HumanEval-Infilling=`88062ff`。
- SciPy 未安装，但它只在 pinned upstream `length_bias.py` fitting utility 出现；actual `llada_cal.llada_cal.generate` 与 evaluator path 不导入 SciPy。上游 README 明示要本地恢复被注释的 evaluator `exec` 调用；adapter 使用一行、in-memory documented enablement overlay，pinned checkout 未修改，并记录 source/enabled hash。
- 12-case smoke：`12/12` unique，missing/duplicate/error=`0/0/0`，forward partition、resume、frozen seal 均通过；随后自动进入 full。
- full tmux=`official-cal-primary-sprint-v1`，pane PID=`2578976`，Python PID=`2579423`。审计时 canonical raw=`659/4990`、unique=`659`、all `status=ok`、duplicate=`0`、failure journal=`0`。只检查完整性，不读取/汇报 CAL 部分准确率。
- full t0=`2026-07-17T10:05:14Z` 的 12 rows；t+10 integrity audit=`2026-07-17T10:16:01Z` 的 209 rows。到本审计的真实吞吐约 `1082.85 rows/hour`，ETA 约 `4.00 hours`；这是外部 t0/t+10 计数，规避旧 progress manifest 的 case-timer bug。该 bug 已在本地 commit 修复，未中断本次稳定 full。

## GPU / M4

- 最近 host-side H200 snapshot（`2026-07-17T10:34:15Z`）：utilization=`100%`、used/free=`59317/84454 MiB`、power=`286W`、temperature=`55C`、ECC=`0`；CAL 占用约 `17294 MiB`。
- 仍有其他用户 GPU 进程。不会停止、重启或抢占它们；sandbox-side `nvidia-smi` 偶发 driver communication error，但 host-side audit 成功且 CAL raw 持续增长。
- M4 的 cost/activation analyzer 已预检并提交。非破坏性 tmux supervisor=`ccfa-execution-sprint-supervisor-v1` 正在等待：CAL t+10 后持续增长、free memory>=25GiB、ECC=0、且不再有第三个 GPU research process。当前 compute process count 不满足，因此 M4 尚未启动，也没有创建/覆盖其 raw output。

存储：`/home/shx` 可用约 `2.4T`，可用 inode=`231,998,365`。下一动作是保持 CAL full，待安全槽出现由 supervisor 启动 M4 的 12-case smoke→148 full。
