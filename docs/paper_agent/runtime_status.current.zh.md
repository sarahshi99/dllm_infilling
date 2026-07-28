# Runtime Status — 2026-07-28 UTC

审计时间：`2026-07-28T19:07:16Z`。Operational decision=`execute_and_monitor`；frozen controller test=`sealed`，`test_evaluation_count=0`。

## M1--M4 当前决定

- M1 RandomSpanLight：完整性 stage-one=`1332/1332`、generic=`148/148`、dependency-cone=`148/148`；576-vs-576 fair delta=`-0.68pp`、help/harm=`0/1`、targeted cone=`17/148`。状态=`reviewed_not_promoted_v0`；历史 5079 仍 `safely_paused_resumable`。
- M2：状态=`reviewed_not_promoted`，结论=`constraints_activated_no_reliable_grouped_advantage`。
- M3：两臂各 `148/148`、256 forwards；birth-death−uniform=`-4.05pp`、help/harm=`8/14`、444 events、mean token-forward=`11,532.11`。状态=`reviewed_not_promoted_v0`。
- M4：mandatory `40,632` MultiLine bank 与 RandomSpanLight visible context 的匹配=`0/164`，所以使用 committed 148-group hash-selected MultiLine-Core manifest。三臂各 `148/148`、missing/duplicate/error=`0/0/0`、best/assembly/repair forwards=`512/512/576`、frozen=`sealed/0`。fair assembly−best=`-12.84pp` CI `[−18.24,−7.43]pp`、help/harm=`0/19`；assembly hash 改变=`83/148` 但结果负向。状态=`reviewed_not_promoted_v0_multilinecore_cross_source`，不进入 296/927/5079、retune 或 fusion。

当前没有 paper primary。M4 与 M1/M3 的数据源不同，因此跨方法 delta 不是 apples-to-apples。

## official CAL

- 标签：**official CAL reproduction on the 4,990-case non-frozen common subset**；CAL=`741e8418`、HumanEval-Infilling=`88062ff`。
- documented runtime overlay 仅恢复上游 README 明示的 evaluator execution line，pinned checkout 未改；SciPy 不在实际 decoder/evaluator import closure。
- smoke=`12/12`，full=`4990/4990`，missing=`0`、canonical error=`0`、failure journal=`0`，完成时间=`2026-07-17T15:29:01Z`。运行期间没有读取或汇报 partial accuracy；截至本审计也没有打开 final outcome analysis。

## GPU / 存储 / 下一步

- H200 NVL snapshot：util=`0%`、used/free=`0/143156 MiB`、temperature=`29C`、ECC=`0`、本项目 GPU process=`0`。M4 full wall=`264.47s`，peak allocation=`15.42 GiB`，无 OOM。
- `/home/shx` 可用约 `2.3T`，inode free=`231,232,594`。
- DreamOn/rho-EOS CPU audit 已完成。下一安全动作：提交/推送 compact evidence；未来 DreamOn smoke 需要新 brief 与 pinned evaluator，rho-EOS 需要单独审计的 faithful FIM protocol。当前没有新 GPU 作业。
