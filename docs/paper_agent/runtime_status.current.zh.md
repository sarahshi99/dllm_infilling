# Runtime Status — 2026-07-16 UTC

审计时间：`2026-07-16T03:56:01Z`。权威分支为 `codex/ccfa-execution-sprint-v1`（审计时 HEAD `9694b29`，base `ce416c4670fbb118cc8ac70d2a3ef9315f4912d1`）。`performance_inspected=false`：本记录只读取 JSONL 的行数、候选键、`status`、末行可解析性、日志错误标记和硬件遥测；没有读取或报告新 M1/M2 的 pass rate、help/harm 或其它性能结论。

## 历史工作保持不动

- 旧共享 MultiLine candidate bank：`40632/40632`、unique=`40632`、duplicate=`0`、`ok=40632`、`error=0`；已完成 score-only historical precursor 分析，仍不得重启、重建、改写或删除。
- 历史 5079-case M1：PID `1576214`/tmux `phase6-m1-multiline-v3` 均不存在，状态为 `safely_paused_resumable`。stage-one=`27217/45711`，generic=`12/5079`，dependency-cone=`12/5079`；三份原 raw 保留、末行可解析、无 duplicate/error，且从未读取部分性能。未来只有胜出方法才可在**原目录**使用 `--auto-full --selected-method-only-5079` 加 existing-key dedup 恢复。
- Frozen controller test：`sealed`，`test_evaluation_count=0`。

## 新 M1 RandomSpanLight

- tmux：`m1-randomspanlight-sprint-v1`；pane PID `2251036`，真实 Python PID `2251038`，状态 `Rl+`、elapsed `00:28:56`。
- 独立目录：stage1 `outputs_clean/m1_randomspanlight_20260715_v1/stage1/`，generic `.../generic/`，dependency-cone `.../dependency_cone/`；日志 `logs/paper_agent/20260716_m1_randomspanlight_sprint_v1.log`。
- 只读完整性：stage-one `70/1332`（unique `70`、duplicate `0`、`ok=70`、`error=0`）；generic `0/148`；M1 dependency-cone `0/148`。当前仍在 12-case smoke 的 stage-one 段；各 raw 最后一行可解析。
- 吞吐：launch t0 目录不存在，故 t0=`0`；启动约 13.25 分钟时为 `33` stage-one rows，M2 启动基线也为 `33`。当前总计 `70/1628`，启动以来约 `145.2 rows/hour`，M2 并发窗口约 `158.2 rows/hour`。按当前 row-equivalent 速度，stage-one ETA 约 `8.69h`，完整 148-route ETA 约 `10.73h`。共享 GPU 的 wall-clock 仅标为 contention measurement。

## 新 M2 Constraint-Homotopy

- 新 M1 的 t+10 审计确认 M1 有行增长、空闲显存 ≥25 GiB、OOM/ECC=0 后启动；不等待新 M1 结束，也没有触碰旧 5079 M1。
- tmux：`m2-constraint-homotopy-sprint-v1`；真实 Python PID `2254866`，状态 `Rsl+`、elapsed `00:14:02`。
- 独立目录：vanilla `outputs_clean/m2_vanilla_randomspanlight_20260715_sprint_v1/`，gradual `...m2_gradual.../`，abrupt `...m2_abrupt.../`；日志 `logs/paper_agent/20260715_m2_constraint_homotopy_sprint_v1.log`。
- 启动后 t0=`0`，当前 vanilla=`12/148`、gradual=`12/148`、abrupt=`6/148`，总计 `30/444`；每份 raw unique、0 duplicate、`status=ok`、末行可解析。约 `128.3 rows/hour`，完整 148-route 粗略 ETA `3.23h`，同样只作 contention operational estimate。

## H200 与安全决定

- H200 NVL：utilization=`100%`，used/free=`97053/46104 MiB`，power=`271.42 W`，temperature=`52°C`，uncorrected ECC=`0`。
- 研究进程显存：M1 PID `2251038`=`17098 MiB`；M2 PID `2254866`=`16408 MiB`。两者仍在增长、日志中未发现 Traceback/CUDA OOM/磁盘错误，磁盘余量 `2.5T`、free inodes=`232121442`。
- 决定：`healthy_continue_two_independent_jobs`。现在不启动第三个研究进程；只在新的 t+10 审计仍同时满足两个任务增长、OOM/ECC=0、free memory ≥`25600 MiB` 时再考虑。M3、M4 等一个研究槽空出后依序启动。

SciPy 仍未安装（本地 import 为 `ModuleNotFoundError`；安装请求被宿主审批服务返回 422），故 corrected official CAL 维持 `waiting_for_scipy_then_12_case_smoke`。这不停止 M1/M2，也不构成 scientific blocker。
