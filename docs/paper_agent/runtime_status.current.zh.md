# Runtime Status — 2026-07-15 UTC

本文件只记录运行完整性、资源和计算预算；本轮没有读取或报告任何部分 pass rate、help/harm 或性能结论，`performance_inspected=false`。

## 10 分钟只读窗口

- `t0=2026-07-15T04:48:08Z`；`t+10m=2026-07-15T04:58:32Z`；窗口为 `624` 秒。
- H200：utilization `99%`，`33,729 MiB` used、`109,427 MiB` free，`275.56 W`，`53 C`，uncorrected ECC=`0`。
- 磁盘：`2.5T` 可用，inode free=`232,141,584`（`1%` used）。两份日志的最新 20 条和全文 error-pattern scan 均无 Traceback、CUDA OOM 或磁盘错误。

## A. 旧共享 MultiLine 候选库（优先任务）

- tmux `phase6-multiline-bank` 和历史 PID `1195368` 均已自然退出；这不是故障。
- raw：`40,632 / 40,632`，`40,632` unique candidate keys，duplicate=`0`，status `ok=40,632` / `error=0`，最后 JSONL 行可解析。
- `full_audit.json` 已于 `2026-07-15T02:35:02Z` 写出：missing/duplicate/error=`0/0/0`，resume no-op=`0`，frozen invariant passed。
- 10 分钟新增 `0` 行，因为任务已完成；ETA=`0h`。
- 决定：`completed_final_audit_passed_do_not_restart`。不发送信号、不重启、不改写 raw。

## B. 新版 M1 MultiLine

- tmux `phase6-m1-multiline-v3` 存在；PID `1576214` 在 t+10 仍为 `Rl+`，elapsed `1-18:17:58`，CPU `103%`，RSS `856,996 KiB`，GPU `20,584 MiB`。
- stage-one：`22,749 → 22,921 / 45,711`；窗口速度 `992.31 rows/hour`，完成 `50.14%`，线性剩余 ETA `22.97h`。
- generic refinement：`12 / 5,079`；dependency-cone refinement：`12 / 5,079`；两者均 `ok`、unique、无 duplicate，最后行可解析。
- total：`22,945 / 55,869`，线性完整 ETA `33.18h`（只作计算预算估计，不含任何质量结论）。
- 由于 stage-one <`80%`，规则要求 `pause_reason=pre_outcome_compute_budget_reconciliation`。已尝试仅对 PID `1576214` 发送 SIGINT；host approval control plane 返回 `422`，因此**实际暂停未完成**。没有使用 SIGKILL、tmux Ctrl-C 或任何绕过方式；三个 raw 目录未被移动、删除、覆盖或重建。
- `resumable=true`；原命令已在 JSON 状态中保存。未来 5079 运行已改为显式 selected-method-only gate，裸 `--auto-full` 不再能直接触发。

## 后续 GPU 任务

M2 已完成 CPU 路由/测试准备，但未实际启动。只有 M1 已安全退出、旧库保持完成状态、H200 free memory ≥`25,600 MiB` 且 OOM/ECC=`0` 时，才启动独立 tmux/log/output 的 M2 12-case smoke；技术通过后最多自动进入 `148` RandomSpanLight，不进入 `927` 或 `5079`。
