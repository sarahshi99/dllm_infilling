# 第一批A后台启动与恢复

已启动并核验正常日志。控制器PID 470433；唯一worker PID 471126；tmux `dreamon_context_foundation_20261015_v1`。当前唯一H200约25GiB空闲，低于32GiB启动阈值，因此等待，不加载模型、不抢占、不消耗GPU作业预算。

从本分支worktree根目录恢复（同一入口识别现有tmux/专用锁，禁止副本）：

```bash
bash scripts/launch_context_foundation_queue.sh
```

查看日志：`analysis_outputs/dreamon_context_foundation_20261015_v1/worker.log`；状态`queue_status.json`、`waiting.local.json`、`gpu_progress.local.json`；PID/绝对路径/精确cd命令仅存本机launch.json。恢复原始张量用同入口加`--restore-missing`，原累计预算不重置。

总GPU作业上限12小时（A1/A2最多4小时、DCD最多2小时），等待最多累计24小时。粗估4–10 GPU作业小时，覆盖不足时如实partial。计划：真实技术smoke→共享状态/反事实→DCD提交与原子ready→4132生成→GPU退出→隔离CPU评分/分析/归档/追加日志/提交推送。网络失败保留本地提交及push命令。

公共git中转协议ready已发布且hash验证；DCD ready只在实际小包完整验证并提交后发布，A_done在自动归档后发布，状态可能complete/partial/blocked，不把启动当实验完成。不启动第二批、不训练、不改思维导图。
