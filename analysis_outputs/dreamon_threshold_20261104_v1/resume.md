# 本轮恢复与运行

共享目录 `/home/shx/projects/dllm_infilling/threshold_eval_20261104_v1`。`run_spec.json` 固定34配置、1033质量案例、独立164计时案例、3次计时。`READY.json` 给出实际执行提交和隔离worktree。`start_worker.sh` 只锁同角色，两个角色直接同卡并发，不等待外部资源。

正常运行先读共享 `status.json`、`nohead.status.json`、`trained.status.json`、`timing.status.json`、各角色 `.heartbeat.json` 和日志末尾；不要重复加载。角色存在时重复入口会安全退出。缺席角色用：

```bash
bash /home/shx/projects/dllm_infilling/threshold_eval_20261104_v1/start_worker.sh --role nohead
bash /home/shx/projects/dllm_infilling/threshold_eval_20261104_v1/start_worker.sh --role trained
```

模型每角色只加载一次并依次处理其所有配置。质量事务键为(kind, config, case_key, repetition=0)，每例SQLite FULL同步提交；恢复跳过已有键，禁止混合执行版本。OOM退出释放本进程上下文后立即最多恢复1次；基础设施最多重试3次；普通答案错误/官方3秒超时不重试。不改变参数、不重抽成功、不终止其它任务。

流水线用独立tmux运行 `python -u -m experiments.threshold_eval.runner --phase pipeline`（使用run_spec指定Python与worktree）。它等待本轮两个角色自然结束与锁释放，随后顺序官方评分、统一计时、完整性审计、分析报告、focused commit和push核实。等待仅为代码/本轮依赖，不是外部GPU空闲。会话断开后继续。

统一计时预热全部34配置；按164案例及3个重复稳定轮换配置顺序；每次cuda同步，时间不含写盘/评测/加载/重型轨迹。所有配置完整492次，不把计时样本并入质量人口。恢复再次预热并只补缺失事务。共享资源观测保留，不能视作独占公平测速。

原始逐例结果在共享`nohead.sqlite`、`trained.sqlite`、`timing.sqlite`；失败尝试表亦在各账本。报告中`per_case.jsonl.gz`与`timing_per_case.jsonl.gz`可独立复算。原始日志、smoke重型画布、GPU心跳日志及权重留本机，不进入普通git。

若代码必须修复，不能让原版本结果继续参与新版本统计；先明确受影响人口，保留原账本证据并新版本重跑受影响行。本轮起点43a52fd不回退主工作区。任务完成后不要在原执行提交检查仍有效时提前提交文档，自动流水线会在所有GPU进程退出后统一提交。
