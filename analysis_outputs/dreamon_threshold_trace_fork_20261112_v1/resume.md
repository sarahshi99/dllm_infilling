# 恢复入口

分支：codex/dreamon-threshold-trace-fork-20261112-v1。工作区：/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-threshold-trace-fork-20261112-v1。

先看status.json和local/run.log；tmux会话threshold_trace_fork_20261112。禁止重复启动模型。若会话已退出，在上述工作区运行：

```bash
TOKENIZERS_PARALLELISM=false tmux new-session -d -s threshold_trace_fork_20261112 -c "$PWD" 'bash experiments/threshold_trace_fork/launch.sh > analysis_outputs/dreamon_threshold_trace_fork_20261112_v1/local/run.log 2>&1'
```

SQLite local/results.sqlite按(kind,key)去重。已完成案例不重复累计，执行源码版本必须一致；仅文档提交允许不同HEAD。固定state_panel/special_state_manifest生成后不根据结果替换。快照位于local/snapshot__*.pt；包含当前候选前整个控制状态和张量深复制，不能只以画布替代。

主流水线状态completed表示生成/分叉/初步汇总完成；最终交付以delivery_verification.json和delivery_completed状态为准。smoke_passed只表示初步技术检查。scientific_validity_blocked先诊断，不能改历史算法迎合数字。OOM限一次进程释放/恢复，基础设施限三次；attempts.log保留退出记录。普通失败与3秒超时不补抽。

若只需重建报告：`/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.threshold_trace_fork.analyze`，无模型加载。

分叉前已修正专项“首次实际动作”定位：不同候选都拒绝不算实际动作差异。回放解码器仍为418e73e；分叉编排的阶段版本见phase_versions.json和SQLite meta。恢复命令会分别验证阶段来源，不重新生成已完成回放；不能用旧编排版本接着生成新分叉。

## 后续技术复核（单模型顺序）

持久队列tmux `threshold_trace_fork_finish_20261112` 等待主会话结束，随后执行原入口逐例详细字段复核，再复核同64面板的log空间KL/TV，最后离线汇总。不会与主模型进程并发加载第二份模型。

```bash
tmux new-session -d -s threshold_trace_fork_finish_20261112 -c "$PWD" 'bash experiments/threshold_trace_fork/finish.sh > analysis_outputs/dreamon_threshold_trace_fork_20261112_v1/local/finish.log 2>&1'
```

先确认会话是否已存在，禁止重复启动。复核日志local/verify_all.log，汇总日志local/enrich.log，退出记录local/validation_attempts.log。仅当主阶段完整且成功时才运行依赖复核。local/PAUSE_REQUESTED存在时队列不会启动新阶段；用户明确恢复后处理该标记。原入口5168次技术对照不是新增主配置或新增独立样本；完全相同文本复用既有官方3秒评分。教师64状态的第二次是数值校验，额外64个教师前向单独计数，不参与任何分支决策。
