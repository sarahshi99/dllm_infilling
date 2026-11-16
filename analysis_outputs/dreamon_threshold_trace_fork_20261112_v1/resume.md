# 恢复入口

分支：codex/dreamon-threshold-trace-fork-20261112-v1。工作区：/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-threshold-trace-fork-20261112-v1。

先看status.json和local/run.log；tmux会话threshold_trace_fork_20261112。禁止重复启动模型。若会话已退出，在上述工作区运行：

```bash
TOKENIZERS_PARALLELISM=false tmux new-session -d -s threshold_trace_fork_20261112 -c "$PWD" 'bash experiments/threshold_trace_fork/launch.sh > analysis_outputs/dreamon_threshold_trace_fork_20261112_v1/local/run.log 2>&1'
```

SQLite local/results.sqlite按(kind,key)去重。已完成案例不重复累计，执行源码版本必须一致；仅文档提交允许不同HEAD。固定state_panel/special_state_manifest生成后不根据结果替换。快照位于local/snapshot__*.pt；包含当前候选前整个控制状态和张量深复制，不能只以画布替代。

状态completed才表示全阶段完成；smoke_passed只表示技术检查。scientific_validity_blocked先诊断，不能改历史算法迎合数字。OOM限一次进程释放/恢复，基础设施限三次；attempts.log保留退出记录。普通失败与3秒超时不补抽。

若只需重建报告：`/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.threshold_trace_fork.analyze`，无模型加载。
