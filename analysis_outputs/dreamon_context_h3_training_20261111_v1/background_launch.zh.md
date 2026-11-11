# 后台运行入口

工作目录：`/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-context-h3-training-20261111-v1`
分支：`codex/dreamon-context-h3-training-20261111-v1`。
公共库与基线审计已通过，53项CPU测试通过，真实公共状态GPU技术检查通过（见technical_checks.json）；临时权重与优化器已丢弃。

启动：
```bash
tmux new-session -d -s dreamon_h3_20261111_v1 -c /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-context-h3-training-20261111-v1 'bash experiments/context_h3_training_v1/launch.sh'
```

唯一worker命令由launch.sh设置CUDA_VISIBLE_DEVICES=0后执行：
`/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -u -m experiments.context_h3_training_v1.pipeline`。

日志：`/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1/pipeline.log`
进度/退出：同目录progress.json、exit.json、shell_exit.json；检查点：checkpoints/{best,last}.pt。

恢复：
```bash
cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-context-h3-training-20261111-v1
bash experiments/context_h3_training_v1/launch.sh
```

无需驻留监督。该有限流水线自动完成5轮训练（含初始和每轮完整验证）、固定best的1033行生成、官方评分、配对汇总及commit/push后退出。遇到数据损坏、最小微批仍OOM、评测显存不足或其他技术阻塞时记录准确阶段及恢复命令，不启动其他头。
