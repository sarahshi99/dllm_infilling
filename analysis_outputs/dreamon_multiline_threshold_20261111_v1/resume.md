# 本轮恢复入口

工作目录：/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-multiline-threshold-eval-20261111-v1；共享目录：/home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1。

先读共享status、角色status/heartbeat/exit和tmux，查角色锁；不得重复启动。两个角色共用一张实时映射H200，不设GPU互斥，不改变外部任务。

```bash
bash /home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1/start_worker.sh --role nohead
bash /home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1/start_worker.sh --role trained
THRESHOLD_TASK_ID=dreamon_multiline_threshold_20261111_v1 /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -u -m experiments.threshold_eval.runner --phase pipeline
```

pipeline会补启动缺席且没有失败记录的角色；已有失败记录先查具体错误，不无限重启。OOM最多立即恢复一次，基础设施最多重试3次。生成完成且两个角色自然释放后才执行官方评分和统一计时，再审计、分析、提交/push与远端核实。READY只表示代码发布，不表示GPU空闲。
原始SQLite事务位于共享目录，缺失key才生成；attempt不是新样本。Git比较相关生成路径允许纯文档HEAD变化，不允许混合生成语义。
本轮质量目标197710（跨行子集162588）；独立计时16728（跨行12954），每配置5815与492/381。旧single-line目录和结果不修改。
