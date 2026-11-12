# 恢复128预算任务

工作目录：/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-multiline-threshold-128-20261112-v1
共享目录：/home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1

用户已授权完整34配置128补充实验。先查status、READY、角色锁和已有进程；勿重复模型实例。显存足够时执行：

```bash
bash /home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1/start.sh
```

启动入口一次检查当前映射GPU容量，不等外部任务退出；不足则记录blocked_resources并返回4。通过后固定16例smoke，smoke进程退出后发布READY，pipeline启动nohead/trained两个角色；完成生成并自然释放后评分、独立计时、分析、focused commit/push。单次OOM原口径恢复一次，其它基础设施最多3次，不按代码得分重抽。

可持久启动命令见commands.txt。64任务继续使用原工作区及账本，绝不在它的运行中改预算。
