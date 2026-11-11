# H3 后台检查点

已真实启动tmux `dreamon_h3_20261111_v1`，worker PID 4008736。当前validation，epoch=0；持久化数量{'scored': 3099, 'validation_0': 4896}。此文件为启动快照，实时状态读progress.json。

公共库220000状态、32配额、来源隔离及停止写入证据通过；公共基线3099条匹配复用。53项CPU检查和真实公共状态GPU技术检查通过。代码从H2已发布9302a1e复用H1正式训练器，H3结构保持B原样，参数79,301,636。H1/H2工作区再次检查均干净。

有效批128的实测更新1.11/0.91秒，纯7815次训练更新外推1.97小时，另需完整验证及生成评分；共享负载与分片IO会改变时间。正式阶段初始验证进行中，训练参数未继承技术检查权重。

后台自动训练、固定best评测、官方评分、配对、归档提交推送并退出，不自动启动其他实验。H2完整1033结果可追加配对，H1当前仅6/1033需待补，不等待其修复。

实时日志：`/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1/pipeline.log`；状态：`/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1/progress.json`；退出：`/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1/exit.json`、`/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1/shell_exit.json`；检查点：`/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1/checkpoints/{best,last}.pt`。

恢复（不会重训已完成阶段）：
```bash
cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-context-h3-training-20261111-v1
bash experiments/context_h3_training_v1/launch.sh
```

最终报告：本目录report.zh.md、generation_comparison.json、training_summary.json及validation_epoch_*.json；原始数据、输出及权重保留服务器，路径见evidence_index.json。启动实证见background_receipt.json。本会话按授权在确认后台任务及日志后结束，不驻留轮询。
