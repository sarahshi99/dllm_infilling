# 128预算补充实验

状态：blocked_resources，GPU生成尚未开始。质量0/197710、评分0、独立计时0/16728；READY未发布，真实模型smoke未执行。CPU测试31+16=47项通过，shell语法和本地diff检查通过。

用户已授权完整34配置，5815官方案例；初始mask64、最大生成长度128、扩展次数预算128、steps256；模型/头、seed42、共同采样、阈值、数据/计时案例、评分和统计与64任务相同。预算参数之外无方法变化。实现提交：4f098d686d59738f407caf4879de12b327257ce1；不是已生成结果的执行版本。

启动尝试：2026-11-12T07:38:58Z，tmux dreamon_multiline128_20261112_v1；启动器退出码4，tmux已退出。H200 UUID GPU-c55d478c-6944-066f-7e9a-8af248e4f6f1，观测空闲619 MiB，新两个角色预留需要40000 MiB。进程与显存证据见status.json。没有启动GPU空闲轮询或排队，没有加载额外模型，也未干预现有64或外部任务。

原64角色PID1475777/1475780仍运行；原工作区git状态干净，相关生成路径与ae0283e一致。两轮账本和路径独立，不能混合记录。

启动日志：/home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1/launch.log；退出状态：同目录launch.exit。资源恢复后在tmux执行唯一入口：

```bash
bash /home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1/start.sh
```

该入口会单次核对容量→固定16案例真实smoke→进程退出释放→发布READY→两个角色并发完整生成→官方bwrap3秒评分→独立计时→审计报告→focused commit/push并核实。已有授权继续有效，不需重新审批；不得中断别人任务或用CPU offload偷偷改变运行设置。

工作区：/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-multiline-threshold-128-20261112-v1；分支codex/dreamon-multiline-threshold-128-20261112-v1。恢复说明resume.md。当前没有质量或速度结果，不作方法效果判断。
