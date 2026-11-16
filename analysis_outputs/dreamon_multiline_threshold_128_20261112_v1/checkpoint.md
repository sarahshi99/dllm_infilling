# 128预算补充实验

状态：smoke_running。2026-11-16用户授权恢复启动；启动时空闲63289 MiB，显存阻塞已解除。质量0/197710、评分0、独立计时0/16728；当前先执行固定16例真实模型smoke，通过且模型自然退出后自动发布READY并启动两个质量角色。

运行：tmux dreamon_multiline128_20261116_v1；smoke PID1574978；启动时间UTC 2026-11-16T09:08:59.202227+00:00。执行提交ad0f9013a0be84aa109abdf83e06f8ff9fc8a4ab。启动日志/home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1/launch.log；smoke日志/home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1/smoke.log；实时状态以共享目录角色status和SQLite账本为准。

设置：初始mask64、最大长度128、扩展次数预算128、steps256；其余34配置、模型/冻结头、seed42/共同采样、官方5815案例和独立计时子集沿用原任务。47项CPU测试已通过；未修改生成代码。新任务独立保存197710质量行及16728独立计时行，不与64预算账本混用。

入口：bash /home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1/start.sh。已有启动器正常运行时不要重复启动。启动器通过smoke后自动串起完整生成→模型自然释放→官方bwrap每例3秒评分→统一计时→审计/报告→focused commit/push与远端核实。未干预其它GPU任务；GPU共享计时不声称独占加速。

本轮启动记录launch_snapshot.json；前次资源失败的退出状态保存在共享launch.exit.20261112T073858Z，旧日志保留。当前没有质量结论。
