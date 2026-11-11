# dreamon_multiline_threshold_20261111_v1 当前执行检查点

状态：quality_generation，UTC 2026-11-11T13:25:57Z。质量636/197710（nohead 290/104670；trained 346/93040）；评分0；独立计时0/16728。尚无完整配置结论，无资源/基础设施失败记录；任务仍在实际运行。

生成版本：ae0283e60fbd60603c3587f2c7394a3b6081a18d；后续报告/测试提交不改变生成代码，Git比较已核实。已推送分支codex/dreamon-multiline-threshold-eval-20261111-v1。CPU41项通过；固定16例smoke及4旧single-line回归通过。5815例全部字段与pinned Arrow一致，输入最长620/2048 token；旧single-line结果未修改。

持久进程：tmux dreamon_multiline_threshold_20261111_v1_pipeline（PID1475763）；nohead PID1475777；trained PID1475780。实际GPU0/UUID GPU-c55d478c-6944-066f-7e9a-8af248e4f6f1。主控无模型驻留，不干预其它任务。真实入口重复调用已验证两个角色均报告already running，不加载副本。

流水线自动继续：完整生成→本轮两个质量模型自然退出→官方bwrap/每例3秒评分→统一164×34×3独立计时→分层/配对10000次bootstrap、审计/报告→fresh tests、diff检查、focused commit、push与远端核实。OOM最多恢复一次，其它基础设施最多重试3次；失败写具体状态与尝试。当前没有等待用户审批的阶段。

实时状态/锁/心跳/事务：/home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1。报告快照：report.zh.md，完整审计未通过前不得声称评测完成或阈值/头有效。smoke粗估逐次耗时合计69.8小时，单卡双角色与外部争用使其不能直接当墙钟截止时间。恢复命令见resume.md，先查已有进程，禁止重复启动。
