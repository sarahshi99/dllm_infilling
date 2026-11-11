你负责 dllm_infilling 官方 MultiLine 阈值任务的 nohead 角色运行与监督。请先读：
/home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1/task_pack/master_prompt.zh.md
以及同目录 planning_spec.json。若共享副本暂不可用，读取：
/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-multiline-threshold-plan-20261111-v1/analysis_outputs/dreamon_multiline_threshold_20261111_v1/task_pack/。

任务ID dreamon_multiline_threshold_20261111_v1。你的范围是18配置，每配置官方5815唯一案例，共104670条；完整配置来自主控发布的run_spec，包含global/left的K1和K2/K4×off/.80/.90/.95，不漏off或负结果。

你不修改公共源码、参数、阈值、随机键或统计。先检查共享角色登记、锁、进程、READY与状态；已有nohead进程时只监督，不能加载第二个模型。代码未准备好可登记到roles/nohead，并用tmux小型等待器等待READY，不用模型持续轮询。若主控状态为资源阻塞，记录角色状态，不另加GPU空闲排队或修改主控入口。

主控发布READY后，使用唯一入口：
bash /home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1/start_worker.sh --role nohead

展示命令后直接执行，不等逐阶段确认。READY是代码准备完成信号；仍须遵守主控声明的实际资源阻塞。与trained同卡并发，每角色只驻留一个模型；不设GPU互斥、不等空卡、不用旧GPU2/3、不干预其它任务。显存容量确实不足时报告具体阻塞，不抢占、不擅改串行。OOM只允许本失败进程原口径恢复一次，其它基础设施最多重试3次；正常代码失败和官方超时不重抽。

统一worker负责实际无头（头调用数必须为0）、共同采样、阈值/断链/结构规则、事务恢复、心跳、逐例落盘。你的本地记录写roles/nohead，不覆盖run_spec、READY、公共status或其它角色账本。

完成后确保104670条唯一质量记录且执行版本统一，进程自然退出并释放模型。官方评分/统一计时/汇总由主控持久流水线负责，你不另开计时模型、不重跑已提交结果、不另行提交公共代码。进度回复列实际完成配置/案例数、PID/tmux/log/账本、重试和阻塞；未完成不宣称完成。无需向其它对话自动发消息；共享目录是协调依据。
