你负责 dllm_infilling 官方 MultiLine 阈值任务的 trained 角色运行与监督。请先读：
/home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1/task_pack/master_prompt.zh.md
以及同目录 planning_spec.json。若共享副本暂不可用，读取：
/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-multiline-threshold-plan-20261111-v1/analysis_outputs/dreamon_multiline_threshold_20261111_v1/task_pack/。

任务ID dreamon_multiline_threshold_20261111_v1。你的范围是16配置，每配置官方5815唯一案例，共93040条：global/left×K2/K4×off/.80/.90/.95；配置以主控run_spec为准，不漏off或负结果，不增加trained K1。

你不修改公共源码、参数、阈值、随机键或统计；不训练/换头/调lambda。头为已有rank256 v2 KL/full/best，FP32、lambda=1，通过原manifest定位；已登记sha256 58fd352ef12ff57fa0eec47a94047559f9dfd84b5ba4478dca3395089c87265a。

先检查共享角色登记、锁、进程、READY与状态；已有trained进程时只监督，不能加载第二个模型。代码未准备好可登记roles/trained，用tmux小型等待器等待READY，不用模型持续轮询。若主控状态为资源阻塞，记录状态，不添加GPU空闲排队、不改主控入口。

主控发布READY后，使用唯一入口：
bash /home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1/start_worker.sh --role trained

展示后直接执行，不等逐阶段确认。READY只是代码完成信号；仍须遵守具体资源阻塞。与nohead同卡并发，每角色只驻留一个模型；不设GPU互斥、不等空卡、不用旧GPU2/3、不干预其它任务。容量不足报告具体阻塞，不抢占、不擅改串行。OOM只允许本失败进程原口径恢复一次，其它基础设施最多重试3次；普通答案失败与官方超时不重抽。

统一worker负责冻结模型和头、合法链内前驱、实际候选T1置信度、拒绝不入记忆/不跨间隔、修正结构词屏障、共同采样、逐例事务与心跳。你的本地记录写roles/trained，不覆盖run_spec、READY、公共status或其它角色账本。发现公共实现问题交给主控记录和修复，不私自运行不同版本。

完成后确保93040条唯一质量记录且版本统一，进程自然退出释放模型。评分/统一计时/分析由主控流水线负责；你不另开计时模型、不重复生成、不另行提交公共代码。回复列真实进度、PID/tmux/log/账本、尝试与阻塞；未完成不宣称完成。无需向其它对话自动发消息；共享目录为协调依据。
