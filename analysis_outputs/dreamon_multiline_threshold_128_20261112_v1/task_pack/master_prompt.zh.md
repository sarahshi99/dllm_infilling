# 官方 MultiLine 阈值实验：补充128预算

用户于2026-11-12明确授权直接补充128实验，其他设置与正在运行的64实验相同。计划版本128-v1；不重新训练、不修改64任务工作区及账本。

候选：继续只用64；另跑96；另跑128。采用128：能检验64容量限制，并与DreamOn论文最大长度口径衔接，避免增加未经请求的预算点。baseline是本任务重新生成的global/left K1及同方法off；同数据、同模型、同硬件共享环境。

官方5815案例（跨行4782）；完整34配置，197710质量行。固定164计时案例、每配置3次，共16728；跨行计时12954。模型DreamOn revision8ccc747、bf16；原rank256 KL/full/best头FP32、lambda1；seed42和共同随机键不变。初始mask64、最大长度128、扩展次数预算128、steps256。阈值、链、结构、评分3秒、统计/分母和报告口径沿用64任务。

修改experiments/threshold_eval/{common,adapter,runner,analyze}.py及参考decoder的预算参数，默认64以保留原行为。先验证128扩展能力、64回归、事务/数据/统计，再固定16案例真实模型smoke；smoke释放后启动两个角色同卡并发。reviewer_gate_disabled：本地diff review和fresh verification，无subagent。

GPU使用实时映射cuda:0；不抢占外部任务、不终止64任务。显存不足记录blocked_resources，不创建空闲轮询器，不擅自改成两个角色串行。OOM仅失败进程退出后原口径重试一次，基础设施最多重试3次。启动时实际资源是当前可能阻塞；共享负载下速度仅探索证据。

任务ID：dreamon_multiline_threshold_128_20261112_v1。
共享目录：/home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1。
报告目录：analysis_outputs/dreamon_multiline_threshold_128_20261112_v1。
命令：THRESHOLD_TASK_ID=dreamon_multiline_threshold_128_20261112_v1 /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -u -m experiments.threshold_eval.runner --phase smoke；通过后bash共享目录/start_worker.sh --role nohead / trained，并持久运行--phase pipeline。

成功：全部事务唯一、版本统一、质量/评分/独立计时完成，跨行及跨度层分母正确；输出report.zh.md、CSV、integrity、attempts、checkpoint与恢复入口。失败/停止条件：代码或smoke不一致、资源不足、限定次数重试失败；不按得分删案例或重抽。当前只获授权补充实验，尚无质量结论；128不会自动提高所有方法的通过率。

完整阈值、结构、评分、统计和展示细则继承同级旧任务包：analysis_outputs/dreamon_multiline_threshold_20261111_v1/task_pack/master_prompt.zh.md。新用户授权将该任务书的max_gen_len与扩展预算64替换为128，并使用本文的新任务ID、分支、工作区、共享/报告路径；其余设置不变。旧任务目录和进程完全独立。
