# DreamOn 外部一阶 Markov 头训练 v1：行动说明

> 2026-09-08 注：以下是原定方案，不代表实际执行全部合规。跨集重复和损失归一化偏差已确认；现状及下一步以同目录 result 文档与修正报告为准。

本轮检验一个严格受限的问题：冻结 DreamOn-v0-7B 后，只看刚提交的左邻 token，rank-256 的低秩加性修正能否逼近重新前向得到的右邻分布。TV-head 与 KL-head 结构、数据、初始化、优化器、顺序和预算完全相同，唯一差异分别为全词表 L1/TV 与正向 KL 分布损失。

数据只使用 `OpenCoder-LLM/opc-sft-stage2@7d28f40d579edd7c24402d17d0c7639f991e6f8d` 的 `educational_instruct`。先按原始记录分组完成规范化、重复合并和 HumanEval 严格去重，再用 seed `20260901` 冻结 80/10/10 split；测试清单在任何训练前封存，训练期间不读取测试指标。

实现采用共享可重放 transition bank，不保存完整词表 logits。轨迹仍由无 head 的官方逐词 DreamOn 生成，只保留空间连续、无结构动作/坐标变化的 offset=1 stale/fresh 状态。训练时在线重放冻结 DreamOn logits，修正在 released logits shift 和动作屏蔽之后、temperature/top-p 之前加入。

执行命令将由 `scripts/manual_launch_dreamon_markov_head_training_20260901.sh` 固定：先 CPU/小 GPU 门禁与 manifest/bank，再独立 TV 进程，确认退出和显存释放后独立 KL 进程；最后只在完整验证门禁允许时打开冻结外部测试一次。长运行写入 `logs/paper_agent/20260901_dreamon_markov_head_training_v1.log`，大型 bank/checkpoint 写入 `/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260901_v1/`。

成功标准、pilot/full/test 门禁、停止条件和禁止范围完全以用户 2026-09-01 授权文本及 `docs/paper_agent/current_action.md` 为准。本轮完成外部训练比较后停止，不自动进入 HumanEval、K=2/K=4、RNN 或控制器实验。

