# 2026-10-13 三起点来源核查与现有头零训练迁移

当前用户已明确授权主控协调A/B/C三个智能体、共享GPU按资源可用串行执行、无需逐批确认；覆盖仓库默认不使用智能体和默认GPU2/3。实际仅GPU0 H200，不抢占六个外部服务。新主控分支从已核查340170ce创建；原工作区未提交文件和旧报告保持原状。

审稿动机：区分确切DreamOn权重的跨模型迁移、同模型校准、固定长度Base辅助实验，隔离训练贡献和二次采样路径。假设为现有v2 KL rank256/lambda1头可直迁，允许负结果；词表/接口不相容或原生规则不能对齐即报告阻塞，不靠重训/改映射解决。

方案选择：优先复用DreamOn自身10×1033诊断；查找另外两份官方DreamOn，缺失保持主表缺失；兼容的Base单列固定64辅助。拒绝把量化版/第三方Java或HF历史提交当三个起点。无新测试集、无927人口拼接、无训练/K4/隐藏状态头。

人口：原1033稳定ID、164基础题开发人口；seed42及原derive_row_seed；固定16个字典序等距稳定ID技术检查，不按正确率筛选。每目标两顺序、四组A=K1无头/B=K2无头/C=K2同类同秩零输出头lambda1/D=K2来源头lambda1；温度0.2/top_p0.9/bf16/初始64。DreamOn max_gen_len64不改128；Base遵循官方固定长度停止和截取。

输出：docs/paper_agent/transfer_20261013/，analysis_outputs/zero_training_transfer_20261013/{controller,a_dreamcoder,b_dream,c_diffucoder}/。公共代码仅主控合入；各智能体独立worktree/adapter/输出。来源检索和CPU审计并行；所有GPU都由主控持有/tmp/dllm_infilling_h200_diagnosis.lock，全轮累计12 GPU作业小时（含加载、技术检查、评测写盘），默认一次1作业。下载不计；共享计时仅成本，不报告正式速度比。

成功条件：来源精确revision可验证、完整普通token映射及规则一致、来源定义strict加载全部参数、一次shift；16行原生K1与无头adapter对齐、C/D同路径、无非法token/未恢复错误；估时可容纳则收益正负均完成1033×8，配对帮助/伤害与164题cluster CI。停止条件：技术错误、接口不兼容、预算不足、明确资源阻塞。质量低本身不停止，不更参。

运行前在对话显示exact command，运行清单保存命令、环境、版本、PID、日志、输出、预算和恢复入口。最终本地diff review+fresh verification；reviewer_gate_disabled只指不另开reviewer，用户明确要求的A/B/C照常执行。提交只含本轮独占文件并push新分支。
