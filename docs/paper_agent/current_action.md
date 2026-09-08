# 当前行动：Markov 训练审计修正与双词元推理准备

<!-- markov-training-audit-20260908 -->
## 2026-09-08 当前状态：训练完成，协议偏差已确认

- 决策：`iterate`；原训练提交 `32e8004`，修正分支 `codex/dreamon-markov-training-audit-fix-20260908`。
- 完整冻结清单存在跨集重复：代码 1,239 组，题目文本 3,102 组（两者重叠，不可相加）。不能再声称严格隔离。
- 已修复无头报表；保守剔除后的外部测试保留 13,387 条，TV/KL 距离缩小 26.69%/27.79%，是事后敏感性结果，不是新的独立测试。
- 旧权重仍属小批次参考损失均值版本；整批归一化已修复代码但未重训。两条训练候选代码与 HumanEval/13 的标准最大公约数实现匹配，实际是否训练需查服务器 SQLite；其余单断言候选需人工复核。
- 下一步：执行 `analysis/audit_markov_bank_membership.py`；实现固定左前沿双词元与全局 top-2 相邻短链。仅在基准隔离问题澄清后，用原验证选定 KL、λ=1 完成 1,033 条 Pass@1/速度比较。确认污染则保留实现与外部检查，暂停正式 HumanEval，不自行重训。
- 权威报告：`analysis_outputs/dreamon_markov_head_training_20260901_v1/report.zh.md`；原权重、原始诊断与冻结清单保留不动。以下旧日期条目仅作历史记录，不是当前运行指令。

原训练方案保留于 `experiments/20260901_dreamon_markov_head_training_action.zh.md`。本轮未启动任何 GPU 任务。
