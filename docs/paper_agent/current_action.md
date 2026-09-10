# 当前行动：Markov 训练审计修正与双词元推理准备

<!-- markov-training-audit-20260908 -->
## 2026-09-10 当前行动：无基准候选的 v2 数据与训练

- action：在新目录重建 OpenCoder `educational_instruct@7d28f40d...` 数据，先建立题意/实际回答代码/代码 AST/测试的传递关联组，再保守排除包含 HumanEval 候选的整个组；split seed 保持 `20260901`。
- exact data command：`HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python analysis/prepare_dreamon_markov_training_data.py --model-snapshot /home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194 --evaluator-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff --result-dir analysis_outputs/dreamon_markov_head_training_20260910_v2 --cache-dir /home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/data --split-seed 20260901`。
- next GPU phases：共享 bank `200000/20000/20000`，随后 TV→释放→KL；H200 单卡串行、`dreamon-repro`、新共同零初始化，旧 v1 权重与优化器不加载。
- success：新实际 bank 无 HumanEval 候选，关联组零跨集，原策略/阶段配额满足；两个头按 `optimizer_batch_v2` 完成既定 pilot/full 验证，至少一个非零修正通过后才打开新外部分布测试并进入六组 SingleLine。
- kill/pause：排除后任一既定 bank 配额不足、两个头均失败、数据或主键完整性错误、或 GPU 被其他任务占用且无法安全串行。禁止缩小规模、重复样本凑数或覆盖 v1。
- outputs：Git 结果目录 `analysis_outputs/dreamon_markov_head_training_20260910_v2/`；服务器数据/权重 `/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/`；运行日志 `logs/paper_agent/20260910_dreamon_markov_v2_k2_eval.log`。

### 2026-09-10 启动前复算审计

- action：从固定 OpenCoder 源数据和 HumanEval 1033 条记录重新计算标准化去重、候选匹配、传递关联组和确定性 split；逐条对照服务器 `prepared_records.jsonl.gz`、完整 split manifest 与 external-test manifest。
- reviewer motivation：防止把数据准备脚本成功退出误当成候选排除、组隔离和 tokenized artifact 均正确。
- exact command：`HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python analysis/audit_dreamon_markov_prepared_data.py --model-snapshot /home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194 --evaluator-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff --prepared-records /home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/data/prepared_records.jsonl.gz --split-manifest analysis_outputs/dreamon_markov_head_training_20260910_v2/split_manifest.jsonl.zst --external-test-manifest analysis_outputs/dreamon_markov_head_training_20260910_v2/external_test_manifest.jsonl.zst --exclusion-audit analysis_outputs/dreamon_markov_head_training_20260910_v2/humaneval_exclusion_audit.json --data-summary analysis_outputs/dreamon_markov_head_training_20260910_v2/data_manifest_summary.json --preparation-audit analysis_outputs/dreamon_markov_head_training_20260910_v2/split_and_dedup_audit.json --output analysis_outputs/dreamon_markov_head_training_20260910_v2/prepared_data_reaudit.json`。
- success：源数据 118278、HumanEval 1033/164；实际 prepared 主键唯一且与 manifest 完全一致；HumanEval 直接候选及其 39 个关联组与 prepared 交集为零；group 零跨 split；token 计数、长度、row seed、词表边界全部一致。
- pause/kill：任一主键、候选排除、group split、token 长度或词表边界不一致；失败时不启动 GPU bank。
- resources：CPU only；不修改源数据或既有 v1/v2 artifact。GPU bank 仍需后续真实生成并做 200000/20000/20000 的 SQLite 成员审计。

## 2026-09-10 当前行动：服务器实际成员核查

- action：只读关联已保存的 transition bank、HumanEval 候选 record_id 与现有验证/外部诊断 sample_key；命令为 `/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python analysis/audit_markov_bank_membership.py --result-dir analysis_outputs/dreamon_markov_head_training_20260901_v1 --bank-db /home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260901_v1/transition_bank.sqlite`。
- reviewer motivation：旧报告已撤回“严格隔离”；只有确认候选是否实际进入 full training，才能决定旧 KL v1 权重是否可用于完整 1,033 条开发/机制评估。
- success：生成 `actual_bank_membership_audit.json`，验证 200,000/20,000/20,000 bank 主键数与两类现有诊断完全一致，并给出每个实际入 train 的 HumanEval 候选的 transition 数。
- pause condition：任何 AST 无文档字符串代码匹配候选实际进入 train，或关键主键不一致，则仅实现 K=2 与外部小样本检查，暂停完整 HumanEval；不重训、不删题、不改旧划分。
- resources：CPU/SQLite only；不加载模型、不做教师前向、不修改 bank；输出写入既有 Markov 审计目录。

## 2026-09-08 当前状态：训练完成，协议偏差已确认

- 决策：`iterate`；原训练提交 `32e8004`，修正分支 `codex/dreamon-markov-training-audit-fix-20260908`。
- 完整冻结清单存在跨集重复：代码 1,239 组，题目文本 3,102 组（两者重叠，不可相加）。不能再声称严格隔离。
- 已修复无头报表；保守剔除后的外部测试保留 13,387 条，TV/KL 距离缩小 26.69%/27.79%，是事后敏感性结果，不是新的独立测试。
- 旧权重仍属小批次参考损失均值版本；整批归一化已修复代码但未重训。两条训练候选代码与 HumanEval/13 的标准最大公约数实现匹配，实际是否训练需查服务器 SQLite；其余单断言候选需人工复核。
- 下一步：执行 `analysis/audit_markov_bank_membership.py`；实现固定左前沿双词元与全局 top-2 相邻短链。仅在基准隔离问题澄清后，用原验证选定 KL、λ=1 完成 1,033 条 Pass@1/速度比较。确认污染则保留实现与外部检查，暂停正式 HumanEval，不自行重训。
- 权威报告：`analysis_outputs/dreamon_markov_head_training_20260901_v1/report.zh.md`；原权重、原始诊断与冻结清单保留不动。以下旧日期条目仅作历史记录，不是当前运行指令。

原训练方案保留于 `experiments/20260901_dreamon_markov_head_training_action.zh.md`。本轮未启动任何 GPU 任务。
