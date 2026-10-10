# DreamOn Markov v2 训练结果（K=2 前置证据）

本结果对应 `dreamon_markov_head_training_20260910_v2`。它回答的是“头是否把 stale DreamOn 分布拉近 fresh DreamOn 分布”，**不回答** SingleLine 代码 Pass@1 或端到端加速。

## 数据与完整性

- prepared 源记录：`106491`；split 为 train/validation/external-test=`86019/10547/9925`。
- 完整 transition bank：`240000`，分别为 `200000/20000/20000`；SQLite `quick_check=ok`，unique sample key=`240000`，18 个 policy/stage 配额均准确。
- 保守 HumanEval 候选关联组排除后，prepared 和 bank 与排除记录的交集均为 `0`，跨 split group=`0`。这不证明 DreamOn 基础模型预训练中未见过 HumanEval。

## 训练与选择

两头共享冻结 DreamOn、rank=256、共同零输出初始化（seed=42）、相同 bank、有效 batch=128；TV 与 KL 仅替换原有分布损失。两者 pilot 和 full（最多五 epoch）均通过既定门槛。按验证 raw TV、stable corruption、aligned reference NLL 的既定顺序选择 **KL, lambda=1**；旧 v1 权重和优化器没有续用。

| 方法 | 验证 raw TV | 相对 no-head 改善 | 分组 CI 下界 |
| --- | ---: | ---: | ---: |
| no-head | 0.063890 | — | — |
| TV | 0.045956 | 28.07% | +0.01472 |
| KL | 0.045349 | 29.02% | +0.01542 |

外部分布测试（20,000 transitions）中，no-head/TV/KL raw TV 分别为 `0.065899/0.047698/0.047099`；KL 相对 no-head 改善 `28.53%`，top-1 agreement 为 `97.315%`。这支持用 KL, lambda=1 做一次预注册的 K=2 推理检验，但不应表述为代码正确率提升。

## 下一阶段与边界

下一阶段只实现两种 K=2：固定左前沿相邻双词元与 global 原 top-2 相邻短链。先检查 K=1/`lambda=0` 等价性、结构动作和真实 logits 对齐；检查通过且完成审阅后，才运行六组各 1033 条、164 个基础题的完整开发/机制评测。该集合已被反复使用，不能称为独立封闭测试。

机器可读主产物在 `analysis_outputs/dreamon_markov_head_training_20260910_v2/`；大型 checkpoint、原始 SQLite bank 和诊断 gzip 保留服务器，不进入普通 Git。
