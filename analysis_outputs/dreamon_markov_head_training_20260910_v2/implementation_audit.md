# Implementation audit

- DreamOn is frozen bf16 inference; Markov softmax/loss is float32.
- Target logits use the released one-position shift by selecting hidden state `target-1` before `lm_head`.
- Markov correction is applied after action masking and before temperature/top-p.
- Structural token correction is exactly zero.
- TV and KL use the same bank, initialization, optimizer, schedule, batch, seed, GPU, and gates.
- TV and KL ran in separate sequential processes.
- 训练数据来自固定 OpenCoder educational_instruct；完整源数据先按保守规则排除 HumanEval 候选关联组，prepared 与实际 transition bank 的成员审计均通过。该结论不证明基础模型预训练未见过 HumanEval。
- No full-vocabulary logits were serialized.
- Reviewer/subagent gate is disabled by repository policy; local diff review and fresh verification are used.
