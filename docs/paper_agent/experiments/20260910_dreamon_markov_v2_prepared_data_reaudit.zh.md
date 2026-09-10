# DreamOn Markov v2 启动前数据复算审计

- 日期：2026-09-10 UTC
- 阶段：新 transition bank 与 TV/KL 正式训练之前。
- 数据：固定 `OpenCoder-LLM/opc-sft-stage2` 的 `educational_instruct@7d28f40d...`；HumanEval-SingleLineInfilling 仅用于离线排除。
- 模型/词表：DreamOn `8ccc747...` 的本地 tokenizer；本动作不加载模型权重，不使用 GPU。
- 比较类型：artifact integrity / decontamination maintenance audit，不是方法效果比较。
- 假设：已经生成的 v2 tokenized 数据应当等于重新计算后的 clean source records 的合法 tokenization 子集，且不包含任何直接 HumanEval 候选或候选所在传递关联组。
- 检查：源记录与 HumanEval 计数、标准化去重、所有变体一致性、候选及组闭包排除、record/source/group/split 对照、manifest 精确一致、row seed、token 数量与长度、token id 词表边界、组跨 split、external-test 子集。
- 成功条件：所有检查错误计数为零；既有摘要计数与复算一致。
- 停止条件：任何候选/关联组进入 prepared，任何组跨 split，任何 manifest/主键/token 不一致。出现时不得启动 GPU bank。
- 已知边界：该检查只能证明本项目已定义的精确文本、AST 和同入口断言保守规则被正确执行，不能证明 DreamOn 基础模型预训练未见过 HumanEval；20万/2万/2万 transition 配额只能在 GPU bank 生成后核查。
- 预期输出：`analysis_outputs/dreamon_markov_head_training_20260910_v2/prepared_data_reaudit.json`，以及对应测试和运行日志。
