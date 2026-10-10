# Markov K=2 推理实现与等价性检查

- 日期：2026-10-10 UTC
- 阶段：在已完成的 v2 数据审计、transition bank 审计及 TV/KL 训练之后；尚未启动 HumanEval-SingleLine 全量生成。
- 目的：实现两种固定的单行双词元解码设计，并用合成状态和真实模型小样本检查其与无头路径的等价性边界。
- 审稿人关注点：K=2 的速度收益只有在第二个词元未额外执行 DreamOn 前向、结构动作不复用失效坐标、且无头对照完全匹配时才可解释。
- 固定配置：v2 选定 KL head、`lambda=1`、DreamOn `8ccc747...`、官方 v2 采样（temperature=0.2、top-p=0.9）、SingleLine 固定逐题种子。头和主模型均只推理。
- 设计 A：固定左前沿；最左 mask 的原分布采样为左词元，只有左词元普通且右邻仍为 mask 时，对右邻 logits 加 Markov bias 后采样。
- 设计 B：先按原置信度固定 global top-2；只有空间相邻时按左到右处理并修正右位。非相邻时保留原 global K=2 无头行为。
- 实现位置：`experiments/dreamon_singleline_order_parallel.py`、`experiments/run_dreamon_singleline_order_parallel.py` 及对应单元测试。
- 成功条件：K=1 旁路 head；`lambda=0` 直接走匹配无头路径；偏置加在 official logits 对齐和动作 mask 后、temperature/top-p 前；第二词元只条件于实际左样本；扩展、EOS 广播删除及“修正后才采到结构词元”均有断言覆盖。
- 停止条件：任何等价性、动作语义或真实 logits 对齐检查失败；失败时不启动 1033 条评测，也不调参或替换 KL/lambda。
- 产出：可运行的双词元实现、精确的无头匹配对照、合成/小样本检查记录。全量评测仅在这些检查通过并经本轮审阅后启动。
