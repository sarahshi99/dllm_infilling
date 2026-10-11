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

## 已完成检查

- 合成单元检查：`dreamon-repro` 环境下 36 项通过。覆盖 global top-2 固定后空间左到右、非相邻 fallback、K=1 head 旁路、左前沿相邻第二词元修正、修正后结构词元停止、`lambda=0` 不调用 head 且不增加采样调用。
- 真实模型小样本：`DreamOn-v0-7B@8ccc747...`、KL `best.pt`、1 条 SingleLine、六组均完成（`6/6`）。两个 K=2 无头组均为 9 个普通词元/5 次 DreamOn 前向；两个 KL 组也为 9/5，并各记录 4 次 Markov 启用。输出在服务器 `analysis_outputs/dreamon_markov_k2_real_smoke_20261010_v1/`。
- 真实模型 `lambda=0`：同一条、同一 seed 下，global 与左前沿两组“加 KL 头”分别与其匹配无头对照的 completion 和逐步 trace 相同（忽略方案标签字段），`markov_enabled_count=0`。输出在 `analysis_outputs/dreamon_markov_k2_lambda0_smoke_20261010_v1/`。
- 环境说明：系统 Python 因缺少 `libcudnn.so.9` 无法加载 PyTorch；所有上述验证均使用训练时的 `/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python`。
