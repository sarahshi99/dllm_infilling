# 2026-10-26 第一批 C：冻结头 K2 Copula CPU 诊断

输入状态：complete；实际 N=256/256，来源题组=220。所有实际完成状态均保留。
输入 manifest SHA256：`7e5805495bc26a7dbfa35a14a54808f32d6e4d4125e4f1d9768b434048d4fb92`。详见 input_reference.json（含原 manifest、ready、分片引用）。

本批回答：在同一候选 A×B 内，硬保 DreamOn 旧边缘是否比现有冻结头 joint R 更接近固定顺序 fresh T。
本批没有回答：真实代码正确率、端到端速度、完整 V² 解码质量、新头训练或 H3/K4 融合效果。T 不是正确代码分布。

## 主表（KL 单位 nats，状态加权均值）

|分布|KL(T‖·)|TV(T,·)|teacher 行边缘加权条件 KL|
|---|---:|---:|---:|
|I|0.415724|0.155517|0.282262|
|R|0.261968|0.118936|0.179168|
|Q|0.374719|0.138374|0.241259|
|Q_T|0.0795482|0.0491751|0.0795482|

完整均值、配对差、来源题组 bootstrap 95% CI、全部分层见 main_table.csv / stratified_table.csv / high_coverage_table.csv。
CI 使用 2000 次来源题组重采样（固定 seed=20261015），同母状态的派生样本整体保留；属于开发探索证据。

高覆盖子集：N=190、来源题组=170；达到预声明样本下限。
阈值固定为 old_p1/p2 候选质量≥.95 且 R_raw/T_raw 块质量均≥.90；这是诊断选择，不是理论充分条件。

## 判断与候选下一步

Q在整体与高覆盖均更差：强保旧边缘在这些状态不合适；这不否定所有依赖建模。软约束等留下一批决定。

算法审计：Q失败=0，Q_T失败=0，适用理论条件下oracle违反=0；CPU合计=0.467167秒。
只有 Q_T 使用 T 的块内边缘，因此是不可部署的 teacher-oracle 检查。旧边缘 Q 不自动继承 KL 改善保证。
KL 链式分解 Δjoint=Δrow+Δconditional；主表及分层同时给出这两项。若只有 Δrow 改善，应视为首位截断修正证据。
order/span 分层方向与置信区间完整发布；样本不足或方向不一致时不推荐部署。

## 概率与支持边界

权威输入是完整词表 log_softmax 直接抽出的 log 字段；float64/log-space 计算。先记录 old_p1×cond2 的块质量，再全块归一化。
逐行先归一化 cond2 会改变 joint 定义，本分析没有这样做。R 与 T 的相同候选不代表相同覆盖。
Q 是 IPF(R;r,c)，Q_T 是 IPF(R;T.row,T.col)，误差上限1e-6、最多1000轮；不加epsilon。
真实零支持、概率下溢、FP64呈现下溢、最大轮数、迭代数、最大边缘误差、特殊质量和所有尾质量见 probability_audit.json / per_state.jsonl。
非有限或失败状态不删：主表会给 unavailable，避免只报成功子集。覆盖不足不能肯定或否定完整词表方法。

## 复现与限制

NumPy/标准库 CPU 分析，无 GPU、训练、生成或模型加载。模型/头/adapter SHA 和来源 split 均随输入引用归档。
逐状态 CSV/JSONL 可移植；r/c、R/T 截断边缘与所有条件候选质量均保留。
文献固定为 [Discrete Copula Diffusion v2 (2025-03-19)](https://arxiv.org/html/2410.01949v2)，§4.1及附录A；本批不复现其AR推理流程。
不以 Q 胜 I 替代 Q/R 比较，不自动搜索软约束参数，不开启第二批。
`reviewer_gate_disabled`：本地串行审查与fresh CPU验证；没有独立reviewer，实数据结论仍受候选/开发来源限制。
