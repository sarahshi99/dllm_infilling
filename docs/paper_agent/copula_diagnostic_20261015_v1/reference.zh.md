# 固定版本文献与适用边界

2026-10-26 阅读：Anji Liu, Oliver Broadrick, Mathias Niepert, Guy Van den Broeck,
Discrete Copula Diffusion, ICLR 2025，arXiv:2410.01949v2，版本日期 2025-03-19。
原文：https://arxiv.org/html/2410.01949v2 ；版本记录：https://arxiv.org/abs/2410.01949v2 。

本任务相关内容为 §4.1 Definition 1、Proposition 2/3/4、Appendix A 式(12)，以及 §5.2 的近似实现。
I-projection 是在指定单位置边缘约束下最小化 KL(Q||R)；正确目标边缘时，KL(T||Q)≤KL(T||R)。
正支持时解为行列乘法缩放，并保留 odds ratios；本文用有限二维 IPF 计算该投影。
论文的 AR 模型结合与近似采样流程未在这里复现，本任务也不额外调用 AR 或模型推理。

理论检查中的 T 是本任务的 fresh 顺序联合参考，只有 Q_T 使用它的真实（候选块内）边缘。
Q 使用 DreamOn 旧近似边缘，通常不满足这一条件。代码正确分布、完整词表概率与候选条件分布也不是同一对象。
因此 Q_T 的数值成功只校验投影算法；Q 的好坏需要实际比较，不能由定理直接推出。
R=old_p1×head_cond2 已有顺序依赖建模。这里不是完整词表 Copula 解码器，不是新论文方法，也没有代码通过率/速度保证。
