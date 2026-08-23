# DreamOn SingleLine：解码顺序 × 并行度 × Markov 前提诊断（v2）

状态：completed / valid-v2。数据是 DreamOn 发布 loader 返回的 `1033` 条 SingleLine development/full-allowed population（`164` 个 HumanEval task groups），不是 held-out 或 frozen-test 证据。六臂均为同一 DreamOn-v0-7B、seed 42、max64、entropy、temperature 0.2、top-p 0.9、expand/delete 协议与 evaluator。

## 运行完整性与 v1 作废说明

v2 完成 `6198 = 1033 × 6` 个 case×variant，0 duplicate，六臂共享 1033 个 task id，39,686 条 step trace 与 25,467 条 L1 stale→fresh transition，failure journal 不存在，进程退出码为 0。详情见 `completeness_audit.json`。

2026-08-22 的 v1 也运行完整，但 L 系列遗漏了 released `pad_eos_to_right` 的 broadcast-delete 语义，使 L1 平均 forward 异常升至 65.4（C1 为 9.9）。v1 仅保留作实现审计，完全不参与本报告的结果、CI 或决策。v2 恢复官方 delete 行为后，C1 真实对齐已验证，且 v2 单例 smoke 的 C1/L1、C2/L2、C4/L4 forward 分别为 10/5/3。

## 质量

| Variant | Pass@1 |
| --- | ---: |
| C1 global-K1 | 951/1033 = 92.06% |
| C2 global-K2 | 901/1033 = 87.22% |
| C4 global-K4 | 814/1033 = 78.80% |
| L1 left-frontier-K1 | 942/1033 = 91.19% |
| L2 left-frontier-K2 | 877/1033 = 84.90% |
| L4 left-frontier-K4 | 713/1033 = 69.02% |

C1 精确匹配历史 DreamOn SingleLine 参考 `951/1033≈92.06%`，因此本环境、1033 loader 与核心协议没有出现可见漂移。

按 164 个 task group 聚类：C2−C1 为 `−4.57pp`，95% CI `[-7.17,-2.11]`；C4−C1 为 `−14.56pp`，CI `[-18.50,-10.83]`。左前沿内部的代价也显著：L2−L1 为 `−6.36pp`，CI `[-9.85,-2.98]`；L4−L1 为 `−24.06pp`，CI `[-28.23,-19.96]`。

同 K 比较中，L1−C1 为 `−0.93pp`，CI `[-2.86,+0.97]`，help/harm=`6/15`，没有可靠差异；L2−C2 为 `−2.72pp`，CI `[-5.98,+0.37]`，help/harm=`20/44`，没有支持 L2 优于 C2；L4−C4 为 `−10.42pp`，CI `[-13.53,-7.62]`，help/harm=`10/111`，明确反对固定左到右 K=4。

失败输出中有 318 个可由 evaluator 文本保守识别为 syntax/compile，682 个为 functional-or-other；runner 没有保存独立 compile 布尔值，故不把这类字符串分类误称为完整失败 taxonomy。

## 有效并行度与效率

| Variant | forwards/case mean | wall s/case mean | normal commits/forward | selected/forward |
| --- | ---: | ---: | ---: | ---: |
| C1 | 9.87 | 0.332 | 0.90 | 1.00 |
| C2 | 5.48 | 0.234 | 1.64 | 1.98 |
| C4 | 4.78 | 0.222 | 1.96 | 3.73 |
| L1 | 10.21 | 0.398 | 0.90 | 1.00 |
| L2 | 5.25 | 0.227 | 1.71 | 2.00 |
| L4 | 2.82 | 0.184 | 3.10 | 4.00 |

因此不存在“requested K>1 实际坍缩为 1”的总体解释：C2 的实际 normal commit 2 的比例为 75.98%，L2 为 80.97%；C4 的 normal commit 4 比例为 37.42%，L4 为 64.63%。C4 的较低 normal commit 主要来自 structural action，不是配置虚设；L4 的结构屏障使实际 commit 更接近 K，但并未保住质量。

K=2/4 确实减少 forward 与 wall time，但没有质量—效率 Pareto 改进：每个 K>1 相对其 K=1 对照都降低 Pass@1。L1 相对 C1 更慢且点估计更低；L2 相比 C2 有略少 forward 和略多 normal commits，却以更低质量交换；L4 既比 C4 质量差，也没有支持固定大块的理由。

## 官方 global top-K 的空间结构

在 C1 的同一 fresh state 上：global top-2 与最左连续 top-2 完全相同的比例为 89.49%，平均 0.93 个相邻 pair、平均 span 2.12、平均左前缀覆盖 1.88/2。global top-4 完全等于最左连续 top-4 的比例为 76.40%，平均 2.74 个相邻 pair、最长连续 run 均值 3.69、span 均值 4.51、左前缀覆盖 3.63/4。

这回答第一个问题：官方 global confidence 通常选择左侧局部成簇的候选，而不是完全分散；但 top-2 仍有约 10.5%、top-4 有约 23.6% 的 step 不等于左连续前缀，不能把它简化为固定左到右。完整分层数据在 `topk_spatial_summary.csv`。

## Markov 前提诊断（L1）

L1 在连续普通 token chain 上提供全词表 stale→fresh 对比：offset 1/2/3 的 transition 数分别为 9516/8489/7462，覆盖 1027/1027/1026 case，均覆盖 164 task groups。

| Offset | top-1 agreement | TV mean (median) | fresh−stale entropy | fresh top-1 stale rank |
| --- | ---: | ---: | ---: | ---: |
| 1 | 93.97% | 0.103 (0.029) | -0.261 | 1.18 |
| 2 | 87.97% | 0.261 (0.191) | -0.827 | 1.42 |
| 3 | 82.10% | 0.365 (0.308) | -1.144 | 2.11 |

新鲜分布更尖锐、margin 更大，且 stale/fresh 差异随 offset 快速增加；早期的 offset-1 TV 为 0.086，中期 0.150，晚期 0.200（晚期样本数较小）。这支持“stale logits 存在可被更新的条件信息”，并反对直接用固定 K=4 把一阶修正外推到长块。

但本 run **没有记录 reference token 的 stale→fresh log-prob/rank/accuracy**，也没有记录右邻进入 fresh global top-2/top-4 的事件。因此无法判断变化是否系统性朝正确方向，不能从这些数字授权训练 DSpark 风格 Markov head。当前研究决定是 **reframe / stop（Markov head training）**：如需恢复该路线，必须预先记录 oracle-only reference direction 与 fresh global-rank promotion，且不得进入生成选择。

## 决策

- 支持：DreamOn global confidence 的候选大多是左侧局部成簇，K=2 存在真实可用并行空间。
- 削弱：硬固定左到右不是可靠的同 K 质量提升；L2 未击败 C2，L4 显著劣于 C4。
- 否定：不能用“实际并行度必坍缩到 1”解释 K>1 的表现；L4 实际 normal commit 为 3.10/forward。
- 不授权：在没有正确方向 oracle 证据及 promotion 记录时，不训练 Markov head，不提出自适应 controller。

本轮是完整允许开发集的机制/证伪诊断，不是外部泛化或冻结测试结论。
