# DreamOn SingleLine Markov 前提补充诊断 v3：行动说明

本轮只运行两个逐词刷新诊断：全局置信度逐词刷新、固定左前沿逐词刷新。它们分别服务于未来的“global top-K 内连续短链 + Markov”和“强制固定左到右 + Markov”设计，但本轮不训练 Markov head，也不自动授权下一阶段。

v2 的 2026-08-23 六臂质量/效率结果继续有效；其 Markov stop 判断由于 reference direction 与 fresh global-rank promotion 两个预注册字段未采集，被本轮补充诊断取代。缺失字段不解释为负面结果。

实现先用测试固定 logits shift、raw/actual decode 分布、top-p 语义、RNG 不变性、reference poison 无泄漏、结构动作终止 chain、schema 与硬门禁。随后执行固定 smoke12；只有轨迹逐步复现 v2 C1/L1 且所有字段门禁通过，才自动启动 1033x2 full。统计以 164 个基础 task group 聚类 bootstrap，offset 1/2/3 分开报告，reference-prefix 完全对齐子集为主要解释。

完整命令、环境、输出路径、成功与终止标准见 `docs/paper_agent/current_action.md`；最终权威结果写入 `analysis_outputs/dreamon_markov_premise_rerun_20260830_v3/` 并推送到新分支。

## 完成状态

2026-08-30 full 已完成 `2066/2066`，所有程序级硬门禁通过，C1/L1 Pass@1 精确复现 `951/1033` 与 `942/1033`。主结果是 mixed-positive：reference-prefix aligned 的 offset-1 raw reference log-prob 在两项诊断中均显著改善，但 actual-decode finite log-prob 的 offset-1 CI 均跨 0；全局置信度轨迹的连续右邻与 top-K retention 很高，offset=2/3 的 TV 与 top-1 disagreement 则显示长块风险。没有训练 Markov head，也没有自动授权下一阶段。
