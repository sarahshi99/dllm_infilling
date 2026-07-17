# M1--M4 方法组合当前登记

更新时间：2026-07-17 UTC。唯一机器可读登记是 `method_portfolio.current.json`；本页是其中文审计摘要。历史报告保留原结论，但不得覆盖这里的当前状态。

| 方法 | 当前状态 | 公平主要比较 | 下一动作 |
|---|---|---|---|
| M1 Abductive Program-State Bridge | 148 reviewed_not_promoted；历史 5079 safely paused/resumable | full vs equal-compute generic（576 vs 576） | 保留 fixed negative/weak evidence，不扩展 |
| M2 Constraint-Homotopy | reviewed_not_promoted | gradual/abrupt vs vanilla（64 vs 64） | 停止本 V0 的扩展 |
| M3 Birth--Death Canvas Diffusion | 148 reviewed_not_promoted | birth-death vs uniform（256 forwards；非 equal-token） | 保留 lower-token/lower-accuracy evidence，不扩展 |
| M4 Semantic Particle Assembly | offline 148 complete，GPU repair safe-slot supervised | assembly-without-repair vs best-single（512 vs 512） | CAL t+10 + ECC=0 + >=25GiB + 无第三研究进程后进入 smoke→148 |

没有论文主方法。M1--M4 是平行独立候选；M2 的负/弱结果不阻止 M3/M4，M1 最先实现也不构成选择。共同路线固定为 `12 smoke → 148 RandomSpanLight → 296 MultiLine-Core → 927 non-frozen SingleLine development → selected-method-only 5079 MultiLine`。只有按各自公平比较满足当前 JSON 所列 promotion 条件的至多一个候选可进入扩展。

所有 deployable 方法仅可使用 prefix、suffix、当前候选与 inference-visible 信号；不得读取 reference/canonical solution、tests、verifier/passed outcome、task ID/group、split label 或 oracle length。Frozen controller test 始终 `sealed`，`test_evaluation_count=0`。

历史边界：`M1-D0/F3/F4` 是固定 proxy/ranking audit，`M4-D0/F1` 是 premise diagnostic，`A1/F2` 是负辅助诊断；它们都不是当前四条完整方法路线。
