# M1 Abductive Program-State Bridge V0 — RandomSpanLight 正式结果

分析器在读取 passed outcome 前由 commit `5a5b5f4` 固定。primary estimand 是 148 个 base task 的 equal-weight macro accuracy；span-micro 仅 descriptive。全部 CI 来自 fixed-seed 10,000 次 task-cluster bootstrap，paired p-value 是 group-aware label-swap；没有行级独立显著性。

完整性：stage-one=`1332/1332`（每 task 8 deployable grid + 1 oracle）、generic=`148/148`、dependency-cone=`148/148`，148 groups 对齐，missing/extra/duplicate/error=`0`；full technical gate passed，frozen=`sealed`、`test_evaluation_count=0`。历史 5079 MultiLine raw 仍是 safely paused/resumable，未恢复。

| Arm | Task macro (95% CI) | Span micro（descriptive） | Standalone forwards | Mean token-forward |
|---|---:|---:|---:|---:|
| fixed64 seed0 | 25.68% [18.92, 33.11] | 25.68% | 64 | 4,096 |
| ordinary confidence best-of-grid | 27.70% [20.95, 35.14] | 27.70% | 512 | 30,720 |
| equal-compute generic remask | 24.32% [17.57, 31.08] | 24.32% | 576 | 34,636.11 |
| M1 score-only selector | 25.68% [18.92, 33.11] | 25.68% | 512 | 30,720 |
| M1 dependency-cone full | 23.65% [16.89, 30.41] | 23.65% | 576 | 34,636.11 |
| oracle ceiling（offline only） | 49.32% [41.22, 57.43] | 49.32% | 64 diagnostic | 1,664.43 |

主要公平比较：M1 full − equal-compute generic=`-0.68pp`，95% CI `[-2.03, 0.00]pp`，wins/losses/ties=`0/1/147`，help/harm=`0/1`，p=`1.0000`。因此 M1 不满足 full-vs-generic point delta>0 且 help>harm 的开发扩展条件。score-only − ordinary confidence=`-2.03pp`，CI `[-9.46, 5.41]pp`，help/harm=`13/16`；full − score-only=`-2.03pp`，CI `[-10.14, 6.08]pp`，help/harm=`16/19`。这些是 fixed comparisons，不触发阈值或融合修改。

Activation audit 表明机制不是纯 no-op，但覆盖很低：dependency cone nonempty/targeted remask=`17/148`（11.49%），fallback=`131/148`（88.51%，其中 97 是 no suffix dependency obligation）；M1 full 与 generic 的候选 hash 仅 `14/148` 不同。故当前结论是 **mechanism sparsely activated and no fair-comparison grouped advantage**，M1 V0 不进入 296/927/5079。完整 compact artifact：`analysis_outputs/m1_randomspanlight_20260717_grouped_v1/`。
