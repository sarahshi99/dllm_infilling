# M3 Birth--Death Canvas Diffusion V0 — RandomSpanLight 正式结果

分析器在读取 passed outcome 前由 commit `5a5b5f4` 固定。primary estimand 是 148 个 base task 的 equal-weight macro accuracy；span-micro 仅 descriptive。全部 CI 来自 fixed-seed 10,000 次 task-cluster bootstrap，paired p-value 是 group-aware label-swap。

完整性：uniform、birth-death 均 `148/148`，row-key/group 对齐，missing/extra/duplicate/error=`0`，每 row=`256` forwards，full technical gate passed；frozen=`sealed`、`test_evaluation_count=0`。这是 equal-forward 比较，**不是** equal-token 比较。

| Arm | Task macro (95% CI) | Span micro（descriptive） | Forwards | Mean token-forward | Mean wall sec |
|---|---:|---:|---:|---:|---:|
| uniform fixed-grid | 29.73% [22.30, 37.16] | 29.73% | 256 | 15,360 | 30.91 |
| birth-death canvas | 25.68% [18.92, 32.43] | 25.68% | 256 | 11,532.11 | 23.39 |

主要公平 paired comparison（birth-death − uniform）=`-4.05pp`，95% CI `[-10.14, 2.03]pp`，wins/losses/ties=`8/14/126`，help/harm=`8/14`，p=`0.2888`。birth-death 的 mean token-forward 比 uniform 低约 24.9%，但其 accuracy point estimate 更低；因此它既不满足 accuracy promotion，也不满足预先规定的“accuracy point estimate 不低于 uniform”的 efficiency promotion。

Activation 明确发生：candidate hash 在 `76/148` rows 不同，所有 148 rows 都发生 birth/death，合计 444 events（round 15/31/47 各 148），uniform events=`0`；birth-death token-forward min/mean/max=`8192/11532.11/24832`。因此结论不是 no-op，而是 **population reallocation activated but current V0 has lower grouped accuracy despite lower token cost**。M3 V0 不进入 296/927/5079。完整 compact artifact：`analysis_outputs/m3_birth_death_canvas_20260717_grouped_v1/`。
