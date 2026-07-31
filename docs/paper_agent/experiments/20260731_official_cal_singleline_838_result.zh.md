# Official-source CAL SingleLine 838 result

日期：2026-07-31 UTC。

准确标签：**official-source CAL, initial length 32, on the 838-row / 143-cluster project-non-frozen SingleLine CAL-Rest subset**。这不是完整 1,033-case reproduction，也不代表 CAL 论文四个初始长度的整表。

完整性 gate：`838/838` exact rows，missing/duplicate/error/failure=`0/0/0/0`；12-case smoke与resume no-op通过；frozen test=`sealed`，`test_evaluation_count=0`。

冻结 grouped analyzer（10,000 次 base-function cluster bootstrap，seed=`20260731`）结果：

- row-level Pass@1：`52.3866%`；
- equal-weight 143-cluster task-macro Pass@1：`41.2711%`；
- task-macro 95% CI：`[36.3798%,46.0499%]`；
- search/decode/total forwards：`13226/25504/38730`；
- token-forwards：`10548607`；
- wall time：`3105.563s` total，`3.7059s/row` mean；
- peak GPU memory：`16560579584` bytes；
- selected length mean/p50/p90/range：`30.434/31/39/3..72`；
- expansion/contraction positive rows：`323/443`；total counts=`1872/3184`；
- pinned upstream adapter未暴露更细 event-level termination reason，838 rows统一登记为`upstream_event_level_reason_not_exposed`。

同 key `official_fixed32` 尚未完成时，不计算 paired delta、paired CI 或 help/harm。Fixed64 sensitivity 也不称 equal-compute。完整 grouped artifact：`analysis_outputs/official_cal_singleline_primary_grouped_20260731_v1/`。
