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

## Fixed controls and paired comparison

`official_fixed32`：row=`48.3294%`；143-cluster macro=`39.3731%`，CI=`[34.6495%,44.0732%]`；forwards/token-forwards=`26816/7338528`；wall=`3652.148s`；peak=`16552904704` bytes。

Primary−Fixed32 same-key paired：row help/harm=`130/96`（row delta=`+4.0573pp`）；cluster help/harm/tie=`47/31/65`；task-macro delta=`+1.8980pp`，95% paired cluster CI=`[-2.2332,+5.7591]pp`。CI跨0。

`project_fixed64_internal` sensitivity：row=`47.8520%`；macro=`39.5620%`，CI=`[34.7851%,44.3053%]`；forwards/token-forwards=`53632/16393280`；wall=`5912.041s`。Primary−Fixed64 row help/harm=`151/113`，cluster=`45/32/66`，macro delta=`+1.7091pp`，CI=`[-2.2374,+5.5446]pp`。Fixed64不称equal-compute或compute-matched。

完整 paired artifact：`analysis_outputs/official_cal_singleline_paired_grouped_20260731_v1/`。
