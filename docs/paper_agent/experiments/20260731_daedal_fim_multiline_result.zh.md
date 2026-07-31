# CAL authors’ DAEDAL FIM adaptation — MultiLine result

日期：2026-07-31 UTC。

准确标签：**CAL authors’ DAEDAL FIM adaptation on the 4,990-row / 143-cluster project-non-frozen MultiLine CAL-Rest common subset**。不得称为原始 DAEDAL 官方 FIM，也不是完整5,715-case reproduction。

Dynamic：row=`11.9238%`；143-cluster macro=`11.0955%`，CI=`[9.5340%,12.7175%]`；forwards/token-forwards=`41746/10224265`；wall=`13815.497s`；peak=`17829322240` bytes。Length mean/p50/p90/range=`12.289/9/19/8..128`；expansion-positive=`2754`，total=`21401`；termination=`4976 filled + 14 max_length`。

Fixed8：row=`11.1022%`；macro=`10.6402%`，CI=`[8.8762%,12.6995%]`；forwards/token-forwards=`25990/6330150`；wall=`8910.884s`；peak=`17733972992` bytes。

Dynamic−Fixed8 paired：row help/harm=`97/56`（row delta=`+0.8216pp`）；cluster help/harm/tie=`35/15/93`；task-macro delta=`+0.4553pp`，95% paired cluster CI=`[-1.3749,+1.7913]pp`。CI跨0；Fixed8不称equal-compute。

完整 artifact：`analysis_outputs/daedal_fim_multiline_paired_20260731_v1/`。Frozen test=`sealed`，`test_evaluation_count=0`。
