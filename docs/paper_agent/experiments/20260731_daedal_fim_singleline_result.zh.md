# CAL authors’ DAEDAL FIM adaptation — SingleLine result

日期：2026-07-31 UTC。

准确标签：**CAL authors’ DAEDAL FIM adaptation on the 838-row / 143-cluster project-non-frozen SingleLine CAL-Rest subset**。不得称为原始 DAEDAL 官方 FIM。

Dynamic：row=`51.4320%`；143-cluster macro=`37.9961%`，CI=`[33.0533%,42.8892%]`；forwards/token-forwards=`6480/1601495`；wall=`3802.091s`；peak=`17751597056` bytes。Final length mean/p50/p90/range=`11.987/8/20/8..128`；expansion-positive=`368`，total expansions=`3341`；termination=`835 filled + 3 max_length`。

Same-decoder Fixed8：row=`50.8353%`；macro=`37.5859%`，CI=`[32.7016%,42.4873%]`；forwards/token-forwards=`3817/928151`；wall=`1106.634s`；peak=`17733972992` bytes。

Dynamic−Fixed8 same-key paired：row help/harm=`54/49`（row delta=`+0.5967pp`）；cluster help/harm/tie=`27/26/90`；task-macro delta=`+0.4102pp`，95% paired cluster CI=`[-2.0624,+2.7344]pp`。CI跨0。Fixed8不称equal-compute。

完整 artifact：`analysis_outputs/daedal_fim_singleline_paired_20260731_v1/`。Frozen test=`sealed`，`test_evaluation_count=0`。
