# LR-DLLM DreamCoder RandomSpan result

日期：2026-07-31 UTC。

准确标签：**paper-guided, author-unverified reimplementation of LR-DLLM on DreamCoder, RandomSpan 1,480-row / 148-cluster project-non-frozen population**。

Primary absolute：row Pass@1=`18.3784%`；equal-weight 148-cluster macro=`18.3784%`，95% CI=`[16.0135%,20.7432%]`；search/decode/total forwards=`76442/20618/97060`；token-forwards=`20649706`；wall=`6976.712s`；peak=`15778112512` bytes。

Same-key Fixed64 absolute：row/macro Pass@1=`33.5135%`，95% CI=`[30.5405%,36.4865%]`；search/decode/total forwards=`0/94720/94720`；token-forwards=`22285184`；wall=`4686.681s`；peak=`15817324032` bytes。

Length mean/p50/p90/range=`13.931/8/28/1..128`；expansion/contraction positive=`847/725`，total=`3604/2801`；termination=`1454 remaining_length_zero + 26 max_gen_reached`。

Same-key paired primary−Fixed64：row help/harm=`104/328`；cluster help/harm/tie=`19/107/22`；task-macro delta=`-15.135pp`，10,000 cluster-bootstrap 95% CI=`[-17.905,-12.365]pp`。该 CI 完全低于 0，是 RandomSpan 上相对 Fixed64 的明确负结果；不与 SingleLine/MultiLine 合并 Mean，也不把 Fixed64 称为 compute-matched。

完整 paired artifact：`analysis_outputs/lrdllm_dreamcoder_randomspan_paired_20260731_v1/`。Frozen test=`sealed`，`test_evaluation_count=0`。
