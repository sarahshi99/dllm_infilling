# LR-DLLM DreamCoder RandomSpan result

日期：2026-07-31 UTC。

准确标签：**paper-guided, author-unverified reimplementation of LR-DLLM on DreamCoder, RandomSpan 1,480-row / 148-cluster project-non-frozen population**。

Primary absolute：row Pass@1=`18.3784%`；equal-weight 148-cluster macro=`18.3784%`，95% CI=`[16.0135%,20.7432%]`；search/decode/total forwards=`76442/20618/97060`；token-forwards=`20649706`；wall=`6976.712s`；peak=`15778112512` bytes。

Length mean/p50/p90/range=`13.931/8/28/1..128`；expansion/contraction positive=`847/725`，total=`3604/2801`；termination=`1454 remaining_length_zero + 26 max_gen_reached`。

RandomSpan Fixed64 control正在运行；完成前不计算paired delta/help-harm。完整 absolute artifact：`analysis_outputs/lrdllm_dreamcoder_randomspan_primary_grouped_20260731_v1/`。Frozen test=`sealed`，`test_evaluation_count=0`。
