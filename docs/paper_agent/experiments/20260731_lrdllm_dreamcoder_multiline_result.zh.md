# LR-DLLM DreamCoder MultiLine result

日期：2026-07-31 UTC。

准确标签：**paper-guided, author-unverified reimplementation of LR-DLLM on DreamCoder, MultiLine 5,079-row / 148-cluster project-non-frozen population**。

Primary absolute：row Pass@1=`31.9945%`；equal-weight 148-cluster macro=`37.1082%`，95% CI=`[32.5411%,41.5975%]`；search/decode/total forwards=`246027/67370/313397`；token-forwards=`76950302`；wall=`13878.672s`；peak=`15788148736` bytes。

Length mean/p50/p90/range=`13.264/9/30/1..128`；expansion/contraction positive=`2518/2448`，total=`7440/7195`；termination=`5063 remaining_length_zero + 16 max_gen_reached`。

MultiLine Fixed64 control正在运行；完成前不计算paired delta/help-harm。完整 absolute artifact：`analysis_outputs/lrdllm_dreamcoder_multiline_primary_grouped_20260731_v1/`。Frozen test=`sealed`，`test_evaluation_count=0`。
