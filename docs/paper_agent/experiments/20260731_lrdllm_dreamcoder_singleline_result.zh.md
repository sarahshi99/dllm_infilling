# LR-DLLM DreamCoder SingleLine result

日期：2026-07-31 UTC。

准确标签：**paper-guided, author-unverified reimplementation of LR-DLLM on DreamCoder, SingleLine 927-row / 148-cluster project-non-frozen population**。截至本结果生成时仍未找到作者代码，因此不得称为 official、paper-faithful 或 exact reproduction。

完整性 gate：`927/927` exact rows，missing/duplicate/error/failure=`0/0/0/0`；12-case technical smoke、64-case mechanism smoke与resume no-op均通过；frozen test=`sealed`，`test_evaluation_count=0`。

冻结 grouped analyzer（10,000 次 base-function cluster bootstrap，seed=`20260731`）结果：

- row-level Pass@1：`72.7077%`；
- equal-weight 148-cluster task-macro Pass@1：`60.5238%`；
- task-macro 95% CI：`[54.9216%,65.8574%]`；
- search/decode/total forwards：`35555/9397/44952`；
- token-forwards：`11288608`；
- wall time：`2631.826s` total，`2.8391s/row` mean；
- peak GPU memory：`15784173568` bytes；
- selected length mean/p50/p90/range：`10.137/7/16/1..128`；
- expansion/contraction positive rows：`465/354`；total counts=`1363/732`；
- termination：`remaining_length_zero=918`，`max_gen_reached=9`。

同 key Fixed64 尚未完成时，不计算 paired delta、paired CI 或 help/harm。完整 grouped artifact：`analysis_outputs/lrdllm_dreamcoder_singleline_grouped_20260731_v1/`。
