# Official-source CAL MultiLine 4,990 Result

准确标签：**official-source CAL, initial length 32, on the 4,990-row / 143-cluster project-non-frozen CAL-Rest common subset**。

Analyzer freeze：`e7a790beb2c7e17bfaa5cdc764c5b82dd05ebd83`。完整 grouped artifact：`analysis_outputs/official_cal_multiline_4990_grouped_20260731_v1/`。

## Result

- row Pass@1：`1643/4990 = 32.9259%`
- equal-weight 143-cluster task-macro Pass@1：`27.8471%`
- 10,000 cluster-bootstrap 95% CI：`[24.3313%,31.2715%]`
- integrity：4990 exact unique rows；missing/extra/duplicate/error/failure=`0/0/0/0/0`
- total search/decode/forward calls：`76,807 / 156,841 / 233,648`
- total token-forwards：`63,545,173`
- mean wall：`3.8762 s/row`；per-case wall sum=`19,342.38 s`
- peak GPU memory：`15.42 GiB`
- selected length：mean=`31.4311`、p50=`32`、p90=`40`、range=`2..77`
- final length vs initial32：contracted/unchanged/expanded=`2418/453/2119`

同 4,990 normalized keys 上没有 completed official_fixed32，因此不报告 paired delta、paired CI 或 help/harm。该结果不是完整 5,715-case reproduction，也不代表论文四种 initial lengths 的整表。

Offline quartile buckets 显示 q1/q2/q3/q4 row Pass@1=`57.31%/50.72%/21.18%/2.33%`；这是 reference-length descriptive analysis，不参与方法。

历史 raw 没有 event-level expansion/contraction 或 termination reason；只报告最终 length relation，不伪造内部事件分布。

Frozen controller test 保持 `sealed`，`test_evaluation_count=0`。
