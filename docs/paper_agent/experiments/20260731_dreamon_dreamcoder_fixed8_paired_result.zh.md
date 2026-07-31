# DreamOn min8 vs DreamCoder Fixed8 paired result

日期：2026-07-31 UTC

## Comparison boundary

- method A：`DreamOn official-source single-H200 reproduction`，released checkpoint `Dream-org/DreamOn-v0-7B@8ccc7475`，initial=`8`，max=`64`。
- method B：`DreamCoder Fixed8 under DreamOn sampling/decoder`，base checkpoint `Dream-org/Dream-Coder-v0-Base-7B@2346ccd3`，`min_gen_len=max_gen_len=8`。
- same project-non-frozen SingleLine manifest=`927 rows / 148 clusters`；same pinned source generator/evaluator、steps=`256`、temperature=`0.2`、top-p=`0.9`、entropy remasking、global seed=`42` 与对应 row seed key。
- checkpoint/training class 不同：A 是 released training-based DreamOn system，B 是 DreamCoder base fixed-canvas control。因此 paired delta 是完整系统差异，不是纯动态长度、纯训练或 compute-matched effect。

## Absolute / paired result

| Arm | Row Pass@1 | Task macro | 95% cluster CI | Total forwards | Token-forwards | Mean wall / row | Peak H200 memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| DreamOn min8/max64 | `90.3991%` | `81.3313%` | `[75.8954%,86.2925%]` | `13,199` | `3,793,167` | `1.7011s` | `24,047,672,320 B` |
| DreamCoder Fixed8 | `60.7335%` | `48.0879%` | `[42.6582%,53.5911%]` | `7,416` | `1,707,152` | `1.0659s` | `23,195,216,896 B` |

DreamOn−Fixed8 paired row help/harm=`286/11`；cluster help/harm/tie=`111/1/36`；task-macro delta=`+33.243pp`，10,000 cluster-bootstrap 95% CI=`[+28.390,+38.417]pp`。

Fixed8 final audit：`927/927` exact unique，missing/extra/duplicate/error/failure/forward-accounting/length-accounting=`0`。Fixed8 final length 恒为`8`；DreamOn min8 final length mean/p50/p90/range=`8.08/7/13/0..39`，expansion-positive=`360`，contraction-positive=`573`。

Artifacts：`analysis_outputs/dreamon_dreamcoder_singleline_min8_paired_20260731_v1/`。其他 Fixed4/16/32/64 arms 未完成前不读取或报告其 partial accuracy。Frozen test=`sealed`，`test_evaluation_count=0`。
