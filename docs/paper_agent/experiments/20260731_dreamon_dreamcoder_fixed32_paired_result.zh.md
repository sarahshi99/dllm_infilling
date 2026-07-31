# DreamOn min32 vs DreamCoder Fixed32 paired result

日期：2026-07-31 UTC

## Comparison boundary

- method A：`DreamOn official-source single-H200 reproduction`，released checkpoint `Dream-org/DreamOn-v0-7B@8ccc7475`，initial=`32`，max=`64`。
- method B：`DreamCoder Fixed32 under DreamOn sampling/decoder`，base checkpoint `Dream-org/Dream-Coder-v0-Base-7B@2346ccd3`，`min_gen_len=max_gen_len=32`。
- same SingleLine non-frozen manifest=`927 rows / 148 clusters`；same pinned generator/evaluator、steps=`256`、temperature=`0.2`、top-p=`0.9`、entropy remasking、seed=`42` 与对应 row seed key。
- checkpoint/training class不同；paired delta 是完整 released system 相对 base fixed-canvas control 的差异，不是纯动态长度、纯训练或 compute-matched effect。

## Absolute / paired result

| Arm | Row Pass@1 | Task macro | 95% cluster CI | Total forwards | Token-forwards | Mean wall / row | Peak H200 memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| DreamOn min32/max64 | `91.2621%` | `83.0372%` | `[77.6890%,87.8429%]` | `14,318` | `4,041,982` | `1.8537s` | `24,018,886,656 B` |
| DreamCoder Fixed32 | `63.2147%` | `63.9771%` | `[58.8659%,69.0882%]` | `29,457` | `7,490,056` | `3.1676s` | `23,576,967,680 B` |

DreamOn−Fixed32 paired row help/harm=`286/26`；cluster help/harm/tie=`89/9/50`；task-macro delta=`+19.060pp`，10,000 cluster-bootstrap 95% CI=`[+13.661,+24.447]pp`。

Fixed32 final audit：`927/927` exact unique，missing/extra/duplicate/error/failure/forward-accounting/length-accounting=`0`。Fixed32 final length mean/p50/p90/range=`31.76/32/32/4..32`，contraction-positive=`11` rows / `218` moves；DreamOn min32 final length mean/p50/p90/range=`8.70/7/16/0..44`。

Artifacts：`analysis_outputs/dreamon_dreamcoder_singleline_min32_paired_20260731_v1/`。Fixed64 未完成前不读取或报告 partial accuracy。Frozen test=`sealed`，`test_evaluation_count=0`。
