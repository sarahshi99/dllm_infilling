# DreamOn min16 vs DreamCoder Fixed16 paired result

日期：2026-07-31 UTC

## Comparison boundary

- method A：`DreamOn official-source single-H200 reproduction`，released checkpoint `Dream-org/DreamOn-v0-7B@8ccc7475`，initial=`16`，max=`64`。
- method B：`DreamCoder Fixed16 under DreamOn sampling/decoder`，base checkpoint `Dream-org/Dream-Coder-v0-Base-7B@2346ccd3`，`min_gen_len=max_gen_len=16`。
- same SingleLine non-frozen manifest=`927 rows / 148 clusters`；same pinned generator/evaluator、steps=`256`、temperature=`0.2`、top-p=`0.9`、entropy remasking、seed=`42` 与对应 row seed key。
- checkpoint/training class不同；paired delta 是完整 released system 相对 base fixed-canvas control 的差异，不是纯动态长度、纯训练或 compute-matched effect。

## Absolute / paired result

| Arm | Row Pass@1 | Task macro | 95% cluster CI | Total forwards | Token-forwards | Mean wall / row | Peak H200 memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| DreamOn min16/max64 | `90.7228%` | `83.1788%` | `[77.9486%,87.9374%]` | `14,806` | `4,287,168` | `1.9014s` | `24,139,347,456 B` |
| DreamCoder Fixed16 | `73.1392%` | `69.2358%` | `[64.1080%,74.1802%]` | `14,783` | `3,520,921` | `1.8501s` | `23,318,941,696 B` |

DreamOn−Fixed16 paired row help/harm=`194/31`；cluster help/harm/tie=`81/9/58`；task-macro delta=`+13.943pp`，10,000 cluster-bootstrap 95% CI=`[+9.618,+18.405]pp`。

Fixed16 final audit：`927/927` exact unique，missing/extra/duplicate/error/failure/forward-accounting/length-accounting=`0`。Fixed16 final length mean/p50/p90/range=`15.94/16/16/4..16`，contraction-positive=`5` rows / `54` moves；DreamOn min16 final length mean/p50/p90/range=`8.62/7/16/0..44`。

Artifacts：`analysis_outputs/dreamon_dreamcoder_singleline_min16_paired_20260731_v1/`。Fixed4/32/64 未完成前不读取或报告 partial accuracy。Frozen test=`sealed`，`test_evaluation_count=0`。
