# DreamOn min4 vs DreamCoder Fixed4 paired result

日期：2026-07-31 UTC

## Comparison boundary

- method A：`DreamOn official-source single-H200 reproduction`，released checkpoint `Dream-org/DreamOn-v0-7B@8ccc7475`，initial=`4`，max=`64`。
- method B：`DreamCoder Fixed4 under DreamOn sampling/decoder`，base checkpoint `Dream-org/Dream-Coder-v0-Base-7B@2346ccd3`，`min_gen_len=max_gen_len=4`。
- same SingleLine non-frozen manifest=`927 rows / 148 clusters`；same pinned generator/evaluator、steps=`256`、temperature=`0.2`、top-p=`0.9`、entropy remasking、seed=`42` 与对应 row seed key。
- checkpoint/training class不同；paired delta 是完整 released system 相对 base fixed-canvas control 的差异，不是纯动态长度、纯训练或 compute-matched effect。

## Absolute / paired result

| Arm | Row Pass@1 | Task macro | 95% cluster CI | Total forwards | Token-forwards | Mean wall / row | Peak H200 memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| DreamOn min4/max64 | `88.4574%` | `78.1111%` | `[72.4789%,83.4275%]` | `12,524` | `3,562,147` | `1.5287s` | `24,032,471,552 B` |
| DreamCoder Fixed4 | `25.1348%` | `19.6466%` | `[16.4233%,23.0425%]` | `3,708` | `838,744` | `0.6193s` | `23,133,082,112 B` |

DreamOn−Fixed4 paired row help/harm=`590/3`；cluster help/harm/tie=`124/0/24`；task-macro delta=`+58.464pp`，10,000 cluster-bootstrap 95% CI=`[+53.091,+63.720]pp`。

Fixed4 final audit：`927/927` exact unique，missing/extra/duplicate/error/failure/forward-accounting/length-accounting=`0`。Fixed4 final length恒为`4`；DreamOn min4 final length mean/p50/p90/range=`7.54/6/13/0..39`，expansion-positive=`725`，contraction-positive=`116`。

Artifacts：`analysis_outputs/dreamon_dreamcoder_singleline_min4_paired_20260731_v1/`。Fixed32/64 未完成前不读取或报告 partial accuracy。Frozen test=`sealed`，`test_evaluation_count=0`。
