# DreamOn min64 vs DreamCoder Fixed64 paired result

日期：2026-07-31 UTC

## Comparison boundary

- method A：`DreamOn official-source single-H200 reproduction`，released checkpoint `Dream-org/DreamOn-v0-7B@8ccc7475`，initial=`64`，max=`64`。
- method B：`DreamCoder Fixed64 under DreamOn sampling/decoder`，base checkpoint `Dream-org/Dream-Coder-v0-Base-7B@2346ccd3`，`min_gen_len=max_gen_len=64`。
- same SingleLine non-frozen manifest=`927 rows / 148 clusters`；same pinned generator/evaluator、steps=`256`、temperature=`0.2`、top-p=`0.9`、entropy remasking、seed=`42` 与对应 row seed key。
- checkpoint/training class不同；paired delta 是完整 released system 相对 base fixed-canvas control 的差异，不是纯动态长度、纯训练或 compute-matched effect。

## Absolute / paired result

| Arm | Row Pass@1 | Task macro | 95% cluster CI | Total forwards | Token-forwards | Mean wall / row | Peak H200 memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| DreamOn min64/max64 | `91.6936%` | `85.1299%` | `[80.1271%,89.6563%]` | `9,168` | `2,595,211` | `1.3476s` | `24,258,207,744 B` |
| DreamCoder Fixed64 | `59.1154%` | `63.9190%` | `[58.7273%,68.9037%]` | `58,078` | `16,684,916` | `5.8004s` | `24,258,207,744 B` |

DreamOn−Fixed64 paired row help/harm=`327/25`；cluster help/harm/tie=`90/7/51`；task-macro delta=`+21.211pp`，10,000 cluster-bootstrap 95% CI=`[+15.657,+26.692]pp`。

Fixed64 final audit：`927/927` exact unique，missing/extra/duplicate/error/failure/forward-accounting/length-accounting=`0`。DreamOn min64同样有`927/927` exact unique canonical rows且final canonical audit通过，但append-only failure journal保留1条并发显存不足导致的OOM；该candidate经resume/dedup成功补齐，因此progress按严格“journal必须为空”规则仍记为`audit_failed`，不能写成failure=`0`。Fixed64 source semantics保留EOS contraction，因此final length mean/p50/p90/range=`62.61/64/64/4..64`，contraction-positive=`35` rows / `1,287` moves；DreamOn min64 final length mean/p50/p90/range=`8.87/7/16/0..44`。

Artifacts：`analysis_outputs/dreamon_dreamcoder_singleline_min64_paired_20260731_v1/`。Frozen test=`sealed`，`test_evaluation_count=0`。
