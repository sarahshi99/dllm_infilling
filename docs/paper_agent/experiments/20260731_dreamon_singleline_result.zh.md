# DreamOn official-source SingleLine result

日期：2026-07-31 UTC

准确标签：**DreamOn official-source single-H200 reproduction**。

该结果使用 `DreamLM/DreamOn@8a0a54918412eda9402a327646f7f067f7160ec8` 与 released checkpoint `Dream-org/DreamOn-v0-7B@8ccc74750e43177327f29dab9e91882ba759e194`（Apache-2.0）。Single H200 只替代官方 8 进程样本分片；本地固定 per-candidate seed-42 以保证 resume/topology 稳定，因此不称为 exact official 8-GPU random-stream reproduction。

## Protocol / population

- population：project non-frozen SingleLine `927 rows / 148 base-function clusters`；frozen intersection=`0`。
- released-source sweep：initial lengths `4/8/16/32/64`，`max_gen_len=64`，steps=`256`，temperature=`0.2`，top-p=`0.9`，entropy remasking，mask expansion 与 EOS contraction。
- paper 正文的 `Lmax=128` 保持为独立且未执行的协议，未与 released source 静默混合。
- grouped inference：equal-weight base-function task macro，10,000 次 cluster bootstrap；row Pass@1 单列。

## Absolute results

| Arm | Row Pass@1 | Task macro | 95% cluster CI | Forward calls | Token-forwards | Mean wall / row | Peak H200 memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| min4 / max64 | `88.4574%` | `78.1111%` | `[72.4789%,83.4275%]` | `12,524` | `3,562,147` | `1.5287s` | `24,032,471,552 B` |
| min8 / max64 | `90.3991%` | `81.3313%` | `[75.8913%,86.3460%]` | `13,199` | `3,793,167` | `1.7011s` | `24,047,672,320 B` |
| min16 / max64 | `90.7228%` | `83.1788%` | `[78.0777%,87.9909%]` | `14,806` | `4,287,168` | `1.9014s` | `24,139,347,456 B` |
| min32 / max64 | `91.2621%` | `83.0372%` | `[77.7295%,87.8439%]` | `14,318` | `4,041,982` | `1.8537s` | `24,018,886,656 B` |
| min64 / max64 | `91.6936%` | `85.1299%` | `[80.2582%,89.7411%]` | `9,168` | `2,595,211` | `1.3476s` | `24,258,207,744 B` |

这些是五个预注册 initial-length arms 的绝对结果，不从 outcome 中事后选择一个新 primary protocol。Matched controls 已完成 Fixed8 与 Fixed16：DreamOn−Fixed8 macro=`+33.243pp`，CI=`[+28.390,+38.417]pp`；DreamOn−Fixed16=`+13.943pp`，CI=`[+9.618,+18.405]pp`。这些是 training-based released system 相对 base fixed controls 的整体差异，不是纯动态长度或 compute-matched effect。Fixed4/32/64仍未完成。

## Dynamic mechanism / integrity

| Arm | Final length mean / p50 / p90 / range | Expansion-positive rows / total moves | Contraction-positive rows / total moves | Termination |
|---|---:|---:|---:|---|
| min4 | `7.54 / 6 / 13 / 0..39` | `725 / 5,396` | `116 / 2,115` | all middle positions filled `927/927` |
| min8 | `8.08 / 7 / 13 / 0..39` | `360 / 5,052` | `573 / 4,978` | all middle positions filled `927/927` |
| min16 | `8.62 / 7 / 16 / 0..44` | `192 / 5,794` | `866 / 12,636` | all middle positions filled `927/927` |
| min32 | `8.70 / 7 / 16 / 0..44` | `198 / 5,132` | `923 / 26,733` | all middle positions filled `927/927` |
| min64 | `8.87 / 7 / 16 / 0..44` | `2 / 2` | `927 / 51,111` | all middle positions filled `927/927` |

五个 canonical raw 均为 `927/927` exact unique success rows，missing/extra/duplicate/canonical error/forward-accounting error/length-accounting error 均为 `0`。min32/min64 首轮与另外五个模型同时加载时，各产生一条 append-only resource OOM failure journal；随后分别从 `153` 与 `152` 个成功 key 原地 resume，dedup 正常，最终 canonical audit 通过。历史 failure journal 不删除、不覆盖。

Artifacts：

- grouped analysis：`analysis_outputs/dreamon_singleline_official_source_20260731_v1/`
- canonical raw/status：`outputs_clean/dreamon_singleline_20260731_v1/`
- logs：`logs/paper_agent/20260731_dreamon_singleline_min<L>.log`

Frozen controller test 保持 `sealed`，`test_evaluation_count=0`。
