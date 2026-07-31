# CAL Phase 1A：MultiLine 4,990 解盲 Action Brief

日期：2026-07-31 UTC

状态：`completed`

## Label and comparison boundary

准确标签固定为：**official-source CAL, initial length 32, on the 4,990-row / 143-cluster project-non-frozen CAL-Rest common subset**。

它不是完整 5,715-case reproduction，也不代表论文四种初始长度的整张表。同一 4,990 normalized keys 上没有 completed `official_fixed32` 时，只报告绝对表现与成本；不与旧 Fixed64 虚构 paired delta，Fixed64 也不称为 equal-compute。

## Inputs

- CAL：`NiuHechang/Calibrated_Adaptive_Length@741e8418a88a732b4c92812424d4f03cab1f7b1f`
- evaluator：`openai/human-eval-infilling@88062ff9859c875d04db115b698ed4b0f0395170`
- model：`GSAI-ML/LLaDA-8B-Base`，cache revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- manifest：`analysis_outputs/official_cal_corrected_protocol_20260715_v1/cal_rest_common_manifest.jsonl`
- raw：`outputs_clean/official_cal_primary_20260715_sprint_v1/official_cal_primary_raw.jsonl`
- analyzer freeze：`e7a790beb2c7e17bfaa5cdc764c5b82dd05ebd83`

## Exact commands

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python - <<'PY'
# Audit exact candidate keys, arm/config hashes, source/evaluator revisions,
# failure journal, forward partition, token budget, wall time and memory fields.
PY
/home/shx/miniconda3/envs/dllm_env/bin/python -m analysis.baseline_grouped_analyzer --manifest analysis_outputs/official_cal_corrected_protocol_20260715_v1/cal_rest_common_manifest.jsonl --method official_cal_primary=outputs_clean/official_cal_primary_20260715_sprint_v1/official_cal_primary_raw.jsonl --expected-rows 4990 --expected-clusters 143 --bootstrap-replicates 10000 --seed 20260731 --output-dir analysis_outputs/official_cal_multiline_4990_grouped_20260731_v1
```

GPU=`none`；log=`none`；output=`analysis_outputs/official_cal_multiline_4990_grouped_20260731_v1/`。

## Gates

- 4990 exact unique rows、143 clusters；
- missing/duplicate/error/failure journal=`0/0/0/0`；
- arm、primary config、CAL/evaluator revisions exact；
- `total_forwards=search_forwards+formal_decode_forwards`；token/wall/memory fields complete；
- 10,000 base-function cluster bootstrap完成；
- frozen=`sealed/0`。

任一 gate 失败即停止 official-primary 解释并记录精确 blocker。

## Outcome

全部 gate 通过。row Pass@1=`1643/4990=32.9259%`；task-macro=`27.8471%`，95% CI=`[24.3313%,31.2715%]`。首次 direct-script invocation 因 repo module path 未加入而失败，未执行统计；改用等价 module entry 后完成，无统计定义变更。
