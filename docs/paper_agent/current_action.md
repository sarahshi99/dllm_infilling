# Current Paper-Agent Action

Timestamp: 2026-06-13 23:14 CST

## Action Name

Close out Route2 trace-gated long-rescue full runs and prepare next error analysis.

## Superpowers Mode

Use local serial workflow only. Do not use `superpowers:subagent-driven-development`, Task/Spawn, reviewer subagents, parallel-agent dispatch, `tool_search`, or Goal.

## Current Phase

The two Route2 full runs requested by the user are complete. No Route2 GPU job is currently running.

## Completed Full Runs

### GPU2: Broad Plateau

- log: `logs/paper_agent/20260613_full_route2_broad_gpu2.log`
- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`
- policy: `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16`
- status: log ended with `COMMAND_EXIT_CODE=0`
- rows: `1033` valid `results.jsonl` rows, `summary.json` present, `39740` step-trace rows
- result: `801/1033 = 77.54%`
- pairwise vs current `midcons`: `7/1/794/231`

### GPU3: Precision Top1/Confidence

- log: `logs/paper_agent/20260613_full_route2_precision_gpu3.log`
- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`
- policy: `top1_median <= 0.464844 AND confidence_max <= 0.84375`
- status: log ended with `COMMAND_EXIT_CODE=0`
- rows: `1033` valid `results.jsonl` rows, `summary.json` present, `38872` step-trace rows
- result: `800/1033 = 77.44%`
- pairwise vs current `midcons`: `5/0/795/233`

## Summary

The comparison baseline is current LLaDA-Base `midcons`: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`, `795/1033 = 76.96%`.

| Run | Pass | Delta vs `midcons` | Trigger count | Trigger true-long precision | Avg sec incl. probe |
|---|---:|---:|---:|---:|---:|
| Route2 broad len24 | `801/1033 = 77.54%` | `+6` tasks / `+0.58pp` | `73` | `53.42%` | `5.0852` |
| Route2 precision len24 | `800/1033 = 77.44%` | `+5` tasks / `+0.48pp` | `57` | `61.40%` | `5.0945` |

Long-failure diagnostic:

- There are `91` baseline failed-long rows.
- Broad triggers `39` of them but rescues only `2`.
- Precision triggers `35` of them but rescues only `1`.
- Therefore the current bottleneck is not just detecting failed-long rows; fixed `len=24` rescue usually does not make them pass.

## Next Required Step

Do a CPU-only Route2 error analysis before launching more GPU work:

1. List triggered-but-still-failed rows and their oracle lengths, selected lengths, trace features, and rescue lengths.
2. List missed failed-long rows and compare their trace feature ranges against triggered rows.
3. Decide whether the next training-free direction should be adaptive rescue length, stronger rescue generation, a different Route1 detector, or a Route3-style multi-canvas design.
