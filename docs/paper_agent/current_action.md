# Current Paper-Agent Action

Timestamp: 2026-06-14 02:40 CST

## Action Name

Close Route2 precision `len32` GPU3 full run and analyze result.

## Current Phase

The user urgently requested that the active experiment be moved off GPU0/GPU1 and run only on GPU3. The earlier GPU1 `len32` partial run was interrupted at about `405/1033` rows and is not used as final evidence.

A clean GPU3-only full run has completed successfully.

## Completed Full Run

- tmux session: `route2_precision_len32_gpu3_20260614`
- log: `logs/paper_agent/20260614_full_route2_precision_len32_gpu3.log`
- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`
- GPU: `CUDA_VISIBLE_DEVICES=3`
- policy: `top1_median <= 0.464844 AND confidence_max <= 0.84375`
- rescue length: `32`
- baseline: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`

Verification:

- log ended with `COMMAND_EXIT_CODE=0`
- `results.jsonl` has `1033` valid rows
- `summary.json` exists
- `step_traces.jsonl` exists and is nonempty
- final pass count: `801/1033 = 77.54%`

## Result Summary

| Run | Pass | Rate | Delta vs `midcons` | Pairwise W/L/TP/TF | Triggers | Trigger true-long precision | Avg sec incl. probe |
|---|---:|---:|---:|---:|---:|---:|---:|
| current `midcons` baseline | `795/1033` | `76.96%` | baseline | baseline | n/a | n/a | `4.311` |
| Route2 broad len24 | `801/1033` | `77.54%` | `+6` | `7/1/794/231` | `73` | `53.42%` | `5.085` |
| Route2 precision len24 | `800/1033` | `77.44%` | `+5` | `5/0/795/233` | `57` | `61.40%` | `5.094` |
| Route2 precision len32 | `801/1033` | `77.54%` | `+6` | `6/0/795/232` | `57` | `61.40%` | `5.462` |

## Interpretation

The GPU3-only `len32` run is a clean low-risk positive result, but not a true-long breakthrough.

Compared with current `midcons`, it gains `+6` tasks with `0` observed losses. Compared with precision `len24`, it gains `+1` task. The longer rescue length does recover `2` oracle `17-24` tasks, but it does not improve the oracle `25+` bucket.

Main diagnostic:

- Risk control remains strong: `0` losses vs `midcons`.
- The precision gate still has good true-long trigger precision: `57` triggers, `61.40%` true-long.
- Rescue generation remains weak on hard long rows: triggered oracle `25+` rows are `0/11` pass.
- Missed failed-long rows remain common, so both gate recall and rescue quality matter.

## Next Required Step

Update experiment docs and dashboards, then do a Route2 error analysis:

1. triggered-but-still-failed true-long rows,
2. missed failed-long rows,
3. whether the next training-free direction should be adaptive rescue length, better rescue decoding, or a stronger trace/probe fusion gate.
