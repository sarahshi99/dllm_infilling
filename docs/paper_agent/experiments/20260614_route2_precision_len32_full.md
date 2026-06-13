# Route2 Precision Len32 Full Run

Timestamp: 2026-06-14 02:40 CST

## Question

The previous Route2 precision `len24` full run was safe but recovered only one failed-long row. This follow-up asks one narrow question:

> If we keep the safer precision trace gate and increase the rescue canvas from `24` to `32`, can we recover more true-long failures without introducing losses?

The user required this clean full run to use GPU3 only. A previous GPU1 partial run was interrupted and is excluded from the final evidence.

## Method Boundary

- Training-free and inference-time only.
- The trigger uses trace/decode features only.
- Oracle length and pass/fail are used only for post-run accounting.
- The method does not use verifier outcomes to choose between primary and rescue outputs.

## Command Evidence

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

## Main Comparison

| Run | Pass | Rate | Delta vs `midcons` | Pairwise W/L/TP/TF | Triggers | Trigger true-long precision | Avg sec incl. probe |
|---|---:|---:|---:|---:|---:|---:|---:|
| current `midcons` baseline | `795/1033` | `76.96%` | baseline | baseline | n/a | n/a | `4.311` |
| Route2 broad len24 | `801/1033` | `77.54%` | `+6` | `7/1/794/231` | `73` | `53.42%` | `5.085` |
| Route2 precision len24 | `800/1033` | `77.44%` | `+5` | `5/0/795/233` | `57` | `61.40%` | `5.094` |
| Route2 precision len32 | `801/1033` | `77.54%` | `+6` | `6/0/795/232` | `57` | `61.40%` | `5.462` |

## Oracle Bucket Comparison

| Bucket | Win | Loss | Tie pass | Tie fail | Route2 pass rate | Baseline pass rate |
|---|---:|---:|---:|---:|---:|---:|
| `<=8` | `2` | `0` | `538` | `58` | `540/598 = 90.30%` | `538/598 = 89.97%` |
| `9-12` | `2` | `0` | `182` | `48` | `184/232 = 79.31%` | `182/232 = 78.45%` |
| `13-16` | `0` | `0` | `53` | `37` | `53/90 = 58.89%` | `53/90 = 58.89%` |
| `17-24` | `2` | `0` | `17` | `63` | `19/82 = 23.17%` | `17/82 = 20.73%` |
| `25+` | `0` | `0` | `5` | `26` | `5/31 = 16.13%` | `5/31 = 16.13%` |

## Trigger Outcomes

| Bucket | Triggers | Wins | Losses | Triggered pass |
|---|---:|---:|---:|---:|
| `<=8` | `6` | `2` | `0` | `2/6 = 33.33%` |
| `9-12` | `6` | `2` | `0` | `2/6 = 33.33%` |
| `13-16` | `10` | `0` | `0` | `0/10 = 0.00%` |
| `17-24` | `24` | `2` | `0` | `2/24 = 8.33%` |
| `25+` | `11` | `0` | `0` | `0/11 = 0.00%` |

## Interpretation

This is a clean incremental positive run. It matches the broad len24 total pass count while keeping the precision policy's zero-loss behavior:

- `+6` tasks vs current `midcons`.
- `0` losses vs current `midcons`.
- `+1` task vs precision len24.
- `+2` tasks in oracle `17-24`.
- no improvement in oracle `25+`.

The result does not solve true-long infilling. The precision gate finds true-long rows at decent precision, but most triggered long rows still fail after `len32` rescue. In particular, triggered oracle `25+` rows are `0/11` pass.

The next useful step is not another blind length increase. It should be an error analysis separating:

1. triggered-but-still-failed true-long rows, where rescue generation/selection is the bottleneck;
2. missed failed-long rows, where gate recall is the bottleneck;
3. short/medium wins, which explain most of the current total pass-rate gain.
