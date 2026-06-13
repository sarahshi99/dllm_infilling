# Route 2 Trace-Gated Long Rescue Full Runs

Timestamp: 2026-06-13 23:14 CST

## Action Name

Completed full Route 2 trace-gated long rescue follow-up on LLaDA-Base `midcons`.

## Current Phase

The CPU-only `trace_feature_audit_v2` found partial trace signal. The user preferred full runs over small-only experiments, so this action tested two Route 2 trace-gated policies on full `1033`-task runs in parallel on GPU `2` and GPU `3`. Both runs are now complete.

## Reviewer Motivation

Offline diagnostics cannot prove pass-rate improvement. A reviewer will care about full-policy behavior: total pass rate, long-bucket gain, short-bucket regression, runtime, and whether a trace-only gate damages already solvable short tasks.

## Method Boundary

- The policy may use only inference-time visible trace/decode features.
- Oracle length, verifier pass/fail, and bucket labels are used only for offline evaluation after generation.
- The policy must not choose between primary and rescue outputs using verifier results.
- Runtime should count both the primary decode used to obtain trace features and the rescue decode when triggered.

## Candidate Policies

| GPU | Name | Gate | Rescue canvas | Offline midcons trace accounting |
|---|---|---|---:|---|
| 2 | `broad_plateau_len24` | `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16` | `24` | `73` triggers, `39` failed-long, `10` short-risk, `1` current-pass risk, precision `0.534` |
| 3 | `precision_top1_conf_len24` | `top1_median <= 0.464844 AND confidence_max <= 0.84375` | `24` | `57` triggers, `35` failed-long, `6` short-risk, `0` current-pass risk, precision `0.614` |

## Success Criteria

- Both 2-sample smokes exit `0`.
- Both full runs produce `1033` valid `results.jsonl` rows and `summary.json`.
- Terminal and docs report total pass rate, pairwise W/L/TP/TF vs current `midcons`, oracle-bucket pass rates, route2 trigger counts, trigger precision, short-bucket losses, and runtime.

## Kill Criteria

- Smoke fails due to import, CUDA, verifier, or malformed output.
- Full run has repeated crashes, OOM, malformed rows, or cannot resume safely.
- Early full-run logs show systematic rescue on short tasks with obvious output corruption.
- Any code path uses oracle/pass/verifier labels to decide whether to keep primary or rescue output.

## Status

Completed at 2026-06-13 23:11 CST. Both full runs exited with `COMMAND_EXIT_CODE=0`.

Verification:

- Broad output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`
- Precision output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`
- Broad rows: `1033` valid `results.jsonl` rows, `summary.json` present, `39740` `step_traces.jsonl` rows.
- Precision rows: `1033` valid `results.jsonl` rows, `summary.json` present, `38872` `step_traces.jsonl` rows.
- Baseline for pairwise accounting: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`, `795/1033 = 76.96%`.

## Final Result

| Run | Pass | Rate | Delta vs current `midcons` | Avg sec incl. probe |
|---|---:|---:|---:|---:|
| current `midcons` baseline | `795/1033` | `76.96%` | baseline | `4.3113` |
| Route2 broad len24 | `801/1033` | `77.54%` | `+6` tasks / `+0.58pp` | `5.0852` |
| Route2 precision len24 | `800/1033` | `77.44%` | `+5` tasks / `+0.48pp` | `5.0945` |

## Trigger And Pairwise Accounting

| Policy | Triggers | Trigger pass | True-long precision | Pairwise W/L/TP/TF vs `midcons` | Short triggers | Primary-pass risk |
|---|---:|---:|---:|---:|---:|---:|
| Route2 broad len24 | `73` (`7.07%`) | `9.59%` | `53.42%` | `7/1/794/231` | `10` | `1` |
| Route2 precision len24 | `57` (`5.52%`) | `8.77%` | `61.40%` | `5/0/795/233` | `6` | `0` |

## Oracle Bucket Pass Rates

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| current `midcons` baseline | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` |
| Route2 broad len24 | `89.97%` | `79.31%` | `61.11%` | `23.17%` | `16.13%` |
| Route2 precision len24 | `90.13%` | `78.88%` | `61.11%` | `21.95%` | `16.13%` |

Bucket pairwise vs current `midcons`:

| Policy | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| Route2 broad len24 | `W1/L1/net+0` | `W2/L0/net+2` | `W2/L0/net+2` | `W2/L0/net+2` | `W0/L0/net+0` |
| Route2 precision len24 | `W1/L0/net+1` | `W1/L0/net+1` | `W2/L0/net+2` | `W1/L0/net+1` | `W0/L0/net+0` |

## Long-Failure Coverage

The important diagnostic is that the gate can find some failed-long rows, but the fixed `len=24` rescue usually does not make them pass.

| Policy | Failed-long total | Failed-long triggered | Rescue wins in failed-long | Triggered but still fail | Failed-long not triggered |
|---|---:|---:|---:|---:|---:|
| Route2 broad len24 | `91` | `39` | `2` | `37` | `52` |
| Route2 precision len24 | `91` | `35` | `1` | `34` | `56` |

## Interpretation

This full run gives a small positive result, not a breakthrough long-length result. Broad gains `+6` tasks over `midcons` but has one primary-pass loss and more short triggers. Precision gains `+5` tasks with zero observed primary-pass loss, so it is safer but lower-coverage.

The core failure mode is now clearer: Route2's trace gate is not completely useless, but fixed-length `24` rescue is weak on the rows it was meant to rescue. In oracle `17-24`, broad gains only `+2` tasks and precision gains only `+1`; in oracle `25+`, both are flat. Among `91` baseline failed-long rows, broad triggers `39` but rescues only `2`, while precision triggers `35` and rescues only `1`.

Research decision: keep this as positive diagnostic evidence and a small pass-rate improvement over `midcons`, but do not frame it as solving true-long infilling. The next useful step is to analyze triggered-but-still-failed and missed failed-long rows, then decide whether a training-free adaptive rescue length or stronger generation-side rescue is justified.

## Follow-Up: Precision Len32

On 2026-06-14, after the user requested GPU3-only execution and excluded the interrupted GPU1 partial run, a clean Route2 precision `len32` full run completed:

- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`
- log: `logs/paper_agent/20260614_full_route2_precision_len32_gpu3.log`
- result: `801/1033 = 77.54%`
- pairwise vs `midcons`: `6/0/795/232`
- trigger count: `57`
- trigger true-long precision: `61.40%`
- avg sec including probe: `5.4622`

The follow-up matches broad len24's total pass count while preserving the precision policy's zero-loss behavior. It improves oracle `17-24` by `+2` tasks, but oracle `25+` remains unchanged. See `docs/paper_agent/experiments/20260614_route2_precision_len32_full.md` for the detailed table.
