# Current Paper-Agent Action

Timestamp: 2026-06-13 21:18 CST

## Action Name

Run full Route 2 trace-gated long rescue on LLaDA-Base `midcons`.

## Current Phase

The CPU-only `trace_feature_audit_v2` found partial trace signal. The user prefers full runs over small-only experiments, so the next action is to implement a trace-gated Route 2 runner, smoke-test the runner, then launch two full `1033`-task runs in parallel on GPU `2` and GPU `3`.

## Reviewer Motivation

Offline diagnostics cannot prove pass-rate improvement. A CCF-A reviewer will care about full-policy behavior: total pass rate, long-bucket gain, short-bucket regression, runtime, and whether a trace-only gate damages already solvable short tasks. Therefore this action runs full policies after a minimal smoke check.

## Method Boundary

- The policy may use only inference-time visible trace/decode features.
- Oracle length, verifier pass/fail, and bucket labels are used only for offline evaluation after generation.
- The policy must not choose between primary and rescue outputs using verifier results.
- Runtime should count both the primary decode used to obtain trace features and the rescue decode when triggered.

## Inputs

- Current `midcons` trace audit: `analysis_outputs/trace_feature_audit_v2_20260613_204721`
- Current same-backbone comparison baseline: `outputs_clean/full_lcal_v3_short_safe_s3_alpha010_compact_sl_gpus23_20260512_135752/results.jsonl`
- Current `midcons` reference result: `outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`

## Candidate Policies

### GPU2: Broad Plateau Gate

- Experiment name: `full_route2_trace_rescue_broad_plateau_len24_gpu2`
- Gate: `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16`
- Offline midcons trace accounting: `73` triggers, `39` failed-long, `10` short-risk, `1` current-pass risk, precision `0.534`
- Rescue canvas: fixed `24`, used as `max(primary_selected_length, 24)`
- Purpose: maximize failed-long coverage while keeping rescue length moderate.

### GPU3: Precision Top1/Confidence Gate

- Experiment name: `full_route2_trace_rescue_precision_top1_conf_len24_gpu3`
- Gate: `top1_median <= 0.464844 AND confidence_max <= 0.84375`
- Offline midcons trace accounting: `57` triggers, `35` failed-long, `6` short-risk, `0` current-pass risk, precision `0.614`
- Rescue canvas: fixed `24`, used as `max(primary_selected_length, 24)`
- Purpose: test a safer lower-risk gate with fewer expected short regressions.

## Success Criteria

- Both 2-sample smokes exit `0`.
- Both full runs produce `1033` valid `results.jsonl` rows and `summary.json`.
- Terminal and docs report total pass rate, pairwise W/L/TP/TF vs current `midcons`, oracle-bucket pass rates, route2 trigger counts, trigger precision, short-bucket losses, and runtime.
- A policy is promising only if it improves overall pass count or improves oracle `>=17` without unacceptable oracle `<=8` loss.

## Kill Criteria

- Smoke fails due to import, CUDA, verifier, or malformed output.
- Full run has repeated crashes, OOM, malformed rows, or cannot resume safely.
- Early full-run logs show systematic rescue on short tasks with obvious output corruption.
- Any code path uses oracle/pass/verifier labels to decide whether to keep primary or rescue output.

## Expected Commands

Use `/home/shx/miniconda3/envs/dllm_env/bin/python`. Full runs should be launched in separate `tmux` sessions with `CUDA_VISIBLE_DEVICES=2` and `CUDA_VISIBLE_DEVICES=3`, respectively.
