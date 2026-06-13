# Route 2 Trace-Gated Long Rescue Full Runs

Timestamp: 2026-06-13 21:18 CST

## Action Name

Run full Route 2 trace-gated long rescue on LLaDA-Base `midcons`.

## Current Phase

The CPU-only `trace_feature_audit_v2` found partial trace signal. The user prefers full runs over small-only experiments, so this action implements a trace-gated Route 2 runner, smoke-tests the runner, then launches two full `1033`-task runs in parallel on GPU `2` and GPU `3`.

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

Not started at creation time.
