# Current Paper-Agent Action

Timestamp: 2026-06-17 17:10 CST

## Action Name

Complete CPU-only Route2 error analysis as Discovery-layer V3 evidence.

## Current Phase

The Route2 precision `len32` full run remains the cleanest follow-up result: `801/1033 = 77.54%`, pairwise `6/0/795/232` against current `midcons`, with `0` losses. This is a small positive result, not a true-long breakthrough.

The new CPU-only diagnostic has now completed:

- script: `analysis/route2_error_analysis.py`
- tests: `tests/test_route2_error_analysis.py`
- output: `analysis_outputs/route2_error_analysis_20260617_165806`
- report: `analysis_outputs/route2_error_analysis_20260617_165806/report.md`

No GPU experiment was launched by this action.

Verification-time GPU note: `nvidia-smi` shows GPU2 and GPU3 are already occupied at roughly `40GB` used each, so no new experiment should be launched there without a later fresh check and explicit action brief.

## Superpowers Alignment

- `superpowers:brainstorming`: used as local protocol fallback to decide whether to continue signal search after `trace_feature_audit_v2`.
- `superpowers:using-git-worktrees`: checked. Work stayed on the dedicated `paper-agent-overnight` branch because this action only adds one CPU analysis script, one focused test file, output artifacts, and paper-agent docs.
- `superpowers:writing-plans`: used as local protocol fallback. The executable plan is `docs/superpowers/plans/2026-06-17-route2-error-analysis-discovery-v3.md`.

The current tool environment does not expose callable `superpowers:*` skill files, so this action follows the local project protocol as fallback.

## Completed Diagnostic Result

Input baseline:

- `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- baseline result: `795/1033 = 76.96%`

Input Route2 result:

- `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl`
- Route2 result: `801/1033 = 77.54%`

Reproduced CPU accounting:

| Metric | Value |
|---|---:|
| Joined rows | `1033` |
| Pairwise W/L/TP/TF | `6/0/795/232` |
| Route2 triggers | `57` |
| Triggered failed-long rows | `33` |
| Missed failed-long rows | `56` |
| Triggered failed-long with rescue length >= oracle | `31/33` |
| Dominant bottleneck | `mixed_rescue_quality_and_gate_recall` |
| Recommended next path | `rescue_generation_quality+gate_recall` |

## Interpretation

This supports the user's concern that signal search should not be abandoned. Route2 did find real signal: all six wins are triggered rescue cases and the policy has no losses in the clean `len32` run.

But the diagnostic also shows that blind canvas-length increases are not the default next mechanism. Among `33` triggered failed-long rows, `31` already have rescue length at least oracle length. That points to rescue generation/selection quality. At the same time, `56` failed-long rows are missed entirely, so gate recall and probe-trace fusion still matter.

## Next Research Move

Do not launch another full GPU run from the current fixed trace gate. The next design should be one of:

1. rescue-generation-quality audit: inspect why triggered long rows fail despite enough length;
2. probe-trace fusion gate: search inference-visible signals for the `56` missed failed-long rows;
3. a combined Discovery V3 model that ranks both rescue-success predictors and missed-long predictors, then distills readable training-free rules.

GPU `2/3` must not be used while other users' tasks are present. Before any future GPU action, check `nvidia-smi`, write a new action brief with success/kill criteria, and only launch if the allocation is clear or explicitly approved.
