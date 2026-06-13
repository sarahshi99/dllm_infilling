# Current Paper-Agent Action

Timestamp: 2026-06-13 CST

## Action Name

Implement and run CPU-only `trace_feature_audit_v2`.

## Current Phase

Post trace-long-rescue negative result. The approved next step is model-assisted offline feature discovery, not GPU execution.

## Reviewer Motivation

The first Route 1/2 formulas triggered zero rows. A reviewer would not accept that as proof trace features are useless. This audit tests richer trace-shape, stop-reason-conditioned, probe-trace fusion, and distilled-rule discovery families under held-out risk constraints.

## Hypothesis

The existing full trace data may contain useful true-long under-selection signals, but v1 missed them because its Boolean formula shape was too narrow.

## Inputs

- Previous local method results: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/results.jsonl`
- Previous local method traces: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/step_traces.jsonl`
- Current `midcons` results: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- Current `midcons` traces: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/step_traces.jsonl`

## Method Boundary

This is CPU-only offline analysis. Labels derived from oracle/pass/fail are allowed only for offline discovery and evaluation. Any learned/fitted model is diagnostic only unless a later plan explicitly changes the paper claim.

## Expected Command

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/trace_feature_audit_v2.py \
  --prev-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/results.jsonl \
  --prev-traces /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/step_traces.jsonl \
  --midcons-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl \
  --midcons-traces /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/step_traces.jsonl \
  --output-dir analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS \
  --folds 5
```

## Success Criteria

- The script runs without GPU.
- Both trace sources report `1033` joined rows.
- Reports include feature coverage, top single rules, top pairwise rules, stop-reason rules, motif rules, shallow-tree rules, model diagnostics, Pareto frontier, and final decision.
- The final decision is one of `policy_candidate`, `diagnostic_only`, `reject`, or `needs_new_data`.

## Kill Criteria

- Missing or malformed trace inputs.
- Joined rows differ from `1033` for either source.
- Candidate search uses oracle/pass labels as policy inputs.
- The only positive signal is opaque and cannot be distilled.
- Any command attempts to launch GPU work.

## Expected Documentation Outputs

- `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/summary.json`
- `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/candidates.csv`
- `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/pareto.csv`
- `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/report.md`
- Updated paper-agent results/dashboard/checkpoint only after the audit command has run.

## Observed Result

Final valid output directory: `analysis_outputs/trace_feature_audit_v2_20260613_204721`.

Decision: `diagnostic_only`.

| Source | Rows | True-long | Failed-long | Short | Decision |
|---|---:|---:|---:|---:|---|
| previous | 1033 | 113 | 96 | 598 | policy_candidate |
| midcons | 1033 | 113 | 91 | 598 | diagnostic_only |

The audit found partial trace signal. The strongest midcons candidate was `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16`, with `18` held-out triggers, `9` failed-long, `2` short-risk, `0` current-pass risk, and `0.500` precision. This is useful diagnostic evidence, but the signal is not stable enough across trace sources to justify a direct full GPU policy run.

No GPU command was launched by this action.
