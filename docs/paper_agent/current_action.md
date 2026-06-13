# Current Paper-Agent Action

Timestamp: 2026-06-13 CST

## Action Name

Design `trace_feature_audit_v2`, a smarter CPU-only trace feature discovery audit.

## Current Phase

Post trace-long-rescue negative result. The two full trace runs are complete, but Route 1/2 triggered `0` rows and Route 3 is not a default fallback. The next safe step is design-only: improve the offline discovery method before any GPU policy run.

## Reviewer Motivation

A reviewer will not accept "trace features failed" unless we show that the failure is not merely a bad hand-written formula. The v2 design explicitly tests nonlinear combinations, time-series shape features, stop-reason-conditioned rules, and risk-controlled selection.

## Hypothesis

Existing trace features may contain useful true-long under-selection signal, but v1 missed it because the formula family was too narrow. Model-assisted discovery can reveal interactions, while the final candidate policy must remain explainable and training-free unless the paper scope changes.

## Baseline And Data

- Previous trace source: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`
- Current `midcons` trace source: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`
- Dataset: HumanEval-SingleLineInfilling, `1033` tasks.
- Comparison type: CPU-only offline diagnostic, not a new pass-rate claim.

## Design Output

Design spec: `docs/superpowers/specs/2026-06-13-trace-feature-audit-v2-design.md`

## Success Criteria

- Explain why learned discovery models are not automatically the main training-free method.
- Define richer trace/probe feature families.
- Define a model-assisted discovery layer.
- Define risk-controlled held-out gates.
- Keep Route 3 non-default unless a stable trace quality score is first discovered.

## Kill Criteria

Do not launch GPU experiments from this action. Do not create `trace_feature_audit_v2.py` until the spec is reviewed and an implementation plan is approved.
