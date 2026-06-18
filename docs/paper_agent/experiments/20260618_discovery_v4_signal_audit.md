# Discovery V4 Signal Audit

Date: 2026-06-18 CST

## Purpose

Run the CPU-only Discovery V4 audit proposed in `docs/superpowers/specs/2026-06-17-discovery-v4-signal-model-design.md`.

The goal is to test whether the current full trace/probe/action logs contain a stable, inference-visible signal that can justify another GPU policy run for true-long recovery.

This is not a new pass-rate claim and does not launch GPU work.

## Inputs

- baseline `midcons`: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- baseline traces: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/step_traces.jsonl`
- Route2 precision len32: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl`
- Route2 precision len24: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958/results.jsonl`
- Route2 broad len24: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958/results.jsonl`

## Output

- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/summary.json`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/report.md`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/policy_shortlist.md`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/row_action_table.csv`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/slice_candidates.csv`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/rule_candidates.csv`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/trace_shape_candidates.csv`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/calibration_residuals.csv`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/weak_signal_votes.csv`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/uplift_diagnostics.csv`

## Verification

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_discovery_v4_signal_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/discovery_v4_signal_audit.py
```

Result: focused tests passed, `Ran 5 tests` / `OK`; compile passed.

## Leakage Check

Two real-data dry runs exposed feature leakage and were not accepted as evidence:

1. `true_long` appeared as a candidate feature. This is oracle-derived and was forbidden.
2. `triggered_rescue_failure_broad_len24` appeared as a candidate feature. This is an outcome label and was forbidden.

The final run excludes oracle/pass/pairwise/risk/outcome target fields from candidate feature columns.

## Final Result

Decision: `route2_polish_only`.

Reason: no stable low-risk V4 signal was found beyond Route2 polish.

| Item | Value |
|---|---:|
| joined rows | `1033` |
| true-long rows | `113` |
| baseline failed-long rows | `91` |
| precision len32 W/L/TP/TF | `6/0/795/232` |
| precision len32 triggers | `57` |
| precision len24 W/L/TP/TF | `5/0/795/233` |
| precision len24 triggers | `57` |
| broad len24 W/L/TP/TF | `7/1/794/231` |
| broad len24 triggers | `73` |

Best final non-leaking candidate:

| Candidate | Triggers | Missed failed-long | Triggered rescue-failure | Short risk | Current-pass risk | True-long precision | Stable folds | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `broad_len24_triggered >= 1` | `73` | `4` | `33` | `10` | `1` | `0.534` | `3/5` | `reject` |

## Interpretation

V4 confirms that the broader Route2 trigger has signal, but it is not safe enough as a policy gate because short risk is too high and missed-long recall remains low.

This result does not mean trace/probe information is useless. It means the current V4 slice/rule/trace-shape/calibration/weak-signal implementation did not find a reviewer-safe rule from existing logs. The clean current positive result remains Route2 precision len32 as conservative polish:

- `801/1033 = 77.54%`
- pairwise `6/0/795/232`
- `0` losses

## Next Decision

Do not launch a GPU full policy from this audit alone.

If continuing true-long recovery, the next design should focus on rescue generation/selection quality for triggered rows rather than another blind length increase. A future GPU run needs a fresh action brief, success/kill criteria, and a GPU availability check.
