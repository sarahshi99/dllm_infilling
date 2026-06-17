# Route2 Error Analysis Discovery V3

Timestamp: 2026-06-17 CST

## Action Name

CPU-only Route2 error analysis as the next Discovery-layer iteration.

## Superpowers Alignment

- `superpowers:brainstorming`: used as local protocol fallback to compare next research routes.
- `superpowers:using-git-worktrees`: checked. Current branch is `paper-agent-overnight`; no extra worktree was needed because the implementation is narrow and CPU-only.
- `superpowers:writing-plans`: used as local protocol fallback. The executable plan is `docs/superpowers/plans/2026-06-17-route2-error-analysis-discovery-v3.md`.

The current environment does not expose callable `superpowers:*` skill files, so the project protocol is used directly.

## Reviewer Motivation

The project should not stop searching for useful true-long signals merely because `trace_feature_audit_v2` ended as `diagnostic_only`. It should also not overclaim Route2's small pass-rate gain as solving true-long.

A reviewer-facing next step must explain:

- why the six Route2 wins happen;
- why many triggered long rows still fail;
- why many failed-long rows are missed;
- whether new Discovery-layer features can separate those cases.

## Inputs

Baseline:

- current `midcons`: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- pass rate: `795/1033 = 76.96%`

Route2 precision `len32`:

- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl`
- pass rate: `801/1033 = 77.54%`
- pairwise: `6/0/795/232`
- triggers: `57`
- trigger true-long precision: `61.40%`

## Implementation

Implemented:

- `analysis/route2_error_analysis.py`
- `tests/test_route2_error_analysis.py`

Final output:

- `analysis_outputs/route2_error_analysis_20260617_165806/summary.json`
- `analysis_outputs/route2_error_analysis_20260617_165806/error_taxonomy.csv`
- `analysis_outputs/route2_error_analysis_20260617_165806/triggered_failed_long.csv`
- `analysis_outputs/route2_error_analysis_20260617_165806/missed_failed_long.csv`
- `analysis_outputs/route2_error_analysis_20260617_165806/wins.csv`
- `analysis_outputs/route2_error_analysis_20260617_165806/feature_contrast.csv`
- `analysis_outputs/route2_error_analysis_20260617_165806/report.md`

No GPU command was launched.

## Result

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

Bucket pairwise:

| Bucket | Win | Loss | Tie pass | Tie fail | Triggers |
|---|---:|---:|---:|---:|---:|
| `<=8` | `2` | `0` | `538` | `58` | `6` |
| `9-12` | `2` | `0` | `182` | `48` | `6` |
| `13-16` | `0` | `0` | `53` | `37` | `10` |
| `17-24` | `2` | `0` | `17` | `63` | `24` |
| `25+` | `0` | `0` | `5` | `26` | `11` |

## Interpretation

This diagnostic confirms Route2 precision `len32` is a clean incremental result: all `6` wins are triggered rescue cases and there are `0` losses.

It also shows why this is not a true-long solution. Among `33` triggered failed-long rows, `31` already have rescue length at least oracle length, so another blind length increase is unlikely to be the default answer. Meanwhile, `56` baseline failed-long rows are not triggered at all, so gate recall remains a real bottleneck.

The next mechanism should therefore be mixed:

- inspect rescue generation/selection quality for triggered long rows that still fail;
- expand probe-trace fusion for missed failed-long rows;
- only test adaptive rescue length if a new offline diagnostic shows length insufficiency is common.

## Claim Boundary

This is not a new SOTA claim and not a new pass-rate result beyond the already recorded Route2 full run. It is diagnostic evidence for the next Discovery-layer design.

The current conservative claim remains:

> Route2 precision `len32` is low-risk incremental evidence, but true-long infilling remains unsolved.

## Verification

Fresh focused verification to run before final commit:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_error_analysis.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/route2_error_analysis.py
git diff --check
```
