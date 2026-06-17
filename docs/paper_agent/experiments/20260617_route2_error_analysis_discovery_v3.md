# Route2 Error Analysis Discovery V3

Timestamp: 2026-06-17 CST

## Action Name

Plan CPU-only Route2 error analysis as the next Discovery-layer iteration.

## Superpowers Alignment

- `superpowers:brainstorming`: used as local protocol fallback to compare next research routes.
- `superpowers:using-git-worktrees`: checked. Current branch is `paper-agent-overnight` and the worktree was clean before this planning action. No extra worktree is created for planning docs.
- `superpowers:writing-plans`: used as local protocol fallback. The executable plan is `docs/superpowers/plans/2026-06-17-route2-error-analysis-discovery-v3.md`.

The current environment does not expose callable `superpowers:*` skill files, so the project protocol is used directly.

## Reviewer Motivation

The project should not stop searching for useful true-long signals merely because `trace_feature_audit_v2` ended as `diagnostic_only`. It should also not overclaim Route2's small pass-rate gain as solving true-long.

A reviewer-facing next step must explain:

- why the six Route2 wins happen;
- why many triggered long rows still fail;
- why many failed-long rows are missed;
- whether new Discovery-layer features can separate those cases.

## Current Evidence

Baseline:

- current `midcons`: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- pass rate: `795/1033 = 76.96%`

Route2 precision `len32`:

- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl`
- pass rate: `801/1033 = 77.54%`
- pairwise: `6/0/795/232`
- triggers: `57`
- trigger true-long precision: `61.40%`

Quick CPU accounting:

- all `6` wins are triggered rescue cases;
- triggered-but-still-failed true-long rows: `33`;
- `31/33` have rescue length greater than or equal to oracle length;
- missed baseline failed-long rows: `56`;
- oracle `25+` remains unchanged.

## Brainstormed Paths

### Path A: Rescue Generation Quality Audit

Ask whether triggered long rows fail because the generated code is wrong despite sufficient length.

Why this is promising: most triggered true-long failures already have enough rescue length by oracle accounting.

Risk: verifier failure categories may be too coarse to explain generation failures.

### Path B: Probe-Trace Fusion For Missed Failed-Long Rows

Ask whether missed failed-long rows have probe curve, selected length, or trace/probe disagreement signals not captured by the current precision gate.

Why this is promising: missed failed-long count remains high at `56`.

Risk: high-confidence missed rows may not expose an inference-visible signal.

### Path C: Adaptive Rescue Length

Ask whether rescue length should vary by trace/probe pattern.

Why this is lower priority now: `31/33` triggered failed-long rows already had rescue length at least oracle length.

Risk: blind length increases add runtime and may not improve pass rate.

### Path D: Stop Or Narrow The Claim

If no signal separates the error classes, keep Route2 as incremental evidence and stop claiming true-long rescue progress under current data.

Risk: narrower paper claim may need stronger backbone/protocol evidence.

## Recommended Route

Run Path A and Path B as one CPU-only diagnostic:

1. classify Route2 outcomes;
2. compare triggered failed-long, missed failed-long, and wins;
3. decide whether Discovery V3 should search rescue-quality features, fusion-gate features, or both.

Do not launch another GPU full run until this diagnostic explains which mechanism is likely to help.

## Expected Implementation

Planned script:

- `analysis/route2_error_analysis.py`

Planned tests:

- `tests/test_route2_error_analysis.py`

Planned output:

- `analysis_outputs/route2_error_analysis_YYYYMMDD_HHMMSS/summary.json`
- `analysis_outputs/route2_error_analysis_YYYYMMDD_HHMMSS/error_taxonomy.csv`
- `analysis_outputs/route2_error_analysis_YYYYMMDD_HHMMSS/triggered_failed_long.csv`
- `analysis_outputs/route2_error_analysis_YYYYMMDD_HHMMSS/missed_failed_long.csv`
- `analysis_outputs/route2_error_analysis_YYYYMMDD_HHMMSS/wins.csv`
- `analysis_outputs/route2_error_analysis_YYYYMMDD_HHMMSS/feature_contrast.csv`
- `analysis_outputs/route2_error_analysis_YYYYMMDD_HHMMSS/report.md`

## Success Criteria

- CPU-only command runs successfully.
- Joins all `1033` baseline and Route2 rows.
- Reproduces pairwise `6/0/795/232`.
- Reproduces triggered failed-long `33` and missed failed-long `56`.
- Reports whether the bottleneck is rescue quality, gate recall, length insufficiency, or mixed.
- Produces a concrete next-path recommendation.

## Kill Criteria

- Known counts cannot be reproduced.
- Required fields are missing from Route2 outputs.
- All useful separators depend on oracle/pass/verifier labels as inference-time inputs.
- The only plausible next action is another blind full GPU run.

## Documentation Outputs

- Design spec: `docs/superpowers/specs/2026-06-17-route2-error-analysis-discovery-v3-design.md`
- Executable plan: `docs/superpowers/plans/2026-06-17-route2-error-analysis-discovery-v3.md`
- This experiment brief.

No GPU command is launched by this planning action.
