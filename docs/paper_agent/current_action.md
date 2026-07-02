# Current Paper-Agent Action

Timestamp: 2026-07-02 00:00 CST

## Action Name

Post-V8 rescue-quality and local-guard writing plan.

## Current Phase

`superpowers:writing-plans` local fallback completed. No GPU experiment is running or launched by this action.

Plan output:

- `docs/superpowers/plans/2026-07-02-post-v8-rescue-quality-and-local-guard-plan.md`

Preceding brainstorm:

- `docs/paper_agent/experiments/20260701_next_step_brainstorm_after_v8.md`

## Plan Summary

The next work should be CPU-only and answer two questions before any new GPU command:

1. **Rescue quality:** why do Route2-triggered rows still fail even when rescue length is already enough?
2. **Local proportional guard:** can V8's few long wins be isolated from its many short/medium losses by an inference-visible guard?

## Evidence Behind The Plan

- Current best LLaDA-Base follow-up remains V6 short override: `802/1033 = 77.64%`.
- V8 global proportional reward is negative:
  - V8a: `786/1033 = 76.09%`;
  - V8b: `781/1033 = 75.61%`;
  - V8c: `782/1033 = 75.70%`.
- Route2/V3 diagnosis:
  - `56` failed-long rows are missed by Route2;
  - `33` triggered failed-long rows remain failures;
  - `31/33` triggered failed-long rows already have rescue length at least oracle.

## Planned Implementation Files

- `analysis/post_v8_rescue_quality_audit.py`
- `tests/test_post_v8_rescue_quality_audit.py`

Planned outputs:

- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/summary.json`
- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/report.md`
- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/row_action_table.csv`
- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/triggered_failure_taxonomy.csv`
- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/v8_changed_row_taxonomy.csv`

## Next Stage

Use `superpowers:executing-plans` local fallback next:

1. Write a short action brief for the CPU audit.
2. Create tests first.
3. Implement the CPU audit.
4. Run verification.
5. Run the audit and report a decision.

Do not launch GPU until CPU evidence produces a credible new action brief with success/kill criteria.
