# Current Paper-Agent Action

Timestamp: 2026-06-17 CST

## Action Name

Brainstorm and plan Route2 error analysis as Discovery-layer V3, not abandonment of true-long signal search.

## Current Phase

The latest implemented Superpowers-style spec, `trace_feature_audit_v2`, has completed and produced `diagnostic_only`. The later user-approved Route2 full runs produced a clean but small gain, with precision `len32` at `801/1033 = 77.54%`, pairwise `6/0/795/232` against `midcons`.

The next step is not to give up on finding useful signals. It is to use Route2's wins, missed failed-long rows, and triggered-but-still-failed rows to improve the v2 Discovery layer.

## Superpowers Alignment

- `superpowers:brainstorming`: active in this action. The brainstorming output is the new design spec and experiment brief.
- `superpowers:using-git-worktrees`: checked. The current workspace is already the dedicated `paper-agent-overnight` branch with a clean worktree. This planning-only action does not create a new worktree. If the next implementation modifies analysis scripts or runners, the implementation plan must re-run this gate.
- `superpowers:writing-plans`: active in this action. The executable plan is written to `docs/superpowers/plans/2026-06-17-route2-error-analysis-discovery-v3.md`.

The current tool environment does not expose callable `superpowers:*` skill files, so this action follows the local project protocol as the fallback.

## Reviewer Motivation

A CCF-A reviewer would not accept either of these shortcuts:

1. "v2 did not find a stable policy, therefore trace/probe signals are useless."
2. "Route2 gained `+6`, therefore true-long is solved."

The right reviewer-facing move is to explain why Route2 wins, where it fails, and which missing signal family should be added to Discovery-layer V3.

## Key Evidence To Explain

- Precision `len32` gains `+6` tasks over `midcons` with `0` losses.
- All six wins are triggered rescue cases.
- Triggered-but-still-failed true-long rows: `33`.
- Among those, `31/33` already have rescue length greater than or equal to oracle length.
- Missed baseline failed-long rows: `56`.
- Oracle `25+` remains unchanged at `16.13%`.

This points away from blind length increases and toward two linked questions:

1. Why does rescue generation fail even when length is apparently enough?
2. Why does the gate miss many failed-long rows whose trace confidence looks high?

## Proposed Next Diagnostic

Create a CPU-only `route2_error_analysis` diagnostic that classifies every row into:

- Route2 wins,
- losses,
- triggered true-long still failed,
- missed failed-long,
- short/medium wins,
- no-change failures.

The diagnostic should output reviewer-readable tables and recommend one of:

- improve rescue decoding/selection,
- improve trace/probe fusion gate recall,
- test adaptive rescue length only where length insufficiency is actually supported,
- stop true-long rescue under current signals.

## Expected Documentation Outputs

- `docs/superpowers/specs/2026-06-17-route2-error-analysis-discovery-v3-design.md`
- `docs/superpowers/plans/2026-06-17-route2-error-analysis-discovery-v3.md`
- `docs/paper_agent/experiments/20260617_route2_error_analysis_discovery_v3.md`

No GPU command is launched by this planning action.
