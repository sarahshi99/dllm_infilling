# Distinct-Candidate Generation Ceiling Action

Date: 2026-07-02 CST

## Decision

Proceed to Phase 1b: `Distinct-Candidate Generation Ceiling`.

Accepted prior verdict: `positive_control_only`.

Stop rules carried forward:

- Do not expand the old A/B/C/D matrix to 9 cases.
- Do not treat `D_oracle_sufficient_steps96` as an independently useful action; action-equivalence audit showed it was output-equivalent to C on all 3 prior cases.
- Do not introduce 96/128/etc. as a new action family by merely increasing nominal step budget.

## Scientific Question

Under oracle-sufficient offline canvas, can a genuinely distinct generation trajectory produce new candidates, and can any such candidate be correct for hard true-long cases?

This pilot cannot answer deployable controller quality, external baseline superiority, remasking novelty, or final backbone incapability.

## Cases

- `SingleLineInfilling/HumanEval/116/L0`: positive-control sanity.
- `SingleLineInfilling/HumanEval/85/L0`: oracle-length-sufficient syntax failure.
- `SingleLineInfilling/HumanEval/113/L3`: missed failed-long semantic/unit-test failure.

## Actions

### A/B Sanity

- `A_primary`: current primary replay, seed 0 only.
- `B_route2_len32`: current Route2 replay when historically triggered; otherwise primary replay, seed 0 only.

### C Baseline

- `C_oracle_sufficient`: oracle-sufficient canvas with current LCAS early commit enabled.
- Seeds: `0,1,2`.

### E No-Early-Commit

- `E_oracle_sufficient_no_early_commit`
- Uses the same oracle-sufficient canvas as C.
- Uses the same pre-registered nominal 64-step schedule as C.
- Disables the `global_gap_early_commit` decision.
- Still records if the decoder reaches `no_remaining_masks`, because once all masks are resolved the current runner may have no further effective update to perform.
- Records per-step remaining-mask counts and token-change counts.
- Records whether E made more effective updates than C.

### F Trace Remask

- `F_oracle_sufficient_trace_remask`
- Stage 1: run C once with oracle-sufficient canvas and early commit enabled.
- Stage 2: select suspicious positions using only internal trace signals from Stage 1.
- Remask rule, fixed before GPU:
  - For each middle position, compute `token_flip_count` from the top-1 prediction sequence across Stage 1 forwards.
  - Compute final top-1 confidence for the position from the final Stage 1 forward.
  - Rank by `token_flip_count` descending, then final confidence ascending, then position index ascending.
  - Remask `max(1, min(4, ceil(0.10 * canvas_len)))` positions.
- Refinement:
  - Run a fixed 16-step refinement schedule on the remasked positions.
  - Disable early commit in the refinement stage.
  - Record remasked indices, scores, count, refinement steps, token-change counts, and final candidate hash.
- The rule must not use unit-test result, oracle code content, pass/fail label, or task-specific manual edits.

F is a diagnostic baseline, not an originality claim about remasking.

## Action-Distinctness Gate

Before the formal pilot, run only `85/L0` with C/E/F seed 0.

Pass the gate if:

- E disables early commit and records whether it adds effective updates versus C.
- F has `remasked_token_count > 0`.
- F executes refinement.
- At least E or F differs from C by candidate hash, or has an auditable distinct trajectory.

If E and F are both hash-equivalent to C and neither produces an effective remask/trajectory change, set verdict `invalid_action_not_distinct` and stop.

## Formal Pilot

If the gate passes, run the 3 listed cases only.

For C/E/F use experimental seeds:

```text
0,1,2
```

Do not add seeds after seeing results.

## Reporting

Report:

- unique candidate hashes;
- compile rate;
- Pass@1 per fixed seed;
- offline candidate-existence Pass@3;
- error-type distribution;
- candidate length;
- cost;
- action-level and case-level tables;
- whether changes are only length changes or actual syntax/semantic differences.

Final verdict must be one of:

- `invalid_action_not_distinct`
- `no_candidate_diversity`
- `candidate_diversity_without_correctness`
- `no_early_commit_signal`
- `trace_remask_signal`
- `multi_seed_candidate_signal`
- `mixed_distinct_candidate_signal`
- `positive_control_only`
