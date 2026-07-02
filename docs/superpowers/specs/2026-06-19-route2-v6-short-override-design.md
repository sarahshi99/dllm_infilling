# Route2 V6 Short-Override Design

Date: 2026-06-19 CST
Branch: `paper-agent-overnight`
Scope: CPU-only design after V5.1 anchor full runs.

## Objective

Design the next CPU-first step after V5.1 exactly matched Route2 precision `len32`.

The immediate research question is:

> Can a tiny, reviewer-readable, inference-visible selector safely choose `len24_s64` on the few rows where it beats the `len32_s64` anchor, without losing known `len32_s64` wins?

This is a selector audit, not a GPU policy run and not a new pass-rate claim.

## Superpowers Alignment

- `superpowers:brainstorming`: compare three post-V5.1 paths.
- `superpowers:using-git-worktrees`: current branch is already a dedicated paper-agent branch; no extra worktree is needed.
- `superpowers:writing-plans`: this spec plus `docs/superpowers/plans/2026-06-19-route2-v6-short-override-plan.md`.
- `superpowers:executing-plans`: serial implementation, no subagents.
- `superpowers:test-driven-development`: focused tests for row joining, feature extraction, rule scoring, and leakage rejection.
- `superpowers:verification-before-completion`: compile, tests, `git diff --check`, and report review.

The runtime environment does not expose callable `superpowers:*` skill files. This document follows the project protocol's Superpowers workflow as local fallback.

## Current Evidence

V5.1 full results:

- margin `0.02`: `801/1033 = 77.54%`, pairwise vs Route2 precision `len32` is `0/0/801/232`;
- margin `0.10`: `801/1033 = 77.54%`, pairwise vs Route2 precision `len32` is `0/0/801/232`;
- triggered rows: `57`;
- selected triggered pass: `6/57`;
- candidate oracle upper bound: `9/57`;
- triggered oracle `25+`: `0/11`.

Residual selector opportunity:

- `SingleLineInfilling/HumanEval/7/L0`, oracle `14`, only `len24_s64` passed;
- `SingleLineInfilling/HumanEval/11/L6`, oracle `22`, only `len24_s64` passed;
- `SingleLineInfilling/HumanEval/128/L2`, oracle `15`, only `len24_s64` passed.

Known risk:

- Previous targeted smoke showed `len24_s64` can lose a known Route2 precision `len32` win on `SingleLineInfilling/HumanEval/60/L0`.

Therefore, the next step should not be another blind full run. It should be an offline selector audit that asks whether the three residual opportunities are distinguishable from the dangerous shorter-candidate cases.

## Brainstorming Result

### Path A: Conservative `len24_s64` Override

Core idea:

- keep `len32_s64` as the default anchor;
- allow `len24_s64` only under a very restrictive inference-visible rule;
- reject if even one known anchor win would be lost.

Why this is attractive:

- directly targets the three V5.1 upper-bound-only rows;
- keeps paper claim training-free and verifier-free;
- cheap to audit using existing logs.

Risk:

- only three positive opportunities, so rules can overfit;
- if the rule depends on task-id-like artifacts or oracle labels, it is invalid.

### Path B: Better Rescue Candidate Family

Core idea:

- V5.1 candidate upper bound is only `9/57`, so selector quality is not the dominant ceiling;
- design new candidates such as different denoising schedules, confidence stop thresholds, or multi-sample same-length candidates.

Why this is attractive:

- addresses the main bottleneck, especially `25+` rows where all current candidates fail.

Risk:

- requires GPU to validate once designed;
- more expensive and needs a fresh action brief.

### Path C: Learned / Compiler-Assisted Selector

Core idea:

- use a learned or syntax-aware selector to exploit subtle differences between candidates.

Why not now:

- changes the claim boundary;
- current V5.1 upper bound is too low for selector sophistication alone to be the main story.

Decision:

Start with Path A as a CPU-only audit. If Path A fails, use its report to motivate Path B rather than spending GPU on selector-only runs.

## Policy Boundary

Allowed policy inputs:

- candidate id;
- requested candidate length and steps;
- decoded middle text;
- candidate text agreement/consensus proxies;
- trace features: confidence, top1, gap, remaining-mask summaries, stop/plateau features;
- parse/compile diagnostics only if explicitly labeled as compiler-assisted, not pure verifier-free;
- primary/anchor candidate trace fields.

Forbidden policy inputs:

- oracle mask length;
- unit-test pass/fail;
- pairwise outcome against baseline or Route2;
- verifier failure class;
- task id, dataset index, human-eval problem number, or any proxy for memorizing rows;
- candidate `offline_passed` fields.

Offline labels may be used only for evaluation tables.

## Audit Design

### Layer 1: Triggered Candidate Table

Build one row per Route2-triggered task with:

- task id for reporting only;
- anchor candidate `len32_s64`;
- shorter candidate `len24_s64`;
- slower candidate `len32_s96`;
- selected policy output;
- offline pass/fail per candidate;
- inference-visible candidate features;
- pairwise labels for offline accounting.

### Layer 2: Candidate Contrast Features

For each candidate pair, compute inference-visible deltas:

- `len24_confidence_last - len32_confidence_last`;
- `len24_top1_median - len32_top1_median`;
- `len24_gap_median - len32_gap_median`;
- remaining-mask ratios for both candidates;
- text length and normalized text length;
- exact text equality between candidates;
- candidate consensus counts across the three outputs;
- parse/compile flags, optionally kept separate.

Avoid task-specific lexical features except broad, non-identifying structural summaries such as line count or indentation consistency.

### Layer 3: Rule Search

Search reviewer-readable rules only:

1. single-feature thresholds;
2. pairwise thresholds;
3. dominance rules such as `len24_confidence >= len32_confidence + margin`;
4. safety vetoes such as `never override when len32 syntax passes and len24 syntax fails`;
5. optional compiler-assisted variants in a separate table.

Primary utility:

```text
 captured_len24_only_wins
- anchor_loss_penalty
- current_pass_loss_penalty
- short_risk_penalty
- complexity_penalty
```

But final decisions are threshold-gated, not score-only.

## Candidate Gate

A `len24_s64` override is worth implementing only if it satisfies all of:

- captures at least `1` of the `3` upper-bound-only rows;
- loses `0` rows where `len32_s64` passed and `len24_s64` failed;
- triggers at most `5` rows total, preferred at most `3`;
- short-bucket risk is `0`;
- rule is at most two clauses plus one safety veto;
- rule uses only allowed inference-visible inputs;
- the same qualitative rule holds for both V5.1 full outputs.

If a rule captures all `3` upper-bound-only rows but has `1` anchor loss, it remains diagnostic-only unless a clear inference-visible veto removes the loss.

## Expected Decisions

The audit should end with exactly one decision:

- `policy_candidate`: implement the override in the V5 runner and consider a smoke/full only after GPU availability check.
- `diagnostic_only`: record signal but do not run GPU.
- `reject_selector_only`: do not continue selector-only work; design better rescue generation candidates.

## Success Criteria

- Reproduce V5.1 summary:
  - `1033` rows;
  - pass `801`;
  - trigger count `57`;
  - candidate upper bound `9`;
  - pairwise vs Route2 precision `len32`: `0/0/801/232`.
- Identify the three upper-bound-only `len24_s64` rows.
- Identify all anchor-risk rows where `len32_s64` passes but `len24_s64` fails.
- Produce candidate rules and a final decision.
- Write a compact report with enough detail for CCF-A reviewer audit.

## Kill Criteria

Stop with `reject_selector_only` if:

- no rule captures at least one `len24_s64`-only win with zero anchor losses;
- all successful rules rely on forbidden labels;
- the only rules are effectively task memorization;
- candidate fields are too sparse for inference-visible rule search;
- the best rule is too complex to be defensible.

## Implementation Shape

Planned files:

- `analysis/route2_v6_short_override_audit.py`
- `tests/test_route2_v6_short_override_audit.py`
- `analysis_outputs/route2_v6_short_override_audit_20260619/summary.json`
- `analysis_outputs/route2_v6_short_override_audit_20260619/report.md`
- `analysis_outputs/route2_v6_short_override_audit_20260619/rule_candidates.csv`
- `analysis_outputs/route2_v6_short_override_audit_20260619/triggered_candidate_table.csv`

The script is CPU-only and must not launch model inference.

## Self-Review

- The design does not claim a new pass rate.
- The design directly uses V5.1 evidence rather than restarting a broad signal search.
- The selector boundary is explicit.
- The gate is intentionally strict because the positive set is tiny.
- The fallback path is clear: if selector-only fails, move to rescue candidate generation.
