# Route2 Rescue Quality V5 Design

Date: 2026-06-18
Branch: `paper-agent-overnight`
Scope: New rescue generation/selection mechanism after Discovery V4 ended as `route2_polish_only`.

## Objective

Design a genuinely new Route2 follow-up that targets rescue generation/selection quality, not another blind canvas-length increase.

The immediate research question is:

> For rows where Route2 fires, can multiple rescue candidates plus an inference-visible selector recover more failures than fixed single-rescue Route2 precision `len32` without increasing short/current-pass risk?

The design remains training-free and inference-time. Hidden unit-test results may be used only for offline reporting and candidate-set upper-bound diagnostics, never for selecting the deployed candidate.

## Current Evidence

Baseline and Route2 evidence:

- current `midcons`: `795/1033 = 76.96%`.
- Route2 precision `len32`: `801/1033 = 77.54%`, pairwise `6/0/795/232`, triggers `57`.
- Route2 precision `len24`: `800/1033 = 77.44%`, pairwise `5/0/795/233`, triggers `57`.
- Route2 broad `len24`: `801/1033 = 77.54%`, pairwise `7/1/794/231`, triggers `73`.

Error analysis:

- precision `len32` has `33` triggered failed-long rows.
- `31/33` triggered failed-long rows already have rescue length at least oracle length.
- precision `len32` misses `56` baseline failed-long rows.
- Discovery V4 found no non-leaking low-risk gate better than Route2 polish.

Implication:

> The next experiment should not simply make rescue length larger. It should ask whether the triggered rows need better candidate generation and candidate selection.

## Superpowers Alignment

- `superpowers:brainstorming`: used as local protocol fallback to compare possible new mechanisms.
- `superpowers:using-git-worktrees`: checked. Current branch is already dedicated to paper-agent work. No new worktree is needed for this design-only step.
- `superpowers:writing-plans`: used as local protocol fallback. The executable plan is `docs/superpowers/plans/2026-06-18-route2-rescue-quality-v5-plan.md`.

The current environment does not expose callable `superpowers:*` skill files, so the project protocol is used directly.

## Design Principle

V5 should make a new action available. It should not be another threshold sweep over the same Route2 action.

The existing runner has one rescue candidate:

```text
primary midcons output
if Route2 trigger:
  run one fixed-length rescue
  always choose rescue
else:
  choose primary
```

V5 changes the triggered branch:

```text
primary midcons output
if precision Route2 trigger:
  generate a small set of rescue candidates
  select one candidate using inference-visible quality signals
else:
  choose primary
```

## Candidate Generation

Use precision Route2 trigger first, not broad trigger, because precision `len32` has `0` losses and higher trigger true-long precision.

For each triggered row, generate candidate rescues with deterministic decode variants:

| Candidate family | Length | Steps | Purpose |
|---|---:|---:|---|
| `rescue_len24_s64` | `max(primary_len, 24)` | `64` | retain existing shorter rescue behavior |
| `rescue_len32_s64` | `max(primary_len, 32)` | `64` | reproduce clean Route2 precision len32 action |
| `rescue_len40_s64` | `max(primary_len, 40)` | `64` | test larger canvas as diversity, not default length claim |
| `rescue_len32_s96` | `max(primary_len, 32)` | `96` | test slower denoising schedule for length-sufficient failures |
| `rescue_len40_s96` | `max(primary_len, 40)` | `96` | test slow larger-canvas fallback for hard triggered rows |

The first implementation may start with a cheaper set:

```text
rescue_len24_s64, rescue_len32_s64, rescue_len32_s96
```

Then add `len40` variants only if the smoke shows stable runtime and no implementation issues.

Rationale:

- Different lengths change the infilling geometry and can change completions even when length is already sufficient.
- Different total steps change the mask reveal schedule and directly target generation quality.
- This is still training-free and uses no verifier feedback during generation.

Name convention:

- `len32_s64` means `max(primary_len, 32)` rescue length and `64` decode steps.
- `len32_s96` means `max(primary_len, 32)` rescue length and `96` decode steps.
- `len24_s64` means `max(primary_len, 24)` rescue length and `64` decode steps.

## Candidate Selection

V5 should report three selectors separately.

### Selector A: Verifier-Free Consensus/Confidence

This is the main training-free policy candidate.

Candidate score:

```text
score =
  consensus_weight * consensus_similarity
+ confidence_weight * final_trace_confidence
+ gap_weight * final_trace_gap
- unresolved_weight * final_remaining_mask_ratio
- length_penalty_weight * normalized_length
```

Inputs are inference-visible:

- candidate middle text;
- candidate trace confidence / top1 / gap;
- remaining mask ratio;
- selected length;
- pairwise text similarity among generated candidates.

No parse/compile/unit-test result is used.

### Selector B: Syntax-Aware, No Hidden Tests

This is a secondary ablation, not the default training-free claim.

It can add:

- Python parse success;
- Python compile success;
- no hidden unit tests.

This variant should be labeled `compiler_assisted_inference_time`, because it changes the method claim.

### Selector C: Oracle Upper Bound, Offline Only

This selector uses hidden unit-test pass/fail only to answer a diagnostic question:

> Did any candidate in the generated candidate set pass?

It is not a deployable policy. It is useful because it separates two failure modes:

- candidate set upper bound improves but Selector A fails: selection problem;
- candidate set upper bound does not improve: generation problem;
- both improve: full policy path is justified.

### V5.1 Selector: Anchor-Protected Len32

Smoke showed that `consensus_confidence` can pick `len24_s64` and lose a known Route2 precision `len32` win. V5.1 therefore adds a conservative selector:

```text
anchor = len32_s64
default = anchor
allow switch only to len32_s96 when its inference-visible score clears a margin
keep len24_s64 as diagnostic-only by default
```

This selector is still training-free and verifier-free. It is intended to preserve the already observed Route2 precision `len32` gains before testing whether slower same-length decoding can add wins.

## Reporting Metrics

The full report must include:

- final selected pass rate and pairwise W/L/TP/TF vs `midcons`;
- pairwise vs Route2 precision `len32`;
- trigger count;
- average candidates per triggered row;
- average total seconds including probe;
- bucket deltas for `<=8`, `9-12`, `13-16`, `17-24`, `25+`;
- candidate-set oracle upper bound pass rate;
- Selector A vs Selector B vs oracle upper bound;
- triggered failed-long rescue recovery count;
- short/current-pass risk count;
- examples where:
  - oracle upper bound passes but Selector A fails;
  - Selector A beats Route2 precision len32;
  - Selector A loses relative to Route2 precision len32.

## Success Criteria

GPU full run is worth claiming as an improvement only if Selector A satisfies all:

- final pass count at least `803/1033`, i.e. at least `+2` over Route2 precision `len32`;
- losses vs `midcons` at most `2`;
- losses vs Route2 precision `len32` at most `3`;
- oracle `<=8` net loss at most `1`;
- oracle `17-24` plus `25+` net gain is positive;
- runtime overhead is reported and not pathological.

Stronger paper-level signal:

- final pass count at least `808/1033`;
- oracle `25+` improves by at least `1`;
- candidate-set oracle upper bound is meaningfully above Selector A, giving a clear next selection story, or Selector A itself improves.

## Kill Criteria

Stop or keep as diagnostic-only if:

- candidate-set oracle upper bound is not above Route2 precision `len32`;
- Selector A regresses against Route2 precision `len32`;
- losses vs `midcons` exceed `2`;
- oracle `<=8` loses more than `1` task;
- runtime overhead is too high for a small gain;
- gains depend only on hidden unit tests, parse/compile, or other non-default selectors.

## GPU Gate

Do not launch a full run until:

- runner implementation has focused unit tests;
- a tiny GPU smoke verifies schema, candidate logging, selector metadata, and summary output;
- GPU `2/3` are checked with `nvidia-smi` and are not occupied by other users' tasks, unless the user explicitly overrides;
- a fresh action brief records command, expected output, success criteria, and kill criteria.

## Implementation Shape

Planned runner:

- `clean_scripts/run_route2_rescue_quality_v5.py`

Planned tests:

- `tests/test_route2_rescue_quality_v5.py`

Planned output:

- `/home/shx/projects/dllm_infilling/outputs_clean/smoke_route2_rescue_quality_v5_*`
- `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_rescue_quality_v5_*`

Key implementation requirements:

- preserve existing Route2 precision trigger by default;
- log every generated candidate with candidate id, length, steps, trace features, syntax fields, and hidden-test pass/fail for offline accounting;
- make deployed selector explicit in result metrics;
- never use hidden test pass/fail as policy input for Selector A/B;
- compute oracle upper bound separately and label it offline-only.

## Claim Boundary

If V5 succeeds, the claim is not "we solved length prediction." The claim is narrower:

> For triggered long-risk rows, rescue generation/selection quality matters; a small inference-time candidate set with a verifier-free selector improves over single fixed rescue.

If V5 fails, it is still useful:

> Existing triggered rows are not rescued by simple deterministic candidate diversity, so the next route needs a stronger generative mechanism or learned length/control model.

## Self-Review

- The design is different from V4 slice mining and from Route2 fixed-length rescue.
- It directly targets the `31/33` length-sufficient triggered failures.
- It keeps hidden verifier outcomes out of deployable policy selection.
- It includes an oracle upper bound to distinguish generation failure from selection failure.
- It requires smoke and GPU availability checks before full runs.
