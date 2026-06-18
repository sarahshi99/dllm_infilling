# Route2 Rescue Quality V5 Plan

> **Superpowers alignment:** Use `superpowers:brainstorming`, `superpowers:using-git-worktrees`, and `superpowers:writing-plans` in local fallback. The current environment does not expose callable `superpowers:*` skill files. Do not launch GPU jobs from this planning step.

**Goal:** Implement and validate a new Route2 rescue generation/selection mechanism that targets triggered rows where fixed `len32` rescue still fails despite enough length.

**Spec:** `docs/superpowers/specs/2026-06-18-route2-rescue-quality-v5-design.md`

**Current evidence:** Route2 precision `len32` is `801/1033 = 77.54%`, pairwise `6/0/795/232`, with `57` triggers. Error analysis found `33` triggered failed-long rows and `31/33` already have rescue length >= oracle. Discovery V4 found no GPU-ready non-leaking gate and ended as `route2_polish_only`.

## Task 1: Action Brief

Write `docs/paper_agent/experiments/20260618_route2_rescue_quality_v5_action.md`.

It must state:

- this is a new mechanism, not a Route2 threshold sweep;
- no GPU command should start until implementation tests and smoke criteria are met;
- main selector is verifier-free consensus/confidence;
- syntax-aware selector is an ablation;
- hidden unit tests are offline oracle upper bound only;
- GPU `2/3` must be checked before any smoke/full run.

## Task 2: Tests First

Create `tests/test_route2_rescue_quality_v5.py`.

Minimum tests:

- candidate quality score uses only inference-visible fields;
- selector A never reads `passed`, hidden unit-test fields, oracle length, or pairwise outcome;
- syntax-aware selector can read parse/compile fields but is labeled separately;
- oracle upper bound is computed only in offline summary;
- selected candidate metadata is logged with selector name and candidate id;
- pairwise summary vs baseline and vs Route2 precision is computed from rows.

Expected red command:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_rescue_quality_v5.py
```

## Task 3: Implement Runner

Create `clean_scripts/run_route2_rescue_quality_v5.py`.

Implementation requirements:

- reuse `clean_scripts/run_route2_trace_rescue.py` for:
  - midcons primary decode;
  - precision Route2 trigger;
  - fixed-length rescue helper;
  - pairwise/bucket summary style;
- add candidate set generation for triggered rows:
  - initial cheap set: `len24_s64`, `len32_s64`, `len32_s96`;
  - optional full set after smoke: add `len40_s64`, `len40_s96`;
- compute verifier-free selector score from:
  - text consensus similarity;
  - final trace confidence/top1/gap;
  - final remaining mask ratio;
  - normalized length penalty;
- optionally compute syntax-aware score from parse/compile diagnostics;
- log every candidate under `route2_rescue_quality_v5.candidates`;
- set final selected result according to the requested selector.

## Task 4: CPU Verification

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_rescue_quality_v5.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_route2_rescue_quality_v5.py
git diff --check
```

Do not proceed to GPU if any command fails.

## Task 5: GPU Smoke Gate

Only after Task 4 passes and GPU `2/3` are free:

- run `max-samples 5` or `max-samples 10`;
- use precision trigger;
- use cheap candidate set;
- selector A: verifier-free consensus/confidence;
- save step traces;
- use HF mirror/proxy settings if model loading needs them.

Smoke success:

- command exits `0`;
- output has requested number of valid `results.jsonl` rows;
- every triggered row has candidate metadata;
- `summary.json` reports selector A, syntax-aware diagnostic, and oracle upper bound;
- no schema mismatch in downstream analysis.

Smoke is not a performance claim.

## Task 6: Full Run Decision

Only launch a full run if the smoke passes and the user approves.

Full run success criteria for Selector A:

- pass count at least `803/1033`;
- losses vs `midcons` at most `2`;
- losses vs Route2 precision `len32` at most `3`;
- oracle `<=8` net loss at most `1`;
- oracle `17-24` plus `25+` net gain positive;
- runtime overhead reported.

Full run outputs:

- `summary.json`;
- `candidate_upper_bound.csv`;
- `candidate_selection_failures.csv`;
- `pairwise_vs_midcons.csv`;
- `pairwise_vs_route2_precision_len32.csv`;
- experiment brief update.

## Task 7: Documentation

Update docs only after smoke/full results are available:

- `docs/paper_agent/experiments/20260618_route2_rescue_quality_v5_action.md`
- `docs/paper_agent/experiment_results.zh.md`
- `docs/paper_agent/paper_agent_dashboard.zh.md`
- `docs/paper_agent/activity_ledger.zh.md`
- English counterparts if already touched by the current milestone.

Do not overwrite unrelated advisor/PPT dirty files.

## Task 8: Commit Boundary

Commit design-only files separately from implementation/results if useful.

Suggested commits:

```bash
git commit -m "docs: plan route2 rescue quality v5"
git commit -m "analysis: add route2 rescue quality v5 runner"
```

## Stop Conditions

Stop before GPU if:

- candidate selector requires hidden verifier outcomes;
- implementation cannot log candidate metadata cleanly;
- smoke cannot reproduce baseline/Route2 pairwise accounting;
- GPU `2/3` are occupied by other users and the user has not explicitly overridden.
