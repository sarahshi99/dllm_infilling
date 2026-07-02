# Proportional Length Widening V7 Action

Date: 2026-06-30 CST

## Action Name

CPU audit for proportional length widening in length estimation.

## Stage And Workflow

Superpowers-style workflow alignment:

- `superpowers:brainstorming`: evaluate the user's idea before changing GPU runners.
- `superpowers:using-git-worktrees`: current branch `paper-agent-overnight` is already the active paper-agent branch; no new worktree is needed for a CPU-only analysis.
- `superpowers:writing-plans`: this action brief defines scope, inputs, success criteria, and kill criteria.
- `superpowers:executing-plans`: implement serially; subagents remain disabled by project protocol.
- `superpowers:test-driven-development`: add focused tests for the proportional selector and audit accounting.
- `superpowers:verification-before-completion`: run tests, compile, and diff hygiene before reporting.

The environment does not expose callable `superpowers:*` skill bodies as project-local tools, so this action follows the documented project protocol as local fallback.

## Reviewer Motivation

The current best LLaDA-Base result is V6 short override, `802/1033 = 77.64%`. It is clean but tiny. The user proposed a simpler alternative: during length estimation, use a ratio-based rule to allow longer predicted lengths to be selected more broadly. This is attractive because it keeps the method training-free and directly targets under-estimated length.

## Hypothesis

If a longer candidate has probe confidence close enough to the best candidate, a proportional rule may safely select that longer length. The threshold can become more permissive as `candidate_len / best_len` grows, e.g.:

```text
threshold(candidate) = max(min_threshold, base_threshold - slope * log(candidate_len / best_len))
```

The policy selects the longest candidate whose score passes this threshold, subject to a maximum expansion factor and optional raw-score confirmation.

## Baseline, Dataset, Model, Metric

- Dataset: HumanEval single-line infilling, `1033` rows.
- Model/backbone: `GSAI-ML/LLaDA-8B-Base`.
- Primary CPU source: current `midcons` trace run:
  `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- Comparison type: offline replay over stored probe candidate scores, not a pass-rate claim.
- Metrics:
  - selected length change count;
  - absolute length error delta;
  - under-selection and over-selection changes;
  - oracle-bucket improvement/worsening counts;
  - short-bucket promotion risk;
  - current-pass promotion risk.

## Planned Command

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/proportional_length_widening_audit.py \
  --results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl \
  --output-dir analysis_outputs/proportional_length_widening_v7_midcons_20260630
```

## GPU / Environment

No GPU work in this action. If CPU audit suggests the current probe grid is too narrow but the idea remains plausible, the next step should be a separate expanded-grid GPU smoke action brief.

## Expected Outcome

The audit should decide one of:

- `reject_existing_grid`: current probe scores do not support proportional widening.
- `needs_expanded_grid_smoke`: existing grid is too narrow, but the idea is plausible enough to collect expanded-grid probe data.
- `policy_candidate`: a conservative rule improves length accounting with bounded short/current-pass risk.

## Success Criteria

- Parse all `1033` rows.
- Write `summary.json`, `policy_sweep.csv`, `row_decisions.csv`, and `report.md`.
- Policy inputs are only probe scores/lengths and current selected length. Oracle/pass labels are offline evaluation only.
- Tests cover proportional thresholding, raw-score confirmation, missing candidates, and bucket accounting.

## Kill Criteria

- Reject if all useful variants worsen absolute length error or short risk.
- Reject if improvement is confined to short buckets rather than true-long/under-selected rows.
- Stop with `needs_expanded_grid_smoke` if the best conclusion is that the current compact grid lacks candidates above `24`.

## Known Risks

- Length accounting is a proxy; pass rate requires GPU generation.
- Prior Route2 analysis showed that many triggered long failures already had sufficient length, so this may not solve generation quality.
- Broadening long estimates can silently increase short/medium false positives.

## Documentation Outputs

- Update this file with audit results.
- If promising, create a new GPU smoke action brief rather than launching full GPU directly.

## CPU Audit Result

Command:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/proportional_length_widening_audit.py \
  --results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl \
  --output-dir analysis_outputs/proportional_length_widening_v7_midcons_20260630
```

Output:

- `analysis_outputs/proportional_length_widening_v7_midcons_20260630/summary.json`
- `analysis_outputs/proportional_length_widening_v7_midcons_20260630/report.md`
- `analysis_outputs/proportional_length_widening_v7_midcons_20260630/policy_sweep.csv`
- `analysis_outputs/proportional_length_widening_v7_midcons_20260630/row_decisions.csv`

Decision: `needs_expanded_grid_smoke`.

Key result:

| Item | Value |
|---|---:|
| rows | `1033` |
| max stored probe length | `24` |
| current avg abs length error | `3.0068` |
| best replay policy | `base0p985_slope0p02_min0p85_minbase12_maxx1p5_noraw` |
| promoted rows | `5` |
| improved rows | `0` |
| worsened rows | `5` |
| failed true-long improved | `0` |
| short promoted | `1` |
| avg abs error delta | `+0.0077` |

Interpretation:

The existing `midcons` stored probe grid is not enough to validate the user's intended "make long length estimates wider" idea. It only contains base probe candidates up to `24`, and replaying proportional widening over that grid does not improve long/failed-long length accounting. The best replay policy promotes only `5` rows, improves `0`, worsens `5`, and does not help oracle `17+`.

This does **not** fully reject the idea. It rejects the **existing-grid replay** version. A faithful test of the idea needs an expanded base probe grid such as:

```text
3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,40,48
```

The next meaningful step, if continuing this direction, is a small GPU smoke that records expanded-grid probe scores and applies proportional widening. It should not be a full run first.

## Verification

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_proportional_length_widening_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/proportional_length_widening_audit.py
git diff --check
```

Result: focused tests passed (`Ran 5 tests`, `OK`), compile passed, and diff check passed.
