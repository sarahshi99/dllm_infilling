# Post-V8 Rescue Quality And Local Guard Plan

> **Superpowers alignment:** This is the `superpowers:writing-plans` stage after `docs/paper_agent/experiments/20260701_next_step_brainstorm_after_v8.md`. The current environment does not expose callable `superpowers:*` skill files, so the project uses the documented local fallback. No subagents. No GPU work in this plan.

Date: 2026-07-02 CST

## Goal

Design and implement a CPU-only audit that answers two questions before any new GPU run:

1. **Rescue-quality question:** For Route2-triggered rows, why do many rows still fail even when rescue length is already enough?
2. **Local proportional-guard question:** Can V8's few long wins be isolated by an inference-visible guard, so proportional length reward is used only locally instead of globally?

The audit should produce a clear next decision:

- `new_rescue_action_candidate`;
- `local_prop_guard_candidate`;
- `diagnostic_only`;
- `stop_true_long_route_for_now`.

## Evidence Base

Current LLaDA-Base full-run anchors:

- `midcons`: `795/1033 = 76.96%`
- Route2 precision `len32`: `801/1033 = 77.54%`, pairwise `6/0/795/232`
- V6 short override: `802/1033 = 77.64%`, pairwise vs Route2 `1/0/801/231`
- V8a: `786/1033 = 76.09%`
- V8b: `781/1033 = 75.61%`
- V8c: `782/1033 = 75.70%`

Critical diagnostics:

- Route2 has real signal, but limited recall.
- Route2 misses `56` failed-long rows.
- Route2 still has `33` triggered failed-long rows.
- `31/33` triggered failed-long rows already have rescue length at least oracle.
- V8b changed `88` rows, but only `12` were true-long and `76` were not.

Interpretation:

- Blind length increase is not the right default.
- Global proportional length reward is not selective enough.
- The next useful CPU work must separate action failure types before proposing a GPU action.

## Non-Goals

- Do not launch GPU experiments.
- Do not tune another global V8 beta.
- Do not claim a new pass-rate result.
- Do not use oracle/pass/verifier labels as inference-time inputs.
- Do not make learned predictors the deployed method in this plan.
- Do not rewrite V5/V6 runner logic unless the audit later justifies a new action.

## Inputs

Required:

- `midcons` baseline:
  `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- Route2 precision `len32`:
  `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl`
- V6 short override:
  `outputs_clean/full_route2_v6_short_override_gpu2_20260620_124754/results.jsonl`
- V8a:
  `outputs_clean/full_v8a_propcal_beta002_gpu1_20260701_102654/results.jsonl`
- V8b:
  `outputs_clean/full_v8b_propcal_beta004_gpu1_20260701_120525/results.jsonl`
- V8c:
  `outputs_clean/full_v8c_propcal_beta004_cap32_gpu1_20260701_134849/results.jsonl`

Optional if present:

- V5.1 outputs and `candidate_upper_bound.csv`;
- Route2 broad/precision `len24` outputs;
- `step_traces.jsonl` files for feature enrichment.

## Planned Files

Implementation:

- `analysis/post_v8_rescue_quality_audit.py`
- `tests/test_post_v8_rescue_quality_audit.py`

Outputs:

- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/summary.json`
- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/report.md`
- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/row_action_table.csv`
- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/triggered_failure_taxonomy.csv`
- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/v8_changed_row_taxonomy.csv`
- `analysis_outputs/post_v8_rescue_quality_audit_<timestamp>/candidate_decision.md`

Documentation:

- `docs/paper_agent/experiments/<timestamp>_post_v8_rescue_quality_audit.md`
- update `docs/paper_agent/current_action.md`
- update `docs/paper_agent/activity_ledger.zh.md`

## Task 1: Action Brief

Write a short experiment brief before coding:

- action name;
- CPU-only scope;
- evidence base;
- input paths;
- expected outputs;
- success criteria;
- kill criteria;
- no-GPU statement.

Expected file:

- `docs/paper_agent/experiments/20260702_post_v8_rescue_quality_audit_action.md`

## Task 2: Tests First

Create focused unit tests for:

- loading/joining row results by `task_id`;
- deriving pass/fail, selected length, oracle length for offline accounting;
- pairwise W/L/TP/TF against `midcons`;
- taxonomy labels:
  - `triggered_rescued`;
  - `triggered_failed_length_insufficient`;
  - `triggered_failed_length_sufficient`;
  - `missed_failed_long`;
  - `v8_changed_win`;
  - `v8_changed_loss`;
  - `v8_changed_neutral`;
- forbidden-input protection: oracle/pass labels may appear only in offline accounting, not in candidate policy feature lists;
- summary JSON schema.

Expected command:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_post_v8_rescue_quality_audit.py
```

## Task 3: Implement Row-Action Table

Build a joined table with one row per `task_id`.

Columns:

- `task_id`;
- oracle bucket for offline accounting;
- `midcons_pass`, `midcons_len`;
- `route2_pass`, `route2_len`, pairwise role vs midcons;
- `v6_pass`, `v6_len`, pairwise role vs Route2 and midcons;
- `v8a_pass`, `v8a_len`, `v8a_delta_len`, pairwise role;
- `v8b_pass`, `v8b_len`, `v8b_delta_len`, pairwise role;
- `v8c_pass`, `v8c_len`, `v8c_delta_len`, pairwise role;
- inference-visible fields available from metrics, such as selected score, raw score, long ratio, stop reason, final source, trace endpoint fields where present.

The row-action table should be readable by future scripts and not depend on hidden verifier details beyond already recorded pass/fail accounting.

## Task 4: Rescue Failure Anatomy

Classify Route2/V6 triggered behavior.

Minimum taxonomy:

- `route2_win`: Route2 fails? No, midcons fails and Route2 passes.
- `route2_loss`: midcons passes and Route2 fails.
- `triggered_failed_length_insufficient`: Route2 triggered, failed, and selected/rescue length is below oracle.
- `triggered_failed_length_sufficient`: Route2 triggered, failed, and selected/rescue length is at least oracle.
- `triggered_candidate_selection_opportunity`: V6 or V5 candidate evidence suggests a different candidate can pass.
- `all_known_actions_fail`: all observed actions fail.

Report:

- counts by oracle bucket;
- examples;
- which failure type dominates;
- whether there is evidence for a new rescue action.

Decision evidence:

- If most failures are length-sufficient and all known actions fail, design should focus on generation quality, not length.
- If candidate-selection opportunities exist beyond V6's one row, design should focus on selector/action expansion.

## Task 5: Local Proportional Guard Audit

Analyze V8 changed rows, especially V8b because it has the strongest long gain and strongest loss signal.

Compute:

- changed rows count;
- true-long changed rows;
- non-true-long changed rows;
- wins/losses by oracle bucket;
- length delta distributions;
- feature differences between changed wins and changed losses;
- overlap with Route2 trigger and V6 trigger/action rows;
- examples of true-long wins and short/medium losses.

Search only simple offline candidate guards:

- selected length bucket;
- long-ratio / raw-ratio margin if present;
- selected score / raw score margin if present;
- stop reason;
- trace endpoint fields if present;
- Route2 trigger membership;
- V8 changed length delta threshold.

Candidate guard gate:

- at least `2` V8 wins retained;
- at most `1` V8 loss retained;
- no more than `1` short-bucket retained loss;
- rule is expressible in at most three clauses;
- labels are not used as policy inputs.

If no guard passes, mark proportional route as `diagnostic_only`.

## Task 6: Decision Report

Write a report with:

- top-line verdict;
- row-action summary table;
- rescue failure taxonomy;
- V8 changed-row taxonomy;
- candidate guard table;
- recommended next route.

Allowed decisions:

### `new_rescue_action_candidate`

Use if the audit identifies a plausible inference-time rescue action, such as:

- alternate schedule;
- targeted multi-candidate generation;
- candidate selector improvement with non-oracle features.

Next step: write a GPU smoke action brief.

### `local_prop_guard_candidate`

Use if the audit finds a high-precision V8 guard.

Next step: write a tiny targeted GPU smoke action brief, not a full run.

### `diagnostic_only`

Use if findings explain failures but no safe action emerges.

Next step: either learned-controller upper-bound diagnostic or paper framing.

### `stop_true_long_route_for_now`

Use if rescue failure anatomy shows current inference-time candidates are exhausted and no local guard exists.

Next step: shift to paper framing / cross-backbone evidence / learned-controller design.

## Verification

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_post_v8_rescue_quality_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/post_v8_rescue_quality_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/post_v8_rescue_quality_audit.py --help
git diff --check -- analysis/post_v8_rescue_quality_audit.py tests/test_post_v8_rescue_quality_audit.py docs/paper_agent/experiments/20260702_post_v8_rescue_quality_audit_action.md
```

Run the audit only after tests and compile pass.

## Success Criteria

The plan succeeds if it produces one of:

- a concrete rescue-action hypothesis with examples and risk bounds;
- a concrete local proportional guard candidate;
- a clear negative diagnosis that prevents another low-value GPU run.

## Kill Criteria

Stop and report diagnostic-only if:

- required input files are missing;
- row joins are incomplete or inconsistent;
- metrics needed for inference-visible guard search are absent;
- all candidate guards either retain too many losses or use forbidden labels;
- no known candidate action has evidence above V6's one-row improvement.

## After This Plan

If the decision is positive, the next Superpowers stage is:

- `superpowers:executing-plans` for CPU implementation already covered here, or
- a new `superpowers:writing-plans` action brief for one GPU smoke if CPU evidence supports it.

Do not jump directly from this plan to a full GPU run.
