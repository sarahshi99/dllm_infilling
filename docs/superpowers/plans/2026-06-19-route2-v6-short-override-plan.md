# Route2 V6 Short-Override Plan

Date: 2026-06-19 CST
Spec: `docs/superpowers/specs/2026-06-19-route2-v6-short-override-design.md`

## Objective

Run a CPU-only audit to decide whether a conservative `len24_s64` override is credible after V5.1.

## Tasks

### Task 1: Build Triggered Candidate Parser

- Read V5.1 `results.jsonl`.
- Extract only Route2-triggered rows.
- Flatten candidate records for `len24_s64`, `len32_s64`, and `len32_s96`.
- Preserve task id for reporting only, not for rule inputs.

Verification:

- Reproduce `57` triggered rows.
- Reproduce candidate upper-bound `9`.

### Task 2: Join References And Reproduce Accounting

- Join baseline `midcons` and Route2 precision `len32` reference by task id.
- Reproduce V5.1 pass `801/1033`.
- Reproduce pairwise vs Route2 precision `len32`: `0/0/801/232`.

Verification:

- Unit tests for pairwise counting.
- Real-data run prints expected summary.

### Task 3: Extract Inference-Visible Candidate Features

- Extract trace features already stored in candidate records.
- Extract candidate syntax fields.
- Compute pairwise feature deltas between `len24_s64` and `len32_s64`.
- Compute simple text/structure summaries without task-specific lexical memorization.

Verification:

- Unit tests ensure forbidden labels are not included in feature columns.
- Unit tests cover missing candidate fields.

### Task 4: Score Conservative Override Rules

- Generate single-feature thresholds.
- Generate pairwise thresholds over delta features.
- Generate dominance rules with safety vetoes.
- Separate pure verifier-free rules from compiler-assisted rules.

Verification:

- Rule scoring reports:
  - total override triggers;
  - captured `len24`-only wins;
  - anchor losses;
  - short risk;
  - current-pass risk;
  - complexity.

### Task 5: Write Audit Report

Outputs:

- `summary.json`
- `triggered_candidate_table.csv`
- `rule_candidates.csv`
- `report.md`

The report must include:

- reproduced V5.1 accounting;
- upper-bound-only rows;
- anchor-risk rows;
- best pure verifier-free rules;
- best compiler-assisted rules, if any;
- final decision: `policy_candidate`, `diagnostic_only`, or `reject_selector_only`.

### Task 6: Verification And Review

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_v6_short_override_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/route2_v6_short_override_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/route2_v6_short_override_audit.py --help
git diff --check
```

Perform local review:

- confirm no GPU command exists in the script;
- confirm oracle/pass labels are excluded from policy feature names;
- confirm report does not overclaim.

## Stop Conditions

Stop after CPU audit and report. Do not launch GPU.

If the audit finds a `policy_candidate`, the next turn should write a fresh GPU action brief and wait for GPU availability/user confirmation.

If the audit returns `diagnostic_only` or `reject_selector_only`, update documentation and move to rescue candidate generation design.
