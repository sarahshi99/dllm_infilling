# Discovery V4 Signal Model Plan

> **Superpowers alignment:** Use `superpowers:brainstorming`, `superpowers:using-git-worktrees`, and `superpowers:writing-plans` in local fallback. The current environment does not expose callable `superpowers:*` skill files. Do not use subagents or GPU jobs in this planning step.

**Goal:** Turn the Route2 error analysis result into a CPU-first Discovery V4 audit that searches for useful inference-visible signals and candidate mechanisms. The plan should avoid blind GPU runs and avoid treating opaque learned models as the final method.

**Current evidence:** Route2 precision `len32` is `801/1033 = 77.54%`, pairwise `6/0/795/232`; Route2 error analysis reports `33` triggered failed-long rows, `56` missed failed-long rows, and `31/33` triggered failed-long rows with rescue length at least oracle. Dominant bottleneck is `mixed_rescue_quality_and_gate_recall`.

## Execution Status

Status as of 2026-06-18 00:00 CST: completed.

- implementation: `analysis/discovery_v4_signal_audit.py`
- tests: `tests/test_discovery_v4_signal_audit.py`
- output: `analysis_outputs/discovery_v4_signal_audit_20260618_000000`
- result brief: `docs/paper_agent/experiments/20260618_discovery_v4_signal_audit.md`
- final decision: `route2_polish_only`

The audit did not find a non-leaking low-risk policy candidate. The best final candidate, `broad_len24_triggered >= 1`, is rejected because it triggers `10` short-risk rows while covering only `4` missed failed-long rows. No GPU action is justified from this plan alone.

## Task 0: Method Lineage Audit

Before implementing the CPU audit, verify that every discovery component has a clear methodological role:

- directly borrowed control framework:
  - risk-controlled selection;
  - held-out fold stability;
  - slice/subgroup coverage-risk reporting;
- discovery microscope only:
  - time-series shape probes;
  - weak-signal fusion;
  - partial uplift/logged-policy diagnostics;
  - MBR/self-consistency quality ideas;
- project-specific fusion:
  - row-action taxonomy for DLLM code infilling;
  - `MissedLongHead` and `RescueQualityHead`;
  - Route2 variants as partial action evidence;
  - distilled inference-visible training-free gates.

Implementation rule:

- no model score may become a policy by default;
- every policy candidate must be traced back to readable clauses or a named low-complexity score;
- if the strongest result is opaque, the decision is `learned_controller_future_direction`, not a training-free claim.

## File Structure

Design files already written:

- `docs/superpowers/specs/2026-06-17-discovery-v4-signal-model-design.md`
- `docs/paper_agent/experiments/20260617_discovery_v4_literature_brainstorm.md`

Planned implementation files:

- `analysis/discovery_v4_signal_audit.py`
- `tests/test_discovery_v4_signal_audit.py`
- `analysis_outputs/discovery_v4_signal_audit_TIMESTAMP/`

Planned documentation updates after implementation:

- `docs/paper_agent/current_action.md`
- `docs/paper_agent/experiment_results.zh.md`
- `docs/paper_agent/experiment_results.en.md`
- `docs/paper_agent/paper_agent_dashboard.zh.md`
- `docs/paper_agent/paper_agent_dashboard.en.md`
- `docs/paper_agent/activity_ledger.zh.md`
- `docs/paper_agent/activity_ledger.en.md`
- `docs/paper_agent/pause_checkpoint.current.md`

## Task 1: Action Brief

Write or update `docs/paper_agent/current_action.md` to state:

- CPU-first only;
- no GPU while GPU `2/3` have other users' tasks;
- V4 searches signals and mechanisms, not just feature lists;
- oracle/pass/verifier labels are offline accounting labels only;
- final policy candidates must be inference-visible and reviewer-readable.

Verification:

```bash
git diff --check -- docs/paper_agent/current_action.md
```

## Task 2: Tests First

Create `tests/test_discovery_v4_signal_audit.py`.

Minimum tests:

- joins baseline, precision len32, precision len24, and broad len24 rows by task id;
- builds action outcome table with W/L/TP/TF for each policy;
- produces missed-long and triggered-failure labels for offline accounting;
- rejects oracle/pass/verifier labels from candidate feature columns;
- bins numeric features reproducibly;
- ranks slices with explicit risk penalties;
- writes expected output files.

Expected red command:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_discovery_v4_signal_audit.py
```

## Task 3: Implement Unified Row/Action Table

Implement:

- load JSONL rows;
- extract inference-visible features;
- merge action outcomes from:
  - current `midcons`;
  - Route2 precision `len32`;
  - Route2 precision `len24`;
  - Route2 broad `len24`;
- compute offline labels separately.

Outputs:

- `row_action_table.csv`
- `action_overlap.csv`
- `policy_delta_by_bucket.csv`

Success checks:

- joined common rows are `1033`;
- precision `len32` pairwise remains `6/0/795/232`;
- broad and precision len24 pairwise numbers match existing docs.

## Task 4: Implement Slice/Subgroup Discovery

Implement a dependency-light constrained slice miner:

- bin selected numeric features by quantiles and named thresholds;
- build single-clause and two-clause slices;
- evaluate:
  - missed failed-long coverage;
  - triggered failed-long coverage;
  - route2 win coverage;
  - short risk;
  - current-pass risk;
  - fold stability;
- sort by constrained utility and Pareto frontier.

Outputs:

- `slice_candidates.csv`
- `slice_pareto_frontier.csv`

Kill criteria:

- no slice triggers at least `5` missed failed-long rows under acceptable risk;
- all top slices rely on oracle/pass/verifier labels.

## Task 5: Implement Discovery Microscopes

Add small, auditable discovery aids:

- shallow decision-tree-like recursive splits if no extra dependency is needed;
- sparse score from standardized features if `sklearn` is available, otherwise skip cleanly;
- trace-shape motif table from existing trace fields;
- calibration residuals conditioned on stop reason and selected length;
- weak-heuristic vote table over hand-defined labeling functions;
- uplift diagnostics from broad/precision/len24/len32 disagreement.

Outputs:

- `rule_candidates.csv`
- `trace_shape_candidates.csv`
- `calibration_residuals.csv`
- `weak_signal_votes.csv`
- `uplift_diagnostics.csv`

Interpretation constraint:

- learned or fitted scores propose hypotheses only.
- final candidate must be distilled to a readable gate/action rule.

## Task 6: Policy Shortlist

Write `policy_shortlist.md` with one of these decisions:

- `probe_trace_fusion_candidate`
- `rescue_quality_candidate`
- `combined_candidate`
- `route2_polish_only`
- `stop_current_true_long_signals`

GPU gate for any candidate:

- held-out missed failed-long coverage at least `8`, or triggered rescue-failure slice at least `10`;
- short-risk count at most `3`;
- current-pass-risk count at most `3`;
- stable on at least `4/5` folds;
- mechanism has a specific action beyond "try more length";
- GPU `2/3` confirmed free before launch.

## Task 7: Documentation And Verification

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_discovery_v4_signal_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/discovery_v4_signal_audit.py
git diff --check
```

Update paper-agent docs with:

- the decision;
- top candidate slices/rules;
- why any GPU action is or is not justified;
- relation to Route2 precision `len32`;
- explicit note that this is not a new pass-rate claim.

## Task 8: Commit Boundary

Commit only after:

- focused tests pass;
- generated outputs reproduce known baseline/Route2 counts;
- docs reflect the result;
- local diff review confirms no oracle/pass labels in policy feature columns;
- `git diff --check` passes.

Suggested commit message:

```bash
git commit -m "docs: plan discovery v4 signal model audit"
```

for this planning-only step, and later:

```bash
git commit -m "analysis: add discovery v4 signal audit"
```

for implementation.
