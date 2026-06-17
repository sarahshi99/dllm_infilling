# Route2 Error Analysis Discovery V3 Plan

> **Superpowers alignment:** Use `superpowers:brainstorming`, `superpowers:using-git-worktrees`, and `superpowers:writing-plans` in local serial fallback. The current environment does not expose callable `superpowers:*` skill files. Do not use subagents, Task/Spawn, reviewer subagents, parallel-agent dispatch, or `tool_search`.

**Goal:** Implement a CPU-only Route2 error analysis that turns the precision `len32` full-run outcomes into a stronger Discovery-layer V3 design. The aim is to continue searching for useful signals, not to abandon true-long rescue prematurely.

**Worktree gate:** Current status is a clean dedicated branch `paper-agent-overnight`. This planning commit can stay in place. Before code implementation, re-check `git status`; if unrelated work appears or runner changes become broad, create an isolated worktree. If the implementation only adds one analysis script, one test file, and paper-agent docs, it may proceed in this branch after confirming cleanliness.

## File Structure

Planned implementation files:

- Create `analysis/route2_error_analysis.py`
- Create `tests/test_route2_error_analysis.py`
- Create output directory `analysis_outputs/route2_error_analysis_YYYYMMDD_HHMMSS/`

Planned documentation files:

- Update `docs/paper_agent/current_action.md`
- Create or update `docs/paper_agent/experiments/20260617_route2_error_analysis_discovery_v3.md`
- Update `docs/paper_agent/experiment_results.zh.md`
- Update `docs/paper_agent/experiment_results.en.md`
- Update `docs/paper_agent/paper_agent_dashboard.zh.md`
- Update `docs/paper_agent/paper_agent_dashboard.en.md`
- Update `docs/paper_agent/activity_ledger.zh.md`
- Update `docs/paper_agent/activity_ledger.en.md`
- Update `docs/paper_agent/pause_checkpoint.current.md`

## Task 1: Write Action Brief

**Files:**

- Modify `docs/paper_agent/current_action.md`
- Create `docs/paper_agent/experiments/20260617_route2_error_analysis_discovery_v3.md`

**Content requirements:**

- State that this is CPU-only.
- State that the purpose is to improve the v2 Discovery layer.
- List inputs:
  - baseline `midcons` results;
  - Route2 precision `len32` results;
  - optional Route2 broad/precision len24 results for comparison.
- Record known summary numbers:
  - precision `len32` pairwise `6/0/795/232`;
  - triggered true-long precision `61.40%`;
  - triggered-but-still-failed true-long `33`;
  - missed failed-long `56`;
  - `31/33` triggered failed-long have rescue length at least oracle length.
- Define success and kill criteria.

**Verification:**

```bash
git diff --check -- docs/paper_agent/current_action.md docs/paper_agent/experiments/20260617_route2_error_analysis_discovery_v3.md
```

## Task 2: Write Tests First

**Files:**

- Create `tests/test_route2_error_analysis.py`

**Test cases:**

- bucket assignment: `<=8`, `9-12`, `13-16`, `17-24`, `25+`;
- pairwise classification: win, loss, tie-pass, tie-fail;
- error taxonomy classification:
  - `triggered_failed_long`;
  - `triggered_rescued_long`;
  - `missed_failed_long`;
  - `short_or_medium_win`;
- rescue length sufficiency accounting;
- report writer creates `summary.json`, CSV files, and `report.md`;
- oracle/pass/verifier labels are not included in inference-feature candidate columns.

**Expected red command:**

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_error_analysis.py
```

## Task 3: Implement CPU Diagnostic

**Files:**

- Create `analysis/route2_error_analysis.py`

**CLI:**

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/route2_error_analysis.py \
  --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl \
  --route2-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl \
  --output-dir analysis_outputs/route2_error_analysis_YYYYMMDD_HHMMSS
```

**Outputs:**

- `summary.json`
- `error_taxonomy.csv`
- `triggered_failed_long.csv`
- `missed_failed_long.csv`
- `wins.csv`
- `feature_contrast.csv`
- `report.md`

**Required summary assertions:**

- rows joined: `1033`;
- pairwise counts: `6/0/795/232`;
- triggered count: `57`;
- triggered-but-still-failed true-long: `33`;
- missed baseline failed-long: `56`;
- rescue-length-at-least-oracle among triggered failed-long: `31/33`.

## Task 4: Run Diagnostic

**Command:**

Use the CLI in Task 3 with a timestamped output directory.

**Terminal output must include:**

- pairwise summary;
- bucket summary;
- triggered failed-long count;
- missed failed-long count;
- dominant bottleneck decision;
- next recommended path.

## Task 5: Update Paper-Agent Docs

**Files:**

- Update result/dashboard/checkpoint/activity ledger bilingual docs.
- Add the diagnostic report path to `experiment_results.*.md`.

**Interpretation constraints:**

- Do not upgrade the central claim unless the diagnostic finds a concrete inference-visible path.
- Do not call Route2 `SOTA`.
- Keep paper-reported numbers separate from local methods.
- If the diagnostic supports generation-side rescue, say so explicitly.
- If it supports probe-trace fusion, define the exact next candidate gate family.

## Task 6: Verification And Local Review

**Commands:**

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_error_analysis.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/route2_error_analysis.py
git diff --check
```

**Reviewer fallback:**

Independent reviewer/subagent is disabled by project policy. Perform local diff review and record residual risk.

## Task 7: Commit Boundary

Commit only after:

- tests pass;
- diagnostic output reproduces known counts;
- docs reflect the result;
- `git diff --check` passes.

Suggested commit message:

```bash
git commit -m "analysis: add route2 error analysis diagnostic"
```

## Expected Decision After Execution

The diagnostic should choose one of:

- `rescue_generation_quality`: prioritize better rescue decoding/selection;
- `gate_recall`: prioritize trace/probe fusion;
- `adaptive_length`: only if length insufficiency is actually common;
- `route2_polish_only`: keep precision `len32` as incremental evidence;
- `stop_true_long_current_signals`: record negative result and stop this branch.

The current prior is `rescue_generation_quality + gate_recall`, because `31/33` triggered failed-long rows already had rescue length at least oracle length, while `56` failed-long rows were missed.
