# Probe-Curve Learned Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only strict-split diagnostic that tests whether multivariate probe-curve features can identify failed true-long infilling cases more safely than any single-feature threshold.

**Architecture:** Add a new analysis script that reuses feature extraction and metric helpers from `analysis/analyze_probe_curve_long_signals.py`. The script creates deterministic folds from `task_id`, fits a dependency-free linear risk scorer on train folds only, selects thresholds on train only, evaluates held-out triggers, and writes compact JSON/Markdown evidence under `docs/paper_agent/`.

**Tech Stack:** Python standard library, existing `analysis/analyze_probe_curve_long_signals.py`, `unittest`, tracked Markdown/JSON summaries.

---

### Task 1: Red Tests For Strict-Split Learned Scoring

**Files:**
- Create: `tests/test_analyze_probe_curve_split_score.py`
- Create later: `analysis/analyze_probe_curve_split_score.py`

- [x] **Step 1: Write the failing test file**

```python
from __future__ import annotations

import unittest

from analysis import analyze_probe_curve_split_score as split_score


def record(task_id: str, *, failed_long: bool, true_long: bool, short: bool, passed: bool, features: dict[str, float]) -> dict:
    return {
        "task_id": task_id,
        "failed_long": failed_long,
        "true_long": true_long,
        "short": short,
        "passed": passed,
        "features": features,
    }


class AnalyzeProbeCurveSplitScoreTest(unittest.TestCase):
    def test_stable_fold_is_deterministic_and_bounded(self) -> None:
        first = split_score.stable_fold("HumanEval/1", folds=5)
        second = split_score.stable_fold("HumanEval/1", folds=5)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 5)

    def test_fit_linear_signal_prefers_features_that_separate_failed_long(self) -> None:
        rows = [
            record("pos-a", failed_long=True, true_long=True, short=False, passed=False, features={"long_low": 0.1, "noise": 0.5}),
            record("pos-b", failed_long=True, true_long=True, short=False, passed=False, features={"long_low": 0.2, "noise": 0.4}),
            record("risk-a", failed_long=False, true_long=False, short=True, passed=True, features={"long_low": 0.9, "noise": 0.5}),
            record("risk-b", failed_long=False, true_long=False, short=True, passed=True, features={"long_low": 0.8, "noise": 0.4}),
        ]
        model = split_score.fit_linear_signal(rows, ["long_low", "noise"])
        scores = {row["task_id"]: split_score.score_record(row, model) for row in rows}
        self.assertLess(scores["risk-a"], scores["pos-a"])
        self.assertLess(scores["risk-b"], scores["pos-b"])
        self.assertLess(model["weights"]["long_low"], 0.0)

    def test_train_threshold_then_evaluate_heldout_uses_selected_threshold(self) -> None:
        train = [
            record("pos-a", failed_long=True, true_long=True, short=False, passed=False, features={"risk": 2.0}),
            record("pos-b", failed_long=True, true_long=True, short=False, passed=False, features={"risk": 1.8}),
            record("risk-a", failed_long=False, true_long=False, short=True, passed=True, features={"risk": -1.0}),
            record("risk-b", failed_long=False, true_long=False, short=True, passed=True, features={"risk": -1.2}),
        ]
        heldout = [
            record("pos-heldout", failed_long=True, true_long=True, short=False, passed=False, features={"risk": 1.9}),
            record("short-heldout", failed_long=False, true_long=False, short=True, passed=True, features={"risk": -0.9}),
        ]
        model = {"features": ["risk"], "means": {"risk": 0.0}, "stdevs": {"risk": 1.0}, "weights": {"risk": 1.0}}
        train_scores = [split_score.score_record(row, model) for row in train]
        threshold = split_score.select_threshold(train, train_scores, min_failed_long_triggers=2, max_short_risk=0.05)
        heldout_scores = [split_score.score_record(row, model) for row in heldout]
        result = split_score.evaluate_scored_records(heldout, heldout_scores, threshold["threshold"])
        self.assertEqual(result["trigger_count"], 1)
        self.assertEqual(result["failed_long_trigger_count"], 1)
        self.assertAlmostEqual(result["short_risk_rate"], 0.0)
```

- [x] **Step 2: Run the test to verify it fails because the module is missing**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_split_score.py
```

Expected: `ImportError` for `analysis.analyze_probe_curve_split_score`.

### Task 2: Implement Dependency-Free Strict-Split Scoring

**Files:**
- Create: `analysis/analyze_probe_curve_split_score.py`
- Modify: none
- Test: `tests/test_analyze_probe_curve_split_score.py`

- [x] **Step 1: Create the script**

Implement:

- `stable_fold(task_id, folds=5)` using `hashlib.sha256`.
- `feature_names(records)` returning numeric feature names present in at least 95% of rows.
- `fit_linear_signal(records, features)` using train-only means, stdevs, and mean-difference weights between failed-long positives and risk negatives (`short` or `passed`).
- `score_record(record, model)` using train-only standardization and learned weights.
- `select_threshold(records, scores, min_failed_long_triggers=8, max_short_risk=0.05)` selecting train thresholds only; if no strict train threshold exists, return the best penalized fallback with `strict_train_pass=False`.
- `evaluate_scored_records(records, scores, threshold)` matching the existing audit metric names.
- `cross_validate(records, folds=5)` aggregating held-out predictions across folds.
- `build_audit(rows)` that uses `analysis.analyze_probe_curve_long_signals.build_record`.
- `write_json`, `write_markdown`, and CLI defaults:
  - input: same A6000 `midcons` `results.jsonl`
  - JSON: `docs/paper_agent/probe_curve_split_score_audit.json`
  - Markdown: `docs/paper_agent/probe_curve_split_score_audit.md`

- [x] **Step 2: Run the new test**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_split_score.py
```

Expected: `OK`.

### Task 3: Generate Evidence And Update Paper-Agent Docs

**Files:**
- Create: `docs/paper_agent/probe_curve_split_score_audit.json`
- Create: `docs/paper_agent/probe_curve_split_score_audit.md`
- Create: `docs/paper_agent/probe_curve_split_score_audit.zh.md`
- Modify: `docs/paper_agent/experiment_results.en.md`
- Modify: `docs/paper_agent/experiment_results.zh.md`
- Modify: `docs/paper_agent/open_questions.en.md`
- Modify: `docs/paper_agent/open_questions.zh.md`
- Modify: `docs/paper_agent/overnight_log.en.md`
- Modify: `docs/paper_agent/overnight_log.zh.md`
- Modify: `docs/paper_agent/paper_agent_dashboard.en.md`
- Modify: `docs/paper_agent/paper_agent_dashboard.zh.md`

- [x] **Step 1: Run the script**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_split_score.py
```

Expected: writes the JSON and English Markdown audit without launching GPU work.

- [x] **Step 2: Write the Chinese audit**

Translate the English audit faithfully into `docs/paper_agent/probe_curve_split_score_audit.zh.md`.

- [x] **Step 3: Update paper-agent docs**

Record whether held-out strict-split scoring passes the GPU gate. If it fails, preserve the negative result and keep GPU work blocked. If it passes, update current docs as a candidate offline gate but do not launch GPU work in the same milestone.

Outcome: the held-out strict-split score failed the gate (`strict_heldout_pass=False`, `22.22%` short-risk), so the negative result was preserved and GPU work remains blocked.

### Task 4: Verification, Review, Commit, Push

**Files:**
- All files touched in Tasks 1-3.

- [ ] **Step 1: Run focused verification**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_long_signals.py tests/test_analyze_probe_curve_split_score.py tests/test_build_paper_agent_evidence_snapshot.py tests/test_diagnose_long_underestimate_policy.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/analyze_probe_curve_long_signals.py analysis/analyze_probe_curve_split_score.py analysis/build_paper_agent_evidence_snapshot.py
git diff --check -- analysis/analyze_probe_curve_split_score.py tests/test_analyze_probe_curve_split_score.py docs/paper_agent docs/superpowers/plans/2026-05-31-probe-curve-learned-diagnostic.md
```

Expected: all commands exit `0`.

- [ ] **Step 2: Use code-review discipline**

If an independent reviewer tool is visible, request review. If not, record the blocked reviewer status and perform local diff review plus fresh verification.

- [ ] **Step 3: Commit and push**

Stage only intended analysis, test, paper-agent, and plan files. Do not stage `AGENTS.md` or `AGENTS.zh.md`.

```bash
git add analysis/analyze_probe_curve_split_score.py tests/test_analyze_probe_curve_split_score.py docs/paper_agent docs/superpowers/plans/2026-05-31-probe-curve-learned-diagnostic.md
git commit -m "analysis: add split probe curve diagnostic"
git push
```
