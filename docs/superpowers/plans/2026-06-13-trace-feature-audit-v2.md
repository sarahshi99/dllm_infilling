# Trace Feature Audit V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` only. Do not use `superpowers:subagent-driven-development`, `superpowers:dispatching-parallel-agents`, Task/Spawn subagents, reviewer subagents, parallel-agent dispatch, or `tool_search` discovery. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only `trace_feature_audit_v2` that discovers and reports stable, risk-controlled trace/probe feature rules for true-long under-selection before any GPU policy run.

**Architecture:** Add one focused analysis script with testable helper functions for trace-shape features, deterministic folds, constrained candidate search, model-assisted diagnostics, and report writing. Keep learned/fitted models diagnostic only; candidate policies must be distilled into readable rules evaluated with held-out accounting. Use the two existing full LLaDA-Base trace outputs as the primary inputs.

**Tech Stack:** Python standard library, existing `analysis.trace_long_rescue_features` helpers, `unittest`, CSV/JSON/Markdown outputs, existing paper-agent docs. Optional `sklearn` may be used only behind a graceful fallback and must not be required for the audit to run.

---

## File Structure

- Create `analysis/trace_feature_audit_v2.py`: CPU-only CLI and reusable helpers for feature extraction, fold assignment, candidate generation/evaluation, model-assisted diagnostics, and report generation.
- Create `tests/test_trace_feature_audit_v2.py`: focused unit tests for trace-shape features, feature/label separation, fold stability, risk accounting, candidate generation, and report writing.
- Modify `docs/paper_agent/current_action.md`: point to the active implementation action and exact input/output paths.
- Modify after running the audit:
  - `docs/paper_agent/experiment_results.zh.md`
  - `docs/paper_agent/experiment_results.en.md`
  - `docs/paper_agent/paper_agent_dashboard.zh.md`
  - `docs/paper_agent/paper_agent_dashboard.en.md`
  - `docs/paper_agent/activity_ledger.zh.md`
  - `docs/paper_agent/activity_ledger.en.md`
  - `docs/paper_agent/pause_checkpoint.current.md`
- Create after running the audit:
  - `docs/paper_agent/experiments/20260613_trace_feature_audit_v2.md`
  - `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/`

Do not modify GPU runners in this plan. Do not create a route-specific GPU policy runner in this plan.

## Task 1: Write The Paper-Agent Action Brief

**Files:**
- Modify: `docs/paper_agent/current_action.md`
- Create: `docs/paper_agent/experiments/20260613_trace_feature_audit_v2.md`

- [ ] **Step 1: Replace `current_action.md` with the audit action**

Use this content:

```markdown
# Current Paper-Agent Action

Timestamp: 2026-06-13 CST

## Action Name

Implement and run CPU-only `trace_feature_audit_v2`.

## Current Phase

Post trace-long-rescue negative result. The approved next step is model-assisted offline feature discovery, not GPU execution.

## Reviewer Motivation

The first Route 1/2 formulas triggered zero rows. A reviewer would not accept that as proof trace features are useless. This audit tests richer trace-shape, stop-reason-conditioned, probe-trace fusion, and distilled-rule discovery families under held-out risk constraints.

## Hypothesis

The existing full trace data may contain useful true-long under-selection signals, but v1 missed them because its Boolean formula shape was too narrow.

## Inputs

- Previous local method results: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/results.jsonl`
- Previous local method traces: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/step_traces.jsonl`
- Current `midcons` results: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- Current `midcons` traces: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/step_traces.jsonl`

## Method Boundary

This is CPU-only offline analysis. Labels derived from oracle/pass/fail are allowed only for offline discovery and evaluation. Any learned/fitted model is diagnostic only unless a later plan explicitly changes the paper claim.

## Expected Command

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/trace_feature_audit_v2.py \
  --prev-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/results.jsonl \
  --prev-traces /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/step_traces.jsonl \
  --midcons-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl \
  --midcons-traces /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/step_traces.jsonl \
  --output-dir analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS \
  --folds 5
```

## Success Criteria

- The script runs without GPU.
- Both trace sources report `1033` joined rows.
- Reports include feature coverage, top single rules, top pairwise rules, stop-reason rules, motif rules, shallow-tree rules, model diagnostics, Pareto frontier, and final decision.
- The final decision is one of `policy_candidate`, `diagnostic_only`, `reject`, or `needs_new_data`.

## Kill Criteria

- Missing or malformed trace inputs.
- Joined rows differ from `1033` for either source.
- Candidate search uses oracle/pass labels as policy inputs.
- The only positive signal is opaque and cannot be distilled.
- Any command attempts to launch GPU work.
```

- [ ] **Step 2: Create the experiment brief**

Create `docs/paper_agent/experiments/20260613_trace_feature_audit_v2.md` with the same content as `current_action.md`, plus this final section:

```markdown
## Expected Documentation Outputs

- `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/summary.json`
- `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/candidates.csv`
- `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/pareto.csv`
- `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/report.md`
- Updated paper-agent results/dashboard/checkpoint only after the audit command has run.
```

- [ ] **Step 3: Verify markdown hygiene**

Run:

```bash
git diff --check -- docs/paper_agent/current_action.md docs/paper_agent/experiments/20260613_trace_feature_audit_v2.md
```

Expected: no output, exit code `0`.

- [ ] **Step 4: Commit Task 1**

Run:

```bash
git add docs/paper_agent/current_action.md docs/paper_agent/experiments/20260613_trace_feature_audit_v2.md
git commit -m "docs: add trace feature audit v2 action"
```

Expected: commit succeeds.

## Task 2: Add Tests For Trace Feature Extraction

**Files:**
- Create: `tests/test_trace_feature_audit_v2.py`
- Create later in Task 3: `analysis/trace_feature_audit_v2.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_trace_feature_audit_v2.py` with:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.trace_feature_audit_v2 import (
    CandidateRule,
    build_feature_record,
    compute_shape_features,
    evaluate_rule,
    fold_id,
    generate_single_feature_rules,
    load_records,
    write_report_outputs,
)


class TraceFeatureAuditV2FeatureTest(unittest.TestCase):
    def test_compute_shape_features_captures_slope_area_and_plateau(self) -> None:
        traces = [
            {"step": 0, "remaining_masks_after_update": 8, "mean_confidence": 0.20},
            {"step": 1, "remaining_masks_after_update": 6, "mean_confidence": 0.25},
            {"step": 2, "remaining_masks_after_update": 6, "mean_confidence": 0.22},
            {"step": 3, "remaining_masks_after_update": 6, "mean_confidence": 0.18},
        ]

        features = compute_shape_features(traces)

        self.assertEqual(features["trace_steps"], 4)
        self.assertAlmostEqual(features["remaining_slope"], -2.0 / 3.0)
        self.assertAlmostEqual(features["remaining_auc_norm"], 26.0 / 32.0)
        self.assertEqual(features["late_remaining_plateau_steps"], 2)
        self.assertAlmostEqual(features["confidence_slope"], -0.02 / 3.0)

    def test_build_feature_record_separates_policy_features_from_labels(self) -> None:
        row = {
            "task_id": "task-a",
            "metrics": {
                "passed": False,
                "oracle_mask_length": 20,
                "selected_mask_length": 8,
                "long_score_max": 0.7,
            },
        }
        traces = [
            {
                "step": 0,
                "remaining_masks_after_update": 8,
                "mean_confidence": 0.20,
                "stop_decision": {"reason": "before_min_stop_step", "mean_gap": 0.30},
            },
            {
                "step": 1,
                "remaining_masks_after_update": 6,
                "mean_confidence": 0.19,
                "stop_decision": {"reason": "mean_gap_below_threshold", "mean_gap": 0.12},
            },
        ]

        record = build_feature_record(row, traces, source_name="midcons")

        self.assertEqual(record["task_id"], "task-a")
        self.assertTrue(record["labels"]["failed_long"])
        self.assertTrue(record["labels"]["true_long"])
        self.assertFalse(record["labels"]["short_risk"])
        self.assertNotIn("oracle_mask_length", record["features"])
        self.assertNotIn("passed", record["features"])
        self.assertEqual(record["features"]["selected_len"], 8)
        self.assertEqual(record["features"]["stop_reason"], "mean_gap_below_threshold")
        self.assertEqual(record["source"], "midcons")

    def test_fold_id_is_deterministic_and_bounded(self) -> None:
        first = fold_id("SingleLineInfilling/HumanEval/0/L0", folds=5)
        second = fold_id("SingleLineInfilling/HumanEval/0/L0", folds=5)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 5)


class TraceFeatureAuditV2RuleTest(unittest.TestCase):
    def test_evaluate_rule_counts_failed_long_short_and_current_pass_risk(self) -> None:
        records = [
            {
                "task_id": "long-fail",
                "features": {"score": 0.9},
                "labels": {"failed_long": True, "true_long": True, "short_risk": False, "current_pass_risk": False},
            },
            {
                "task_id": "short-pass",
                "features": {"score": 0.8},
                "labels": {"failed_long": False, "true_long": False, "short_risk": True, "current_pass_risk": True},
            },
            {
                "task_id": "safe",
                "features": {"score": 0.1},
                "labels": {"failed_long": False, "true_long": False, "short_risk": False, "current_pass_risk": False},
            },
        ]
        rule = CandidateRule(name="score_ge_0.5", clauses=[["score", ">=", 0.5]])

        summary = evaluate_rule(rule, records)

        self.assertEqual(summary["trigger_count"], 2)
        self.assertEqual(summary["failed_long_count"], 1)
        self.assertEqual(summary["short_risk_count"], 1)
        self.assertEqual(summary["current_pass_risk_count"], 1)
        self.assertAlmostEqual(summary["true_long_precision"], 0.5)

    def test_generate_single_feature_rules_uses_numeric_thresholds(self) -> None:
        records = [
            {"features": {"score": 0.1}, "labels": {}},
            {"features": {"score": 0.5}, "labels": {}},
            {"features": {"score": 0.9}, "labels": {}},
        ]

        rules = generate_single_feature_rules(records, feature_names=["score"])

        self.assertTrue(any(rule.name.startswith("score_ge_") for rule in rules))
        self.assertTrue(any(rule.name.startswith("score_le_") for rule in rules))


class TraceFeatureAuditV2IoTest(unittest.TestCase):
    def test_load_records_joins_results_and_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            results = temp / "results.jsonl"
            traces = temp / "step_traces.jsonl"
            results.write_text(
                json.dumps(
                    {
                        "task_id": "task-a",
                        "metrics": {
                            "passed": False,
                            "oracle_mask_length": 20,
                            "selected_mask_length": 8,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            traces.write_text(
                json.dumps(
                    {
                        "task_id": "task-a",
                        "step": 0,
                        "remaining_masks_after_update": 8,
                        "mean_confidence": 0.2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_records(results, traces, source_name="previous")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["task_id"], "task-a")
        self.assertEqual(records[0]["features"]["trace_steps"], 1)

    def test_write_report_outputs_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = {
                "decision": "reject",
                "sources": {"midcons": {"rows": 1}},
                "top_candidates": [{"name": "score_ge_0.5", "trigger_count": 1}],
            }
            write_report_outputs(output_dir, summary, candidates=[{"name": "score_ge_0.5", "trigger_count": 1}])

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "candidates.csv").exists())
            self.assertTrue((output_dir / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_feature_audit_v2.py
```

Expected: failure with `ModuleNotFoundError` or missing symbols from `analysis.trace_feature_audit_v2`.

## Task 3: Implement Core Feature Extraction

**Files:**
- Create: `analysis/trace_feature_audit_v2.py`
- Test: `tests/test_trace_feature_audit_v2.py`

- [ ] **Step 1: Create the module with core helpers**

Create `analysis/trace_feature_audit_v2.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import join_results_and_traces, load_jsonl, metric


JsonDict = Dict[str, Any]


def _num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    parsed = _num(value)
    if parsed is None:
        return None
    return int(parsed)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _trace_step(trace: Mapping[str, Any]) -> int:
    step = _int(trace.get("step"))
    return -1 if step is None else step


def _decision(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    decision = trace.get("stop_decision") or {}
    if isinstance(decision, Mapping):
        return decision
    return {}


def _series_from_trace(traces: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for trace in traces:
        parsed = _num(trace.get(key))
        if parsed is None:
            parsed = _num(_decision(trace).get(key))
        if parsed is not None:
            values.append(parsed)
    return values


def _remaining_series(traces: Sequence[Mapping[str, Any]]) -> List[float]:
    return [
        parsed
        for parsed in (_num(trace.get("remaining_masks_after_update")) for trace in traces)
        if parsed is not None
    ]


def _slope(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return (values[-1] - values[0]) / float(len(values) - 1)


def _last_window(values: Sequence[float], size: int = 3) -> List[float]:
    if not values:
        return []
    return list(values[-size:])


def _count_equal_transitions(values: Sequence[float]) -> int:
    count = 0
    for left, right in zip(values, values[1:]):
        if abs(right - left) <= 1e-9:
            count += 1
    return count


def _max_plateau(values: Sequence[float]) -> int:
    best = 0
    current = 0
    for left, right in zip(values, values[1:]):
        if abs(right - left) <= 1e-9:
            current += 1
        else:
            best = max(best, current)
            current = 0
    return max(best, current)


def _last_decrease_index(values: Sequence[float]) -> Optional[int]:
    last: Optional[int] = None
    for index, (left, right) in enumerate(zip(values, values[1:]), start=1):
        if right < left:
            last = index
    return last


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(mean(values))


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(median(values))


def compute_shape_features(traces: Iterable[Mapping[str, Any]]) -> JsonDict:
    ordered = sorted((dict(trace) for trace in traces), key=_trace_step)
    remaining = _remaining_series(ordered)
    confidence = _series_from_trace(ordered, "mean_confidence")
    gap = _series_from_trace(ordered, "mean_gap")
    top1 = _series_from_trace(ordered, "mean_top1")
    late_remaining = _last_window(remaining, size=max(2, min(4, len(remaining))))
    first_remaining = remaining[0] if remaining else None
    remaining_auc = sum(remaining) if remaining else None
    remaining_auc_norm = (
        _safe_div(remaining_auc, float(len(remaining)) * max(abs(first_remaining), 1.0))
        if remaining_auc is not None and first_remaining is not None
        else None
    )
    last_decrease = _last_decrease_index(remaining)

    return {
        "trace_steps": len(ordered),
        "remaining_first": first_remaining,
        "remaining_last": remaining[-1] if remaining else None,
        "remaining_min": min(remaining) if remaining else None,
        "remaining_max": max(remaining) if remaining else None,
        "remaining_median": _median_or_none(remaining),
        "remaining_slope": _slope(remaining),
        "remaining_late_slope": _slope(late_remaining),
        "remaining_auc_norm": remaining_auc_norm,
        "remaining_equal_transition_frac": _safe_div(float(_count_equal_transitions(remaining)), float(max(len(remaining) - 1, 1))),
        "late_remaining_plateau_steps": _count_equal_transitions(late_remaining),
        "max_remaining_plateau_steps": _max_plateau(remaining),
        "last_remaining_decrease_step": last_decrease,
        "confidence_first": confidence[0] if confidence else None,
        "confidence_last": confidence[-1] if confidence else None,
        "confidence_min": min(confidence) if confidence else None,
        "confidence_max": max(confidence) if confidence else None,
        "confidence_median": _median_or_none(confidence),
        "confidence_slope": _slope(confidence),
        "confidence_late_slope": _slope(_last_window(confidence)),
        "gap_last": gap[-1] if gap else None,
        "gap_min": min(gap) if gap else None,
        "gap_median": _median_or_none(gap),
        "top1_last": top1[-1] if top1 else None,
        "top1_min": min(top1) if top1 else None,
        "top1_median": _median_or_none(top1),
    }


def fold_id(task_id: str, *, folds: int) -> int:
    if folds <= 1:
        return 0
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % folds


def _oracle_bucket(oracle_len: Optional[int]) -> str:
    if oracle_len is None:
        return "unknown"
    if oracle_len <= 8:
        return "<=8"
    if oracle_len <= 12:
        return "9-12"
    if oracle_len <= 16:
        return "13-16"
    if oracle_len <= 24:
        return "17-24"
    return "25+"


def build_feature_record(row: Mapping[str, Any], traces: Iterable[Mapping[str, Any]], *, source_name: str) -> JsonDict:
    ordered = sorted((dict(trace) for trace in traces), key=_trace_step)
    shape = compute_shape_features(ordered)
    oracle_len = _int(metric(row, "oracle_mask_length"))
    selected_len = _int(metric(row, "selected_mask_length", metric(row, "mask_length")))
    passed = bool(metric(row, "passed", False))
    last_decision = _decision(ordered[-1]) if ordered else {}
    stop_reason = last_decision.get("reason")
    final_remaining = shape.get("remaining_last")
    final_remaining_ratio = (
        _safe_div(float(final_remaining), float(max(selected_len, 1)))
        if final_remaining is not None and selected_len is not None
        else None
    )

    features: JsonDict = {
        "selected_len": selected_len,
        "stop_reason": str(stop_reason) if stop_reason is not None else "missing",
        "final_remaining_ratio_by_selected": final_remaining_ratio,
        "long_score_max": _num(metric(row, "long_score_max")),
        "best_long_len": _num(metric(row, "best_long_len")),
        "best_len": _num(metric(row, "best_len")),
        "base_selected_length": _num(metric(row, "base_selected_length")),
        "final_selected_length": _num(metric(row, "final_selected_length")),
    }
    features.update(shape)

    labels = {
        "passed": passed,
        "oracle_len": oracle_len,
        "oracle_bucket": _oracle_bucket(oracle_len),
        "failed_long": bool(oracle_len is not None and oracle_len >= 17 and not passed),
        "true_long": bool(oracle_len is not None and oracle_len >= 17),
        "short_risk": bool(oracle_len is not None and oracle_len <= 8),
        "current_pass_risk": passed,
    }
    return {
        "task_id": str(row.get("task_id")),
        "source": source_name,
        "features": features,
        "labels": labels,
    }


def load_records(results_path: str | Path, traces_path: str | Path, *, source_name: str) -> List[JsonDict]:
    rows = load_jsonl(results_path)
    traces = load_jsonl(traces_path)
    joined = join_results_and_traces(rows, traces)
    return [
        build_feature_record(payload["row"], payload["traces"], source_name=source_name)
        for task_id, payload in sorted(joined.items())
    ]
```

- [ ] **Step 2: Run tests and verify partial pass**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_feature_audit_v2.py
```

Expected: import succeeds, feature/fold/load tests pass, rule/report tests fail because `CandidateRule`, `evaluate_rule`, `generate_single_feature_rules`, and `write_report_outputs` are not implemented yet.

## Task 4: Implement Rule Evaluation And Candidate Generation

**Files:**
- Modify: `analysis/trace_feature_audit_v2.py`
- Test: `tests/test_trace_feature_audit_v2.py`

- [ ] **Step 1: Add rule dataclass and evaluation helpers**

Append this code before `load_records` or after it:

```python
@dataclass(frozen=True)
class CandidateRule:
    name: str
    clauses: List[List[Any]]
    family: str = "manual"

    def matches(self, record: Mapping[str, Any]) -> bool:
        features = record.get("features") or {}
        for feature_name, op, threshold in self.clauses:
            value = features.get(str(feature_name))
            if isinstance(threshold, str):
                matched = str(value) == threshold
            else:
                parsed = _num(value)
                if parsed is None:
                    return False
                if op == ">=":
                    matched = parsed >= float(threshold)
                elif op == "<=":
                    matched = parsed <= float(threshold)
                else:
                    raise ValueError(f"Unsupported operator: {op}")
            if not matched:
                return False
        return True


def evaluate_rule(rule: CandidateRule, records: Sequence[Mapping[str, Any]]) -> JsonDict:
    triggered = [record for record in records if rule.matches(record)]
    trigger_count = len(triggered)
    failed_long_count = sum(1 for record in triggered if record["labels"].get("failed_long"))
    true_long_count = sum(1 for record in triggered if record["labels"].get("true_long"))
    short_risk_count = sum(1 for record in triggered if record["labels"].get("short_risk"))
    current_pass_risk_count = sum(1 for record in triggered if record["labels"].get("current_pass_risk"))
    total_failed_long = sum(1 for record in records if record["labels"].get("failed_long"))
    return {
        "name": rule.name,
        "family": rule.family,
        "clauses": rule.clauses,
        "trigger_count": trigger_count,
        "failed_long_count": failed_long_count,
        "true_long_count": true_long_count,
        "short_risk_count": short_risk_count,
        "current_pass_risk_count": current_pass_risk_count,
        "failed_long_recall": _safe_div(float(failed_long_count), float(total_failed_long)),
        "true_long_precision": _safe_div(float(true_long_count), float(trigger_count)),
        "short_risk_rate": _safe_div(float(short_risk_count), float(trigger_count)),
        "current_pass_risk_rate": _safe_div(float(current_pass_risk_count), float(trigger_count)),
    }


def _numeric_values(records: Sequence[Mapping[str, Any]], feature_name: str) -> List[float]:
    values: List[float] = []
    for record in records:
        parsed = _num((record.get("features") or {}).get(feature_name))
        if parsed is not None and math.isfinite(parsed):
            values.append(parsed)
    return sorted(set(values))


def _thresholds(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    if len(values) <= 8:
        return list(values)
    positions = [0.10, 0.25, 0.50, 0.75, 0.90]
    thresholds: List[float] = []
    for position in positions:
        index = min(len(values) - 1, max(0, int(round(position * (len(values) - 1)))))
        thresholds.append(values[index])
    return sorted(set(thresholds))


def generate_single_feature_rules(records: Sequence[Mapping[str, Any]], *, feature_names: Sequence[str]) -> List[CandidateRule]:
    rules: List[CandidateRule] = []
    for feature_name in feature_names:
        values = _numeric_values(records, feature_name)
        for threshold in _thresholds(values):
            threshold_label = f"{threshold:.6g}".replace("-", "neg").replace(".", "p")
            rules.append(
                CandidateRule(
                    name=f"{feature_name}_ge_{threshold_label}",
                    family="single",
                    clauses=[[feature_name, ">=", threshold]],
                )
            )
            rules.append(
                CandidateRule(
                    name=f"{feature_name}_le_{threshold_label}",
                    family="single",
                    clauses=[[feature_name, "<=", threshold]],
                )
            )
    return rules


def generate_pairwise_rules(single_rules: Sequence[CandidateRule], *, max_rules: int = 200) -> List[CandidateRule]:
    selected = list(single_rules[:max_rules])
    rules: List[CandidateRule] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if left.clauses[0][0] == right.clauses[0][0]:
                continue
            rules.append(
                CandidateRule(
                    name=f"{left.name}_AND_{right.name}",
                    family="pairwise",
                    clauses=[left.clauses[0], right.clauses[0]],
                )
            )
    return rules


def generate_stop_reason_rules(records: Sequence[Mapping[str, Any]], base_rules: Sequence[CandidateRule]) -> List[CandidateRule]:
    reasons = sorted({str((record.get("features") or {}).get("stop_reason", "missing")) for record in records})
    rules: List[CandidateRule] = []
    for reason in reasons:
        for rule in base_rules:
            rules.append(
                CandidateRule(
                    name=f"stop_{reason}_AND_{rule.name}",
                    family="stop_reason",
                    clauses=[["stop_reason", "==", reason]] + rule.clauses,
                )
            )
    return rules
```

- [ ] **Step 2: Run tests**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_feature_audit_v2.py
```

Expected: all tests pass except the report-output test if `write_report_outputs` is not implemented yet.

## Task 5: Implement Cross-Fold Discovery And Model Diagnostics

**Files:**
- Modify: `analysis/trace_feature_audit_v2.py`
- Test: `tests/test_trace_feature_audit_v2.py`

- [ ] **Step 1: Add feature selection and ranking helpers**

Append this code:

```python
def numeric_feature_names(records: Sequence[Mapping[str, Any]]) -> List[str]:
    names = set()
    for record in records:
        for name, value in (record.get("features") or {}).items():
            parsed = _num(value)
            if parsed is not None and math.isfinite(parsed):
                names.add(name)
    return sorted(names)


def constrained_score(summary: Mapping[str, Any]) -> float:
    failed_long = float(summary.get("failed_long_count") or 0)
    true_precision = float(summary.get("true_long_precision") or 0.0)
    short_risk = float(summary.get("short_risk_count") or 0)
    pass_risk = float(summary.get("current_pass_risk_count") or 0)
    trigger_count = float(summary.get("trigger_count") or 0)
    if trigger_count == 0:
        return -1e9
    return failed_long * 10.0 + true_precision * 5.0 - short_risk * 4.0 - pass_risk * 2.0


def candidate_decision(summary: Mapping[str, Any]) -> str:
    failed_long = int(summary.get("failed_long_count") or 0)
    short_risk = int(summary.get("short_risk_count") or 0)
    pass_risk = int(summary.get("current_pass_risk_count") or 0)
    precision = float(summary.get("true_long_precision") or 0.0)
    if failed_long >= 10 and short_risk <= 5 and pass_risk <= 5 and precision >= 0.35:
        return "policy_candidate"
    if failed_long >= 5 and short_risk <= 10:
        return "diagnostic_only"
    return "reject"


def evaluate_candidates(candidates: Sequence[CandidateRule], records: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    summaries = [evaluate_rule(candidate, records) for candidate in candidates]
    for summary in summaries:
        summary["score"] = constrained_score(summary)
        summary["decision"] = candidate_decision(summary)
    return sorted(summaries, key=lambda item: (item["score"], item["failed_long_count"]), reverse=True)


def split_records(records: Sequence[Mapping[str, Any]], *, heldout_fold: int, folds: int) -> Tuple[List[JsonDict], List[JsonDict]]:
    train: List[JsonDict] = []
    heldout: List[JsonDict] = []
    for record in records:
        task_id = str(record["task_id"])
        if fold_id(task_id, folds=folds) == heldout_fold:
            heldout.append(dict(record))
        else:
            train.append(dict(record))
    return train, heldout


def discover_for_source(records: Sequence[Mapping[str, Any]], *, folds: int) -> JsonDict:
    feature_names = numeric_feature_names(records)
    fold_outputs: List[JsonDict] = []
    all_heldout: List[JsonDict] = []
    stable_feature_counts: Dict[str, int] = {}

    for heldout_fold in range(folds):
        train, heldout = split_records(records, heldout_fold=heldout_fold, folds=folds)
        single_rules = generate_single_feature_rules(train, feature_names=feature_names)
        train_ranked = evaluate_candidates(single_rules, train)
        top_train_rules = [
            CandidateRule(name=item["name"], family=item["family"], clauses=item["clauses"])
            for item in train_ranked[:50]
        ]
        pairwise_rules = generate_pairwise_rules(top_train_rules, max_rules=50)
        stop_rules = generate_stop_reason_rules(train, top_train_rules[:20])
        all_rules = top_train_rules + pairwise_rules + stop_rules
        heldout_ranked = evaluate_candidates(all_rules, heldout)
        best = heldout_ranked[0] if heldout_ranked else {"name": "none", "decision": "reject", "score": -1e9}
        for clause in best.get("clauses", []):
            stable_feature_counts[str(clause[0])] = stable_feature_counts.get(str(clause[0]), 0) + 1
        fold_outputs.append({"heldout_fold": heldout_fold, "best": best, "heldout_rows": len(heldout)})
        all_heldout.extend(heldout_ranked[:20])

    ranked = sorted(all_heldout, key=lambda item: (item.get("score", -1e9), item.get("failed_long_count", 0)), reverse=True)
    source_decision = "reject"
    if any(item.get("decision") == "policy_candidate" for item in ranked[:20]):
        source_decision = "policy_candidate"
    elif any(item.get("decision") == "diagnostic_only" for item in ranked[:20]):
        source_decision = "diagnostic_only"

    return {
        "rows": len(records),
        "true_long": sum(1 for record in records if record["labels"].get("true_long")),
        "failed_long": sum(1 for record in records if record["labels"].get("failed_long")),
        "short": sum(1 for record in records if record["labels"].get("short_risk")),
        "feature_count": len(feature_names),
        "folds": fold_outputs,
        "top_candidates": ranked[:100],
        "stable_features": sorted(stable_feature_counts.items(), key=lambda item: (-item[1], item[0])),
        "decision": source_decision,
    }
```

- [ ] **Step 2: Add a manual shapelet motif family**

Append this code:

```python
def generate_motif_rules(records: Sequence[Mapping[str, Any]]) -> List[CandidateRule]:
    feature_names = [
        "late_remaining_plateau_steps",
        "remaining_equal_transition_frac",
        "remaining_late_slope",
        "confidence_late_slope",
        "gap_last",
        "final_remaining_ratio_by_selected",
    ]
    rules = generate_single_feature_rules(records, feature_names=feature_names)
    motif_rules: List[CandidateRule] = []
    for rule in rules:
        motif_rules.append(
            CandidateRule(
                name=f"motif_{rule.name}",
                family="motif",
                clauses=rule.clauses,
            )
        )
    return motif_rules
```

- [ ] **Step 3: Add optional sklearn diagnostic fallback**

Append this code:

```python
def optional_sklearn_diagnostic(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    names = numeric_feature_names(records)
    if not names:
        return {"available": True, "reason": "no_numeric_features", "top_weights": []}
    x_rows: List[List[float]] = []
    y: List[int] = []
    for record in records:
        features = record.get("features") or {}
        x_rows.append([float(_num(features.get(name)) or 0.0) for name in names])
        y.append(1 if record["labels"].get("failed_long") else 0)
    if len(set(y)) < 2:
        return {"available": True, "reason": "one_class", "top_weights": []}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=0),
    )
    model.fit(x_rows, y)
    logistic = model.named_steps["logisticregression"]
    weights = list(logistic.coef_[0])
    ranked = sorted(zip(names, weights), key=lambda item: abs(item[1]), reverse=True)
    return {
        "available": True,
        "top_weights": [{"feature": name, "weight": weight} for name, weight in ranked[:20]],
    }
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_feature_audit_v2.py
```

Expected: still all tests pass except report output if not implemented yet. The optional sklearn diagnostic may report unavailable; that is acceptable.

## Task 6: Implement Report Outputs And CLI

**Files:**
- Modify: `analysis/trace_feature_audit_v2.py`
- Test: `tests/test_trace_feature_audit_v2.py`

- [ ] **Step 1: Add report writing helpers**

Append this code:

```python
def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def write_report_outputs(output_dir: str | Path, summary: Mapping[str, Any], *, candidates: Sequence[Mapping[str, Any]]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=_json_default)

    fieldnames = [
        "source",
        "name",
        "family",
        "decision",
        "trigger_count",
        "failed_long_count",
        "true_long_count",
        "short_risk_count",
        "current_pass_risk_count",
        "true_long_precision",
        "failed_long_recall",
        "short_risk_rate",
        "current_pass_risk_rate",
        "score",
    ]
    with (path / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({field: candidate.get(field) for field in fieldnames})

    pareto_rows = sorted(
        candidates,
        key=lambda item: (
            int(item.get("failed_long_count") or 0),
            -int(item.get("short_risk_count") or 0),
            -int(item.get("current_pass_risk_count") or 0),
        ),
        reverse=True,
    )[:50]
    with (path / "pareto.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in pareto_rows:
            writer.writerow({field: candidate.get(field) for field in fieldnames})

    lines = [
        "# Trace Feature Audit V2 Report",
        "",
        f"Decision: `{summary.get('decision')}`",
        "",
        "## Source Summary",
        "",
        "| Source | Rows | True-long | Failed-long | Short | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for source, source_summary in sorted((summary.get("sources") or {}).items()):
        lines.append(
            "| {source} | {rows} | {true_long} | {failed_long} | {short} | {decision} |".format(
                source=source,
                rows=source_summary.get("rows", 0),
                true_long=source_summary.get("true_long", 0),
                failed_long=source_summary.get("failed_long", 0),
                short=source_summary.get("short", 0),
                decision=source_summary.get("decision", "reject"),
            )
        )
    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
            "| Source | Candidate | Family | Decision | Triggers | Failed-long | Short risk | Current-pass risk | Precision |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in list(candidates)[:20]:
        lines.append(
            "| {source} | `{name}` | `{family}` | `{decision}` | {trigger_count} | {failed_long_count} | {short_risk_count} | {current_pass_risk_count} | {precision} |".format(
                source=candidate.get("source", "unknown"),
                name=candidate.get("name", "unknown"),
                family=candidate.get("family", "unknown"),
                decision=candidate.get("decision", "reject"),
                trigger_count=candidate.get("trigger_count", 0),
                failed_long_count=candidate.get("failed_long_count", 0),
                short_risk_count=candidate.get("short_risk_count", 0),
                current_pass_risk_count=candidate.get("current_pass_risk_count", 0),
                precision="{:.3f}".format(float(candidate.get("true_long_precision") or 0.0)),
            )
        )
    (path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 2: Add CLI orchestration**

Append this code at the end of the module:

```python
def run_audit(args: argparse.Namespace) -> JsonDict:
    sources = {
        "previous": load_records(args.prev_results, args.prev_traces, source_name="previous"),
        "midcons": load_records(args.midcons_results, args.midcons_traces, source_name="midcons"),
    }
    source_summaries: Dict[str, JsonDict] = {}
    all_candidates: List[JsonDict] = []
    diagnostics: Dict[str, JsonDict] = {}

    for source_name, records in sources.items():
        source_summary = discover_for_source(records, folds=args.folds)
        motif_ranked = evaluate_candidates(generate_motif_rules(records), records)
        for item in motif_ranked[:30]:
            item["source"] = source_name
        for item in source_summary["top_candidates"]:
            item["source"] = source_name
        source_summary["top_motif_candidates"] = motif_ranked[:30]
        source_summaries[source_name] = source_summary
        all_candidates.extend(source_summary["top_candidates"][:50])
        all_candidates.extend(motif_ranked[:30])
        diagnostics[source_name] = {
            "sklearn_sparse_logistic": optional_sklearn_diagnostic(records),
        }

    all_candidates = sorted(
        all_candidates,
        key=lambda item: (item.get("score", -1e9), item.get("failed_long_count", 0)),
        reverse=True,
    )
    decision = "reject"
    if any(item.get("decision") == "policy_candidate" for item in all_candidates[:30]):
        decision = "policy_candidate"
    elif any(item.get("decision") == "diagnostic_only" for item in all_candidates[:30]):
        decision = "diagnostic_only"

    summary = {
        "decision": decision,
        "sources": source_summaries,
        "diagnostics": diagnostics,
        "top_candidates": all_candidates[:100],
        "notes": [
            "CPU-only offline audit; no GPU policy run launched.",
            "Oracle/pass labels are used only for offline evaluation.",
            "Learned diagnostics are discovery aids, not the final training-free policy.",
        ],
    }
    write_report_outputs(args.output_dir, summary, candidates=all_candidates)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only trace feature audit v2")
    parser.add_argument("--prev-results", required=True)
    parser.add_argument("--prev-traces", required=True)
    parser.add_argument("--midcons-results", required=True)
    parser.add_argument("--midcons-traces", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_audit(args)
    print(json.dumps({"decision": summary["decision"], "sources": {k: v["decision"] for k, v in summary["sources"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests and compile**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_feature_audit_v2.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/trace_feature_audit_v2.py
```

Expected: `Ran 7 tests` or more with `OK`; py_compile exits `0`.

- [ ] **Step 4: Commit Tasks 2-6**

Run:

```bash
git add analysis/trace_feature_audit_v2.py tests/test_trace_feature_audit_v2.py
git commit -m "analysis: add trace feature audit v2"
```

Expected: commit succeeds.

## Task 7: Run The CPU-Only Audit On Full Trace Outputs

**Files:**
- Use: `analysis/trace_feature_audit_v2.py`
- Create: `analysis_outputs/trace_feature_audit_v2_YYYYMMDD_HHMMSS/`

- [ ] **Step 1: Print the exact action in terminal**

Run:

```bash
printf '%s\n' \
'TRACE FEATURE AUDIT V2' \
'Mode: CPU-only offline diagnostic' \
'GPU: none' \
'Inputs: previous and midcons LLaDA-Base full trace outputs' \
'Outputs: analysis_outputs/trace_feature_audit_v2_STAMP' \
'Kill: missing/malformed rows, joined rows not 1033, oracle/pass used as policy inputs'
```

Expected: terminal prints the action summary.

- [ ] **Step 2: Run the audit**

Run:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/trace_feature_audit_v2.py \
  --prev-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/results.jsonl \
  --prev-traces /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552/step_traces.jsonl \
  --midcons-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl \
  --midcons-traces /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/step_traces.jsonl \
  --output-dir "analysis_outputs/trace_feature_audit_v2_${STAMP}" \
  --folds 5
```

Expected: command exits `0`, prints a compact JSON decision, and creates `summary.json`, `candidates.csv`, `pareto.csv`, and `report.md`.

- [ ] **Step 3: Verify output files**

Run:

```bash
OUT_DIR=$(ls -dt analysis_outputs/trace_feature_audit_v2_* | head -n 1)
printf 'OUT_DIR=%s\n' "$OUT_DIR"
test -s "$OUT_DIR/summary.json"
test -s "$OUT_DIR/candidates.csv"
test -s "$OUT_DIR/pareto.csv"
test -s "$OUT_DIR/report.md"
/home/shx/miniconda3/envs/dllm_env/bin/python -m json.tool "$OUT_DIR/summary.json" >/tmp/trace_feature_audit_v2_summary_check.json
```

Expected: all commands exit `0`.

- [ ] **Step 4: Print compact terminal tables**

Run:

```bash
OUT_DIR=$(ls -dt analysis_outputs/trace_feature_audit_v2_* | head -n 1)
/home/shx/miniconda3/envs/dllm_env/bin/python -c "import csv, itertools, json, os; out=os.environ['OUT_DIR']; s=json.load(open(os.path.join(out,'summary.json'))); print('decision', s['decision']); print('sources', {k:v['decision'] for k,v in s['sources'].items()}); rows=list(csv.DictReader(open(os.path.join(out,'candidates.csv')))); print('top candidates'); [print(r['source'], r['family'], r['decision'], r['name'], r['trigger_count'], r['failed_long_count'], r['short_risk_count'], r['current_pass_risk_count'], r['true_long_precision']) for r in rows[:10]]"
```

Expected: terminal prints the final decision and top 10 candidates.

## Task 8: Update Paper-Agent Docs With Audit Result

**Files:**
- Modify: `docs/paper_agent/experiment_results.zh.md`
- Modify: `docs/paper_agent/experiment_results.en.md`
- Modify: `docs/paper_agent/paper_agent_dashboard.zh.md`
- Modify: `docs/paper_agent/paper_agent_dashboard.en.md`
- Modify: `docs/paper_agent/activity_ledger.zh.md`
- Modify: `docs/paper_agent/activity_ledger.en.md`
- Modify: `docs/paper_agent/pause_checkpoint.current.md`
- Modify: `docs/paper_agent/current_action.md`

- [ ] **Step 1: Generate compact result sections from `summary.json`**

Run:

```bash
OUT_DIR=$(ls -dt analysis_outputs/trace_feature_audit_v2_* | head -n 1)
OUT_DIR="$OUT_DIR" /home/shx/miniconda3/envs/dllm_env/bin/python - <<'PY'
import csv
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
candidates = list(csv.DictReader((out_dir / "candidates.csv").open("r", encoding="utf-8")))
top = candidates[:5]

def fmt_float(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"

def source_table():
    lines = [
        "| Source | Rows | True-long | Failed-long | Short | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for source in ["previous", "midcons"]:
        data = summary["sources"].get(source, {})
        lines.append(
            f"| {source} | {data.get('rows', 0)} | {data.get('true_long', 0)} | {data.get('failed_long', 0)} | {data.get('short', 0)} | {data.get('decision', 'reject')} |"
        )
    return "\n".join(lines)

def candidate_table():
    lines = [
        "| Source | Family | Decision | Candidate | Triggers | Failed-long | Short risk | Current-pass risk | True-long precision |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| {row.get('source', 'unknown')} | {row.get('family', 'unknown')} | {row.get('decision', 'reject')} | `{row.get('name', 'unknown')}` | {row.get('trigger_count', 0)} | {row.get('failed_long_count', 0)} | {row.get('short_risk_count', 0)} | {row.get('current_pass_risk_count', 0)} | {fmt_float(row.get('true_long_precision'))} |"
        )
    return "\n".join(lines)

decision = summary.get("decision", "reject")
interpretation_en = {
    "policy_candidate": "The audit found at least one readable candidate worth a separate Route 2 GPU policy-runner plan. This section is still diagnostic and does not report a new pass rate.",
    "diagnostic_only": "The audit found partial signal, but not enough evidence to launch a GPU policy runner without another design step.",
    "needs_new_data": "The audit indicates that the available single-canvas traces are insufficient for the CPU-only question.",
    "reject": "The audit did not find a stable low-risk candidate; do not default to Route 3 from this result alone.",
}.get(decision, "The audit produced an unrecognized decision and should be inspected before any GPU work.")
interpretation_zh = {
    "policy_candidate": "audit 找到了至少一个可读候选规则，值得单独写 Route 2 GPU policy-runner 计划。这里仍然只是 diagnostic，不是新 pass rate。",
    "diagnostic_only": "audit 找到了部分信号，但不足以在没有新设计的情况下启动 GPU policy runner。",
    "needs_new_data": "audit 表明现有 single-canvas traces 不足以回答 CPU-only 问题。",
    "reject": "audit 没有找到稳定低风险候选规则；不能因为这个结果直接转向 Route 3。",
}.get(decision, "audit 产生了未知 decision，启动任何 GPU 工作前必须人工检查。")

en = f"""## Trace Feature Audit V2

`trace_feature_audit_v2` is a CPU-only offline diagnostic. It does not report a new pass rate and did not launch GPU work. It tests whether richer trace-shape, stop-reason-conditioned, probe-trace fusion, and model-assisted discovery families can find a readable long-rescue gate.

{source_table()}

Top candidates:

{candidate_table()}

Interpretation: {interpretation_en}
"""

zh = f"""## Trace Feature Audit V2

`trace_feature_audit_v2` 是 CPU-only offline diagnostic。它不报告新的 pass rate，也没有启动 GPU 工作。它检查更丰富的 trace-shape、stop-reason-conditioned、probe-trace fusion 和 model-assisted discovery family 是否能找到可读的 long-rescue gate。

{source_table()}

Top candidates:

{candidate_table()}

Interpretation：{interpretation_zh}
"""

(out_dir / "docs_section.en.md").write_text(en, encoding="utf-8")
(out_dir / "docs_section.zh.md").write_text(zh, encoding="utf-8")
print(out_dir / "docs_section.en.md")
print(out_dir / "docs_section.zh.md")
PY
```

Expected: the command prints paths to `docs_section.en.md` and `docs_section.zh.md`.

- [ ] **Step 2: Insert generated sections into result docs**

Run:

```bash
OUT_DIR=$(ls -dt analysis_outputs/trace_feature_audit_v2_* | head -n 1)
OUT_DIR="$OUT_DIR" /home/shx/miniconda3/envs/dllm_env/bin/python - <<'PY'
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
targets = [
    (
        Path("docs/paper_agent/experiment_results.en.md"),
        out_dir / "docs_section.en.md",
        "Interpretation: no route met the offline continuation rule. This is diagnostic negative evidence; no route-specific GPU policy full run should be launched from these traces.",
    ),
    (
        Path("docs/paper_agent/experiment_results.zh.md"),
        out_dir / "docs_section.zh.md",
        "Interpretation：没有任何 route 满足 offline continuation rule。这是 diagnostic negative evidence；不应基于这批 traces 启动 route-specific GPU policy full run。",
    ),
]

for target, section_path, marker in targets:
    text = target.read_text(encoding="utf-8")
    section = section_path.read_text(encoding="utf-8").strip() + "\n\n"
    if "## Trace Feature Audit V2" in text:
        continue
    if marker not in text:
        raise SystemExit(f"marker not found in {target}")
    text = text.replace(marker + "\n\n", marker + "\n\n" + section, 1)
    target.write_text(text, encoding="utf-8")
PY
```

Expected: both result docs now contain `## Trace Feature Audit V2`.

- [ ] **Step 3: Update dashboard, checkpoint, and current action**

Run:

```bash
OUT_DIR=$(ls -dt analysis_outputs/trace_feature_audit_v2_* | head -n 1)
OUT_DIR="$OUT_DIR" /home/shx/miniconda3/envs/dllm_env/bin/python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
decision = summary.get("decision", "reject")
line_en = f"- Trace feature audit v2 completed as CPU-only diagnostic. Decision: `{decision}`. No GPU policy runner was launched; do not default to Route 3 unless a later design justifies it.\n"
line_zh = f"- Trace feature audit v2 已作为 CPU-only diagnostic 完成。Decision: `{decision}`。本动作未启动 GPU policy runner；除非后续新设计证明必要，否则不要默认转向 Route 3。\n"

append_targets = [
    (Path("docs/paper_agent/paper_agent_dashboard.en.md"), "## Latest Result Summary", line_en),
    (Path("docs/paper_agent/paper_agent_dashboard.zh.md"), "## Latest Result Summary", line_zh),
    (Path("docs/paper_agent/pause_checkpoint.current.md"), "## Latest Evidence And Results", line_en),
    (Path("docs/paper_agent/activity_ledger.en.md"), None, line_en),
    (Path("docs/paper_agent/activity_ledger.zh.md"), None, line_zh),
]
for path, marker, line in append_targets:
    text = path.read_text(encoding="utf-8")
    if "Trace feature audit v2" in text or "Trace feature audit v2 已" in text:
        continue
    if marker and marker in text:
        text = text.replace(marker + "\n", marker + "\n\n" + line, 1)
    else:
        text = text.rstrip() + "\n\n" + line
    path.write_text(text, encoding="utf-8")

current_action = f"""# Current Paper-Agent Action

Timestamp: 2026-06-13 CST

## Action Name

Close out CPU-only `trace_feature_audit_v2`.

## Current Phase

The audit has completed. See `{out_dir}` and updated experiment results.

## Decision

`{decision}`

## Next Required Step

If the decision is `policy_candidate`, write a new action brief and implementation plan for a Route 2 GPU policy runner. If the decision is `diagnostic_only`, `reject`, or `needs_new_data`, do not launch GPU work until a new design is approved.
"""
Path("docs/paper_agent/current_action.md").write_text(current_action, encoding="utf-8")
PY
```

Expected: dashboard, checkpoint, ledgers, and current action mention the audit decision.

- [ ] **Step 4: Run doc hygiene**

Run:

```bash
git diff --check -- docs/paper_agent/experiment_results.zh.md docs/paper_agent/experiment_results.en.md docs/paper_agent/paper_agent_dashboard.zh.md docs/paper_agent/paper_agent_dashboard.en.md docs/paper_agent/activity_ledger.zh.md docs/paper_agent/activity_ledger.en.md docs/paper_agent/pause_checkpoint.current.md docs/paper_agent/current_action.md
```

Expected: no output, exit code `0`.

- [ ] **Step 5: Commit Task 8**

Run:

```bash
git add docs/paper_agent/experiment_results.zh.md docs/paper_agent/experiment_results.en.md docs/paper_agent/paper_agent_dashboard.zh.md docs/paper_agent/paper_agent_dashboard.en.md docs/paper_agent/activity_ledger.zh.md docs/paper_agent/activity_ledger.en.md docs/paper_agent/pause_checkpoint.current.md docs/paper_agent/current_action.md analysis_outputs/trace_feature_audit_v2_*
git commit -m "docs: record trace feature audit v2 results"
```

Expected: commit succeeds.

## Task 9: Final Verification And Local Review

**Files:**
- Verify all files touched by this plan.

- [ ] **Step 1: Run focused tests**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_feature_audit_v2.py tests/test_trace_long_rescue_features.py tests/test_analyze_probe_curve_split_score.py
```

Expected: all tests pass.

- [ ] **Step 2: Run compile checks**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/trace_feature_audit_v2.py analysis/trace_long_rescue_features.py analysis/analyze_trace_long_rescue_routes.py analysis/print_trace_long_rescue_report.py
```

Expected: exit code `0`.

- [ ] **Step 3: Run git hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` exits `0`. `git status --short` is empty after all planned commits.

- [ ] **Step 4: Local review fallback**

Because this project disables subagents, do not use `superpowers:requesting-code-review` reviewer dispatch. Instead, run:

```bash
git show --stat --oneline HEAD
git show --stat --oneline HEAD~1
```

Expected: the final commits only include the planned audit script, tests, analysis outputs, and paper-agent docs.

## Self-Review Checklist

- Spec coverage: Tasks cover action brief, feature families, deterministic folds, model-assisted discovery, risk-controlled candidate ranking, reporting, docs, and no-GPU boundary.
- Placeholder scan: no placeholder markers, angle-bracket placeholders, or unspecified commands.
- Type consistency: tests and implementation consistently use `features`, `labels`, `CandidateRule`, `source`, `task_id`, and deterministic `fold_id`.
- Method boundary: oracle/pass labels are used only in `labels` for offline evaluation; policy feature dictionaries intentionally exclude oracle length and pass/fail.
- Route 3 discipline: this plan does not create or run multi-canvas audit. It only reports whether a trace-quality signal exists.
- Safety: no GPU command, no destructive command, no raw output overwrite.
