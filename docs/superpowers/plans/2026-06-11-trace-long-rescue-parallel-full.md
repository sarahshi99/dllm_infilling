# Trace Long Rescue Parallel Full Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` only, or execute this plan manually in the current session. Do not use `superpowers:subagent-driven-development`, `superpowers:dispatching-parallel-agents`, Task/Spawn subagents, reviewer subagents, parallel-agent dispatch, or `tool_search` discovery. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate three pure inference-time, training-free long-length recovery routes for LLaDA-Base, using full trace-enabled runs on GPUs `2/3`, then report paper-positioned results against literature reports, previous local methods, and the current method.

**Architecture:** Reuse existing LLaDA runners to collect full `step_traces.jsonl` for the previous local method and current `midcons` method. Add focused trace-analysis utilities that evaluate three route families: trace-only true-long detector, risk-controlled long rescue, and multi-canvas trace reranking. Only after offline trace gates are computed should GPU full variants be launched, each with separate output/log paths and paper-agent action briefs.

**Tech Stack:** Python, existing `expvision_dllm_clean` runners, HumanEval-SingleLineInfilling, `jq`, `tmux`, `/home/shx/miniconda3/envs/dllm_env/bin/python`, GPUs `2/3`.

---

## Fixed Research Decisions

- Method family: pure inference-time / training-free.
- Success gate A: oracle `>=17` bucket improves, overall pass count does not decrease, and oracle `<=8` short-bucket net loss is at most `2` tasks.
- Exploratory gate B: allow long-bucket improvement with small overall regression only as diagnostic evidence, not as main claim.
- Development backbone: `GSAI-ML/LLaDA-8B-Base`.
- Scope: LLaDA-Base full trace first, plus small cross-backbone offline sanity using already completed LLaDA-MoE, DreamCoder-Base, DiffuCoder-Base rows where fields permit.
- Signal family: generation trace / decode dynamics only. Do not use post-generation static code features or verifier outcomes as policy inputs.
- Routes to evaluate as three route families. Implementation and experiment supervision must remain serial; do not use parallel subagents:
  - Route 1: trace-only true-long detector.
  - Route 2: risk-controlled long rescue.
  - Route 3: multi-canvas trace reranking without verifier.

## File Structure

- Create `analysis/trace_long_rescue_features.py`: parse `results.jsonl` and `step_traces.jsonl`, join rows by `task_id`, extract trace features without using verifier output as input.
- Create `analysis/analyze_trace_long_rescue_routes.py`: compute route-1/2/3 offline trigger candidates and gate-A/B summaries from trace features and existing result labels.
- Create `analysis/print_trace_long_rescue_report.py`: read concrete full-trace output directories and offline analysis summaries, print terminal comparison tables, and write Chinese/English markdown snippets for paper-agent docs.
- Create `tests/test_trace_long_rescue_features.py`: unit tests for trace parsing, joining, feature extraction, and gate summaries.
- Create `docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md`: paper-agent action brief for full trace collection and route analysis.
- Modify `docs/paper_agent/current_action.md`: point the current action to the trace-long-rescue plan.
- Later, after offline gates: create route-specific runner(s), likely `clean_scripts/run_lcal_trace_long_rescue.py`, only if the offline analysis identifies a route worth a GPU full run.

## Task 1: Write Paper-Agent Action Brief Before Any Full Trace Run

**Files:**
- Create: `docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md`
- Modify: `docs/paper_agent/current_action.md`

- [x] **Step 1: Write the experiment brief**

Create `docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md` with this content:

```markdown
# Trace Long Rescue Full-Trace Plan

Timestamp: 2026-06-11 CST

## Action Name

Collect full LLaDA-Base generation traces and evaluate three pure inference-time long-length recovery routes.

## Current Phase And Plan Version

Post-backbone-matrix long-length recovery design under experiment plan `v3`.

## Reviewer Motivation

Current methods improve short/medium behavior but still under-select true-long infilling. A CCF-A reviewer will need evidence that any long-length recovery is not just a short-bucket tradeoff.

## Hypothesis

Generation trace dynamics can distinguish true-long under-selection from safe short/medium cases better than probe-curve scalar fields alone.

## Fixed Success Gate

Gate A: oracle `>=17` bucket improves, overall pass count does not decrease, and oracle `<=8` short-bucket net loss is at most `2` tasks.

Gate B: long-bucket gains with small overall regression are exploratory only.

## Routes

1. Trace-only true-long detector.
2. Risk-controlled long rescue.
3. Multi-canvas trace reranking without verifier.

## Baselines And Comparisons

- Literature reported values remain external reported numbers.
- Previous local method: A6000 control `787/1033 = 76.19%`.
- Current local method: `midcons` `795/1033 = 76.96%`.
- New trace policies must compare against both previous local method and current method.

## Full Trace Collection Commands

Previous local method trace run on GPU `2`:

```bash
tmux new-session -d -s trace_llada_base_prev_20260611 -c /home/shx/projects/dllm_infilling/git_workspace "script -q -e -c \"HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_cal_lite_lcas_v3.py --model-path GSAI-ML/LLaDA-8B-Base --probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24 --length-alpha 0.06 --lcas-policy lcas_v3b --save-step-traces --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_trace_llada_base_cal_lite_lcas_v3b_gpu2\" logs/paper_agent/20260611_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log"
```

Current method trace run on GPU `3`:

```bash
tmux new-session -d -s trace_llada_base_midcons_20260611 -c /home/shx/projects/dllm_infilling/git_workspace "script -q -e -c \"HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path GSAI-ML/LLaDA-8B-Base --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_trace_llada_base_midcons_gpu3 --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8 --save-step-traces\" logs/paper_agent/20260611_full_trace_llada_base_midcons_gpu3.log"
```

## Success Criteria

- Both logs end with `COMMAND_EXIT_CODE="0"`.
- Both outputs have `1033` valid `results.jsonl` rows.
- Both outputs have non-empty `step_traces.jsonl`.
- Trace analysis can report route 1/2/3 Gate A and Gate B summaries.

## Kill Criteria

- Import failure, OOM, verifier failure, malformed rows, missing `summary.json`, empty traces, or pass-rate collapse relative to previous local method/current method.

## Expected Documentation Outputs

- Updated `experiment_results.*.md`.
- Updated dashboard, checkpoint, and activity ledger.
- Terminal table: literature reported numbers / previous local method / current method / trace route candidates.
```

- [x] **Step 2: Update `docs/paper_agent/current_action.md`**

Replace the old closeout action with a short pointer to this trace plan:

```markdown
# Current Paper-Agent Action

Timestamp: 2026-06-11 CST

## Action Name

Prepare full LLaDA-Base trace collection and parallel trace-long-rescue route analysis.

## Current Phase

Post-backbone-matrix long-length recovery. The approved direction is pure inference-time / training-free, success gate A, LLaDA-Base main development, and trace/dynamics signals only.

## Next Required Step

Follow `docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md`. Do not launch GPU runs until the exact commands, log paths, output paths, success criteria, and kill criteria have been printed in terminal.
```

- [x] **Step 3: Verify markdown hygiene**

Run:

```bash
git diff --check -- docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md docs/paper_agent/current_action.md
```

Expected: no output, exit `0`.

## Task 2: Implement Trace Feature Extraction

**Files:**
- Create: `analysis/trace_long_rescue_features.py`
- Create: `tests/test_trace_long_rescue_features.py`

- [x] **Step 1: Write failing tests for trace parsing and feature extraction**

Create `tests/test_trace_long_rescue_features.py`:

```python
import unittest

from analysis.trace_long_rescue_features import (
    compute_gate_summary,
    extract_trace_features,
    join_results_and_traces,
)


class TraceLongRescueFeaturesTest(unittest.TestCase):
    def test_join_results_and_traces_groups_by_task_id(self):
        rows = [
            {"task_id": "a", "metrics": {"passed": False, "oracle_mask_length": 18}},
            {"task_id": "b", "metrics": {"passed": True, "oracle_mask_length": 6}},
        ]
        traces = [
            {"task_id": "a", "step": 0, "remaining_masks_after_update": 18, "stop_decision": {"remaining_mask_ratio": 1.0}},
            {"task_id": "a", "step": 1, "remaining_masks_after_update": 12, "stop_decision": {"remaining_mask_ratio": 0.67}},
            {"task_id": "b", "step": 0, "remaining_masks_after_update": 2, "stop_decision": {"remaining_mask_ratio": 0.33}},
        ]

        joined = join_results_and_traces(rows, traces)

        self.assertEqual(len(joined), 2)
        self.assertEqual([item["step"] for item in joined["a"]["traces"]], [0, 1])
        self.assertEqual([item["step"] for item in joined["b"]["traces"]], [0])

    def test_extract_trace_features_uses_decode_dynamics(self):
        row = {
            "task_id": "a",
            "metrics": {
                "passed": False,
                "oracle_mask_length": 18,
                "selected_mask_length": 8,
                "remaining_masks_at_stop_or_final": 6,
                "remaining_mask_ratio_at_stop_or_final": 0.75,
                "mean_gap_at_stop_or_final": 0.12,
                "mean_top1_at_stop_or_final": 0.40,
            },
        }
        traces = [
            {"step": 0, "remaining_masks_before_update": 8, "remaining_masks_after_update": 8, "mean_confidence": 0.20, "stop_decision": {"reason": "too_many_remaining_masks"}},
            {"step": 1, "remaining_masks_before_update": 8, "remaining_masks_after_update": 6, "mean_confidence": 0.30, "stop_decision": {"reason": "mean_gap_below_threshold"}},
        ]

        features = extract_trace_features(row, traces)

        self.assertEqual(features["task_id"], "a")
        self.assertTrue(features["failed_long"])
        self.assertEqual(features["oracle_bucket"], "17-24")
        self.assertEqual(features["final_remaining_masks"], 6)
        self.assertAlmostEqual(features["final_remaining_mask_ratio"], 0.75)
        self.assertAlmostEqual(features["max_remaining_mask_ratio"], 1.0)
        self.assertEqual(features["last_stop_reason"], "mean_gap_below_threshold")

    def test_compute_gate_summary_counts_short_losses_and_long_gains(self):
        base_rows = {
            "short_loss": {"metrics": {"passed": True, "oracle_mask_length": 5}},
            "long_gain": {"metrics": {"passed": False, "oracle_mask_length": 20}},
            "neutral": {"metrics": {"passed": True, "oracle_mask_length": 10}},
        }
        candidate_passed = {
            "short_loss": False,
            "long_gain": True,
            "neutral": True,
        }

        summary = compute_gate_summary(base_rows, candidate_passed)

        self.assertEqual(summary["overall_delta"], 0)
        self.assertEqual(summary["short_net_loss"], 1)
        self.assertEqual(summary["long_net_gain"], 1)
        self.assertTrue(summary["gate_a_passed"])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_long_rescue_features.py
```

Expected: import failure because `analysis.trace_long_rescue_features` does not exist.

- [x] **Step 3: Implement trace feature extraction**

Create `analysis/trace_long_rescue_features.py`:

```python
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    metrics = row.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        return default
    return metrics.get(key, default)


def oracle_bucket(oracle_len: Optional[int]) -> str:
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


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def join_results_and_traces(
    result_rows: Iterable[Mapping[str, Any]],
    trace_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    joined: Dict[str, Dict[str, Any]] = {}
    for row in result_rows:
        task_id = str(row.get("task_id"))
        joined[task_id] = {"row": dict(row), "traces": []}

    grouped: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for trace in trace_rows:
        grouped[str(trace.get("task_id"))].append(dict(trace))

    for task_id, traces in grouped.items():
        if task_id not in joined:
            continue
        joined[task_id]["traces"] = sorted(traces, key=lambda item: int(item.get("step", -1)))
    return joined


def _trace_remaining_ratio(trace: Mapping[str, Any]) -> Optional[float]:
    decision = trace.get("stop_decision") or {}
    if isinstance(decision, Mapping):
        ratio = _optional_float(decision.get("remaining_mask_ratio"))
        if ratio is not None:
            return ratio
    before = _optional_float(trace.get("remaining_masks_before_update"))
    after = _optional_float(trace.get("remaining_masks_after_update"))
    if before and before > 0 and after is not None:
        return after / before
    return None


def extract_trace_features(row: Mapping[str, Any], traces: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    traces = list(traces)
    oracle_len = _optional_int(metric(row, "oracle_mask_length"))
    selected_len = _optional_int(metric(row, "selected_mask_length"))
    passed = bool(metric(row, "passed", False))
    remaining_ratios = [ratio for ratio in (_trace_remaining_ratio(trace) for trace in traces) if ratio is not None]
    confidences = [
        confidence
        for confidence in (_optional_float(trace.get("mean_confidence")) for trace in traces)
        if confidence is not None
    ]
    last_trace = traces[-1] if traces else {}
    last_decision = last_trace.get("stop_decision") or {}

    return {
        "task_id": row.get("task_id"),
        "passed": passed,
        "oracle_len": oracle_len,
        "oracle_bucket": oracle_bucket(oracle_len),
        "selected_len": selected_len,
        "selected_minus_oracle": None
        if selected_len is None or oracle_len is None
        else int(selected_len) - int(oracle_len),
        "failed_long": bool((oracle_len is not None and oracle_len >= 17) and not passed),
        "short": bool(oracle_len is not None and oracle_len <= 8),
        "trace_steps": len(traces),
        "final_remaining_masks": _optional_int(metric(row, "remaining_masks_at_stop_or_final")),
        "final_remaining_mask_ratio": _optional_float(metric(row, "remaining_mask_ratio_at_stop_or_final")),
        "final_mean_gap": _optional_float(metric(row, "mean_gap_at_stop_or_final")),
        "final_mean_top1": _optional_float(metric(row, "mean_top1_at_stop_or_final")),
        "max_remaining_mask_ratio": max(remaining_ratios) if remaining_ratios else None,
        "last_remaining_mask_ratio": remaining_ratios[-1] if remaining_ratios else None,
        "min_mean_confidence": min(confidences) if confidences else None,
        "last_mean_confidence": confidences[-1] if confidences else None,
        "last_stop_reason": str(last_decision.get("reason")) if isinstance(last_decision, Mapping) else None,
    }


def compute_gate_summary(
    base_rows: Mapping[str, Mapping[str, Any]],
    candidate_passed: Mapping[str, bool],
) -> Dict[str, Any]:
    common_task_ids = sorted(set(base_rows).intersection(candidate_passed))
    overall_delta = 0
    short_net_loss = 0
    long_net_gain = 0

    for task_id in common_task_ids:
        row = base_rows[task_id]
        before = bool(metric(row, "passed", False))
        after = bool(candidate_passed[task_id])
        if before == after:
            continue
        delta = 1 if after else -1
        overall_delta += delta
        oracle_len = _optional_int(metric(row, "oracle_mask_length"))
        if oracle_len is not None and oracle_len <= 8 and delta < 0:
            short_net_loss += 1
        if oracle_len is not None and oracle_len >= 17 and delta > 0:
            long_net_gain += 1

    return {
        "common": len(common_task_ids),
        "overall_delta": overall_delta,
        "short_net_loss": short_net_loss,
        "long_net_gain": long_net_gain,
        "gate_a_passed": bool(long_net_gain > 0 and overall_delta >= 0 and short_net_loss <= 2),
        "gate_b_exploratory": bool(long_net_gain > 0),
    }
```

- [x] **Step 4: Run tests and verify they pass**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_long_rescue_features.py
```

Expected: `Ran 3 tests` and `OK`.

- [x] **Step 5: Commit this task only if requested**

Do not stage unrelated dirty files. If committing is requested, stage only:

```bash
git add analysis/trace_long_rescue_features.py tests/test_trace_long_rescue_features.py
git commit -m "analysis: add trace long rescue feature extraction"
```

## Task 3: Implement Offline Route Analyzer

**Files:**
- Create: `analysis/analyze_trace_long_rescue_routes.py`
- Create: `analysis/print_trace_long_rescue_report.py`
- Modify: `tests/test_trace_long_rescue_features.py`

- [x] **Step 1: Add failing tests for route scoring**

Append to `tests/test_trace_long_rescue_features.py`:

```python
from analysis.analyze_trace_long_rescue_routes import (
    route1_trace_only_trigger,
    route2_risk_controlled_trigger,
    route3_rerank_choice,
)


class TraceLongRescueRoutesTest(unittest.TestCase):
    def test_route1_trace_only_trigger_requires_underselection_and_unstable_trace(self):
        features = {
            "selected_minus_oracle": -10,
            "final_remaining_mask_ratio": 0.75,
            "final_mean_gap": 0.10,
            "short": False,
        }
        self.assertTrue(route1_trace_only_trigger(features))

        features["short"] = True
        self.assertFalse(route1_trace_only_trigger(features))

    def test_route2_is_more_conservative_than_route1(self):
        features = {
            "selected_minus_oracle": -10,
            "final_remaining_mask_ratio": 0.75,
            "final_mean_gap": 0.10,
            "last_mean_confidence": 0.30,
            "short": False,
            "selected_len": 8,
        }
        self.assertTrue(route2_risk_controlled_trigger(features))

        features["selected_len"] = 13
        self.assertFalse(route2_risk_controlled_trigger(features))

    def test_route3_picks_highest_trace_quality_without_verifier(self):
        candidates = [
            {"length": 8, "trace_quality": 0.20, "length_penalty": 0.00},
            {"length": 20, "trace_quality": 0.55, "length_penalty": 0.05},
            {"length": 32, "trace_quality": 0.56, "length_penalty": 0.30},
        ]
        self.assertEqual(route3_rerank_choice(candidates)["length"], 20)
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_long_rescue_features.py
```

Expected: import failure or missing function failure.

- [x] **Step 3: Implement offline route analyzer**

Create `analysis/analyze_trace_long_rescue_routes.py`:

```python
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from analysis.trace_long_rescue_features import (
    extract_trace_features,
    join_results_and_traces,
    load_jsonl,
)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def route1_trace_only_trigger(features: Mapping[str, Any]) -> bool:
    if bool(features.get("short")):
        return False
    return (
        _num(features.get("selected_minus_oracle")) <= -6
        and _num(features.get("final_remaining_mask_ratio")) >= 0.50
        and _num(features.get("final_mean_gap"), 1.0) <= 0.25
    )


def route2_risk_controlled_trigger(features: Mapping[str, Any]) -> bool:
    if not route1_trace_only_trigger(features):
        return False
    return (
        _num(features.get("selected_len")) <= 12
        and _num(features.get("last_mean_confidence"), 1.0) <= 0.45
    )


def route3_rerank_choice(candidates: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    candidates = [dict(candidate) for candidate in candidates]
    if not candidates:
        raise ValueError("route3_rerank_choice requires at least one candidate")
    return max(
        candidates,
        key=lambda item: (
            _num(item.get("trace_quality")) - _num(item.get("length_penalty")),
            -_num(item.get("length")),
        ),
    )


def summarize_triggers(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    route1 = [item for item in features if route1_trace_only_trigger(item)]
    route2 = [item for item in features if route2_risk_controlled_trigger(item)]
    return {
        "rows": len(features),
        "route1_trigger_count": len(route1),
        "route1_failed_long_count": sum(1 for item in route1 if item.get("failed_long")),
        "route1_short_count": sum(1 for item in route1 if item.get("short")),
        "route2_trigger_count": len(route2),
        "route2_failed_long_count": sum(1 for item in route2 if item.get("failed_long")),
        "route2_short_count": sum(1 for item in route2 if item.get("short")),
    }


def write_markdown(path: str, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Trace Long Rescue Route Analysis",
        "",
        "Generated from full trace outputs. Route summaries are offline diagnostics, not GPU policy results.",
        "",
        "| Route | Trigger count | Failed-long triggers | Short triggers |",
        "|---|---:|---:|---:|",
        f"| Route 1 trace-only | `{summary['route1_trigger_count']}` | `{summary['route1_failed_long_count']}` | `{summary['route1_short_count']}` |",
        f"| Route 2 risk-controlled | `{summary['route2_trigger_count']}` | `{summary['route2_failed_long_count']}` | `{summary['route2_short_count']}` |",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    traces = load_jsonl(args.traces)
    joined = join_results_and_traces(rows, traces)
    features = [
        extract_trace_features(payload["row"], payload["traces"])
        for payload in joined.values()
    ]
    summary = summarize_triggers(features)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "features.jsonl"), "w", encoding="utf-8") as handle:
        for feature in features:
            handle.write(json.dumps(feature, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_markdown(os.path.join(args.output_dir, "summary.md"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run tests and verify they pass**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_long_rescue_features.py
```

Expected: `Ran 6 tests` and `OK`.

- [x] **Step 5: Implement terminal/doc report renderer**

Create `analysis/print_trace_long_rescue_report.py`:

```python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def pass_rate(run_dir: Path) -> Tuple[int, int, str]:
    rows = load_jsonl(run_dir / "results.jsonl")
    passed = sum(1 for row in rows if bool((row.get("metrics") or {}).get("passed", False)))
    total = len(rows)
    rate = 0.0 if total == 0 else passed * 100.0 / total
    return passed, total, f"{passed}/{total} = {rate:.2f}%"


def route_decision(route_name: str, failed_long: int, short: int) -> str:
    if route_name == "Route 1 trace-only":
        return "continue" if failed_long >= 10 and short <= 5 else "stop"
    if route_name == "Route 2 risk-controlled":
        return "continue" if failed_long >= 5 and short <= 2 else "stop"
    return "pending"


def route_rows(label: str, summary: Mapping[str, Any]) -> List[Tuple[str, str, int, int, int, str]]:
    rows: List[Tuple[str, str, int, int, int, str]] = []
    route_specs = [
        (
            "Route 1 trace-only",
            int(summary.get("route1_trigger_count", 0)),
            int(summary.get("route1_failed_long_count", 0)),
            int(summary.get("route1_short_count", 0)),
        ),
        (
            "Route 2 risk-controlled",
            int(summary.get("route2_trigger_count", 0)),
            int(summary.get("route2_failed_long_count", 0)),
            int(summary.get("route2_short_count", 0)),
        ),
        ("Route 3 multi-canvas rerank", 0, 0, 0),
    ]
    for route_name, trigger_count, failed_long_count, short_count in route_specs:
        rows.append(
            (
                label,
                route_name,
                trigger_count,
                failed_long_count,
                short_count,
                route_decision(route_name, failed_long_count, short_count),
            )
        )
    return rows


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    headers = list(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_sections(
    prev_run: Path,
    midcons_run: Path,
    prev_summary: Mapping[str, Any],
    midcons_summary: Mapping[str, Any],
) -> Tuple[str, str, str]:
    prev_passed, prev_total, prev_rate = pass_rate(prev_run)
    mid_passed, mid_total, mid_rate = pass_rate(midcons_run)
    run_rows = [
        ("previous local method trace", prev_run, prev_total, count_lines(prev_run / "step_traces.jsonl"), prev_rate),
        ("current midcons trace", midcons_run, mid_total, count_lines(midcons_run / "step_traces.jsonl"), mid_rate),
    ]
    route_table_rows = route_rows("previous local method", prev_summary) + route_rows("current midcons", midcons_summary)

    terminal = "\n".join(
        [
            "TRACE LONG RESCUE FULL DIAGNOSTICS",
            markdown_table(["Run", "Output", "Rows", "Trace rows", "Pass rate"], run_rows),
            "",
            markdown_table(
                ["Trace source", "Route", "Trigger count", "Failed-long triggers", "Short triggers", "Decision"],
                route_table_rows,
            ),
        ]
    )
    zh = "\n".join(
        [
            "## LLaDA-Base Full Trace Long-Rescue Diagnostics",
            "",
            "Full trace collection completed for the previous local method and the current `midcons` method. These diagnostics are trace-signal evidence only; they are not a new SOTA claim.",
            "",
            markdown_table(["Run", "Output", "Results rows", "Trace rows", "Pass rate"], run_rows),
            "",
            "Offline route analysis:",
            "",
            markdown_table(
                ["Trace source", "Route", "Trigger count", "Failed-long triggers", "Short triggers", "Decision"],
                route_table_rows,
            ),
            "",
        ]
    )
    en = zh
    return terminal, zh, en


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev-run", required=True)
    parser.add_argument("--midcons-run", required=True)
    parser.add_argument("--prev-analysis", required=True)
    parser.add_argument("--midcons-analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    prev_run = Path(args.prev_run)
    midcons_run = Path(args.midcons_run)
    prev_summary = load_json(Path(args.prev_analysis) / "summary.json")
    midcons_summary = load_json(Path(args.midcons_analysis) / "summary.json")
    terminal, zh, en = render_sections(prev_run, midcons_run, prev_summary, midcons_summary)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    (output_dir / "docs_section.zh.md").write_text(zh, encoding="utf-8")
    (output_dir / "docs_section.en.md").write_text(en, encoding="utf-8")
    print(terminal)


if __name__ == "__main__":
    main()
```

- [x] **Step 6: Compile scripts**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/trace_long_rescue_features.py analysis/analyze_trace_long_rescue_routes.py analysis/print_trace_long_rescue_report.py
```

Expected: exit `0`.

## Task 4: Launch Full Trace Collection On GPUs 2/3

**Files:**
- Use existing: `clean_scripts/run_cal_lite_lcas_v3.py`
- Use existing: `clean_scripts/run_lcal_official_bounded_repair.py`
- Logs: `logs/paper_agent/20260611_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log`, `logs/paper_agent/20260611_full_trace_llada_base_midcons_gpu3.log`
- Outputs: `outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_*`, `outputs_clean/full_trace_llada_base_midcons_gpu3_*`

- [x] **Step 1: Print the experiment design in terminal**

Run:

```bash
printf '%s\n' \
'EXPERIMENT DESIGN: full LLaDA-Base trace collection' \
'Model: GSAI-ML/LLaDA-8B-Base' \
'Dataset: HumanEval-SingleLineInfilling test split, 1033 tasks' \
'GPU: previous local method on GPU 2, current midcons on GPU 3' \
'Metric: pass rate, oracle bucket pass, pairwise, trace route trigger quality' \
'Success: logs exit 0, 1033 rows each, non-empty step_traces.jsonl' \
'Kill: import/OOM/verifier failure, malformed rows, missing summaries, empty traces'
```

- [x] **Step 2: Check GPUs**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

Expected: GPUs `2` and `3` have enough free memory and are not occupied by unrelated heavy jobs.

- [x] **Step 3: Start previous local method full trace run on GPU 2**

Run:

```bash
tmux new-session -d -s trace_llada_base_prev_20260611 -c /home/shx/projects/dllm_infilling/git_workspace "script -q -e -c \"HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_cal_lite_lcas_v3.py --model-path GSAI-ML/LLaDA-8B-Base --probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24 --length-alpha 0.06 --lcas-policy lcas_v3b --save-step-traces --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_trace_llada_base_cal_lite_lcas_v3b_gpu2\" logs/paper_agent/20260611_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log"
```

- [x] **Step 4: Start current method full trace run on GPU 3**

Run:

```bash
tmux new-session -d -s trace_llada_base_midcons_20260611 -c /home/shx/projects/dllm_infilling/git_workspace "script -q -e -c \"HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path GSAI-ML/LLaDA-8B-Base --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_trace_llada_base_midcons_gpu3 --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8 --save-step-traces\" logs/paper_agent/20260611_full_trace_llada_base_midcons_gpu3.log"
```

- [x] **Step 5: Monitor until both runs finish**

Run:

```bash
tmux ls
tail -5 logs/paper_agent/20260611_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log
tail -5 logs/paper_agent/20260611_full_trace_llada_base_midcons_gpu3.log
```

Expected: both logs eventually end with `COMMAND_EXIT_CODE="0"`.

- [x] **Step 6: Verify row counts and trace counts**

Run:

```bash
PREV_DIR=$(ls -dt /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_* | head -n 1)
MIDCONS_DIR=$(ls -dt /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_* | head -n 1)
printf 'PREV_DIR=%s\nMIDCONS_DIR=%s\n' "$PREV_DIR" "$MIDCONS_DIR"
wc -l "$PREV_DIR/results.jsonl" "$MIDCONS_DIR/results.jsonl"
wc -l "$PREV_DIR/step_traces.jsonl" "$MIDCONS_DIR/step_traces.jsonl"
jq -s 'length' "$PREV_DIR/results.jsonl"
jq -s 'length' "$MIDCONS_DIR/results.jsonl"
```

Expected: `PREV_DIR` and `MIDCONS_DIR` print concrete output directories, both `results.jsonl` files report `1033` rows, and both trace files have more than `1033` rows.

## Task 5: Run Offline Route Analysis On Full Trace Outputs

**Files:**
- Use: `analysis/analyze_trace_long_rescue_routes.py`
- Create output directories under `analysis_outputs/trace_long_rescue_llada_base_prev_YYYYMMDD_HHMMSS/` and `analysis_outputs/trace_long_rescue_llada_base_midcons_YYYYMMDD_HHMMSS/`.

- [x] **Step 1: Analyze previous local method traces**

Run:

```bash
ANALYSIS_STAMP=$(date +%Y%m%d_%H%M%S)
PREV_DIR=$(ls -dt /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_* | head -n 1)
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_trace_long_rescue_routes.py \
  --results "$PREV_DIR/results.jsonl" \
  --traces "$PREV_DIR/step_traces.jsonl" \
  --output-dir "analysis_outputs/trace_long_rescue_llada_base_prev_${ANALYSIS_STAMP}"
```

Expected: printed JSON with route trigger counts and written `summary.json`, `summary.md`, `features.jsonl`.

- [x] **Step 2: Analyze current method traces**

Run:

```bash
ANALYSIS_STAMP=$(date +%Y%m%d_%H%M%S)
MIDCONS_DIR=$(ls -dt /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_* | head -n 1)
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_trace_long_rescue_routes.py \
  --results "$MIDCONS_DIR/results.jsonl" \
  --traces "$MIDCONS_DIR/step_traces.jsonl" \
  --output-dir "analysis_outputs/trace_long_rescue_llada_base_midcons_${ANALYSIS_STAMP}"
```

Expected: printed JSON with route trigger counts and written `summary.json`, `summary.md`, `features.jsonl`.

- [x] **Step 3: Decide which route deserves GPU policy implementation**

Use this rule:

```text
Route 1: continue if it triggers at least 10 failed-long rows and at most 5 short rows.
Route 2: continue if it triggers at least 5 failed-long rows and at most 2 short rows.
Route 3: continue only if route 1 or route 2 show enough trace signal to justify extra multi-canvas cost.
```

Expected: one of:

- route 1/2/3 has enough signal for a GPU policy runner;
- no route has enough signal, in which case document the negative result and do not launch a policy full run.

## Task 6: Update Paper-Agent Results And Dashboard

**Files:**
- Modify: `docs/paper_agent/experiment_results.zh.md`
- Modify: `docs/paper_agent/experiment_results.en.md`
- Modify: `docs/paper_agent/paper_agent_dashboard.zh.md`
- Modify: `docs/paper_agent/paper_agent_dashboard.en.md`
- Modify: `docs/paper_agent/activity_ledger.zh.md`
- Modify: `docs/paper_agent/activity_ledger.en.md`
- Modify: `docs/paper_agent/pause_checkpoint.current.md`

- [x] **Step 1: Print and save the trace diagnostic report**

Run:

```bash
PREV_DIR=$(ls -dt /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_* | head -n 1)
MIDCONS_DIR=$(ls -dt /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_* | head -n 1)
PREV_ANALYSIS=$(ls -dt analysis_outputs/trace_long_rescue_llada_base_prev_* | head -n 1)
MIDCONS_ANALYSIS=$(ls -dt analysis_outputs/trace_long_rescue_llada_base_midcons_* | head -n 1)
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/print_trace_long_rescue_report.py \
  --prev-run "$PREV_DIR" \
  --midcons-run "$MIDCONS_DIR" \
  --prev-analysis "$PREV_ANALYSIS" \
  --midcons-analysis "$MIDCONS_ANALYSIS" \
  --output-dir analysis_outputs/trace_long_rescue_report_20260611
```

Expected: terminal prints a table with run directory, row count, trace count, pass rate, route trigger count, failed-long trigger count, short trigger count, and continue/stop decision. The command writes:

- `analysis_outputs/trace_long_rescue_report_20260611/docs_section.zh.md`
- `analysis_outputs/trace_long_rescue_report_20260611/docs_section.en.md`

- [x] **Step 2: Insert generated report sections into result docs**

Open `analysis_outputs/trace_long_rescue_report_20260611/docs_section.zh.md` and insert its complete contents into `docs/paper_agent/experiment_results.zh.md` under the cross-backbone comparison table. Open `analysis_outputs/trace_long_rescue_report_20260611/docs_section.en.md` and insert its complete contents into `docs/paper_agent/experiment_results.en.md` under the corresponding English comparison table.

- [x] **Step 3: Update dashboard and activity ledger**

Add a compact entry with:

```text
Trace full runs completed or blocked; route-1/2/3 offline decision recorded; no SOTA claim from trace diagnostics alone.
```

- [x] **Step 4: Run doc hygiene**

Run:

```bash
git diff --check -- docs/paper_agent/experiment_results.zh.md docs/paper_agent/experiment_results.en.md docs/paper_agent/paper_agent_dashboard.zh.md docs/paper_agent/paper_agent_dashboard.en.md docs/paper_agent/activity_ledger.zh.md docs/paper_agent/activity_ledger.en.md docs/paper_agent/pause_checkpoint.current.md
```

Expected: no output, exit `0`.

## Task 7: Plan GPU Policy Runner Only After Offline Gate

**Files:**
- Create later only if warranted: `clean_scripts/run_lcal_trace_long_rescue.py`
- Create later only if warranted: `tests/test_trace_long_rescue_policy.py`

- [x] **Step 1: If no route passes offline continuation rule, stop**

Write in docs:

```text
No route met the offline continuation rule. The trace features are diagnostic negative evidence, and no route-specific GPU full run should be launched yet.
```

- [ ] **Step 2: If route 1 or 2 passes, write a new action brief before code**

Create:

```text
docs/paper_agent/experiments/20260611_trace_long_rescue_policy_runner.md
```

Include exact route, code files, command, GPU set, baseline, success gate A, kill criteria, output/log path.

- [ ] **Step 3: Use TDD before policy runner implementation**

Before production code, invoke `superpowers:test-driven-development` and write failing tests for:

```python
def test_trace_policy_does_not_trigger_on_short_confident_case():
    ...

def test_trace_policy_triggers_on_underselected_long_unstable_case():
    ...

def test_policy_summary_reports_gate_a():
    ...
```

- [ ] **Step 4: Launch route-specific full run only after tests pass**

Use `tmux`, GPU `2/3`, unique experiment name, and separate logs. Do not overwrite trace collection outputs.

## Self-Review Checklist

- Spec coverage: covers the user's chosen pure inference-time/training-free boundary, gate A, exploratory gate B, LLaDA-Base development, small cross-backbone offline sanity, B trace signals, and the three parallel route families.
- Placeholder scan: no `TBD` and no angle-bracket placeholders. Runtime directories are resolved with concrete shell variables from output-directory globs.
- Type consistency: tests and implementation use `task_id`, `metrics`, `oracle_mask_length`, `selected_mask_length`, and `step_traces.jsonl` fields already present in existing runners.
- Scope check: the plan separates trace collection, offline analysis, and route-specific GPU policy implementation so a failed trace signal does not force unnecessary runner work.
- Safety: all GPU commands use unique experiment names and logs, GPUs `2/3`, and no destructive commands.
