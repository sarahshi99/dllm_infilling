# Long Underestimate Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only offline sweep that identifies candidate long-underestimation triggers before spending GPU on a new long-tail policy.

**Architecture:** Add one analysis script that loads `results.jsonl`, sweeps gate combinations, computes oracle-labeled diagnostic rates, and writes compact JSON/CSV/Markdown artifacts. Keep it independent of model loading and existing runners.

**Tech Stack:** Python standard library, existing JSONL result format, outputs under `analysis_outputs/long_underestimate_detector/`, direct `unittest` tests.

---

### Task 1: Diagnostic Sweep Script

**Files:**
- Create: `analysis/diagnose_long_underestimate_policy.py`
- Test: `tests/test_diagnose_long_underestimate_policy.py`

- [x] **Step 1: Write tests for rule evaluation**

Test a tiny set of synthetic rows where one row is a failed long under-selection, one is a passing short row, and one is a passing medium row. Verify trigger count, true-long precision, short-risk rate, and failed-long recall.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. /home/shx/miniconda3/envs/dllm_env/bin/python tests/test_diagnose_long_underestimate_policy.py
```

Expected before implementation: import failure for `analysis.diagnose_long_underestimate_policy`.

- [x] **Step 3: Implement minimal analysis script**

Implement:

- `load_jsonl(path)`
- `Rule` dataclass
- `row_triggers(row, rule)`
- `evaluate_rule(rows, rule)`
- `sweep_rules(rows)`
- CLI writing `sweep.json`, `sweep.csv`, and `sweep.md`

- [x] **Step 4: Verify tests and compile**

Run:

```bash
PYTHONPATH=. /home/shx/miniconda3/envs/dllm_env/bin/python tests/test_diagnose_long_underestimate_policy.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/diagnose_long_underestimate_policy.py
```

- [x] **Step 5: Run on A6000 midcons**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/diagnose_long_underestimate_policy.py \
  --results outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626/results.jsonl \
  --output-dir analysis_outputs/long_underestimate_detector/a6000_midcons
```

- [x] **Step 6: Interpret**

Read `analysis_outputs/long_underestimate_detector/a6000_midcons/sweep.md`. If a rule meets the success criteria in the design spec, create a GPU runner plan. If not, document that heuristic long rescue is not currently promising.

Outcome: no rule passed the strict success criteria. The best rule had `35.48%` true-long precision and `40.86%` short-risk rate, so no GPU runner should be launched from this heuristic family.
