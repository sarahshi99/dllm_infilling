# A6000 Midcons Longrescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible A6000 experiment path that reruns union control and midcons, then evaluates mid-rescue precision and true-long rescue candidates.

**Architecture:** Keep the existing LCAL official bounded repair runner as the central execution path. Add small A6000 run scripts, focused decision-policy arguments, and compact pairwise analysis utilities so every candidate is compared against the same A6000 control.

**Tech Stack:** Python 3, bash, JSONL results, existing `expvision_dllm_clean` package, existing `clean_scripts` runners.

---

## File Structure

- Create `clean_scripts/run_lcal_a6000_baselines.sh`: runs A6000 union control and A6000 midcons using GPUs `0,1`.
- Create `clean_scripts/run_lcal_a6000_candidates.sh`: runs candidate policies after baseline outputs exist.
- Modify `clean_scripts/run_lcal_official_bounded_repair.py`: add optional mid-rescue precision guards and true-long rescue flags.
- Create `analysis/analyze_lcal_pairwise.py`: compares two `results.jsonl` files by pass/fail, oracle bucket, selected length transition, and final source.
- Create `analysis_outputs/a6000_midcons_longrescue/README.md`: documents commands, output directories, and comparison order.

## Task 1: Add A6000 Baseline Run Script

**Files:**
- Create: `clean_scripts/run_lcal_a6000_baselines.sh`

- [ ] **Step 1: Create the baseline script**

Add this file:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/a6000-midcons-longrescue

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

PYTHON=/home/shx/miniconda3/envs/dllm_env/bin/python
RUNNER=clean_scripts/run_lcal_official_bounded_repair.py
BASELINE=outputs_clean/202604/full_lcas_v3b_alpha006_compact_sl_20260429_180958/results.jsonl

COMMON_ARGS=(
  --baseline-results "$BASELINE"
  --repair-max-s3-len 5
  --repair-min-official-len 6
  --repair-max-official-len 9
  --repair-min-delta 1
  --repair-max-delta 8
  --suspicion-max-s3-len 5
  --suspicion-min-official-len 16
  --suspicion-max-official-len 64
  --suspicion-min-delta 1
)

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control \
  "${COMMON_ARGS[@]}"

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000 \
  "${COMMON_ARGS[@]}" \
  --official-eval-max-s3-len 12 \
  --mid-rescue-max-s3-len 12 \
  --mid-rescue-source base \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 13 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 7 \
  --mid-rescue-min-long-ratio 0.8
```

- [ ] **Step 2: Make the script executable**

Run:

```bash
chmod +x clean_scripts/run_lcal_a6000_baselines.sh
```

Expected: command exits with status `0`.

- [ ] **Step 3: Syntax-check Python runner and bash script**

Run:

```bash
bash -n clean_scripts/run_lcal_a6000_baselines.sh
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_lcal_official_bounded_repair.py
```

Expected: both commands exit with status `0`.

- [ ] **Step 4: Commit**

Run:

```bash
git add clean_scripts/run_lcal_a6000_baselines.sh
git commit -m "Add A6000 baseline run script"
```

## Task 2: Add Pairwise LCAL Analysis Utility

**Files:**
- Create: `analysis/analyze_lcal_pairwise.py`

- [ ] **Step 1: Create the utility**

Implement a script with these functions:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_task(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["task_id"]): row for row in rows}


def metric(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    return row.get("metrics", {}).get(key, default)


def passed(row: Dict[str, Any]) -> bool:
    return bool(metric(row, "passed", False))


def oracle_len(row: Dict[str, Any]) -> Optional[int]:
    value = metric(row, "oracle_mask_length")
    return None if value is None else int(value)


def bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    if length <= 8:
        return "<=8"
    if length <= 12:
        return "9-12"
    if length <= 16:
        return "13-16"
    if length <= 24:
        return "17-24"
    return "25+"


def final_source(row: Dict[str, Any]) -> str:
    return str(metric(row, "final_source", "unknown"))


def selected_len(row: Dict[str, Any]) -> Optional[int]:
    value = metric(row, "selected_mask_length", metric(row, "mask_length"))
    return None if value is None else int(value)


def rate(values: Iterable[bool]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(1 for value in vals if value) / len(vals)


def compare(base_rows: List[Dict[str, Any]], cand_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = by_task(base_rows)
    cand = by_task(cand_rows)
    task_ids = sorted(set(base) & set(cand))
    pairwise: List[Dict[str, Any]] = []
    for task_id in task_ids:
        b = base[task_id]
        c = cand[task_id]
        b_pass = passed(b)
        c_pass = passed(c)
        if c_pass and not b_pass:
            outcome = "win"
        elif b_pass and not c_pass:
            outcome = "loss"
        elif b_pass and c_pass:
            outcome = "both_pass"
        else:
            outcome = "both_fail"
        olen = oracle_len(c)
        if olen is None:
            olen = oracle_len(b)
        pairwise.append(
            {
                "task_id": task_id,
                "outcome": outcome,
                "oracle_len": olen,
                "oracle_bucket": bucket(olen),
                "base_passed": b_pass,
                "candidate_passed": c_pass,
                "base_selected_len": selected_len(b),
                "candidate_selected_len": selected_len(c),
                "base_final_source": final_source(b),
                "candidate_final_source": final_source(c),
            }
        )

    buckets = ["<=8", "9-12", "13-16", "17-24", "25+", "unknown"]
    bucket_rows = []
    for name in buckets:
        rows = [row for row in pairwise if row["oracle_bucket"] == name]
        if rows:
            bucket_rows.append(
                {
                    "oracle_bucket": name,
                    "count": len(rows),
                    "wins": sum(1 for row in rows if row["outcome"] == "win"),
                    "losses": sum(1 for row in rows if row["outcome"] == "loss"),
                    "base_pass_rate": rate(row["base_passed"] for row in rows),
                    "candidate_pass_rate": rate(row["candidate_passed"] for row in rows),
                }
            )

    outcomes = Counter(row["outcome"] for row in pairwise)
    return {
        "summary": {
            "common": len(task_ids),
            "wins": outcomes["win"],
            "losses": outcomes["loss"],
            "net": outcomes["win"] - outcomes["loss"],
            "base_pass_rate": rate(row["base_passed"] for row in pairwise),
            "candidate_pass_rate": rate(row["candidate_passed"] for row in pairwise),
            "candidate_source_histogram": dict(Counter(row["candidate_final_source"] for row in pairwise)),
        },
        "buckets": bucket_rows,
        "pairwise": pairwise,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LCAL-style result JSONL files.")
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--candidate-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = compare(load_jsonl(args.base_results), load_jsonl(args.candidate_results))
    (out_dir / "summary.json").write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")
    write_csv(out_dir / "bucket_summary.csv", payload["buckets"])
    write_csv(out_dir / "pairwise.csv", payload["pairwise"])
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check the utility**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/analyze_lcal_pairwise.py
```

Expected: command exits with status `0`.

- [ ] **Step 3: Smoke-test on existing local outputs**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_lcal_pairwise.py \
  --base-results /home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus01_control_20260521_174051/results.jsonl \
  --candidate-results /home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_gpus01_20260520_201658/results.jsonl \
  --output-dir analysis_outputs/a6000_midcons_longrescue/smoke_pairwise_midcons_vs_gpus01_control
```

Expected: `summary.json`, `bucket_summary.csv`, and `pairwise.csv` are created; printed summary has `common` equal to `1033`.

- [ ] **Step 4: Commit**

Run:

```bash
git add analysis/analyze_lcal_pairwise.py analysis_outputs/a6000_midcons_longrescue/smoke_pairwise_midcons_vs_gpus01_control
git commit -m "Add LCAL pairwise analysis utility"
```

## Task 3: Add Mid-Rescue Precision Guards

**Files:**
- Modify: `clean_scripts/run_lcal_official_bounded_repair.py`

- [ ] **Step 1: Add CLI arguments**

Add these parser arguments near the existing `--mid-rescue-*` options:

```python
parser.add_argument("--mid-rescue-min-support-count", type=int, default=None)
parser.add_argument("--mid-rescue-best-long-lens", type=str, default=None)
parser.add_argument("--mid-rescue-veto-s3-le", type=int, default=None)
parser.add_argument("--mid-rescue-veto-official-len", type=int, default=None)
```

- [ ] **Step 2: Extend `BoundedRepairSettings`**

Add fields:

```python
mid_rescue_min_support_count: Optional[int] = None
mid_rescue_best_long_lens_csv: Optional[str] = None
mid_rescue_veto_s3_le: Optional[int] = None
mid_rescue_veto_official_len: Optional[int] = None
```

- [ ] **Step 3: Add helper functions**

Add near `_source_matches`:

```python
def _parse_optional_int_set(csv_text: Optional[str]) -> Optional[set[int]]:
    if csv_text is None or not str(csv_text).strip():
        return None
    return {int(part.strip()) for part in str(csv_text).split(",") if part.strip()}


def _mid_rescue_precision_passed(
    *,
    s3_selected: int,
    official_selected: int,
    settings: BoundedRepairSettings,
    support_count: Optional[int],
    best_long_len: Optional[int],
) -> Dict[str, Any]:
    support_passed = (
        settings.mid_rescue_min_support_count is None
        or (support_count is not None and int(support_count) >= int(settings.mid_rescue_min_support_count))
    )
    allowed_lens = _parse_optional_int_set(settings.mid_rescue_best_long_lens_csv)
    best_long_len_passed = allowed_lens is None or (
        best_long_len is not None and int(best_long_len) in allowed_lens
    )
    vetoed_short_jump = (
        settings.mid_rescue_veto_s3_le is not None
        and settings.mid_rescue_veto_official_len is not None
        and int(s3_selected) <= int(settings.mid_rescue_veto_s3_le)
        and int(official_selected) >= int(settings.mid_rescue_veto_official_len)
        and not support_passed
    )
    return {
        "support_passed": bool(support_passed),
        "best_long_len_passed": bool(best_long_len_passed),
        "vetoed_short_jump": bool(vetoed_short_jump),
        "passed": bool(support_passed and best_long_len_passed and not vetoed_short_jump),
    }
```

- [ ] **Step 4: Wire helper into `_repair_decision`**

Extend `_repair_decision` signature:

```python
support_count: Optional[int] = None,
best_long_len: Optional[int] = None,
```

Inside the mid-rescue block, compute:

```python
precision = _mid_rescue_precision_passed(
    s3_selected=s3_selected,
    official_selected=int(official_selected),
    settings=settings,
    support_count=support_count,
    best_long_len=best_long_len,
)
```

Require `precision["passed"]` in `mid_rescue_triggered`.

- [ ] **Step 5: Preserve diagnostics**

Add returned fields from `_repair_decision`:

```python
"mid_rescue_precision_support_passed": precision["support_passed"] if mid_rescue_enabled else None,
"mid_rescue_precision_best_long_len_passed": precision["best_long_len_passed"] if mid_rescue_enabled else None,
"mid_rescue_precision_vetoed_short_jump": precision["vetoed_short_jump"] if mid_rescue_enabled else None,
```

Also copy these fields into `meta` and `result["metrics"]` next to the existing `official_repair_mid_rescue_*` fields.

- [ ] **Step 6: Pass metadata into `_repair_decision`**

Both `_repair_decision` calls in `resolve_mask_length_bounded_repair` must pass:

```python
support_count=meta.get("support_count"),
best_long_len=meta.get("best_long_len"),
```

- [ ] **Step 7: Syntax-check**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_lcal_official_bounded_repair.py
```

Expected: command exits with status `0`.

- [ ] **Step 8: Run a 5-sample smoke candidate**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py \
  --max-samples 5 \
  --experiment-name smoke_a6000_mid_precision \
  --baseline-results outputs_clean/202604/full_lcas_v3b_alpha006_compact_sl_20260429_180958/results.jsonl \
  --official-eval-max-s3-len 12 \
  --repair-max-s3-len 5 \
  --repair-min-official-len 6 \
  --repair-max-official-len 9 \
  --repair-min-delta 1 \
  --repair-max-delta 8 \
  --suspicion-max-s3-len 5 \
  --suspicion-min-official-len 16 \
  --suspicion-max-official-len 64 \
  --suspicion-min-delta 1 \
  --mid-rescue-max-s3-len 12 \
  --mid-rescue-source base \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 13 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 7 \
  --mid-rescue-min-long-ratio 0.8 \
  --mid-rescue-min-support-count 2 \
  --mid-rescue-best-long-lens 13,14,15,16 \
  --mid-rescue-veto-s3-le 5 \
  --mid-rescue-veto-official-len 13
```

Expected: run completes and `outputs_clean/smoke_a6000_mid_precision_*` contains `summary.json`.

- [ ] **Step 9: Commit**

Run:

```bash
git add clean_scripts/run_lcal_official_bounded_repair.py
git commit -m "Add mid-rescue precision guards"
```

## Task 4: Add True-Long Rescue Policy Flags

**Files:**
- Modify: `clean_scripts/run_lcal_official_bounded_repair.py`

- [ ] **Step 1: Add CLI arguments**

Add near the mid-rescue arguments:

```python
parser.add_argument("--true-long-max-s3-len", type=int, default=None)
parser.add_argument("--true-long-min-official-len", type=int, default=None)
parser.add_argument("--true-long-max-official-len", type=int, default=64)
parser.add_argument("--true-long-min-delta", type=int, default=8)
parser.add_argument("--true-long-min-long-ratio", type=float, default=None)
parser.add_argument("--true-long-min-raw-ratio", type=float, default=None)
parser.add_argument("--true-long-min-support-count", type=int, default=None)
parser.add_argument("--true-long-best-long-lens", type=str, default=None)
parser.add_argument("--true-long-source", type=str, default="any")
```

- [ ] **Step 2: Extend `BoundedRepairSettings`**

Add fields:

```python
true_long_max_s3_len: Optional[int] = None
true_long_min_official_len: Optional[int] = None
true_long_max_official_len: int = 64
true_long_min_delta: int = 8
true_long_min_long_ratio: Optional[float] = None
true_long_min_raw_ratio: Optional[float] = None
true_long_min_support_count: Optional[int] = None
true_long_best_long_lens_csv: Optional[str] = None
true_long_source: str = "any"
```

- [ ] **Step 3: Add true-long decision block**

Inside `_repair_decision`, after the mid-rescue block, compute a separate `true_long_triggered`:

```python
true_long_enabled = settings.true_long_min_official_len is not None
true_long_len_passed = False
true_long_delta_passed = False
true_long_ratio_passed = False
true_long_raw_ratio_passed = False
true_long_support_passed = False
true_long_best_long_len_passed = False
true_long_source_passed = False
true_long_triggered = False
if true_long_enabled:
    true_long_max_s3_len = (
        int(settings.true_long_max_s3_len)
        if settings.true_long_max_s3_len is not None
        else official_eval_max
    )
    true_long_len_passed = (
        int(s3_selected) <= true_long_max_s3_len
        and int(settings.true_long_min_official_len) <= int(official_selected) <= int(settings.true_long_max_official_len)
    )
    true_long_delta_passed = int(delta) >= int(settings.true_long_min_delta)
    true_long_ratio_passed = (
        settings.true_long_min_long_ratio is None
        or (long_ratio is not None and float(long_ratio) >= float(settings.true_long_min_long_ratio))
    )
    true_long_raw_ratio_passed = (
        settings.true_long_min_raw_ratio is None
        or (raw_long_ratio is not None and float(raw_long_ratio) >= float(settings.true_long_min_raw_ratio))
    )
    true_long_support_passed = (
        settings.true_long_min_support_count is None
        or (support_count is not None and int(support_count) >= int(settings.true_long_min_support_count))
    )
    true_long_lens = _parse_optional_int_set(settings.true_long_best_long_lens_csv)
    true_long_best_long_len_passed = true_long_lens is None or (
        best_long_len is not None and int(best_long_len) in true_long_lens
    )
    true_long_source_passed = _source_matches(pre_repair_source, settings.true_long_source)
    true_long_triggered = bool(
        true_long_len_passed
        and true_long_delta_passed
        and true_long_ratio_passed
        and true_long_raw_ratio_passed
        and true_long_support_passed
        and true_long_best_long_len_passed
        and true_long_source_passed
    )
```

Extend `_repair_decision` signature with `raw_long_ratio`.

- [ ] **Step 4: Update trigger priority**

Use this priority:

```python
triggered = bool(bounded_triggered or suspicion_triggered or mid_rescue_triggered or true_long_triggered)
if triggered:
    if bounded_triggered:
        reason = "official_bounded_repair"
    elif suspicion_triggered:
        reason = "official_long_suspicion"
    elif mid_rescue_triggered:
        reason = "official_mid_rescue"
    else:
        reason = "official_true_long_rescue"
```

- [ ] **Step 5: Add diagnostics to config, meta, and metrics**

Add all `true_long_*` settings to `config_payload["lcal_official_bounded_repair"]["repair"]`.

Add decision booleans to `meta` and `result["metrics"]`:

```python
"official_repair_true_long_len_passed": decision["true_long_len_passed"],
"official_repair_true_long_delta_passed": decision["true_long_delta_passed"],
"official_repair_true_long_ratio_passed": decision["true_long_ratio_passed"],
"official_repair_true_long_raw_ratio_passed": decision["true_long_raw_ratio_passed"],
"official_repair_true_long_support_passed": decision["true_long_support_passed"],
"official_repair_true_long_best_long_len_passed": decision["true_long_best_long_len_passed"],
"official_repair_true_long_source_passed": decision["true_long_source_passed"],
```

- [ ] **Step 6: Update summary counts**

In `summarize_bounded_repair_results`, add:

```python
true_long_triggered = [
    m for m in repair_triggered if str(m.get("official_repair_reason")) == "official_true_long_rescue"
]
```

Add count, rate, oracle bucket histogram, pass rate, and true-long precision fields matching the existing mid-rescue summary pattern.

- [ ] **Step 7: Syntax-check**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_lcal_official_bounded_repair.py
```

Expected: command exits with status `0`.

- [ ] **Step 8: Run a 5-sample smoke true-long candidate**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py \
  --max-samples 5 \
  --experiment-name smoke_a6000_true_long_rescue \
  --baseline-results outputs_clean/202604/full_lcas_v3b_alpha006_compact_sl_20260429_180958/results.jsonl \
  --official-eval-max-s3-len 12 \
  --repair-max-s3-len 5 \
  --repair-min-official-len 6 \
  --repair-max-official-len 9 \
  --repair-min-delta 1 \
  --repair-max-delta 8 \
  --suspicion-max-s3-len 5 \
  --suspicion-min-official-len 16 \
  --suspicion-max-official-len 64 \
  --suspicion-min-delta 1 \
  --true-long-max-s3-len 12 \
  --true-long-min-official-len 17 \
  --true-long-max-official-len 64 \
  --true-long-min-delta 8 \
  --true-long-min-long-ratio 0.85 \
  --true-long-min-raw-ratio 0.85 \
  --true-long-min-support-count 2 \
  --true-long-best-long-lens 16,20,24,28,32,40 \
  --true-long-source any
```

Expected: run completes and summary contains `official_true_long_rescue` fields.

- [ ] **Step 9: Commit**

Run:

```bash
git add clean_scripts/run_lcal_official_bounded_repair.py
git commit -m "Add true-long rescue policy flags"
```

## Task 5: Add Candidate Run Script

**Files:**
- Create: `clean_scripts/run_lcal_a6000_candidates.sh`

- [ ] **Step 1: Create the candidate script**

Add a script with three candidate runs:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/a6000-midcons-longrescue

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

PYTHON=/home/shx/miniconda3/envs/dllm_env/bin/python
RUNNER=clean_scripts/run_lcal_official_bounded_repair.py
BASELINE=outputs_clean/202604/full_lcas_v3b_alpha006_compact_sl_20260429_180958/results.jsonl

COMMON_ARGS=(
  --baseline-results "$BASELINE"
  --official-eval-max-s3-len 12
  --repair-max-s3-len 5
  --repair-min-official-len 6
  --repair-max-official-len 9
  --repair-min-delta 1
  --repair-max-delta 8
  --suspicion-max-s3-len 5
  --suspicion-min-official-len 16
  --suspicion-max-official-len 64
  --suspicion-min-delta 1
)

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000 \
  "${COMMON_ARGS[@]}" \
  --mid-rescue-max-s3-len 12 \
  --mid-rescue-source base \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 13 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 7 \
  --mid-rescue-min-long-ratio 0.8 \
  --mid-rescue-min-support-count 2 \
  --mid-rescue-best-long-lens 13,14,15,16 \
  --mid-rescue-veto-s3-le 5 \
  --mid-rescue-veto-official-len 13

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000 \
  "${COMMON_ARGS[@]}" \
  --true-long-max-s3-len 12 \
  --true-long-min-official-len 17 \
  --true-long-max-official-len 64 \
  --true-long-min-delta 8 \
  --true-long-min-long-ratio 0.85 \
  --true-long-min-raw-ratio 0.85 \
  --true-long-min-support-count 2 \
  --true-long-best-long-lens 16,20,24,28,32,40 \
  --true-long-source any

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000 \
  "${COMMON_ARGS[@]}" \
  --mid-rescue-max-s3-len 12 \
  --mid-rescue-source base \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 13 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 7 \
  --mid-rescue-min-long-ratio 0.8 \
  --mid-rescue-min-support-count 2 \
  --mid-rescue-best-long-lens 13,14,15,16 \
  --mid-rescue-veto-s3-le 5 \
  --mid-rescue-veto-official-len 13 \
  --true-long-max-s3-len 12 \
  --true-long-min-official-len 17 \
  --true-long-max-official-len 64 \
  --true-long-min-delta 8 \
  --true-long-min-long-ratio 0.85 \
  --true-long-min-raw-ratio 0.85 \
  --true-long-min-support-count 2 \
  --true-long-best-long-lens 16,20,24,28,32,40 \
  --true-long-source any
```

- [ ] **Step 2: Make executable and syntax-check**

Run:

```bash
chmod +x clean_scripts/run_lcal_a6000_candidates.sh
bash -n clean_scripts/run_lcal_a6000_candidates.sh
```

Expected: both commands exit with status `0`.

- [ ] **Step 3: Commit**

Run:

```bash
git add clean_scripts/run_lcal_a6000_candidates.sh
git commit -m "Add A6000 candidate run script"
```

## Task 6: Run And Compare A6000 Full Experiments

**Files:**
- Modify: `analysis_outputs/a6000_midcons_longrescue/README.md`
- Modify: `analysis_outputs/experiment_scoreboard.md`

- [ ] **Step 1: Run A6000 baselines**

Run:

```bash
clean_scripts/run_lcal_a6000_baselines.sh
```

Expected: two full output directories appear under `outputs_clean/`, one for A6000 control and one for A6000 midcons.

- [ ] **Step 2: Compare midcons to A6000 control**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_lcal_pairwise.py \
  --base-results outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_*/results.jsonl \
  --candidate-results outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_*/results.jsonl \
  --output-dir analysis_outputs/a6000_midcons_longrescue/midcons_vs_a6000_control
```

Expected: comparison summary has `common` equal to `1033`.

- [ ] **Step 3: Run A6000 candidates**

Run:

```bash
clean_scripts/run_lcal_a6000_candidates.sh
```

Expected: three candidate output directories appear under `outputs_clean/`.

- [ ] **Step 4: Compare every candidate to A6000 control**

Run `analysis/analyze_lcal_pairwise.py` once per candidate, writing outputs under:

```text
analysis_outputs/a6000_midcons_longrescue/mid_precision_vs_a6000_control
analysis_outputs/a6000_midcons_longrescue/true_long_vs_a6000_control
analysis_outputs/a6000_midcons_longrescue/combined_vs_a6000_control
```

Expected: each comparison summary has `common` equal to `1033`.

- [ ] **Step 5: Update scoreboard**

Add a new A6000 section to `analysis_outputs/experiment_scoreboard.md` with:

```markdown
## A6000 Controlled Runs

| Run | Method | Pass | Delta vs A6000 control | <=8 | 9-12 | 13-16 | 17-24 | 25+ | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
```

Fill the table from each run's `summary.json` and pairwise comparison outputs.

- [ ] **Step 6: Commit**

Run:

```bash
git add analysis_outputs/a6000_midcons_longrescue analysis_outputs/experiment_scoreboard.md
git commit -m "Record A6000 controlled experiment results"
```
