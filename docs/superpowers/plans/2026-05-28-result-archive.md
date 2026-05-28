# Historical Result Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a compact committed archive of meaningful historical experiment results while keeping large raw outputs local.

**Architecture:** Add one registry builder that scans local `outputs_clean/` run directories, extracts config and summary metadata when present, and writes human-readable plus machine-readable registries. Keep manual interpretation in markdown so research judgment is visible rather than buried in code.

**Tech Stack:** Python 3, JSON/JSONL, Markdown, existing local `outputs_clean/` and `analysis_outputs/` directories.

---

## File Structure

- Create `analysis/build_run_registry.py`: scans run directories and writes `docs/results/run_registry.json` plus `docs/results/run_registry.md`.
- Modify `docs/results/run_registry.md`: replace bootstrap text with generated table and interpretation notes.
- Create `docs/results/archive_notes.md`: records triage decisions and artifact policy.
- Modify `analysis_outputs/experiment_scoreboard.md`: link to the registry and clarify which historical runs are canonical.

## Task 1: Add Registry Builder

**Files:**
- Create: `analysis/build_run_registry.py`

- [ ] **Step 1: Create the script**

Add this implementation:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


STATUS_BY_NAME = {
    "full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus23_20260519_175826": "global_best",
    "full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus01_control_20260521_174051": "env_control",
    "full_lcal_official_bounded_repair_union_eval12_nomiddle_s3_off6_9_delta1_8_susp16_gpus01_control_20260521_182208": "env_control",
    "full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_gpus01_20260520_201658": "candidate",
    "full_lcal_official_bounded_repair_union_midaggr_off11_15_d3_8_r08_gpus01_20260520_210248": "candidate",
}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_jsonl(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def first_jsonl(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def classify_run(name: str) -> str:
    if name in STATUS_BY_NAME:
        return STATUS_BY_NAME[name]
    if "smoke" in name:
        return "diagnostic"
    if "control" in name:
        return "env_control"
    if "midcons" in name or "midaggr" in name:
        return "candidate"
    if "alpha" in name or "ratio" in name or "short_safe" in name:
        return "superseded"
    return "diagnostic"


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    config = load_json(run_dir / "config.json")
    first_row = first_jsonl(run_dir / "results.jsonl")
    metrics = first_row.get("metrics", {})
    model = (
        config.get("model", {}).get("model_path")
        or config.get("model_path")
        or first_row.get("model_path")
        or "unknown"
    )
    experiment_name = (
        config.get("logging", {}).get("experiment_name")
        or summary.get("experiment_name")
        or run_dir.name
    )
    pass_count = summary.get("pass_count") or summary.get("num_passed") or summary.get("passed_count")
    sample_count = summary.get("num_samples") or summary.get("total") or count_jsonl(run_dir / "results.jsonl")
    pass_rate = summary.get("pass_rate")
    if pass_rate is None and pass_count is not None and sample_count:
        pass_rate = float(pass_count) / float(sample_count)
    return {
        "run_id": run_dir.name,
        "experiment_name": experiment_name,
        "status": classify_run(run_dir.name),
        "raw_path": str(run_dir),
        "model": model,
        "sample_count": sample_count,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "mask_length_source": summary.get("mask_length_source") or metrics.get("mask_length_source") or "unknown",
        "final_source_histogram": summary.get("final_source_histogram")
        or summary.get("official_repair_source_histogram")
        or {},
        "oracle_bucket_pass_rates": summary.get("oracle_bucket_pass_rates")
        or summary.get("bucket_pass_rates")
        or {},
    }


def iter_run_dirs(outputs_dir: Path) -> Iterable[Path]:
    for path in sorted(outputs_dir.iterdir()):
        if path.is_dir() and (path / "results.jsonl").exists():
            yield path
    archive_202604 = outputs_dir / "202604"
    if archive_202604.exists():
        for path in sorted(archive_202604.iterdir()):
            if path.is_dir() and (path / "results.jsonl").exists():
                yield path


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Run Registry",
        "",
        "This registry records meaningful local experiment outputs without committing raw `results.jsonl` files to normal git.",
        "",
        "| Run | Status | Samples | Pass | Rate | Model | Raw Path |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        rate = row["pass_rate"]
        rate_text = "unknown" if rate is None else f"{100.0 * float(rate):.2f}%"
        lines.append(
            f"| `{row['run_id']}` | `{row['status']}` | {row['sample_count'] or 'unknown'} | "
            f"{row['pass_count'] or 'unknown'} | {rate_text} | `{row['model']}` | `{row['raw_path']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact registry for local experiment outputs.")
    parser.add_argument("--outputs-dir", default="/home/shx/projects/dllm_infilling/outputs_clean")
    parser.add_argument("--json-out", default="docs/results/run_registry.json")
    parser.add_argument("--md-out", default="docs/results/run_registry.md")
    args = parser.parse_args()

    rows = [summarize_run(path) for path in iter_run_dirs(Path(args.outputs_dir))]
    rows.sort(key=lambda row: (row["status"], row["run_id"]))
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, rows)
    print(f"wrote {len(rows)} registry rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/build_run_registry.py
```

Expected: command exits with status `0`.

- [ ] **Step 3: Generate registry**

Run:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_run_registry.py \
  --outputs-dir /home/shx/projects/dllm_infilling/outputs_clean \
  --json-out docs/results/run_registry.json \
  --md-out docs/results/run_registry.md
```

Expected: output prints the number of registry rows and writes both registry files.

- [ ] **Step 4: Commit**

Run:

```bash
git add analysis/build_run_registry.py docs/results/run_registry.json docs/results/run_registry.md
git commit -m "Build historical run registry"
```

## Task 2: Add Archive Notes

**Files:**
- Create: `docs/results/archive_notes.md`

- [ ] **Step 1: Add notes file**

Create:

```markdown
# Result Archive Notes

Raw experiment outputs remain local under `/home/shx/projects/dllm_infilling/outputs_clean`.

Normal git stores compact records:

- `docs/results/run_registry.md`
- `docs/results/run_registry.json`
- `analysis_outputs/experiment_scoreboard.md`
- compact CSV/JSON diagnostics under `analysis_outputs/`

Canonical historical runs:

- `old_union_gpus23`: old global checkpoint, `787/1033 = 76.19%`.
- `union_gpus01_control`: same-policy control on the newer environment, `785/1033 = 75.99%`.
- `eval12_nomiddle_gpus01_control`: gate-control run, same pass/fail set as `union_gpus01_control`.
- `midcons`: best newer-environment candidate, `791/1033 = 76.57%`.
- `midaggr`: record-only candidate, more wins but too many short/mid losses.

Artifact policy:

- Do not commit raw full `outputs_clean/` to normal git.
- Use Git LFS or external storage only for selected raw files that must be preserved verbatim.
- Never delete local raw outputs during archive work without explicit user approval.
```

- [ ] **Step 2: Commit**

Run:

```bash
git add docs/results/archive_notes.md
git commit -m "Document result archive policy"
```

## Task 3: Link Scoreboard To Registry

**Files:**
- Modify: `analysis_outputs/experiment_scoreboard.md`

- [ ] **Step 1: Add registry note near the top**

Add this paragraph below the `Updated:` line:

```markdown
For a compact index of local raw run directories, see `docs/results/run_registry.md`. Raw `outputs_clean/` files remain local unless a run is explicitly selected for Git LFS or external artifact storage.
```

- [ ] **Step 2: Verify markdown references**

Run:

```bash
rg -n "run_registry|outputs_clean|old_union_gpus23|midcons" analysis_outputs/experiment_scoreboard.md docs/results/run_registry.md docs/results/archive_notes.md
```

Expected: command prints matches in all three files.

- [ ] **Step 3: Commit**

Run:

```bash
git add analysis_outputs/experiment_scoreboard.md
git commit -m "Link scoreboard to run registry"
```
