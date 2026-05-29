#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_nested(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def passed_from_row(row: Dict[str, Any]) -> Optional[bool]:
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and metrics.get("passed") is not None:
        return bool(metrics["passed"])
    verification = row.get("verification")
    if isinstance(verification, dict):
        if verification.get("passed") is not None:
            return bool(verification["passed"])
        tier_values = [value for value in verification.values() if isinstance(value, dict)]
        if tier_values and all(value.get("passed") is not None for value in tier_values):
            return all(bool(value.get("passed")) for value in tier_values)
    return None


def result_jsonl_stats(path: Path) -> Dict[str, Optional[int]]:
    if not path.exists():
        return {"sample_count": None, "pass_count": None}
    sample_count = 0
    pass_count = 0
    saw_pass_field = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample_count += 1
            passed = passed_from_row(json.loads(line))
            if passed is not None:
                saw_pass_field = True
                pass_count += int(passed)
    return {"sample_count": sample_count, "pass_count": pass_count if saw_pass_field else None}


def pass_count_from_summary(summary: Dict[str, Any]) -> Optional[int]:
    for key in ("pass_count", "num_passed", "passed_count"):
        value = summary.get(key)
        if value is not None:
            return int(value)
    return None


def sample_count_from_summary(summary: Dict[str, Any]) -> Optional[int]:
    for key in ("num_samples", "total", "total_count"):
        value = summary.get(key)
        if value is not None:
            return int(value)
    return None


def first_jsonl(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def run_family(run_dir: Path, root: Path) -> str:
    try:
        return str(run_dir.relative_to(root).parts[0])
    except (ValueError, IndexError):
        return "unknown"


def summarize_run(run_dir: Path, root: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.jsonl"
    summary = load_json(summary_path)
    config = load_json(run_dir / "config.json")
    first_row = first_jsonl(results_path)
    jsonl_stats = result_jsonl_stats(results_path)

    model = (
        get_nested(config, "model", "model_path")
        or config.get("model_path")
        or first_row.get("model_path")
        or "unknown"
    )
    experiment_name = get_nested(config, "logging", "experiment_name") or run_dir.name
    sample_count = sample_count_from_summary(summary) or jsonl_stats["sample_count"]
    pass_count = pass_count_from_summary(summary)
    if pass_count is None:
        pass_count = jsonl_stats["pass_count"]
    pass_rate = summary.get("pass_rate")
    if pass_rate is None and pass_count is not None and sample_count:
        pass_rate = float(pass_count) / float(sample_count)
    if pass_count is None and pass_rate is not None and sample_count:
        pass_count = round(float(pass_rate) * int(sample_count))

    return {
        "run_id": run_dir.name,
        "family": run_family(run_dir, root),
        "experiment_name": experiment_name,
        "raw_path": str(run_dir),
        "has_summary": summary_path.exists(),
        "model": model,
        "sample_count": sample_count,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "mask_length_source": summary.get("mask_length_source")
        or get_nested(config, "decode", "mask_length_source")
        or "unknown",
        "prompt_formats": summary.get("prompt_formats") or summary.get("canvas_formats") or [],
        "decode_backends": summary.get("decode_backends") or [],
        "exact_length_match_rate": summary.get("exact_length_match_rate"),
        "avg_abs_selected_minus_oracle_length": summary.get("avg_abs_selected_minus_oracle_length"),
    }


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*/*")):
        if path.is_dir() and (path / "results.jsonl").exists():
            yield path


def should_include(row: Dict[str, Any], include_smoke: bool, include_incomplete: bool) -> bool:
    run_id = row["run_id"].lower()
    if "smoke" in run_id and not include_smoke:
        return False
    if row["has_summary"]:
        return True
    if include_incomplete:
        return True
    return row["sample_count"] == 1033


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Model Generalization Registry",
        "",
        "This registry records compact metadata for local cross-model infilling runs.",
        "Smoke runs are excluded by default; raw `results.jsonl` files remain local.",
        "",
        "| Run | Family | Model | Samples | Pass | Rate | Mask Source | Raw Path |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        rate = row["pass_rate"]
        rate_text = "unknown" if rate is None else f"{100.0 * float(rate):.2f}%"
        sample_count = row["sample_count"] if row["sample_count"] is not None else "unknown"
        pass_count = row["pass_count"] if row["pass_count"] is not None else "unknown"
        lines.append(
            f"| `{row['run_id']}` | `{row['family']}` | `{row['model']}` | "
            f"{sample_count} | {pass_count} | {rate_text} | "
            f"`{row['mask_length_source']}` | `{row['raw_path']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact registry for model generalization runs.")
    parser.add_argument(
        "--runs-dir",
        default="/home/shx/projects/dllm_infilling/model_generalization_runs",
    )
    parser.add_argument("--json-out", default="docs/results/model_generalization_registry.json")
    parser.add_argument("--md-out", default="docs/results/model_generalization_registry.md")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()

    root = Path(args.runs_dir)
    rows = [
        row
        for row in (summarize_run(path, root) for path in iter_run_dirs(root))
        if should_include(
            row,
            include_smoke=args.include_smoke,
            include_incomplete=args.include_incomplete,
        )
    ]
    rows.sort(key=lambda row: (row["model"], row["family"], row["run_id"]))

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, rows)
    print(f"wrote {len(rows)} model-generalization rows")


if __name__ == "__main__":
    main()
