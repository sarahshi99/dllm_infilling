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
    lower = name.lower()
    if "smoke" in lower:
        return "diagnostic"
    if "control" in lower:
        return "env_control"
    if "midcons" in lower or "midaggr" in lower:
        return "candidate"
    if "alpha" in lower or "ratio" in lower or "short_safe" in lower:
        return "superseded"
    return "diagnostic"


def get_nested(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def pass_count_from_summary(summary: Dict[str, Any]) -> Optional[int]:
    for key in ("pass_count", "num_passed", "passed_count"):
        value = summary.get(key)
        if value is not None:
            return int(value)
    required = summary.get("required_metrics") or {}
    value = required.get("pass_count")
    return None if value is None else int(value)


def sample_count_from_summary(summary: Dict[str, Any]) -> Optional[int]:
    for key in ("num_samples", "total", "total_count"):
        value = summary.get(key)
        if value is not None:
            return int(value)
    required = summary.get("required_metrics") or {}
    value = required.get("num_samples")
    return None if value is None else int(value)


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    config = load_json(run_dir / "config.json")
    first_row = first_jsonl(run_dir / "results.jsonl")
    metrics = first_row.get("metrics", {})
    model = (
        get_nested(config, "model", "model_path")
        or config.get("model_path")
        or first_row.get("model_path")
        or "unknown"
    )
    experiment_name = (
        get_nested(config, "logging", "experiment_name")
        or summary.get("experiment_name")
        or run_dir.name
    )
    sample_count = sample_count_from_summary(summary) or count_jsonl(run_dir / "results.jsonl")
    pass_count = pass_count_from_summary(summary)
    pass_rate = summary.get("pass_rate")
    if pass_rate is None and pass_count is not None and sample_count:
        pass_rate = float(pass_count) / float(sample_count)
    if pass_count is None and pass_rate is not None and sample_count:
        pass_count = round(float(pass_rate) * int(sample_count))

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
        sample_count = row["sample_count"] if row["sample_count"] is not None else "unknown"
        pass_count = row["pass_count"] if row["pass_count"] is not None else "unknown"
        lines.append(
            f"| `{row['run_id']}` | `{row['status']}` | {sample_count} | {pass_count} | "
            f"{rate_text} | `{row['model']}` | `{row['raw_path']}` |"
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
