#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BUCKETS = ["<=8", "9-12", "13-16", "17-24", "25+"]


@dataclass(frozen=True)
class RunSpec:
    label: str
    method: str
    prefix: str
    comparison_dir: Optional[str]
    status: str


RUN_SPECS = [
    RunSpec(
        label="a6000_control",
        method="union control: S3 short-safe + bounded repair + long suspicion",
        prefix="full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_",
        comparison_dir=None,
        status="A6000 baseline",
    ),
    RunSpec(
        label="midcons",
        method="union + conservative mid rescue off11..13, delta3..7, ratio>=0.8",
        prefix="full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_",
        comparison_dir="midcons_vs_a6000_control",
        status="A6000 mid baseline",
    ),
    RunSpec(
        label="mid_precision",
        method="mid rescue + support/best-len/short-jump guards",
        prefix="full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_",
        comparison_dir="mid_precision_vs_a6000_control",
        status="candidate",
    ),
    RunSpec(
        label="true_long",
        method="true-long rescue off>=17, delta>=8, ratio>=0.85, support>=2",
        prefix="full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_",
        comparison_dir="true_long_vs_a6000_control",
        status="candidate",
    ),
    RunSpec(
        label="combined",
        method="mid precision + true-long rescue",
        prefix="full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_",
        comparison_dir="combined_vs_a6000_control",
        status="candidate",
    ),
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_run_dir(outputs_dir: Path, prefix: str) -> Optional[Path]:
    matches = sorted(path for path in outputs_dir.glob(f"{prefix}*") if path.is_dir())
    return matches[-1] if matches else None


def pass_count(summary: Dict[str, Any]) -> Optional[int]:
    for key in ("pass_count", "num_passed", "passed_count"):
        value = summary.get(key)
        if value is not None:
            return int(value)
    required = summary.get("required_metrics")
    if isinstance(required, dict) and required.get("pass_count") is not None:
        return int(required["pass_count"])
    sample_count = summary.get("num_samples")
    rate = summary.get("pass_rate")
    if sample_count is not None and rate is not None:
        return round(float(rate) * int(sample_count))
    return None


def sample_count(summary: Dict[str, Any]) -> Optional[int]:
    for key in ("num_samples", "total", "total_count"):
        value = summary.get(key)
        if value is not None:
            return int(value)
    required = summary.get("required_metrics")
    if isinstance(required, dict) and required.get("num_samples") is not None:
        return int(required["num_samples"])
    return None


def pass_rate(summary: Dict[str, Any]) -> Optional[float]:
    rate = summary.get("pass_rate")
    if rate is not None:
        return float(rate)
    passed = pass_count(summary)
    total = sample_count(summary)
    if passed is None or not total:
        return None
    return float(passed) / float(total)


def bucket_rate(summary: Dict[str, Any], bucket: str) -> Optional[float]:
    bucket_rates = summary.get("oracle_bucket_pass_rate")
    if isinstance(bucket_rates, dict) and bucket_rates.get(bucket) is not None:
        return float(bucket_rates[bucket])
    required = summary.get("required_metrics")
    key = f"{bucket}_bucket_pass_rate"
    if isinstance(required, dict) and required.get(key) is not None:
        return float(required[key])
    return None


def format_rate(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{100.0 * float(value):.2f}%"


def format_pass(summary: Dict[str, Any]) -> str:
    passed = pass_count(summary)
    total = sample_count(summary)
    rate = pass_rate(summary)
    if passed is None or total is None:
        return f"`unknown` `{format_rate(rate)}`"
    return f"`{passed}/{total}` `{format_rate(rate)}`"


def format_delta(comparison_summary: Optional[Dict[str, Any]], *, is_control: bool = False) -> str:
    if comparison_summary is None:
        return "baseline" if is_control else "comparison missing"
    net = int(comparison_summary.get("net", 0))
    wins = int(comparison_summary.get("wins", 0))
    losses = int(comparison_summary.get("losses", 0))
    sign = "+" if net > 0 else ""
    return f"`{sign}{net}` ({wins}W/{losses}L)"


def iter_rows(outputs_dir: Path, analysis_dir: Path) -> Iterable[Dict[str, str]]:
    for spec in RUN_SPECS:
        run_dir = latest_run_dir(outputs_dir, spec.prefix)
        if run_dir is None:
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)

        comparison_summary = None
        if spec.comparison_dir is not None:
            comparison_path = analysis_dir / spec.comparison_dir / "summary.json"
            if comparison_path.exists():
                comparison_summary = load_json(comparison_path)

        row = {
            "label": spec.label,
            "method": spec.method,
            "pass": format_pass(summary),
            "delta": format_delta(comparison_summary, is_control=spec.comparison_dir is None),
            "status": spec.status,
            "run_dir": str(run_dir),
        }
        for bucket in BUCKETS:
            row[bucket] = f"`{format_rate(bucket_rate(summary, bucket))}`"
        yield row


def build_section(outputs_dir: Path, analysis_dir: Path, generated_at: Optional[str] = None) -> str:
    timestamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [
        "## A6000 Controlled Runs",
        "",
        f"Generated: {timestamp}",
        "",
        "| Run | Method | Pass | Delta vs A6000 control | <=8 | 9-12 | 13-16 | 17-24 | 25+ | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in iter_rows(outputs_dir, analysis_dir):
        lines.append(
            f"| `{row['label']}` | {row['method']} | {row['pass']} | {row['delta']} | "
            f"{row['<=8']} | {row['9-12']} | {row['13-16']} | {row['17-24']} | "
            f"{row['25+']} | {row['status']} |"
        )
    lines.append("")
    lines.append("### A6000 Run Directories")
    lines.append("")
    for row in iter_rows(outputs_dir, analysis_dir):
        lines.append(f"- `{row['label']}`: `{row['run_dir']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an A6000 scoreboard markdown section.")
    parser.add_argument("--outputs-dir", default="outputs_clean")
    parser.add_argument("--analysis-dir", default="analysis_outputs/a6000_midcons_longrescue")
    parser.add_argument("--output", default="analysis_outputs/a6000_midcons_longrescue/a6000_scoreboard_section.md")
    args = parser.parse_args()

    section = build_section(outputs_dir=Path(args.outputs_dir), analysis_dir=Path(args.analysis_dir))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(section, encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
