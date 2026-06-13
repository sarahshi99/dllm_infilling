#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


ROUTE_ORDER = (
    "route1_trace_only",
    "route2_risk_controlled",
    "route3_multi_canvas_rerank",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
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


def _route_items(summary: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    routes = summary.get("routes") or {}
    items = [routes[key] for key in ROUTE_ORDER if key in routes]
    if items:
        return items
    return [
        {
            "route_name": "Route 1 trace-only detector",
            "trigger_count": summary.get("route1_trigger_count", 0),
            "failed_long_trigger_count": summary.get("route1_failed_long_count", 0),
            "short_risk_count": summary.get("route1_short_count", 0),
            "current_pass_risk_count": "n/a",
            "gate_a_pessimistic_passed": "n/a",
            "gate_b_pessimistic_exploratory": "n/a",
            "decision": "n/a",
        },
        {
            "route_name": "Route 2 risk-controlled rescue",
            "trigger_count": summary.get("route2_trigger_count", 0),
            "failed_long_trigger_count": summary.get("route2_failed_long_count", 0),
            "short_risk_count": summary.get("route2_short_count", 0),
            "current_pass_risk_count": "n/a",
            "gate_a_pessimistic_passed": "n/a",
            "gate_b_pessimistic_exploratory": "n/a",
            "decision": "n/a",
        },
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def route_rows(label: str, summary: Mapping[str, Any]) -> List[Tuple[Any, ...]]:
    rows: List[Tuple[Any, ...]] = []
    for route in _route_items(summary):
        rows.append(
            (
                label,
                route.get("route_name", "unknown"),
                route.get("trigger_count", 0),
                route.get("failed_long_trigger_count", 0),
                route.get("short_risk_count", 0),
                route.get("current_pass_risk_count", 0),
                _fmt(route.get("gate_a_pessimistic_passed")),
                _fmt(route.get("gate_b_pessimistic_exploratory")),
                route.get("decision", "n/a"),
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
    _, prev_total, prev_rate = pass_rate(prev_run)
    _, mid_total, mid_rate = pass_rate(midcons_run)
    run_rows = [
        (
            "previous local method trace",
            prev_run,
            prev_total,
            count_lines(prev_run / "step_traces.jsonl"),
            prev_rate,
        ),
        (
            "current midcons trace",
            midcons_run,
            mid_total,
            count_lines(midcons_run / "step_traces.jsonl"),
            mid_rate,
        ),
    ]
    route_table_rows = route_rows("previous local method", prev_summary) + route_rows(
        "current midcons",
        midcons_summary,
    )

    run_table = markdown_table(["Run", "Output", "Rows", "Trace rows", "Pass rate"], run_rows)
    route_table = markdown_table(
        [
            "Trace source",
            "Route",
            "Triggers",
            "Failed-long",
            "Short",
            "Current-pass risk",
            "Gate A",
            "Gate B",
            "Decision",
        ],
        route_table_rows,
    )

    terminal = "\n".join(
        [
            "TRACE LONG RESCUE FULL DIAGNOSTICS",
            run_table,
            "",
            route_table,
        ]
    )
    en = "\n".join(
        [
            "## LLaDA-Base Full Trace Long-Rescue Diagnostics",
            "",
            "Full trace collection completed for the previous local method and the current `midcons` method. These diagnostics use trace/decode dynamics for triggers and labels only for offline Gate A/B accounting; they are not a new SOTA claim.",
            "",
            run_table,
            "",
            "Offline route analysis:",
            "",
            route_table,
            "",
        ]
    )
    zh = en
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "docs_section.zh.md").write_text(zh, encoding="utf-8")
    (output_dir / "docs_section.en.md").write_text(en, encoding="utf-8")
    print(terminal)


if __name__ == "__main__":
    main()
