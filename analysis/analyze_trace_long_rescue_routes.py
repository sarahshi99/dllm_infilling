#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import (
    compute_gate_summary,
    extract_trace_features,
    join_results_and_traces,
    load_jsonl,
    metric,
)


RouteTrigger = Callable[[Mapping[str, Any]], bool]


def _maybe_num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any, default: float = 0.0) -> float:
    parsed = _maybe_num(value)
    return default if parsed is None else parsed


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def route1_trace_only_trigger(features: Mapping[str, Any]) -> bool:
    trace_steps = _num(features.get("trace_steps"))
    if trace_steps <= 0:
        return False

    selected_len = _maybe_num(features.get("selected_len"))
    if selected_len is not None and selected_len > 16:
        return False

    if features.get("last_stop_reason") == "no_remaining_masks":
        return False

    remaining_ratio = _maybe_num(features.get("final_remaining_mask_ratio"))
    remaining_masks = _maybe_num(features.get("final_remaining_masks"))
    unstable_remaining = bool(
        (remaining_ratio is not None and remaining_ratio >= 0.50)
        or (remaining_masks is not None and remaining_masks >= 4)
    )

    mean_gap = _maybe_num(features.get("final_mean_gap"))
    last_confidence = _maybe_num(features.get("last_mean_confidence"))
    final_top1 = _maybe_num(features.get("final_mean_top1"))
    uncertain_decode = bool(
        (mean_gap is not None and mean_gap <= 0.25)
        or (last_confidence is not None and last_confidence <= 0.45)
        or (final_top1 is not None and final_top1 <= 0.55)
    )
    return unstable_remaining and uncertain_decode


def route2_risk_controlled_trigger(features: Mapping[str, Any]) -> bool:
    if not route1_trace_only_trigger(features):
        return False

    selected_len = _maybe_num(features.get("selected_len"))
    if selected_len is not None and selected_len > 12:
        return False

    remaining_ratio = _maybe_num(features.get("final_remaining_mask_ratio"))
    remaining_masks = _maybe_num(features.get("final_remaining_masks"))
    strong_remaining = bool(
        (remaining_ratio is not None and remaining_ratio >= 0.65)
        or (remaining_masks is not None and remaining_masks >= 6)
    )

    mean_gap = _maybe_num(features.get("final_mean_gap"))
    last_confidence = _maybe_num(features.get("last_mean_confidence"))
    final_top1 = _maybe_num(features.get("final_mean_top1"))
    strong_uncertainty = bool(
        (mean_gap is not None and mean_gap <= 0.15)
        or (last_confidence is not None and last_confidence <= 0.40)
        or (final_top1 is not None and final_top1 <= 0.45)
    )
    return strong_remaining and strong_uncertainty


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


def build_records(
    result_rows: Iterable[Mapping[str, Any]],
    trace_rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    joined = join_results_and_traces(result_rows, trace_rows)
    records: List[Dict[str, Any]] = []
    for task_id in sorted(joined):
        payload = joined[task_id]
        features = extract_trace_features(payload["row"], payload["traces"])
        features["route1_trace_only_triggered"] = route1_trace_only_trigger(features)
        features["route2_risk_controlled_triggered"] = route2_risk_controlled_trigger(features)
        records.append(
            {
                "task_id": task_id,
                "row": payload["row"],
                "traces": payload["traces"],
                "features": features,
            }
        )
    return records


def _pessimistic_candidate_passed(
    records: Iterable[Mapping[str, Any]],
    trigger: RouteTrigger,
) -> Dict[str, bool]:
    candidate_passed: Dict[str, bool] = {}
    for record in records:
        task_id = str(record["task_id"])
        row = record["row"]
        features = record["features"]
        before = bool(metric(row, "passed", False))
        after = before
        if trigger(features):
            if bool(features.get("failed_long")):
                after = True
            elif before:
                after = False
        candidate_passed[task_id] = after
    return candidate_passed


def summarize_route(
    records: List[Mapping[str, Any]],
    *,
    route_key: str,
    route_name: str,
    trigger: RouteTrigger,
    min_failed_long: int,
    max_short: int,
) -> Dict[str, Any]:
    triggered = [record for record in records if trigger(record["features"])]
    failed_long_total = sum(1 for record in records if record["features"].get("failed_long"))
    true_long_trigger_count = sum(1 for record in triggered if record["features"].get("true_long"))
    failed_long_trigger_count = sum(1 for record in triggered if record["features"].get("failed_long"))
    short_risk_count = sum(1 for record in triggered if record["features"].get("short"))
    current_pass_risk_count = sum(1 for record in triggered if record["features"].get("passed"))
    trigger_count = len(triggered)
    base_rows = {str(record["task_id"]): record["row"] for record in records}
    gate_summary = compute_gate_summary(base_rows, _pessimistic_candidate_passed(records, trigger))
    continuation_passed = failed_long_trigger_count >= min_failed_long and short_risk_count <= max_short

    return {
        "route_key": route_key,
        "route_name": route_name,
        "trigger_count": trigger_count,
        "true_long_trigger_count": true_long_trigger_count,
        "failed_long_trigger_count": failed_long_trigger_count,
        "short_risk_count": short_risk_count,
        "current_pass_risk_count": current_pass_risk_count,
        "true_long_precision": _rate(true_long_trigger_count, trigger_count),
        "failed_long_precision": _rate(failed_long_trigger_count, trigger_count),
        "failed_long_recall": _rate(failed_long_trigger_count, failed_long_total),
        "short_risk_rate": _rate(short_risk_count, trigger_count),
        "current_pass_risk_rate": _rate(current_pass_risk_count, trigger_count),
        "gate_a_pessimistic_passed": gate_summary["gate_a_passed"],
        "gate_b_pessimistic_exploratory": gate_summary["gate_b_exploratory"],
        "gate_summary_pessimistic": gate_summary,
        "continuation_rule": {
            "min_failed_long": min_failed_long,
            "max_short": max_short,
            "passed": continuation_passed,
        },
        "decision": "continue" if continuation_passed else "stop",
    }


def summarize_route3(records: List[Mapping[str, Any]], prior_signal: bool) -> Dict[str, Any]:
    base_rows = {str(record["task_id"]): record["row"] for record in records}
    unchanged = {task_id: bool(metric(row, "passed", False)) for task_id, row in base_rows.items()}
    gate_summary = compute_gate_summary(base_rows, unchanged)
    decision = "pending_requires_multicanvas_trace" if prior_signal else "stop_no_trace_signal"
    return {
        "route_key": "route3_multi_canvas_rerank",
        "route_name": "Route 3 multi-canvas trace rerank",
        "trigger_count": 0,
        "true_long_trigger_count": 0,
        "failed_long_trigger_count": 0,
        "short_risk_count": 0,
        "current_pass_risk_count": 0,
        "true_long_precision": None,
        "failed_long_precision": None,
        "failed_long_recall": None,
        "short_risk_rate": None,
        "current_pass_risk_rate": None,
        "gate_a_pessimistic_passed": False,
        "gate_b_pessimistic_exploratory": False,
        "gate_summary_pessimistic": gate_summary,
        "continuation_rule": {
            "requires_route1_or_route2_signal": True,
            "passed": prior_signal,
        },
        "decision": decision,
        "note": "Single-canvas traces cannot evaluate multi-canvas reranking without another trace collection run.",
    }


def summarize_records(records: List[Mapping[str, Any]], *, trace_row_count: int) -> Dict[str, Any]:
    route1 = summarize_route(
        records,
        route_key="route1_trace_only",
        route_name="Route 1 trace-only detector",
        trigger=route1_trace_only_trigger,
        min_failed_long=10,
        max_short=5,
    )
    route2 = summarize_route(
        records,
        route_key="route2_risk_controlled",
        route_name="Route 2 risk-controlled rescue",
        trigger=route2_risk_controlled_trigger,
        min_failed_long=5,
        max_short=2,
    )
    prior_signal = bool(
        route1["continuation_rule"]["passed"] or route2["continuation_rule"]["passed"]
    )
    route3 = summarize_route3(records, prior_signal)
    routes = {
        "route1_trace_only": route1,
        "route2_risk_controlled": route2,
        "route3_multi_canvas_rerank": route3,
    }
    failed_long_total = sum(1 for record in records if record["features"].get("failed_long"))
    true_long_total = sum(1 for record in records if record["features"].get("true_long"))
    short_total = sum(1 for record in records if record["features"].get("short"))
    passed_total = sum(1 for record in records if record["features"].get("passed"))
    rows_with_traces = sum(1 for record in records if record["features"].get("trace_steps", 0) > 0)
    summary = {
        "rows": len(records),
        "trace_rows": trace_row_count,
        "rows_with_traces": rows_with_traces,
        "totals": {
            "passed": passed_total,
            "failed_long": failed_long_total,
            "true_long": true_long_total,
            "short": short_total,
        },
        "routes": routes,
        "route1_trigger_count": route1["trigger_count"],
        "route1_failed_long_count": route1["failed_long_trigger_count"],
        "route1_short_count": route1["short_risk_count"],
        "route2_trigger_count": route2["trigger_count"],
        "route2_failed_long_count": route2["failed_long_trigger_count"],
        "route2_short_count": route2["short_risk_count"],
        "route3_trigger_count": route3["trigger_count"],
        "route3_failed_long_count": route3["failed_long_trigger_count"],
        "route3_short_count": route3["short_risk_count"],
        "any_route_continues": any(
            route["decision"].startswith("continue") for route in routes.values()
        ),
    }
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: str | Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Trace Long Rescue Route Analysis",
        "",
        "Generated from full trace outputs. Route summaries are offline diagnostics, not GPU policy results.",
        "",
        "| Route | Triggers | Failed-long | Short | Current-pass risk | Gate A | Gate B | Decision |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for route in (summary.get("routes") or {}).values():
        lines.append(
            "| {name} | `{triggers}` | `{failed}` | `{short}` | `{risk}` | {gate_a} | {gate_b} | `{decision}` |".format(
                name=route["route_name"],
                triggers=route["trigger_count"],
                failed=route["failed_long_trigger_count"],
                short=route["short_risk_count"],
                risk=route["current_pass_risk_count"],
                gate_a=_fmt(route["gate_a_pessimistic_passed"]),
                gate_b=_fmt(route["gate_b_pessimistic_exploratory"]),
                decision=route["decision"],
            )
        )
    lines.append("")
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    traces = load_jsonl(args.traces)
    records = build_records(rows, traces)
    summary = summarize_records(records, trace_row_count=len(traces))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "features.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record["features"], ensure_ascii=False) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
