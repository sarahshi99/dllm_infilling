from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "task_id" not in row:
                raise ValueError(f"Missing task_id in {path} line {line_no}")
            rows.append(row)
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


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _trace_step(trace: Mapping[str, Any]) -> int:
    step = _optional_int(trace.get("step"))
    return -1 if step is None else step


def join_results_and_traces(
    result_rows: Iterable[Mapping[str, Any]],
    trace_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    joined: Dict[str, Dict[str, Any]] = {}
    for row in result_rows:
        task_id = row.get("task_id")
        if task_id is None:
            continue
        joined[str(task_id)] = {"row": dict(row), "traces": []}

    grouped: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for trace in trace_rows:
        task_id = trace.get("task_id")
        if task_id is None:
            continue
        grouped[str(task_id)].append(dict(trace))

    for task_id, traces in grouped.items():
        if task_id not in joined:
            continue
        joined[task_id]["traces"] = sorted(traces, key=_trace_step)
    return joined


def _trace_remaining_ratio(trace: Mapping[str, Any]) -> Optional[float]:
    decision = trace.get("stop_decision") or {}
    if isinstance(decision, Mapping):
        ratio = _optional_float(decision.get("remaining_mask_ratio"))
        if ratio is not None:
            return ratio

    after = _optional_float(trace.get("remaining_masks_after_update"))
    target = _optional_float(trace.get("target_masks"))
    if after is not None and target and target > 0:
        return after / target

    before = _optional_float(trace.get("remaining_masks_before_update"))
    if after is not None and before and before > 0:
        return after / before
    return None


def _trace_decision(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    decision = trace.get("stop_decision") or {}
    if isinstance(decision, Mapping):
        return decision
    return {}


def extract_trace_features(row: Mapping[str, Any], traces: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered_traces = sorted((dict(trace) for trace in traces), key=_trace_step)
    oracle_len = _optional_int(metric(row, "oracle_mask_length"))
    selected_len = _optional_int(metric(row, "selected_mask_length", metric(row, "mask_length")))
    selected_minus_oracle = _optional_int(metric(row, "selected_minus_oracle_length"))
    if selected_minus_oracle is None and selected_len is not None and oracle_len is not None:
        selected_minus_oracle = selected_len - oracle_len

    passed = bool(metric(row, "passed", False))
    remaining_ratios = [
        ratio for ratio in (_trace_remaining_ratio(trace) for trace in ordered_traces) if ratio is not None
    ]
    confidences = [
        confidence
        for confidence in (_optional_float(trace.get("mean_confidence")) for trace in ordered_traces)
        if confidence is not None
    ]
    remaining_after = [
        remaining
        for remaining in (_optional_int(trace.get("remaining_masks_after_update")) for trace in ordered_traces)
        if remaining is not None
    ]
    last_trace = ordered_traces[-1] if ordered_traces else {}
    last_decision = _trace_decision(last_trace)

    final_remaining_masks = _optional_int(metric(row, "remaining_masks_at_stop_or_final"))
    if final_remaining_masks is None and remaining_after:
        final_remaining_masks = remaining_after[-1]
    final_remaining_mask_ratio = _first_number(
        metric(row, "remaining_mask_ratio_at_stop_or_final"),
        remaining_ratios[-1] if remaining_ratios else None,
    )
    final_mean_gap = _first_number(metric(row, "mean_gap_at_stop_or_final"), last_decision.get("mean_gap"))
    final_mean_top1 = _first_number(metric(row, "mean_top1_at_stop_or_final"), last_decision.get("mean_top1"))

    return {
        "task_id": row.get("task_id"),
        "passed": passed,
        "oracle_len": oracle_len,
        "oracle_bucket": oracle_bucket(oracle_len),
        "selected_len": selected_len,
        "selected_minus_oracle": selected_minus_oracle,
        "true_long": bool(oracle_len is not None and oracle_len >= 17),
        "failed_long": bool((oracle_len is not None and oracle_len >= 17) and not passed),
        "short": bool(oracle_len is not None and oracle_len <= 8),
        "trace_steps": len(ordered_traces),
        "final_remaining_masks": final_remaining_masks,
        "final_remaining_mask_ratio": final_remaining_mask_ratio,
        "final_mean_gap": final_mean_gap,
        "final_mean_top1": final_mean_top1,
        "max_remaining_mask_ratio": max(remaining_ratios) if remaining_ratios else None,
        "last_remaining_mask_ratio": remaining_ratios[-1] if remaining_ratios else None,
        "min_mean_confidence": min(confidences) if confidences else None,
        "last_mean_confidence": confidences[-1] if confidences else None,
        "last_stop_reason": str(last_decision.get("reason")) if last_decision.get("reason") is not None else None,
    }


def compute_gate_summary(
    base_rows: Mapping[str, Mapping[str, Any]],
    candidate_passed: Mapping[str, bool],
) -> Dict[str, Any]:
    common_task_ids = sorted(set(base_rows).intersection(candidate_passed))
    overall_delta = 0
    short_delta = 0
    long_delta = 0
    bucket_delta: Counter[str] = Counter()
    bucket_total: Counter[str] = Counter()

    for task_id in common_task_ids:
        row = base_rows[task_id]
        before = bool(metric(row, "passed", False))
        after = bool(candidate_passed[task_id])
        oracle_len = _optional_int(metric(row, "oracle_mask_length"))
        bucket = oracle_bucket(oracle_len)
        bucket_total[bucket] += 1

        if before == after:
            continue
        delta = 1 if after else -1
        overall_delta += delta
        bucket_delta[bucket] += delta
        if oracle_len is not None and oracle_len <= 8:
            short_delta += delta
        if oracle_len is not None and oracle_len >= 17:
            long_delta += delta

    short_net_loss = max(0, -short_delta)
    long_net_gain = max(0, long_delta)
    return {
        "common": len(common_task_ids),
        "overall_delta": overall_delta,
        "short_delta": short_delta,
        "short_net_loss": short_net_loss,
        "long_delta": long_delta,
        "long_net_gain": long_net_gain,
        "bucket_delta": dict(bucket_delta),
        "bucket_total": dict(bucket_total),
        "gate_a_passed": bool(long_net_gain > 0 and overall_delta >= 0 and short_net_loss <= 2),
        "gate_b_exploratory": bool(long_net_gain > 0 and overall_delta >= -5 and short_net_loss <= 5),
    }
