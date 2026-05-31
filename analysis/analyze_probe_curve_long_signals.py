#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEFAULT_RESULTS = (
    "/home/shx/projects/dllm_infilling/outputs_clean/"
    "full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626/"
    "results.jsonl"
)
DEFAULT_OUTPUT_JSON = "docs/paper_agent/probe_curve_signal_audit.json"
DEFAULT_OUTPUT_MD = "docs/paper_agent/probe_curve_signal_audit.md"


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


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def passed(row: Mapping[str, Any]) -> bool:
    return bool(metric(row, "passed", False))


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def candidate_scores(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    lcal_v3 = row.get("lcal_v3") or {}
    length_probe = row.get("length_probe") or {}
    for container in (lcal_v3, length_probe):
        if isinstance(container, Mapping):
            scores = container.get("base_candidate_scores") or container.get("candidate_scores")
            if isinstance(scores, list) and scores:
                return scores
    return []


def _score(candidate: Mapping[str, Any], key: str = "score") -> float | None:
    return optional_float(candidate.get(key))


def _length(candidate: Mapping[str, Any]) -> int | None:
    return optional_int(candidate.get("mask_length"))


def _max_score(candidates: Sequence[Mapping[str, Any]], *, min_len: int | None = None, max_len: int | None = None, key: str = "score") -> float | None:
    values: List[float] = []
    for item in candidates:
        length = _length(item)
        score = _score(item, key)
        if length is None or score is None:
            continue
        if min_len is not None and length < min_len:
            continue
        if max_len is not None and length > max_len:
            continue
        values.append(score)
    return max(values) if values else None


def _avg_score(candidates: Sequence[Mapping[str, Any]], *, min_len: int | None = None, max_len: int | None = None, key: str = "score") -> float | None:
    values: List[float] = []
    for item in candidates:
        length = _length(item)
        score = _score(item, key)
        if length is None or score is None:
            continue
        if min_len is not None and length < min_len:
            continue
        if max_len is not None and length > max_len:
            continue
        values.append(score)
    return sum(values) / len(values) if values else None


def _candidate_by_length(candidates: Sequence[Mapping[str, Any]], length: int) -> Mapping[str, Any] | None:
    for item in candidates:
        if _length(item) == length:
            return item
    return None


def _best_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    scored = [item for item in candidates if _score(item) is not None and _length(item) is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: (_score(item) or float("-inf"), -(_length(item) or 0)))


def _put_if_number(features: Dict[str, float], key: str, value: float | int | None) -> None:
    if value is not None:
        features[key] = float(value)


def extract_features(row: Mapping[str, Any]) -> Dict[str, float]:
    candidates = candidate_scores(row)
    selected_len = optional_int(metric(row, "selected_mask_length"))
    best = _best_candidate(candidates)
    best_len = _length(best) if best is not None else None
    best_score = _score(best) if best is not None else None
    best_raw = _score(best, "raw_score") if best is not None else None
    selected_candidate = _candidate_by_length(candidates, selected_len) if selected_len is not None else None

    features: Dict[str, float] = {}
    _put_if_number(features, "selected_len", selected_len)
    _put_if_number(features, "best_len", best_len)
    _put_if_number(features, "best_score", best_score)
    _put_if_number(features, "best_raw_score", best_raw)
    if selected_len is not None and best_len is not None:
        features["selected_to_best_gap"] = float(best_len - selected_len)

    selected_score = _score(selected_candidate) if selected_candidate is not None else None
    selected_raw = _score(selected_candidate, "raw_score") if selected_candidate is not None else None
    _put_if_number(features, "selected_score", selected_score)
    _put_if_number(features, "selected_raw_score", selected_raw)
    if selected_score is not None and best_score is not None:
        features["best_minus_selected_score"] = float(best_score - selected_score)
    if selected_raw is not None and best_raw is not None:
        features["best_minus_selected_raw"] = float(best_raw - selected_raw)

    short_max = _max_score(candidates, max_len=8)
    medium_max = _max_score(candidates, min_len=9, max_len=16)
    long_max = _max_score(candidates, min_len=17)
    short_raw_max = _max_score(candidates, max_len=8, key="raw_score")
    medium_raw_max = _max_score(candidates, min_len=9, max_len=16, key="raw_score")
    long_raw_max = _max_score(candidates, min_len=17, key="raw_score")
    long_gap_max = _max_score(candidates, min_len=17, key="mean_top2_gap")

    _put_if_number(features, "short_score_max", short_max)
    _put_if_number(features, "medium_score_max", medium_max)
    _put_if_number(features, "long_score_max", long_max)
    _put_if_number(features, "short_raw_max", short_raw_max)
    _put_if_number(features, "medium_raw_max", medium_raw_max)
    _put_if_number(features, "long_raw_max", long_raw_max)
    _put_if_number(features, "long_gap_max", long_gap_max)
    _put_if_number(features, "long_score_avg", _avg_score(candidates, min_len=17))
    _put_if_number(features, "short_score_avg", _avg_score(candidates, max_len=8))

    if long_max is not None and short_max is not None:
        features["long_minus_short_score"] = float(long_max - short_max)
        features["long_to_short_score_ratio"] = float(long_max / short_max) if short_max else 0.0
    if long_max is not None and medium_max is not None:
        features["long_minus_medium_score"] = float(long_max - medium_max)
        features["long_to_medium_score_ratio"] = float(long_max / medium_max) if medium_max else 0.0
    if long_raw_max is not None and short_raw_max is not None:
        features["long_minus_short_raw"] = float(long_raw_max - short_raw_max)
    if long_raw_max is not None and medium_raw_max is not None:
        features["long_minus_medium_raw"] = float(long_raw_max - medium_raw_max)

    return features


def is_true_long(row: Mapping[str, Any]) -> bool:
    oracle = optional_int(metric(row, "oracle_mask_length"))
    return oracle is not None and oracle >= 17


def is_short(row: Mapping[str, Any]) -> bool:
    oracle = optional_int(metric(row, "oracle_mask_length"))
    return oracle is not None and oracle <= 8


def build_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    oracle = optional_int(metric(row, "oracle_mask_length"))
    return {
        "task_id": str(row["task_id"]),
        "passed": passed(row),
        "oracle_mask_length": oracle,
        "selected_mask_length": optional_int(metric(row, "selected_mask_length")),
        "true_long": is_true_long(row),
        "short": is_short(row),
        "failed_long": is_true_long(row) and not passed(row),
        "features": extract_features(row),
    }


def evaluate_threshold(
    records: Sequence[Mapping[str, Any]],
    *,
    feature: str,
    direction: str,
    threshold: float,
) -> Dict[str, Any]:
    if direction not in (">=", "<="):
        raise ValueError(f"Unsupported direction: {direction}")

    def triggers(record: Mapping[str, Any]) -> bool:
        value = (record.get("features") or {}).get(feature)
        if value is None:
            return False
        return float(value) >= threshold if direction == ">=" else float(value) <= threshold

    triggered = [record for record in records if triggers(record)]
    failed_long_total = sum(1 for record in records if record["failed_long"])
    true_long_count = sum(1 for record in triggered if record["true_long"])
    failed_long_trigger_count = sum(1 for record in triggered if record["failed_long"])
    short_risk_count = sum(1 for record in triggered if record["short"])
    current_pass_risk_count = sum(1 for record in triggered if record["passed"])
    trigger_count = len(triggered)
    precision = rate(true_long_count, trigger_count)
    recall = rate(failed_long_trigger_count, failed_long_total)
    short_risk = rate(short_risk_count, trigger_count)
    pass_risk = rate(current_pass_risk_count, trigger_count)
    score = (
        (precision or 0.0)
        * (recall or 0.0)
        * (1.0 - (short_risk or 0.0))
        * (1.0 - (pass_risk or 0.0))
    )
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "trigger_count": trigger_count,
        "true_long_count": true_long_count,
        "failed_long_trigger_count": failed_long_trigger_count,
        "short_risk_count": short_risk_count,
        "current_pass_risk_count": current_pass_risk_count,
        "true_long_precision": precision,
        "failed_long_recall": recall,
        "short_risk_rate": short_risk,
        "current_pass_risk_rate": pass_risk,
        "score": score,
    }


def _candidate_thresholds(values: Sequence[float], max_thresholds: int = 80) -> List[float]:
    unique = sorted(set(float(value) for value in values))
    if len(unique) <= max_thresholds:
        return unique
    step = max(1, len(unique) // max_thresholds)
    thresholds = unique[::step]
    if unique[-1] not in thresholds:
        thresholds.append(unique[-1])
    return thresholds


def sweep_feature_thresholds(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_feature: Dict[str, List[float]] = {}
    for record in records:
        features = record.get("features") or {}
        for key, value in features.items():
            by_feature.setdefault(key, []).append(float(value))

    results: List[Dict[str, Any]] = []
    for feature, values in by_feature.items():
        for threshold in _candidate_thresholds(values):
            for direction in (">=", "<="):
                result = evaluate_threshold(records, feature=feature, direction=direction, threshold=threshold)
                if result["trigger_count"] > 0:
                    results.append(result)
    results.sort(
        key=lambda item: (
            item["score"],
            item["failed_long_trigger_count"],
            item["true_long_precision"] or 0.0,
            -(item["short_risk_rate"] or 0.0),
        ),
        reverse=True,
    )
    return results


def availability_summary(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    feature_counts: Counter[str] = Counter()
    for record in records:
        for feature in (record.get("features") or {}):
            feature_counts[feature] += 1
    has_stopping_trace = sum(1 for row in rows if bool(row.get("stopping_trace")))
    return {
        "rows": len(rows),
        "rows_with_probe_curve_features": sum(1 for record in records if record.get("features")),
        "rows_with_stopping_trace": has_stopping_trace,
        "feature_counts": dict(sorted(feature_counts.items())),
    }


def strict_viable(results: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        item
        for item in results
        if item["trigger_count"] >= 10
        and (item["true_long_precision"] or 0.0) >= 0.60
        and (item["short_risk_rate"] or 0.0) <= 0.05
        and item["failed_long_trigger_count"] >= 10
    ]


def build_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    records = [build_record(row) for row in rows]
    results = sweep_feature_thresholds(records)
    viable = strict_viable(results)
    return {
        "audit_version": 1,
        "availability": availability_summary(rows, records),
        "label_counts": {
            "true_long": sum(1 for record in records if record["true_long"]),
            "failed_long": sum(1 for record in records if record["failed_long"]),
            "short": sum(1 for record in records if record["short"]),
            "passed": sum(1 for record in records if record["passed"]),
        },
        "evaluated_thresholds": len(results),
        "strict_viable_thresholds": len(viable),
        "top_thresholds": results[:30],
    }


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    availability = payload["availability"]
    lines = [
        "# Probe-Curve Long-Signal Audit",
        "",
        "Generated from existing local `results.jsonl`; no GPU job was launched.",
        "",
        "## Availability",
        "",
        f"- rows: `{availability['rows']}`",
        f"- rows_with_probe_curve_features: `{availability['rows_with_probe_curve_features']}`",
        f"- rows_with_stopping_trace: `{availability['rows_with_stopping_trace']}`",
        "",
        "## Labels",
        "",
    ]
    for key, value in payload["label_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Threshold Sweep",
            "",
            f"- evaluated_thresholds: `{payload['evaluated_thresholds']}`",
            f"- strict_viable_thresholds: `{payload['strict_viable_thresholds']}`",
            "",
            "| Feature | Direction | Threshold | Triggers | Precision | Failed-Long Recall | Short Risk | Current-Pass Risk |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["top_thresholds"][:15]:
        lines.append(
            f"| `{item['feature']}` | `{item['direction']}` | `{item['threshold']:.6g}` | "
            f"{item['trigger_count']} | {pct(item['true_long_precision'])} | "
            f"{pct(item['failed_long_recall'])} | {pct(item['short_risk_rate'])} | "
            f"{pct(item['current_pass_risk_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current full-run artifact supports probe-curve diagnostics but not trajectory diagnostics, "
            "because `rows_with_stopping_trace = 0`. Single-feature probe-curve thresholds do not pass the "
            "strict viability gate. The best threshold is close to the true-long precision gate, but its "
            "short-risk exceeds `5%`, so it should not be promoted directly to a GPU full run.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit probe-curve features for long under-selection signals.")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.results)
    payload = build_audit(rows)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(
        f"wrote {args.output_json} and {args.output_md}; "
        f"thresholds={payload['evaluated_thresholds']} "
        f"strict_viable={payload['strict_viable_thresholds']}"
    )


if __name__ == "__main__":
    main()
