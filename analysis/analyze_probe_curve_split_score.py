#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis import analyze_probe_curve_long_signals as probe_audit


DEFAULT_RESULTS = probe_audit.DEFAULT_RESULTS
DEFAULT_OUTPUT_JSON = "docs/paper_agent/probe_curve_split_score_audit.json"
DEFAULT_OUTPUT_MD = "docs/paper_agent/probe_curve_split_score_audit.md"


def stable_fold(task_id: str, folds: int = 5) -> int:
    if folds <= 0:
        raise ValueError("folds must be positive")
    digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % folds


def _feature_value(record: Mapping[str, Any], name: str) -> float | None:
    value = (record.get("features") or {}).get(name)
    if value is None:
        return None
    return float(value)


def feature_names(records: Sequence[Mapping[str, Any]], min_coverage: float = 0.95) -> List[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for name, value in (record.get("features") or {}).items():
            if value is not None:
                counts[name] += 1
    required = max(1, int(len(records) * min_coverage))
    return sorted(name for name, count in counts.items() if count >= required)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 1.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    stdev = variance ** 0.5
    return stdev if stdev > 1e-12 else 1.0


def _is_risk_negative(record: Mapping[str, Any]) -> bool:
    return bool(record.get("short")) or bool(record.get("passed"))


def fit_linear_signal(records: Sequence[Mapping[str, Any]], features: Sequence[str]) -> Dict[str, Any]:
    means: Dict[str, float] = {}
    stdevs: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    positives = [record for record in records if record.get("failed_long")]
    risk_negatives = [record for record in records if _is_risk_negative(record)]

    for name in features:
        values = [_feature_value(record, name) for record in records]
        observed = [value for value in values if value is not None]
        mean = _mean(observed)
        stdev = _stdev(observed, mean)
        means[name] = mean
        stdevs[name] = stdev

        pos_values = [
            ((_feature_value(record, name) or mean) - mean) / stdev
            for record in positives
        ]
        neg_values = [
            ((_feature_value(record, name) or mean) - mean) / stdev
            for record in risk_negatives
        ]
        weights[name] = _mean(pos_values) - _mean(neg_values)

    return {
        "features": list(features),
        "means": means,
        "stdevs": stdevs,
        "weights": weights,
        "positive_count": len(positives),
        "risk_negative_count": len(risk_negatives),
    }


def score_record(record: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    score = 0.0
    for name in model["features"]:
        mean = float(model["means"][name])
        stdev = float(model["stdevs"][name])
        value = _feature_value(record, name)
        standardized = ((mean if value is None else value) - mean) / stdev
        score += float(model["weights"][name]) * standardized
    return score


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _objective(result: Mapping[str, Any]) -> float:
    precision = result["true_long_precision"] or 0.0
    recall = result["failed_long_recall"] or 0.0
    short_risk = result["short_risk_rate"] or 0.0
    pass_risk = result["current_pass_risk_rate"] or 0.0
    return precision * recall * (1.0 - short_risk) * (1.0 - pass_risk)


def evaluate_trigger_flags(records: Sequence[Mapping[str, Any]], triggers: Sequence[bool]) -> Dict[str, Any]:
    if len(records) != len(triggers):
        raise ValueError("records and triggers must have the same length")
    triggered = [record for record, trigger in zip(records, triggers) if trigger]
    failed_long_total = sum(1 for record in records if record["failed_long"])
    true_long_count = sum(1 for record in triggered if record["true_long"])
    failed_long_trigger_count = sum(1 for record in triggered if record["failed_long"])
    short_risk_count = sum(1 for record in triggered if record["short"])
    current_pass_risk_count = sum(1 for record in triggered if record["passed"])
    trigger_count = len(triggered)
    result = {
        "trigger_count": trigger_count,
        "true_long_count": true_long_count,
        "failed_long_trigger_count": failed_long_trigger_count,
        "short_risk_count": short_risk_count,
        "current_pass_risk_count": current_pass_risk_count,
        "true_long_precision": _rate(true_long_count, trigger_count),
        "failed_long_recall": _rate(failed_long_trigger_count, failed_long_total),
        "short_risk_rate": _rate(short_risk_count, trigger_count),
        "current_pass_risk_rate": _rate(current_pass_risk_count, trigger_count),
    }
    result["score"] = _objective(result)
    return result


def evaluate_scored_records(
    records: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    threshold: float,
) -> Dict[str, Any]:
    return evaluate_trigger_flags(records, [score >= threshold for score in scores])


def _candidate_thresholds(scores: Sequence[float]) -> List[float]:
    return sorted(set(float(score) for score in scores))


def _passes_strict_gate(
    result: Mapping[str, Any],
    *,
    min_failed_long_triggers: int,
    max_short_risk: float,
    min_precision: float,
) -> bool:
    return (
        result["failed_long_trigger_count"] >= min_failed_long_triggers
        and (result["true_long_precision"] or 0.0) >= min_precision
        and (result["short_risk_rate"] or 0.0) <= max_short_risk
    )


def select_threshold(
    records: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    min_failed_long_triggers: int = 8,
    max_short_risk: float = 0.05,
    min_precision: float = 0.60,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for threshold in _candidate_thresholds(scores):
        result = evaluate_scored_records(records, scores, threshold)
        if result["trigger_count"] == 0:
            continue
        result["threshold"] = threshold
        result["strict_train_pass"] = _passes_strict_gate(
            result,
            min_failed_long_triggers=min_failed_long_triggers,
            max_short_risk=max_short_risk,
            min_precision=min_precision,
        )
        candidates.append(result)

    if not candidates:
        raise ValueError("No threshold candidates available")

    strict = [candidate for candidate in candidates if candidate["strict_train_pass"]]
    if strict:
        return max(strict, key=lambda item: (item["score"], item["failed_long_trigger_count"], -item["trigger_count"]))
    return max(candidates, key=lambda item: (item["score"], item["failed_long_trigger_count"], -item["short_risk_count"]))


def _top_weights(model: Mapping[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    items = sorted(
        model["weights"].items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    return [{"feature": name, "weight": weight} for name, weight in items[:limit]]


def cross_validate(records: Sequence[Mapping[str, Any]], folds: int = 5) -> Dict[str, Any]:
    heldout_records: List[Mapping[str, Any]] = []
    heldout_triggers: List[bool] = []
    fold_results: List[Dict[str, Any]] = []

    for fold in range(folds):
        train = [record for record in records if stable_fold(record["task_id"], folds) != fold]
        heldout = [record for record in records if stable_fold(record["task_id"], folds) == fold]
        names = feature_names(train)
        model = fit_linear_signal(train, names)
        train_scores = [score_record(record, model) for record in train]
        threshold = select_threshold(train, train_scores)
        heldout_scores = [score_record(record, model) for record in heldout]
        triggers = [score >= threshold["threshold"] for score in heldout_scores]
        heldout_eval = evaluate_trigger_flags(heldout, triggers)
        train_eval = evaluate_scored_records(train, train_scores, threshold["threshold"])
        heldout_records.extend(heldout)
        heldout_triggers.extend(triggers)
        fold_results.append(
            {
                "fold": fold,
                "train_size": len(train),
                "heldout_size": len(heldout),
                "feature_count": len(names),
                "threshold": threshold["threshold"],
                "strict_train_pass": threshold["strict_train_pass"],
                "train_eval": train_eval,
                "heldout_eval": heldout_eval,
                "top_weights": _top_weights(model),
            }
        )

    aggregate = evaluate_trigger_flags(heldout_records, heldout_triggers)
    aggregate["strict_heldout_pass"] = _passes_strict_gate(
        aggregate,
        min_failed_long_triggers=10,
        max_short_risk=0.05,
        min_precision=0.60,
    )
    return {
        "folds": folds,
        "aggregate_heldout": aggregate,
        "fold_results": fold_results,
    }


def build_audit(rows: Sequence[Mapping[str, Any]], folds: int = 5) -> Dict[str, Any]:
    records = [probe_audit.build_record(row) for row in rows]
    names = feature_names(records)
    return {
        "audit_version": 1,
        "split_discipline": "deterministic_sha256_task_id_folds_train_thresholds_only",
        "rows": len(records),
        "feature_count": len(names),
        "features": names,
        "label_counts": {
            "true_long": sum(1 for record in records if record["true_long"]),
            "failed_long": sum(1 for record in records if record["failed_long"]),
            "short": sum(1 for record in records if record["short"]),
            "passed": sum(1 for record in records if record["passed"]),
        },
        "cross_validation": cross_validate(records, folds=folds),
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
    cv = payload["cross_validation"]
    aggregate = cv["aggregate_heldout"]
    lines = [
        "# Probe-Curve Strict-Split Score Audit",
        "",
        "Generated from existing local `results.jsonl`; no GPU job was launched.",
        "",
        "## Split Discipline",
        "",
        f"- split_discipline: `{payload['split_discipline']}`",
        f"- folds: `{cv['folds']}`",
        "- thresholds are selected on train folds only and evaluated on held-out folds.",
        "",
        "## Labels And Features",
        "",
        f"- rows: `{payload['rows']}`",
        f"- feature_count: `{payload['feature_count']}`",
    ]
    for key, value in payload["label_counts"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Aggregate Held-Out Result",
            "",
            f"- strict_heldout_pass: `{aggregate['strict_heldout_pass']}`",
            f"- trigger_count: `{aggregate['trigger_count']}`",
            f"- true_long_precision: `{pct(aggregate['true_long_precision'])}`",
            f"- failed_long_recall: `{pct(aggregate['failed_long_recall'])}`",
            f"- short_risk_rate: `{pct(aggregate['short_risk_rate'])}`",
            f"- current_pass_risk_rate: `{pct(aggregate['current_pass_risk_rate'])}`",
            "",
            "## Fold Results",
            "",
            "| Fold | Train | Held-Out | Strict Train Pass | Threshold | Held-Out Triggers | Precision | Failed-Long Recall | Short Risk | Current-Pass Risk |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in cv["fold_results"]:
        heldout = fold["heldout_eval"]
        lines.append(
            f"| {fold['fold']} | {fold['train_size']} | {fold['heldout_size']} | "
            f"`{fold['strict_train_pass']}` | {fold['threshold']:.6g} | "
            f"{heldout['trigger_count']} | {pct(heldout['true_long_precision'])} | "
            f"{pct(heldout['failed_long_recall'])} | {pct(heldout['short_risk_rate'])} | "
            f"{pct(heldout['current_pass_risk_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit tests whether a simple multivariate probe-curve score has an offline safety signal "
            "under strict split discipline. A positive result requires the aggregate held-out score to satisfy "
            "the same safety posture used by the paper-agent plan: enough failed-long triggers, at least "
            "`60%` true-long precision, and at most `5%` short-risk. Passing this diagnostic would only justify "
            "a later smoke plan; it would not by itself justify a full GPU run.",
            "",
        ]
    )
    if aggregate["strict_heldout_pass"]:
        lines.append(
            "This run passes the offline gate and should be treated only as a candidate for a later smoke plan."
        )
    else:
        lines.append(
            "This run does not pass that gate: held-out short-risk is "
            f"`{pct(aggregate['short_risk_rate'])}`, well above `5%`, and true-long precision is "
            f"`{pct(aggregate['true_long_precision'])}`. The result is therefore a negative diagnostic "
            "result against launching a GPU smoke run from the current probe-curve fields alone."
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict-split probe-curve scoring diagnostic.")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = probe_audit.load_jsonl(args.results)
    payload = build_audit(rows, folds=args.folds)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    aggregate = payload["cross_validation"]["aggregate_heldout"]
    print(
        f"wrote {args.output_json} and {args.output_md}; "
        f"strict_heldout_pass={aggregate['strict_heldout_pass']} "
        f"heldout_triggers={aggregate['trigger_count']} "
        f"short_risk={pct(aggregate['short_risk_rate'])}"
    )


if __name__ == "__main__":
    main()
