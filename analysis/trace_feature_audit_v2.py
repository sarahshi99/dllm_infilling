#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import join_results_and_traces, load_jsonl, metric


JsonDict = Dict[str, Any]


def _num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    parsed = _num(value)
    if parsed is None:
        return None
    return int(parsed)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _trace_step(trace: Mapping[str, Any]) -> int:
    step = _int(trace.get("step"))
    return -1 if step is None else step


def _decision(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    decision = trace.get("stop_decision") or {}
    if isinstance(decision, Mapping):
        return decision
    return {}


def _series_from_trace(traces: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for trace in traces:
        parsed = _num(trace.get(key))
        if parsed is None:
            parsed = _num(_decision(trace).get(key))
        if parsed is not None:
            values.append(parsed)
    return values


def _remaining_series(traces: Sequence[Mapping[str, Any]]) -> List[float]:
    return [
        parsed
        for parsed in (_num(trace.get("remaining_masks_after_update")) for trace in traces)
        if parsed is not None
    ]


def _slope(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return (values[-1] - values[0]) / float(len(values) - 1)


def _last_window(values: Sequence[float], size: int = 3) -> List[float]:
    if not values:
        return []
    return list(values[-size:])


def _count_equal_transitions(values: Sequence[float]) -> int:
    count = 0
    for left, right in zip(values, values[1:]):
        if abs(right - left) <= 1e-9:
            count += 1
    return count


def _max_plateau(values: Sequence[float]) -> int:
    best = 0
    current = 0
    for left, right in zip(values, values[1:]):
        if abs(right - left) <= 1e-9:
            current += 1
        else:
            best = max(best, current)
            current = 0
    return max(best, current)


def _last_decrease_index(values: Sequence[float]) -> Optional[int]:
    last: Optional[int] = None
    for index, (left, right) in enumerate(zip(values, values[1:]), start=1):
        if right < left:
            last = index
    return last


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(median(values))


def compute_shape_features(traces: Iterable[Mapping[str, Any]]) -> JsonDict:
    ordered = sorted((dict(trace) for trace in traces), key=_trace_step)
    remaining = _remaining_series(ordered)
    confidence = _series_from_trace(ordered, "mean_confidence")
    gap = _series_from_trace(ordered, "mean_gap")
    top1 = _series_from_trace(ordered, "mean_top1")
    late_remaining = _last_window(remaining, size=max(2, min(4, len(remaining))))
    first_remaining = remaining[0] if remaining else None
    remaining_auc = sum(remaining) if remaining else None
    remaining_auc_norm = (
        _safe_div(remaining_auc, float(len(remaining)) * max(abs(first_remaining), 1.0))
        if remaining_auc is not None and first_remaining is not None
        else None
    )
    last_decrease = _last_decrease_index(remaining)

    return {
        "trace_steps": len(ordered),
        "remaining_first": first_remaining,
        "remaining_last": remaining[-1] if remaining else None,
        "remaining_min": min(remaining) if remaining else None,
        "remaining_max": max(remaining) if remaining else None,
        "remaining_median": _median_or_none(remaining),
        "remaining_slope": _slope(remaining),
        "remaining_late_slope": _slope(late_remaining),
        "remaining_auc_norm": remaining_auc_norm,
        "remaining_equal_transition_frac": _safe_div(
            float(_count_equal_transitions(remaining)), float(max(len(remaining) - 1, 1))
        ),
        "late_remaining_plateau_steps": _count_equal_transitions(late_remaining),
        "max_remaining_plateau_steps": _max_plateau(remaining),
        "last_remaining_decrease_step": last_decrease,
        "confidence_first": confidence[0] if confidence else None,
        "confidence_last": confidence[-1] if confidence else None,
        "confidence_min": min(confidence) if confidence else None,
        "confidence_max": max(confidence) if confidence else None,
        "confidence_median": _median_or_none(confidence),
        "confidence_slope": _slope(confidence),
        "confidence_late_slope": _slope(_last_window(confidence)),
        "gap_last": gap[-1] if gap else None,
        "gap_min": min(gap) if gap else None,
        "gap_median": _median_or_none(gap),
        "top1_last": top1[-1] if top1 else None,
        "top1_min": min(top1) if top1 else None,
        "top1_median": _median_or_none(top1),
    }


def fold_id(task_id: str, *, folds: int) -> int:
    if folds <= 1:
        return 0
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % folds


def _oracle_bucket(oracle_len: Optional[int]) -> str:
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


def build_feature_record(row: Mapping[str, Any], traces: Iterable[Mapping[str, Any]], *, source_name: str) -> JsonDict:
    ordered = sorted((dict(trace) for trace in traces), key=_trace_step)
    shape = compute_shape_features(ordered)
    oracle_len = _int(metric(row, "oracle_mask_length"))
    selected_len = _int(metric(row, "selected_mask_length", metric(row, "mask_length")))
    passed = bool(metric(row, "passed", False))
    last_decision = _decision(ordered[-1]) if ordered else {}
    stop_reason = last_decision.get("reason")
    final_remaining = shape.get("remaining_last")
    final_remaining_ratio = (
        _safe_div(float(final_remaining), float(max(selected_len, 1)))
        if final_remaining is not None and selected_len is not None
        else None
    )

    features: JsonDict = {
        "selected_len": selected_len,
        "stop_reason": str(stop_reason) if stop_reason is not None else "missing",
        "final_remaining_ratio_by_selected": final_remaining_ratio,
        "long_score_max": _num(metric(row, "long_score_max")),
        "best_long_len": _num(metric(row, "best_long_len")),
        "best_len": _num(metric(row, "best_len")),
        "base_selected_length": _num(metric(row, "base_selected_length")),
        "final_selected_length": _num(metric(row, "final_selected_length")),
    }
    features.update(shape)

    labels = {
        "passed": passed,
        "oracle_len": oracle_len,
        "oracle_bucket": _oracle_bucket(oracle_len),
        "failed_long": bool(oracle_len is not None and oracle_len >= 17 and not passed),
        "true_long": bool(oracle_len is not None and oracle_len >= 17),
        "short_risk": bool(oracle_len is not None and oracle_len <= 8),
        "current_pass_risk": passed,
    }
    return {
        "task_id": str(row.get("task_id")),
        "source": source_name,
        "features": features,
        "labels": labels,
    }


@dataclass(frozen=True)
class CandidateRule:
    name: str
    clauses: List[List[Any]]
    family: str = "manual"

    def matches(self, record: Mapping[str, Any]) -> bool:
        features = record.get("features") or {}
        for feature_name, op, threshold in self.clauses:
            value = features.get(str(feature_name))
            if isinstance(threshold, str):
                matched = str(value) == threshold
            else:
                parsed = _num(value)
                if parsed is None:
                    return False
                if op == ">=":
                    matched = parsed >= float(threshold)
                elif op == "<=":
                    matched = parsed <= float(threshold)
                else:
                    raise ValueError(f"Unsupported operator: {op}")
            if not matched:
                return False
        return True


def evaluate_rule(rule: CandidateRule, records: Sequence[Mapping[str, Any]]) -> JsonDict:
    triggered = [record for record in records if rule.matches(record)]
    trigger_count = len(triggered)
    failed_long_count = sum(1 for record in triggered if record["labels"].get("failed_long"))
    true_long_count = sum(1 for record in triggered if record["labels"].get("true_long"))
    short_risk_count = sum(1 for record in triggered if record["labels"].get("short_risk"))
    current_pass_risk_count = sum(1 for record in triggered if record["labels"].get("current_pass_risk"))
    total_failed_long = sum(1 for record in records if record["labels"].get("failed_long"))
    return {
        "name": rule.name,
        "family": rule.family,
        "clauses": rule.clauses,
        "trigger_count": trigger_count,
        "failed_long_count": failed_long_count,
        "true_long_count": true_long_count,
        "short_risk_count": short_risk_count,
        "current_pass_risk_count": current_pass_risk_count,
        "failed_long_recall": _safe_div(float(failed_long_count), float(total_failed_long)),
        "true_long_precision": _safe_div(float(true_long_count), float(trigger_count)),
        "short_risk_rate": _safe_div(float(short_risk_count), float(trigger_count)),
        "current_pass_risk_rate": _safe_div(float(current_pass_risk_count), float(trigger_count)),
    }


def _numeric_values(records: Sequence[Mapping[str, Any]], feature_name: str) -> List[float]:
    values: List[float] = []
    for record in records:
        parsed = _num((record.get("features") or {}).get(feature_name))
        if parsed is not None and math.isfinite(parsed):
            values.append(parsed)
    return sorted(set(values))


def _thresholds(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    if len(values) <= 8:
        return list(values)
    positions = [0.10, 0.25, 0.50, 0.75, 0.90]
    thresholds: List[float] = []
    for position in positions:
        index = min(len(values) - 1, max(0, int(round(position * (len(values) - 1)))))
        thresholds.append(values[index])
    return sorted(set(thresholds))


def generate_single_feature_rules(records: Sequence[Mapping[str, Any]], *, feature_names: Sequence[str]) -> List[CandidateRule]:
    rules: List[CandidateRule] = []
    for feature_name in feature_names:
        values = _numeric_values(records, feature_name)
        for threshold in _thresholds(values):
            threshold_label = f"{threshold:.6g}".replace("-", "neg").replace(".", "p")
            rules.append(
                CandidateRule(
                    name=f"{feature_name}_ge_{threshold_label}",
                    family="single",
                    clauses=[[feature_name, ">=", threshold]],
                )
            )
            rules.append(
                CandidateRule(
                    name=f"{feature_name}_le_{threshold_label}",
                    family="single",
                    clauses=[[feature_name, "<=", threshold]],
                )
            )
    return rules


def generate_pairwise_rules(single_rules: Sequence[CandidateRule], *, max_rules: int = 200) -> List[CandidateRule]:
    selected = list(single_rules[:max_rules])
    rules: List[CandidateRule] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if left.clauses[0][0] == right.clauses[0][0]:
                continue
            rules.append(
                CandidateRule(
                    name=f"{left.name}_AND_{right.name}",
                    family="pairwise",
                    clauses=[left.clauses[0], right.clauses[0]],
                )
            )
    return rules


def generate_stop_reason_rules(records: Sequence[Mapping[str, Any]], base_rules: Sequence[CandidateRule]) -> List[CandidateRule]:
    reasons = sorted({str((record.get("features") or {}).get("stop_reason", "missing")) for record in records})
    rules: List[CandidateRule] = []
    for reason in reasons:
        for rule in base_rules:
            rules.append(
                CandidateRule(
                    name=f"stop_{reason}_AND_{rule.name}",
                    family="stop_reason",
                    clauses=[["stop_reason", "==", reason]] + rule.clauses,
                )
            )
    return rules


def numeric_feature_names(records: Sequence[Mapping[str, Any]]) -> List[str]:
    names = set()
    for record in records:
        for name, value in (record.get("features") or {}).items():
            parsed = _num(value)
            if parsed is not None and math.isfinite(parsed):
                names.add(name)
    return sorted(names)


def constrained_score(summary: Mapping[str, Any]) -> float:
    failed_long = float(summary.get("failed_long_count") or 0)
    true_precision = float(summary.get("true_long_precision") or 0.0)
    short_risk = float(summary.get("short_risk_count") or 0)
    pass_risk = float(summary.get("current_pass_risk_count") or 0)
    trigger_count = float(summary.get("trigger_count") or 0)
    if trigger_count == 0:
        return -1e9
    return failed_long * 10.0 + true_precision * 5.0 - short_risk * 4.0 - pass_risk * 2.0


def candidate_decision(summary: Mapping[str, Any]) -> str:
    failed_long = int(summary.get("failed_long_count") or 0)
    short_risk = int(summary.get("short_risk_count") or 0)
    pass_risk = int(summary.get("current_pass_risk_count") or 0)
    precision = float(summary.get("true_long_precision") or 0.0)
    if failed_long >= 10 and short_risk <= 5 and pass_risk <= 5 and precision >= 0.35:
        return "policy_candidate"
    if failed_long >= 5 and short_risk <= 10:
        return "diagnostic_only"
    return "reject"


def decision_priority(decision: Any) -> int:
    return {"policy_candidate": 2, "diagnostic_only": 1, "reject": 0}.get(str(decision), 0)


def evaluate_candidates(candidates: Sequence[CandidateRule], records: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    summaries = [evaluate_rule(candidate, records) for candidate in candidates]
    for summary in summaries:
        summary["score"] = constrained_score(summary)
        summary["decision"] = candidate_decision(summary)
    return sorted(summaries, key=lambda item: (item["score"], item["failed_long_count"]), reverse=True)


def _rule_from_summary(summary: Mapping[str, Any]) -> CandidateRule:
    return CandidateRule(
        name=str(summary.get("name", "unknown")),
        family=str(summary.get("family", "unknown")),
        clauses=list(summary.get("clauses") or []),
    )


def _evaluate_train_selected_rule(
    train_summary: Mapping[str, Any],
    heldout_records: Sequence[Mapping[str, Any]],
    *,
    heldout_fold: int,
    selection_rank: int,
) -> JsonDict:
    heldout_summary = evaluate_rule(_rule_from_summary(train_summary), heldout_records)
    heldout_summary["score"] = constrained_score(heldout_summary)
    heldout_summary["decision"] = candidate_decision(heldout_summary)
    heldout_summary["heldout_fold"] = heldout_fold
    heldout_summary["selection_rank"] = selection_rank
    heldout_summary["train_score"] = train_summary.get("score")
    heldout_summary["train_decision"] = train_summary.get("decision")
    heldout_summary["train_trigger_count"] = train_summary.get("trigger_count")
    heldout_summary["train_failed_long_count"] = train_summary.get("failed_long_count")
    heldout_summary["train_short_risk_count"] = train_summary.get("short_risk_count")
    heldout_summary["train_current_pass_risk_count"] = train_summary.get("current_pass_risk_count")
    return heldout_summary


def split_records(records: Sequence[Mapping[str, Any]], *, heldout_fold: int, folds: int) -> Tuple[List[JsonDict], List[JsonDict]]:
    train: List[JsonDict] = []
    heldout: List[JsonDict] = []
    for record in records:
        task_id = str(record["task_id"])
        if fold_id(task_id, folds=folds) == heldout_fold:
            heldout.append(dict(record))
        else:
            train.append(dict(record))
    return train, heldout


def generate_motif_rules(records: Sequence[Mapping[str, Any]]) -> List[CandidateRule]:
    feature_names = [
        "late_remaining_plateau_steps",
        "remaining_equal_transition_frac",
        "remaining_late_slope",
        "confidence_late_slope",
        "gap_last",
        "final_remaining_ratio_by_selected",
    ]
    return [
        CandidateRule(name=f"motif_{rule.name}", family="motif", clauses=rule.clauses)
        for rule in generate_single_feature_rules(records, feature_names=feature_names)
    ]


def discover_for_source(records: Sequence[Mapping[str, Any]], *, folds: int) -> JsonDict:
    feature_names = numeric_feature_names(records)
    fold_outputs: List[JsonDict] = []
    all_heldout: List[JsonDict] = []
    stable_feature_counts: Dict[str, int] = {}

    for heldout_fold in range(folds):
        train, heldout = split_records(records, heldout_fold=heldout_fold, folds=folds)
        single_rules = generate_single_feature_rules(train, feature_names=feature_names)
        train_ranked_single = evaluate_candidates(single_rules, train)
        top_train_rules = [
            CandidateRule(name=item["name"], family=item["family"], clauses=item["clauses"])
            for item in train_ranked_single[:50]
        ]
        pairwise_rules = generate_pairwise_rules(top_train_rules, max_rules=50)
        stop_rules = generate_stop_reason_rules(train, top_train_rules[:20])
        motif_rules = generate_motif_rules(train)
        all_rules = top_train_rules + pairwise_rules + stop_rules + motif_rules
        train_ranked = evaluate_candidates(all_rules, train)
        heldout_evaluated = [
            _evaluate_train_selected_rule(train_summary, heldout, heldout_fold=heldout_fold, selection_rank=rank)
            for rank, train_summary in enumerate(train_ranked[:100], start=1)
        ]
        best = heldout_evaluated[0] if heldout_evaluated else {"name": "none", "decision": "reject", "score": -1e9}
        for clause in best.get("clauses", []):
            stable_feature_counts[str(clause[0])] = stable_feature_counts.get(str(clause[0]), 0) + 1
        fold_outputs.append(
            {
                "heldout_fold": heldout_fold,
                "top_train_candidate": train_ranked[0] if train_ranked else None,
                "top_train_selected_heldout": best,
                "heldout_rows": len(heldout),
            }
        )
        all_heldout.extend(heldout_evaluated[:20])

    ranked = sorted(
        all_heldout,
        key=lambda item: (
            decision_priority(item.get("decision")),
            -int(item.get("selection_rank") or 9999),
            item.get("score", -1e9),
            item.get("failed_long_count", 0),
        ),
        reverse=True,
    )
    source_decision = "reject"
    if any(item.get("decision") == "policy_candidate" for item in ranked[:20]):
        source_decision = "policy_candidate"
    elif any(item.get("decision") == "diagnostic_only" for item in ranked[:20]):
        source_decision = "diagnostic_only"

    return {
        "rows": len(records),
        "true_long": sum(1 for record in records if record["labels"].get("true_long")),
        "failed_long": sum(1 for record in records if record["labels"].get("failed_long")),
        "short": sum(1 for record in records if record["labels"].get("short_risk")),
        "feature_count": len(feature_names),
        "folds": fold_outputs,
        "top_candidates": ranked[:100],
        "stable_features": sorted(stable_feature_counts.items(), key=lambda item: (-item[1], item[0])),
        "decision": source_decision,
    }


def optional_sklearn_diagnostic(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    names = numeric_feature_names(records)
    if not names:
        return {"available": True, "reason": "no_numeric_features", "top_weights": []}
    x_rows: List[List[float]] = []
    y: List[int] = []
    for record in records:
        features = record.get("features") or {}
        x_rows.append([float(_num(features.get(name)) or 0.0) for name in names])
        y.append(1 if record["labels"].get("failed_long") else 0)
    if len(set(y)) < 2:
        return {"available": True, "reason": "one_class", "top_weights": []}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=0),
    )
    model.fit(x_rows, y)
    logistic = model.named_steps["logisticregression"]
    weights = list(logistic.coef_[0])
    ranked = sorted(zip(names, weights), key=lambda item: abs(item[1]), reverse=True)
    return {
        "available": True,
        "top_weights": [{"feature": name, "weight": weight} for name, weight in ranked[:20]],
    }


def load_records(results_path: str | Path, traces_path: str | Path, *, source_name: str) -> List[JsonDict]:
    rows = load_jsonl(results_path)
    traces = load_jsonl(traces_path)
    joined = join_results_and_traces(rows, traces)
    return [
        build_feature_record(payload["row"], payload["traces"], source_name=source_name)
        for _task_id, payload in sorted(joined.items())
    ]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def write_report_outputs(output_dir: str | Path, summary: Mapping[str, Any], *, candidates: Sequence[Mapping[str, Any]]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=_json_default)

    fieldnames = [
        "source",
        "name",
        "family",
        "decision",
        "trigger_count",
        "failed_long_count",
        "true_long_count",
        "short_risk_count",
        "current_pass_risk_count",
        "true_long_precision",
        "failed_long_recall",
        "short_risk_rate",
        "current_pass_risk_rate",
        "score",
        "heldout_fold",
        "selection_rank",
        "train_score",
        "train_decision",
        "train_trigger_count",
        "train_failed_long_count",
        "train_short_risk_count",
        "train_current_pass_risk_count",
    ]
    with (path / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({field: candidate.get(field) for field in fieldnames})

    pareto_rows = sorted(
        candidates,
        key=lambda item: (
            int(item.get("failed_long_count") or 0),
            -int(item.get("short_risk_count") or 0),
            -int(item.get("current_pass_risk_count") or 0),
        ),
        reverse=True,
    )[:50]
    with (path / "pareto.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in pareto_rows:
            writer.writerow({field: candidate.get(field) for field in fieldnames})

    lines = [
        "# Trace Feature Audit V2 Report",
        "",
        f"Decision: `{summary.get('decision')}`",
        "",
        f"Decision reason: {summary.get('decision_reason', 'not recorded')}",
        "",
        "## Source Summary",
        "",
        "| Source | Rows | True-long | Failed-long | Short | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for source, source_summary in sorted((summary.get("sources") or {}).items()):
        lines.append(
            "| {source} | {rows} | {true_long} | {failed_long} | {short} | {decision} |".format(
                source=source,
                rows=source_summary.get("rows", 0),
                true_long=source_summary.get("true_long", 0),
                failed_long=source_summary.get("failed_long", 0),
                short=source_summary.get("short", 0),
                decision=source_summary.get("decision", "reject"),
            )
        )
    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
            "| Source | Candidate | Family | Decision | Fold | Train rank | Triggers | Failed-long | Short risk | Current-pass risk | Precision |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in list(candidates)[:20]:
        lines.append(
            "| {source} | `{name}` | `{family}` | `{decision}` | {heldout_fold} | {selection_rank} | {trigger_count} | {failed_long_count} | {short_risk_count} | {current_pass_risk_count} | {precision} |".format(
                source=candidate.get("source", "unknown"),
                name=candidate.get("name", "unknown"),
                family=candidate.get("family", "unknown"),
                decision=candidate.get("decision", "reject"),
                heldout_fold=candidate.get("heldout_fold", ""),
                selection_rank=candidate.get("selection_rank", ""),
                trigger_count=candidate.get("trigger_count", 0),
                failed_long_count=candidate.get("failed_long_count", 0),
                short_risk_count=candidate.get("short_risk_count", 0),
                current_pass_risk_count=candidate.get("current_pass_risk_count", 0),
                precision="{:.3f}".format(float(candidate.get("true_long_precision") or 0.0)),
            )
        )
    (path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> JsonDict:
    sources = {
        "previous": load_records(args.prev_results, args.prev_traces, source_name="previous"),
        "midcons": load_records(args.midcons_results, args.midcons_traces, source_name="midcons"),
    }
    source_summaries: Dict[str, JsonDict] = {}
    all_candidates: List[JsonDict] = []
    diagnostics: Dict[str, JsonDict] = {}

    for source_name, records in sources.items():
        source_summary = discover_for_source(records, folds=args.folds)
        motif_ranked = evaluate_candidates(generate_motif_rules(records), records)
        for item in motif_ranked[:30]:
            item["source"] = source_name
        for item in source_summary["top_candidates"]:
            item["source"] = source_name
        source_summary["top_full_source_motif_diagnostics"] = motif_ranked[:30]
        source_summaries[source_name] = source_summary
        all_candidates.extend(source_summary["top_candidates"][:50])
        diagnostics[source_name] = {
            "sklearn_sparse_logistic": optional_sklearn_diagnostic(records),
        }

    all_candidates = sorted(
        all_candidates,
        key=lambda item: (
            decision_priority(item.get("decision")),
            -int(item.get("selection_rank") or 9999),
            item.get("score", -1e9),
            item.get("failed_long_count", 0),
        ),
        reverse=True,
    )
    source_decisions = {key: value["decision"] for key, value in source_summaries.items()}
    if source_decisions and all(value == "policy_candidate" for value in source_decisions.values()):
        decision = "policy_candidate"
        decision_reason = "All trace sources produced held-out train-selected policy candidates."
    elif any(value in {"policy_candidate", "diagnostic_only"} for value in source_decisions.values()):
        decision = "diagnostic_only"
        decision_reason = "At least one source has signal, but the policy-candidate gate is not stable across both trace sources."
    else:
        decision = "reject"
        decision_reason = "No source produced a held-out train-selected diagnostic or policy candidate."

    summary = {
        "decision": decision,
        "decision_reason": decision_reason,
        "sources": source_summaries,
        "diagnostics": diagnostics,
        "top_candidates": all_candidates[:100],
        "notes": [
            "CPU-only offline audit; no GPU policy run launched.",
            "Oracle/pass labels are used only for offline evaluation.",
            "Learned diagnostics are discovery aids, not the final training-free policy.",
        ],
    }
    write_report_outputs(args.output_dir, summary, candidates=all_candidates)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only trace feature audit v2")
    parser.add_argument("--prev-results", required=True)
    parser.add_argument("--prev-traces", required=True)
    parser.add_argument("--midcons-results", required=True)
    parser.add_argument("--midcons-traces", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_audit(args)
    compact = {
        "decision": summary["decision"],
        "sources": {key: value["decision"] for key, value in summary["sources"].items()},
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
