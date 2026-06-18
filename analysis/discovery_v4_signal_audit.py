#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_feature_audit_v2 import compute_shape_features
from analysis.trace_long_rescue_features import join_results_and_traces, load_jsonl, metric, oracle_bucket


JsonDict = Dict[str, Any]


DEFAULT_POLICY_NAMES = ["precision_len32", "precision_len24", "broad_len24"]
FORBIDDEN_FEATURE_TOKENS = (
    "oracle",
    "true_long",
    "passed",
    "pairwise",
    "bucket",
    "risk",
    "win",
    "loss",
    "failed",
    "failure",
    "rescued",
    "class",
    "verification",
)


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int(value: Any) -> Optional[int]:
    parsed = _num(value)
    if parsed is None:
        return None
    return int(parsed)


def _bool(value: Any) -> bool:
    return bool(value)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _rows_by_task(rows: Iterable[Mapping[str, Any]], *, name: str) -> Dict[str, Mapping[str, Any]]:
    by_task: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id is None:
            raise ValueError(f"Missing task_id in {name}")
        by_task[str(task_id)] = row
    return by_task


def _pairwise(policy_passed: bool, baseline_passed: bool) -> str:
    if policy_passed and not baseline_passed:
        return "win"
    if (not policy_passed) and baseline_passed:
        return "loss"
    if policy_passed and baseline_passed:
        return "tie_pass"
    return "tie_fail"


def _fold_id(task_id: str, *, folds: int) -> int:
    if folds <= 1:
        return 0
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % folds


def _last_decision_from_traces(traces: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not traces:
        return {}
    decision = traces[-1].get("stop_decision") or {}
    return decision if isinstance(decision, Mapping) else {}


def _prefixed_trace_features(
    row: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> JsonDict:
    shape = compute_shape_features(traces)
    last_decision = _last_decision_from_traces(traces)
    features: JsonDict = {}
    selected_len = _int(metric(row, "selected_mask_length", metric(row, "mask_length")))
    for key, value in shape.items():
        features[f"{prefix}_{key}"] = value
    final_remaining = _num(shape.get("remaining_last"))
    if final_remaining is not None and selected_len is not None:
        features[f"{prefix}_final_remaining_ratio_by_selected"] = _safe_div(
            float(final_remaining), float(max(selected_len, 1))
        )
    features[f"{prefix}_stop_reason"] = str(
        metric(row, "stop_reason", last_decision.get("reason", "missing"))
    )
    return features


def _probe_features(row: Mapping[str, Any], *, prefix: str) -> JsonDict:
    selected_len = _int(metric(row, "selected_mask_length", metric(row, "mask_length")))
    best_len = _num(metric(row, "best_len"))
    best_long_len = _num(metric(row, "best_long_len"))
    best_score = _num(metric(row, "best_score"))
    best_long_score = _num(metric(row, "best_long_score", metric(row, "long_score_max")))
    long_ratio = _num(metric(row, "long_ratio"))
    raw_long_ratio = _num(metric(row, "raw_long_ratio"))
    features: JsonDict = {
        f"{prefix}_selected_len": selected_len,
        f"{prefix}_best_len": best_len,
        f"{prefix}_best_long_len": best_long_len,
        f"{prefix}_best_score": best_score,
        f"{prefix}_best_long_score": best_long_score,
        f"{prefix}_long_ratio": long_ratio,
        f"{prefix}_raw_long_ratio": raw_long_ratio,
        f"{prefix}_base_selected_length": _num(metric(row, "base_selected_length")),
        f"{prefix}_final_selected_length": _num(metric(row, "final_selected_length")),
        f"{prefix}_mean_top1_final": _num(metric(row, "mean_top1_at_stop_or_final")),
        f"{prefix}_mean_gap_final": _num(metric(row, "mean_gap_at_stop_or_final")),
        f"{prefix}_remaining_mask_ratio_final": _num(metric(row, "remaining_mask_ratio_at_stop_or_final")),
    }
    if best_long_len is not None and selected_len is not None:
        features[f"{prefix}_best_long_minus_selected"] = best_long_len - float(selected_len)
        features[f"{prefix}_selected_below_best_long"] = bool(float(selected_len) < best_long_len)
    if best_score is not None and best_long_score is not None:
        features[f"{prefix}_best_long_score_gap"] = best_score - best_long_score
    if long_ratio is not None:
        features[f"{prefix}_long_ratio_gap_to_one"] = 1.0 - long_ratio
    return features


def _policy_features(policy_row: Mapping[str, Any], *, policy_name: str) -> JsonDict:
    return {
        f"{policy_name}_triggered": _bool(metric(policy_row, "route2_trace_rescue_triggered", False)),
        f"{policy_name}_selected_len": _int(metric(policy_row, "selected_mask_length")),
        f"{policy_name}_primary_selected_len": _int(metric(policy_row, "route2_primary_selected_mask_length")),
        f"{policy_name}_rescue_len": _int(metric(policy_row, "route2_rescue_length")),
        f"{policy_name}_final_source": metric(policy_row, "route2_final_source"),
        f"{policy_name}_primary_stop_reason": metric(policy_row, "route2_primary_stop_reason"),
        f"{policy_name}_trace_top1_last": _num(metric(policy_row, "route2_trace_top1_last")),
        f"{policy_name}_trace_top1_median": _num(metric(policy_row, "route2_trace_top1_median")),
        f"{policy_name}_trace_confidence_max": _num(metric(policy_row, "route2_trace_confidence_max")),
        f"{policy_name}_trace_max_remaining_plateau_steps": _num(
            metric(policy_row, "route2_trace_max_remaining_plateau_steps")
        ),
        f"{policy_name}_combined_total_sec_including_probe": _num(
            metric(policy_row, "route2_combined_total_sec_including_probe")
        ),
    }


def _policy_outcome_fields(
    baseline_row: Mapping[str, Any],
    policy_row: Mapping[str, Any],
    *,
    policy_name: str,
    oracle_len: Optional[int],
) -> JsonDict:
    baseline_passed = _bool(metric(baseline_row, "passed", False))
    policy_passed = _bool(metric(policy_row, "passed", False))
    true_long = bool(oracle_len is not None and oracle_len >= 17)
    baseline_failed = not baseline_passed
    triggered = _bool(metric(policy_row, "route2_trace_rescue_triggered", False))
    pairwise = _pairwise(policy_passed, baseline_passed)
    return {
        f"{policy_name}_passed": policy_passed,
        f"{policy_name}_pairwise": pairwise,
        f"{policy_name}_win": pairwise == "win",
        f"{policy_name}_loss": pairwise == "loss",
        f"missed_failed_long_{policy_name}": bool(true_long and baseline_failed and not triggered),
        f"triggered_rescue_failure_{policy_name}": bool(true_long and triggered and not policy_passed),
        f"rescued_long_{policy_name}": bool(true_long and baseline_failed and triggered and policy_passed),
    }


def build_row_action_table(
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_trace_rows: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[JsonDict]:
    baseline = _rows_by_task(baseline_rows, name="baseline")
    policy_maps = {name: _rows_by_task(rows, name=name) for name, rows in policies.items()}
    traces_by_task = join_results_and_traces(baseline_rows, baseline_trace_rows)
    common_task_ids = set(baseline)
    for policy_map in policy_maps.values():
        common_task_ids &= set(policy_map)

    rows: List[JsonDict] = []
    for task_id in sorted(common_task_ids):
        base_row = baseline[task_id]
        oracle_len = _int(metric(base_row, "oracle_mask_length"))
        baseline_passed = _bool(metric(base_row, "passed", False))
        true_long = bool(oracle_len is not None and oracle_len >= 17)
        short_risk = bool(oracle_len is not None and oracle_len <= 8)
        trace_payload = traces_by_task.get(task_id, {})
        traces = list(trace_payload.get("traces") or [])

        row: JsonDict = {
            "task_id": task_id,
            "oracle_length": oracle_len,
            "oracle_bucket": oracle_bucket(oracle_len),
            "true_long": true_long,
            "short_risk": short_risk,
            "current_pass_risk": baseline_passed,
            "baseline_passed": baseline_passed,
            "baseline_failed_long": bool(true_long and not baseline_passed),
            "fold5": _fold_id(task_id, folds=5),
        }
        row.update(_probe_features(base_row, prefix="baseline"))
        row.update(_prefixed_trace_features(base_row, traces, prefix="baseline"))

        for policy_name, policy_map in policy_maps.items():
            policy_row = policy_map[task_id]
            row.update(_policy_features(policy_row, policy_name=policy_name))
            row.update(
                _policy_outcome_fields(
                    base_row,
                    policy_row,
                    policy_name=policy_name,
                    oracle_len=oracle_len,
                )
            )
        rows.append(row)
    return rows


def candidate_feature_columns(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    names: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key in names:
                continue
            lowered = key.lower()
            if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
                continue
            if key in {"task_id", "fold5"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                names.append(key)
    return names


@dataclass(frozen=True)
class CandidateSlice:
    name: str
    family: str
    clauses: List[List[Any]]

    def matches(self, row: Mapping[str, Any]) -> bool:
        for feature_name, op, threshold in self.clauses:
            value = row.get(str(feature_name))
            if op == "==":
                if str(value) != str(threshold):
                    return False
                continue
            parsed = _num(value)
            if parsed is None:
                return False
            if op == ">=":
                if parsed < float(threshold):
                    return False
            elif op == "<=":
                if parsed > float(threshold):
                    return False
            else:
                raise ValueError(f"Unsupported op: {op}")
        return True


def _threshold_label(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("/", "_").replace(" ", "_")
    parsed = _num(value)
    if parsed is None:
        return str(value)
    return f"{parsed:.6g}".replace("-", "neg").replace(".", "p")


def _numeric_values(rows: Sequence[Mapping[str, Any]], column: str) -> List[float]:
    values = sorted({_num(row.get(column)) for row in rows if _num(row.get(column)) is not None})
    return [float(value) for value in values]


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


def generate_single_clause_slices(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
    family: str = "single",
) -> List[CandidateSlice]:
    slices: List[CandidateSlice] = []
    for column in feature_columns:
        numeric_values = _numeric_values(rows, column)
        if numeric_values:
            for threshold in _thresholds(numeric_values):
                label = _threshold_label(threshold)
                slices.append(CandidateSlice(f"{column}_ge_{label}", family, [[column, ">=", threshold]]))
                slices.append(CandidateSlice(f"{column}_le_{label}", family, [[column, "<=", threshold]]))
            continue

        values = [str(row.get(column)) for row in rows if row.get(column) is not None]
        counts = Counter(values)
        for value, count in counts.most_common(8):
            if count < 3:
                continue
            label = _threshold_label(value)
            slices.append(CandidateSlice(f"{column}_eq_{label}", family, [[column, "==", value]]))
    return slices


def generate_pairwise_slices(base_slices: Sequence[CandidateSlice], *, max_base: int = 60) -> List[CandidateSlice]:
    selected = list(base_slices[:max_base])
    pairs: List[CandidateSlice] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if left.clauses[0][0] == right.clauses[0][0]:
                continue
            pairs.append(
                CandidateSlice(
                    name=f"{left.name}_AND_{right.name}",
                    family="pairwise",
                    clauses=[left.clauses[0], right.clauses[0]],
                )
            )
    return pairs


def _target_fields(policy_name: str) -> Tuple[str, str, str]:
    return (
        f"missed_failed_long_{policy_name}",
        f"triggered_rescue_failure_{policy_name}",
        f"{policy_name}_win",
    )


def evaluate_slice(candidate: CandidateSlice, rows: Sequence[Mapping[str, Any]], *, policy_name: str = "precision_len32") -> JsonDict:
    triggered = [row for row in rows if candidate.matches(row)]
    trigger_count = len(triggered)
    missed_field, rescue_failure_field, win_field = _target_fields(policy_name)
    missed_count = sum(1 for row in triggered if row.get(missed_field))
    rescue_failure_count = sum(1 for row in triggered if row.get(rescue_failure_field))
    route2_win_count = sum(1 for row in triggered if row.get(win_field))
    true_long_count = sum(1 for row in triggered if row.get("true_long"))
    short_risk_count = sum(1 for row in triggered if row.get("short_risk"))
    current_pass_risk_count = sum(1 for row in triggered if row.get("current_pass_risk"))
    total_missed = sum(1 for row in rows if row.get(missed_field))
    score = (
        missed_count * 8.0
        + rescue_failure_count * 2.0
        + route2_win_count * 5.0
        + (true_long_count / max(trigger_count, 1)) * 4.0
        - short_risk_count * 5.0
        - current_pass_risk_count * 3.0
    )
    decision = "reject"
    if missed_count >= 8 and short_risk_count <= 3 and current_pass_risk_count <= 3:
        decision = "probe_trace_fusion_candidate"
    elif rescue_failure_count >= 10 and short_risk_count <= 5:
        decision = "rescue_quality_candidate"
    elif missed_count >= 5 and short_risk_count <= 8:
        decision = "diagnostic_only"
    return {
        "name": candidate.name,
        "family": candidate.family,
        "clauses": json.dumps(candidate.clauses, ensure_ascii=False),
        "trigger_count": trigger_count,
        "missed_failed_long_count": missed_count,
        "triggered_rescue_failure_count": rescue_failure_count,
        "route2_win_count": route2_win_count,
        "true_long_count": true_long_count,
        "short_risk_count": short_risk_count,
        "current_pass_risk_count": current_pass_risk_count,
        "missed_failed_long_recall": _safe_div(float(missed_count), float(total_missed)),
        "true_long_precision": _safe_div(float(true_long_count), float(trigger_count)),
        "short_risk_rate": _safe_div(float(short_risk_count), float(trigger_count)),
        "current_pass_risk_rate": _safe_div(float(current_pass_risk_count), float(trigger_count)),
        "score": score,
        "decision": decision,
    }


def _slice_from_summary(summary: Mapping[str, Any]) -> CandidateSlice:
    clauses = summary.get("clauses")
    if isinstance(clauses, str):
        clauses = json.loads(clauses)
    return CandidateSlice(
        name=str(summary.get("name", "unknown")),
        family=str(summary.get("family", "unknown")),
        clauses=list(clauses or []),
    )


def _with_fold_stability(
    summaries: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int,
    policy_name: str,
) -> List[JsonDict]:
    enriched: List[JsonDict] = []
    for summary in summaries:
        candidate = _slice_from_summary(summary)
        good_folds = 0
        fold_missed_counts: List[int] = []
        for fold in range(folds):
            heldout = [row for row in rows if int(row.get("fold5", 0)) == fold]
            fold_summary = evaluate_slice(candidate, heldout, policy_name=policy_name)
            fold_missed_counts.append(int(fold_summary.get("missed_failed_long_count") or 0))
            if (
                int(fold_summary.get("missed_failed_long_count") or 0) >= 1
                and int(fold_summary.get("short_risk_count") or 0) <= 2
                and int(fold_summary.get("current_pass_risk_count") or 0) <= 2
            ):
                good_folds += 1
        item = dict(summary)
        item["stable_folds"] = good_folds
        item["fold_missed_failed_long_counts"] = json.dumps(fold_missed_counts)
        if item.get("decision") in {"probe_trace_fusion_candidate", "diagnostic_only"} and good_folds < 3:
            item["decision"] = "diagnostic_only" if int(item.get("missed_failed_long_count") or 0) >= 5 else "reject"
        enriched.append(item)
    return enriched


def discover_slices(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_name: str,
    folds: int,
) -> Tuple[List[JsonDict], List[JsonDict], List[JsonDict]]:
    columns = candidate_feature_columns(rows)
    single = generate_single_clause_slices(rows, feature_columns=columns, family="single")
    ranked_single = sorted(
        (evaluate_slice(candidate, rows, policy_name=policy_name) for candidate in single),
        key=lambda item: (float(item.get("score") or -1e9), int(item.get("missed_failed_long_count") or 0)),
        reverse=True,
    )
    top_single_candidates = [_slice_from_summary(item) for item in ranked_single[:80]]
    pairwise = generate_pairwise_slices(top_single_candidates, max_base=80)
    ranked_pairwise = sorted(
        (evaluate_slice(candidate, rows, policy_name=policy_name) for candidate in pairwise),
        key=lambda item: (float(item.get("score") or -1e9), int(item.get("missed_failed_long_count") or 0)),
        reverse=True,
    )
    trace_columns = [
        column
        for column in columns
        if any(token in column for token in ["trace", "remaining", "confidence", "top1", "gap", "plateau"])
    ]
    trace_slices = generate_single_clause_slices(rows, feature_columns=trace_columns, family="trace_shape")
    ranked_trace = sorted(
        (evaluate_slice(candidate, rows, policy_name=policy_name) for candidate in trace_slices),
        key=lambda item: (float(item.get("score") or -1e9), int(item.get("missed_failed_long_count") or 0)),
        reverse=True,
    )
    return (
        _with_fold_stability(ranked_single[:200], rows, folds=folds, policy_name=policy_name),
        _with_fold_stability(ranked_pairwise[:200], rows, folds=folds, policy_name=policy_name),
        _with_fold_stability(ranked_trace[:200], rows, folds=folds, policy_name=policy_name),
    )


def weak_signal_votes(row: Mapping[str, Any]) -> JsonDict:
    selected_len = _num(row.get("baseline_selected_len"))
    best_long_len = _num(row.get("baseline_best_long_len"))
    long_ratio = _num(row.get("baseline_long_ratio"))
    plateau = _num(row.get("baseline_late_remaining_plateau_steps"))
    top1_last = _num(row.get("baseline_top1_last"))
    gap_final = _num(row.get("baseline_mean_gap_final"))
    triggered = bool(row.get("precision_len32_triggered"))

    signals = {
        "weak_selected_le12": bool(selected_len is not None and selected_len <= 12),
        "weak_best_long_ge17": bool(best_long_len is not None and best_long_len >= 17),
        "weak_long_ratio_ge085": bool(long_ratio is not None and long_ratio >= 0.85),
        "weak_plateau_ge2": bool(plateau is not None and plateau >= 2),
        "weak_high_top1": bool(top1_last is not None and top1_last >= 0.95),
        "weak_high_gap": bool(gap_final is not None and gap_final >= 0.8),
        "weak_untriggered_by_precision": not triggered,
    }
    signals["weak_vote_score"] = sum(1 for value in signals.values() if value is True)
    return signals


def build_weak_signal_rows(rows: Sequence[Mapping[str, Any]], *, policy_name: str) -> List[JsonDict]:
    output: List[JsonDict] = []
    for row in rows:
        votes = weak_signal_votes(row)
        candidate_row = {
            "task_id": row.get("task_id"),
            **votes,
            "missed_failed_long": row.get(f"missed_failed_long_{policy_name}"),
            "triggered_rescue_failure": row.get(f"triggered_rescue_failure_{policy_name}"),
            "true_long": row.get("true_long"),
            "short_risk": row.get("short_risk"),
            "current_pass_risk": row.get("current_pass_risk"),
        }
        output.append(candidate_row)
    return output


def calibration_residuals(rows: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        reason = str(row.get("baseline_stop_reason", "missing"))
        value = _num(row.get("baseline_top1_last"))
        if value is not None:
            groups[reason].append(value)
    medians = {reason: median(values) for reason, values in groups.items() if values}
    output: List[JsonDict] = []
    for row in rows:
        reason = str(row.get("baseline_stop_reason", "missing"))
        value = _num(row.get("baseline_top1_last"))
        residual = value - medians[reason] if value is not None and reason in medians else None
        output.append(
            {
                "task_id": row.get("task_id"),
                "stop_reason": reason,
                "baseline_top1_last": value,
                "stop_reason_top1_median": medians.get(reason),
                "baseline_top1_stop_residual": residual,
                "true_long": row.get("true_long"),
                "short_risk": row.get("short_risk"),
                "missed_failed_long_precision_len32": row.get("missed_failed_long_precision_len32"),
            }
        )
    return output


def summarize_policy_deltas(rows: Sequence[Mapping[str, Any]], *, policy_names: Sequence[str]) -> List[JsonDict]:
    output: List[JsonDict] = []
    for policy_name in policy_names:
        wins = sum(1 for row in rows if row.get(f"{policy_name}_pairwise") == "win")
        losses = sum(1 for row in rows if row.get(f"{policy_name}_pairwise") == "loss")
        tie_pass = sum(1 for row in rows if row.get(f"{policy_name}_pairwise") == "tie_pass")
        tie_fail = sum(1 for row in rows if row.get(f"{policy_name}_pairwise") == "tie_fail")
        triggers = sum(1 for row in rows if row.get(f"{policy_name}_triggered"))
        output.append(
            {
                "policy": policy_name,
                "wins": wins,
                "losses": losses,
                "tie_pass": tie_pass,
                "tie_fail": tie_fail,
                "net_delta": wins - losses,
                "triggers": triggers,
            }
        )

        bucket_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            bucket_counts[str(row.get("oracle_bucket"))][str(row.get(f"{policy_name}_pairwise"))] += 1
        for bucket, counts in sorted(bucket_counts.items()):
            output.append(
                {
                    "policy": policy_name,
                    "bucket": bucket,
                    "wins": counts.get("win", 0),
                    "losses": counts.get("loss", 0),
                    "tie_pass": counts.get("tie_pass", 0),
                    "tie_fail": counts.get("tie_fail", 0),
                    "net_delta": counts.get("win", 0) - counts.get("loss", 0),
                    "triggers": sum(
                        1
                        for row in rows
                        if str(row.get("oracle_bucket")) == bucket and row.get(f"{policy_name}_triggered")
                    ),
                }
            )
    return output


def action_overlap(rows: Sequence[Mapping[str, Any]], *, policy_names: Sequence[str]) -> List[JsonDict]:
    output: List[JsonDict] = []
    for left_index, left in enumerate(policy_names):
        for right in policy_names[left_index + 1 :]:
            output.append(
                {
                    "left_policy": left,
                    "right_policy": right,
                    "both_triggered": sum(
                        1 for row in rows if row.get(f"{left}_triggered") and row.get(f"{right}_triggered")
                    ),
                    "left_only_triggered": sum(
                        1 for row in rows if row.get(f"{left}_triggered") and not row.get(f"{right}_triggered")
                    ),
                    "right_only_triggered": sum(
                        1 for row in rows if row.get(f"{right}_triggered") and not row.get(f"{left}_triggered")
                    ),
                    "both_win": sum(
                        1 for row in rows if row.get(f"{left}_pairwise") == "win" and row.get(f"{right}_pairwise") == "win"
                    ),
                    "left_win_right_not": sum(
                        1 for row in rows if row.get(f"{left}_pairwise") == "win" and row.get(f"{right}_pairwise") != "win"
                    ),
                    "right_win_left_not": sum(
                        1 for row in rows if row.get(f"{right}_pairwise") == "win" and row.get(f"{left}_pairwise") != "win"
                    ),
                }
            )
    return output


def uplift_diagnostics(rows: Sequence[Mapping[str, Any]], *, policy_names: Sequence[str]) -> List[JsonDict]:
    output: List[JsonDict] = []
    for policy_name in policy_names:
        for triggered in [False, True]:
            subset = [row for row in rows if bool(row.get(f"{policy_name}_triggered")) is triggered]
            if not subset:
                continue
            output.append(
                {
                    "policy": policy_name,
                    "slice": f"triggered_{triggered}",
                    "rows": len(subset),
                    "wins": sum(1 for row in subset if row.get(f"{policy_name}_pairwise") == "win"),
                    "losses": sum(1 for row in subset if row.get(f"{policy_name}_pairwise") == "loss"),
                    "missed_failed_long": sum(1 for row in subset if row.get(f"missed_failed_long_{policy_name}")),
                    "triggered_rescue_failure": sum(
                        1 for row in subset if row.get(f"triggered_rescue_failure_{policy_name}")
                    ),
                    "short_risk": sum(1 for row in subset if row.get("short_risk")),
                    "current_pass_risk": sum(1 for row in subset if row.get("current_pass_risk")),
                }
            )
    output.extend(action_overlap(rows, policy_names=policy_names))
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report(summary: Mapping[str, Any], top_slices: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Discovery V4 Signal Audit",
        "",
        f"Decision: `{summary.get('decision')}`",
        "",
        f"Decision reason: {summary.get('decision_reason')}",
        "",
        "## Overview",
        "",
        f"- Rows: `{summary.get('row_count')}`",
        f"- True-long rows: `{summary.get('true_long_count')}`",
        f"- Baseline failed-long rows: `{summary.get('baseline_failed_long_count')}`",
        "",
        "## Policy Deltas",
        "",
        "| Policy | Bucket | Win | Loss | Tie pass | Tie fail | Net | Triggers |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.get("policy_deltas") or []:
        lines.append(
            "| {policy} | {bucket} | {wins} | {losses} | {tie_pass} | {tie_fail} | {net_delta} | {triggers} |".format(
                policy=item.get("policy", ""),
                bucket=item.get("bucket", "all"),
                wins=item.get("wins", 0),
                losses=item.get("losses", 0),
                tie_pass=item.get("tie_pass", 0),
                tie_fail=item.get("tie_fail", 0),
                net_delta=item.get("net_delta", 0),
                triggers=item.get("triggers", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Top Slices",
            "",
            "| Candidate | Family | Decision | Stable folds | Triggers | Missed-long | Rescue-failure | Short risk | Current-pass risk | Precision |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in top_slices[:30]:
        lines.append(
            "| `{name}` | `{family}` | `{decision}` | `{stable}` | `{triggers}` | `{missed}` | `{rescue}` | `{short}` | `{pass_risk}` | `{precision:.3f}` |".format(
                name=item.get("name", ""),
                family=item.get("family", ""),
                decision=item.get("decision", ""),
                stable=item.get("stable_folds", ""),
                triggers=item.get("trigger_count", 0),
                missed=item.get("missed_failed_long_count", 0),
                rescue=item.get("triggered_rescue_failure_count", 0),
                short=item.get("short_risk_count", 0),
                pass_risk=item.get("current_pass_risk_count", 0),
                precision=float(item.get("true_long_precision") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- This is a CPU-only discovery audit, not a new pass-rate claim.",
            "- Oracle/pass labels are used for offline accounting only.",
            "- Learned or fitted discovery scores remain microscopes unless separately promoted to a learned-controller design.",
            "- A GPU run requires a held-out low-risk candidate and a new action brief.",
            "",
        ]
    )
    return "\n".join(lines)


def _policy_shortlist(summary: Mapping[str, Any], top_slices: Sequence[Mapping[str, Any]]) -> str:
    decision = summary.get("decision")
    lines = [
        "# Discovery V4 Policy Shortlist",
        "",
        f"Decision: `{decision}`",
        "",
        f"Reason: {summary.get('decision_reason')}",
        "",
    ]
    if top_slices:
        best = top_slices[0]
        lines.extend(
            [
                "## Best Candidate",
                "",
                f"- Name: `{best.get('name')}`",
                f"- Family: `{best.get('family')}`",
                f"- Clauses: `{best.get('clauses')}`",
                f"- Stable folds: `{best.get('stable_folds')}`",
                f"- Missed failed-long: `{best.get('missed_failed_long_count')}`",
                f"- Short risk: `{best.get('short_risk_count')}`",
                f"- Current-pass risk: `{best.get('current_pass_risk_count')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Next Action",
            "",
        ]
    )
    if decision in {"probe_trace_fusion_candidate", "rescue_quality_candidate", "combined_candidate"}:
        lines.append("Write a GPU action brief only after checking GPU availability and reviewing the candidate clauses.")
    elif decision == "diagnostic_only":
        lines.append("Do not launch GPU yet. Inspect top diagnostic slices and decide whether to design a better mechanism.")
    else:
        lines.append("Keep Route2 precision len32 as conservative polish and avoid a blind full policy run.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    output_dir: str | Path,
    *,
    summary: Mapping[str, Any],
    row_action_table: Sequence[Mapping[str, Any]],
    slice_candidates: Sequence[Mapping[str, Any]],
    rule_candidates: Sequence[Mapping[str, Any]],
    trace_shape_candidates: Sequence[Mapping[str, Any]],
    calibration_residuals: Sequence[Mapping[str, Any]],
    weak_signal_rows: Sequence[Mapping[str, Any]],
    uplift_rows: Sequence[Mapping[str, Any]],
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(path / "row_action_table.csv", row_action_table)
    _write_csv(path / "slice_candidates.csv", slice_candidates)
    _write_csv(path / "rule_candidates.csv", rule_candidates)
    _write_csv(path / "trace_shape_candidates.csv", trace_shape_candidates)
    _write_csv(path / "calibration_residuals.csv", calibration_residuals)
    _write_csv(path / "weak_signal_votes.csv", weak_signal_rows)
    _write_csv(path / "uplift_diagnostics.csv", uplift_rows)
    (path / "policy_shortlist.md").write_text(
        _policy_shortlist(summary, list(slice_candidates)[:20]),
        encoding="utf-8",
    )
    (path / "report.md").write_text(_report(summary, list(slice_candidates)[:30]), encoding="utf-8")


def _decide(top_slices: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    if not top_slices:
        return "route2_polish_only", "No candidate slices were generated."
    best = top_slices[0]
    if (
        best.get("decision") == "probe_trace_fusion_candidate"
        and int(best.get("stable_folds") or 0) >= 4
        and int(best.get("missed_failed_long_count") or 0) >= 8
    ):
        return "probe_trace_fusion_candidate", "A low-risk missed-long slice passed the held-out stability gate."
    if (
        best.get("decision") == "rescue_quality_candidate"
        and int(best.get("triggered_rescue_failure_count") or 0) >= 10
    ):
        return "rescue_quality_candidate", "A triggered rescue-failure slice is large enough for mechanism design."
    if any(item.get("decision") in {"probe_trace_fusion_candidate", "rescue_quality_candidate"} for item in top_slices[:10]):
        return "diagnostic_only", "Some candidates are promising but not stable enough for a GPU action brief."
    if any(int(item.get("missed_failed_long_count") or 0) >= 5 for item in top_slices[:20]):
        return "diagnostic_only", "Weak missed-long signal exists, but risk/stability gates are not satisfied."
    return "route2_polish_only", "No stable low-risk V4 signal was found beyond Route2 polish."


def run_audit(args: argparse.Namespace) -> JsonDict:
    baseline_rows = load_jsonl(args.baseline_results)
    baseline_traces = load_jsonl(args.baseline_traces)
    policies = {
        "precision_len32": load_jsonl(args.route2_len32_results),
        "precision_len24": load_jsonl(args.route2_len24_results),
        "broad_len24": load_jsonl(args.route2_broad_results),
    }
    rows = build_row_action_table(baseline_rows, baseline_trace_rows=baseline_traces, policies=policies)
    policy_names = list(policies)
    policy_deltas = summarize_policy_deltas(rows, policy_names=policy_names)
    single_candidates, pairwise_candidates, trace_shape_candidates = discover_slices(
        rows,
        policy_name="precision_len32",
        folds=args.folds,
    )
    all_candidates = sorted(
        single_candidates + pairwise_candidates + trace_shape_candidates,
        key=lambda item: (
            str(item.get("decision")) in {"probe_trace_fusion_candidate", "rescue_quality_candidate"},
            int(item.get("stable_folds") or 0),
            float(item.get("score") or -1e9),
            int(item.get("missed_failed_long_count") or 0),
        ),
        reverse=True,
    )
    decision, reason = _decide(all_candidates)
    weak_rows = build_weak_signal_rows(rows, policy_name="precision_len32")
    calibration_rows = calibration_residuals(rows)
    uplift_rows = uplift_diagnostics(rows, policy_names=policy_names)
    summary = {
        "decision": decision,
        "decision_reason": reason,
        "row_count": len(rows),
        "true_long_count": sum(1 for row in rows if row.get("true_long")),
        "baseline_failed_long_count": sum(1 for row in rows if row.get("baseline_failed_long")),
        "short_risk_count": sum(1 for row in rows if row.get("short_risk")),
        "current_pass_risk_count": sum(1 for row in rows if row.get("current_pass_risk")),
        "policy_deltas": policy_deltas,
        "action_overlap": action_overlap(rows, policy_names=policy_names),
        "top_slices": all_candidates[:30],
        "inputs": {
            "baseline_results": args.baseline_results,
            "baseline_traces": args.baseline_traces,
            "route2_len32_results": args.route2_len32_results,
            "route2_len24_results": args.route2_len24_results,
            "route2_broad_results": args.route2_broad_results,
        },
        "notes": [
            "CPU-only Discovery V4 audit; no GPU job launched.",
            "Oracle/pass labels are offline accounting fields only.",
            "Candidate features exclude oracle/pass/pairwise/risk/target labels.",
        ],
    }
    write_outputs(
        args.output_dir,
        summary=summary,
        row_action_table=rows,
        slice_candidates=all_candidates[:300],
        rule_candidates=pairwise_candidates[:300],
        trace_shape_candidates=trace_shape_candidates[:300],
        calibration_residuals=calibration_rows,
        weak_signal_rows=weak_rows,
        uplift_rows=uplift_rows,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only Discovery V4 signal audit.")
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--baseline-traces", required=True)
    parser.add_argument("--route2-len32-results", required=True)
    parser.add_argument("--route2-len24-results", required=True)
    parser.add_argument("--route2-broad-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_audit(args)
    top = (summary.get("top_slices") or [{}])[0]
    print("===== Discovery V4 Signal Audit =====", flush=True)
    print(f"output_dir: {args.output_dir}", flush=True)
    print(f"rows: {summary['row_count']}", flush=True)
    print(f"decision: {summary['decision']}", flush=True)
    print(f"reason: {summary['decision_reason']}", flush=True)
    print(
        "top_candidate: "
        f"{top.get('name')} decision={top.get('decision')} "
        f"missed={top.get('missed_failed_long_count')} "
        f"short_risk={top.get('short_risk_count')} "
        f"pass_risk={top.get('current_pass_risk_count')} "
        f"stable_folds={top.get('stable_folds')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
