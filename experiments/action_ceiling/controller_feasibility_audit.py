#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl, metric
from experiments.action_ceiling.action_ceiling_matrix import current_branch, current_commit, git_capture, parse_task_id_group


ACTION_ORDER = ("KEEP_PRIMARY", "EXPAND_16", "EXPAND_24", "EXPAND_32", "EXPAND_48")
EXPANSION_ACTIONS = ACTION_ORDER[1:]
ACTION_RANK = {action: idx for idx, action in enumerate(ACTION_ORDER)}
MIN_ACTION_LABELS = ("KEEP", "EXPAND_16", "EXPAND_24", "EXPAND_32", "EXPAND_48", "UNRECOVERABLE")
BANK_DIR = "analysis_outputs/controller_action_bank_h200_20260707_tier1_offline"
SPLIT_DIR = "analysis_outputs/grouped_split_20260702_accel2"
H200_AUDIT_DIR = "analysis_outputs/h200_repro_audit_20260707_tier1_v2"
TEST_LOCK = "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"

PROBE_FEATURES = [
    "selected_len",
    "best_len",
    "best_score",
    "best_long_len",
    "best_long_score",
    "long_ratio",
    "raw_long_ratio",
    "selected_score",
    "selected_raw_score",
    "long_minus_best_score",
    "long_len_minus_selected",
    "official_selected_length",
    "s3_selected_length",
    "primary_selected_length",
]
TRACE_FEATURES = [
    "mean_final_confidence",
    "mean_gap_at_stop_or_final",
    "mean_top1_at_stop_or_final",
    "remaining_mask_ratio_at_stop_or_final",
    "remaining_masks_at_stop_or_final",
    "stop_step",
    "route2_trace_top1_last",
    "route2_trace_top1_median",
    "route2_trace_confidence_max",
    "route2_trace_max_remaining_plateau_steps",
]
ACTION_FEATURES = ["candidate_actual_canvas", "canvas_delta_from_primary", "normalized_canvas_cost", "action_rank"]
FEATURE_VARIANTS = {
    "probe_only": PROBE_FEATURES,
    "trace_only": TRACE_FEATURES,
    "probe_trace_fused": PROBE_FEATURES + TRACE_FEATURES,
}
FORBIDDEN_FEATURES = {
    "oracle_length",
    "oracle_bucket",
    "reference_code",
    "unit_test_result",
    "pass_fail",
    "primary_passed",
    "action_passed",
    "benefit_label",
    "harm_label",
    "error_type",
    "task_id",
    "task_group",
    "split",
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_rate(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def row_passed(row: Mapping[str, Any]) -> bool:
    if "passed" in row:
        return _to_bool(row.get("passed"))
    return _to_bool(row.get("action_passed"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_task(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        out[str(row["task_id"])] = row
    return out


def rows_by_task_action(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["task_id"]), str(row["action"])): row for row in rows}


def load_bank_rows(bank_dir: Path) -> list[dict[str, Any]]:
    path = bank_dir / "action_bank.jsonl"
    if path.exists():
        return load_jsonl(path)
    return [dict(row) for row in read_csv_rows(bank_dir / "action_bank.csv")]


def h200_run_paths(audit_dir: Path) -> dict[str, str]:
    rows = read_csv_rows(audit_dir / "old_vs_h200_baselines.csv")
    return {row["run"]: row["h200_path"] for row in rows if row.get("h200_path")}


def primary_results_from_bank(bank_dir: Path, audit_dir: Path) -> str:
    manifest = load_json_if_exists(bank_dir / "run_manifest.json")
    primary = manifest.get("primary_results")
    if primary:
        return str(primary)
    paths = h200_run_paths(audit_dir)
    return str(Path(paths["midcons"]) / "results.jsonl")


def extract_base_features(primary: Mapping[str, Any], route2: Mapping[str, Any] | None = None) -> dict[str, float]:
    m = primary.get("metrics") or {}
    r = (route2 or {}).get("metrics") or {}
    selected = _to_float(metric(primary, "selected_mask_length", metric(primary, "mask_length")), 0.0)
    best_long_score = _to_float(m.get("best_long_score"), 0.0)
    best_score = _to_float(m.get("best_score"), 0.0)
    return {
        "selected_len": selected,
        "best_len": _to_float(m.get("best_len"), 0.0),
        "best_score": best_score,
        "best_long_len": _to_float(m.get("best_long_len"), 0.0),
        "best_long_score": best_long_score,
        "long_ratio": _to_float(m.get("long_ratio"), 0.0),
        "raw_long_ratio": _to_float(m.get("raw_long_ratio"), 0.0),
        "selected_score": _to_float(m.get("selected_score"), 0.0),
        "selected_raw_score": _to_float(m.get("selected_raw_score"), 0.0),
        "long_minus_best_score": best_long_score - best_score,
        "long_len_minus_selected": _to_float(m.get("best_long_len"), 0.0) - selected,
        "official_selected_length": _to_float(m.get("official_selected_length"), 0.0),
        "s3_selected_length": _to_float(m.get("s3_selected_length"), 0.0),
        "primary_selected_length": selected,
        "mean_final_confidence": _to_float(m.get("mean_final_confidence"), 0.0),
        "mean_gap_at_stop_or_final": _to_float(m.get("mean_gap_at_stop_or_final"), 0.0),
        "mean_top1_at_stop_or_final": _to_float(m.get("mean_top1_at_stop_or_final"), 0.0),
        "remaining_mask_ratio_at_stop_or_final": _to_float(m.get("remaining_mask_ratio_at_stop_or_final"), 0.0),
        "remaining_masks_at_stop_or_final": _to_float(m.get("remaining_masks_at_stop_or_final"), 0.0),
        "stop_step": _to_float(m.get("stop_step"), 0.0),
        "route2_trace_top1_last": _to_float(r.get("route2_trace_top1_last"), 0.0),
        "route2_trace_top1_median": _to_float(r.get("route2_trace_top1_median"), 0.0),
        "route2_trace_confidence_max": _to_float(r.get("route2_trace_confidence_max"), 0.0),
        "route2_trace_max_remaining_plateau_steps": _to_float(r.get("route2_trace_max_remaining_plateau_steps"), 0.0),
    }


def minimum_passing_action(actions: Mapping[str, Mapping[str, Any]]) -> str:
    for action in ACTION_ORDER:
        row = actions.get(action)
        if row is not None and row_passed(row):
            return "KEEP" if action == "KEEP_PRIMARY" else action
    return "UNRECOVERABLE"


def task_records(bank_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in bank_rows:
        grouped[str(row["task_id"])][str(row["action"])] = row
    records: list[dict[str, Any]] = []
    for task_id, actions in sorted(grouped.items()):
        keep = actions["KEEP_PRIMARY"]
        primary_pass = row_passed(keep)
        expansion_rows = [actions[action] for action in EXPANSION_ACTIONS if action in actions]
        any_expand_pass = any(row_passed(row) for row in expansion_rows)
        any_expand_fail = any(not row_passed(row) for row in expansion_rows)
        min_action = minimum_passing_action(actions)
        records.append(
            {
                "task_id": task_id,
                "task_group": str(keep.get("task_group") or parse_task_id_group(task_id)),
                "split": str(keep.get("split")),
                "oracle_bucket": str(keep.get("oracle_bucket")),
                "primary_passed": primary_pass,
                "recoverable": (not primary_pass) and any_expand_pass,
                "harmable": primary_pass and any_expand_fail,
                "minimum_passing_action": min_action,
                "minimum_passing_rank": MIN_ACTION_LABELS.index(min_action),
                "no_action_passes": min_action == "UNRECOVERABLE",
                "keep_best": min_action == "KEEP",
                "expansion_best": min_action in EXPANSION_ACTIONS,
                "best_pass_action_count": sum(1 for row in actions.values() if row_passed(row)),
                "actions": actions,
            }
        )
    return records


def feature_rows(
    bank_rows: Sequence[Mapping[str, Any]],
    *,
    primary_results: str,
    route2_results: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_by_task = rows_by_task(load_jsonl(primary_results))
    route2_by_task = rows_by_task(load_jsonl(route2_results)) if route2_results else {}
    records = task_records(bank_rows)
    task_features: list[dict[str, Any]] = []
    action_features: list[dict[str, Any]] = []
    action_by_key = rows_by_task_action(bank_rows)
    for record in records:
        task_id = record["task_id"]
        primary_row = primary_by_task.get(task_id)
        if primary_row is None:
            continue
        base = extract_base_features(primary_row, route2_by_task.get(task_id))
        keep = action_by_key[(task_id, "KEEP_PRIMARY")]
        primary_selected = _to_float(keep.get("primary_selected_length"), base.get("primary_selected_length", 0.0))
        task_row = {
            "task_id": task_id,
            "task_group": record["task_group"],
            "split": record["split"],
            "oracle_bucket": record["oracle_bucket"],
            "primary_passed": record["primary_passed"],
            "recoverable": record["recoverable"],
            "harmable": record["harmable"],
            "minimum_passing_action": record["minimum_passing_action"],
            "minimum_passing_rank": record["minimum_passing_rank"],
            **base,
        }
        task_features.append(task_row)
        for action in ACTION_ORDER:
            bank = action_by_key.get((task_id, action))
            if bank is None:
                continue
            actual_canvas = _to_float(bank.get("actual_canvas"), primary_selected)
            action_passed = row_passed(bank)
            action_features.append(
                {
                    **task_row,
                    "action": action,
                    "action_rank": ACTION_RANK[action],
                    "candidate_actual_canvas": actual_canvas,
                    "canvas_delta_from_primary": max(0.0, actual_canvas - primary_selected),
                    "normalized_canvas_cost": actual_canvas / 48.0,
                    "action_passed": action_passed,
                    "benefit_label": (not record["primary_passed"]) and action != "KEEP_PRIMARY" and action_passed,
                    "harm_label": record["primary_passed"] and action != "KEEP_PRIMARY" and (not action_passed),
                    "inference_cost_sec": _to_float(bank.get("inference_cost_sec"), 0.0),
                    "actual_canvas": actual_canvas,
                }
            )
    return task_features, action_features


class LogisticModel:
    def __init__(self, feature_names: Sequence[str], *, c_value: float = 1.0) -> None:
        self.feature_names = list(feature_names)
        self.c_value = float(c_value)
        self.mean: list[float] = []
        self.std: list[float] = []
        self.weights: list[float] = []

    def fit(self, rows: Sequence[Mapping[str, Any]], labels: Sequence[int], *, epochs: int = 500, lr: float = 0.08) -> None:
        if not rows:
            self.mean = [0.0 for _ in self.feature_names]
            self.std = [1.0 for _ in self.feature_names]
            self.weights = [0.0 for _ in range(len(self.feature_names) + 1)]
            return
        x = np.array([[float(row.get(name, 0.0) or 0.0) for name in self.feature_names] for row in rows], dtype=np.float64)
        y = np.array(labels, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        z = (x - mean) / std
        z = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
        w = np.zeros(z.shape[1], dtype=np.float64)
        pos = max(1.0, float(y.sum()))
        neg = max(1.0, float(len(y) - y.sum()))
        sample_weight = np.where(y > 0.5, neg / pos, 1.0)
        l2 = 1.0 / max(1e-6, self.c_value)
        for _ in range(int(epochs)):
            logits = np.clip(z @ w, -40.0, 40.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (z.T @ ((p - y) * sample_weight)) / max(1, len(y))
            grad[1:] += l2 * 0.01 * w[1:]
            w -= lr * grad
        self.mean = [float(value) for value in mean]
        self.std = [float(value) for value in std]
        self.weights = [float(value) for value in w]

    def predict_one(self, row: Mapping[str, Any]) -> float:
        logit = max(-40.0, min(40.0, self.linear_score(row)))
        return float(1.0 / (1.0 + math.exp(-logit)))

    def linear_score(self, row: Mapping[str, Any]) -> float:
        if not self.weights:
            return 0.0
        values = [float(row.get(name, 0.0) or 0.0) for name in self.feature_names]
        z = [1.0] + [(value - mean) / std for value, mean, std in zip(values, self.mean, self.std)]
        return float(sum(value * weight for value, weight in zip(z, self.weights)))

    def predict_many(self, rows: Sequence[Mapping[str, Any]]) -> list[float]:
        return [self.predict_one(row) for row in rows]

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "c_value": self.c_value,
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "LogisticModel":
        model = cls(payload.get("feature_names") or [], c_value=float(payload.get("c_value") or 1.0))
        model.mean = [float(value) for value in (payload.get("mean") or [])]
        model.std = [float(value) for value in (payload.get("std") or [])]
        model.weights = [float(value) for value in (payload.get("weights") or [])]
        return model


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        rank += j - i
        i = j
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hit = 0
    total = 0.0
    for idx, (_, label) in enumerate(ordered, start=1):
        if label:
            hit += 1
            total += hit / idx
    return float(total / positives)


def ranking_rows(task_feature_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    train_rows = [row for row in task_feature_rows if row.get("split") == "train"]
    outputs: list[dict[str, Any]] = []
    for target in ("recoverable", "harmable"):
        for variant, columns in FEATURE_VARIANTS.items():
            model = LogisticModel(columns, c_value=1.0)
            model.fit(train_rows, [1 if row.get(target) else 0 for row in train_rows])
            for split in ("train", "calibration", "validation"):
                rows = [row for row in task_feature_rows if row.get("split") == split]
                labels = [1 if row.get(target) else 0 for row in rows]
                scores = model.predict_many(rows)
                positives = sum(labels)
                top_k = positives if positives > 0 else min(10, len(rows))
                ordered = sorted(zip(scores, labels, rows), key=lambda item: item[0], reverse=True)
                top_hits = sum(label for _, label, _ in ordered[:top_k])
                outputs.append(
                    {
                        "target": target,
                        "feature_variant": variant,
                        "split": split,
                        "rows": len(rows),
                        "positives": positives,
                        "base_rate": _safe_rate(positives, len(rows)),
                        "auc": auc_score(labels, scores),
                        "average_precision": average_precision(labels, scores),
                        "top_k": top_k,
                        "top_k_hits": top_hits,
                        "top_k_precision": _safe_rate(top_hits, top_k),
                    }
                )
    return outputs


def label_distribution(records: Sequence[Mapping[str, Any]], action_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("overall", "train", "calibration", "validation"):
        recs = list(records) if split == "overall" else [row for row in records if row.get("split") == split]
        acts = list(action_rows) if split == "overall" else [row for row in action_rows if row.get("split") == split]
        expand = [row for row in acts if row.get("action") != "KEEP_PRIMARY"]
        min_counts = Counter(str(row.get("minimum_passing_action")) for row in recs)
        row = {
            "split": split,
            "task_count": len(recs),
            "primary_pass_count": sum(1 for row in recs if row.get("primary_passed")),
            "primary_fail_count": sum(1 for row in recs if not row.get("primary_passed")),
            "row_level_recoverable_count": sum(1 for row in recs if row.get("recoverable")),
            "row_level_harmable_count": sum(1 for row in recs if row.get("harmable")),
            "action_row_benefit_count": sum(1 for row in expand if row.get("benefit_label")),
            "action_row_harm_count": sum(1 for row in expand if row.get("harm_label")),
            "action_row_neutral_count": sum(1 for row in expand if not row.get("benefit_label") and not row.get("harm_label")),
            "no_action_passes": sum(1 for row in recs if row.get("no_action_passes")),
            "keep_best": sum(1 for row in recs if row.get("keep_best")),
            "expansion_best": sum(1 for row in recs if row.get("expansion_best")),
        }
        for label in MIN_ACTION_LABELS:
            row[f"minimum_{label.lower()}"] = min_counts[label]
        rows.append(row)
    return rows


def per_action_diagnostics(action_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("overall", "train", "calibration", "validation"):
        split_rows = list(action_rows) if split == "overall" else [row for row in action_rows if row.get("split") == split]
        for action in ACTION_ORDER:
            subset = [row for row in split_rows if row.get("action") == action]
            rows.append(
                {
                    "split": split,
                    "action": action,
                    "rows": len(subset),
                    "pass_count": sum(1 for row in subset if row.get("action_passed")),
                    "pass_rate": _safe_rate(sum(1 for row in subset if row.get("action_passed")), len(subset)),
                    "benefit_count": sum(1 for row in subset if row.get("benefit_label")),
                    "harm_count": sum(1 for row in subset if row.get("harm_label")),
                    "neutral_count": sum(1 for row in subset if not row.get("benefit_label") and not row.get("harm_label")),
                    "mean_cost_sec": _mean([float(row.get("inference_cost_sec") or 0.0) for row in subset]),
                }
            )
    return rows


def action_transition_matrix(action_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, bool, bool]] = Counter()
    for row in action_rows:
        counts[(str(row.get("split")), str(row.get("action")), bool(row.get("primary_passed")), bool(row.get("action_passed")))] += 1
    rows = []
    for (split, action, primary_passed, action_passed), count in sorted(counts.items()):
        rows.append(
            {
                "split": split,
                "action": action,
                "primary_passed": primary_passed,
                "action_passed": action_passed,
                "count": count,
            }
        )
    return rows


def binomial_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return total


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float | None:
    if n == 0:
        return None
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = (lo + hi) / 2.0
        if binomial_cdf(k, n, mid) >= alpha:
            lo = mid
        else:
            hi = mid
    return hi


def wilson_upper(k: int, n: int, z: float = 1.96) -> float | None:
    if n == 0:
        return None
    phat = k / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return min(1.0, (centre + margin) / denom)


def zero_harm_n_for_bound(bound: float, alpha: float = 0.05) -> int:
    return int(math.ceil(math.log(alpha) / math.log(1.0 - bound)))


def risk_sample_complexity(calibration_task_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for harm_bound in (0.05, 0.10, 0.15):
        required = zero_harm_n_for_bound(harm_bound)
        rows.append(
            {
                "risk_quantity": "P(harm | intervene)",
                "harm_bound": harm_bound,
                "confidence": 0.95,
                "zero_harm_interventions_required_cp": required,
                "calibration_task_count": calibration_task_count,
                "calibration_can_certify_with_zero_harm": calibration_task_count >= required,
                "zero_harm_cp_upper_at_full_calibration": clopper_pearson_upper(0, calibration_task_count),
                "zero_harm_wilson_upper_at_full_calibration": wilson_upper(0, calibration_task_count),
            }
        )
    for n in (1, 2, 5, 10, 20, 40, 60, 80, calibration_task_count):
        if n <= 0:
            continue
        rows.append(
            {
                "risk_quantity": "P(harm | intervene)",
                "harm_bound": "curve",
                "confidence": 0.95,
                "intervention_count": n,
                "harm_count": 0,
                "clopper_pearson_upper": clopper_pearson_upper(0, n),
                "wilson_upper": wilson_upper(0, n),
            }
        )
    return rows


def oracle_coverage_curve(records: Sequence[Mapping[str, Any]], action_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row["task_id"]), str(row["action"])): row for row in action_rows}
    rows: list[dict[str, Any]] = []
    for split in ("train", "calibration", "validation"):
        split_records = [row for row in records if row.get("split") == split]
        for max_rank, action in enumerate(ACTION_ORDER):
            selected = []
            for record in split_records:
                task_id = str(record["task_id"])
                allowed = ACTION_ORDER[: max_rank + 1]
                passing = [by_key[(task_id, candidate)] for candidate in allowed if row_passed(by_key[(task_id, candidate)])]
                selected.append(passing[0] if passing else by_key[(task_id, "KEEP_PRIMARY")])
            wins = losses = pass_count = interventions = harms = benefits = 0
            for selected_row, record in zip(selected, split_records):
                before = bool(record["primary_passed"])
                after = row_passed(selected_row)
                pass_count += int(after)
                interventions += int(str(selected_row.get("action")) != "KEEP_PRIMARY")
                wins += int((not before) and after)
                losses += int(before and not after)
                benefits += int((not before) and after and str(selected_row.get("action")) != "KEEP_PRIMARY")
                harms += int(before and not after and str(selected_row.get("action")) != "KEEP_PRIMARY")
            rows.append(
                {
                    "split": split,
                    "oracle_action_ceiling": action,
                    "tasks": len(split_records),
                    "pass_count": pass_count,
                    "wins_vs_keep": wins,
                    "losses_vs_keep": losses,
                    "intervention_count": interventions,
                    "benefit_count": benefits,
                    "harm_count": harms,
                    "conditional_harm_rate": _safe_rate(harms, interventions),
                    "population_harm_rate": _safe_rate(harms, len(split_records)),
                    "net_gain_probability": _safe_rate(benefits - harms, len(split_records)),
                }
            )
    return rows


def classify_audit_verdict(label_rows: Sequence[Mapping[str, Any]], ranking: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    overall = next(row for row in label_rows if row["split"] == "overall")
    validation = next(row for row in label_rows if row["split"] == "validation")
    calibration = next(row for row in label_rows if row["split"] == "calibration")
    reasons: list[str] = []
    if int(overall["row_level_recoverable_count"]) == 0:
        return "insufficient_action_bank_signal", ["no row-level recoverable cases in action bank"]
    best_val = max(
        (
            float(row.get("auc") or 0.0)
            for row in ranking
            if row.get("split") == "validation" and row.get("target") == "recoverable"
        ),
        default=0.0,
    )
    base_rate = _safe_rate(int(validation["row_level_recoverable_count"]), int(validation["task_count"])) or 0.0
    best_top = max(
        (
            float(row.get("top_k_precision") or 0.0)
            for row in ranking
            if row.get("split") == "validation" and row.get("target") == "recoverable"
        ),
        default=0.0,
    )
    if best_val < 0.55 and best_top <= base_rate * 1.25:
        reasons.append("feature ranking is weak for validation recoverability")
    if int(calibration["row_level_recoverable_count"]) < zero_harm_n_for_bound(0.05):
        reasons.append("calibration has fewer recoverable rows than zero-harm 5% certification needs")
    if int(overall["action_row_harm_count"]) > 4 * max(1, int(overall["action_row_benefit_count"])):
        reasons.append("non-KEEP action harm labels strongly outnumber benefit labels")
    if best_val >= 0.60 or best_top > max(base_rate * 1.5, base_rate + 0.05):
        reasons.append("some recoverability ranking signal exists outside the first controller")
    if len(reasons) >= 2:
        return "mixed_controller_failure", reasons
    if reasons and "feature ranking" in reasons[0]:
        return "feature_ranking_failure", reasons
    if reasons and "calibration" in reasons[0]:
        return "risk_certification_sample_limited", reasons
    if reasons and "harm labels" in reasons[0]:
        return "label_imbalance_dominant", reasons
    if reasons and "ranking signal" in reasons[0]:
        return "controller_signal_exists_but_first_model_misses_it", reasons
    return "mixed_controller_failure", ["weak but nonzero action-bank signal with strict risk constraints"]


def render_report(summary: Mapping[str, Any], label_rows: Sequence[Mapping[str, Any]], ranking: Sequence[Mapping[str, Any]]) -> str:
    validation = next(row for row in label_rows if row["split"] == "validation")
    best_recover = max(
        [row for row in ranking if row["split"] == "validation" and row["target"] == "recoverable"],
        key=lambda row: float(row.get("auc") or 0.0),
    )
    lines = [
        "# H200 Controller Feasibility Audit",
        "",
        f"audit_verdict: `{summary['audit_verdict']}`",
        f"branch: `{summary['branch']}`",
        f"commit: `{summary['commit']}`",
        "",
        "## Label Distribution",
        "",
        f"- validation recoverable rows: `{validation['row_level_recoverable_count']}/{validation['task_count']}`",
        f"- validation harmable rows: `{validation['row_level_harmable_count']}/{validation['task_count']}`",
        f"- validation action-row benefit/harm: `{validation['action_row_benefit_count']}/{validation['action_row_harm_count']}`",
        "",
        "## Feature Ranking",
        "",
        f"- best validation recoverability feature variant: `{best_recover['feature_variant']}`",
        f"- AUC: `{best_recover['auc']}`",
        f"- top-k recoverable precision: `{best_recover['top_k_precision']}` with k=`{best_recover['top_k']}`",
        "",
        "## Risk Certification",
        "",
        f"- zero-harm interventions required for 5% conditional harm upper95: `{zero_harm_n_for_bound(0.05)}`",
        "- Conditional risk is `P(harm | intervene)`.",
        "- Population policy harm is `P(intervene and harm)` over all validation rows.",
        "- Net gain is `P(benefit) - P(harm)`.",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in summary.get("audit_reasons", []))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit H200 controller action-bank feasibility and risk certification limits.")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-root", default="analysis_outputs")
    parser.add_argument("--bank-dir", default=BANK_DIR)
    parser.add_argument("--h200-audit-dir", default=H200_AUDIT_DIR)
    parser.add_argument("--test-lock", default=TEST_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"controller_feasibility_h200_{timestamp}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    bank_dir = Path(args.bank_dir)
    audit_dir = Path(args.h200_audit_dir)
    run_paths = h200_run_paths(audit_dir)
    primary_results = primary_results_from_bank(bank_dir, audit_dir)
    route2_results = str(Path(run_paths["route2"]) / "results.jsonl")
    bank_rows = load_bank_rows(bank_dir)
    records = task_records(bank_rows)
    task_feature_rows, action_feature_rows = feature_rows(bank_rows, primary_results=primary_results, route2_results=route2_results)
    labels = label_distribution(records, action_feature_rows)
    ranking = ranking_rows(task_feature_rows)
    risk_rows = risk_sample_complexity(
        int(next(row["task_count"] for row in labels if row["split"] == "calibration"))
    )
    per_action = per_action_diagnostics(action_feature_rows)
    transition = action_transition_matrix(action_feature_rows)
    coverage = oracle_coverage_curve(records, action_feature_rows)
    verdict, reasons = classify_audit_verdict(labels, ranking)
    test_lock = load_json_if_exists(Path(args.test_lock))
    summary = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "audit_verdict": verdict,
        "audit_reasons": reasons,
        "bank_dir": str(bank_dir),
        "primary_results": primary_results,
        "route2_results": route2_results,
        "test_lock_status": test_lock.get("test_status"),
        "test_evaluation_count": test_lock.get("test_evaluation_count"),
        "h200_evidence_base": True,
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }
    best_ranking = max(
        [row for row in ranking if row["split"] == "validation" and row["target"] == "recoverable"],
        key=lambda row: float(row.get("auc") or 0.0),
    )
    summary["validation_recoverability_ranking"] = best_ranking

    write_csv(output_dir / "label_distribution.csv", labels)
    write_csv(output_dir / "action_transition_matrix.csv", transition)
    write_csv(output_dir / "risk_sample_complexity.csv", risk_rows)
    write_csv(output_dir / "ranking_diagnostics.csv", ranking)
    write_csv(output_dir / "per_action_diagnostics.csv", per_action)
    write_csv(output_dir / "oracle_coverage_curve.csv", coverage)
    (output_dir / "feature_signal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary, labels, ranking), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "audit_verdict": verdict}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
