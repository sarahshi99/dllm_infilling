#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.action_ceiling.action_ceiling_matrix import current_branch, current_commit, git_capture
from experiments.action_ceiling.controller_feasibility_audit import (
    ACTION_ORDER,
    ACTION_RANK,
    EXPANSION_ACTIONS,
    FEATURE_VARIANTS,
    FORBIDDEN_FEATURES,
    LogisticModel,
    auc_score,
    clopper_pearson_upper,
    load_json_if_exists,
    read_csv_rows,
    row_passed,
    wilson_upper,
    write_csv,
)


BANK_DIR = "analysis_outputs/controller_action_bank_h200_20260707_tier1_offline"
V1_DIR = "analysis_outputs/controller_validation_h200_20260707_v1_replay"
FEASIBILITY_DIR = "analysis_outputs/controller_feasibility_h200_20260707_phase3_v2_feasibility"
TEST_LOCK = "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"
C_VALUES = (0.1, 1.0, 10.0)
MAIN_FEATURE_VARIANTS = ("probe_only", "probe_trace_fused")
NEGATIVE_CONTROL_FEATURE_VARIANTS = ("trace_only",)
VARIANTS = ("ordinal_only", "pairwise_only", "ordinal_pairwise_harm")
ACTION_COST = {action: idx for idx, action in enumerate(ACTION_ORDER)}


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


def _safe_rate(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in read_csv_rows(path)]


def parse_feature_table(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_dicts(path)
    for row in rows:
        for key, value in list(row.items()):
            if key in {"task_id", "task_group", "split", "oracle_bucket", "action", "generated_text_sha256"}:
                continue
            if value in {"True", "False", "true", "false"}:
                row[key] = _to_bool(value)
            else:
                try:
                    row[key] = float(value) if value not in {None, ""} else value
                except (TypeError, ValueError):
                    pass
    return rows


def rows_by_task_action(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["task_id"]), str(row["action"])): row for row in rows}


def task_rows_from_action_rows(action_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[str(row["task_id"])].append(row)
    out: list[dict[str, Any]] = []
    for task_id, rows in sorted(grouped.items()):
        keep = next(row for row in rows if row["action"] == "KEEP_PRIMARY")
        primary_pass = _to_bool(keep["primary_passed"])
        expansions = [row for row in rows if row["action"] != "KEEP_PRIMARY"]
        min_action = "UNRECOVERABLE"
        for action in ACTION_ORDER:
            candidate = next(row for row in rows if row["action"] == action)
            if _to_bool(candidate["action_passed"]):
                min_action = "KEEP" if action == "KEEP_PRIMARY" else action
                break
        out.append(
            {
                **{key: keep.get(key) for key in keep if key not in {"action", "action_passed", "benefit_label", "harm_label", "underallocation_label", "generated_text_sha256", "inference_cost_sec"}},
                "task_id": task_id,
                "primary_passed": primary_pass,
                "recoverable": (not primary_pass) and any(_to_bool(row["action_passed"]) for row in expansions),
                "harmable": primary_pass and any(not _to_bool(row["action_passed"]) for row in expansions),
                "minimum_passing_action": min_action,
                "minimum_passing_rank": {"KEEP": 0, "EXPAND_16": 1, "EXPAND_24": 2, "EXPAND_32": 3, "EXPAND_48": 4, "UNRECOVERABLE": 5}[min_action],
            }
        )
    return out


def feature_columns(feature_variant: str, *, include_action: bool = False) -> list[str]:
    cols = list(FEATURE_VARIANTS[feature_variant])
    if include_action:
        cols += ["candidate_actual_canvas", "canvas_delta_from_primary", "normalized_canvas_cost", "action_rank"]
    leaked = [col for col in cols if col in FORBIDDEN_FEATURES]
    if leaked:
        raise ValueError(f"forbidden feature columns: {leaked}")
    return cols


def add_action_rank(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        enriched = dict(row)
        enriched["action_rank"] = ACTION_RANK[str(row["action"])]
        out.append(enriched)
    return out


def one_vs_rest_ordinal_models(task_rows: Sequence[Mapping[str, Any]], feature_variant: str, c_value: float) -> list[LogisticModel]:
    cols = feature_columns(feature_variant)
    train = [row for row in task_rows if row["split"] == "train"]
    models = []
    for threshold in range(1, 6):
        model = LogisticModel(cols, c_value=c_value)
        model.fit(train, [1 if int(row["minimum_passing_rank"]) >= threshold else 0 for row in train])
        models.append(model)
    return models


def monotonic_cumulative_probs(models: Sequence[LogisticModel], row: Mapping[str, Any]) -> list[float]:
    probs = [model.predict_one(row) for model in models]
    projected: list[float] = []
    running = 1.0
    for prob in probs:
        running = min(running, max(0.0, min(1.0, prob)))
        projected.append(running)
    return projected


def rank_distribution_from_cumulative(cumulative: Sequence[float]) -> list[float]:
    probs = []
    prev = 1.0
    for value in cumulative:
        probs.append(max(0.0, prev - value))
        prev = value
    probs.append(max(0.0, prev))
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def ordinal_candidate_action(row: Mapping[str, Any], models: Sequence[LogisticModel], recover_model: LogisticModel, threshold: float) -> tuple[str, dict[str, Any]]:
    p_recover = recover_model.predict_one(row)
    cumulative = monotonic_cumulative_probs(models, row)
    distribution = rank_distribution_from_cumulative(cumulative)
    expected_rank = sum(idx * prob for idx, prob in enumerate(distribution))
    if p_recover < threshold:
        return "KEEP_PRIMARY", {"p_recover": p_recover, "expected_rank": expected_rank, "ordinal_distribution": distribution}
    rank = int(math.ceil(expected_rank))
    rank = max(1, min(4, rank))
    return ACTION_ORDER[rank], {"p_recover": p_recover, "expected_rank": expected_rank, "ordinal_distribution": distribution}


def pairwise_training_rows(action_rows: Sequence[Mapping[str, Any]], feature_variant: str) -> tuple[list[dict[str, Any]], list[int]]:
    cols = feature_columns(feature_variant, include_action=True)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in action_rows:
        if row["split"] == "train":
            by_task[str(row["task_id"])].append(row)
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    for task_id, task_actions in by_task.items():
        for left in task_actions:
            for right in task_actions:
                if ACTION_RANK[str(left["action"])] >= ACTION_RANK[str(right["action"])]:
                    continue
                left_pass = _to_bool(left["action_passed"])
                right_pass = _to_bool(right["action_passed"])
                if left_pass == right_pass:
                    if left_pass:
                        preferred, other = (left, right)
                    else:
                        continue
                else:
                    preferred, other = (left, right) if left_pass else (right, left)
                diff = {name: _to_float(preferred.get(name)) - _to_float(other.get(name)) for name in cols}
                rows.append(diff)
                labels.append(1)
                rows.append({name: -value for name, value in diff.items()})
                labels.append(0)
    return rows, labels


def fit_pairwise_model(action_rows: Sequence[Mapping[str, Any]], feature_variant: str, c_value: float) -> LogisticModel:
    cols = feature_columns(feature_variant, include_action=True)
    rows, labels = pairwise_training_rows(action_rows, feature_variant)
    model = LogisticModel(cols, c_value=c_value)
    model.fit(rows, labels)
    return model


def fit_binary_task_model(task_rows: Sequence[Mapping[str, Any]], feature_variant: str, label: str, c_value: float) -> LogisticModel:
    train = [row for row in task_rows if row["split"] == "train"]
    model = LogisticModel(feature_columns(feature_variant), c_value=c_value)
    model.fit(train, [1 if row[label] else 0 for row in train])
    return model


def fit_harm_guard(action_rows: Sequence[Mapping[str, Any]], feature_variant: str, c_value: float) -> LogisticModel:
    train = [row for row in action_rows if row["split"] == "train" and row["action"] != "KEEP_PRIMARY"]
    model = LogisticModel(feature_columns(feature_variant, include_action=True), c_value=c_value)
    model.fit(train, [1 if _to_bool(row["harm_label"]) else 0 for row in train])
    return model


def select_for_variant(
    *,
    variant: str,
    feature_variant: str,
    task_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, Any],
    split: str,
    threshold: float,
    harm_threshold: float,
) -> list[dict[str, Any]]:
    task_by_id = {str(row["task_id"]): row for row in task_rows if row["split"] == split}
    by_key = rows_by_task_action([row for row in action_rows if row["split"] == split])
    selected: list[dict[str, Any]] = []
    for task_id, task in sorted(task_by_id.items()):
        chosen_action = "KEEP_PRIMARY"
        scores: dict[str, Any] = {}
        if variant == "ordinal_only":
            chosen_action, scores = ordinal_candidate_action(task, models["ordinal"], models["recover"], threshold)
        elif variant == "pairwise_only":
            candidates = []
            for action in ACTION_ORDER:
                row = by_key[(task_id, action)]
                score = models["pairwise"].linear_score(row) - 0.02 * ACTION_COST[action]
                candidates.append((score, action, row))
            candidates.sort(key=lambda item: (item[0], -ACTION_RANK[item[1]]), reverse=True)
            best_score, best_action, _ = candidates[0]
            if best_action != "KEEP_PRIMARY" and best_score >= threshold:
                chosen_action = best_action
            scores = {"pairwise_score": best_score}
        else:
            ordinal_action, ordinal_scores = ordinal_candidate_action(task, models["ordinal"], models["recover"], threshold)
            candidates = []
            for action in ACTION_ORDER:
                row = by_key[(task_id, action)]
                pairwise_score = models["pairwise"].linear_score(row) - 0.02 * ACTION_COST[action]
                harm_prob = models["harm"].predict_one(row) if action != "KEEP_PRIMARY" else 0.0
                ordinal_bonus = 0.05 if action == ordinal_action else 0.0
                candidates.append((pairwise_score + ordinal_bonus - 0.75 * harm_prob, action, row, harm_prob, pairwise_score))
            candidates.sort(key=lambda item: (item[0], -ACTION_RANK[item[1]]), reverse=True)
            best_score, best_action, _, harm_prob, pairwise_score = candidates[0]
            if best_action != "KEEP_PRIMARY" and best_score >= threshold and harm_prob <= harm_threshold:
                chosen_action = best_action
            scores = {**ordinal_scores, "pairwise_score": pairwise_score, "harm_prob": harm_prob, "combined_score": best_score}
        selected_row = dict(by_key[(task_id, chosen_action)])
        selected_row["selected_action"] = chosen_action
        selected_row["intervened"] = chosen_action != "KEEP_PRIMARY"
        selected_row.update(scores)
        selected.append(selected_row)
    return selected


def evaluate_selection(selected: Sequence[Mapping[str, Any]], primary_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    wins = losses = ties_pass = ties_fail = 0
    interventions = harms = benefits = 0
    costs = []
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        task_id = str(row["task_id"])
        before = _to_bool(primary_rows[task_id]["action_passed"])
        after = _to_bool(row["action_passed"])
        bucket = str(row.get("oracle_bucket"))
        costs.append(_to_float(row.get("inference_cost_sec")))
        if after and not before:
            wins += 1
            benefits += int(_to_bool(row.get("intervened")))
            bucket_counts[bucket]["wins"] += 1
        elif before and not after:
            losses += 1
            harms += int(_to_bool(row.get("intervened")))
            bucket_counts[bucket]["losses"] += 1
        elif after:
            ties_pass += 1
        else:
            ties_fail += 1
        interventions += int(_to_bool(row.get("intervened")))
    total = len(selected)
    return {
        "pass_count": wins + ties_pass,
        "total": total,
        "pass_rate": _safe_rate(wins + ties_pass, total),
        "wins_vs_primary": wins,
        "losses_vs_primary": losses,
        "ties_pass": ties_pass,
        "ties_fail": ties_fail,
        "intervention_count": interventions,
        "intervention_coverage": _safe_rate(interventions, total),
        "intervention_benefit_count": benefits,
        "intervention_harm_count": harms,
        "conditional_harm_rate": _safe_rate(harms, interventions),
        "conditional_harm_upper95": clopper_pearson_upper(harms, interventions),
        "population_harm_rate": _safe_rate(harms, total),
        "population_harm_upper95": clopper_pearson_upper(harms, total),
        "population_harm_wilson_upper95": wilson_upper(harms, total),
        "net_gain_probability": _safe_rate(benefits - harms, total),
        "mean_cost_sec": _mean(costs),
        "bucket_summary": {bucket: dict(counter) for bucket, counter in sorted(bucket_counts.items())},
    }


def calibration_candidates(
    *,
    variant: str,
    feature_variant: str,
    task_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    raw = select_for_variant(
        variant=variant,
        feature_variant=feature_variant,
        task_rows=task_rows,
        action_rows=action_rows,
        models=models,
        split="calibration",
        threshold=-999.0,
        harm_threshold=1.0,
    )
    scores = []
    harms = [1.0, 0.5, 0.25, 0.1, 0.05]
    for row in raw:
        if row["selected_action"] == "KEEP_PRIMARY":
            continue
        for key in ("combined_score", "pairwise_score", "p_recover"):
            if key in row:
                scores.append(float(row[key]))
                break
    return [999.0] + sorted(set(scores), reverse=True) + [0.0, -999.0], harms


def calibrate_policy(
    *,
    variant: str,
    feature_variant: str,
    task_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, Any],
    primary_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds, harm_thresholds = calibration_candidates(
        variant=variant,
        feature_variant=feature_variant,
        task_rows=task_rows,
        action_rows=action_rows,
        models=models,
    )
    conditional_curve: list[dict[str, Any]] = []
    population_curve: list[dict[str, Any]] = []
    selected_operating: dict[str, Any] | None = None
    for threshold in thresholds:
        for harm_threshold in harm_thresholds:
            selected = select_for_variant(
                variant=variant,
                feature_variant=feature_variant,
                task_rows=task_rows,
                action_rows=action_rows,
                models=models,
                split="calibration",
                threshold=threshold,
                harm_threshold=harm_threshold,
            )
            metrics = evaluate_selection(selected, primary_rows)
            row = {
                "variant": variant,
                "feature_variant": feature_variant,
                "threshold": threshold,
                "harm_threshold": harm_threshold,
                **{key: value for key, value in metrics.items() if key != "bucket_summary"},
            }
            conditional_curve.append(row)
            population_curve.append(row)
            if (metrics.get("population_harm_upper95") is not None and metrics["population_harm_upper95"] <= 0.05):
                if selected_operating is None or (
                    int(metrics["wins_vs_primary"]) - int(metrics["losses_vs_primary"]),
                    int(metrics["intervention_count"]),
                    -float(metrics["population_harm_upper95"]),
                ) > (
                    int(selected_operating["wins_vs_primary"]) - int(selected_operating["losses_vs_primary"]),
                    int(selected_operating["intervention_count"]),
                    -float(selected_operating["population_harm_upper95"]),
                ):
                    selected_operating = row
    if selected_operating is None:
        selected_operating = {
            "variant": variant,
            "feature_variant": feature_variant,
            "threshold": 999.0,
            "harm_threshold": 0.05,
            "calibration_no_population_safe_nonzero_point": True,
            "pass_count": sum(1 for row in primary_rows.values() if _to_bool(row["action_passed"])),
            "wins_vs_primary": 0,
            "losses_vs_primary": 0,
            "intervention_count": 0,
            "population_harm_upper95": 0.0,
        }
    return selected_operating, conditional_curve, population_curve


def cv_select_c(task_rows: Sequence[Mapping[str, Any]], feature_variant: str, label: str) -> tuple[float, list[dict[str, Any]]]:
    train = [row for row in task_rows if row["split"] == "train"]
    groups = sorted({str(row["task_group"]) for row in train})
    folds = {group: idx % 5 for idx, group in enumerate(groups)}
    rows = []
    best_c = C_VALUES[0]
    best_auc = -1.0
    for c_value in C_VALUES:
        labels_all: list[int] = []
        scores_all: list[float] = []
        for fold in range(5):
            fit_rows = [row for row in train if folds[str(row["task_group"])] != fold]
            eval_rows = [row for row in train if folds[str(row["task_group"])] == fold]
            model = LogisticModel(feature_columns(feature_variant), c_value=c_value)
            model.fit(fit_rows, [1 if row[label] else 0 for row in fit_rows])
            labels = [1 if row[label] else 0 for row in eval_rows]
            scores = [model.predict_one(row) for row in eval_rows]
            labels_all.extend(labels)
            scores_all.extend(scores)
        auc = auc_score(labels_all, scores_all)
        rows.append({"label": label, "feature_variant": feature_variant, "C": c_value, "grouped_cv_auc": auc})
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_c = c_value
    return best_c, rows


def train_variant(task_rows: Sequence[Mapping[str, Any]], action_rows: Sequence[Mapping[str, Any]], variant: str, feature_variant: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cv_rows: list[dict[str, Any]] = []
    recover_c, rows = cv_select_c(task_rows, feature_variant, "recoverable")
    cv_rows.extend(rows)
    models: dict[str, Any] = {
        "recover": fit_binary_task_model(task_rows, feature_variant, "recoverable", recover_c),
    }
    if variant in {"ordinal_only", "ordinal_pairwise_harm"}:
        models["ordinal"] = one_vs_rest_ordinal_models(task_rows, feature_variant, recover_c)
    if variant in {"pairwise_only", "ordinal_pairwise_harm"}:
        models["pairwise"] = fit_pairwise_model(action_rows, feature_variant, recover_c)
    if variant == "ordinal_pairwise_harm":
        models["harm"] = fit_harm_guard(action_rows, feature_variant, recover_c)
    cv_rows.append({"variant": variant, "feature_variant": feature_variant, "selected_C": recover_c})
    return models, cv_rows


def serialize_models(models: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, model in models.items():
        if isinstance(model, list):
            out[name] = [item.to_json() for item in model]
        else:
            out[name] = model.to_json()
    return out


def baseline_rows(v1_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv_dicts(v1_dir / "validation_baselines.csv")
    controller_summary = load_json_if_exists(v1_dir / "summary.json")
    gate = controller_summary.get("validation_gate") or {}
    rows.append(
        {
            "baseline": "controller_v1_h200_replay",
            "validation_pass_count": int(round(_to_float(gate.get("validation_controller_pass_rate")) * 127)),
            "validation_total": 127,
            "validation_pass_rate": gate.get("validation_controller_pass_rate"),
            "validation_wins_vs_primary": 0,
            "validation_losses_vs_primary": 0,
            "validation_intervention_count": 0,
            "validation_mean_cost_sec": gate.get("validation_controller_mean_cost_sec"),
        }
    )
    return rows


def validation_gate(row: Mapping[str, Any], v6_baseline: Mapping[str, Any], controller_v1: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    pass_count = int(row.get("validation_pass_count") or 0)
    v6_pass_count = int(float(v6_baseline.get("validation_pass_count") or 0))
    wins = int(row.get("validation_wins_vs_primary") or 0)
    losses = int(row.get("validation_losses_vs_primary") or 0)
    interventions = int(row.get("validation_intervention_count") or 0)
    population_upper = row.get("validation_population_harm_upper95")
    population_ok = population_upper is not None and float(population_upper) <= 0.05
    bucket = row.get("validation_bucket_summary") or {}
    short = bucket.get("<=8") or {}
    short_ok = int(short.get("losses", 0)) <= int(short.get("wins", 0))
    mean_cost = _to_float(row.get("validation_mean_cost_sec"), 999.0)
    v6_cost = _to_float(v6_baseline.get("validation_mean_cost_sec"), 999.0)
    v1_interventions = int(controller_v1.get("validation_intervention_count") or 0)
    long_gain = sum(int((bucket.get(name) or {}).get("wins", 0)) - int((bucket.get(name) or {}).get("losses", 0)) for name in ("17-24", "25+"))
    opportunity = (
        pass_count >= v6_pass_count + 2
        or (long_gain >= 2 and pass_count >= v6_pass_count)
        or (pass_count >= v6_pass_count and mean_cost <= 0.9 * v6_cost)
        or (interventions > v1_interventions and wins >= losses and population_ok)
    )
    flags = {
        "nonzero_intervention": interventions > 0,
        "population_harm_upper95_le_5pct": population_ok,
        "wins_gt_losses": wins > losses,
        "not_below_h200_v6": pass_count >= v6_pass_count,
        "short_bucket_no_net_regression": short_ok,
        "opportunity_condition": opportunity,
        "long_bucket_net_gain": long_gain,
    }
    return all(flags.values()), flags


def render_report(summary: Mapping[str, Any], validation_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# H200 Controller V2 Validation",
        "",
        f"controller_verdict: `{summary['controller_verdict']}`",
        f"validation_gate_passed: `{summary['validation_gate_passed']}`",
        f"test_decision: `{summary['test_decision']}`",
        "",
        "## Variants",
        "",
        "| Variant | Feature | Pass | Wins | Losses | Interventions | Pop Harm Upper95 | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in validation_rows:
        lines.append(
            f"| `{row['variant']}` | `{row['feature_variant']}` | {row['validation_pass_count']}/{row['validation_total']} | "
            f"{row['validation_wins_vs_primary']} | {row['validation_losses_vs_primary']} | {row['validation_intervention_count']} | "
            f"{row.get('validation_population_harm_upper95')} | `{row.get('gate_passed')}` |"
        )
    if summary.get("selected_policy"):
        lines.extend(["", "## Selected Policy", "", f"`{json.dumps(summary['selected_policy'], sort_keys=True)}`"])
    lines.extend(
        [
            "",
            "Frozen test remains sealed unless validation gate passes.",
            "",
        ]
    )
    return "\n".join(lines)


def feature_schema_payload() -> dict[str, Any]:
    return {
        "feature_variants": {name: feature_columns(name) for name in FEATURE_VARIANTS},
        "action_feature_columns": ["candidate_actual_canvas", "canvas_delta_from_primary", "normalized_canvas_cost", "action_rank"],
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "test_features_materialized": False,
        "source": "H200 train/calibration/validation action_training_table only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate bounded H200 Controller V2 variants.")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-root", default="analysis_outputs")
    parser.add_argument("--v1-dir", default=V1_DIR)
    parser.add_argument("--bank-dir", default=BANK_DIR)
    parser.add_argument("--feasibility-dir", default=FEASIBILITY_DIR)
    parser.add_argument("--test-lock", default=TEST_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"controller_v2_h200_{timestamp}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    v1_dir = Path(args.v1_dir)
    action_rows = add_action_rank(parse_feature_table(v1_dir / "action_training_table.csv"))
    action_rows = [row for row in action_rows if row["split"] in {"train", "calibration", "validation"}]
    task_rows = task_rows_from_action_rows(action_rows)
    primary_by_task = {str(row["task_id"]): row for row in action_rows if row["action"] == "KEEP_PRIMARY"}
    validation_rows: list[dict[str, Any]] = []
    train_cv_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    conditional_curve: list[dict[str, Any]] = []
    population_curve: list[dict[str, Any]] = []
    model_payloads: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    action_selection: list[dict[str, Any]] = []

    baselines = baseline_rows(v1_dir)
    v6_baseline = next(row for row in baselines if row["baseline"] == "v6")
    controller_v1 = next(row for row in baselines if row["baseline"] == "controller_v1_h200_replay")

    for variant in VARIANTS:
        allowed_features = MAIN_FEATURE_VARIANTS + (() if variant != "pairwise_only" else NEGATIVE_CONTROL_FEATURE_VARIANTS)
        for feature_variant in allowed_features:
            models, cv_rows = train_variant(task_rows, action_rows, variant, feature_variant)
            train_cv_rows.extend({"variant": variant, **row} for row in cv_rows)
            model_payloads[f"{variant}:{feature_variant}"] = serialize_models(models)
            operating, cond, pop = calibrate_policy(
                variant=variant,
                feature_variant=feature_variant,
                task_rows=task_rows,
                action_rows=action_rows,
                models=models,
                primary_rows=primary_by_task,
            )
            calibration_rows.append(operating)
            conditional_curve.extend(cond)
            population_curve.extend(pop)
            selected = select_for_variant(
                variant=variant,
                feature_variant=feature_variant,
                task_rows=task_rows,
                action_rows=action_rows,
                models=models,
                split="validation",
                threshold=float(operating.get("threshold") or 999.0),
                harm_threshold=float(operating.get("harm_threshold") or 0.05),
            )
            metrics = evaluate_selection(selected, primary_by_task)
            row = {
                "variant": variant,
                "feature_variant": feature_variant,
                "threshold": operating.get("threshold"),
                "harm_threshold": operating.get("harm_threshold"),
                **{f"validation_{key}": value for key, value in metrics.items() if key != "bucket_summary"},
                "validation_bucket_summary": metrics["bucket_summary"],
            }
            gate_passed, gate_flags = validation_gate(row, v6_baseline, controller_v1)
            row.update(gate_flags)
            row["gate_passed"] = gate_passed
            validation_rows.append(row)
            for selected_row in selected:
                action_selection.append(
                    {
                        "variant": variant,
                        "feature_variant": feature_variant,
                        "task_id": selected_row["task_id"],
                        "split": selected_row["split"],
                        "selected_action": selected_row["selected_action"],
                        "intervened": selected_row["intervened"],
                        "action_passed": selected_row["action_passed"],
                        "primary_passed": selected_row["primary_passed"],
                    }
                )
                predictions.append(
                    {
                        "variant": variant,
                        "feature_variant": feature_variant,
                        "task_id": selected_row["task_id"],
                        "p_recover": selected_row.get("p_recover"),
                        "pairwise_score": selected_row.get("pairwise_score"),
                        "harm_prob": selected_row.get("harm_prob"),
                        "combined_score": selected_row.get("combined_score"),
                    }
                )

    passing = [row for row in validation_rows if row["gate_passed"]]
    selected_policy = None
    if passing:
        selected = max(
            passing,
            key=lambda row: (
                int(row["validation_pass_count"]),
                int(row["validation_wins_vs_primary"]) - int(row["validation_losses_vs_primary"]),
                -_to_float(row["validation_population_harm_upper95"], 1.0),
                -_to_float(row["validation_mean_cost_sec"], 999.0),
            ),
        )
        selected_policy = {
            "variant": selected["variant"],
            "feature_variant": selected["feature_variant"],
            "threshold": selected["threshold"],
            "harm_threshold": selected["harm_threshold"],
            "validation_pass_count": selected["validation_pass_count"],
            "validation_wins_vs_primary": selected["validation_wins_vs_primary"],
            "validation_losses_vs_primary": selected["validation_losses_vs_primary"],
        }
        verdict = "second_generation_validation_passed"
    else:
        best = max(validation_rows, key=lambda row: (int(row["validation_pass_count"]), int(row["validation_wins_vs_primary"]) - int(row["validation_losses_vs_primary"])))
        if int(best["validation_intervention_count"]) == 0:
            verdict = "second_generation_no_signal_test_sealed"
        elif int(best["validation_wins_vs_primary"]) <= int(best["validation_losses_vs_primary"]):
            verdict = "feature_ranking_failure_test_sealed"
        elif best.get("population_harm_upper95_le_5pct") is False:
            verdict = "risk_certification_limited_test_sealed"
        else:
            verdict = "second_generation_weak_signal_test_sealed"

    test_lock = load_json_if_exists(Path(args.test_lock))
    summary = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "controller_verdict": verdict,
        "validation_gate_passed": bool(passing),
        "test_decision": "authorized_once" if passing else "sealed",
        "test_lock_status": test_lock.get("test_status"),
        "test_evaluation_count": test_lock.get("test_evaluation_count"),
        "selected_policy": selected_policy,
        "feasibility_dir": args.feasibility_dir,
        "v1_dir": args.v1_dir,
        "bank_dir": args.bank_dir,
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }

    (output_dir / "feasibility_audit_link.json").write_text(json.dumps({"path": args.feasibility_dir}, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "target_schema.json").write_text(json.dumps({"targets": ["recoverable", "harmable", "minimum_passing_action", "pairwise_action_preference"], "test_labels_used": False}, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "feature_schema.json").write_text(json.dumps(feature_schema_payload(), indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_dir / "train_cv_results.csv", train_cv_rows)
    (output_dir / "ordinal_model.json").write_text(json.dumps({key: value for key, value in model_payloads.items() if "ordinal" in value}, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "pairwise_model.json").write_text(json.dumps({key: value.get("pairwise") for key, value in model_payloads.items() if value.get("pairwise")}, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "harm_guard_model.json").write_text(json.dumps({key: value.get("harm") for key, value in model_payloads.items() if value.get("harm")}, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_dir / "calibration_results.csv", calibration_rows)
    write_csv(output_dir / "conditional_risk_curve.csv", conditional_curve)
    write_csv(output_dir / "population_risk_curve.csv", population_curve)
    write_csv(output_dir / "validation_predictions.csv", predictions)
    write_csv(output_dir / "validation_action_selection.csv", action_selection)
    write_csv(output_dir / "validation_baselines.csv", baselines)
    write_csv(output_dir / "validation_ablation.csv", validation_rows)
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if selected_policy:
        payload = {
            **selected_policy,
            "models": model_payloads[f"{selected_policy['variant']}:{selected_policy['feature_variant']}"],
            "feature_schema_sha256": hashlib.sha256((output_dir / "feature_schema.json").read_bytes()).hexdigest(),
            "action_schema_source": args.bank_dir,
            "frozen_commit": current_commit(),
        }
        (output_dir / "selected_policy.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary, validation_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "controller_verdict": verdict, "validation_gate_passed": bool(passing)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
