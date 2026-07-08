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
from typing import Any, Mapping, Sequence

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
    wilson_upper,
    write_csv,
)


V1_DIR = "analysis_outputs/controller_validation_h200_20260707_v1_replay"
V2_DIR = "analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation"
TEST_LOCK = "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"
TOPK_VALUES = (5, 10, 15, 20, 30, 40, 60)
LOGISTIC_C = (0.1, 1.0, 10.0)
FEATURE_VARIANT_NAMES = ("probe_only", "probe_trace_fused")
FAMILIES = ("targeted_missed_long", "two_stage_rejector", "oracle_win_distillation")
FAMILY_DISPLAY = {
    "targeted_missed_long": "Family A",
    "two_stage_rejector": "Family B",
    "oracle_win_distillation": "Family C",
}
ACTION_FEATURES = ("candidate_actual_canvas", "canvas_delta_from_primary", "normalized_canvas_cost", "action_rank")
FROZEN_TEST_ALLOWED_FAMILIES = {"targeted_missed_long", "two_stage_rejector"}


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
    return None if den == 0 else float(num) / float(den)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def parse_table(path: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in read_csv_rows(path)]
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


def feature_columns(feature_variant: str, *, action: bool = False) -> list[str]:
    cols = list(FEATURE_VARIANTS[feature_variant])
    if action:
        cols += list(ACTION_FEATURES)
    leaked = [col for col in cols if col in FORBIDDEN_FEATURES]
    if leaked:
        raise ValueError(f"forbidden features in schema: {leaked}")
    return cols


def add_action_rank(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        enriched = dict(row)
        enriched["action_rank"] = ACTION_RANK[str(row["action"])]
        out.append(enriched)
    return out


def grouped_task_rows(action_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[str(row["task_id"])].append(row)
    task_rows: list[dict[str, Any]] = []
    for task_id, rows in sorted(grouped.items()):
        keep = next(row for row in rows if row["action"] == "KEEP_PRIMARY")
        expansions = [row for row in rows if row["action"] != "KEEP_PRIMARY"]
        primary_pass = _to_bool(keep["action_passed"])
        any_expand_pass = any(_to_bool(row["action_passed"]) for row in expansions)
        any_expand_fail = any(not _to_bool(row["action_passed"]) for row in expansions)
        min_rank = len(ACTION_ORDER)
        min_action = "UNRECOVERABLE"
        for action in ACTION_ORDER:
            row = next(item for item in rows if item["action"] == action)
            if _to_bool(row["action_passed"]):
                min_action = "KEEP" if action == "KEEP_PRIMARY" else action
                min_rank = ACTION_RANK[action]
                break
        task_rows.append(
            {
                **{key: keep.get(key) for key in keep if key not in {"action", "action_passed", "benefit_label", "harm_label", "underallocation_label", "generated_text_sha256", "inference_cost_sec"}},
                "task_id": task_id,
                "primary_passed": primary_pass,
                "primary_failed": not primary_pass,
                "recoverable": (not primary_pass) and any_expand_pass,
                "harmable": primary_pass and any_expand_fail,
                "oracle_win": (not primary_pass) and any_expand_pass,
                "minimum_passing_action": min_action,
                "minimum_passing_rank": min_rank,
            }
        )
    return task_rows


def rows_by_task_action(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["task_id"]), str(row["action"])): row for row in rows}


def rows_by_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["task_id"]): row for row in rows}


def fit_task_model(task_rows: Sequence[Mapping[str, Any]], label: str, feature_variant: str, c_value: float) -> LogisticModel:
    train = [row for row in task_rows if row["split"] == "train"]
    model = LogisticModel(feature_columns(feature_variant), c_value=c_value)
    model.fit(train, [1 if row[label] else 0 for row in train])
    return model


def fit_action_model(action_rows: Sequence[Mapping[str, Any]], label: str, feature_variant: str, c_value: float) -> LogisticModel:
    train = [row for row in action_rows if row["split"] == "train" and row["action"] != "KEEP_PRIMARY"]
    model = LogisticModel(feature_columns(feature_variant, action=True), c_value=c_value)
    model.fit(train, [1 if row[label] else 0 for row in train])
    return model


def cv_select_c(task_rows: Sequence[Mapping[str, Any]], label: str, feature_variant: str) -> tuple[float, list[dict[str, Any]]]:
    train = [row for row in task_rows if row["split"] == "train"]
    groups = sorted({str(row["task_group"]) for row in train})
    fold_by_group = {group: idx % 5 for idx, group in enumerate(groups)}
    rows: list[dict[str, Any]] = []
    best_c = LOGISTIC_C[0]
    best_auc = -1.0
    for c_value in LOGISTIC_C:
        labels_all: list[int] = []
        scores_all: list[float] = []
        for fold in range(5):
            fit_rows = [row for row in train if fold_by_group[str(row["task_group"])] != fold]
            eval_rows = [row for row in train if fold_by_group[str(row["task_group"])] == fold]
            model = LogisticModel(feature_columns(feature_variant), c_value=c_value)
            model.fit(fit_rows, [1 if row[label] else 0 for row in fit_rows])
            labels = [1 if row[label] else 0 for row in eval_rows]
            scores = [model.predict_one(row) for row in eval_rows]
            labels_all.extend(labels)
            scores_all.extend(scores)
        auc = auc_score(labels_all, scores_all)
        rows.append({"feature_variant": feature_variant, "target": label, "C": c_value, "grouped_cv_auc": auc})
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_c = c_value
    return best_c, rows


def train_family_models(task_rows: Sequence[Mapping[str, Any]], action_rows: Sequence[Mapping[str, Any]], family: str, feature_variant: str) -> tuple[dict[str, LogisticModel], list[dict[str, Any]]]:
    cv_rows: list[dict[str, Any]] = []
    recover_c, rows = cv_select_c(task_rows, "recoverable", feature_variant)
    cv_rows.extend({"family": family, **row} for row in rows)
    models = {
        "primary_fail": fit_task_model(task_rows, "primary_failed", feature_variant, recover_c),
        "recover": fit_task_model(task_rows, "recoverable", feature_variant, recover_c),
        "harmable": fit_task_model(task_rows, "harmable", feature_variant, recover_c),
        "action_pass": fit_action_model(action_rows, "action_passed", feature_variant, recover_c),
        "action_harm": fit_action_model(action_rows, "harm_label", feature_variant, recover_c),
    }
    if family == "oracle_win_distillation":
        oracle_c, rows = cv_select_c(task_rows, "oracle_win", feature_variant)
        cv_rows.extend({"family": family, **row} for row in rows)
        models["oracle_win"] = fit_task_model(task_rows, "oracle_win", feature_variant, oracle_c)
    cv_rows.append({"family": family, "feature_variant": feature_variant, "selected_C": recover_c})
    return models, cv_rows


def choose_lowest_safe_action(task_id: str, by_key: Mapping[tuple[str, str], Mapping[str, Any]], action_pass_model: LogisticModel, action_harm_model: LogisticModel, min_pass_prob: float = 0.45, max_harm_prob: float = 0.45) -> tuple[str, dict[str, float]]:
    best_action = "KEEP_PRIMARY"
    best_score = -999.0
    best_meta: dict[str, float] = {"action_pass_prob": 0.0, "action_harm_prob": 0.0, "action_score": 0.0}
    for action in EXPANSION_ACTIONS:
        row = by_key[(task_id, action)]
        pass_prob = action_pass_model.predict_one(row)
        harm_prob = action_harm_model.predict_one(row)
        score = pass_prob - 0.75 * harm_prob - 0.015 * ACTION_RANK[action]
        if pass_prob >= min_pass_prob and harm_prob <= max_harm_prob:
            return action, {"action_pass_prob": pass_prob, "action_harm_prob": harm_prob, "action_score": score}
        if score > best_score:
            best_action = action
            best_score = score
            best_meta = {"action_pass_prob": pass_prob, "action_harm_prob": harm_prob, "action_score": score}
    return best_action, best_meta


def score_validation_candidates(
    *,
    family: str,
    feature_variant: str,
    task_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, LogisticModel],
) -> list[dict[str, Any]]:
    by_key = rows_by_task_action(action_rows)
    validation_tasks = [row for row in task_rows if row["split"] == "validation"]
    candidates: list[dict[str, Any]] = []
    for task in validation_tasks:
        task_id = str(task["task_id"])
        p_fail = models["primary_fail"].predict_one(task)
        p_recover = models["recover"].predict_one(task)
        p_harmable = models["harmable"].predict_one(task)
        action, action_meta = choose_lowest_safe_action(task_id, by_key, models["action_pass"], models["action_harm"])
        if family == "targeted_missed_long":
            score = p_fail * p_recover * (1.0 - p_harmable) + 0.05 * action_meta["action_pass_prob"]
        elif family == "two_stage_rejector":
            score = (p_recover - 0.65 * p_harmable + 0.15 * p_fail) + 0.25 * action_meta["action_score"]
        else:
            p_oracle = models["oracle_win"].predict_one(task)
            score = p_oracle + 0.25 * p_recover - 0.25 * p_harmable
        candidates.append(
            {
                "family": family,
                "feature_variant": feature_variant,
                "task_id": task_id,
                "task_group": task["task_group"],
                "oracle_bucket": task["oracle_bucket"],
                "primary_passed": task["primary_passed"],
                "recoverable": task["recoverable"],
                "oracle_win": task["oracle_win"],
                "score": score,
                "p_primary_fail": p_fail,
                "p_recover": p_recover,
                "p_harmable": p_harmable,
                "selected_action_if_intervened": action,
                **action_meta,
            }
        )
    return candidates


def evaluate_selected(selected: Sequence[Mapping[str, Any]], primary_by_task: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    wins = losses = ties_pass = ties_fail = interventions = harms = benefits = short_losses = missed_long_wins = oracle_hits = 0
    costs: list[float] = []
    canvas_counts: Counter[str] = Counter()
    for row in selected:
        task_id = str(row["task_id"])
        before = _to_bool(primary_by_task[task_id]["action_passed"])
        after = _to_bool(row["action_passed"])
        intervened = str(row["selected_action"]) != "KEEP_PRIMARY"
        bucket = str(row.get("oracle_bucket"))
        costs.append(_to_float(row.get("inference_cost_sec")))
        canvas_counts[str(int(_to_float(row.get("actual_canvas"))))] += 1
        if after and not before:
            wins += 1
            benefits += int(intervened)
            if bucket in {"17-24", "25+"}:
                missed_long_wins += 1
        elif before and not after:
            losses += 1
            harms += int(intervened)
            if bucket == "<=8":
                short_losses += 1
        elif after:
            ties_pass += 1
        else:
            ties_fail += 1
        interventions += int(intervened)
        oracle_hits += int(intervened and _to_bool(row.get("oracle_win")))
    total = len(selected)
    return {
        "interventions": interventions,
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
        "pass_count": wins + ties_pass,
        "pass_rate": _safe_rate(wins + ties_pass, total),
        "population_harm": _safe_rate(harms, total),
        "population_harm_upper95": clopper_pearson_upper(harms, total),
        "population_harm_wilson_upper95": wilson_upper(harms, total),
        "conditional_harm": _safe_rate(harms, interventions),
        "conditional_harm_upper95": clopper_pearson_upper(harms, interventions),
        "short_bucket_losses": short_losses,
        "missed_long_wins": missed_long_wins,
        "selected_canvas_distribution": dict(sorted(canvas_counts.items())),
        "average_cost": _mean(costs),
        "topk_oracle_win_hits": oracle_hits,
    }


def selected_for_topk(candidates: Sequence[Mapping[str, Any]], action_by_key: Mapping[tuple[str, str], Mapping[str, Any]], primary_by_task: Mapping[str, Mapping[str, Any]], k: int) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda row: float(row["score"]), reverse=True)
    chosen = {str(row["task_id"]): str(row["selected_action_if_intervened"]) for row in ranked[:k]}
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: str(row["task_id"])):
        task_id = str(candidate["task_id"])
        action = chosen.get(task_id, "KEEP_PRIMARY")
        row = dict(action_by_key[(task_id, action)])
        row.update(
            {
                "family": candidate["family"],
                "feature_variant": candidate["feature_variant"],
                "score": candidate["score"],
                "selected_action": action,
                "oracle_win": candidate["oracle_win"],
            }
        )
        selected.append(row)
    return selected


def gate_flags(curve_row: Mapping[str, Any], v6_pass_count: int, v6_cost: float, family: str) -> dict[str, bool]:
    exploratory = (
        int(curve_row["interventions"]) > 0
        and int(curve_row["pass_count"]) >= v6_pass_count
        and int(curve_row["wins"]) > int(curve_row["losses"])
        and float(curve_row["population_harm_upper95"] or 1.0) <= 0.075
        and int(curve_row["short_bucket_losses"]) <= 0
        and (
            int(curve_row["net"]) >= 2
            or (int(curve_row["missed_long_wins"]) >= 2 and int(curve_row["pass_count"]) >= v6_pass_count)
            or (int(curve_row["pass_count"]) >= v6_pass_count and _to_float(curve_row["average_cost"], 999.0) <= 0.9 * v6_cost)
        )
    )
    frozen = (
        exploratory
        and family in FROZEN_TEST_ALLOWED_FAMILIES
        and float(curve_row["population_harm_upper95"] or 1.0) <= 0.05
        and int(curve_row["net"]) >= 2
    )
    return {"exploratory_gate_passed": exploratory, "frozen_test_gate_passed": frozen}


def baseline_rows(v1_dir: Path, v2_dir: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in read_csv_rows(v1_dir / "validation_baselines.csv")]
    v1_summary = load_json_if_exists(v1_dir / "summary.json")
    v1_gate = v1_summary.get("validation_gate") or {}
    rows.append(
        {
            "baseline": "controller_v1_h200_replay",
            "validation_pass_count": int(round(_to_float(v1_gate.get("validation_controller_pass_rate")) * 127)),
            "validation_total": 127,
            "validation_pass_rate": v1_gate.get("validation_controller_pass_rate"),
            "validation_wins_vs_primary": 0,
            "validation_losses_vs_primary": 0,
            "validation_intervention_count": 0,
            "validation_mean_cost_sec": v1_gate.get("validation_controller_mean_cost_sec"),
        }
    )
    if (v2_dir / "validation_ablation.csv").exists():
        v2_rows = read_csv_rows(v2_dir / "validation_ablation.csv")
        nonzero = [row for row in v2_rows if int(float(row.get("validation_intervention_count") or 0)) > 0]
        best = max(nonzero, key=lambda row: int(float(row.get("validation_pass_count") or 0)), default=None)
        if best:
            rows.append(
                {
                    "baseline": "controller_v2_best_nonzero",
                    "validation_pass_count": best.get("validation_pass_count"),
                    "validation_total": best.get("validation_total"),
                    "validation_pass_rate": best.get("validation_pass_rate"),
                    "validation_wins_vs_primary": best.get("validation_wins_vs_primary"),
                    "validation_losses_vs_primary": best.get("validation_losses_vs_primary"),
                    "validation_intervention_count": best.get("validation_intervention_count"),
                    "validation_mean_cost_sec": best.get("validation_mean_cost_sec"),
                }
            )
    return rows


def validate_bank_contract(bank_dir: Path, action_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not bank_dir.exists():
        raise FileNotFoundError(bank_dir)
    summary = load_json_if_exists(bank_dir / "summary.json")
    schema = load_json_if_exists(bank_dir / "action_schema.json")
    expected_rows = int(summary.get("row_count") or 0)
    expected_tasks = int(summary.get("task_count") or 0)
    actual_rows = len(action_rows)
    actual_tasks = len({str(row["task_id"]) for row in action_rows})
    if expected_rows and expected_rows != actual_rows:
        raise ValueError(f"bank row_count mismatch: summary={expected_rows} table={actual_rows}")
    if expected_tasks and expected_tasks != actual_tasks:
        raise ValueError(f"bank task_count mismatch: summary={expected_tasks} table={actual_tasks}")
    actions = [item.get("action") for item in schema.get("actions", [])]
    if actions and tuple(actions) != ACTION_ORDER:
        raise ValueError(f"unexpected action schema: {actions}")
    return {
        "bank_dir": str(bank_dir),
        "bank_summary_row_count": expected_rows or actual_rows,
        "bank_summary_task_count": expected_tasks or actual_tasks,
        "bank_action_schema_version": schema.get("version"),
    }


def validate_test_lock(test_lock: Mapping[str, Any]) -> dict[str, Any]:
    status = test_lock.get("test_status")
    count = int(test_lock.get("test_evaluation_count") or 0)
    if status != "sealed" or count != 0:
        raise ValueError(f"frozen test lock is not sealed/0: status={status!r} count={count}")
    return {"test_status": status, "test_evaluation_count": count}


def route_decision(best_exploratory: Mapping[str, Any] | None, best_frozen: Mapping[str, Any] | None) -> str:
    if best_frozen is not None:
        return "v3_controller_positive_validation_signal"
    if best_exploratory is not None:
        return "weak_validation_signal_test_sealed"
    return "controller_route_negative_test_sealed"


def render_report(summary: Mapping[str, Any], best_by_family: Mapping[str, Mapping[str, Any] | None]) -> str:
    lines = [
        "# H200 Controller V3 Candidate Screen",
        "",
        f"route_decision: `{summary['route_decision']}`",
        f"exploratory_gate_passed: `{summary['exploratory_gate_passed']}`",
        f"frozen_test_gate_passed: `{summary['frozen_test_gate_passed']}`",
        f"test_status: `{summary['test_status']}`",
        "",
        "## Family Best Top-k",
        "",
        "| Family | Feature | k | Pass | Wins | Losses | Net | Pop Harm Upper95 | Oracle-win Hits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        row = best_by_family.get(family)
        if not row:
            lines.append(f"| `{family}` | failed |  |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| `{family}` | `{row['feature_variant']}` | {row['k']} | {row['pass_count']}/127 | "
            f"{row['wins']} | {row['losses']} | {row['net']} | {row['population_harm_upper95']} | {row['topk_oracle_win_hits']} |"
        )
    lines.extend(
        [
            "",
            "## Feature Safety",
            "",
            "- Forbidden feature check: passed.",
            "- Test labels/features/results were not materialized.",
            "- Family C is diagnostic and cannot alone authorize frozen test.",
            "",
        ]
    )
    if summary["route_decision"] == "controller_route_negative_test_sealed":
        lines.extend(
            [
                "## Route Recommendation",
                "",
                "- Stop further controller development on this HumanEval validation split.",
                "- Shift to diagnostic/mixed paper framing.",
                "- Validate the central claim on a second backbone or second regime.",
                "- Do not continue writing new controllers against the same validation split.",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H200 Controller V3 candidate screen over validation split.")
    parser.add_argument("--bank-dir", default="analysis_outputs/controller_action_bank_h200_20260707_tier1_offline")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-root", default="analysis_outputs")
    parser.add_argument("--v1-dir", default=V1_DIR)
    parser.add_argument("--v2-dir", default=V2_DIR)
    parser.add_argument("--test-lock", default=TEST_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"controller_v3_h200_{timestamp}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    v1_dir = Path(args.v1_dir)
    action_rows = add_action_rank(parse_table(v1_dir / "action_training_table.csv"))
    action_rows = [row for row in action_rows if row["split"] in {"train", "calibration", "validation"}]
    bank_contract = validate_bank_contract(Path(args.bank_dir), action_rows)
    test_lock = load_json_if_exists(Path(args.test_lock))
    lock_contract = validate_test_lock(test_lock)
    task_rows = grouped_task_rows(action_rows)
    action_by_key = rows_by_task_action(action_rows)
    primary_by_task = {str(row["task_id"]): row for row in action_rows if row["action"] == "KEEP_PRIMARY"}
    baselines = baseline_rows(v1_dir, Path(args.v2_dir))
    v6 = next(row for row in baselines if row["baseline"] == "v6")
    v6_pass_count = int(float(v6["validation_pass_count"]))
    v6_cost = _to_float(v6.get("validation_mean_cost_sec"), 999.0)

    config = {
        "bank_dir": args.bank_dir,
        "v1_dir": args.v1_dir,
        "v2_dir": args.v2_dir,
        "families": list(FAMILIES),
        "feature_variants": list(FEATURE_VARIANT_NAMES),
        "topk_values": list(TOPK_VALUES),
        "allowed_logistic_C": list(LOGISTIC_C),
        "tree_depth": [2, 3, 4],
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "note": "V3 implementation uses weighted logistic variants from the allowed model class; tree grids are recorded as allowed but not selected.",
        **bank_contract,
    }
    feature_schema = {
        "feature_variants": {name: feature_columns(name) for name in FEATURE_VARIANT_NAMES},
        "action_feature_columns": list(ACTION_FEATURES),
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "forbidden_feature_check_passed": True,
        "test_features_materialized": False,
    }

    model_registry: dict[str, Any] = {}
    train_cv_results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    action_selection_rows: list[dict[str, Any]] = []
    family_errors: dict[str, str] = {}

    for family in FAMILIES:
        for feature_variant in FEATURE_VARIANT_NAMES:
            try:
                models, cv_rows = train_family_models(task_rows, action_rows, family, feature_variant)
                train_cv_results.extend(cv_rows)
                model_registry[f"{family}:{feature_variant}"] = {name: model.to_json() for name, model in models.items()}
                candidates = score_validation_candidates(
                    family=family,
                    feature_variant=feature_variant,
                    task_rows=task_rows,
                    action_rows=action_rows,
                    models=models,
                )
                predictions.extend(candidates)
                for k in TOPK_VALUES:
                    selected = selected_for_topk(candidates, action_by_key, primary_by_task, k)
                    metrics = evaluate_selected(selected, primary_by_task)
                    row = {
                        "family": family,
                        "family_display": FAMILY_DISPLAY[family],
                        "feature_variant": feature_variant,
                        "k": k,
                        **metrics,
                    }
                    row.update(gate_flags(row, v6_pass_count, v6_cost, family))
                    topk_rows.append(row)
                    for item in selected:
                        action_selection_rows.append(
                            {
                                "family": family,
                                "feature_variant": feature_variant,
                                "k": k,
                                "task_id": item["task_id"],
                                "selected_action": item["selected_action"],
                                "action_passed": item["action_passed"],
                                "primary_passed": item["primary_passed"],
                                "score": item["score"],
                            }
                        )
            except Exception as exc:  # keep other families running as requested
                family_errors[family] = repr(exc)

    best_by_family: dict[str, Mapping[str, Any] | None] = {}
    for family in FAMILIES:
        rows = [row for row in topk_rows if row["family"] == family]
        best_by_family[family] = max(rows, key=lambda row: (int(row["pass_count"]), int(row["net"]), -_to_float(row["population_harm_upper95"], 1.0)), default=None)
    exploratory = [row for row in topk_rows if row.get("exploratory_gate_passed")]
    frozen = [row for row in topk_rows if row.get("frozen_test_gate_passed")]
    best_exploratory = max(exploratory, key=lambda row: (int(row["pass_count"]), int(row["net"])), default=None)
    best_frozen = max(frozen, key=lambda row: (int(row["pass_count"]), int(row["net"])), default=None)
    decision = route_decision(best_exploratory, best_frozen)
    validation_ablation = list(best_by_family.values())
    validation_ablation = [dict(row) for row in validation_ablation if row]
    v3_baseline_rows = [
        {
            "baseline": f"controller_v3_{row['family']}",
            "validation_pass_count": row["pass_count"],
            "validation_total": 127,
            "validation_pass_rate": row["pass_rate"],
            "validation_wins_vs_primary": row["wins"],
            "validation_losses_vs_primary": row["losses"],
            "validation_intervention_count": row["interventions"],
            "validation_mean_cost_sec": row["average_cost"],
            "feature_variant": row["feature_variant"],
            "k": row["k"],
            "population_harm_upper95": row["population_harm_upper95"],
            "conditional_harm_upper95": row["conditional_harm_upper95"],
            "exploratory_gate_passed": row["exploratory_gate_passed"],
            "frozen_test_gate_passed": row["frozen_test_gate_passed"],
        }
        for row in validation_ablation
    ]
    oracle_diag = [
        {
            "family": row["family"],
            "feature_variant": row["feature_variant"],
            "k": row["k"],
            "topk_oracle_win_hits": row["topk_oracle_win_hits"],
            "wins": row["wins"],
            "losses": row["losses"],
            "net": row["net"],
        }
        for row in topk_rows
        if row["family"] == "oracle_win_distillation"
    ]
    pareto = [
        {
            "family": row["family"],
            "feature_variant": row["feature_variant"],
            "k": row["k"],
            "pass_count": row["pass_count"],
            "population_harm_upper95": row["population_harm_upper95"],
            "average_cost": row["average_cost"],
            "net": row["net"],
        }
        for row in topk_rows
    ]
    summary = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "controller_v3_started": True,
        "controller_v3_completed": len(family_errors) == 0,
        "controller_v3_families_completed": [family for family in FAMILIES if family not in family_errors],
        "family_errors": family_errors,
        "controller_v3_output_dir": str(output_dir),
        "exploratory_gate_passed": bool(best_exploratory),
        "frozen_test_gate_passed": bool(best_frozen),
        "selected_exploratory_policy": best_exploratory,
        "selected_frozen_policy": best_frozen,
        "route_decision": decision,
        **lock_contract,
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }

    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "model_registry.json").write_text(json.dumps(model_registry, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_dir / "train_cv_results.csv", train_cv_results)
    write_csv(output_dir / "topk_policy_curves.csv", topk_rows)
    write_csv(output_dir / "validation_predictions.csv", predictions)
    write_csv(output_dir / "validation_action_selection.csv", action_selection_rows)
    write_csv(output_dir / "validation_ablation.csv", validation_ablation)
    write_csv(output_dir / "validation_baselines.csv", [*baselines, *v3_baseline_rows])
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_dir / "oracle_win_distillation.csv", oracle_diag)
    write_csv(output_dir / "cost_risk_pareto.csv", pareto)
    (output_dir / "report.md").write_text(render_report(summary, best_by_family), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "route_decision": decision, "families_completed": summary["controller_v3_families_completed"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
