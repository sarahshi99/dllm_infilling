#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import builtins
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import tokenize
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


FORBIDDEN_DEPLOYABLE_FEATURES = {
    "reference_code",
    "reference_middle",
    "oracle_length",
    "oracle_mask_length",
    "unit_test_result",
    "passed",
    "pass_fail",
    "error_type",
    "task_id",
    "task_group",
    "split",
    "split_label",
    "row_key",
}
CONTROL_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
BUILTIN_NAMES = set(dir(builtins))
COMMON_FEATURES = ["canvas_log2", "seed_value"]
PREFIX_FEATURES = [
    "prefix_candidate_def_use_count",
    "prefix_candidate_use_coverage",
    "prefix_available_use_coverage",
    "prefix_boundary_indent_match",
]
SUFFIX_FEATURES = [
    "suffix_required_identifier_count",
    "suffix_required_recovered_count",
    "suffix_required_recovery",
    "candidate_suffix_def_use_count",
    "suffix_boundary_indent_match",
]
COMBINED_FEATURES = PREFIX_FEATURES + SUFFIX_FEATURES + [
    "full_parse_passed",
    "candidate_control_structure_count",
    "combined_bridge_satisfaction",
    "semantic_horizon_prediction",
]
TOKEN_LENGTH_FEATURES = [
    "candidate_middle_tokens",
    "canvas_tokens",
    "candidate_canvas_fill_ratio",
    "prefix_tokens",
    "suffix_tokens",
]
CONFIDENCE_FEATURES = ["ordinary_confidence"]
SUPERVISED_PROBE_FEATURE_FAMILIES = {
    "prefix_only": COMMON_FEATURES + PREFIX_FEATURES,
    "suffix_only": COMMON_FEATURES + SUFFIX_FEATURES,
    "combined_bridge": COMMON_FEATURES + COMBINED_FEATURES,
    "token_length": COMMON_FEATURES + TOKEN_LENGTH_FEATURES,
    "ordinary_confidence": COMMON_FEATURES + CONFIDENCE_FEATURES,
}
DEPLOYABLE_PROXY_SCORE_KEYS = (
    "deployable_proxy_prefix_only",
    "deployable_proxy_suffix_only",
    "deployable_proxy_token_canvas",
    "deployable_proxy_ordinary_confidence",
    "deployable_proxy_combined",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def mean(values: Iterable[float]) -> float:
    usable = [float(value) for value in values if math.isfinite(float(value))]
    return sum(usable) / len(usable) if usable else float("nan")


def safe_ratio(numerator: float, denominator: float, empty: float = 1.0) -> float:
    return float(numerator) / float(denominator) if denominator else float(empty)


def hash_group(group: str) -> str:
    return hashlib.sha256(group.encode("utf-8")).hexdigest()[:16]


def validate_feature_names(feature_names: Sequence[str]) -> None:
    forbidden = sorted(set(feature_names) & FORBIDDEN_DEPLOYABLE_FEATURES)
    if forbidden:
        raise ValueError(f"Forbidden deployable feature(s): {forbidden}")


def clipped(value: Any) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def deployable_bridge_formula_spec() -> dict[str, Any]:
    return {
        "mechanism_name": "AST/def-use bridge proxy V0",
        "provenance": "fixed_before_outcomes",
        "parameter_source": "hand_fixed_constants_only",
        "fit_inputs": [],
        "selection_inputs": [
            "prefix_candidate_use_coverage",
            "prefix_available_use_coverage",
            "prefix_boundary_indent_match",
            "prefix_candidate_def_use_count",
            "suffix_required_recovery",
            "suffix_boundary_indent_match",
            "candidate_suffix_def_use_count",
            "full_parse_passed",
            "candidate_control_structure_count",
            "ordinary_confidence",
            "candidate_canvas_fill_ratio",
        ],
        "scores": {
            "deployable_proxy_prefix_only": "mean(prefix_candidate_use_coverage, prefix_available_use_coverage, prefix_boundary_indent_match, min(prefix_candidate_def_use_count/3, 1))",
            "deployable_proxy_suffix_only": "mean(suffix_required_recovery, suffix_boundary_indent_match, min(candidate_suffix_def_use_count/3, 1))",
            "deployable_proxy_token_canvas": "clip(candidate_canvas_fill_ratio, 0, 1)",
            "deployable_proxy_ordinary_confidence": "clip(ordinary_confidence, 0, 1)",
            "deployable_proxy_combined": "0.30*prefix_only + 0.30*suffix_only + 0.20*full_parse_passed + 0.10*min(candidate_control_structure_count/3,1) + 0.10*ordinary_confidence",
        },
        "scope_boundary": "This is a deterministic AST/def-use/boundary proxy, not full program-state analysis, backward obligations, bridge anchors, or denoising intervention.",
        "offline_gate_note": "Functional outcomes are used after deterministic selection only to evaluate and authorize the preregistered proxy; they do not construct, fit, or alter its score.",
    }


def deployable_proxy_scores(row: Mapping[str, Any]) -> dict[str, float]:
    prefix = mean(
        [
            clipped(row.get("prefix_candidate_use_coverage")),
            clipped(row.get("prefix_available_use_coverage")),
            clipped(row.get("prefix_boundary_indent_match")),
            clipped(float(row.get("prefix_candidate_def_use_count", 0.0) or 0.0) / 3.0),
        ]
    )
    suffix = mean(
        [
            clipped(row.get("suffix_required_recovery")),
            clipped(row.get("suffix_boundary_indent_match")),
            clipped(float(row.get("candidate_suffix_def_use_count", 0.0) or 0.0) / 3.0),
        ]
    )
    token_canvas = clipped(row.get("candidate_canvas_fill_ratio"))
    confidence = clipped(row.get("ordinary_confidence"))
    combined = (
        0.30 * prefix
        + 0.30 * suffix
        + 0.20 * clipped(row.get("full_parse_passed"))
        + 0.10 * clipped(float(row.get("candidate_control_structure_count", 0.0) or 0.0) / 3.0)
        + 0.10 * confidence
    )
    return {
        "deployable_proxy_prefix_only": prefix,
        "deployable_proxy_suffix_only": suffix,
        "deployable_proxy_token_canvas": token_canvas,
        "deployable_proxy_ordinary_confidence": confidence,
        "deployable_proxy_combined": combined,
    }


def forbidden_feature_audit() -> dict[str, Any]:
    return {
        "passed": True,
        "forbidden_features": sorted(FORBIDDEN_DEPLOYABLE_FEATURES),
        "labels_used_for_offline_evaluation": True,
        "reference_used_for_offline_evaluation": True,
        "supervised_probe_diagnostic": {
            "outcome_labels_used_for_fit": True,
            "reference_used_for_fit": False,
            "outcome_labels_used_for_selection": False,
            "reference_used_for_selection": False,
            "deployable_authorization_role": "none",
        },
        "deployable_bridge_proxy": {
            "outcome_labels_used_for_fit": False,
            "reference_used_for_fit": False,
            "outcome_labels_used_for_selection": False,
            "reference_used_for_selection": False,
            "fitted_parameters": False,
            "formula_provenance": "fixed_before_outcomes",
            "outcome_labels_used_for_offline_gate_evaluation": True,
        },
    }


def _token_names(text: str) -> set[str]:
    try:
        return {
            token.string
            for token in tokenize.generate_tokens(StringIO(text).readline)
            if token.type == tokenize.NAME and token.string not in {"True", "False", "None"}
        }
    except (tokenize.TokenError, IndentationError):
        return set()


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(text):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def _node_offset(node: ast.AST, offsets: Sequence[int]) -> int:
    return offsets[int(getattr(node, "lineno", 1)) - 1] + int(getattr(node, "col_offset", 0))


def _region(offset: int, prefix_end: int, middle_end: int) -> str:
    if offset < prefix_end:
        return "prefix"
    if offset < middle_end:
        return "middle"
    return "suffix"


def region_info(prefix: str, middle: str, suffix: str) -> tuple[dict[str, dict[str, set[str]]], bool]:
    full = prefix + middle + suffix
    info = {
        name: {"defs": set(), "uses": set(), "controls": set()}
        for name in ["prefix", "middle", "suffix"]
    }
    try:
        tree = ast.parse(full)
    except SyntaxError:
        info["prefix"]["uses"] = _token_names(prefix)
        info["middle"]["uses"] = _token_names(middle)
        info["suffix"]["uses"] = _token_names(suffix)
        return info, False
    offsets = _line_offsets(full)
    prefix_end = len(prefix)
    middle_end = prefix_end + len(middle)
    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        region = _region(_node_offset(node, offsets), prefix_end, middle_end)
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                info[region]["defs"].add(node.id)
            elif isinstance(node.ctx, ast.Load):
                info[region]["uses"].add(node.id)
        elif isinstance(node, ast.arg):
            info[region]["defs"].add(node.arg)
        elif isinstance(node, CONTROL_NODES):
            info[region]["controls"].add(type(node).__name__)
    return info, True


def semantic_units(prefix: str, middle: str, suffix: str) -> set[str]:
    info, _ = region_info(prefix, middle, suffix)
    units = {f"def:{name}" for name in info["middle"]["defs"]}
    units.update(f"use:{name}" for name in info["middle"]["uses"])
    units.update(f"control:{name}" for name in info["middle"]["controls"])
    prefix_bridge = info["prefix"]["defs"] & info["middle"]["uses"]
    suffix_required = (info["suffix"]["uses"] - info["prefix"]["defs"] - info["suffix"]["defs"] - BUILTIN_NAMES)
    suffix_bridge = info["middle"]["defs"] & suffix_required
    units.update(f"defuse:prefix:{name}" for name in prefix_bridge)
    units.update(f"defuse:suffix:{name}" for name in suffix_bridge)
    return units


def _indent(text: str, first: bool) -> int:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0
    line = lines[0] if first else lines[-1]
    return len(line) - len(line.lstrip(" \t"))


def bridge_features(prefix: str, middle: str, suffix: str) -> dict[str, float]:
    info, full_parse_passed = region_info(prefix, middle, suffix)
    prefix_defs = info["prefix"]["defs"]
    middle_defs = info["middle"]["defs"]
    middle_uses = info["middle"]["uses"]
    suffix_defs = info["suffix"]["defs"]
    suffix_uses = info["suffix"]["uses"]
    prefix_carried = prefix_defs & middle_uses
    suffix_required = suffix_uses - prefix_defs - suffix_defs - BUILTIN_NAMES
    suffix_recovered = middle_defs & suffix_required
    prefix_use_coverage = safe_ratio(len(prefix_carried), len(middle_uses), empty=0.0)
    prefix_available_coverage = safe_ratio(len(prefix_carried), len(prefix_defs), empty=0.0)
    suffix_recovery = safe_ratio(len(suffix_recovered), len(suffix_required), empty=1.0)
    prefix_indent = _indent(prefix, first=False)
    middle_first_indent = _indent(middle, first=True)
    middle_last_indent = _indent(middle, first=False)
    suffix_indent = _indent(suffix, first=True)
    prefix_indent_match = 1.0 / (1.0 + abs(prefix_indent - middle_first_indent))
    suffix_indent_match = 1.0 / (1.0 + abs(middle_last_indent - suffix_indent))
    control_count = len(info["middle"]["controls"])
    combined = mean(
        [
            prefix_use_coverage,
            prefix_available_coverage,
            suffix_recovery,
            prefix_indent_match,
            suffix_indent_match,
            1.0 if full_parse_passed else 0.0,
        ]
    )
    horizon = len(prefix_carried) + len(suffix_recovered) + control_count + (1 if full_parse_passed else 0)
    return {
        "prefix_candidate_def_use_count": float(len(prefix_carried)),
        "prefix_candidate_use_coverage": prefix_use_coverage,
        "prefix_available_use_coverage": prefix_available_coverage,
        "prefix_boundary_indent_match": prefix_indent_match,
        "suffix_required_identifier_count": float(len(suffix_required)),
        "suffix_required_recovered_count": float(len(suffix_recovered)),
        "suffix_required_recovery": suffix_recovery,
        "candidate_suffix_def_use_count": float(len(suffix_recovered)),
        "suffix_boundary_indent_match": suffix_indent_match,
        "full_parse_passed": 1.0 if full_parse_passed else 0.0,
        "candidate_control_structure_count": float(control_count),
        "combined_bridge_satisfaction": combined,
        "semantic_horizon_prediction": float(horizon),
    }


def reference_recovery(prefix: str, reference: str, candidate: str, suffix: str) -> dict[str, float]:
    reference_units = semantic_units(prefix, reference, suffix)
    candidate_units = semantic_units(prefix, candidate, suffix)
    categories = {
        "required_identifier": lambda unit: unit.startswith("def:") or unit.startswith("use:"),
        "def_use": lambda unit: unit.startswith("defuse:"),
        "control_structure": lambda unit: unit.startswith("control:"),
    }
    result: dict[str, float] = {}
    for name, predicate in categories.items():
        ref = {unit for unit in reference_units if predicate(unit)}
        cand = {unit for unit in candidate_units if predicate(unit)}
        matched = ref & cand
        result[f"{name}_precision_offline"] = safe_ratio(len(matched), len(cand), empty=1.0)
        result[f"{name}_coverage_offline"] = safe_ratio(len(matched), len(ref), empty=1.0)
        result[f"{name}_matched_units_offline"] = float(len(matched))
        result[f"{name}_reference_units_offline"] = float(len(ref))
    horizon_ref = {unit for unit in reference_units if unit.startswith("defuse:") or unit.startswith("control:")}
    horizon_matched = horizon_ref & candidate_units
    result["reference_semantic_horizon_offline"] = float(len(horizon_ref))
    result["semantic_horizon_recovery_offline"] = safe_ratio(len(horizon_matched), len(horizon_ref), empty=1.0)
    result["semantic_unit_precision_offline"] = safe_ratio(len(reference_units & candidate_units), len(candidate_units), empty=1.0)
    result["semantic_unit_coverage_offline"] = safe_ratio(len(reference_units & candidate_units), len(reference_units), empty=1.0)
    return result


def rankdata(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for pos in range(index, end):
            ranks[ordered[pos][0]] = rank
        index = end
    return ranks


def auc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    pairs = [(int(bool(label)), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata([score for _, score in pairs])
    positive_rank_sum = sum(rank for rank, (label, _) in zip(ranks, pairs) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def spearman(targets: Sequence[float], predictions: Sequence[float]) -> float:
    pairs = [(float(target), float(pred)) for target, pred in zip(targets, predictions) if math.isfinite(float(target)) and math.isfinite(float(pred))]
    if len(pairs) < 2:
        return float("nan")
    target_ranks = rankdata([target for target, _ in pairs])
    pred_ranks = rankdata([pred for _, pred in pairs])
    if statistics.pstdev(target_ranks) == 0 or statistics.pstdev(pred_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(np.asarray(target_ranks), np.asarray(pred_ranks))[0, 1])


def grouped_bootstrap_metric(
    rows: Sequence[Mapping[str, Any]],
    group_key: str,
    metric: Callable[[list[Mapping[str, Any]]], float],
    *,
    seed: int = 20260711,
    replicates: int = 1000,
) -> dict[str, float | int]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    group_names = sorted(groups)
    estimate = float(metric(list(rows)))
    if not group_names:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_replicates_requested": replicates,
            "bootstrap_replicates_valid": 0,
            "group_count": 0,
        }
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        sampled: list[Mapping[str, Any]] = []
        for group in (rng.choice(group_names) for _ in group_names):
            sampled.extend(groups[group])
        value = float(metric(sampled))
        if math.isfinite(value):
            samples.append(value)
    return {
        "estimate": estimate,
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid": len(samples),
        "group_count": len(group_names),
    }


def grouped_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    group_key: str,
    metric_a: Callable[[list[Mapping[str, Any]]], float],
    metric_b: Callable[[list[Mapping[str, Any]]], float],
    *,
    seed: int = 20260711,
    replicates: int = 1000,
) -> dict[str, float | int]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    group_names = sorted(groups)
    estimate = float(metric_a(list(rows)) - metric_b(list(rows)))
    if not group_names:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_replicates_requested": replicates,
            "bootstrap_replicates_valid": 0,
            "group_count": 0,
        }
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        sampled: list[Mapping[str, Any]] = []
        for group in (rng.choice(group_names) for _ in group_names):
            sampled.extend(groups[group])
        value_a = float(metric_a(sampled))
        value_b = float(metric_b(sampled))
        if math.isfinite(value_a) and math.isfinite(value_b):
            samples.append(value_a - value_b)
    return {
        "estimate": estimate,
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid": len(samples),
        "group_count": len(group_names),
    }


def pairwise_ranking_accuracy(
    rows: Sequence[Mapping[str, Any]],
    score_key: str,
    *,
    group_key: str = "group",
    cross_canvas_only: bool = False,
) -> float:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    correct = 0.0
    pairs = 0
    for group_rows in groups.values():
        positives = [row for row in group_rows if bool(row["passed"])]
        negatives = [row for row in group_rows if not bool(row["passed"])]
        for positive in positives:
            for negative in negatives:
                if cross_canvas_only and "canvas_tokens" in positive and "canvas_tokens" in negative:
                    if int(positive["canvas_tokens"]) == int(negative["canvas_tokens"]):
                        continue
                pscore = float(positive[score_key])
                nscore = float(negative[score_key])
                correct += 1.0 if pscore > nscore else 0.5 if pscore == nscore else 0.0
                pairs += 1
    return correct / pairs if pairs else float("nan")


class LogisticModel:
    def __init__(self, feature_names: Sequence[str]) -> None:
        validate_feature_names(feature_names)
        self.feature_names = list(feature_names)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.constant: float | None = None

    def fit(self, rows: Sequence[Mapping[str, Any]], label_key: str = "passed") -> None:
        x = np.asarray([[float(row.get(name, 0.0) or 0.0) for name in self.feature_names] for row in rows], dtype=np.float64)
        y = np.asarray([1.0 if bool(row[label_key]) else 0.0 for row in rows], dtype=np.float64)
        if not len(y) or y.min() == y.max():
            self.constant = float(y.mean()) if len(y) else 0.0
            return
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        z = (x - self.mean) / self.std
        z = np.concatenate([np.ones((len(z), 1)), z], axis=1)
        weights = np.zeros(z.shape[1], dtype=np.float64)
        positive = max(1.0, float(y.sum()))
        negative = max(1.0, float(len(y) - y.sum()))
        sample_weights = np.where(y > 0, negative / positive, 1.0)
        for _ in range(700):
            logits = np.clip(z @ weights, -40.0, 40.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = z.T @ ((probabilities - y) * sample_weights) / len(y)
            gradient[1:] += 0.01 * weights[1:]
            weights -= 0.07 * gradient
        self.weights = weights

    def predict(self, row: Mapping[str, Any]) -> float:
        if self.constant is not None:
            return self.constant
        assert self.mean is not None and self.std is not None and self.weights is not None
        x = np.asarray([float(row.get(name, 0.0) or 0.0) for name in self.feature_names], dtype=np.float64)
        z = np.concatenate([[1.0], (x - self.mean) / self.std])
        logit = float(np.clip(z @ self.weights, -40.0, 40.0))
        return float(1.0 / (1.0 + math.exp(-logit)))

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
            "weights": self.weights.tolist() if self.weights is not None else [],
            "constant": self.constant,
        }


class LinearModel:
    def __init__(self, feature_names: Sequence[str]) -> None:
        validate_feature_names(feature_names)
        self.feature_names = list(feature_names)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, rows: Sequence[Mapping[str, Any]], target_key: str) -> None:
        x = np.asarray([[float(row.get(name, 0.0) or 0.0) for name in self.feature_names] for row in rows], dtype=np.float64)
        y = np.asarray([float(row[target_key]) for row in rows], dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        z = (x - self.mean) / self.std
        z = np.concatenate([np.ones((len(z), 1)), z], axis=1)
        ridge = np.eye(z.shape[1], dtype=np.float64) * 0.05
        ridge[0, 0] = 0.0
        self.weights = np.linalg.pinv(z.T @ z + ridge) @ z.T @ y

    def predict(self, row: Mapping[str, Any]) -> float:
        assert self.mean is not None and self.std is not None and self.weights is not None
        x = np.asarray([float(row.get(name, 0.0) or 0.0) for name in self.feature_names], dtype=np.float64)
        z = np.concatenate([[1.0], (x - self.mean) / self.std])
        return float(np.clip(z @ self.weights, 0.0, 1.0))


def fold_for_group(group: str, folds: int = 5) -> int:
    return int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % folds


def out_of_fold_predictions(
    rows: list[dict[str, Any]],
    feature_names: Sequence[str],
    *,
    target_key: str,
    kind: str,
    folds: int = 5,
) -> tuple[list[float], dict[str, Any]]:
    predictions = [0.0] * len(rows)
    fold_models: dict[str, Any] = {}
    for fold in range(folds):
        train = [row for row in rows if fold_for_group(str(row["group_key"]), folds) != fold]
        held_indices = [index for index, row in enumerate(rows) if fold_for_group(str(row["group_key"]), folds) == fold]
        model: LogisticModel | LinearModel
        if kind == "logistic":
            model = LogisticModel(feature_names)
            model.fit(train, label_key=target_key)
        elif kind == "linear":
            model = LinearModel(feature_names)
            model.fit(train, target_key=target_key)
        else:
            raise ValueError(kind)
        fold_models[str(fold)] = model.to_json() if isinstance(model, LogisticModel) else {"feature_names": list(feature_names)}
        for index in held_indices:
            predictions[index] = model.predict(rows[index])
    return predictions, fold_models


def corrected_deployable_gate(
    *,
    global_auc: float,
    within_task_accuracy: float,
    cross_canvas_accuracy: float,
    comparisons: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = ["prefix_only", "suffix_only", "token_canvas", "ordinary_confidence"]
    conditions = {
        "within_task_accuracy_above_chance": math.isfinite(within_task_accuracy) and within_task_accuracy > 0.5,
        "cross_canvas_accuracy_above_chance": math.isfinite(cross_canvas_accuracy) and cross_canvas_accuracy > 0.5,
        "positive_primary_delta_vs_all_baselines": all(
            math.isfinite(float(comparisons.get(name, {}).get("delta_primary", float("nan"))))
            and float(comparisons[name]["delta_primary"]) > 0.0
            and math.isfinite(float(comparisons[name].get("delta_ci_low", float("nan"))))
            and float(comparisons[name]["delta_ci_low"]) > 0.0
            for name in required
        ),
        "positive_selection_net_vs_fixed64": int(selection.get("vs_fixed64", {}).get("net", 0)) > 0,
        "positive_selection_net_vs_confidence": int(selection.get("vs_confidence", {}).get("net", 0)) > 0,
        "no_short_net_regression_vs_fixed64": int(selection.get("vs_fixed64", {}).get("short_net", -1)) >= 0,
        "no_short_net_regression_vs_confidence": int(selection.get("vs_confidence", {}).get("short_net", -1)) >= 0,
    }
    return {
        "passed": all(conditions.values()),
        "mechanism_name": "AST/def-use bridge proxy V0",
        "primary_metric": "cross_canvas_within_task_pairwise_accuracy",
        "global_auc": global_auc,
        "global_auc_role": "secondary_diagnostic_only",
        "within_task_pairwise_accuracy": within_task_accuracy,
        "cross_canvas_within_task_pairwise_accuracy": cross_canvas_accuracy,
        "conditions": conditions,
        "failed_conditions": [name for name, passed in conditions.items() if not passed],
        "comparisons": dict(comparisons),
        "selection": dict(selection),
    }


def bank_integrity(rows: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = [row for row in rows if row.get("candidate_kind") in {"deployable_grid", "oracle_sufficient_diagnostic_ceiling"}]
    deployable = [row for row in rows if row.get("candidate_kind") == "deployable_grid"]
    counts = Counter(str(row.get("candidate_key", "")) for row in rows)
    duplicate_keys = sorted(key for key, count in counts.items() if count > 1)
    expected_base = len(manifest) * 9
    return {
        "passed": len(manifest) == 148 and len(base) == expected_base and len(deployable) == 148 * 8 and not duplicate_keys,
        "manifest_rows": len(manifest),
        "base_candidate_rows": len(base),
        "expected_base_candidate_rows": expected_base,
        "deployable_rows": len(deployable),
        "expected_deployable_rows": 148 * 8,
        "alpha_rows": sum(row.get("candidate_kind") == "alpha_renamed_auxiliary" for row in rows),
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_keys": duplicate_keys[:100],
        "error_row_count": sum(row.get("status") != "ok" for row in rows),
    }


def feature_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    prefix = str(raw["prefix_text"])
    middle = str(raw["middle_text"])
    suffix = str(raw["suffix_text"])
    features = bridge_features(prefix, middle, suffix)
    canvas = int(raw["canvas_tokens"])
    candidate_tokens = int(raw.get("candidate_middle_tokens", 0) or 0)
    features.update(
        {
            "canvas_log2": math.log2(max(1, canvas)),
            "seed_value": float(raw["seed"]),
            "candidate_middle_tokens": float(candidate_tokens),
            "canvas_tokens": float(canvas),
            "candidate_canvas_fill_ratio": safe_ratio(candidate_tokens, canvas, empty=0.0),
            "prefix_tokens": float(len(_token_names(prefix))),
            "suffix_tokens": float(len(_token_names(suffix))),
            "ordinary_confidence": float((raw.get("metrics") or {}).get("mean_final_confidence") or 0.0),
        }
    )
    features.update(reference_recovery(prefix, str(raw["reference_middle"]), middle, suffix))
    row = {
        "candidate_key": raw["candidate_key"],
        "row_key": raw["row_key"],
        "group_key": hash_group(str(raw["task_group"])),
        "canvas_tokens": canvas,
        "seed": int(raw["seed"]),
        "passed": bool(raw["passed"]),
        "length_bucket_offline_only": raw["length_bucket"],
        "candidate_middle_sha256": raw.get("candidate_middle_sha256", ""),
        "candidate_full_ast_sha256": raw.get("candidate_full_ast_sha256", ""),
        **features,
    }
    row.update(deployable_proxy_scores(row))
    return row


def select_candidate(rows: Sequence[Mapping[str, Any]], score_key: str) -> Mapping[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row[score_key]),
            -int(row["canvas_tokens"]),
            -int(row["seed"]),
        ),
    )


def deterministic_selection_summary(feature_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        groups[str(row["row_key"])].append(row)
    selections: list[dict[str, Any]] = []
    score_keys = {
        "combined": "deployable_proxy_combined",
        "confidence": "deployable_proxy_ordinary_confidence",
        "prefix_only": "deployable_proxy_prefix_only",
        "suffix_only": "deployable_proxy_suffix_only",
        "token_canvas": "deployable_proxy_token_canvas",
    }
    for row_key, rows in sorted(groups.items()):
        fixed = next(row for row in rows if int(row["canvas_tokens"]) == 64 and int(row["seed"]) == 0)
        chosen = {name: select_candidate(rows, key) for name, key in score_keys.items()}
        result = {
            "row_key": row_key,
            "group_key": rows[0]["group_key"],
            "length_bucket_offline_only": rows[0]["length_bucket_offline_only"],
            "fixed64_passed": bool(fixed["passed"]),
        }
        for name, row in chosen.items():
            result[f"{name}_candidate_key"] = row["candidate_key"]
            result[f"{name}_passed"] = bool(row["passed"])
        selections.append(result)

    def paired(method: str, baseline: str) -> dict[str, int]:
        wins = sum(bool(row[f"{method}_passed"]) and not bool(row[f"{baseline}_passed"]) for row in selections)
        losses = sum(not bool(row[f"{method}_passed"]) and bool(row[f"{baseline}_passed"]) for row in selections)
        short = [row for row in selections if row["length_bucket_offline_only"] == "short"]
        short_wins = sum(bool(row[f"{method}_passed"]) and not bool(row[f"{baseline}_passed"]) for row in short)
        short_losses = sum(not bool(row[f"{method}_passed"]) and bool(row[f"{baseline}_passed"]) for row in short)
        return {
            "wins": wins,
            "losses": losses,
            "net": wins - losses,
            "short_wins": short_wins,
            "short_losses": short_losses,
            "short_net": short_wins - short_losses,
        }

    return selections, {
        "vs_fixed64": paired("combined", "fixed64"),
        "vs_confidence": paired("combined", "confidence"),
    }


def f1_analysis(deployable_raw: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in deployable_raw:
        groups[str(row["row_key"])].append(row)
    task_rows: list[dict[str, Any]] = []
    for row_key, candidates in sorted(groups.items()):
        reference_units = semantic_units(
            str(candidates[0]["prefix_text"]),
            str(candidates[0]["reference_middle"]),
            str(candidates[0]["suffix_text"]),
        )
        candidate_units = [semantic_units(str(row["prefix_text"]), str(row["middle_text"]), str(row["suffix_text"])) for row in candidates]
        counts = Counter(unit for units in candidate_units for unit in units)
        consensus = {unit for unit, count in counts.items() if count >= math.ceil(len(candidates) / 2)}
        all_fail = not any(bool(row["passed"]) for row in candidates)
        union_correct = set().union(*(units & reference_units for units in candidate_units))
        best_correct = max((len(units & reference_units) for units in candidate_units), default=0)
        task_rows.append(
            {
                "row_key": row_key,
                "group_key": hash_group(str(candidates[0]["task_group"])),
                "length_bucket_offline_only": candidates[0]["length_bucket"],
                "candidate_count": len(candidates),
                "unique_candidate_hashes": len({str(row["candidate_middle_sha256"]) for row in candidates}),
                "parsable_candidate_count": sum(bool(row.get("candidate_full_ast_sha256")) for row in candidates),
                "unique_parsable_ast_hashes": len({str(row["candidate_full_ast_sha256"]) for row in candidates if row.get("candidate_full_ast_sha256")}),
                "mean_semantic_unit_precision_offline": mean(
                    safe_ratio(len(units & reference_units), len(units), empty=1.0) for units in candidate_units
                ),
                "mean_semantic_unit_coverage_offline": mean(
                    safe_ratio(len(units & reference_units), len(reference_units), empty=1.0) for units in candidate_units
                ),
                "all_deployable_candidates_fail": all_fail,
                "all_fail_union_correct_semantic_units": len(union_correct) if all_fail else "",
                "all_fail_best_single_correct_semantic_units": best_correct if all_fail else "",
                "all_fail_complementary_correct_units": len(union_correct) - best_correct if all_fail else "",
                "consensus_landmark_count": len(consensus),
                "consensus_landmark_precision_offline": safe_ratio(len(consensus & reference_units), len(consensus), empty=1.0),
            }
        )
    all_fail_rows = [row for row in task_rows if row["all_deployable_candidates_fail"]]
    summary = {
        "verdict": "f1_completed_diagnostic_only",
        "task_count": len(task_rows),
        "candidate_count": len(deployable_raw),
        "mean_unique_candidate_hashes": mean(float(row["unique_candidate_hashes"]) for row in task_rows),
        "mean_unique_parsable_ast_hashes": mean(float(row["unique_parsable_ast_hashes"]) for row in task_rows),
        "mean_semantic_unit_precision_offline": mean(float(row["mean_semantic_unit_precision_offline"]) for row in task_rows),
        "mean_semantic_unit_coverage_offline": mean(float(row["mean_semantic_unit_coverage_offline"]) for row in task_rows),
        "all_fail_task_count": len(all_fail_rows),
        "all_fail_tasks_with_complementary_correct_units": sum(int(row["all_fail_complementary_correct_units"]) > 0 for row in all_fail_rows),
        "mean_all_fail_complementary_correct_units": mean(float(row["all_fail_complementary_correct_units"]) for row in all_fail_rows),
        "mean_consensus_landmark_precision_offline": mean(float(row["consensus_landmark_precision_offline"]) for row in task_rows),
        "reference_use": "offline_diagnostic_only",
    }
    return task_rows, summary


def f2_analysis(
    deployable_raw: Sequence[Mapping[str, Any]],
    alpha_raw: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alpha_by_key = {
        (str(row["row_key"]), int(row["canvas_tokens"]), int(row["seed"])): row
        for row in alpha_raw
        if row.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    for original in deployable_raw:
        alpha = alpha_by_key.get((str(original["row_key"]), int(original["canvas_tokens"]), int(original["seed"])))
        if alpha is None:
            continue
        original_units = semantic_units(str(original["prefix_text"]), str(original["middle_text"]), str(original["suffix_text"]))
        inverse_units = semantic_units(
            str(original["prefix_text"]),
            str(alpha.get("inverse_renamed_middle_text", "")),
            str(original["suffix_text"]),
        )
        union = original_units | inverse_units
        rows.append(
            {
                "candidate_key": original["candidate_key"],
                "row_key": original["row_key"],
                "group_key": hash_group(str(original["task_group"])),
                "canvas_tokens": int(original["canvas_tokens"]),
                "seed": int(original["seed"]),
                "passed": bool(original["passed"]),
                "canvas_log2": math.log2(max(1, int(original["canvas_tokens"]))),
                "seed_value": float(original["seed"]),
                "ordinary_confidence": float((original.get("metrics") or {}).get("mean_final_confidence") or 0.0),
                "equiv_exact_middle_hash": 1.0 if original.get("candidate_middle_sha256") == alpha.get("inverse_renamed_middle_sha256") else 0.0,
                "equiv_ast_match": 1.0 if original.get("candidate_full_ast_sha256") and original.get("candidate_full_ast_sha256") == alpha.get("inverse_renamed_full_ast_sha256") else 0.0,
                "equiv_semantic_jaccard": safe_ratio(len(original_units & inverse_units), len(union), empty=1.0),
                "equiv_confidence_abs_delta": abs(
                    float((original.get("metrics") or {}).get("mean_final_confidence") or 0.0)
                    - float((alpha.get("metrics") or {}).get("mean_final_confidence") or 0.0)
                ),
            }
        )
    if not rows:
        return [], {
            "verdict": "f2_no_reference_verified_alpha_pairs",
            "paired_candidate_count": 0,
            "paired_task_count": 0,
            "transformation": "strict_local_alpha_renaming_reference_verified",
            "interpretation_guard": "equivariance_or_stability_is_not_correctness",
            "baseline_controls": ["canvas", "seed", "ordinary_confidence"],
        }
    baseline_features = COMMON_FEATURES + CONFIDENCE_FEATURES
    enhanced_features = baseline_features + [
        "equiv_exact_middle_hash",
        "equiv_ast_match",
        "equiv_semantic_jaccard",
        "equiv_confidence_abs_delta",
    ]
    validate_feature_names(baseline_features)
    validate_feature_names(enhanced_features)
    baseline_scores, _ = out_of_fold_predictions(rows, baseline_features, target_key="passed", kind="logistic")
    enhanced_scores, _ = out_of_fold_predictions(rows, enhanced_features, target_key="passed", kind="logistic")
    for row, baseline, enhanced in zip(rows, baseline_scores, enhanced_scores):
        row["baseline_control_score"] = baseline
        row["equivariance_augmented_score"] = enhanced
    baseline_metric = grouped_bootstrap_metric(
        rows,
        "group_key",
        lambda sample: auc([bool(row["passed"]) for row in sample], [float(row["baseline_control_score"]) for row in sample]),
        replicates=bootstrap_replicates,
    )
    enhanced_metric = grouped_bootstrap_metric(
        rows,
        "group_key",
        lambda sample: auc([bool(row["passed"]) for row in sample], [float(row["equivariance_augmented_score"]) for row in sample]),
        replicates=bootstrap_replicates,
    )
    delta = grouped_bootstrap_delta(
        rows,
        "group_key",
        lambda sample: auc([bool(row["passed"]) for row in sample], [float(row["equivariance_augmented_score"]) for row in sample]),
        lambda sample: auc([bool(row["passed"]) for row in sample], [float(row["baseline_control_score"]) for row in sample]),
        replicates=bootstrap_replicates,
    )
    summary = {
        "verdict": "f2_equivariance_predictive_increment" if float(delta["ci_low"]) > 0 else "f2_equivariance_not_independently_predictive",
        "paired_candidate_count": len(rows),
        "paired_task_count": len({row["group_key"] for row in rows}),
        "transformation": "strict_local_alpha_renaming_reference_verified",
        "interpretation_guard": "equivariance_or_stability_is_not_correctness",
        "baseline_controls": ["canvas", "seed", "ordinary_confidence"],
        "baseline_auc": baseline_metric,
        "equivariance_augmented_auc": enhanced_metric,
        "delta_auc": delta,
        "exact_middle_hash_rate": mean(float(row["equiv_exact_middle_hash"]) for row in rows),
        "ast_match_rate": mean(float(row["equiv_ast_match"]) for row in rows),
        "mean_semantic_jaccard": mean(float(row["equiv_semantic_jaccard"]) for row in rows),
    }
    return rows, summary


def f3_analysis(feature_rows: list[dict[str, Any]], bootstrap_replicates: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model_registry: dict[str, Any] = {}
    supervised_metrics: list[dict[str, Any]] = []
    for family, features in SUPERVISED_PROBE_FEATURE_FAMILIES.items():
        validate_feature_names(features)
        pass_scores, fold_models = out_of_fold_predictions(feature_rows, features, target_key="passed", kind="logistic")
        horizon_scores, _ = out_of_fold_predictions(
            feature_rows,
            features,
            target_key="semantic_horizon_recovery_offline",
            kind="linear",
        )
        for row, pass_score, horizon_score in zip(feature_rows, pass_scores, horizon_scores):
            row[f"supervised_probe_score_{family}"] = pass_score
            row[f"supervised_horizon_prediction_{family}"] = horizon_score
        auc_ci = grouped_bootstrap_metric(
            feature_rows,
            "group_key",
            lambda sample, key=f"supervised_probe_score_{family}": auc([bool(row["passed"]) for row in sample], [float(row[key]) for row in sample]),
            replicates=bootstrap_replicates,
        )
        horizon_mae_ci = grouped_bootstrap_metric(
            feature_rows,
            "group_key",
            lambda sample, key=f"supervised_horizon_prediction_{family}": mean(
                abs(float(row[key]) - float(row["semantic_horizon_recovery_offline"])) for row in sample
            ),
            replicates=bootstrap_replicates,
        )
        horizon_spearman = spearman(
            [float(row["semantic_horizon_recovery_offline"]) for row in feature_rows],
            [float(row[f"supervised_horizon_prediction_{family}"]) for row in feature_rows],
        )
        supervised_metrics.append(
            {
                "feature_family": family,
                "candidate_ranking_auc": auc_ci["estimate"],
                "candidate_ranking_auc_ci_low": auc_ci["ci_low"],
                "candidate_ranking_auc_ci_high": auc_ci["ci_high"],
                "semantic_horizon_mae": horizon_mae_ci["estimate"],
                "semantic_horizon_mae_ci_low": horizon_mae_ci["ci_low"],
                "semantic_horizon_mae_ci_high": horizon_mae_ci["ci_high"],
                "semantic_horizon_spearman": horizon_spearman,
                "feature_count": len(features),
            }
        )
        full_model = LogisticModel(features)
        full_model.fit(feature_rows)
        model_registry[family] = {
            "track": "supervised_probe_diagnostic",
            "outcome_labels_used_for_fit": True,
            "deployable_authorization_role": "none",
            "features": features,
            "fold_models": fold_models,
            "full_model": full_model.to_json(),
        }

    deployable_metrics: list[dict[str, Any]] = []
    proxy_names = {
        "prefix_only": "deployable_proxy_prefix_only",
        "suffix_only": "deployable_proxy_suffix_only",
        "token_canvas": "deployable_proxy_token_canvas",
        "ordinary_confidence": "deployable_proxy_ordinary_confidence",
        "combined": "deployable_proxy_combined",
    }
    for family, score_key in proxy_names.items():
        global_auc = grouped_bootstrap_metric(
            feature_rows,
            "group_key",
            lambda sample, key=score_key: auc([bool(row["passed"]) for row in sample], [float(row[key]) for row in sample]),
            replicates=bootstrap_replicates,
        )
        within = grouped_bootstrap_metric(
            feature_rows,
            "group_key",
            lambda sample, key=score_key: pairwise_ranking_accuracy(sample, key, group_key="group_key"),
            replicates=bootstrap_replicates,
        )
        cross = grouped_bootstrap_metric(
            feature_rows,
            "group_key",
            lambda sample, key=score_key: pairwise_ranking_accuracy(sample, key, group_key="group_key", cross_canvas_only=True),
            replicates=bootstrap_replicates,
        )
        deployable_metrics.append(
            {
                "score_family": family,
                "score_key": score_key,
                "global_auc_secondary": global_auc["estimate"],
                "global_auc_ci_low": global_auc["ci_low"],
                "global_auc_ci_high": global_auc["ci_high"],
                "within_task_pairwise_accuracy": within["estimate"],
                "within_task_ci_low": within["ci_low"],
                "within_task_ci_high": within["ci_high"],
                "cross_canvas_pairwise_accuracy_primary": cross["estimate"],
                "cross_canvas_ci_low": cross["ci_low"],
                "cross_canvas_ci_high": cross["ci_high"],
            }
        )

    comparisons: dict[str, dict[str, Any]] = {}
    for baseline in ["prefix_only", "suffix_only", "token_canvas", "ordinary_confidence"]:
        delta = grouped_bootstrap_delta(
            feature_rows,
            "group_key",
            lambda sample: pairwise_ranking_accuracy(sample, "deployable_proxy_combined", group_key="group_key", cross_canvas_only=True),
            lambda sample, key=proxy_names[baseline]: pairwise_ranking_accuracy(sample, key, group_key="group_key", cross_canvas_only=True),
            replicates=bootstrap_replicates,
        )
        comparisons[baseline] = {
            "delta_primary": delta["estimate"],
            "delta_ci_low": delta["ci_low"],
            "delta_ci_high": delta["ci_high"],
            "bootstrap_replicates_valid": delta["bootstrap_replicates_valid"],
        }
    selections, selection_summary = deterministic_selection_summary(feature_rows)
    deployable_by_family = {row["score_family"]: row for row in deployable_metrics}
    combined_metrics = deployable_by_family["combined"]
    gate = corrected_deployable_gate(
        global_auc=float(combined_metrics["global_auc_secondary"]),
        within_task_accuracy=float(combined_metrics["within_task_pairwise_accuracy"]),
        cross_canvas_accuracy=float(combined_metrics["cross_canvas_pairwise_accuracy_primary"]),
        comparisons=comparisons,
        selection=selection_summary,
    )
    recovery = {
        key: grouped_bootstrap_metric(
            feature_rows,
            "group_key",
            lambda sample, metric_key=key: mean(float(row[metric_key]) for row in sample),
            replicates=bootstrap_replicates,
        )
        for key in [
            "required_identifier_precision_offline",
            "required_identifier_coverage_offline",
            "def_use_precision_offline",
            "def_use_coverage_offline",
            "control_structure_precision_offline",
            "control_structure_coverage_offline",
        ]
    }
    summary = {
        "verdict": "f3_corrected_deployable_gate_passed" if gate["passed"] else "f3_corrected_deployable_gate_failed",
        "mechanism_name": "AST/def-use bridge proxy V0",
        "candidate_count": len(feature_rows),
        "task_count": len({row["group_key"] for row in feature_rows}),
        "grouped_bootstrap_replicates": bootstrap_replicates,
        "supervised_probe_diagnostic": {
            "outcome_labels_used_for_fit": True,
            "deployable_authorization_role": "none",
            "metrics": supervised_metrics,
        },
        "deployable_bridge_proxy": {
            "formula_provenance": "fixed_before_outcomes",
            "outcome_labels_used_for_fit": False,
            "reference_used_for_fit": False,
            "outcome_labels_used_for_selection": False,
            "reference_used_for_selection": False,
            "metrics": deployable_metrics,
            "comparisons": comparisons,
            "selection": selection_summary,
        },
        "corrected_deployable_gate": gate,
        "semantic_recovery_offline": recovery,
        "forbidden_feature_audit": forbidden_feature_audit(),
    }
    return deployable_metrics, summary, model_registry, selections


def f4_analysis(feature_rows: list[dict[str, Any]], bootstrap_replicates: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for family, score_key in {
        "prefix_only": "deployable_proxy_prefix_only",
        "suffix_only": "deployable_proxy_suffix_only",
        "token_canvas": "deployable_proxy_token_canvas",
        "ordinary_confidence": "deployable_proxy_ordinary_confidence",
        "combined": "deployable_proxy_combined",
    }.items():
        all_pairs = grouped_bootstrap_metric(
            feature_rows,
            "group_key",
            lambda sample, key=score_key: pairwise_ranking_accuracy(sample, key, group_key="group_key"),
            replicates=bootstrap_replicates,
        )
        cross_canvas = grouped_bootstrap_metric(
            feature_rows,
            "group_key",
            lambda sample, key=score_key: pairwise_ranking_accuracy(
                sample,
                key,
                group_key="group_key",
                cross_canvas_only=True,
            ),
            replicates=bootstrap_replicates,
        )
        metrics.append(
            {
                "feature_family": family,
                "within_task_pairwise_accuracy": all_pairs["estimate"],
                "within_task_pairwise_ci_low": all_pairs["ci_low"],
                "within_task_pairwise_ci_high": all_pairs["ci_high"],
                "cross_canvas_pairwise_accuracy": cross_canvas["estimate"],
                "cross_canvas_pairwise_ci_low": cross_canvas["ci_low"],
                "cross_canvas_pairwise_ci_high": cross_canvas["ci_high"],
            }
        )
    best = max(metrics, key=lambda row: float(row["cross_canvas_pairwise_accuracy"]) if math.isfinite(float(row["cross_canvas_pairwise_accuracy"])) else -1.0)
    summary = {
        "verdict": "f4_deterministic_inference_visible_ranking_measured",
        "candidate_count": len(feature_rows),
        "task_count": len({row["group_key"] for row in feature_rows}),
        "best_cross_canvas_family": best["feature_family"],
        "best_cross_canvas_pairwise_accuracy": best["cross_canvas_pairwise_accuracy"],
        "metrics": metrics,
    }
    return metrics, summary


def render_report(f1: Mapping[str, Any], f2: Mapping[str, Any], f3: Mapping[str, Any], f4: Mapping[str, Any]) -> str:
    gate = f3["corrected_deployable_gate"]
    lines = [
        "# Phase 5 Premise Falsification",
        "",
        "All results use the shared full RandomSpanLight candidate bank. Reference code and functional outcomes are offline labels only; task IDs, split labels, oracle length, reference code, verifier outcomes, and error types are absent from deployable feature families.",
        "",
        "## F1 Candidate/Fragment Diagnostic",
        "",
        f"Tasks/candidates: `{f1['task_count']}` / `{f1['candidate_count']}`.",
        f"Mean unique candidate hashes: `{f1['mean_unique_candidate_hashes']:.4f}`.",
        f"Mean unique parsable AST hashes: `{f1['mean_unique_parsable_ast_hashes']:.4f}`.",
        f"All-fail tasks with complementary reference-matching semantic units: `{f1['all_fail_tasks_with_complementary_correct_units']}/{f1['all_fail_task_count']}`.",
        f"Mean consensus-landmark precision: `{f1['mean_consensus_landmark_precision_offline']:.4f}`.",
        "",
        "## F2 Alpha-Renaming Equivariance",
        "",
        f"Verdict: `{f2['verdict']}`.",
        f"Paired candidates/tasks: `{f2['paired_candidate_count']}` / `{f2['paired_task_count']}`.",
        f"Controlled baseline AUROC: `{f2['baseline_auc']['estimate']:.4f}`.",
        f"Equivariance-augmented AUROC: `{f2['equivariance_augmented_auc']['estimate']:.4f}`.",
        f"Delta 95% grouped CI: `[{f2['delta_auc']['ci_low']:.4f}, {f2['delta_auc']['ci_high']:.4f}]`.",
        "Equivariance/stability is not correctness.",
        "",
        "## F3A Supervised Probe Diagnostic",
        "",
        "These grouped-OOF logistic probes use functional pass/fail labels. They answer only whether feature families contain information and cannot authorize or supply scores to V0.",
        "",
        "| Family | Pass AUROC | 95% CI | Horizon MAE | Horizon Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in f3["supervised_probe_diagnostic"]["metrics"]:
        lines.append(
            f"| `{row['feature_family']}` | `{row['candidate_ranking_auc']:.4f}` | "
            f"`[{row['candidate_ranking_auc_ci_low']:.4f}, {row['candidate_ranking_auc_ci_high']:.4f}]` | "
            f"`{row['semantic_horizon_mae']:.4f}` | `{row['semantic_horizon_spearman']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## F3B/F4 Deterministic AST/Def-Use Bridge Proxy",
            "",
            f"Verdict: `{f3['verdict']}`.",
            f"Corrected gate passed: `{gate['passed']}`.",
            f"Failed conditions: `{gate['failed_conditions']}`.",
            "Global AUROC is secondary diagnostic only.",
            "",
            "| Family | Cross-canvas pairwise accuracy | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in f4["metrics"]:
        lines.append(
            f"| `{row['feature_family']}` | `{row['cross_canvas_pairwise_accuracy']:.4f}` | "
            f"`[{row['cross_canvas_pairwise_ci_low']:.4f}, {row['cross_canvas_pairwise_ci_high']:.4f}]` |"
        )
    lines.extend(
        [
            "",
            "## Method Gate",
            "",
            "AST/def-use bridge proxy V0 is authorized only by the corrected within-task/cross-canvas, deterministic-baseline, paired-selection, and short-safety gate.",
            f"Current authorization: `{'authorized' if gate['passed'] else 'killed_for_this_round'}`.",
            "This proxy is not full Semantic Bridge Projection or Abductive Program-State Bridge; genuine program-state analysis, backward obligations, bridge anchors, and denoising intervention remain future stages.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    bank_dir = Path(args.bank_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = bank_dir / "candidate_bank_raw.jsonl"
    compact_bank_dir = Path(args.compact_bank_dir).resolve() if args.compact_bank_dir else None
    manifest_path = compact_bank_dir / "manifest.csv" if compact_bank_dir else bank_dir / "manifest.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    raw_rows = read_jsonl(raw_path)
    manifest = read_csv(manifest_path)
    integrity = bank_integrity(raw_rows, manifest)
    if not integrity["passed"]:
        write_json(output_dir / "bank_integrity.json", integrity)
        raise RuntimeError(f"Candidate bank integrity failed: {integrity}")
    deployable_raw = [row for row in raw_rows if row.get("candidate_kind") == "deployable_grid" and row.get("status") == "ok"]
    alpha_raw = [row for row in raw_rows if row.get("candidate_kind") == "alpha_renamed_auxiliary" and row.get("status") == "ok"]

    f1_rows, f1_summary = f1_analysis(deployable_raw)
    f2_rows, f2_summary = f2_analysis(deployable_raw, alpha_raw, args.bootstrap_replicates)
    features = [feature_row(row) for row in deployable_raw]
    f3_metrics, f3_summary, model_registry, selection_preview = f3_analysis(features, args.bootstrap_replicates)
    f4_metrics, f4_summary = f4_analysis(features, args.bootstrap_replicates)

    compact_feature_fields = [
        "candidate_key",
        "row_key",
        "group_key",
        "canvas_tokens",
        "seed",
        "passed",
        "length_bucket_offline_only",
        *sorted({name for names in SUPERVISED_PROBE_FEATURE_FAMILIES.values() for name in names}),
        "semantic_horizon_recovery_offline",
        *[f"supervised_probe_score_{family}" for family in SUPERVISED_PROBE_FEATURE_FAMILIES],
        *[f"supervised_horizon_prediction_{family}" for family in SUPERVISED_PROBE_FEATURE_FAMILIES],
        *DEPLOYABLE_PROXY_SCORE_KEYS,
    ]
    write_csv(output_dir / "f1_task_diagnostics.csv", f1_rows)
    write_json(output_dir / "f1_summary.json", f1_summary)
    write_csv(
        output_dir / "f2_equivariance_predictions.csv",
        [
            {key: value for key, value in row.items() if key not in {"row_key"}}
            for row in f2_rows
        ],
    )
    write_json(output_dir / "f2_summary.json", f2_summary)
    write_csv(output_dir / "f3_supervised_probe_oof_predictions.csv", features, compact_feature_fields)
    write_csv(
        output_dir / "f3_deployable_proxy_scores.csv",
        [
            {
                "candidate_key": row["candidate_key"],
                "row_key": row["row_key"],
                "canvas_tokens": row["canvas_tokens"],
                "seed": row["seed"],
                **{key: row[key] for key in DEPLOYABLE_PROXY_SCORE_KEYS},
            }
            for row in features
        ],
        ["candidate_key", "row_key", "canvas_tokens", "seed", *DEPLOYABLE_PROXY_SCORE_KEYS],
    )
    write_csv(output_dir / "f3_deployable_proxy_metrics.csv", f3_metrics)
    write_csv(output_dir / "f3_deployable_selection_preview.csv", selection_preview)
    write_json(output_dir / "f3_summary.json", f3_summary)
    write_json(output_dir / "f3_supervised_probe_model_registry.json", model_registry)
    formula = deployable_bridge_formula_spec()
    write_json(output_dir / "deployable_bridge_formula.json", formula)
    (output_dir / "deployable_bridge_formula.md").write_text(
        "# AST/Def-Use Bridge Proxy V0 Fixed Formula\n\n"
        f"Provenance: `{formula['provenance']}`. Parameters: `{formula['parameter_source']}`.\n\n"
        + "\n".join(f"- `{name}`: `{value}`" for name, value in formula["scores"].items())
        + f"\n\nScope boundary: {formula['scope_boundary']}\n",
        encoding="utf-8",
    )
    write_json(output_dir / "f3_gate.json", f3_summary["corrected_deployable_gate"])
    write_csv(output_dir / "f4_pairwise_ranking.csv", f4_metrics)
    write_json(output_dir / "f4_summary.json", f4_summary)
    write_json(output_dir / "bank_integrity.json", integrity)
    write_json(output_dir / "forbidden_feature_audit.json", f3_summary["forbidden_feature_audit"])
    (output_dir / "report.md").write_text(render_report(f1_summary, f2_summary, f3_summary, f4_summary), encoding="utf-8")
    summary = {
        "verdict": "phase5_premise_falsification_completed",
        "bank_integrity": integrity,
        "f1": f1_summary,
        "f2": f2_summary,
        "f3": f3_summary,
        "f4": f4_summary,
        "ast_def_use_bridge_proxy_v0_authorized": bool(f3_summary["corrected_deployable_gate"]["passed"]),
        "frozen_test_status": "sealed",
        "test_evaluation_count": 0,
    }
    write_json(output_dir / "summary.json", summary)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Phase 5 premise falsification on the shared candidate bank")
    root.add_argument("--bank-dir", required=True)
    root.add_argument("--compact-bank-dir")
    root.add_argument("--output-dir", required=True)
    root.add_argument("--bootstrap-replicates", type=int, default=1000)
    return root


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
