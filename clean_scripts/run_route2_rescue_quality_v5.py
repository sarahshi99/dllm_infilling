#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analysis.trace_feature_audit_v2 import compute_shape_features
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask, load_humaneval_infilling
from expvision_dllm_clean.evaluation import summarize_results
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import load_model_and_tokenizer, set_global_seed

import run_route2_trace_rescue as route2_runner


JsonDict = Dict[str, Any]


CONSENSUS_CONFIDENCE_ALLOWED_FIELDS = [
    "middle_text",
    "text_consensus",
    "trace_features.confidence_last",
    "trace_features.confidence_max",
    "trace_features.top1_last",
    "trace_features.top1_median",
    "trace_features.gap_last",
    "trace_features.gap_median",
    "trace_features.final_remaining_ratio_by_selected",
    "selected_mask_length",
]

SYNTAX_AWARE_ALLOWED_FIELDS = CONSENSUS_CONFIDENCE_ALLOWED_FIELDS + [
    "parse_passed",
    "compile_passed",
]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    min_length: int
    steps: int


CHEAP_CANDIDATE_SPECS = [
    CandidateSpec("len24_s64", min_length=24, steps=64),
    CandidateSpec("len32_s64", min_length=32, steps=64),
    CandidateSpec("len32_s96", min_length=32, steps=96),
]

FULL_CANDIDATE_SPECS = CHEAP_CANDIDATE_SPECS + [
    CandidateSpec("len40_s64", min_length=40, steps=64),
    CandidateSpec("len40_s96", min_length=40, steps=96),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Route2 rescue-quality V5 multi-candidate rescue.")
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=64, help="Primary midcons decode steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lcas-policy", type=str, default="lcas_v3b", choices=["lcas_v3a", "lcas_v3b"])
    parser.add_argument("--route2-policy", type=str, default="precision_top1_conf", choices=sorted(route2_runner.POLICIES))
    parser.add_argument("--candidate-set", type=str, default="cheap", choices=["cheap", "full"])
    parser.add_argument("--selector", type=str, default="consensus_confidence", choices=["consensus_confidence", "syntax_aware"])
    parser.add_argument("--baseline-results", type=str, default=None)
    parser.add_argument("--route2-reference-results", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs_clean")
    parser.add_argument("--experiment-name", type=str, default="route2_rescue_quality_v5")
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--save-full-text-per-step", action="store_true")
    return parser.parse_args()


def candidate_specs(name: str) -> List[CandidateSpec]:
    if name == "cheap":
        return list(CHEAP_CANDIDATE_SPECS)
    if name == "full":
        return list(FULL_CANDIDATE_SPECS)
    raise ValueError(f"unknown candidate set: {name}")


def make_cfg(args: argparse.Namespace) -> ExperimentConfig:
    cfg = route2_runner.make_cfg(args)
    cfg.decode.save_step_traces = True
    cfg.decode.save_full_text_per_step = bool(args.save_full_text_per_step)
    return cfg


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _metric(result: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return (result.get("metrics") or {}).get(key, default)


def _result_passed(result: Mapping[str, Any]) -> bool:
    metrics = result.get("metrics") or {}
    if "passed" in metrics:
        return bool(metrics.get("passed"))
    verification = result.get("verification") or {}
    tier3 = verification.get("tier3_unit_tests") or {}
    return bool(tier3.get("passed"))


def _syntax_from_result(result: Mapping[str, Any]) -> JsonDict:
    diagnostics = result.get("diagnostics") or {}
    return {
        "parse_passed": bool(diagnostics.get("final_full_code_parse_passed", False)),
        "compile_passed": bool(diagnostics.get("final_full_code_compile_passed", False)),
        "parse_error": diagnostics.get("final_full_code_parse_error"),
        "compile_error": diagnostics.get("final_full_code_compile_error"),
    }


def _trace_features_from_result(result: Mapping[str, Any]) -> JsonDict:
    features = compute_shape_features(result.get("step_traces") or [])
    selected_len = _metric(result, "selected_mask_length", _metric(result, "mask_length"))
    features["selected_len"] = selected_len
    remaining_last = features.get("remaining_last")
    if remaining_last is not None and selected_len is not None:
        features["final_remaining_ratio_by_selected"] = _safe_div(float(remaining_last), float(max(int(selected_len), 1)))
    else:
        features["final_remaining_ratio_by_selected"] = None
    return features


def build_candidate_record(spec: CandidateSpec, result: Mapping[str, Any]) -> JsonDict:
    metrics = result.get("metrics") or {}
    selected_len = metrics.get("selected_mask_length", metrics.get("mask_length"))
    trace_features = _trace_features_from_result(result)
    return {
        "candidate_id": spec.candidate_id,
        "requested_min_length": int(spec.min_length),
        "steps": int(spec.steps),
        "selected_mask_length": None if selected_len is None else int(selected_len),
        "middle_text": result.get("middle_text") or "",
        "code": result.get("code") or "",
        "trace_features": trace_features,
        "syntax": _syntax_from_result(result),
        "total_sec_including_probe": _num(metrics.get("total_sec_including_probe")),
        "decode_sec": _num(metrics.get("decode_sec")),
        "verification_sec": _num(metrics.get("verification_sec")),
        "offline_passed": _result_passed(result),
        "offline_oracle_mask_length": metrics.get("oracle_mask_length"),
    }


def selector_metadata(selector_name: str) -> JsonDict:
    if selector_name == "consensus_confidence":
        return {
            "selector": selector_name,
            "claim_boundary": "pure_inference_time_verifier_free",
            "used_policy_fields": list(CONSENSUS_CONFIDENCE_ALLOWED_FIELDS),
        }
    if selector_name == "syntax_aware":
        return {
            "selector": selector_name,
            "claim_boundary": "compiler_assisted_inference_time",
            "used_policy_fields": list(SYNTAX_AWARE_ALLOWED_FIELDS),
        }
    if selector_name == "oracle_upper_bound":
        return {
            "selector": selector_name,
            "claim_boundary": "offline_only_not_deployable",
            "used_policy_fields": ["hidden_unit_test_passed"],
        }
    raise ValueError(f"unknown selector: {selector_name}")


def policy_candidate_view(candidate: Mapping[str, Any], *, include_syntax: bool = False) -> JsonDict:
    trace = candidate.get("trace_features") or {}
    view: JsonDict = {
        "candidate_id": candidate.get("candidate_id"),
        "middle_text": candidate.get("middle_text") or "",
        "selected_mask_length": candidate.get("selected_mask_length"),
        "trace_features": {
            "confidence_last": trace.get("confidence_last"),
            "confidence_max": trace.get("confidence_max"),
            "confidence_median": trace.get("confidence_median"),
            "top1_last": trace.get("top1_last"),
            "top1_median": trace.get("top1_median"),
            "gap_last": trace.get("gap_last"),
            "gap_median": trace.get("gap_median"),
            "final_remaining_ratio_by_selected": trace.get("final_remaining_ratio_by_selected"),
        },
    }
    if include_syntax:
        syntax = candidate.get("syntax") or {}
        view["syntax"] = {
            "parse_passed": bool(syntax.get("parse_passed")),
            "compile_passed": bool(syntax.get("compile_passed")),
        }
    return view


def _text_consensus(candidate: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> float:
    text = str(candidate.get("middle_text") or "")
    others = [str(other.get("middle_text") or "") for other in candidates if other is not candidate]
    if not others:
        return 1.0
    return sum(SequenceMatcher(None, text, other).ratio() for other in others) / float(len(others))


def _candidate_base_score(candidate: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Tuple[float, JsonDict]:
    trace = candidate.get("trace_features") or {}
    selected_len = _num(candidate.get("selected_mask_length"), default=64.0)
    confidence = max(
        _num(trace.get("confidence_last")),
        _num(trace.get("confidence_max")),
        _num(trace.get("confidence_median")),
    )
    top1 = max(_num(trace.get("top1_last")), _num(trace.get("top1_median")))
    gap = max(_num(trace.get("gap_last")), _num(trace.get("gap_median")))
    remaining_ratio = _num(trace.get("final_remaining_ratio_by_selected"))
    consensus = _text_consensus(candidate, candidates)
    length_penalty = selected_len / 64.0
    score = (
        1.00 * consensus
        + 1.00 * confidence
        + 0.50 * top1
        + 0.50 * gap
        - 1.25 * remaining_ratio
        - 0.05 * length_penalty
    )
    parts = {
        "text_consensus": consensus,
        "confidence": confidence,
        "top1": top1,
        "gap": gap,
        "final_remaining_ratio_by_selected": remaining_ratio,
        "length_penalty": length_penalty,
    }
    return score, parts


def score_consensus_confidence(candidate: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Tuple[float, JsonDict]:
    score, parts = _candidate_base_score(candidate, candidates)
    return score, parts


def score_syntax_aware(candidate: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Tuple[float, JsonDict]:
    score, parts = _candidate_base_score(candidate, candidates)
    syntax = candidate.get("syntax") or {}
    parse_bonus = 0.20 if bool(syntax.get("parse_passed")) else -0.20
    compile_bonus = 0.20 if bool(syntax.get("compile_passed")) else -0.20
    score += parse_bonus + compile_bonus
    parts = dict(parts)
    parts.update({"parse_bonus": parse_bonus, "compile_bonus": compile_bonus})
    return score, parts


def select_candidate(candidates: Sequence[Mapping[str, Any]], *, selector_name: str) -> Tuple[JsonDict, JsonDict]:
    if not candidates:
        raise ValueError("select_candidate requires at least one candidate")
    metadata = selector_metadata(selector_name)
    score_fn = score_syntax_aware if selector_name == "syntax_aware" else score_consensus_confidence
    include_syntax = selector_name == "syntax_aware"
    policy_views = [policy_candidate_view(candidate, include_syntax=include_syntax) for candidate in candidates]
    originals_by_id = {str(candidate.get("candidate_id")): dict(candidate) for candidate in candidates}
    scored: List[Tuple[float, int, str, JsonDict, JsonDict]] = []
    for candidate in policy_views:
        score, parts = score_fn(candidate, policy_views)
        selected_len = int(candidate.get("selected_mask_length") or 999999)
        candidate_id = str(candidate.get("candidate_id"))
        scored.append((float(score), -selected_len, candidate_id, originals_by_id[candidate_id], parts))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_score, _neg_len, best_id, best_candidate, best_parts = scored[0]
    candidate_scores = {candidate_id: score for score, _len_key, candidate_id, _candidate, _parts in scored}
    return best_candidate, {
        **metadata,
        "selected_candidate_id": best_id,
        "selected_score": best_score,
        "selected_score_parts": best_parts,
        "candidate_scores": candidate_scores,
    }


def oracle_upper_bound_summary(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    passing = [str(candidate.get("candidate_id")) for candidate in candidates if bool(candidate.get("offline_passed"))]
    return {
        "offline_only": True,
        "any_candidate_passed": bool(passing),
        "passing_candidate_ids": passing,
        "candidate_count": len(candidates),
    }


def pairwise_vs_references(
    results: Sequence[Mapping[str, Any]],
    references: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    return {
        name: route2_runner.pairwise_vs_baseline(results, rows)
        for name, rows in references.items()
        if rows
    }


def add_route2_rescue_quality_v5_metadata(
    final_result: JsonDict,
    primary_result: Mapping[str, Any],
    *,
    route2_policy_name: str,
    route2_triggered: bool,
    candidates: Sequence[Mapping[str, Any]],
    selected_candidate: Optional[Mapping[str, Any]],
    selector_name: str,
    selector_scores: Mapping[str, Any],
) -> None:
    final_metrics = final_result["metrics"]
    primary_metrics = primary_result["metrics"]
    primary_total = _num(primary_metrics.get("total_sec_including_probe"))
    candidate_total = sum(_num(candidate.get("total_sec_including_probe")) for candidate in candidates)
    combined_total = primary_total + (candidate_total if route2_triggered else 0.0)
    selected_id = str(selected_candidate.get("candidate_id")) if selected_candidate else "primary"

    if route2_triggered:
        final_metrics["decode_sec"] = _num(primary_metrics.get("decode_sec")) + sum(
            _num(candidate.get("decode_sec")) for candidate in candidates
        )
        final_metrics["verification_sec"] = _num(primary_metrics.get("verification_sec")) + sum(
            _num(candidate.get("verification_sec")) for candidate in candidates
        )
        final_metrics["total_sec"] = _num(primary_metrics.get("total_sec")) + sum(
            _num(candidate.get("decode_sec")) + _num(candidate.get("verification_sec")) for candidate in candidates
        )
        final_metrics["length_probe_sec"] = _num(primary_metrics.get("length_probe_sec"))
        final_metrics["total_sec_including_probe"] = combined_total

    final_metrics.update(
        {
            "mask_length_source": "route2_rescue_quality_v5",
            "route2_policy": route2_policy_name,
            "route2_trace_rescue_triggered": bool(route2_triggered),
            "route2_final_source": "rescue_quality_v5" if route2_triggered else "primary",
            "route2_primary_passed": bool(primary_metrics.get("passed")),
            "route2_primary_selected_mask_length": primary_metrics.get("selected_mask_length"),
            "route2_primary_final_source": primary_metrics.get("final_source"),
            "route2_primary_stop_reason": primary_metrics.get("stop_reason"),
            "route2_primary_total_sec_including_probe": primary_total,
            "route2_rescue_total_sec_including_probe": candidate_total if route2_triggered else 0.0,
            "route2_combined_total_sec_including_probe": combined_total,
            "route2_rescue_quality_v5_triggered": bool(route2_triggered),
            "route2_rescue_quality_v5_selector": selector_name,
            "route2_rescue_quality_v5_selected_candidate_id": selected_id,
            "route2_rescue_quality_v5_candidate_count": len(candidates),
            "route2_rescue_quality_v5_candidate_upper_bound_passed": any(
                bool(candidate.get("offline_passed")) for candidate in candidates
            ),
        }
    )
    final_result["route2_rescue_quality_v5"] = {
        "route2_policy": route2_policy_name,
        "triggered": bool(route2_triggered),
        "selector": selector_name,
        "selector_scores": dict(selector_scores),
        "selected_candidate_id": selected_id,
        "candidates": [dict(candidate) for candidate in candidates],
        "oracle_upper_bound": oracle_upper_bound_summary(candidates),
        "primary_metrics": {
            key: primary_metrics.get(key)
            for key in [
                "passed",
                "selected_mask_length",
                "oracle_mask_length",
                "final_source",
                "stop_reason",
                "total_sec_including_probe",
            ]
        },
    }


def result_payload(result: Mapping[str, Any], dataset_subset: str) -> JsonDict:
    payload = route2_runner.result_payload(result, dataset_subset)
    payload["middle_text"] = result.get("middle_text")
    payload["route2_rescue_quality_v5"] = result.get("route2_rescue_quality_v5")
    return payload


def summarize_v5(
    results: List[JsonDict],
    *,
    references: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_results_path: Optional[str],
    route2_reference_results_path: Optional[str],
) -> JsonDict:
    summary = summarize_results(results)
    metrics = [item["metrics"] for item in results]
    triggered = [m for m in metrics if bool(m.get("route2_rescue_quality_v5_triggered"))]
    triggered_oracle = [m for m in triggered if m.get("oracle_mask_length") is not None]
    buckets = sorted({route2_runner.oracle_bucket(m.get("oracle_mask_length")) for m in metrics})
    upper_bound_hits = [
        bool((row.get("route2_rescue_quality_v5") or {}).get("oracle_upper_bound", {}).get("any_candidate_passed"))
        for row in results
        if bool((row.get("route2_rescue_quality_v5") or {}).get("triggered"))
    ]
    summary.update(
        {
            "mask_length_source": "route2_rescue_quality_v5",
            "route2_trigger_count": len(triggered),
            "route2_trigger_rate": len(triggered) / len(metrics) if metrics else None,
            "route2_trigger_true_long_precision": route2_runner.rate(
                int(m.get("oracle_mask_length")) >= 17 for m in triggered_oracle
            ),
            "route2_trigger_oracle_bucket_histogram": dict(
                sorted(Counter(route2_runner.oracle_bucket(m.get("oracle_mask_length")) for m in triggered).items())
            ),
            "route2_rescue_quality_v5_selector_histogram": dict(
                sorted(Counter(m.get("route2_rescue_quality_v5_selector") for m in metrics).items())
            ),
            "route2_rescue_quality_v5_candidate_count_histogram": dict(
                sorted(Counter(m.get("route2_rescue_quality_v5_candidate_count") for m in metrics).items())
            ),
            "route2_rescue_quality_v5_oracle_upper_bound_pass_rate_on_triggered": route2_runner.rate(upper_bound_hits),
            "oracle_bucket_pass_rate": {
                bucket: route2_runner.rate(
                    bool(m.get("passed")) for m in metrics if route2_runner.oracle_bucket(m.get("oracle_mask_length")) == bucket
                )
                for bucket in buckets
            },
            "baseline_results": baseline_results_path,
            "route2_reference_results": route2_reference_results_path,
            "pairwise_vs_references": pairwise_vs_references(results, references),
        }
    )
    return summary


def write_candidate_upper_bound_csv(run_dir: str, results: Sequence[Mapping[str, Any]]) -> None:
    path = os.path.join(run_dir, "candidate_upper_bound.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "triggered",
                "selected_candidate_id",
                "any_candidate_passed",
                "passing_candidate_ids",
                "candidate_count",
            ],
        )
        writer.writeheader()
        for row in results:
            v5 = row.get("route2_rescue_quality_v5") or {}
            upper = v5.get("oracle_upper_bound") or {}
            writer.writerow(
                {
                    "task_id": row.get("task_id"),
                    "triggered": bool(v5.get("triggered")),
                    "selected_candidate_id": v5.get("selected_candidate_id"),
                    "any_candidate_passed": bool(upper.get("any_candidate_passed")),
                    "passing_candidate_ids": ",".join(upper.get("passing_candidate_ids") or []),
                    "candidate_count": upper.get("candidate_count", 0),
                }
            )


def run_candidate_rescue(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
    *,
    primary_selected_length: int,
    spec: CandidateSpec,
) -> Tuple[JsonDict, JsonDict]:
    rescue_cfg = copy.deepcopy(cfg)
    rescue_cfg.decode.total_steps = int(spec.steps)
    actual_length = max(int(primary_selected_length), int(spec.min_length))
    rescue_result = route2_runner.run_fixed_rescue(task, tokenizer, model, rescue_cfg, actual_length)
    candidate = build_candidate_record(spec, rescue_result)
    candidate["actual_rescue_length"] = actual_length
    return rescue_result, candidate


def run_experiment(
    cfg: ExperimentConfig,
    settings: route2_runner.midcons_runner.BoundedRepairSettings,
    *,
    route2_policy_name: str,
    specs: Sequence[CandidateSpec],
    selector_name: str,
    baseline_results_path: Optional[str],
    route2_reference_results_path: Optional[str],
) -> JsonDict:
    set_global_seed(cfg.decode.seed)
    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)
    rule = route2_runner.route2_rule(route2_policy_name)
    config_payload = cfg.to_dict()
    config_payload["route2_rescue_quality_v5"] = {
        "route2_policy": route2_policy_name,
        "clauses": route2_runner.POLICIES[route2_policy_name],
        "candidate_specs": [asdict(spec) for spec in specs],
        "selector": selector_metadata(selector_name),
        "primary_method": "lcal_official_bounded_repair_midcons",
        "oracle_upper_bound_boundary": "offline_only_not_deployable",
    }
    config_payload["baseline_results"] = baseline_results_path
    config_payload["route2_reference_results"] = route2_reference_results_path
    logger.save_config(config_payload)

    print("=" * 80, flush=True)
    print("Starting Route2 rescue-quality V5", flush=True)
    print(f"experiment_name       = {cfg.logging.experiment_name}", flush=True)
    print(f"route2_policy         = {route2_policy_name}", flush=True)
    print(f"selector              = {selector_name}", flush=True)
    print(f"candidate_specs       = {[asdict(spec) for spec in specs]}", flush=True)
    print(f"model_path            = {cfg.model.model_path}", flush=True)
    print(f"max_samples           = {cfg.data.max_samples}", flush=True)
    print(f"baseline              = {baseline_results_path}", flush=True)
    print(f"route2_reference      = {route2_reference_results_path}", flush=True)
    print(f"run_dir               = {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)

    tokenizer, model = load_model_and_tokenizer(cfg.model)
    tasks = load_humaneval_infilling(
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        dataset_subset=cfg.data.dataset_subset,
    )

    results: List[JsonDict] = []
    total_tasks = len(tasks)
    print(f"Loaded {total_tasks} tasks", flush=True)
    print("-" * 80, flush=True)

    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{total_tasks}] task_id={task.task_id} | primary start", flush=True)
        primary = route2_runner.run_primary(task, tokenizer, model, cfg, settings)
        record = route2_runner.trace_record_from_result(primary)
        triggered = rule.matches(record)
        candidates: List[JsonDict] = []
        candidate_results: Dict[str, JsonDict] = {}
        selected_candidate: Optional[JsonDict] = None
        selector_scores: JsonDict = selector_metadata(selector_name)
        final_result = primary

        if triggered:
            primary_selected = int(primary["metrics"]["selected_mask_length"])
            print(
                f"[{idx}/{total_tasks}] task_id={task.task_id} | route2 triggered | "
                f"primary_len={primary_selected} | candidates={','.join(spec.candidate_id for spec in specs)}",
                flush=True,
            )
            for spec in specs:
                rescue_result, candidate = run_candidate_rescue(
                    task,
                    tokenizer,
                    model,
                    cfg,
                    primary_selected_length=primary_selected,
                    spec=spec,
                )
                candidates.append(candidate)
                candidate_results[str(candidate["candidate_id"])] = rescue_result
            selected_candidate, selector_scores = select_candidate(candidates, selector_name=selector_name)
            selected_id = str(selected_candidate["candidate_id"])
            final_result = candidate_results[selected_id]
            print(
                f"[{idx}/{total_tasks}] task_id={task.task_id} | selected={selected_id} | "
                f"score={selector_scores['selected_score']:.4f}",
                flush=True,
            )

        final_result = copy.deepcopy(final_result)
        add_route2_rescue_quality_v5_metadata(
            final_result,
            primary,
            route2_policy_name=route2_policy_name,
            route2_triggered=triggered,
            candidates=candidates,
            selected_candidate=selected_candidate,
            selector_name=selector_name if triggered else "primary",
            selector_scores=selector_scores,
        )
        final_result["dataset_subset"] = cfg.data.dataset_subset
        results.append(final_result)
        logger.log_result(result_payload(final_result, cfg.data.dataset_subset))

        for trace in primary.get("step_traces") or []:
            trace_payload = dict(trace)
            trace_payload["route2_rescue_quality_v5_phase"] = "primary"
            logger.log_trace(trace_payload)
        for candidate_id, candidate_result in candidate_results.items():
            for trace in candidate_result.get("step_traces") or []:
                trace_payload = dict(trace)
                trace_payload["route2_rescue_quality_v5_phase"] = "candidate"
                trace_payload["route2_rescue_quality_v5_candidate_id"] = candidate_id
                logger.log_trace(trace_payload)

        m = final_result["metrics"]
        print(
            f"[{idx}/{total_tasks}] task_id={task.task_id} | "
            f"{'PASS' if m['passed'] else 'FAIL'} | "
            f"triggered={m['route2_rescue_quality_v5_triggered']} | "
            f"selected={m['route2_rescue_quality_v5_selected_candidate_id']} | "
            f"primary_pass={m['route2_primary_passed']} | "
            f"final_len={m['selected_mask_length']} | "
            f"oracle_len={m['oracle_mask_length']} | "
            f"combined_sec={m['route2_combined_total_sec_including_probe']:.3f}",
            flush=True,
        )

    baseline_rows = route2_runner.load_jsonl(baseline_results_path)
    route2_reference_rows = route2_runner.load_jsonl(route2_reference_results_path)
    references = {
        "midcons": baseline_rows,
        "route2_precision_len32": route2_reference_rows,
    }
    summary = summarize_v5(
        results,
        references=references,
        baseline_results_path=baseline_results_path,
        route2_reference_results_path=route2_reference_results_path,
    )
    logger.save_json("summary.json", summary)
    write_candidate_upper_bound_csv(logger.run_dir, results)

    print("-" * 80, flush=True)
    print("Route2 rescue-quality V5 finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)
    return {"results": results, "summary": summary, "run_dir": logger.run_dir}


def main() -> None:
    args = parse_args()
    cfg = make_cfg(args)
    settings = route2_runner.default_midcons_settings()
    output = run_experiment(
        cfg,
        settings,
        route2_policy_name=args.route2_policy,
        specs=candidate_specs(args.candidate_set),
        selector_name=args.selector,
        baseline_results_path=args.baseline_results,
        route2_reference_results_path=args.route2_reference_results,
    )
    print("===== Route2 Rescue Quality V5 Summary =====")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")
    print(f"\nRun directory: {output['run_dir']}")


if __name__ == "__main__":
    main()
