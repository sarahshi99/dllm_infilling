from __future__ import annotations

from typing import Any, Dict, List, Optional


def _avg(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"num_samples": 0}

    num_samples = len(results)
    passed = sum(1 for r in results if r["metrics"]["passed"])
    avg_decode_sec = sum(r["metrics"]["decode_sec"] for r in results) / num_samples
    avg_total_sec = sum(r["metrics"]["total_sec"] for r in results) / num_samples
    early_stop_count = sum(1 for r in results if r["metrics"].get("early_stop"))
    backslide_count = sum(1 for r in results if r["metrics"].get("backslide") is True)
    rescue_count = sum(1 for r in results if r["metrics"].get("rescue"))

    first_pass_steps = [r["metrics"]["first_pass_step"] for r in results if r["metrics"].get("first_pass_step") is not None]
    avg_first_pass_step = sum(first_pass_steps) / len(first_pass_steps) if first_pass_steps else None

    wasted_steps = [
        r["metrics"]["wasted_steps_after_first_pass"]
        for r in results
        if r["metrics"].get("wasted_steps_after_first_pass") is not None
    ]
    avg_wasted_steps = sum(wasted_steps) / len(wasted_steps) if wasted_steps else None

    inference_verifier_calls = [float(r["metrics"].get("inference_verifier_calls", 0)) for r in results]
    inference_verifier_sec = [float(r["metrics"].get("inference_verifier_sec", 0.0)) for r in results]
    final_verifier_calls = [float(r["metrics"].get("final_verifier_calls", 0)) for r in results]
    final_verifier_sec = [float(r["metrics"].get("final_verifier_sec", 0.0)) for r in results]
    verifier_calls_total = [float(r["metrics"].get("verifier_calls_total", 0)) for r in results]
    verifier_total_sec = [float(r["metrics"].get("verifier_total_sec", 0.0)) for r in results]
    mean_final_confidence = [
        float(r["metrics"].get("mean_final_confidence"))
        for r in results
        if r["metrics"].get("mean_final_confidence") is not None
    ]
    selected_snapshot_steps = [
        float(r["metrics"].get("selected_snapshot_step"))
        for r in results
        if r["metrics"].get("selected_snapshot_step") is not None
    ]

    summary = {
        "num_samples": num_samples,
        "pass_rate": passed / num_samples,
        "avg_decode_sec": avg_decode_sec,
        "avg_total_sec": avg_total_sec,
        "early_stop_rate": early_stop_count / num_samples,
        "backslide_rate": backslide_count / num_samples,
        "rescue_rate": rescue_count / num_samples,
        "avg_first_pass_step": avg_first_pass_step,
        "avg_wasted_steps_after_first_pass": avg_wasted_steps,
        "avg_inference_verifier_calls": _avg(inference_verifier_calls),
        "avg_inference_verifier_sec": _avg(inference_verifier_sec),
        "avg_final_verifier_calls": _avg(final_verifier_calls),
        "avg_final_verifier_sec": _avg(final_verifier_sec),
        "avg_verifier_calls_total": _avg(verifier_calls_total),
        "avg_verifier_total_sec": _avg(verifier_total_sec),
        "avg_mean_final_confidence": _avg(mean_final_confidence),
        "avg_selected_snapshot_step": _avg(selected_snapshot_steps),
    }

    modes = sorted({str(r["metrics"].get("g_control_mode")) for r in results if r["metrics"].get("g_control_mode") is not None})
    if modes:
        summary["g_control_modes"] = modes

    rm_modes = sorted({str(r["metrics"].get("rm_control_mode")) for r in results if r["metrics"].get("rm_control_mode") is not None})
    if rm_modes:
        summary["rm_control_modes"] = rm_modes

    s_modes = sorted({str(r["metrics"].get("s_control_mode")) for r in results if r["metrics"].get("s_control_mode") is not None})
    if s_modes:
        summary["s_control_modes"] = s_modes

    dataset_subsets = sorted({str(r.get("dataset_subset")) for r in results if r.get("dataset_subset")})
    if dataset_subsets:
        summary["dataset_subsets"] = dataset_subsets

    sampling_policies = sorted({str(r["metrics"].get("sampling_policy_version")) for r in results if r["metrics"].get("sampling_policy_version")})
    if sampling_policies:
        summary["sampling_policy_versions"] = sampling_policies

    feature_schemas = sorted({str(r["metrics"].get("feature_schema_version")) for r in results if r["metrics"].get("feature_schema_version")})
    if feature_schemas:
        summary["feature_schema_versions"] = feature_schemas

    selector_versions = sorted({str(r["metrics"].get("selector_version")) for r in results if r["metrics"].get("selector_version")})
    if selector_versions:
        summary["selector_versions"] = selector_versions

    verifier_versions = sorted({str(r["metrics"].get("verifier_version")) for r in results if r["metrics"].get("verifier_version")})
    if verifier_versions:
        summary["verifier_versions"] = verifier_versions

    baseline_chain_versions = sorted({str(r["metrics"].get("baseline_chain_version")) for r in results if r["metrics"].get("baseline_chain_version")})
    if baseline_chain_versions:
        summary["baseline_chain_versions"] = baseline_chain_versions

    return summary