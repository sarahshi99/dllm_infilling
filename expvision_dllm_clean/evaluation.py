from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


def _avg(values: Iterable[float | int | None]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    return (sum(filtered) / len(filtered)) if filtered else None


def _rate(values: Iterable[bool]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _histogram(values: Iterable[int | None]) -> Dict[str, int]:
    counter: Counter[int] = Counter()
    for value in values:
        if value is not None:
            counter[int(value)] += 1
    return {str(key): counter[key] for key in sorted(counter)}


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"num_samples": 0}

    num_samples = len(results)
    pass_rate = sum(1 for item in results if item["metrics"]["passed"]) / num_samples

    metrics = [item["metrics"] for item in results]

    selected_lengths = [m.get("selected_mask_length") for m in metrics]
    oracle_lengths = [m.get("oracle_mask_length") for m in metrics]
    length_diffs = [m.get("selected_minus_oracle_length") for m in metrics]
    abs_length_diffs = [m.get("abs_selected_minus_oracle_length") for m in metrics]

    valid_diffs = [d for d in length_diffs if d is not None]
    valid_abs_diffs = [d for d in abs_length_diffs if d is not None]

    summary = {
        "num_samples": num_samples,
        "pass_rate": pass_rate,
        "avg_decode_sec": _avg(m.get("decode_sec") for m in metrics),
        "avg_verification_sec": _avg(m.get("verification_sec") for m in metrics),
        "avg_total_sec": _avg(m.get("total_sec") for m in metrics),
        "avg_length_probe_sec": _avg(m.get("length_probe_sec") for m in metrics),
        "avg_total_sec_including_probe": _avg(m.get("total_sec_including_probe") for m in metrics),
        "avg_mask_length": _avg(m.get("mask_length") for m in metrics),
        "avg_selected_mask_length": _avg(selected_lengths),
        "avg_oracle_mask_length": _avg(oracle_lengths),
        "avg_selected_score": _avg(m.get("selected_score") for m in metrics),
        "avg_selected_raw_score": _avg(m.get("selected_raw_score") for m in metrics),
        "avg_selected_adjusted_score": _avg(m.get("selected_adjusted_score") for m in metrics),
        "oracle_length_available_rate": (
            sum(1 for m in metrics if m.get("oracle_mask_length") is not None) / num_samples
        ),
        "avg_selected_minus_oracle_length": _avg(valid_diffs),
        "avg_abs_selected_minus_oracle_length": _avg(valid_abs_diffs),
        "exact_length_match_rate": _rate(d == 0 for d in valid_diffs),
        "within_1_length_rate": _rate(abs(d) <= 1 for d in valid_diffs),
        "within_2_length_rate": _rate(abs(d) <= 2 for d in valid_diffs),
        "under_select_rate": _rate(d < 0 for d in valid_diffs),
        "over_select_rate": _rate(d > 0 for d in valid_diffs),
        "same_as_fixed_length_rate": _rate((v == 64) for v in selected_lengths if v is not None),
        "selected_length_histogram": _histogram(selected_lengths),
        "oracle_length_histogram": _histogram(oracle_lengths),
        "score_modes": sorted({str(m.get("score_mode")) for m in metrics if m.get("score_mode") is not None}),
        "length_alphas": sorted({float(m.get("length_alpha")) for m in metrics if m.get("length_alpha") is not None}),
    }

    win_over_fixed = sum(1 for item in results if item.get("comparison_to_fixed") == "win")
    loss_over_fixed = sum(1 for item in results if item.get("comparison_to_fixed") == "loss")
    tie_over_fixed = sum(1 for item in results if item.get("comparison_to_fixed") == "tie")
    if win_over_fixed or loss_over_fixed or tie_over_fixed:
        summary["pairwise_vs_fixed"] = {
            "win": win_over_fixed,
            "loss": loss_over_fixed,
            "tie": tie_over_fixed,
        }

    return summary