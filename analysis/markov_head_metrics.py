"""CPU-only metrics shared by training and retrospective report repair."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Mapping, Sequence
import numpy as np


def no_head_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(summary)
    for suffix in ("raw_tv", "top1_agreement", "reference_nll", "reference_rank"):
        result["head_" + suffix] = result.get("baseline_" + suffix)
    for field in ("relative_raw_tv_improvement", "cluster_tv_improvement",
                  "cluster_tv_improvement_ci95_low", "cluster_tv_improvement_ci95_high",
                  "mismatch_recovery", "stable_corruption",
                  "recovered_count", "corrupted_count",
                  "top_p_support_enter_count_mean", "top_p_support_exit_count_mean"):
        result[field] = 0.0
    return result

def clustered_ci(rows: Sequence[Mapping[str, Any]], field: str, *, seed: int, reps: int) -> tuple[float, float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            groups[str(row["problem_group_id"])].append(float(value))
    values = np.asarray([np.mean(items) for items in groups.values()], dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(reps, len(values)))
    bootstrap = values[indexes].mean(1)
    return float(values.mean()), float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def summarize(rows: Sequence[Mapping[str, Any]], *, bootstrap_reps: int) -> dict[str, Any]:
    baseline = float(np.mean([row["baseline_tv"] for row in rows]))
    head = float(np.mean([row["head_tv"] for row in rows]))
    improvement, low, high = clustered_ci(
        rows, "tv_improvement", seed=20260901, reps=bootstrap_reps
    )
    aligned = [row for row in rows if row["reference_aligned"]]
    mismatch = [row for row in rows if not row["baseline_top1_agreement"]]
    stable = [row for row in rows if row["baseline_top1_agreement"]]
    return {
        "transitions": len(rows),
        "problem_groups": len({row["problem_group_id"] for row in rows}),
        "baseline_raw_tv": baseline,
        "head_raw_tv": head,
        "relative_raw_tv_improvement": (baseline - head) / baseline if baseline else 0.0,
        "cluster_tv_improvement": improvement,
        "cluster_tv_improvement_ci95_low": low,
        "cluster_tv_improvement_ci95_high": high,
        "baseline_top1_agreement": float(np.mean([row["baseline_top1_agreement"] for row in rows])),
        "head_top1_agreement": float(np.mean([row["head_top1_agreement"] for row in rows])),
        "mismatch_transitions": len(mismatch),
        "stable_transitions": len(stable),
        "mismatch_recovery": (
            float(np.mean([row["mismatch_recovery"] for row in mismatch])) if mismatch else 0.0
        ),
        "stable_corruption": (
            float(np.mean([row["stable_corruption"] for row in stable])) if stable else 0.0
        ),
        "aligned_transitions": len(aligned),
        "baseline_reference_nll": (
            float(np.mean([row["baseline_reference_nll"] for row in aligned])) if aligned else None
        ),
        "head_reference_nll": (
            float(np.mean([row["head_reference_nll"] for row in aligned])) if aligned else None
        ),
    }
