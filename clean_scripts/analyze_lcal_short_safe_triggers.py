#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_BASE_GRID = "3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24"
DEFAULT_WEAK_GRID = "13,14,15,16"
DEFAULT_STRONG_GRID = "13,14,15,16,20,24,28,32,40"


def parse_csv_ints(text: str) -> List[int]:
    values: List[int] = []
    seen = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Length values must be positive, got {value}")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("No length values parsed")
    return values


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def length_bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    value = int(length)
    if value <= 8:
        return "<=8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    if value <= 24:
        return "17-24"
    return "25+"


def avg(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def rate(values: Iterable[bool]) -> Optional[float]:
    vals = list(values)
    return sum(1 for v in vals if v) / len(vals) if vals else None


def hist(values: Iterable[Any]) -> Dict[str, int]:
    counter: Counter[str] = Counter(str(v) for v in values)
    return dict(sorted(counter.items(), key=lambda kv: (kv[0])))


def get_metric(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    return row.get("metrics", {}).get(key, default)


def get_probe(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("length_probe", {})


def get_base_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    probe = get_probe(row)
    candidates = probe.get("base_candidate_scores") or probe.get("candidate_scores")
    if not candidates:
        raise KeyError(
            f"Missing base_candidate_scores/candidate_scores for task_id={row.get('task_id')}"
        )
    return list(candidates)


def get_existing_long_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(get_probe(row).get("long_candidate_scores") or [])


def raw_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("raw_score", candidate.get("mean_top1_prob", candidate.get("score"))))


def adjusted_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("score", candidate.get("adjusted_score", raw_score(candidate))))


def power_score(candidate: Dict[str, Any], alpha: float, score_mode: str = "length_power") -> float:
    raw = raw_score(candidate)
    length = int(candidate["mask_length"])
    if score_mode == "raw":
        return raw
    if score_mode == "length_power":
        return raw * (float(length) ** float(alpha))
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def pick_best(
    candidates: List[Dict[str, Any]],
    tie_break: str,
    score_fn,
) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("Cannot pick best from empty candidate list")
    if tie_break not in {"shorter", "longer"}:
        raise ValueError(f"Unsupported tie_break={tie_break}")
    if tie_break == "shorter":
        return max(candidates, key=lambda c: (score_fn(c), -int(c["mask_length"])))
    return max(candidates, key=lambda c: (score_fn(c), int(c["mask_length"])))


def candidate_by_length(candidates: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(c["mask_length"]): c for c in candidates}


def base_curve_info(row: Dict[str, Any], tie_break: str) -> Dict[str, Any]:
    base_candidates = get_base_candidates(row)
    best = pick_best(base_candidates, tie_break=tie_break, score_fn=adjusted_score)
    long_candidates = [c for c in base_candidates if int(c["mask_length"]) >= 13]
    best_long = pick_best(long_candidates, tie_break=tie_break, score_fn=adjusted_score)

    best_raw = pick_best(base_candidates, tie_break=tie_break, score_fn=raw_score)
    best_long_raw = pick_best(long_candidates, tie_break=tie_break, score_fn=raw_score)

    best_score = adjusted_score(best)
    best_long_score = adjusted_score(best_long)
    best_raw_score = raw_score(best_raw)
    best_long_raw_score = raw_score(best_long_raw)

    support_count = sum(
        1
        for c in long_candidates
        if adjusted_score(c) >= 0.0  # placeholder, overwritten by policy thresholds
    )

    return {
        "best_len": int(best["mask_length"]),
        "best_score": best_score,
        "best_long_len": int(best_long["mask_length"]),
        "best_long_score": best_long_score,
        "long_ratio": best_long_score / best_score if best_score else None,
        "best_raw_len": int(best_raw["mask_length"]),
        "best_raw_score": best_raw_score,
        "best_long_raw_len": int(best_long_raw["mask_length"]),
        "best_long_raw_score": best_long_raw_score,
        "raw_long_ratio": best_long_raw_score / best_raw_score if best_raw_score else None,
        "long_support_count_placeholder": support_count,
    }


def existing_or_base_final_length(row: Dict[str, Any]) -> int:
    # For S0, prefer the actual final length from the current LCAL-v3 result.
    value = get_metric(row, "selected_mask_length")
    if value is not None:
        return int(value)
    value = get_metric(row, "lcal_v3_base_selected_mask_length")
    if value is not None:
        return int(value)
    return int(base_curve_info(row, tie_break="shorter")["best_len"])


def select_from_lengths(
    row: Dict[str, Any],
    lengths: List[int],
    alpha: float,
    tie_break: str,
    score_mode: str,
) -> Optional[int]:
    base_map = candidate_by_length(get_base_candidates(row))
    long_map = candidate_by_length(get_existing_long_candidates(row))
    candidates: List[Dict[str, Any]] = []
    for length in lengths:
        if length in long_map:
            candidates.append(long_map[length])
        elif length in base_map:
            candidates.append(base_map[length])
    if not candidates:
        return None
    best = pick_best(
        candidates,
        tie_break=tie_break,
        score_fn=lambda c: power_score(c, alpha=alpha, score_mode=score_mode),
    )
    return int(best["mask_length"])


def jump_cap_for_base(base_len: int, cap_base_le8: int, cap_base_9_12: int) -> Optional[int]:
    if base_len <= 8:
        return cap_base_le8
    if 9 <= base_len <= 12:
        return cap_base_9_12
    return None


def simulate_policy(
    row: Dict[str, Any],
    policy: str,
    tie_break: str,
    ratio_threshold: float,
    long_score_floor: float,
    raw_ratio_threshold: float,
    support_count_threshold: int,
    strong_min_len: int,
    weak_min_base_len: int,
    weak_max_base_len: int,
    weak_grid: List[int],
    strong_grid: List[int],
    long_alpha: float,
    score_mode: str,
    cap_base_le8: int,
    cap_base_9_12: int,
) -> Dict[str, Any]:
    info = base_curve_info(row, tie_break=tie_break)
    base_len = int(get_metric(row, "lcal_v3_base_selected_mask_length", info["best_len"]))
    if base_len != int(info["best_len"]):
        # Prefer logged base_selected_length for reproducibility, but keep best_len for diagnostics.
        pass

    best_score = float(info["best_score"])
    best_long_score = float(info["best_long_score"])
    long_ratio = info["long_ratio"]
    raw_long_ratio = info["raw_long_ratio"]

    long_score_floor_passed = best_long_score >= long_score_floor
    ratio_passed = long_ratio is not None and long_ratio >= ratio_threshold and long_score_floor_passed
    raw_ratio_passed = raw_long_ratio is not None and raw_long_ratio >= raw_ratio_threshold
    support_count = sum(
        1
        for c in get_base_candidates(row)
        if int(c["mask_length"]) >= strong_min_len and adjusted_score(c) >= ratio_threshold * best_score
    )
    support_passed = support_count >= support_count_threshold

    strong_trigger = base_len >= strong_min_len
    weak_window = weak_min_base_len <= base_len <= weak_max_base_len

    weak_trigger = False
    trigger_reason = "none"
    correction_kind = "none"
    final_len = base_len
    correction_len: Optional[int] = None

    if policy == "s0":
        if strong_trigger:
            trigger_reason = "strong_base_len"
            correction_kind = "strong_existing"
            final_len = existing_or_base_final_length(row)
            correction_len = final_len
    elif policy == "s1":
        if strong_trigger:
            trigger_reason = "strong_base_len"
            correction_kind = "strong_full"
            correction_len = select_from_lengths(row, strong_grid, long_alpha, tie_break, score_mode)
        elif weak_window and ratio_passed:
            trigger_reason = "weak_ratio"
            correction_kind = "weak_grid"
            correction_len = select_from_lengths(row, weak_grid, long_alpha, tie_break, score_mode)
    elif policy == "s2":
        if strong_trigger:
            trigger_reason = "strong_base_len"
            correction_kind = "strong_full"
            correction_len = select_from_lengths(row, strong_grid, long_alpha, tie_break, score_mode)
        elif weak_window and ratio_passed:
            cap = jump_cap_for_base(base_len, cap_base_le8, cap_base_9_12)
            effective_grid = [length for length in weak_grid if cap is None or length <= cap]
            trigger_reason = "weak_ratio_jump_cap"
            correction_kind = "weak_grid_capped"
            correction_len = select_from_lengths(row, effective_grid, long_alpha, tie_break, score_mode)
    elif policy == "s3":
        if strong_trigger:
            trigger_reason = "strong_base_len"
            correction_kind = "strong_full"
            correction_len = select_from_lengths(row, strong_grid, long_alpha, tie_break, score_mode)
        elif weak_window and ratio_passed and raw_ratio_passed:
            cap = jump_cap_for_base(base_len, cap_base_le8, cap_base_9_12)
            effective_grid = [length for length in weak_grid if cap is None or length <= cap]
            trigger_reason = "weak_ratio_raw_confirmed_jump_cap"
            correction_kind = "weak_grid_capped"
            correction_len = select_from_lengths(row, effective_grid, long_alpha, tie_break, score_mode)
    elif policy == "s4":
        if strong_trigger:
            trigger_reason = "strong_base_len"
            correction_kind = "strong_full"
            correction_len = select_from_lengths(row, strong_grid, long_alpha, tie_break, score_mode)
        elif weak_window and ratio_passed and support_passed:
            cap = jump_cap_for_base(base_len, cap_base_le8, cap_base_9_12)
            effective_grid = [length for length in weak_grid if cap is None or length <= cap]
            trigger_reason = "weak_ratio_support_count_jump_cap"
            correction_kind = "weak_grid_capped"
            correction_len = select_from_lengths(row, effective_grid, long_alpha, tie_break, score_mode)
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    triggered = correction_len is not None
    if triggered:
        final_len = max(base_len, int(correction_len))

    oracle_len = get_metric(row, "oracle_mask_length")
    oracle_len = None if oracle_len is None else int(oracle_len)
    diff = None if oracle_len is None else final_len - oracle_len

    return {
        "policy": policy,
        "task_id": row.get("task_id"),
        "oracle_length": oracle_len,
        "oracle_bucket": length_bucket(oracle_len),
        "base_len": base_len,
        "final_len": final_len,
        "final_minus_base": final_len - base_len,
        "selected_minus_oracle": diff,
        "abs_selected_minus_oracle": None if diff is None else abs(diff),
        "triggered": bool(triggered),
        "strong_trigger": bool(triggered and trigger_reason == "strong_base_len"),
        "weak_trigger": bool(triggered and trigger_reason.startswith("weak_")),
        "trigger_reason": trigger_reason,
        "correction_kind": correction_kind,
        "correction_len": correction_len,
        "best_len": info["best_len"],
        "best_score": info["best_score"],
        "best_long_len": info["best_long_len"],
        "best_long_score": info["best_long_score"],
        "long_ratio": info["long_ratio"],
        "raw_long_ratio": info["raw_long_ratio"],
        "long_score_floor_passed": bool(long_score_floor_passed),
        "ratio_passed": bool(ratio_passed),
        "raw_ratio_passed": bool(raw_ratio_passed),
        "support_count": int(support_count),
        "support_passed": bool(support_passed),
    }


def summarize_simulations(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    short = [item for item in items if item["oracle_length"] is not None and item["oracle_length"] <= 8]
    true_long = [item for item in items if item["oracle_length"] is not None and item["oracle_length"] >= 17]
    medium = [item for item in items if item["oracle_length"] is not None and 9 <= item["oracle_length"] <= 16]
    weak = [item for item in items if item["weak_trigger"]]
    strong = [item for item in items if item["strong_trigger"]]

    def selected_under(item: Dict[str, Any]) -> bool:
        return item["selected_minus_oracle"] is not None and item["selected_minus_oracle"] < 0

    def selected_under_by3(item: Dict[str, Any]) -> bool:
        return item["selected_minus_oracle"] is not None and item["selected_minus_oracle"] <= -3

    def selected_over(item: Dict[str, Any]) -> bool:
        return item["selected_minus_oracle"] is not None and item["selected_minus_oracle"] > 0

    short_large_jump = [item for item in short if item["final_minus_base"] >= 4]
    short_changed = [item for item in short if item["final_minus_base"] != 0]

    return {
        "num_samples": n,
        "trigger_count": sum(1 for item in items if item["triggered"]),
        "trigger_rate": rate(item["triggered"] for item in items),
        "strong_trigger_count": len(strong),
        "strong_trigger_rate": len(strong) / n if n else None,
        "weak_trigger_count": len(weak),
        "weak_trigger_rate": len(weak) / n if n else None,
        "true_long_count": len(true_long),
        "true_long_trigger_count": sum(1 for item in true_long if item["triggered"]),
        "true_long_recall": rate(item["triggered"] for item in true_long),
        "true_long_under_select_rate": rate(selected_under(item) for item in true_long),
        "true_long_under_by3_rate": rate(selected_under_by3(item) for item in true_long),
        "short_count": len(short),
        "short_false_trigger_count": sum(1 for item in short if item["triggered"]),
        "short_false_trigger_rate": rate(item["triggered"] for item in short),
        "short_large_jump_count": len(short_large_jump),
        "short_large_jump_rate": len(short_large_jump) / len(short) if short else None,
        "short_changed_count": len(short_changed),
        "short_changed_rate": len(short_changed) / len(short) if short else None,
        "short_over_select_rate": rate(selected_over(item) for item in short),
        "medium_count": len(medium),
        "medium_trigger_count": sum(1 for item in medium if item["triggered"]),
        "medium_trigger_rate": rate(item["triggered"] for item in medium),
        "weak_trigger_precision_true_long": (
            sum(1 for item in weak if item["oracle_length"] is not None and item["oracle_length"] >= 17) / len(weak)
            if weak
            else None
        ),
        "avg_final_minus_base_all": avg(item["final_minus_base"] for item in items),
        "avg_final_minus_base_short": avg(item["final_minus_base"] for item in short),
        "avg_selected_minus_oracle_all": avg(item["selected_minus_oracle"] for item in items),
        "avg_selected_minus_oracle_true_long": avg(item["selected_minus_oracle"] for item in true_long),
        "final_length_histogram": hist(item["final_len"] for item in items),
        "trigger_reason_histogram": hist(item["trigger_reason"] for item in items),
        "oracle_bucket_trigger_rate": {
            bucket: rate(item["triggered"] for item in items if item["oracle_bucket"] == bucket)
            for bucket in ["<=8", "9-12", "13-16", "17-24", "25+", "unknown"]
        },
        "oracle_bucket_avg_final_minus_base": {
            bucket: avg(item["final_minus_base"] for item in items if item["oracle_bucket"] == bucket)
            for bucket in ["<=8", "9-12", "13-16", "17-24", "25+", "unknown"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline short-safe LCAL trigger diagnostics using saved results.jsonl."
    )
    parser.add_argument("--results", required=True, help="Path to LCAL-v3/T2 results.jsonl with candidate scores.")
    parser.add_argument("--output-json", required=True, help="Path to write summary JSON.")
    parser.add_argument("--output-csv", default=None, help="Optional path to write per-sample simulated trigger rows.")
    parser.add_argument("--policies", default="s0,s1,s2,s3,s4")
    parser.add_argument("--ratio-threshold", type=float, default=0.97)
    parser.add_argument("--long-score-floor", type=float, default=0.55)
    parser.add_argument("--raw-ratio-threshold", type=float, default=0.97)
    parser.add_argument("--support-count-threshold", type=int, default=2)
    parser.add_argument("--strong-min-len", type=int, default=13)
    parser.add_argument("--weak-min-base-len", type=int, default=8)
    parser.add_argument("--weak-max-base-len", type=int, default=12)
    parser.add_argument("--weak-grid", default=DEFAULT_WEAK_GRID)
    parser.add_argument("--strong-grid", default=DEFAULT_STRONG_GRID)
    parser.add_argument("--long-alpha", type=float, default=0.10)
    parser.add_argument("--score-mode", default="length_power", choices=["raw", "length_power"])
    parser.add_argument("--tie-break", default="shorter", choices=["shorter", "longer"])
    parser.add_argument("--cap-base-le8", type=int, default=14)
    parser.add_argument("--cap-base-9-12", type=int, default=16)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    policies = [p.strip().lower() for p in args.policies.split(",") if p.strip()]
    weak_grid = parse_csv_ints(args.weak_grid)
    strong_grid = parse_csv_ints(args.strong_grid)

    all_items: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Any] = {
        "source_results": args.results,
        "num_rows": len(rows),
        "settings": {
            "policies": policies,
            "ratio_threshold": args.ratio_threshold,
            "long_score_floor": args.long_score_floor,
            "raw_ratio_threshold": args.raw_ratio_threshold,
            "support_count_threshold": args.support_count_threshold,
            "strong_min_len": args.strong_min_len,
            "weak_min_base_len": args.weak_min_base_len,
            "weak_max_base_len": args.weak_max_base_len,
            "weak_grid": weak_grid,
            "strong_grid": strong_grid,
            "long_alpha": args.long_alpha,
            "score_mode": args.score_mode,
            "tie_break": args.tie_break,
            "cap_base_le8": args.cap_base_le8,
            "cap_base_9_12": args.cap_base_9_12,
        },
        "policies": {},
    }

    for policy in policies:
        items = [
            simulate_policy(
                row=row,
                policy=policy,
                tie_break=args.tie_break,
                ratio_threshold=args.ratio_threshold,
                long_score_floor=args.long_score_floor,
                raw_ratio_threshold=args.raw_ratio_threshold,
                support_count_threshold=args.support_count_threshold,
                strong_min_len=args.strong_min_len,
                weak_min_base_len=args.weak_min_base_len,
                weak_max_base_len=args.weak_max_base_len,
                weak_grid=weak_grid,
                strong_grid=strong_grid,
                long_alpha=args.long_alpha,
                score_mode=args.score_mode,
                cap_base_le8=args.cap_base_le8,
                cap_base_9_12=args.cap_base_9_12,
            )
            for row in rows
        ]
        all_items[policy] = items
        summary["policies"][policy] = summarize_simulations(items)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "policy",
            "task_id",
            "oracle_length",
            "oracle_bucket",
            "base_len",
            "final_len",
            "final_minus_base",
            "selected_minus_oracle",
            "triggered",
            "strong_trigger",
            "weak_trigger",
            "trigger_reason",
            "correction_kind",
            "correction_len",
            "best_len",
            "best_score",
            "best_long_len",
            "best_long_score",
            "long_ratio",
            "raw_long_ratio",
            "long_score_floor_passed",
            "ratio_passed",
            "raw_ratio_passed",
            "support_count",
            "support_passed",
        ]
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for policy in policies:
                for item in all_items[policy]:
                    writer.writerow({key: item.get(key) for key in fieldnames})

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
