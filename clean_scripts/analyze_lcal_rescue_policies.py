#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_POLICIES = (
    "r0_s3_existing,"
    "r1_strong_only_shortest,"
    "r2_s3_shortest,"
    "r3_base_le8_extreme,"
    "r4_base_le8_support2,"
    "r5_official_len17_guarded,"
    "r6_official_len17_cap"
)
DEFAULT_STRONG_GRID = "13,14,15,16,20,24,28,32,40"
DEFAULT_WEAK_GRID = "13,14,15,16"


def parse_csv_ints(text: str) -> List[int]:
    values: List[int] = []
    seen = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"length values must be positive, got {value}")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("no length values parsed")
    return values


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    return row.get("metrics", {}).get(key, default)


def probe(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("length_probe", {})


def base_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = probe(row).get("base_candidate_scores") or probe(row).get("candidate_scores")
    if not candidates:
        raise KeyError(f"missing base candidates for task_id={row.get('task_id')}")
    return list(candidates)


def existing_long_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(probe(row).get("long_candidate_scores") or [])


def raw_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("raw_score", candidate.get("mean_top1_prob", candidate.get("score"))))


def adjusted_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("score", candidate.get("adjusted_score", raw_score(candidate))))


def power_score(candidate: Dict[str, Any], alpha: float, score_mode: str) -> float:
    if score_mode == "raw":
        return raw_score(candidate)
    if score_mode == "length_power":
        return raw_score(candidate) * (int(candidate["mask_length"]) ** float(alpha))
    raise ValueError(f"unsupported score_mode={score_mode}")


def pick_best(candidates: Iterable[Dict[str, Any]], score_fn, tie_break: str) -> Dict[str, Any]:
    items = list(candidates)
    if not items:
        raise ValueError("cannot pick best from empty candidates")
    if tie_break == "shorter":
        return max(items, key=lambda item: (score_fn(item), -int(item["mask_length"])))
    if tie_break == "longer":
        return max(items, key=lambda item: (score_fn(item), int(item["mask_length"])))
    raise ValueError(f"unsupported tie_break={tie_break}")


def by_length(candidates: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for candidate in candidates:
        length = int(candidate["mask_length"])
        current = out.get(length)
        if current is None or raw_score(candidate) > raw_score(current):
            out[length] = candidate
    return out


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
    vals = [float(value) for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def rate(values: Iterable[bool]) -> Optional[float]:
    vals = list(values)
    return sum(1 for value in vals if value) / len(vals) if vals else None


def hist(values: Iterable[Any]) -> Dict[str, int]:
    counter = Counter(str(value) for value in values)
    return dict(sorted(counter.items(), key=lambda kv: kv[0]))


def cap_for_base(base_len: int, cap_le8: int, cap_9_12: int, cap_13_16: int, cap_ge20: int) -> int:
    if base_len <= 8:
        return int(cap_le8)
    if base_len <= 12:
        return int(cap_9_12)
    if base_len <= 16:
        return int(cap_13_16)
    return int(cap_ge20)


def adjacent_support_run(lengths: List[int], supported: set[int]) -> int:
    best_run = 0
    current = 0
    for length in lengths:
        if length in supported:
            current += 1
            best_run = max(best_run, current)
        else:
            current = 0
    return best_run


def curve_info(row: Dict[str, Any], tie_break: str, support_ratio: float) -> Dict[str, Any]:
    candidates = base_candidates(row)
    long_candidates = [candidate for candidate in candidates if int(candidate["mask_length"]) >= 13]
    best = pick_best(candidates, adjusted_score, tie_break)
    best_long = pick_best(long_candidates, adjusted_score, tie_break)
    best_raw = pick_best(candidates, raw_score, tie_break)
    best_long_raw = pick_best(long_candidates, raw_score, tie_break)

    best_score = adjusted_score(best)
    best_long_score = adjusted_score(best_long)
    best_raw_score = raw_score(best_raw)
    best_long_raw_score = raw_score(best_long_raw)

    supported_lengths = {
        int(candidate["mask_length"])
        for candidate in long_candidates
        if adjusted_score(candidate) >= float(support_ratio) * best_score
    }
    grid_order = [13, 14, 15, 16, 20, 24]
    base_len = metric(row, "lcal_v3_base_selected_mask_length", int(best["mask_length"]))

    return {
        "base_len": int(base_len),
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
        "support_count": len(supported_lengths),
        "support_lengths": sorted(supported_lengths),
        "adjacent_support_run": adjacent_support_run(grid_order, supported_lengths),
    }


def available_candidates_for_lengths(row: Dict[str, Any], lengths: List[int]) -> List[Dict[str, Any]]:
    base_map = by_length(base_candidates(row))
    long_map = by_length(existing_long_candidates(row))
    out: List[Dict[str, Any]] = []
    for length in lengths:
        if length in long_map:
            out.append(long_map[length])
        elif length in base_map:
            out.append(base_map[length])
    return out


def select_correction_length(
    row: Dict[str, Any],
    base_len: int,
    grid: List[int],
    cap: int,
    long_alpha: float,
    score_mode: str,
    tie_break: str,
    selection_rule: str,
    shortest_supported_ratio: float,
) -> Tuple[Optional[int], Optional[str], Optional[float], int]:
    effective_lengths = [length for length in grid if length <= cap]
    candidates = available_candidates_for_lengths(row, effective_lengths)
    min_correction_len = max(13, int(base_len))
    candidates = [candidate for candidate in candidates if int(candidate["mask_length"]) >= min_correction_len]
    if not candidates:
        return None, None, None, 0

    def score_fn(candidate: Dict[str, Any]) -> float:
        return power_score(candidate, alpha=long_alpha, score_mode=score_mode)

    best = pick_best(candidates, score_fn, tie_break)
    best_score = score_fn(best)
    if selection_rule == "argmax":
        selected = best
        reason = "argmax"
    elif selection_rule == "shortest_supported":
        supported = [
            candidate
            for candidate in candidates
            if score_fn(candidate) >= float(shortest_supported_ratio) * best_score
        ]
        selected = min(supported, key=lambda candidate: int(candidate["mask_length"])) if supported else best
        reason = "shortest_supported"
    else:
        raise ValueError(f"unsupported selection_rule={selection_rule}")
    return int(selected["mask_length"]), reason, float(score_fn(selected)), len(candidates)


def official_selected_len(official_row: Optional[Dict[str, Any]]) -> Optional[int]:
    if not official_row:
        return None
    value = official_row.get("metrics", {}).get("selected_mask_length")
    if value is not None:
        return int(value)
    value = official_row.get("official_cal", {}).get("selected_mask_length")
    return None if value is None else int(value)


def official_search_steps(official_row: Optional[Dict[str, Any]]) -> Optional[int]:
    if not official_row:
        return None
    value = official_row.get("metrics", {}).get("official_cal_search_steps")
    if value is not None:
        return int(value)
    value = official_row.get("official_cal", {}).get("search_steps")
    return None if value is None else int(value)


def trigger_for_policy(
    policy: str,
    info: Dict[str, Any],
    off_len: Optional[int],
    args: argparse.Namespace,
) -> Tuple[bool, str, str, List[int], int]:
    base_len = int(info["base_len"])
    ratio = info["long_ratio"]
    raw_ratio = info["raw_long_ratio"]
    best_long_score = float(info["best_long_score"])
    support_count = int(info["support_count"])
    adjacent_run = int(info["adjacent_support_run"])
    cap = cap_for_base(base_len, args.cap_base_le8, args.cap_base_9_12, args.cap_base_13_16, args.cap_base_ge20)
    ratio_pass = (
        ratio is not None
        and ratio >= float(args.ratio_threshold)
        and best_long_score >= float(args.long_score_floor)
    )
    raw_pass = raw_ratio is not None and raw_ratio >= float(args.raw_ratio_threshold)
    weak_window = int(args.weak_min_base_len) <= base_len <= int(args.weak_max_base_len)

    if policy == "r1_strong_only_shortest":
        if base_len >= args.strong_min_len:
            return True, "strong_base_len", "strong_shortest", args.strong_grid_values, cap
        return False, "none", "none", [], cap

    if policy == "r2_s3_shortest":
        if base_len >= args.strong_min_len:
            return True, "strong_base_len", "strong_shortest", args.strong_grid_values, cap
        if weak_window and ratio_pass and raw_pass:
            return True, "weak_ratio_raw", "weak_shortest_capped", args.weak_grid_values, cap
        return False, "none", "none", [], cap

    if policy == "r3_base_le8_extreme":
        if base_len >= args.strong_min_len:
            return True, "strong_base_len", "strong_shortest", args.strong_grid_values, cap
        if weak_window and ratio_pass and raw_pass:
            return True, "weak_ratio_raw", "weak_shortest_capped", args.weak_grid_values, cap
        if base_len <= 8:
            extreme = (
                ratio is not None
                and ratio >= float(args.extreme_ratio_threshold)
                and best_long_score >= float(args.long_score_floor)
                and raw_ratio is not None
                and raw_ratio >= float(args.extreme_raw_ratio_threshold)
                and support_count >= int(args.extreme_support_count)
                and adjacent_run >= int(args.extreme_adjacent_run)
            )
            if extreme:
                return True, "short_extreme_ratio", "short_extreme_capped", args.weak_grid_values, cap
        return False, "none", "none", [], cap

    if policy == "r4_base_le8_support2":
        if base_len >= args.strong_min_len:
            return True, "strong_base_len", "strong_shortest", args.strong_grid_values, cap
        if weak_window and ratio_pass and raw_pass:
            return True, "weak_ratio_raw", "weak_shortest_capped", args.weak_grid_values, cap
        if base_len <= 8:
            support2 = (
                ratio is not None
                and ratio >= float(args.ratio_threshold)
                and best_long_score >= float(args.long_score_floor)
                and raw_ratio is not None
                and raw_ratio >= float(args.raw_ratio_threshold)
                and support_count >= 2
            )
            if support2:
                return True, "short_support2_ratio_raw", "short_support2_capped", args.weak_grid_values, cap
        return False, "none", "none", [], cap

    if policy == "r5_official_len17_guarded":
        if base_len >= args.strong_min_len:
            return True, "strong_base_len", "strong_shortest", args.strong_grid_values, cap
        if weak_window and ratio_pass and raw_pass:
            return True, "weak_ratio_raw", "weak_shortest_capped", args.weak_grid_values, cap
        official_guarded = (
            off_len is not None
            and off_len >= 17
            and base_len <= 12
            and best_long_score >= float(args.official_long_score_floor)
            and ratio is not None
            and ratio >= float(args.official_ratio_threshold)
            and (raw_ratio is None or raw_ratio >= float(args.official_raw_ratio_threshold))
        )
        if official_guarded:
            return True, "official_len17_guarded", "official_shortest_capped", args.weak_grid_values, cap
        return False, "none", "none", [], cap

    if policy == "r6_official_len17_cap":
        if base_len >= args.strong_min_len:
            return True, "strong_base_len", "strong_shortest", args.strong_grid_values, cap
        if weak_window and ratio_pass and raw_pass:
            return True, "weak_ratio_raw", "weak_shortest_capped", args.weak_grid_values, cap
        if off_len is not None and off_len >= 17 and base_len <= 12:
            return True, "official_len17", "official_shortest_capped", args.weak_grid_values, cap
        return False, "none", "none", [], cap

    raise ValueError(f"unsupported policy={policy}")


def existing_s3_item(row: Dict[str, Any], info: Dict[str, Any], off_len: Optional[int], off_steps: Optional[int]) -> Dict[str, Any]:
    oracle = metric(row, "oracle_mask_length")
    oracle = None if oracle is None else int(oracle)
    base_len = int(metric(row, "lcal_v3_base_selected_mask_length", info["base_len"]))
    final_len = int(metric(row, "selected_mask_length", metric(row, "final_selected_length", base_len)))
    triggered = bool(metric(row, "lcal_v3_long_triggered", final_len != base_len))
    trigger_reason = str(metric(row, "trigger_reason", "unknown" if triggered else "none"))
    weak = bool(metric(row, "lcal_v3_weak_triggered", trigger_reason.startswith("weak")))
    strong = bool(metric(row, "lcal_v3_strong_triggered", trigger_reason == "strong_base_len"))
    diff = None if oracle is None else final_len - oracle
    return {
        "policy": "r0_s3_existing",
        "task_id": row.get("task_id"),
        "oracle_length": oracle,
        "oracle_bucket": length_bucket(oracle),
        "base_len": base_len,
        "final_len": final_len,
        "final_minus_base": final_len - base_len,
        "selected_minus_oracle": diff,
        "triggered": triggered,
        "strong_trigger": strong,
        "weak_trigger": weak,
        "rescue_trigger": bool(triggered and not strong and not weak),
        "trigger_reason": trigger_reason,
        "correction_kind": str(metric(row, "correction_kind", "")),
        "correction_len": metric(row, "long_selected_length"),
        "correction_select_rule": "existing_s3",
        "correction_candidate_count": len(existing_long_candidates(row)),
        "cap": None,
        "best_len": info["best_len"],
        "best_score": info["best_score"],
        "best_long_len": info["best_long_len"],
        "best_long_score": info["best_long_score"],
        "long_ratio": info["long_ratio"],
        "raw_long_ratio": info["raw_long_ratio"],
        "support_count": info["support_count"],
        "support_lengths": ",".join(str(v) for v in info["support_lengths"]),
        "adjacent_support_run": info["adjacent_support_run"],
        "official_selected_len": off_len,
        "official_search_steps": off_steps,
        "uses_official_hint": False,
    }


def simulate_policy(
    row: Dict[str, Any],
    official_row: Optional[Dict[str, Any]],
    policy: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    info = curve_info(row, tie_break=args.tie_break, support_ratio=args.support_ratio)
    off_len = official_selected_len(official_row)
    off_steps = official_search_steps(official_row)
    if policy == "r0_s3_existing":
        return existing_s3_item(row, info, off_len, off_steps)

    oracle = metric(row, "oracle_mask_length")
    oracle = None if oracle is None else int(oracle)
    base_len = int(info["base_len"])
    triggered, trigger_reason, correction_kind, grid, cap = trigger_for_policy(policy, info, off_len, args)
    correction_len = None
    correction_select_rule = None
    correction_score = None
    correction_candidate_count = 0
    if triggered:
        correction_len, correction_select_rule, correction_score, correction_candidate_count = select_correction_length(
            row=row,
            base_len=base_len,
            grid=grid,
            cap=cap,
            long_alpha=args.long_alpha,
            score_mode=args.score_mode,
            tie_break=args.tie_break,
            selection_rule=args.selection_rule,
            shortest_supported_ratio=args.shortest_supported_ratio,
        )
        if correction_len is None:
            triggered = False
            trigger_reason = "none_no_available_candidate"
            correction_kind = "none"

    final_len = base_len if correction_len is None else max(base_len, int(correction_len))
    diff = None if oracle is None else final_len - oracle
    strong = bool(triggered and trigger_reason == "strong_base_len")
    weak = bool(triggered and trigger_reason.startswith("weak_"))
    uses_official_hint = bool(triggered and trigger_reason.startswith("official_"))

    return {
        "policy": policy,
        "task_id": row.get("task_id"),
        "oracle_length": oracle,
        "oracle_bucket": length_bucket(oracle),
        "base_len": base_len,
        "final_len": final_len,
        "final_minus_base": final_len - base_len,
        "selected_minus_oracle": diff,
        "triggered": bool(triggered),
        "strong_trigger": strong,
        "weak_trigger": weak,
        "rescue_trigger": bool(triggered and not strong and not weak),
        "trigger_reason": trigger_reason,
        "correction_kind": correction_kind,
        "correction_len": correction_len,
        "correction_select_rule": correction_select_rule,
        "correction_score": correction_score,
        "correction_candidate_count": correction_candidate_count,
        "cap": cap,
        "best_len": info["best_len"],
        "best_score": info["best_score"],
        "best_long_len": info["best_long_len"],
        "best_long_score": info["best_long_score"],
        "long_ratio": info["long_ratio"],
        "raw_long_ratio": info["raw_long_ratio"],
        "support_count": info["support_count"],
        "support_lengths": ",".join(str(v) for v in info["support_lengths"]),
        "adjacent_support_run": info["adjacent_support_run"],
        "official_selected_len": off_len,
        "official_search_steps": off_steps,
        "uses_official_hint": uses_official_hint,
    }


def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    true_long = [item for item in items if item["oracle_length"] is not None and item["oracle_length"] >= 17]
    short = [item for item in items if item["oracle_length"] is not None and item["oracle_length"] <= 8]
    medium = [item for item in items if item["oracle_length"] is not None and 9 <= item["oracle_length"] <= 16]
    weak = [item for item in items if item["weak_trigger"]]
    rescue = [item for item in items if item["rescue_trigger"]]

    def under(item: Dict[str, Any]) -> bool:
        return item["selected_minus_oracle"] is not None and item["selected_minus_oracle"] < 0

    def under_by3(item: Dict[str, Any]) -> bool:
        return item["selected_minus_oracle"] is not None and item["selected_minus_oracle"] <= -3

    def over(item: Dict[str, Any]) -> bool:
        return item["selected_minus_oracle"] is not None and item["selected_minus_oracle"] > 0

    def over_by3(item: Dict[str, Any]) -> bool:
        return item["selected_minus_oracle"] is not None and item["selected_minus_oracle"] >= 3

    return {
        "num_samples": n,
        "trigger_count": sum(1 for item in items if item["triggered"]),
        "trigger_rate": rate(item["triggered"] for item in items),
        "strong_trigger_count": sum(1 for item in items if item["strong_trigger"]),
        "strong_trigger_rate": rate(item["strong_trigger"] for item in items),
        "weak_trigger_count": len(weak),
        "weak_trigger_rate": rate(item["weak_trigger"] for item in items),
        "rescue_trigger_count": len(rescue),
        "rescue_trigger_rate": rate(item["rescue_trigger"] for item in items),
        "official_hint_trigger_count": sum(1 for item in items if item["uses_official_hint"]),
        "true_long_count": len(true_long),
        "true_long_trigger_count": sum(1 for item in true_long if item["triggered"]),
        "true_long_recall": rate(item["triggered"] for item in true_long),
        "true_long_final_ge13_count": sum(1 for item in true_long if item["final_len"] >= 13),
        "true_long_final_ge13_rate": rate(item["final_len"] >= 13 for item in true_long),
        "true_long_final_ge17_count": sum(1 for item in true_long if item["final_len"] >= 17),
        "true_long_final_ge17_rate": rate(item["final_len"] >= 17 for item in true_long),
        "true_long_under_select_rate": rate(under(item) for item in true_long),
        "true_long_under_by3_rate": rate(under_by3(item) for item in true_long),
        "short_count": len(short),
        "short_false_trigger_count": sum(1 for item in short if item["triggered"]),
        "short_false_trigger_rate": rate(item["triggered"] for item in short),
        "short_changed_count": sum(1 for item in short if item["final_minus_base"] != 0),
        "short_changed_rate": rate(item["final_minus_base"] != 0 for item in short),
        "short_large_jump_count": sum(1 for item in short if item["final_minus_base"] >= 4),
        "short_large_jump_rate": rate(item["final_minus_base"] >= 4 for item in short),
        "short_over_select_rate": rate(over(item) for item in short),
        "short_over_by3_rate": rate(over_by3(item) for item in short),
        "medium_count": len(medium),
        "medium_trigger_count": sum(1 for item in medium if item["triggered"]),
        "medium_trigger_rate": rate(item["triggered"] for item in medium),
        "weak_trigger_precision_true_long": (
            rate(item["oracle_length"] is not None and item["oracle_length"] >= 17 for item in weak)
            if weak
            else None
        ),
        "rescue_trigger_precision_true_long": (
            rate(item["oracle_length"] is not None and item["oracle_length"] >= 17 for item in rescue)
            if rescue
            else None
        ),
        "avg_final_minus_base_all": avg(item["final_minus_base"] for item in items),
        "avg_final_minus_base_short": avg(item["final_minus_base"] for item in short),
        "avg_final_minus_base_true_long": avg(item["final_minus_base"] for item in true_long),
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
        description="Offline diagnostics for short-safe long-rescue LCAL policies."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--official-results", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--strong-grid", default=DEFAULT_STRONG_GRID)
    parser.add_argument("--weak-grid", default=DEFAULT_WEAK_GRID)
    parser.add_argument("--selection-rule", default="shortest_supported", choices=["argmax", "shortest_supported"])
    parser.add_argument("--shortest-supported-ratio", type=float, default=0.985)
    parser.add_argument("--long-alpha", type=float, default=0.10)
    parser.add_argument("--score-mode", default="length_power", choices=["raw", "length_power"])
    parser.add_argument("--tie-break", default="shorter", choices=["shorter", "longer"])
    parser.add_argument("--support-ratio", type=float, default=0.97)
    parser.add_argument("--ratio-threshold", type=float, default=0.97)
    parser.add_argument("--raw-ratio-threshold", type=float, default=0.97)
    parser.add_argument("--long-score-floor", type=float, default=0.55)
    parser.add_argument("--extreme-ratio-threshold", type=float, default=0.995)
    parser.add_argument("--extreme-raw-ratio-threshold", type=float, default=0.97)
    parser.add_argument("--extreme-support-count", type=int, default=1)
    parser.add_argument("--extreme-adjacent-run", type=int, default=1)
    parser.add_argument("--official-ratio-threshold", type=float, default=0.90)
    parser.add_argument("--official-raw-ratio-threshold", type=float, default=0.85)
    parser.add_argument("--official-long-score-floor", type=float, default=0.55)
    parser.add_argument("--strong-min-len", type=int, default=13)
    parser.add_argument("--weak-min-base-len", type=int, default=8)
    parser.add_argument("--weak-max-base-len", type=int, default=12)
    parser.add_argument("--cap-base-le8", type=int, default=14)
    parser.add_argument("--cap-base-9-12", type=int, default=16)
    parser.add_argument("--cap-base-13-16", type=int, default=24)
    parser.add_argument("--cap-base-ge20", type=int, default=40)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    official_map: Dict[str, Dict[str, Any]] = {}
    if args.official_results:
        official_map = {row["task_id"]: row for row in load_jsonl(args.official_results)}

    args.strong_grid_values = parse_csv_ints(args.strong_grid)
    args.weak_grid_values = parse_csv_ints(args.weak_grid)
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]

    all_items: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Any] = {
        "source_results": args.results,
        "official_results": args.official_results,
        "num_rows": len(rows),
        "settings": {
            "policies": policies,
            "strong_grid": args.strong_grid_values,
            "weak_grid": args.weak_grid_values,
            "selection_rule": args.selection_rule,
            "shortest_supported_ratio": args.shortest_supported_ratio,
            "long_alpha": args.long_alpha,
            "score_mode": args.score_mode,
            "tie_break": args.tie_break,
            "support_ratio": args.support_ratio,
            "ratio_threshold": args.ratio_threshold,
            "raw_ratio_threshold": args.raw_ratio_threshold,
            "long_score_floor": args.long_score_floor,
            "extreme_ratio_threshold": args.extreme_ratio_threshold,
            "extreme_raw_ratio_threshold": args.extreme_raw_ratio_threshold,
            "extreme_support_count": args.extreme_support_count,
            "extreme_adjacent_run": args.extreme_adjacent_run,
            "official_ratio_threshold": args.official_ratio_threshold,
            "official_raw_ratio_threshold": args.official_raw_ratio_threshold,
            "official_long_score_floor": args.official_long_score_floor,
            "strong_min_len": args.strong_min_len,
            "weak_min_base_len": args.weak_min_base_len,
            "weak_max_base_len": args.weak_max_base_len,
            "cap_base_le8": args.cap_base_le8,
            "cap_base_9_12": args.cap_base_9_12,
            "cap_base_13_16": args.cap_base_13_16,
            "cap_base_ge20": args.cap_base_ge20,
            "available_length_note": (
                "Offline correction can only use saved base candidates plus saved long candidates. "
                "For previously untriggered rows, lengths 28/32/40 are usually unavailable until full decode."
            ),
            "official_hint_note": (
                "Policies r5/r6 use saved official CAL outputs only as diagnostics. "
                "A valid full experiment must compute the same official hint on the fly."
            ),
        },
        "policies": {},
    }

    for policy in policies:
        items = [
            simulate_policy(row, official_map.get(row["task_id"]), policy, args)
            for row in rows
        ]
        all_items[policy] = items
        summary["policies"][policy] = summarize(items)

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
            "rescue_trigger",
            "trigger_reason",
            "correction_kind",
            "correction_len",
            "correction_select_rule",
            "correction_score",
            "correction_candidate_count",
            "cap",
            "best_len",
            "best_score",
            "best_long_len",
            "best_long_score",
            "long_ratio",
            "raw_long_ratio",
            "support_count",
            "support_lengths",
            "adjacent_support_run",
            "official_selected_len",
            "official_search_steps",
            "uses_official_hint",
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
