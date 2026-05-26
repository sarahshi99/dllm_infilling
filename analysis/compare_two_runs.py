#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {line_idx} in {path}: {exc}") from exc
    return rows


def normalize_row(row: Dict[str, Any], label: str) -> Dict[str, Any]:
    metrics = row.get("metrics", {})
    diagnostics = row.get("diagnostics", {})
    verification = row.get("verification", {})
    tier3 = verification.get("tier3_unit_tests", {})

    return {
        "task_id": row["task_id"],
        f"{label}_passed": bool(metrics.get("passed", False)),
        f"{label}_decode_sec": metrics.get("decode_sec"),
        f"{label}_verification_sec": metrics.get("verification_sec"),
        f"{label}_total_sec": metrics.get("total_sec"),
        f"{label}_mask_length": metrics.get("mask_length"),
        f"{label}_oracle_mask_length": metrics.get("oracle_mask_length"),
        f"{label}_mean_final_confidence": metrics.get("mean_final_confidence"),
        f"{label}_decoded_middle_text": diagnostics.get("decoded_middle_text"),
        f"{label}_reference_middle_text": diagnostics.get("reference_middle_text"),
        f"{label}_tier3_error": tier3.get("error_message"),
        f"{label}_code": row.get("code"),
    }


def build_index(rows: List[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        task_id = row["task_id"]
        if task_id in index:
            raise ValueError(f"Duplicate task_id detected in {label}: {task_id}")
        index[task_id] = normalize_row(row, label)
    return index


def safe_mean(values: Iterable[float | int | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def length_bin(length_value: int | None) -> str:
    if length_value is None:
        return "unknown"
    if length_value <= 8:
        return "<=8"
    if length_value <= 12:
        return "9-12"
    if length_value <= 16:
        return "13-16"
    return ">=17"


def compare_runs(
    fixed_rows: List[Dict[str, Any]],
    oracle_rows: List[Dict[str, Any]],
    fixed_label: str = "fixed",
    oracle_label: str = "oracle",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    fixed_index = build_index(fixed_rows, fixed_label)
    oracle_index = build_index(oracle_rows, oracle_label)

    fixed_ids = set(fixed_index.keys())
    oracle_ids = set(oracle_index.keys())

    common_ids = sorted(fixed_ids & oracle_ids)
    only_fixed = sorted(fixed_ids - oracle_ids)
    only_oracle = sorted(oracle_ids - fixed_ids)

    pairwise_rows: List[Dict[str, Any]] = []
    bin_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for task_id in common_ids:
        f = fixed_index[task_id]
        o = oracle_index[task_id]

        fixed_passed = bool(f[f"{fixed_label}_passed"])
        oracle_passed = bool(o[f"{oracle_label}_passed"])

        if (not fixed_passed) and oracle_passed:
            outcome = "win"
        elif fixed_passed and (not oracle_passed):
            outcome = "loss"
        elif fixed_passed and oracle_passed:
            outcome = "both_pass"
        else:
            outcome = "both_fail"

        oracle_len = o.get(f"{oracle_label}_oracle_mask_length")
        if oracle_len is None:
            oracle_len = f.get(f"{fixed_label}_oracle_mask_length")

        row = {
            "task_id": task_id,
            "outcome": outcome,
            "oracle_length": oracle_len,
            "length_bin": length_bin(oracle_len),
            "fixed_passed": fixed_passed,
            "oracle_passed": oracle_passed,
            "fixed_mask_length": f.get(f"{fixed_label}_mask_length"),
            "oracle_mask_length": o.get(f"{oracle_label}_mask_length"),
            "fixed_total_sec": f.get(f"{fixed_label}_total_sec"),
            "oracle_total_sec": o.get(f"{oracle_label}_total_sec"),
            "fixed_decode_sec": f.get(f"{fixed_label}_decode_sec"),
            "oracle_decode_sec": o.get(f"{oracle_label}_decode_sec"),
            "fixed_mean_final_confidence": f.get(f"{fixed_label}_mean_final_confidence"),
            "oracle_mean_final_confidence": o.get(f"{oracle_label}_mean_final_confidence"),
            "reference_middle_text": (
                o.get(f"{oracle_label}_reference_middle_text")
                or f.get(f"{fixed_label}_reference_middle_text")
            ),
            "fixed_decoded_middle_text": f.get(f"{fixed_label}_decoded_middle_text"),
            "oracle_decoded_middle_text": o.get(f"{oracle_label}_decoded_middle_text"),
            "fixed_tier3_error": f.get(f"{fixed_label}_tier3_error"),
            "oracle_tier3_error": o.get(f"{oracle_label}_tier3_error"),
        }
        pairwise_rows.append(row)
        bin_buckets[row["length_bin"]].append(row)

    outcome_counter = Counter(row["outcome"] for row in pairwise_rows)

    overall_summary: Dict[str, Any] = {
        "num_common_samples": len(common_ids),
        "num_only_fixed": len(only_fixed),
        "num_only_oracle": len(only_oracle),
        "fixed_pass_rate": safe_mean(int(row["fixed_passed"]) for row in pairwise_rows),
        "oracle_pass_rate": safe_mean(int(row["oracle_passed"]) for row in pairwise_rows),
        "absolute_pass_rate_gain": (
            safe_mean(int(row["oracle_passed"]) for row in pairwise_rows)
            - safe_mean(int(row["fixed_passed"]) for row in pairwise_rows)
        ),
        "fixed_avg_total_sec": safe_mean(row["fixed_total_sec"] for row in pairwise_rows),
        "oracle_avg_total_sec": safe_mean(row["oracle_total_sec"] for row in pairwise_rows),
        "wins": outcome_counter["win"],
        "losses": outcome_counter["loss"],
        "both_pass": outcome_counter["both_pass"],
        "both_fail": outcome_counter["both_fail"],
        "ties": outcome_counter["both_pass"] + outcome_counter["both_fail"],
        "only_fixed_task_ids": only_fixed,
        "only_oracle_task_ids": only_oracle,
    }

    fixed_avg = overall_summary["fixed_avg_total_sec"]
    oracle_avg = overall_summary["oracle_avg_total_sec"]
    if fixed_avg is not None and oracle_avg is not None and fixed_avg > 0:
        overall_summary["relative_total_sec_reduction"] = 1.0 - (oracle_avg / fixed_avg)
        overall_summary["speedup_ratio"] = fixed_avg / oracle_avg
    else:
        overall_summary["relative_total_sec_reduction"] = None
        overall_summary["speedup_ratio"] = None

    length_bin_summary: List[Dict[str, Any]] = []
    for bin_name in ["<=8", "9-12", "13-16", ">=17", "unknown"]:
        rows = bin_buckets.get(bin_name, [])
        if not rows:
            continue
        length_bin_summary.append(
            {
                "length_bin": bin_name,
                "num_samples": len(rows),
                "wins": sum(1 for row in rows if row["outcome"] == "win"),
                "losses": sum(1 for row in rows if row["outcome"] == "loss"),
                "both_pass": sum(1 for row in rows if row["outcome"] == "both_pass"),
                "both_fail": sum(1 for row in rows if row["outcome"] == "both_fail"),
                "fixed_pass_rate": safe_mean(int(row["fixed_passed"]) for row in rows),
                "oracle_pass_rate": safe_mean(int(row["oracle_passed"]) for row in rows),
                "absolute_pass_rate_gain": (
                    safe_mean(int(row["oracle_passed"]) for row in rows)
                    - safe_mean(int(row["fixed_passed"]) for row in rows)
                ),
                "fixed_avg_total_sec": safe_mean(row["fixed_total_sec"] for row in rows),
                "oracle_avg_total_sec": safe_mean(row["oracle_total_sec"] for row in rows),
            }
        )

    return pairwise_rows, overall_summary, length_bin_summary


def write_json(path: str, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two run result JSONL files.")
    parser.add_argument("--fixed", type=str, required=True, help="Path to fixed run results.jsonl")
    parser.add_argument("--oracle", type=str, required=True, help="Path to oracle run results.jsonl")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save comparison outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fixed_rows = load_jsonl(args.fixed)
    oracle_rows = load_jsonl(args.oracle)

    pairwise_rows, overall_summary, length_bin_summary = compare_runs(
        fixed_rows=fixed_rows,
        oracle_rows=oracle_rows,
        fixed_label="fixed",
        oracle_label="oracle",
    )

    write_json(os.path.join(args.out_dir, "compare_summary.json"), overall_summary)
    write_json(os.path.join(args.out_dir, "length_bin_summary.json"), length_bin_summary)
    write_jsonl(os.path.join(args.out_dir, "pairwise_outcomes.jsonl"), pairwise_rows)
    write_jsonl(
        os.path.join(args.out_dir, "wins.jsonl"),
        [row for row in pairwise_rows if row["outcome"] == "win"],
    )
    write_jsonl(
        os.path.join(args.out_dir, "losses.jsonl"),
        [row for row in pairwise_rows if row["outcome"] == "loss"],
    )

    print("=" * 80)
    print("Comparison finished")
    for key, value in overall_summary.items():
        if key in {"only_fixed_task_ids", "only_oracle_task_ids"}:
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    print("-" * 80)
    print("Saved files:")
    print(os.path.join(args.out_dir, "compare_summary.json"))
    print(os.path.join(args.out_dir, "length_bin_summary.json"))
    print(os.path.join(args.out_dir, "pairwise_outcomes.jsonl"))
    print(os.path.join(args.out_dir, "wins.jsonl"))
    print(os.path.join(args.out_dir, "losses.jsonl"))
    print("=" * 80)


if __name__ == "__main__":
    main()