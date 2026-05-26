#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"results.jsonl not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    return rows


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def avg(values: Iterable[Any]) -> Optional[float]:
    filtered = [safe_float(v) for v in values]
    filtered = [v for v in filtered if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def rate(values: Iterable[bool]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def histogram(values: Iterable[Any]) -> Dict[str, int]:
    counter: Counter[int] = Counter()
    for value in values:
        if value is None:
            continue
        try:
            counter[int(value)] += 1
        except (TypeError, ValueError):
            continue
    return {str(key): counter[key] for key in sorted(counter)}


def pass_bool(row: Dict[str, Any]) -> bool:
    return bool(row.get("metrics", {}).get("passed", False))


def metric(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    return row.get("metrics", {}).get(key, default)


def verification_pass(row: Dict[str, Any], tier: str) -> bool:
    return bool(row.get("verification", {}).get(tier, {}).get("passed", False))


def verification_error(row: Dict[str, Any]) -> str:
    verification = row.get("verification", {})

    if not verification.get("tier1_parse_compile", {}).get("passed", False):
        tier = verification.get("tier1_parse_compile", {})
        return f"tier1:{tier.get('error_type')}:{tier.get('error_message')}"

    if "tier2_smoke_exec" in verification and not verification.get("tier2_smoke_exec", {}).get("passed", False):
        tier = verification.get("tier2_smoke_exec", {})
        return f"tier2:{tier.get('error_type')}:{tier.get('error_message')}"

    if "tier3_unit_tests" in verification and not verification.get("tier3_unit_tests", {}).get("passed", False):
        tier = verification.get("tier3_unit_tests", {})
        return f"tier3:{tier.get('error_type')}:{tier.get('error_message')}"

    return ""


def oracle_bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    if length <= 4:
        return "01_<=4"
    if 5 <= length <= 8:
        return "02_5-8"
    if 9 <= length <= 12:
        return "03_9-12"
    if 13 <= length <= 16:
        return "04_13-16"
    if 17 <= length <= 24:
        return "05_17-24"
    return "06_25+"


def selected_bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    if length <= 4:
        return "01_<=4"
    if 5 <= length <= 8:
        return "02_5-8"
    if 9 <= length <= 12:
        return "03_9-12"
    if 13 <= length <= 16:
        return "04_13-16"
    if 17 <= length <= 24:
        return "05_17-24"
    return "06_25+"


def get_probe_lengths(row: Dict[str, Any]) -> List[int]:
    probe = row.get("length_probe", {})
    values = probe.get("probe_lengths")
    if isinstance(values, list):
        output: List[int] = []
        for value in values:
            try:
                output.append(int(value))
            except (TypeError, ValueError):
                pass
        return output

    candidates = probe.get("candidate_scores")
    if isinstance(candidates, list):
        output = []
        for candidate in candidates:
            try:
                output.append(int(candidate["mask_length"]))
            except (KeyError, TypeError, ValueError):
                pass
        return output

    return []


def nearest_probe_length(oracle_length: Optional[int], probe_lengths: List[int]) -> Optional[int]:
    if oracle_length is None or not probe_lengths:
        return None
    return min(probe_lengths, key=lambda value: (abs(value - oracle_length), value))


def length_status(diff: Optional[int]) -> str:
    if diff is None:
        return "unknown"
    if diff <= -3:
        return "under_by_3plus"
    if diff in {-2, -1}:
        return "under_by_1_2"
    if diff == 0:
        return "exact"
    if diff in {1, 2}:
        return "over_by_1_2"
    return "over_by_3plus"


def candidate_score_for_length(row: Dict[str, Any], length: Optional[int]) -> Optional[float]:
    if length is None:
        return None
    candidates = row.get("length_probe", {}).get("candidate_scores")
    if not isinstance(candidates, list):
        return None

    for candidate in candidates:
        try:
            if int(candidate.get("mask_length")) == int(length):
                return safe_float(candidate.get("score"))
        except (TypeError, ValueError):
            continue
    return None


def raw_and_adjusted_score_for_selected(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    selected = metric(row, "selected_mask_length")
    candidates = row.get("length_probe", {}).get("candidate_scores")
    if selected is None or not isinstance(candidates, list):
        return None, None

    for candidate in candidates:
        try:
            if int(candidate.get("mask_length")) == int(selected):
                raw_score = candidate.get("raw_score", candidate.get("mean_top1_prob", candidate.get("score")))
                adjusted_score = candidate.get("adjusted_score", candidate.get("score"))
                return safe_float(raw_score), safe_float(adjusted_score)
        except (TypeError, ValueError):
            continue

    return None, None


def make_row_map(rows: List[Dict[str, Any]], name: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not task_id:
            raise ValueError(f"Missing task_id in {name}")
        if task_id in result:
            raise ValueError(f"Duplicate task_id in {name}: {task_id}")
        result[task_id] = row
    return result


def pairwise_summary(
    main_map: Dict[str, Dict[str, Any]],
    other_map: Dict[str, Dict[str, Any]],
    other_name: str,
) -> Dict[str, Any]:
    common_ids = sorted(set(main_map) & set(other_map))
    if not common_ids:
        raise ValueError(f"No overlapping task_id with comparison run: {other_name}")

    win_ids: List[str] = []
    loss_ids: List[str] = []
    tie_pass_ids: List[str] = []
    tie_fail_ids: List[str] = []

    for task_id in common_ids:
        main_pass = pass_bool(main_map[task_id])
        other_pass = pass_bool(other_map[task_id])
        if main_pass and not other_pass:
            win_ids.append(task_id)
        elif not main_pass and other_pass:
            loss_ids.append(task_id)
        elif main_pass and other_pass:
            tie_pass_ids.append(task_id)
        else:
            tie_fail_ids.append(task_id)

    def subset_stats(task_ids: List[str]) -> Dict[str, Any]:
        rows = [main_map[task_id] for task_id in task_ids]
        return {
            "n": len(rows),
            "avg_oracle_length": avg(metric(row, "oracle_mask_length") for row in rows),
            "avg_selected_length": avg(metric(row, "selected_mask_length") for row in rows),
            "avg_selected_minus_oracle": avg(metric(row, "selected_minus_oracle_length") for row in rows),
            "oracle_bucket_histogram": histogram(metric(row, "oracle_mask_length") for row in rows),
            "selected_bucket_histogram": histogram(metric(row, "selected_mask_length") for row in rows),
        }

    return {
        "comparison_name": other_name,
        "common_samples": len(common_ids),
        "win": len(win_ids),
        "loss": len(loss_ids),
        "tie_pass": len(tie_pass_ids),
        "tie_fail": len(tie_fail_ids),
        "win_stats": subset_stats(win_ids),
        "loss_stats": subset_stats(loss_ids),
        "tie_pass_stats": subset_stats(tie_pass_ids),
        "tie_fail_stats": subset_stats(tie_fail_ids),
    }


def summarize_main(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [row.get("metrics", {}) for row in rows]
    diffs = [m.get("selected_minus_oracle_length") for m in metrics if m.get("selected_minus_oracle_length") is not None]
    abs_diffs = [m.get("abs_selected_minus_oracle_length") for m in metrics if m.get("abs_selected_minus_oracle_length") is not None]

    return {
        "num_samples": len(rows),
        "pass_count": sum(1 for row in rows if pass_bool(row)),
        "pass_rate": rate(pass_bool(row) for row in rows),
        "avg_decode_sec": avg(m.get("decode_sec") for m in metrics),
        "avg_verification_sec": avg(m.get("verification_sec") for m in metrics),
        "avg_total_sec": avg(m.get("total_sec") for m in metrics),
        "avg_length_probe_sec": avg(m.get("length_probe_sec") for m in metrics),
        "avg_total_sec_including_probe": avg(m.get("total_sec_including_probe") for m in metrics),
        "avg_mask_length": avg(m.get("mask_length") for m in metrics),
        "avg_selected_mask_length": avg(m.get("selected_mask_length") for m in metrics),
        "avg_oracle_mask_length": avg(m.get("oracle_mask_length") for m in metrics),
        "avg_selected_score": avg(m.get("selected_score") for m in metrics),
        "avg_selected_minus_oracle_length": avg(diffs),
        "avg_abs_selected_minus_oracle_length": avg(abs_diffs),
        "exact_length_match_rate": rate(d == 0 for d in diffs),
        "within_1_length_rate": rate(abs(d) <= 1 for d in diffs),
        "within_2_length_rate": rate(abs(d) <= 2 for d in diffs),
        "under_select_rate": rate(d < 0 for d in diffs),
        "over_select_rate": rate(d > 0 for d in diffs),
        "selected_length_histogram": histogram(m.get("selected_mask_length") for m in metrics),
        "oracle_length_histogram": histogram(m.get("oracle_mask_length") for m in metrics),
        "selected_minus_oracle_histogram": histogram(diffs),
    }


def make_bucket_summary(rows: List[Dict[str, Any]], comparison_maps: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[oracle_bucket(metric(row, "oracle_mask_length"))].append(row)

    output: List[Dict[str, Any]] = []
    for bucket_name in sorted(buckets):
        bucket_rows = buckets[bucket_name]
        task_ids = [row["task_id"] for row in bucket_rows]
        record: Dict[str, Any] = {
            "oracle_bucket": bucket_name,
            "n": len(bucket_rows),
            "main_pass_rate": rate(pass_bool(row) for row in bucket_rows),
            "main_pass_count": sum(1 for row in bucket_rows if pass_bool(row)),
            "avg_oracle_length": avg(metric(row, "oracle_mask_length") for row in bucket_rows),
            "avg_selected_length": avg(metric(row, "selected_mask_length") for row in bucket_rows),
            "avg_selected_minus_oracle": avg(metric(row, "selected_minus_oracle_length") for row in bucket_rows),
            "avg_abs_selected_minus_oracle": avg(metric(row, "abs_selected_minus_oracle_length") for row in bucket_rows),
            "under_select_rate": rate((metric(row, "selected_minus_oracle_length") or 0) < 0 for row in bucket_rows),
            "over_select_rate": rate((metric(row, "selected_minus_oracle_length") or 0) > 0 for row in bucket_rows),
            "exact_length_match_rate": rate(metric(row, "selected_minus_oracle_length") == 0 for row in bucket_rows),
            "avg_total_sec_including_probe": avg(metric(row, "total_sec_including_probe") for row in bucket_rows),
        }

        for name, cmp_map in comparison_maps.items():
            matched = [cmp_map[task_id] for task_id in task_ids if task_id in cmp_map]
            record[f"{name}_pass_rate"] = rate(pass_bool(row) for row in matched)
            record[f"{name}_pass_count"] = sum(1 for row in matched if pass_bool(row))

        output.append(record)

    return output


def make_failure_rows(
    rows: List[Dict[str, Any]],
    comparison_maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    failure_rows: List[Dict[str, Any]] = []

    for row in rows:
        if pass_bool(row):
            continue

        task_id = row["task_id"]
        oracle_len = metric(row, "oracle_mask_length")
        selected_len = metric(row, "selected_mask_length")
        diff = metric(row, "selected_minus_oracle_length")
        probe_lengths = get_probe_lengths(row)
        nearest_probe = nearest_probe_length(oracle_len, probe_lengths)
        oracle_available_in_candidates = (
            oracle_len is not None and probe_lengths and int(oracle_len) in set(probe_lengths)
        )

        raw_selected_score, adjusted_selected_score = raw_and_adjusted_score_for_selected(row)

        record: Dict[str, Any] = {
            "task_id": task_id,
            "main_passed": False,
            "oracle_length": oracle_len,
            "selected_length": selected_len,
            "selected_minus_oracle": diff,
            "abs_selected_minus_oracle": metric(row, "abs_selected_minus_oracle_length"),
            "oracle_bucket": oracle_bucket(oracle_len),
            "selected_bucket": selected_bucket(selected_len),
            "length_status": length_status(diff),
            "oracle_length_in_candidates": oracle_available_in_candidates,
            "nearest_probe_to_oracle": nearest_probe,
            "nearest_probe_minus_oracle": None if nearest_probe is None or oracle_len is None else nearest_probe - int(oracle_len),
            "selected_score": metric(row, "selected_score"),
            "selected_raw_score": raw_selected_score,
            "selected_adjusted_score": adjusted_selected_score,
            "oracle_length_candidate_score": candidate_score_for_length(row, oracle_len),
            "length_probe_sec": metric(row, "length_probe_sec"),
            "total_sec": metric(row, "total_sec"),
            "total_sec_including_probe": metric(row, "total_sec_including_probe"),
            "tier1_passed": verification_pass(row, "tier1_parse_compile"),
            "tier2_passed": verification_pass(row, "tier2_smoke_exec"),
            "tier3_passed": verification_pass(row, "tier3_unit_tests"),
            "verification_error": verification_error(row),
        }

        for name, cmp_map in comparison_maps.items():
            cmp_row = cmp_map.get(task_id)
            record[f"{name}_passed"] = None if cmp_row is None else pass_bool(cmp_row)
            record[f"{name}_selected_length"] = None if cmp_row is None else metric(cmp_row, "selected_mask_length", metric(cmp_row, "mask_length"))
            record[f"{name}_mask_length"] = None if cmp_row is None else metric(cmp_row, "mask_length")

        oracle_row = comparison_maps.get("oracle", {}).get(task_id)
        if oracle_row is not None:
            record["oracle_rescues_main_failure"] = bool(pass_bool(oracle_row))
        else:
            record["oracle_rescues_main_failure"] = None

        failure_rows.append(record)

    failure_rows.sort(
        key=lambda item: (
            str(item["oracle_bucket"]),
            item["length_status"],
            -abs(int(item["selected_minus_oracle"] or 0)),
            item["task_id"],
        )
    )
    return failure_rows


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_compare_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--compare must be in NAME=PATH format, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f"Empty comparison name in: {value}")
    if not path:
        raise ValueError(f"Empty comparison path in: {value}")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline failure analysis for CAL-lite / length-adaptive infilling runs."
    )
    parser.add_argument("--main-results", required=True, help="Path to the main results.jsonl, e.g. alpha=0.06.")
    parser.add_argument("--main-name", default="main", help="Name used in output summaries.")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Optional comparison run in NAME=PATH format. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    main_rows = load_jsonl(args.main_results)
    main_map = make_row_map(main_rows, args.main_name)

    comparison_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    comparison_paths: Dict[str, str] = {}

    for item in args.compare:
        name, path = parse_compare_arg(item)
        rows = load_jsonl(path)
        comparison_maps[name] = make_row_map(rows, name)
        comparison_paths[name] = path

    pairwise = {
        name: pairwise_summary(main_map, cmp_map, name)
        for name, cmp_map in comparison_maps.items()
    }

    bucket_rows = make_bucket_summary(main_rows, comparison_maps)
    failure_rows = make_failure_rows(main_rows, comparison_maps)

    failure_status_counter = Counter(row["length_status"] for row in failure_rows)
    failure_bucket_counter = Counter(row["oracle_bucket"] for row in failure_rows)
    failure_oracle_rescue_counter = Counter(str(row.get("oracle_rescues_main_failure")) for row in failure_rows)

    summary = {
        "main_name": args.main_name,
        "main_results": str(args.main_results),
        "comparison_paths": comparison_paths,
        "main_summary": summarize_main(main_rows),
        "pairwise": pairwise,
        "failure_summary": {
            "num_failures": len(failure_rows),
            "failure_length_status_histogram": dict(sorted(failure_status_counter.items())),
            "failure_oracle_bucket_histogram": dict(sorted(failure_bucket_counter.items())),
            "failure_oracle_rescue_histogram": dict(sorted(failure_oracle_rescue_counter.items())),
            "avg_failure_oracle_length": avg(row["oracle_length"] for row in failure_rows),
            "avg_failure_selected_length": avg(row["selected_length"] for row in failure_rows),
            "avg_failure_selected_minus_oracle": avg(row["selected_minus_oracle"] for row in failure_rows),
            "avg_failure_abs_selected_minus_oracle": avg(row["abs_selected_minus_oracle"] for row in failure_rows),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "analysis_summary.json", summary)
    write_csv(output_dir / "bucket_summary.csv", bucket_rows)
    write_csv(output_dir / "failure_cases.csv", failure_rows)

    print("=" * 80)
    print("Length failure analysis finished")
    print(f"main_name: {args.main_name}")
    print(f"main_results: {args.main_results}")
    print(f"output_dir: {output_dir}")
    print("-" * 80)
    print(json.dumps(summary["main_summary"], ensure_ascii=False, indent=2))
    print("-" * 80)
    print("Pairwise comparisons:")
    for name, item in pairwise.items():
        print(
            f"{name}: "
            f"win={item['win']} "
            f"loss={item['loss']} "
            f"tie_pass={item['tie_pass']} "
            f"tie_fail={item['tie_fail']}"
        )
    print("-" * 80)
    print("Failure summary:")
    print(json.dumps(summary["failure_summary"], ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()