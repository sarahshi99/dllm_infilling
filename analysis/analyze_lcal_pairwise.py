#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
                raise ValueError(f"Failed to parse {path}:{line_idx}: {exc}") from exc
    return rows


def by_task(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in output:
            raise ValueError(f"Duplicate task_id in results: {task_id}")
        output[task_id] = row
    return output


def metric(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    return row.get("metrics", {}).get(key, default)


def passed(row: Dict[str, Any]) -> bool:
    return bool(metric(row, "passed", False))


def oracle_len(row: Dict[str, Any]) -> Optional[int]:
    value = metric(row, "oracle_mask_length")
    return None if value is None else int(value)


def bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    if length <= 8:
        return "<=8"
    if length <= 12:
        return "9-12"
    if length <= 16:
        return "13-16"
    if length <= 24:
        return "17-24"
    return "25+"


def selected_len(row: Dict[str, Any]) -> Optional[int]:
    value = metric(row, "selected_mask_length", metric(row, "mask_length"))
    return None if value is None else int(value)


def final_source(row: Dict[str, Any]) -> str:
    return str(metric(row, "final_source", "unknown"))


def rate(values: Iterable[bool]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(1 for value in vals if value) / len(vals)


def compare(base_rows: List[Dict[str, Any]], candidate_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = by_task(base_rows)
    candidate = by_task(candidate_rows)
    task_ids = sorted(set(base) & set(candidate))
    pairwise: List[Dict[str, Any]] = []

    for task_id in task_ids:
        base_row = base[task_id]
        candidate_row = candidate[task_id]
        base_passed = passed(base_row)
        candidate_passed = passed(candidate_row)
        if candidate_passed and not base_passed:
            outcome = "win"
        elif base_passed and not candidate_passed:
            outcome = "loss"
        elif base_passed and candidate_passed:
            outcome = "both_pass"
        else:
            outcome = "both_fail"

        length = oracle_len(candidate_row)
        if length is None:
            length = oracle_len(base_row)

        pairwise.append(
            {
                "task_id": task_id,
                "outcome": outcome,
                "oracle_len": length,
                "oracle_bucket": bucket(length),
                "base_passed": base_passed,
                "candidate_passed": candidate_passed,
                "base_selected_len": selected_len(base_row),
                "candidate_selected_len": selected_len(candidate_row),
                "base_final_source": final_source(base_row),
                "candidate_final_source": final_source(candidate_row),
            }
        )

    bucket_rows: List[Dict[str, Any]] = []
    for name in ["<=8", "9-12", "13-16", "17-24", "25+", "unknown"]:
        rows = [row for row in pairwise if row["oracle_bucket"] == name]
        if not rows:
            continue
        bucket_rows.append(
            {
                "oracle_bucket": name,
                "count": len(rows),
                "wins": sum(1 for row in rows if row["outcome"] == "win"),
                "losses": sum(1 for row in rows if row["outcome"] == "loss"),
                "net": (
                    sum(1 for row in rows if row["outcome"] == "win")
                    - sum(1 for row in rows if row["outcome"] == "loss")
                ),
                "base_pass_rate": rate(row["base_passed"] for row in rows),
                "candidate_pass_rate": rate(row["candidate_passed"] for row in rows),
            }
        )

    outcomes = Counter(row["outcome"] for row in pairwise)
    return {
        "summary": {
            "common": len(task_ids),
            "base_only": len(set(base) - set(candidate)),
            "candidate_only": len(set(candidate) - set(base)),
            "wins": outcomes["win"],
            "losses": outcomes["loss"],
            "net": outcomes["win"] - outcomes["loss"],
            "both_pass": outcomes["both_pass"],
            "both_fail": outcomes["both_fail"],
            "base_pass_rate": rate(row["base_passed"] for row in pairwise),
            "candidate_pass_rate": rate(row["candidate_passed"] for row in pairwise),
            "candidate_source_histogram": dict(Counter(row["candidate_final_source"] for row in pairwise)),
        },
        "buckets": bucket_rows,
        "pairwise": pairwise,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LCAL-style result JSONL files.")
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--candidate-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = compare(load_jsonl(args.base_results), load_jsonl(args.candidate_results))
    (output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "bucket_summary.json").write_text(
        json.dumps(payload["buckets"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "bucket_summary.csv", payload["buckets"])
    write_csv(output_dir / "pairwise.csv", payload["pairwise"])
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
