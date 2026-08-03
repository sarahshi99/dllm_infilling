#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


NONDETERMINISTIC_FIELDS = {"wall_time_seconds", "peak_cuda_memory_bytes"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def durable_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def deterministic_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): deterministic_projection(item)
            for key, item in value.items()
            if key not in NONDETERMINISTIC_FIELDS
        }
    if isinstance(value, list):
        return [deterministic_projection(item) for item in value]
    return value


def seed_resume(source_dir: Path, resume_dir: Path, rows: int) -> None:
    source = read_jsonl(source_dir / "predictions.jsonl")
    if not 0 < rows < len(source):
        raise RuntimeError("resume seed rows must be positive and smaller than source")
    destination = resume_dir / "predictions.jsonl"
    expected = source[:rows]
    if destination.exists():
        existing = read_jsonl(destination)
        if existing[:rows] != expected:
            raise RuntimeError("existing resume seed does not match source prefix")
        return
    durable_write_jsonl(destination, expected)


def compare(reference_dir: Path, candidate_dir: Path, output: Path, label: str) -> None:
    reference = read_jsonl(reference_dir / "predictions.jsonl")
    candidate = read_jsonl(candidate_dir / "predictions.jsonl")
    reference_ids = [row["task_id"] for row in reference]
    candidate_ids = [row["task_id"] for row in candidate]
    projected_reference = deterministic_projection(reference)
    projected_candidate = deterministic_projection(candidate)
    mismatches = [
        task_id
        for task_id, left, right in zip(
            reference_ids, projected_reference, projected_candidate
        )
        if left != right
    ]
    passed = (
        len(reference) == len(candidate)
        and reference_ids == candidate_ids
        and not mismatches
    )
    payload = {
        "label": label,
        "passed": passed,
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "reference_rows": len(reference),
        "candidate_rows": len(candidate),
        "task_order_matches": reference_ids == candidate_ids,
        "excluded_nondeterministic_fields": sorted(NONDETERMINISTIC_FIELDS),
        "mismatch_task_ids": mismatches,
    }
    write_json(output, payload)
    if not passed:
        raise RuntimeError(f"{label} failed: {mismatches}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed-resume")
    seed.add_argument("--source-dir", type=Path, required=True)
    seed.add_argument("--resume-dir", type=Path, required=True)
    seed.add_argument("--rows", type=int, required=True)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--reference-dir", type=Path, required=True)
    comparison.add_argument("--candidate-dir", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "seed-resume":
        seed_resume(args.source_dir, args.resume_dir, args.rows)
    else:
        compare(
            args.reference_dir,
            args.candidate_dir,
            args.output,
            args.label,
        )


if __name__ == "__main__":
    main()
