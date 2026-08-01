#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_ROWS = 642
EXPECTED_BASE_PROBLEMS = 115
ALLOWED_GENERATION_FIELDS = ("task_id", "base_problem_id", "prompt", "suffix")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_problem_id(task_id: str) -> str:
    match = re.search(r"HumanEval/(\d+)", task_id)
    return match.group(1) if match else task_id


def load_dataset_by_task(path: Path) -> dict[str, dict[str, str]]:
    by_task: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            task_id = raw["task_id"]
            by_task[task_id] = {
                "task_id": task_id,
                "base_problem_id": base_problem_id(task_id),
                "prompt": raw["prompt"],
                "suffix": raw["suffix"],
            }
    return by_task


def prepare(frozen_manifest: Path, dataset: Path, output: Path) -> dict[str, Any]:
    manifest = read_jsonl(frozen_manifest)
    task_ids = [row["task_id"] for row in manifest]
    if len(task_ids) != EXPECTED_ROWS or len(set(task_ids)) != EXPECTED_ROWS:
        raise RuntimeError("Frozen manifest must contain 642 unique task IDs")
    base_ids = {base_problem_id(task_id) for task_id in task_ids}
    if len(base_ids) != EXPECTED_BASE_PROBLEMS:
        raise RuntimeError(f"Expected 115 base problems, found {len(base_ids)}")

    dataset_by_task = load_dataset_by_task(dataset)
    missing = [task_id for task_id in task_ids if task_id not in dataset_by_task]
    if missing:
        raise RuntimeError(f"Dataset is missing {len(missing)} frozen tasks")
    rows = [dataset_by_task[task_id] for task_id in task_ids]
    if any(tuple(row) != ALLOWED_GENERATION_FIELDS for row in rows):
        raise AssertionError("Sanitized generation rows contain unexpected fields")
    write_jsonl(output, rows)
    reloaded = read_jsonl(output)
    if reloaded != rows:
        raise AssertionError("Generation population round-trip mismatch")
    return {
        "manifest_sha256": sha256(frozen_manifest),
        "generation_population_sha256": sha256(output),
        "rows": len(rows),
        "unique_task_ids": len({row["task_id"] for row in rows}),
        "base_problems": len({row["base_problem_id"] for row in rows}),
        "allowed_fields": list(ALLOWED_GENERATION_FIELDS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            prepare(args.frozen_manifest, args.dataset, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
