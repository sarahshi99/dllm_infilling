from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def base_task_id(sample_id: str) -> str:
    match = re.search(r"HumanEval/(\d+)", sample_id)
    if match is None:
        raise ValueError(f"Cannot parse base task from {sample_id!r}")
    return match.group(1)


def stable_hash(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def largest_remainder_allocation(
    group_sizes: Mapping[str, int], total: int, seed: str
) -> dict[str, int]:
    population = sum(group_sizes.values())
    if total < 0 or total > population:
        raise ValueError(f"total={total} is outside [0, {population}]")
    if population == 0:
        return {key: 0 for key in group_sizes}
    exact = {key: total * size / population for key, size in group_sizes.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    order = sorted(
        group_sizes,
        key=lambda key: (-(exact[key] - allocation[key]), stable_hash(seed, key)),
    )
    for key in order[:remaining]:
        allocation[key] += 1
    if sum(allocation.values()) != total:
        raise AssertionError("largest-remainder allocation did not reach requested total")
    if any(allocation[key] > group_sizes[key] for key in group_sizes):
        raise AssertionError("allocation exceeds a group population")
    return allocation


def proportional_sample(
    rows: Sequence[Mapping[str, Any]], total: int, seed: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[base_task_id(str(row["task_id"]))].append(row)
    allocation = largest_remainder_allocation(
        {key: len(value) for key, value in groups.items()}, total=total, seed=seed
    )
    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: int(item)):
        ordered = sorted(
            groups[key], key=lambda row: stable_hash(seed, str(row["task_id"]))
        )
        selected.extend(dict(row) for row in ordered[: allocation[key]])
    selected.sort(key=lambda row: (int(base_task_id(str(row["task_id"]))), str(row["task_id"])))
    if len(selected) != total or len({row["task_id"] for row in selected}) != total:
        raise AssertionError("proportional sample is incomplete or duplicated")
    return selected, allocation
