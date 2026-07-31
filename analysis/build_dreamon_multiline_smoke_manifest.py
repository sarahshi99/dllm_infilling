#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FORBIDDEN_FIELDS = {
    "canonical_solution",
    "completion",
    "evaluator_outcome",
    "evaluator_result",
    "oracle_length",
    "passed",
    "reference_code",
    "test",
    "tests",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_full(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 5079:
        raise RuntimeError("DreamOn MultiLine full manifest must contain 5079 rows")
    groups = {str(row.get("task_group") or "") for row in rows}
    if "" in groups or len(groups) != 148:
        raise RuntimeError("DreamOn MultiLine full manifest must contain 148 nonblank clusters")
    forbidden = sorted({field for row in rows for field in FORBIDDEN_FIELDS if field in row})
    if forbidden:
        raise RuntimeError(f"DreamOn MultiLine manifest contains forbidden fields: {forbidden}")
    task_ids = [str(row.get("task_id") or "") for row in rows]
    if any(not value for value in task_ids) or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("DreamOn MultiLine manifest task ids are blank or duplicated")


def select_smoke(rows: Sequence[Mapping[str, Any]], count: int = 12) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in rows:
        group = str(row["task_group"])
        if group in used:
            continue
        selected.append(dict(row))
        used.add(group)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"cannot select {count} distinct MultiLine clusters")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable file exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def build(full_manifest: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(full_manifest)
    validate_full(rows)
    smoke = select_smoke(rows)
    smoke_path = output_dir / "dreamon_multiline_technical_smoke12_manifest.jsonl"
    write_immutable(smoke_path, jsonl_bytes(smoke))
    summary = {
        "builder": "analysis/build_dreamon_multiline_smoke_manifest.py",
        "full_manifest": str(full_manifest.resolve()),
        "full_manifest_sha256": sha256(full_manifest),
        "full_rows": 5079,
        "full_clusters": 148,
        "smoke_manifest": str(smoke_path.resolve()),
        "smoke_manifest_sha256": sha256(smoke_path),
        "smoke_rows": 12,
        "smoke_clusters": 12,
        "selection": "first row from each of the first 12 distinct task_group values in immutable full-manifest order",
        "forbidden_fields_used": [],
        "sealed_files_opened": False,
        "test_evaluation_count": 0,
    }
    summary_path = output_dir / "dreamon_multiline_technical_smoke12_manifest_summary.json"
    write_immutable(
        summary_path,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build an immutable outcome-blind DreamOn MultiLine smoke manifest.")
    result.add_argument("--full-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(build(args.full_manifest, args.output_dir), ensure_ascii=False, indent=2, sort_keys=True))
