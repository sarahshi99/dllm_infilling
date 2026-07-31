#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASETS = {
    "singleline": ("HumanEval-SingleLineInfilling", 1033, 927),
    "randomspan": ("HumanEval-RandomSpanInfilling", 1640, 1480),
    "multiline": ("HumanEval-MultiLineInfilling", 5815, 5079),
}
TASK_GROUP_PATTERN = re.compile(r"HumanEval/\d+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_group(task_id: str) -> str:
    match = TASK_GROUP_PATTERN.search(task_id)
    if not match:
        raise ValueError(f"cannot extract base-function group from {task_id!r}")
    return match.group(0)


def read_allowed_groups(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        group_index = header.index("task_group")
        frozen_index = header.index("frozen_controller_test_row")
        groups: set[str] = set()
        for row in reader:
            if str(row[frozen_index]).strip().lower() not in {"false", "0"}:
                raise RuntimeError("allowed-group certificate contains a frozen row")
            groups.add(str(row[group_index]))
    if len(groups) != 148:
        raise RuntimeError("allowed-group certificate must contain exactly 148 groups")
    return groups


def read_official_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for source_row_id, line in enumerate(handle):
            source = json.loads(line)
            safe = {
                "source_row_id": source_row_id,
                "task_id": str(source["task_id"]),
                "task_group": task_group(str(source["task_id"])),
                "prompt": str(source["prompt"]),
                "suffix": str(source["suffix"]),
                "entry_point": str(source["entry_point"]),
            }
            rows.append(safe)
    return rows


def row_fingerprint(row: Mapping[str, Any]) -> str:
    safe = {key: row[key] for key in ("task_id", "prompt", "suffix", "entry_point")}
    payload = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_rows(dataset_name: str, rows: Sequence[Mapping[str, Any]], allowed_groups: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row["task_group"]) not in allowed_groups:
            continue
        fingerprint = row_fingerprint(row)
        selected.append(
            {
                "candidate_key": f"{dataset_name}:source_row={int(row['source_row_id'])}:sha256={fingerprint[:16]}",
                "dataset": dataset_name,
                "source_row_id": int(row["source_row_id"]),
                "task_id": str(row["task_id"]),
                "task_group": str(row["task_group"]),
                "safe_row_sha256": fingerprint,
            }
        )
    return selected


def one_per_group(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
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
    raise RuntimeError(f"cannot select {count} distinct clusters")


def mechanism_rows(
    manifest: Sequence[Mapping[str, Any]],
    safe_rows_by_source: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    enriched = [
        {
            **dict(row),
            "context_character_count": len(str(safe_rows_by_source[int(row["source_row_id"])]["prompt"]))
            + len(str(safe_rows_by_source[int(row["source_row_id"])]["suffix"])),
        }
        for row in manifest
    ]
    enriched.sort(key=lambda row: (int(row["context_character_count"]), str(row["task_group"]), int(row["source_row_id"])))
    buckets: list[list[dict[str, Any]]] = [[], [], [], []]
    for index, row in enumerate(enriched):
        bucket = min(index * 4 // len(enriched), 3)
        buckets[bucket].append(row)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for bucket_index, bucket_rows in enumerate(buckets):
        bucket_count = 0
        for row in bucket_rows:
            group = str(row["task_group"])
            if group in used:
                continue
            selected.append({**row, "context_length_bucket": bucket_index})
            used.add(group)
            bucket_count += 1
            if bucket_count == 16:
                break
        if bucket_count != 16:
            raise RuntimeError(f"context bucket {bucket_index} cannot provide 16 distinct unused clusters")
    return selected


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable manifest already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def build(args: argparse.Namespace) -> dict[str, Any]:
    evaluator_data_root = Path(args.evaluator_data_root).resolve()
    allowed_path = Path(args.allowed_singleline_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    allowed_groups = read_allowed_groups(allowed_path)
    summary: dict[str, Any] = {
        "builder": "analysis/build_lrdllm_common_manifests.py",
        "allowed_group_certificate": str(allowed_path),
        "allowed_group_certificate_sha256": sha256(allowed_path),
        "allowed_groups": len(allowed_groups),
        "selection_fields": ["task_id", "task_group", "source_row_id", "prompt", "suffix", "entry_point"],
        "forbidden_fields_used": [],
        "sealed_files_opened": False,
        "test_evaluation_count": 0,
        "datasets": {},
    }
    for short_name, (dataset_name, official_count, allowed_count) in DATASETS.items():
        source_path = evaluator_data_root / f"{dataset_name}.jsonl.gz"
        safe_rows = read_official_rows(source_path)
        if len(safe_rows) != official_count:
            raise RuntimeError(f"{dataset_name} official row count mismatch")
        manifest = manifest_rows(dataset_name, safe_rows, allowed_groups)
        if len(manifest) != allowed_count or len({row["task_group"] for row in manifest}) != 148:
            raise RuntimeError(f"{dataset_name} non-frozen population mismatch")
        full_path = output_dir / f"lrdllm_common_{short_name}_nonfrozen_manifest.jsonl"
        write_immutable(full_path, jsonl_bytes(manifest))
        paths: dict[str, Any] = {
            "official_rows": official_count,
            "nonfrozen_rows": allowed_count,
            "nonfrozen_clusters": 148,
            "source_path": str(source_path),
            "source_sha256": sha256(source_path),
            "full_manifest": str(full_path),
            "full_manifest_sha256": sha256(full_path),
        }
        if short_name == "singleline":
            smoke = one_per_group(manifest, 12)
            smoke_path = output_dir / "lrdllm_common_singleline_technical_smoke12_manifest.jsonl"
            write_immutable(smoke_path, jsonl_bytes(smoke))
            mechanism = mechanism_rows(manifest, {int(row["source_row_id"]): row for row in safe_rows})
            mechanism_path = output_dir / "lrdllm_common_singleline_mechanism_smoke64_manifest.jsonl"
            write_immutable(mechanism_path, jsonl_bytes(mechanism))
            paths.update(
                {
                    "technical_smoke_rows": 12,
                    "technical_smoke_clusters": 12,
                    "technical_smoke_manifest": str(smoke_path),
                    "technical_smoke_manifest_sha256": sha256(smoke_path),
                    "mechanism_smoke_rows": 64,
                    "mechanism_smoke_clusters": 64,
                    "mechanism_smoke_context_buckets": dict(Counter(row["context_length_bucket"] for row in mechanism)),
                    "mechanism_smoke_manifest": str(mechanism_path),
                    "mechanism_smoke_manifest_sha256": sha256(mechanism_path),
                }
            )
        summary["datasets"][dataset_name] = paths
    summary_path = output_dir / "lrdllm_common_manifest_summary.json"
    write_immutable(summary_path, (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build immutable non-frozen LR-DLLM/common-protocol manifests without opening sealed files.")
    result.add_argument("--evaluator-data-root", required=True)
    result.add_argument("--allowed-singleline-manifest", required=True)
    result.add_argument("--output-dir", required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
