#!/usr/bin/env python3
"""Prepare a pinned, smoke-only ExecRepoBench evaluator manifest.

This tool deliberately never runs a model or produces benchmark scores.  It
checks the pinned dataset/evaluator checkouts, audits the public record schema,
and writes a small repository-grouped plan with one case per observed fill type.
The plan contains identifiers and hashes only; benchmark source code stays in
the immutable dataset checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_REPOSITORY = "https://huggingface.co/datasets/CSJianYang/ExecRepoBench"
DATASET_COMMIT = "fa61028ce495c9ceff58398b8a7c47b5ae9f5276"
EVALUATOR_REPOSITORY = "https://github.com/QwenLM/Qwen3-Coder.git"
EVALUATOR_COMMIT = "33bc6aabd7791ad7b32f7e92104f11f2359ba890"
EVALUATOR_ENTRYPOINT = "qwencoder-eval/base/benchmarks/ExecRepoBench"
DATA_FILE_NAME = "exec_repo_bench.jsonl"
REQUIRED_FIELDS = (
    "repo_name",
    "file_name",
    "prefix_code",
    "suffix_code",
    "middle_code",
    "context_code",
    "fill_type",
)


class AuditError(ValueError):
    """Raised when a supposedly pinned, smoke-only setup is invalid."""


def _canonical_sha256(record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise AuditError(f"{path} is not a checked-out git revision: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _verify_checkout(path: Path, *, expected_commit: str, label: str) -> dict[str, str]:
    observed = _git_head(path)
    if observed != expected_commit:
        raise AuditError(f"{label} commit mismatch: expected {expected_commit}, got {observed}")
    return {"path": str(path), "commit": observed}


def _license_audit(root: Path, relative_path: str) -> dict[str, str]:
    path = root / relative_path
    if not path.is_file():
        raise AuditError(f"required license audit file is missing: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?im)^\s*license\s*:\s*([^\r\n]+)", content)
    if match:
        hint = match.group(1).strip()
    elif "Apache License" in content:
        hint = "Apache License"
    elif "MIT License" in content:
        hint = "MIT License"
    else:
        hint = "unparsed_license_text"
    return {"path": relative_path, "sha256": _file_sha256(path), "declared_license_hint": hint}


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"invalid JSON at line {index + 1}: {error.msg}") from error
            if not isinstance(row, dict):
                raise AuditError(f"line {index + 1} is not a JSON object")
            records.append(row)
    if not records:
        raise AuditError(f"no records found in {path}")
    return records


def _validate_record(record: Mapping[str, Any], *, index: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise AuditError(f"record {index} is missing required fields: {', '.join(missing)}")
    for field in ("repo_name", "file_name", "prefix_code", "suffix_code", "middle_code", "fill_type"):
        if not isinstance(record[field], str) or not record[field]:
            raise AuditError(f"record {index} field {field} must be a non-empty string")
    if not isinstance(record["context_code"], list):
        raise AuditError(f"record {index} field context_code must be a list")


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    for index, record in enumerate(records):
        _validate_record(record, index=index)
    fill_types = {str(record["fill_type"]) for record in records}
    if len(fill_types) != 6:
        raise AuditError(f"ExecRepoBench smoke requires exactly six fill types, found {len(fill_types)}")
    if len({str(record["repo_name"]) for record in records}) < 2:
        raise AuditError("ExecRepoBench smoke requires at least two repositories")


def _pick_smoke_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose one record per fill type while preferring distinct repositories."""
    by_fill: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        by_fill[str(record["fill_type"])].append((index, record))

    chosen: list[tuple[int, Mapping[str, Any]]] = []
    seen_repos: set[str] = set()
    for fill_type in sorted(by_fill):
        candidates = by_fill[fill_type]
        selected = min(
            candidates,
            key=lambda item: (
                str(item[1]["repo_name"]) in seen_repos,
                str(item[1]["repo_name"]),
                item[0],
            ),
        )
        chosen.append(selected)
        seen_repos.add(str(selected[1]["repo_name"]))

    if len({str(record["repo_name"]) for _, record in chosen}) < 2:
        raise AuditError("six-fill smoke selection could not cover multiple repositories")

    return [
        {
            "record_index": index,
            "record_sha256": _canonical_sha256(record),
            "repo_name": record["repo_name"],
            "file_name": record["file_name"],
            "fill_type": record["fill_type"],
        }
        for index, record in chosen
    ]


def build_audit(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_records(records)
    repo_counts = Counter(str(record["repo_name"]) for record in records)
    fill_counts = Counter(str(record["fill_type"]) for record in records)
    smoke_rows = _pick_smoke_rows(records)
    audit = {
        "mode": "smoke_plan_only",
        "final_external_results_opened": False,
        "record_count": len(records),
        "repository_count": len(repo_counts),
        "fill_type_count": len(fill_counts),
        "required_fields": list(REQUIRED_FIELDS),
        "repository_counts": dict(sorted(repo_counts.items())),
        "fill_type_counts": dict(sorted(fill_counts.items())),
        "smoke_case_count": len(smoke_rows),
        "smoke_repository_count": len({row["repo_name"] for row in smoke_rows}),
        "smoke_fill_types": [row["fill_type"] for row in smoke_rows],
    }
    return audit, smoke_rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_counts(path: Path, header: str, counts: Mapping[str, int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[header, "record_count"])
        writer.writeheader()
        writer.writerows({header: key, "record_count": value} for key, value in sorted(counts.items()))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, required=True)
    result.add_argument("--evaluator-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _verify_checkout(args.dataset_root, expected_commit=DATASET_COMMIT, label="dataset")
    evaluator = _verify_checkout(args.evaluator_root, expected_commit=EVALUATOR_COMMIT, label="Qwen evaluator")
    data_path = args.dataset_root / DATA_FILE_NAME
    entrypoint = args.evaluator_root / EVALUATOR_ENTRYPOINT
    if not data_path.is_file():
        raise AuditError(f"pinned dataset file is missing: {data_path}")
    if not entrypoint.is_dir():
        raise AuditError(f"pinned evaluator entrypoint is missing: {entrypoint}")

    records = _load_records(data_path)
    audit, smoke_rows = build_audit(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": {
            "repository": DATASET_REPOSITORY,
            **dataset,
            "data_file": DATA_FILE_NAME,
            "data_file_sha256": _file_sha256(data_path),
            "license": _license_audit(args.dataset_root, "README.md"),
        },
        "evaluator": {
            "repository": EVALUATOR_REPOSITORY,
            **evaluator,
            "entrypoint": EVALUATOR_ENTRYPOINT,
            "license": _license_audit(args.evaluator_root, "LICENSE"),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "git_version": subprocess.run(["git", "--version"], check=True, capture_output=True, text=True).stdout.strip(),
        },
        **audit,
    }
    _write_json(args.output_dir / "audit_manifest.json", manifest)
    _write_counts(args.output_dir / "repository_counts.csv", "repo_name", audit["repository_counts"])
    _write_counts(args.output_dir / "fill_type_counts.csv", "fill_type", audit["fill_type_counts"])
    with (args.output_dir / "smoke_plan.jsonl").open("w", encoding="utf-8") as handle:
        for row in smoke_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
