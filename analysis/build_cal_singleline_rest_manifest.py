#!/usr/bin/env python3
"""Build immutable seed-42 SingleLine CAL-Rest x project-non-frozen manifests.

The project allowed set comes from an existing 927-row manifest that already
records `included_not_frozen_controller_test`.  This builder never opens the
sealed controller files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


EVALUATOR_COMMIT = "88062ff9859c875d04db115b698ed4b0f0395170"
EVALUATION_FIELDS = ("prompt", "suffix", "canonical_solution", "test", "entry_point")
PUBLIC_FIELDS = (
    "candidate_key",
    "source_row_id",
    "task_id",
    "task_group",
    "dataset",
    "population",
    "evaluator_commit",
    "seed",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "task_id": str(row.get("task_id") or ""),
                "task_group": str(row.get("task_group") or ""),
                "frozen_controller_test_row": str(row.get("frozen_controller_test_row") or ""),
                "frozen_controller_test_exclusion_flag": str(
                    row.get("frozen_controller_test_exclusion_flag") or ""
                ),
                "source_dataset": str(row.get("source_dataset") or ""),
            }
            for row in csv.DictReader(handle)
        ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def evaluation_hash(row: Mapping[str, Any]) -> str:
    payload = {field: normalize_text(row.get(field, "")) for field in EVALUATION_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def task_group(task_id: str) -> str:
    match = re.search(r"HumanEval/(\d+)", task_id)
    if not match:
        raise ValueError(f"cannot derive task group from {task_id!r}")
    return f"HumanEval/{match.group(1)}"


def seed42_demo_and_rest(
    rows: Sequence[Mapping[str, Any]], *, demo_count: int = 100
) -> tuple[list[tuple[int, Mapping[str, Any]]], list[tuple[int, Mapping[str, Any]]]]:
    indexed = list(enumerate(rows))
    random.Random(42).shuffle(indexed)
    return indexed[:demo_count], indexed[demo_count:]


def _allowed_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    allowed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in allowed:
            raise RuntimeError("allowed SingleLine manifest has blank or duplicate task_id")
        if str(row.get("frozen_controller_test_row")) != "False":
            raise RuntimeError("allowed SingleLine manifest contains a frozen row")
        if str(row.get("frozen_controller_test_exclusion_flag")) != "included_not_frozen_controller_test":
            raise RuntimeError("allowed SingleLine manifest lacks the non-frozen inclusion flag")
        if str(row.get("source_dataset")) != "HumanEval-SingleLineInfilling":
            raise RuntimeError("allowed SingleLine manifest has the wrong source dataset")
        allowed[task_id] = row
    return allowed


def build_common_manifest(
    official_rows: Sequence[Mapping[str, Any]],
    allowed_rows: Sequence[Mapping[str, Any]],
    *,
    demo_count: int = 100,
) -> list[dict[str, Any]]:
    allowed = _allowed_map(allowed_rows)
    _, rest = seed42_demo_and_rest(official_rows, demo_count=demo_count)
    result: list[dict[str, Any]] = []
    for source_row_id, row in rest:
        task_id = str(row["task_id"])
        if task_id not in allowed:
            continue
        group = task_group(task_id)
        if str(allowed[task_id]["task_group"]) != group:
            raise RuntimeError(f"allowed task_group mismatch for {task_id}")
        result.append(
            {
                "candidate_key": task_id,
                "source_row_id": int(source_row_id),
                "task_id": task_id,
                "task_group": group,
                "dataset": "HumanEval-SingleLineInfilling",
                "population": "CAL-Rest_intersection_project_non-frozen",
                "evaluator_commit": EVALUATOR_COMMIT,
                "seed": 42,
                "_technical_length": len(normalize_text(row.get("canonical_solution", "")).encode("utf-8")),
            }
        )
    return result


def choose_smoke(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    def technical_length(row: Mapping[str, Any]) -> int:
        value = row.get("_technical_length", row.get("technical_length"))
        if value is None:
            raise RuntimeError("smoke selection requires an internal technical length")
        return int(value)

    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["task_group"])].append(row)
    representatives = [
        min(group_rows, key=lambda row: (technical_length(row), int(row["source_row_id"])))
        for _, group_rows in sorted(by_group.items())
    ]
    if count <= 0 or count > len(representatives):
        raise ValueError("invalid smoke count")
    ordered = sorted(representatives, key=lambda row: (technical_length(row), str(row["task_group"])))
    buckets: list[list[Mapping[str, Any]]] = [ordered[index::4] for index in range(4)]
    per_bucket, remainder = divmod(count, 4)
    selected: list[dict[str, Any]] = []
    for bucket_index, candidates in enumerate(buckets):
        take = per_bucket + int(bucket_index < remainder)
        for row in candidates[:take]:
            public = {field: row[field] for field in PUBLIC_FIELDS}
            public["smoke_order"] = len(selected)
            public["technical_stratum"] = f"q{bucket_index + 1}"
            selected.append(public)
    if len(selected) != count or len({row["task_group"] for row in selected}) != count:
        raise RuntimeError("smoke manifest must contain unique base-function groups")
    return selected


def demo_mapping(
    singleline_rows: Sequence[Mapping[str, Any]], multiline_rows: Sequence[Mapping[str, Any]]
) -> list[tuple[int, int]]:
    demo, _ = seed42_demo_and_rest(singleline_rows)
    multi_by_hash: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(multiline_rows):
        multi_by_hash[evaluation_hash(row)].append(index)
    mapping: list[tuple[int, int]] = []
    for single_index, row in demo:
        matches = multi_by_hash[evaluation_hash(row)]
        if len(matches) != 1:
            raise RuntimeError(f"SingleLine demo row {single_index} does not map uniquely to MultiLine")
        mapping.append((single_index, matches[0]))
    if len(mapping) != 100 or len({multi for _, multi in mapping}) != 100:
        raise RuntimeError("seed-42 demo mapping is not one-to-one")
    return mapping


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--official-singleline", type=Path, required=True)
    result.add_argument("--official-multiline", type=Path, required=True)
    result.add_argument("--multiline-common-manifest", type=Path, required=True)
    result.add_argument("--allowed-singleline-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--smoke-cases", type=int, default=12)
    return result


def main() -> int:
    args = parser().parse_args()
    singleline = read_jsonl(args.official_singleline)
    multiline = read_jsonl(args.official_multiline)
    allowed = read_csv(args.allowed_singleline_manifest)
    common = build_common_manifest(singleline, allowed)
    mapping = demo_mapping(singleline, multiline)
    existing_multiline_common = read_jsonl(args.multiline_common_manifest)
    mapped_multi_ids = {multi for _, multi in mapping}
    observed_multi_ids = {int(row["source_row_id"]) for row in existing_multiline_common}
    if mapped_multi_ids & observed_multi_ids:
        raise RuntimeError("existing MultiLine CAL-Rest common manifest contains a seed-42 demo row")
    if not (
        len(singleline) == 1033
        and len(allowed) == 927
        and len(common) == 838
        and len({row["task_group"] for row in common}) == 143
        and len(existing_multiline_common) == 4990
        and len({str(row["task_group"]) for row in existing_multiline_common}) == 143
    ):
        raise RuntimeError("CAL population counts do not match the preregistered ledger")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "cal_singleline_rest_nonfrozen_manifest.jsonl"
    smoke_path = args.output_dir / "cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl"
    public_common = [{field: row[field] for field in PUBLIC_FIELDS} for row in common]
    write_jsonl(full_path, public_common)
    write_jsonl(smoke_path, choose_smoke(common, args.smoke_cases))
    mapping_payload = json.dumps(mapping, separators=(",", ":")).encode("utf-8")
    summary = {
        "schema_version": 1,
        "status": "immutable_manifest_frozen_outcome_blind",
        "official_singleline_rows": 1033,
        "seed42_demo_rows": 100,
        "seed42_rest_rows": 933,
        "project_nonfrozen_rows": 927,
        "cal_rest_intersection_nonfrozen_rows": 838,
        "cal_rest_intersection_nonfrozen_clusters": 143,
        "multiline_cal_rest_common_rows": 4990,
        "multiline_cal_rest_common_clusters": 143,
        "demo_mapping_count": 100,
        "demo_mapping_sha256": hashlib.sha256(mapping_payload).hexdigest(),
        "official_singleline_sha256": sha256(args.official_singleline),
        "official_multiline_sha256": sha256(args.official_multiline),
        "allowed_singleline_manifest_sha256": sha256(args.allowed_singleline_manifest),
        "multiline_common_manifest_sha256": sha256(args.multiline_common_manifest),
        "full_manifest_sha256": sha256(full_path),
        "smoke_manifest_sha256": sha256(smoke_path),
        "frozen_controller_test": {"status": "sealed", "test_evaluation_count": 0},
        "sealed_files_opened": False,
        "forbidden_fields_present": [],
    }
    (args.output_dir / "cal_singleline_rest_nonfrozen_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
