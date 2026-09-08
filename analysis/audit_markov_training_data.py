#!/usr/bin/env python3
"""Retrospective source/manifest audit. Never changes a split or training bank."""
from __future__ import annotations
import argparse
import csv
import gzip
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.markov_data_identity import (
    code_key, connected_groups, human_index, human_matches, response_code, text_key,
)


def read_zst(path):
    import zstandard
    with open(path, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
        return [json.loads(line) for line in io.TextIOWrapper(r) if line.strip()]


def write_json(path, data):
    content = (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(content, mtime=0))
    else:
        path.write_bytes(content)


def run(args):
    import pyarrow.parquet as pq
    result = Path(args.result_dir)
    source = pq.read_table(args.source_parquet).to_pylist()
    if len(source) != 118278:
        raise ValueError("Wrong pinned OpenCoder population")
    by_id = {int(row["seq_id"]): row for row in source}
    if len(by_id) != len(source):
        raise ValueError("Duplicate source IDs")
    manifest = read_zst(result / "split_manifest.jsonl.zst")
    if len({r["record_id"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate manifest records")
    records = [{**by_id[int(m["source_seq_id"])], **m} for m in manifest]
    with gzip.open(args.humaneval, "rt") as f:
        human = human_index([json.loads(line) for line in f])
    # Record direct equality evidence separately from transitive conservative groups.
    equality = {}
    for field, getter in (
        ("instruction", lambda r: text_key(r["instruction"])),
        ("code", lambda r: text_key(r["code"])),
        ("response_code", lambda r: text_key(response_code(r))),
    ):
        groups = defaultdict(list)
        for r in records:
            key = getter(r)
            if key:
                groups[key].append(r)
        cross = [rs for rs in groups.values() if len({r["split"] for r in rs}) > 1]
        equality[field] = {
            "cross_split_groups": len(cross),
            "cross_split_rows": sum(map(len, cross)),
            "rows_by_split": dict(Counter(r["split"] for rs in cross for r in rs)),
        }
    representatives = connected_groups(records)
    components = defaultdict(list)
    for row, representative in zip(records, representatives):
        components[representative].append(row)
    cross_components = [rs for rs in components.values() if len({r["split"] for r in rs}) > 1]
    group_map = {row["problem_group_id"]: "source/" + str(records[representative]["source_seq_id"])
                 for representative, rows in components.items() for row in rows}
    excluded = set()
    evidence_rows = []
    for representative, rows in components.items():
        splits = {r["split"] for r in rows}
        # Validation must not share train; test must not share train OR validation
        # (validation selected head/lambda). Exclude whole OLD groups conservatively.
        for row in rows:
            unsafe = (row["split"] == "validation" and "train" in splits) or (
                row["split"] == "external_test" and bool(splits & {"train", "validation"})
            )
            if unsafe:
                excluded.add(row["problem_group_id"])
            if len(splits) > 1:
                evidence_rows.append({
                    "component_source_seq_id": records[representative]["source_seq_id"],
                    "source_seq_id": row["source_seq_id"], "record_id": row["record_id"],
                    "problem_group_id": row["problem_group_id"], "split": row["split"],
                    "component_splits": ";".join(sorted(splits)),
                    "exclude_evaluation_group": unsafe,
                })
    candidates = []
    response_diff = 0
    parse_fail = Counter()
    for row in records:
        if text_key(row["code"]) != text_key(response_code(row)):
            response_diff += 1
        if code_key(response_code(row)) is None:
            parse_fail[row["split"]] += 1
        matches = human_matches(row, human)
        if matches:
            candidates.append({
                "source_seq_id": row["source_seq_id"], "record_id": row["record_id"],
                "problem_group_id": row["problem_group_id"], "split": row["split"],
                "entry_point": row["entry_point"], "matches": matches,
            })
    for candidate in candidates:
        if candidate["split"] != "train":
            excluded.add(candidate["problem_group_id"])
    for name, rows in (("data_overlap_records.csv", evidence_rows),):
        with (result / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["source_seq_id"], lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    write_json(result / "humaneval_overlap_candidates.json", candidates)
    write_json(result / "sensitivity_excluded_problem_groups.json", sorted(excluded))
    write_json(result / "sensitivity_group_map.json.gz", group_map)
    audit = {
        "audit_date_utc": "2026-09-08", "audit_type": "retrospective_source_manifest",
        "dataset_revision": "7d28f40d579edd7c24402d17d0c7639f991e6f8d",
        "source_rows": len(source), "prepared_manifest_rows": len(records),
        "missing_source_ids": 0, "direct_equality": equality,
        "conservative_cross_split_components": len(cross_components),
        "conservative_cross_split_rows": sum(map(len, cross_components)),
        "sensitivity_excluded_old_problem_groups": len(excluded),
        "humaneval_base_tasks": human["base_tasks"],
        "humaneval_candidate_records": len(candidates),
        "humaneval_candidates_by_split": dict(Counter(r["split"] for r in candidates)),
        "response_code_differs_from_code_field": response_diff,
        "response_code_ast_parse_failures": dict(parse_fail),
        "training_bank_available_here": False,
        "actual_training_sample_overlap": "not verified; server SQLite sample membership required",
        "frozen_split_changed": False,
        "next_action": "join candidate record/group IDs to actual training bank before HumanEval generation",
        "limitations": [
            "Exact text/AST/test checks do not prove absence of semantic or renamed duplicates.",
            "One matching assert is a candidate for manual review, not proof of benchmark copying.",
            "Sensitivity exclusions use the full training/validation pool, not the unavailable sampled bank.",
            "Existing test was already opened; filtered results are post-hoc sensitivity, not a new held-out test.",
        ],
    }
    write_json(result / "data_overlap_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--humaneval", required=True)
    parser.add_argument("--result-dir", required=True)
    run(parser.parse_args())
