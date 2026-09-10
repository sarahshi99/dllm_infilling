#!/usr/bin/env python3
"""Recompute and audit the v2 prepared Markov training data.

This is read-only with respect to source data and existing artifacts.  It
reconstructs the pinned source population and the same conservative grouping
pipeline, then compares every prepared row and manifest entry, including the
actual token ids produced by the pinned tokenizer.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.prepare_dreamon_markov_training_data import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_SOURCE_ROWS,
    tokenize_record,
)
from experiments.dreamon_markov_head_training import deduplicate_and_split_records
from experiments.lrdllm_dreamcoder_adapter import load_dataset_rows

def read_zst(path: Path) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["zstd", "-q", "-d", "-c", str(path)],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        json.loads(line)
        for line in io.StringIO(completed.stdout.decode("utf-8"))
        if line.strip()
    ]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_diff(left: set[str], right: set[str]) -> dict[str, list[str]]:
    return {
        "only_left": sorted(left - right),
        "only_right": sorted(right - left),
    }


def run(args: argparse.Namespace) -> int:
    prepared_path = Path(args.prepared_records).resolve()
    split_manifest_path = Path(args.split_manifest).resolve()
    external_manifest_path = Path(args.external_test_manifest).resolve()
    exclusion_path = Path(args.exclusion_audit).resolve()
    summary_path = Path(args.data_summary).resolve()
    preparation_audit_path = Path(args.preparation_audit).resolve()

    dataset = load_dataset(
        DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION
    )
    source_rows = [dict(dataset[index]) for index in range(len(dataset))]
    source_ids = [int(row["seq_id"]) for row in source_rows]
    source_id_set = set(source_ids)
    if len(source_rows) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"source row count {len(source_rows)} != {EXPECTED_SOURCE_ROWS}")
    if len(source_id_set) != len(source_ids):
        raise RuntimeError("source seq_id is not unique")

    human_rows = load_dataset_rows(
        Path(args.evaluator_root).resolve(), "HumanEval-SingleLineInfilling"
    )
    clean_records, rebuild_audit = deduplicate_and_split_records(
        source_rows, human_eval_rows=human_rows, split_seed=int(args.split_seed)
    )
    clean_by_id = {str(row["record_id"]): row for row in clean_records}
    if len(clean_by_id) != len(clean_records):
        raise RuntimeError("rebuilt clean record_id is not unique")

    tokenizer = AutoTokenizer.from_pretrained(
        Path(args.model_snapshot).resolve(),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.expand_token_id = int(getattr(tokenizer, "expand_token_id", 151667))
    model_config = AutoConfig.from_pretrained(
        Path(args.model_snapshot).resolve(), trust_remote_code=True, local_files_only=True
    )

    prepared = read_gzip_jsonl(prepared_path)
    manifest = read_zst(split_manifest_path)
    external_manifest = read_zst(external_manifest_path)
    prepared_by_id = {str(row["record_id"]): row for row in prepared}
    manifest_by_id = {str(row["record_id"]): row for row in manifest}
    external_by_id = {str(row["record_id"]): row for row in external_manifest}

    errors: list[dict[str, Any]] = []

    def error(kind: str, **details: Any) -> None:
        if len(errors) < 100:
            errors.append({"kind": kind, **details})

    if len(prepared_by_id) != len(prepared):
        error("duplicate_prepared_record_id", rows=len(prepared), unique=len(prepared_by_id))
    if len(manifest_by_id) != len(manifest):
        error("duplicate_manifest_record_id", rows=len(manifest), unique=len(manifest_by_id))
    if len(external_by_id) != len(external_manifest):
        error("duplicate_external_record_id", rows=len(external_manifest), unique=len(external_by_id))

    prepared_ids = set(prepared_by_id)
    manifest_ids = set(manifest_by_id)
    external_ids = set(external_by_id)
    id_diffs = {
        "manifest_vs_prepared": set_diff(manifest_ids, prepared_ids),
        "external_vs_prepared_external": set_diff(
            external_ids, {rid for rid, row in prepared_by_id.items() if row.get("split") == "external_test"}
        ),
    }
    for name, diff in id_diffs.items():
        if diff["only_left"] or diff["only_right"]:
            error("record_id_set_mismatch", name=name, **{k: v[:20] for k, v in diff.items()})

    exclusion = load_json(exclusion_path)
    excluded_ids = {
        str(record_id)
        for group in exclusion.get("groups", [])
        for record_id in group.get("member_record_ids", [])
    }
    excluded_group_ids = {str(group["problem_group_id"]) for group in exclusion.get("groups", [])}
    if prepared_ids & excluded_ids:
        error("excluded_record_in_prepared", record_ids=sorted(prepared_ids & excluded_ids)[:20])
    if any(str(row["problem_group_id"]) in excluded_group_ids for row in prepared):
        error("excluded_group_in_prepared")

    split_counts = Counter(str(row.get("split")) for row in prepared)
    if rebuild_audit["source_rows"] != EXPECTED_SOURCE_ROWS:
        error("rebuild_source_count", actual=rebuild_audit["source_rows"])
    exclusion_count_checks = {
        "excluded_groups": (exclusion.get("excluded_groups"), rebuild_audit["humaneval_excluded_groups"]),
        "excluded_group_records": (exclusion.get("excluded_group_records"), rebuild_audit["humaneval_decontaminated_rows"]),
    }
    for field, (saved_value, rebuilt_value) in exclusion_count_checks.items():
        if saved_value != rebuilt_value:
            error("exclusion_audit_count_mismatch", field=field, saved=saved_value, rebuilt=rebuilt_value)
    if rebuild_audit["humaneval_base_tasks"] != 164:
        error("human_base_task_count", actual=rebuild_audit["humaneval_base_tasks"])
    if rebuild_audit["humaneval_variant_consistency"].get("variants") != 1033:
        error("human_variant_count", actual=rebuild_audit["humaneval_variant_consistency"].get("variants"))
    if rebuild_audit["humaneval_variant_consistency"].get("parse_failures"):
        error("human_variant_parse_failures", details=rebuild_audit["humaneval_variant_consistency"]["parse_failures"])
    if rebuild_audit["humaneval_variant_consistency"].get("inconsistent_base_tasks"):
        error("human_variant_inconsistency", details=rebuild_audit["humaneval_variant_consistency"]["inconsistent_base_tasks"])

    # Retokenize every clean source record.  This proves both that each saved
    # row is exact and that every omitted row really fails the frozen rules.
    tokenizer_mismatches = 0
    expected_prepared_ids: set[str] = set()
    tokenization_exclusions = Counter()
    source_seq_ids: set[int] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    token_id_min = None
    token_id_max = None
    for expected in clean_records:
        record_id = str(expected["record_id"])
        rebuilt, reason = tokenize_record(expected, tokenizer)
        if rebuilt is None:
            tokenization_exclusions[str(reason)] += 1
            continue
        expected_prepared_ids.add(record_id)
        saved = prepared_by_id.get(record_id)
        if saved is None:
            error("tokenizable_record_missing_from_prepared", record_id=record_id)
            continue
        source_seq_ids.add(int(saved["source_seq_id"]))
        group = str(saved["problem_group_id"])
        group_splits[group].add(str(saved["split"]))
        for field in ("source_seq_id", "problem_group_id", "split", "entry_point"):
            if str(saved.get(field)) != str(expected.get(field)):
                error("identity_field_mismatch", record_id=record_id, field=field, saved=saved.get(field), expected=expected.get(field))
        if int(saved["source_seq_id"]) not in source_id_set:
            error("unknown_source_seq_id", record_id=record_id, source_seq_id=saved["source_seq_id"])
        for key in ("row_seed", "prefix_ids", "middle_ids", "suffix_ids", "prefix_token_count", "middle_token_count", "suffix_token_count", "full_reference_length", "initial_canvas_length"):
            if saved.get(key) != rebuilt.get(key):
                tokenizer_mismatches += 1
                if tokenizer_mismatches <= 20:
                    error("tokenization_mismatch", record_id=record_id, field=key, saved=saved.get(key), rebuilt=rebuilt.get(key))
        all_ids = rebuilt["prefix_ids"] + rebuilt["middle_ids"] + rebuilt["suffix_ids"]
        if all_ids:
            token_id_min = min(token_id_min, min(all_ids)) if token_id_min is not None else min(all_ids)
            token_id_max = max(token_id_max, max(all_ids)) if token_id_max is not None else max(all_ids)

    unexpected_prepared_ids = sorted(prepared_ids - expected_prepared_ids)
    missing_prepared_ids = sorted(expected_prepared_ids - prepared_ids)
    if unexpected_prepared_ids:
        error("non_tokenizable_record_in_prepared", record_ids=unexpected_prepared_ids[:20])
    if missing_prepared_ids:
        error("tokenizable_record_missing_from_prepared_set", record_ids=missing_prepared_ids[:20])

    if token_id_min is None or token_id_min < 0:
        error("invalid_minimum_token_id", value=token_id_min)
    if token_id_max is None or token_id_max >= int(model_config.vocab_size):
        error("token_id_outside_model_vocabulary", maximum=token_id_max, vocab_size=int(model_config.vocab_size))

    cross_split_groups = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) != 1}
    if cross_split_groups:
        error("prepared_group_cross_split", groups=list(cross_split_groups.items())[:20])
    if len(source_seq_ids) != len(prepared):
        error("duplicate_prepared_source_seq_id", rows=len(prepared), unique=len(source_seq_ids))

    compact_fields = (
        "record_id", "problem_group_id", "source_seq_id", "split", "row_seed",
        "prefix_token_count", "middle_token_count", "suffix_token_count",
        "full_reference_length", "initial_canvas_length",
    )
    for record_id, saved in manifest_by_id.items():
        prepared_row = prepared_by_id.get(record_id)
        if prepared_row is None:
            continue
        for field in compact_fields:
            if saved.get(field) != prepared_row.get(field):
                error("split_manifest_field_mismatch", record_id=record_id, field=field)
    for record_id, saved in external_by_id.items():
        prepared_row = prepared_by_id.get(record_id)
        if prepared_row is None:
            continue
        for field in compact_fields:
            if saved.get(field) != prepared_row.get(field):
                error("external_manifest_field_mismatch", record_id=record_id, field=field)

    summary = load_json(summary_path)
    preparation_audit = load_json(preparation_audit_path)
    summary_checks = {
        "source_rows": (summary.get("actual_source_rows"), EXPECTED_SOURCE_ROWS),
        "prepared_rows": (summary.get("available_rows_after_normalize_dedup_decontam_tokenize"), len(prepared)),
        "split_counts": (summary.get("split_counts"), dict(split_counts)),
    }
    for name, (actual, expected) in summary_checks.items():
        if actual != expected:
            error("summary_mismatch", field=name, actual=actual, expected=expected)
    if preparation_audit.get("tokenization_exclusions") != dict(tokenization_exclusions):
        error(
            "tokenization_exclusion_mismatch",
            saved=preparation_audit.get("tokenization_exclusions"),
            rebuilt=dict(tokenization_exclusions),
        )

    audit = {
        "status": "passed" if not errors else "failed",
        "source": {
            "dataset_id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "rows": len(source_rows),
            "unique_seq_ids": len(source_id_set),
        },
        "human_eval": {
            "variants": len(human_rows),
            "base_tasks": rebuild_audit["humaneval_base_tasks"],
            "variant_consistency": rebuild_audit["humaneval_variant_consistency"],
            "direct_candidate_rows": rebuild_audit["humaneval_direct_candidate_rows"],
            "excluded_groups": rebuild_audit["humaneval_excluded_groups"],
            "excluded_records": rebuild_audit["humaneval_decontaminated_rows"],
            "prepared_intersection": len(prepared_ids & excluded_ids),
        },
        "rebuild": {
            "normalized_duplicate_rows_removed": rebuild_audit["normalized_duplicate_rows_removed"],
            "invalid_rows_removed": rebuild_audit["invalid_rows_removed"],
            "available_rows_before_tokenization": rebuild_audit["available_rows_after_filters"],
            "problem_group_count": rebuild_audit["problem_group_count"],
            "split_counts_before_tokenization": rebuild_audit["split_counts"],
            "records_omitted_by_tokenization": sum(tokenization_exclusions.values()),
            "tokenization_exclusions": dict(tokenization_exclusions),
        },
        "prepared": {
            "rows": len(prepared),
            "unique_record_ids": len(prepared_by_id),
            "unique_source_seq_ids": len(source_seq_ids),
            "split_counts": dict(split_counts),
            "problem_groups": len(group_splits),
            "cross_split_groups": cross_split_groups,
            "token_id_min": token_id_min,
            "token_id_max": token_id_max,
            "tokenizer_vocab_size": len(tokenizer),
            "model_vocab_size": int(model_config.vocab_size),
            "tokenization_mismatch_count": tokenizer_mismatches,
        },
        "manifests": {
            "split_rows": len(manifest),
            "external_rows": len(external_manifest),
            "manifest_record_id_match": all(
                not diff["only_left"] and not diff["only_right"] for diff in id_diffs.values()
            ),
        },
        "errors": errors,
        "source_artifacts_unchanged": True,
        "gpu_used": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0 if audit["status"] == "passed" else 2


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--evaluator-root", required=True)
    parser.add_argument("--prepared-records", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--external-test-manifest", required=True)
    parser.add_argument("--exclusion-audit", required=True)
    parser.add_argument("--data-summary", required=True)
    parser.add_argument("--preparation-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=20260901)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
