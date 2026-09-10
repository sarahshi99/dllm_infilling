#!/usr/bin/env python3
"""Read-only full membership and integrity audit for the v2 transition bank."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sqlite3
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.build_dreamon_markov_transition_bank import (
    MODEL_REVISION,
    POLICIES,
    SOURCE_REVISION,
    STAGES,
    TOKENIZER_REVISION,
    capacities,
)

MASK_TOKEN_ID = 151666
EXPAND_TOKEN_ID = 151667
EOS_TOKEN_ID = 151643
MODEL_VOCAB_SIZE = 152064


def load_prepared(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row["record_id"])
            if record_id in records:
                raise RuntimeError(f"duplicate prepared record_id: {record_id}")
            records[record_id] = row
    return records


def run(args: argparse.Namespace) -> int:
    bank_path = Path(args.bank_db).resolve()
    prepared = load_prepared(Path(args.prepared_records).resolve())
    exclusion = json.loads(Path(args.exclusion_audit).read_text(encoding="utf-8"))
    excluded_records = {
        str(record_id)
        for group in exclusion.get("groups", [])
        for record_id in group.get("member_record_ids", [])
    }
    excluded_groups = {
        str(group["problem_group_id"]) for group in exclusion.get("groups", [])
    }
    targets = {
        "train": int(args.train_target),
        "validation": int(args.validation_target),
        "external_test": int(args.external_test_target),
    }
    expected_counts = {
        (split, policy, stage): count
        for split, target in targets.items()
        for (policy, stage), count in capacities(target).items()
    }
    errors: list[dict[str, Any]] = []

    def error(kind: str, **details: Any) -> None:
        if len(errors) < 100:
            errors.append({"kind": kind, **details})

    connection = sqlite3.connect(f"file:{bank_path}?mode=ro", uri=True)
    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    if quick_check != "ok":
        error("sqlite_quick_check", result=quick_check)

    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(samples)")
    }
    required_columns = {
        "split", "policy", "stage", "slot", "sample_key",
        "problem_group_id", "record_id", "payload",
    }
    if not required_columns.issubset(columns):
        error("sample_schema_missing_columns", missing=sorted(required_columns - columns))

    selected = Counter()
    slots: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    contributions = Counter()
    sample_keys: set[str] = set()
    record_ids: set[str] = set()
    payload_errors = Counter()
    total = 0
    cursor = connection.execute(
        "SELECT split,policy,stage,slot,sample_key,problem_group_id,record_id,payload "
        "FROM samples ORDER BY split,policy,stage,slot"
    )
    for split, policy, stage, slot, sample_key, group_id, record_id, compressed in cursor:
        total += 1
        split = str(split)
        policy = str(policy)
        stage = str(stage)
        sample_key = str(sample_key)
        group_id = str(group_id)
        record_id = str(record_id)
        stratum = (split, policy, stage)
        selected[stratum] += 1
        slots[stratum].add(int(slot))
        contributions[(split, policy, group_id)] += 1
        if sample_key in sample_keys:
            error("duplicate_sample_key", sample_key=sample_key)
        sample_keys.add(sample_key)
        record_ids.add(record_id)
        expected = prepared.get(record_id)
        if expected is None:
            error("bank_record_not_prepared", record_id=record_id)
        else:
            for field, actual in (
                ("split", split),
                ("problem_group_id", group_id),
            ):
                if str(expected[field]) != actual:
                    error(
                        "bank_prepared_identity_mismatch",
                        record_id=record_id,
                        field=field,
                        bank=actual,
                        prepared=expected[field],
                    )
        if record_id in excluded_records:
            error("excluded_record_in_bank", record_id=record_id)
        if group_id in excluded_groups:
            error("excluded_group_in_bank", problem_group_id=group_id)
        try:
            payload = json.loads(zlib.decompress(compressed))
        except Exception as exc:
            payload_errors[type(exc).__name__] += 1
            error("payload_decode_failure", sample_key=sample_key, exception=type(exc).__name__)
            continue
        for field, actual in (
            ("sample_key", sample_key),
            ("record_id", record_id),
            ("problem_group_id", group_id),
            ("split", split),
            ("trajectory_policy", policy),
            ("generation_stage", stage),
        ):
            if str(payload.get(field)) != actual:
                error(
                    "payload_column_mismatch",
                    sample_key=sample_key,
                    field=field,
                    payload=payload.get(field),
                    column=actual,
                )
        frozen_fields = {
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "source_revision": SOURCE_REVISION,
            "offset": 1,
            "temperature": 0.2,
            "top_p": 0.9,
        }
        for field, frozen in frozen_fields.items():
            if payload.get(field) != frozen:
                error(
                    "frozen_field_mismatch",
                    sample_key=sample_key,
                    field=field,
                    actual=payload.get(field),
                    expected=frozen,
                )
        if expected is not None:
            for field in ("source_seq_id", "row_seed"):
                if payload.get(field) != expected.get(field):
                    error(
                        "payload_prepared_mismatch",
                        sample_key=sample_key,
                        field=field,
                        payload=payload.get(field),
                        prepared=expected.get(field),
                    )
        stale = payload.get("stale_input_ids")
        fresh = payload.get("fresh_input_ids")
        target = payload.get("target_position")
        local_target = payload.get("target_local_position")
        if not isinstance(stale, list) or not isinstance(fresh, list) or len(stale) != len(fresh):
            error("invalid_replay_lengths", sample_key=sample_key)
            continue
        if not isinstance(target, int) or not 0 <= target < len(fresh):
            error("invalid_target_position", sample_key=sample_key, target=target)
            continue
        if fresh[target] != MASK_TOKEN_ID or stale[target] != MASK_TOKEN_ID:
            error(
                "target_not_active_mask",
                sample_key=sample_key,
                stale=stale[target],
                fresh=fresh[target],
            )
        if not isinstance(local_target, int):
            error("invalid_local_target_position", sample_key=sample_key, value=local_target)
        elif expected is not None and target != local_target + len(expected["prefix_ids"]):
            error("target_coordinate_mismatch", sample_key=sample_key)
        if target < 1 or fresh[target - 1] != payload.get("previous_token_id"):
            error("predecessor_token_mismatch", sample_key=sample_key)
        previous = payload.get("previous_token_id")
        reference = payload.get("reference_token_id")
        for name, token in (("previous_token_id", previous), ("reference_token_id", reference)):
            if not isinstance(token, int) or not 0 <= token < MODEL_VOCAB_SIZE:
                error("token_outside_model_vocabulary", sample_key=sample_key, field=name, token=token)
        if previous in {MASK_TOKEN_ID, EXPAND_TOKEN_ID, EOS_TOKEN_ID}:
            error("structural_predecessor_in_bank", sample_key=sample_key, token=previous)

    for stratum, expected in expected_counts.items():
        actual = selected[stratum]
        if actual != expected:
            error("stratum_count_mismatch", stratum=stratum, actual=actual, expected=expected)
        expected_slots = set(range(expected))
        if slots[stratum] != expected_slots:
            error(
                "stratum_slots_not_contiguous",
                stratum=stratum,
                count=len(slots[stratum]),
                expected=expected,
            )
    unexpected_strata = sorted(set(selected) - set(expected_counts))
    if unexpected_strata:
        error("unexpected_strata", strata=unexpected_strata)
    if total != sum(targets.values()):
        error("bank_total_mismatch", actual=total, expected=sum(targets.values()))

    maximum_contribution = max(contributions.values(), default=0)
    if maximum_contribution > int(args.per_problem_cap):
        error(
            "per_problem_policy_cap_exceeded",
            actual=maximum_contribution,
            maximum=int(args.per_problem_cap),
        )

    counters = {
        (str(split), str(policy), str(stage)): int(seen)
        for split, policy, stage, seen in connection.execute(
            "SELECT split,policy,stage,seen FROM counters"
        )
    }
    for stratum, capacity in expected_counts.items():
        minimum_seen = math.ceil(capacity * float(args.oversample))
        if counters.get(stratum, -1) < minimum_seen:
            error(
                "insufficient_seen_for_reservoir",
                stratum=stratum,
                actual=counters.get(stratum),
                minimum=minimum_seen,
            )
    progress = {
        str(split): int(index)
        for split, index in connection.execute(
            "SELECT split,next_record_index FROM progress"
        )
    }
    prepared_split_counts = Counter(str(row["split"]) for row in prepared.values())
    for split in targets:
        index = progress.get(split)
        if index is None or index < 0 or index > prepared_split_counts[split]:
            error(
                "invalid_resume_progress",
                split=split,
                index=index,
                prepared_rows=prepared_split_counts[split],
            )
    connection.close()

    output = {
        "status": "passed" if not errors else "failed",
        "sqlite_quick_check": quick_check,
        "targets": targets,
        "total_samples": total,
        "unique_sample_keys": len(sample_keys),
        "sampled_record_ids": len(record_ids),
        "selected_counts": [
            {
                "split": split,
                "policy": policy,
                "generation_stage": stage,
                "count": selected[(split, policy, stage)],
            }
            for split, policy, stage in sorted(expected_counts)
        ],
        "maximum_selected_per_problem_policy": maximum_contribution,
        "per_problem_policy_cap": int(args.per_problem_cap),
        "excluded_record_intersection": len(record_ids & excluded_records),
        "payload_decode_errors": dict(payload_errors),
        "resume_progress": progress,
        "errors": errors,
        "bank_modified": False,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if output["status"] == "passed" else 2


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-db", required=True)
    parser.add_argument("--prepared-records", required=True)
    parser.add_argument("--exclusion-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-target", type=int, default=200000)
    parser.add_argument("--validation-target", type=int, default=20000)
    parser.add_argument("--external-test-target", type=int, default=20000)
    parser.add_argument("--per-problem-cap", type=int, default=16)
    parser.add_argument("--oversample", type=float, default=1.25)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
