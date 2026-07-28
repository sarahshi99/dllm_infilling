#!/usr/bin/env python3
"""Freeze an outcome-blind 148-group M4 selection from the immutable 40,632-row bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.m2_constraint_homotopy import load_frozen_groups, read_jsonl, write_json
from experiments.m4_semantic_particle_assembly import build_multiline_core_selection_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(bank_raw: Path, output: Path) -> dict[str, object]:
    frozen_groups, lock = load_frozen_groups()
    selected_rows = build_multiline_core_selection_manifest(read_jsonl(bank_raw), frozen_groups)
    if len(selected_rows) != 148 or len({row["task_group"] for row in selected_rows}) != 148:
        raise RuntimeError("M4 MultiLine-Core selection must contain exactly 148 task groups")
    payload: dict[str, object] = {
        "schema_version": 1,
        "selection_rule": "minimum sha256('m4_multiline_core_v1|task_group|row_key') per task group",
        "source_bank": str(bank_raw),
        "source_bank_sha256": sha256_file(bank_raw),
        "source_bank_rows": 40632,
        "selected_row_count": len(selected_rows),
        "selected_task_group_count": len({row["task_group"] for row in selected_rows}),
        "frozen_test_status": lock.get("test_status"),
        "test_evaluation_count": lock.get("test_evaluation_count"),
        "selected_rows": selected_rows,
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank-raw", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.candidate_bank_raw).resolve(), Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
