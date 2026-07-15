#!/usr/bin/env python3
"""Audit the corrected official CAL Rest/common evaluation population.

This CPU-only audit pins source revisions, reproduces CAL's seed-42
SingleLine demo split, maps its 100 demo rows to MultiLine by normalized
evaluation-field hashes, and emits a deterministic 12-case smoke manifest.
It does not generate samples or inspect outcomes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
EXPECTED_CAL_COMMIT = "741e8418a88a732b4c92812424d4f03cab1f7b1f"
EXPECTED_HUMANEVAL_COMMIT = "88062ff9859c875d04db115b698ed4b0f0395170"
REQUIRED_PACKAGES = ("torch", "transformers", "numpy", "scipy", "tqdm", "datasets", "accelerate")
EVALUATION_FIELDS = ("prompt", "suffix", "canonical_solution", "test", "entry_point")
PRIMARY_CAL_CONFIG = {
    "initial_gen_length": 32,
    "span": 1,
    "dstep": 4,
    "max_gen_length": 128,
    "use_bias": True,
    "temperature": 0.0,
    "cfg_scale": 0.0,
    "oracle": False,
    "steps": None,
    "block_length": None,
}
OFFICIAL_FIXED32_CONFIG = {
    **PRIMARY_CAL_CONFIG,
    "dstep": -1,
    "use_bias": False,
    "arm": "official_fixed32_same_decoder",
}
PROJECT_FIXED64_CONFIG = {
    "canvas_tokens": 64,
    "steps": 64,
    "arm": "project_internal_fixed64_control_not_same_compute_as_official_cal",
}


def git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def evaluation_hash(row: Mapping[str, Any]) -> str:
    payload = {field: normalize_text(row.get(field, "")) for field in EVALUATION_FIELDS}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def task_group(task_id: str) -> str:
    match = re.search(r"HumanEval/(\d+)", task_id)
    if not match:
        raise ValueError(f"cannot derive HumanEval group from {task_id!r}")
    return f"HumanEval/{match.group(1)}"


def seed42_demo_and_rest(rows: Sequence[Mapping[str, Any]]) -> tuple[list[tuple[int, Mapping[str, Any]]], list[tuple[int, Mapping[str, Any]]]]:
    indexed = list(enumerate(rows))
    random.Random(42).shuffle(indexed)
    return indexed[:100], indexed[100:]


def frozen_groups() -> set[str]:
    lock_path = REPO / "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"
    grouped_path = REPO / "analysis_outputs/grouped_split_20260702_accel2/test_tasks.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("test_status") != "sealed" or int(lock.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError("frozen controller test is not sealed with evaluation count zero")
    groups = {str(item) for item in lock.get("test_task_ids", [])}
    groups.update(str(item) for item in json.loads(grouped_path.read_text(encoding="utf-8")))
    return groups


def map_demo_to_multiline(
    demo_rows: Sequence[tuple[int, Mapping[str, Any]]], multiline_rows: Sequence[Mapping[str, Any]]
) -> dict[int, int]:
    by_hash: dict[str, list[int]] = defaultdict(list)
    for source_row_id, row in enumerate(multiline_rows):
        by_hash[evaluation_hash(row)].append(source_row_id)
    mapped: dict[int, int] = {}
    for singleline_source_id, row in demo_rows:
        matches = by_hash[evaluation_hash(row)]
        if len(matches) != 1:
            raise RuntimeError(
                f"CAL demo row {singleline_source_id} does not map uniquely to MultiLine: {matches[:5]}"
            )
        mapped[int(singleline_source_id)] = int(matches[0])
    if len(set(mapped.values())) != len(mapped):
        raise RuntimeError("CAL demo mapping is not one-to-one")
    return mapped


def field_hash_comparison(
    official_rows: Sequence[Mapping[str, Any]], current_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    official_by_id = {str(row["task_id"]): row for row in official_rows}
    current_by_id = {str(row["task_id"]): row for row in current_rows}
    if set(official_by_id) != set(current_by_id) or len(official_by_id) != len(official_rows) or len(current_by_id) != len(current_rows):
        raise RuntimeError("official/current MultiLine task IDs are not an exact one-to-one mapping")
    records: list[dict[str, Any]] = []
    mismatch_count = 0
    for task_id in sorted(official_by_id):
        official_hash = evaluation_hash(official_by_id[task_id])
        current_hash = evaluation_hash(current_by_id[task_id])
        mismatch_count += int(official_hash != current_hash)
        records.append(
            {
                "task_id_sha256": hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
                "task_group": task_group(task_id),
                "official_evaluation_hash": official_hash,
                "current_evaluation_hash": current_hash,
                "match": official_hash == current_hash,
            }
        )
    if mismatch_count:
        raise RuntimeError(f"official/current evaluation-field hash mismatch count={mismatch_count}")
    return records


def _length_bucket_representatives(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["task_group"])].append(row)
    representatives: list[dict[str, Any]] = []
    for group, group_rows in sorted(by_group.items()):
        chosen = min(group_rows, key=lambda row: (int(row["completion_utf8_bytes"]), int(row["source_row_id"])))
        representatives.append(dict(chosen))
    return representatives


def choose_smoke(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    representatives = _length_bucket_representatives(rows)
    if count <= 0 or count > len(representatives):
        raise ValueError("invalid smoke count")
    lengths = sorted(int(row["completion_utf8_bytes"]) for row in representatives)
    cut1 = lengths[(len(lengths) - 1) // 4]
    cut2 = lengths[(len(lengths) - 1) // 2]
    cut3 = lengths[(3 * (len(lengths) - 1)) // 4]

    def bucket(value: int) -> str:
        if value <= cut1:
            return "q1_short"
        if value <= cut2:
            return "q2_medium"
        if value <= cut3:
            return "q3_long"
        return "q4_extreme"

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representatives:
        grouped[bucket(int(row["completion_utf8_bytes"]))].append(row)
    selected: list[dict[str, Any]] = []
    per_bucket, remainder = divmod(count, 4)
    for index, label in enumerate(("q1_short", "q2_medium", "q3_long", "q4_extreme")):
        candidates = sorted(grouped[label], key=lambda row: (int(row["completion_utf8_bytes"]), int(row["source_row_id"])))
        take = per_bucket + (1 if index < remainder else 0)
        if len(candidates) < take:
            raise RuntimeError(f"insufficient unique base tasks in smoke length bucket {label}")
        for row in candidates[:take]:
            selected.append(
                {
                    "smoke_order": len(selected),
                    "source_row_id": int(row["source_row_id"]),
                    "task_id": str(row["task_id"]),
                    "task_group": str(row["task_group"]),
                    "task_id_sha256": hashlib.sha256(str(row["task_id"]).encode("utf-8")).hexdigest(),
                    "length_bucket": label,
                    "completion_utf8_bytes_offline_only": int(row["completion_utf8_bytes"]),
                    "demo_linked": False,
                }
            )
    if len({row["task_group"] for row in selected}) != len(selected):
        raise RuntimeError("smoke manifest must deduplicate base tasks")
    return selected


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(
    *,
    official_cal_root: Path,
    humaneval_root: Path,
    current_dataset: Path,
    output_dir: Path,
    smoke_cases: int,
) -> dict[str, Any]:
    if git_head(official_cal_root) != EXPECTED_CAL_COMMIT:
        raise RuntimeError("official CAL checkout is not the pinned commit")
    if git_head(humaneval_root) != EXPECTED_HUMANEVAL_COMMIT:
        raise RuntimeError("HumanEval-Infilling checkout is not the pinned commit")
    cal_script = official_cal_root / "llada_cal" / "llada_code_infilling.py"
    official_multi = humaneval_root / "data" / "HumanEval-MultiLineInfilling.jsonl.gz"
    official_single = humaneval_root / "data" / "HumanEval-SingleLineInfilling.jsonl.gz"
    evaluator = humaneval_root / "human_eval_infilling" / "evaluate_functional_correctness.py"
    if not all(path.is_file() for path in (cal_script, official_multi, official_single, evaluator, current_dataset)):
        raise RuntimeError("pinned CAL source, evaluator, or dataset input is missing")

    official_multi_rows = read_jsonl(official_multi)
    current_rows = read_jsonl(current_dataset)
    single_rows = read_jsonl(official_single)
    comparison = field_hash_comparison(official_multi_rows, current_rows)
    demo, single_rest = seed42_demo_and_rest(single_rows)
    demo_to_multi = map_demo_to_multiline(demo, official_multi_rows)
    demo_multi_ids = set(demo_to_multi.values())
    multi_rest = [
        {**dict(row), "source_row_id": source_row_id}
        for source_row_id, row in enumerate(current_rows)
        if source_row_id not in demo_multi_ids
    ]
    frozen = frozen_groups()
    single_rest_nonfrozen = [row for _, row in single_rest if task_group(str(row["task_id"])) not in frozen]
    common = [
        {
            **row,
            "task_group": task_group(str(row["task_id"])),
            "completion_utf8_bytes": len(normalize_text(row.get("canonical_solution", "")).encode("utf-8")),
        }
        for row in multi_rest
        if task_group(str(row["task_id"])) not in frozen
    ]
    if not (
        len(single_rows) == 1033
        and len(demo) == 100
        and len(single_rest) == 933
        and len(single_rest_nonfrozen) == 838
        and len(official_multi_rows) == 5815
        and len(demo_multi_ids) == 100
        and len(multi_rest) == 5715
        and len(common) == 4990
    ):
        raise RuntimeError(
            "corrected CAL population counts failed: "
            + json.dumps(
                {
                    "single_full": len(single_rows),
                    "demo": len(demo),
                    "single_rest": len(single_rest),
                    "single_rest_nonfrozen": len(single_rest_nonfrozen),
                    "multi_full": len(official_multi_rows),
                    "mapped_demo_exclusion": len(demo_multi_ids),
                    "multi_rest": len(multi_rest),
                    "common": len(common),
                },
                sort_keys=True,
            )
        )
    smoke = choose_smoke(common, smoke_cases)
    package_status = {name: bool(importlib.util.find_spec(name)) for name in REQUIRED_PACKAGES}
    source = cal_script.read_text(encoding="utf-8")
    payload = {
        "status": "corrected_protocol_cpu_audited_gpu_smoke_pending",
        "official_cal": {
            "repository": "NiuHechang/Calibrated_Adaptive_Length",
            "commit": EXPECTED_CAL_COMMIT,
            "license_file_present": (official_cal_root / "LICENSE").is_file(),
            "license_note": "upstream checkout has no LICENSE file; no upstream source is copied into paper artifacts",
            "llada_script": str(cal_script),
            "llada_script_sha256": sha256(cal_script),
            "model": "GSAI-ML/LLaDA-8B-Base",
            "prompt_contract": "official prefix + masked middle + suffix",
            "primary_multiline_config": PRIMARY_CAL_CONFIG,
            "official_fixed32_config": OFFICIAL_FIXED32_CONFIG,
            "project_fixed64_control": PROJECT_FIXED64_CONFIG,
            "forward_contract": "search_forwards + formal_decode_forwards; adapter records both and total per case",
            "seed": 42,
            "source_has_explicit_seed_cli": "seed" in source and "add_argument(\"--seed\"" in source,
        },
        "humaneval_infilling": {
            "repository": "openai/human-eval-infilling",
            "commit": EXPECTED_HUMANEVAL_COMMIT,
            "license": "MIT",
            "official_multiline_sha256": sha256(official_multi),
            "official_singleline_sha256": sha256(official_single),
            "current_multiline_sha256": sha256(current_dataset),
            "evaluator_path": str(evaluator),
            "evaluator_sha256": sha256(evaluator),
            "field_hash_task_count": len(comparison),
            "field_hash_mismatch_count": 0,
        },
        "population": {
            "singleline_full": 1033,
            "singleline_demo_seed42": 100,
            "singleline_rest": 933,
            "project_nonfrozen_singleline_rest": 838,
            "multiline_full": 5815,
            "mapped_demo_exclusion": 100,
            "multiline_rest": 5715,
            "project_nonfrozen_multiline_rest_common": 4990,
            "frozen_group_count": len(frozen),
            "official_cal_exact_population_name": "CAL-Rest non-frozen common population",
            "not_official_cal_exact": "5079 project non-frozen MultiLine spans",
        },
        "smoke": {
            "case_count": len(smoke),
            "base_task_deduplicated": True,
            "stratifier": "canonical_solution_utf8_bytes_offline_only quartiles",
            "demo_linked_rows_excluded": True,
            "execution_started": False,
            "performance_gate": False,
        },
        "frozen_controller_test": {"test_status": "sealed", "test_evaluation_count": 0},
        "environment": {
            "python": sys.version.split()[0],
            "required_package_present": package_status,
            "missing_required_packages": sorted(name for name, present in package_status.items() if not present),
        },
    }
    write_json(output_dir / "source_audit.json", payload)
    write_csv(output_dir / "evaluation_field_hash_comparison.csv", comparison)
    with (output_dir / "cal_rest_common_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in common:
            handle.write(json.dumps({key: row[key] for key in ("source_row_id", "task_id", "task_group", "completion_utf8_bytes")}, sort_keys=True) + "\n")
    with (output_dir / "smoke_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in smoke:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--official-cal-root", type=Path, required=True)
    result.add_argument("--humaneval-root", type=Path, required=True)
    result.add_argument("--current-dataset", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--smoke-cases", type=int, default=12)
    return result


def main(argv: Iterable[str] | None = None) -> None:
    args = parser().parse_args(argv)
    print(json.dumps(audit(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
