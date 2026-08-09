from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .manifests import base_task_id, proportional_sample, read_jsonl, write_json, write_jsonl
from .protocol import (
    DATA_PATH,
    FIXED_CHECKSUM_PATH,
    FIXED_FULL_ROWS,
    FIXED_MANIFEST_PATH,
    FIXED_METADATA_PATH,
    FIXED_SELECTION_SEED,
    PILOT_CHECKSUM_PATH,
    PILOT_MANIFEST_PATH,
    PILOT_METADATA_PATH,
    PILOT_ROWS,
    REPO_ROOT,
    config_hash,
    generation_config,
    sha256_bytes,
    sha256_file,
)


FROZEN_PILOT_RESULTS = (
    REPO_ROOT / "repro_results/dreamon_progressive_v2_hard_v2_pilot30/predictions.jsonl"
)


def _manifest_row(source: Mapping[str, Any], role: str, selection_order: int) -> dict[str, Any]:
    task_id = str(source["task_id"])
    return {
        "task_id": task_id,
        "sample_id": task_id,
        "base_task_id": base_task_id(task_id),
        "group_id": base_task_id(task_id),
        "manifest_role": role,
        "selection_order": selection_order,
        "prompt": source["prompt"],
        "suffix": source["suffix"],
        "canonical_solution": source["canonical_solution"],
        "entry_point": source["entry_point"],
        "test": source["test"],
    }


def _seal(path: Path, rows: list[dict[str, Any]], checksum_path: Path) -> str:
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    digest = sha256_bytes(serialized)
    if path.exists() and path.read_bytes() != serialized:
        raise RuntimeError(f"sealed manifest differs from requested content: {path}")
    if not path.exists():
        write_jsonl(path, rows)
    checksum_line = f"{digest}  {path.name}\n"
    if checksum_path.exists() and checksum_path.read_text(encoding="utf-8") != checksum_line:
        raise RuntimeError(f"sealed checksum differs: {checksum_path}")
    checksum_path.write_text(checksum_line, encoding="utf-8")
    return digest


def build() -> dict[str, Any]:
    write_json(
        Path(__file__).resolve().parent / "config.json",
        {"generation": generation_config(), "config_hash": config_hash()},
    )
    population = read_jsonl(DATA_PATH)
    if len(population) != 5815:
        raise AssertionError(f"expected 5815 source rows, found {len(population)}")
    by_id = {str(row["task_id"]): row for row in population}
    if len(by_id) != len(population):
        raise AssertionError("source population contains duplicate task IDs")

    frozen_rows = read_jsonl(FROZEN_PILOT_RESULTS)
    pilot_ids = [str(row["task_id"]) for row in frozen_rows]
    if len(pilot_ids) != PILOT_ROWS or len(set(pilot_ids)) != PILOT_ROWS:
        raise AssertionError("frozen mechanism pilot must contain exactly 30 unique rows")
    missing = sorted(set(pilot_ids) - set(by_id))
    if missing:
        raise AssertionError(f"pilot IDs missing from 5815 population: {missing}")
    pilot_rows = [
        _manifest_row(by_id[task_id], "pilot30_development_mechanism", order)
        for order, task_id in enumerate(pilot_ids)
    ]
    pilot_sha = _seal(PILOT_MANIFEST_PATH, pilot_rows, PILOT_CHECKSUM_PATH)
    pilot_meta = {
        "manifest_id": f"pilot30:{pilot_sha}",
        "role": "development/mechanism population",
        "not_frozen_test": True,
        "rows": PILOT_ROWS,
        "construction": "Exact task-ID reuse of the previously frozen DreamOn V2-Hard-v2 mechanism Pilot-30; no selection used Frontier-Gated DreamOn outcomes.",
        "source_results_path": str(FROZEN_PILOT_RESULTS),
        "source_results_sha256": sha256_file(FROZEN_PILOT_RESULTS),
        "population_path": str(DATA_PATH),
        "population_rows": len(population),
        "population_sha256": sha256_file(DATA_PATH),
        "manifest_sha256": pilot_sha,
    }
    write_json(PILOT_METADATA_PATH, pilot_meta)

    pilot_set = set(pilot_ids)
    remaining = [row for row in population if str(row["task_id"]) not in pilot_set]
    fixed_sources, allocation = proportional_sample(
        remaining, total=FIXED_FULL_ROWS, seed=FIXED_SELECTION_SEED
    )
    fixed_rows = [
        _manifest_row(source, "fixed_full_1000_development_validation", order)
        for order, source in enumerate(fixed_sources)
    ]
    fixed_sha = _seal(FIXED_MANIFEST_PATH, fixed_rows, FIXED_CHECKSUM_PATH)
    fixed_meta = {
        "manifest_id": f"fixed_full_1000:{fixed_sha}",
        "role": "fixed-full-1000 development/validation population",
        "not_official_5815_full": True,
        "not_frozen_test": True,
        "rows": FIXED_FULL_ROWS,
        "source_population_rows": len(population),
        "eligible_after_pilot_exclusion": len(remaining),
        "excluded_pilot_rows": len(pilot_set),
        "construction": "Base-task proportional allocation by largest remainder, then stable SHA256 order within task; no reference correctness or Frontier-Gated DreamOn result was read for selection.",
        "selection_seed": FIXED_SELECTION_SEED,
        "population_path": str(DATA_PATH),
        "population_sha256": sha256_file(DATA_PATH),
        "pilot_manifest_sha256": pilot_sha,
        "allocation": allocation,
        "manifest_sha256": fixed_sha,
    }
    write_json(FIXED_METADATA_PATH, fixed_meta)
    return {"pilot": pilot_meta, "fixed_full_1000": fixed_meta}


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
