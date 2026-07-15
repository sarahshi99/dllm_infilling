#!/usr/bin/env python3
"""Audit pinned official CAL sources against the current MultiLine manifest.

This is intentionally CPU-only: it produces provenance and a 12-case technical
smoke manifest without loading a model or evaluating any functional outcome.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
EXPECTED_CAL_COMMIT = "741e8418a88a732b4c92812424d4f03cab1f7b1f"
EXPECTED_HUMANEVAL_COMMIT = "88062ff9859c875d04db115b698ed4b0f0395170"
REQUIRED_PACKAGES = ("torch", "transformers", "numpy", "scipy", "tqdm", "datasets", "accelerate")


def git_head(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


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


def task_group(task_id: str) -> str:
    match = re.search(r"HumanEval/(\d+)", task_id)
    if not match:
        raise ValueError(f"cannot derive HumanEval group from {task_id!r}")
    return f"HumanEval/{match.group(1)}"


def choose_smoke(rows: list[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ValueError("invalid smoke count")
    indexes = [round(index * (len(rows) - 1) / (count - 1)) if count > 1 else 0 for index in range(count)]
    selected: list[dict[str, Any]] = []
    for order, index in enumerate(indexes):
        row = rows[index]
        task_id = str(row["task_id"])
        selected.append(
            {
                "smoke_order": order,
                "source_row_id": index,
                "task_id": task_id,
                "task_group": task_group(task_id),
                "task_id_sha256": hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
            }
        )
    return selected


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    evaluator = humaneval_root / "human_eval_infilling" / "evaluate_functional_correctness.py"
    official_data = humaneval_root / "data" / "HumanEval-MultiLineInfilling.jsonl.gz"
    if not all(path.is_file() for path in (cal_script, evaluator, official_data, current_dataset)):
        raise RuntimeError("pinned source, evaluator, or data file is missing")

    official_rows = read_jsonl(official_data)
    current_rows = read_jsonl(current_dataset)
    official_ids = [str(row["task_id"]) for row in official_rows]
    current_ids = [str(row["task_id"]) for row in current_rows]
    if set(official_ids) != set(current_ids) or len(official_ids) != len(current_ids):
        raise RuntimeError("official/current MultiLine task mapping is not exact")

    frozen = set(json.loads((REPO / "analysis_outputs/grouped_split_20260702_accel2/test_tasks.json").read_text()))
    allowed = [row for row in current_rows if task_group(str(row["task_id"])) not in frozen]
    if len(allowed) != 5079:
        raise RuntimeError(f"expected 5079 non-frozen MultiLine rows, got {len(allowed)}")
    smoke = choose_smoke(allowed, smoke_cases)
    package_status = {name: bool(importlib.util.find_spec(name)) for name in REQUIRED_PACKAGES}
    source = cal_script.read_text(encoding="utf-8")
    audit_payload = {
        "status": "source_and_mapping_audited_gpu_execution_pending",
        "official_cal": {
            "repository": "https://github.com/NiuHechang/Calibrated_Adaptive_Length",
            "commit": EXPECTED_CAL_COMMIT,
            "license_file_present": (official_cal_root / "LICENSE").is_file(),
            "llada_script": str(cal_script),
            "llada_script_sha256": sha256(cal_script),
            "model_default": "GSAI-ML/LLaDA-8B-Base",
            "benchmark_default": "single-line; smoke contract uses multi-line exact mapping",
            "prompt_contract": "official prefix + masked middle + suffix",
            "initial_generation_length": 8,
            "max_generation_length": 64,
            "span": 1,
            "dstep": 4,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "use_bias_default": False,
            "oracle_default": False,
            "formal_decode_steps": "selected length when --steps is omitted",
            "forward_count_contract": "one model forward per CAL probe plus selected-length formal decode forwards",
            "seed_contract": "no explicit seed CLI/set_seed found in llada_code_infilling.py",
            "source_uses_official_evaluator": "evaluate_functional_correctness.py --benchmark_name=<split>",
            "source_contains_oracle_switch": "--oracle; prohibited for deployable comparison smoke",
            "source_text_mentions_gpu_ids": "--gpu_ids" in source,
        },
        "humaneval_infilling": {
            "repository": "https://github.com/openai/human-eval-infilling",
            "commit": EXPECTED_HUMANEVAL_COMMIT,
            "license": "MIT",
            "official_multiline_rows": len(official_rows),
            "current_multiline_rows": len(current_rows),
            "exact_task_id_intersection": len(set(official_ids).intersection(current_ids)),
            "current_non_frozen_allowed_rows": len(allowed),
            "official_data_sha256": sha256(official_data),
            "current_data_sha256": sha256(current_dataset),
            "evaluator_path": str(evaluator),
            "evaluator_sha256": sha256(evaluator),
        },
        "environment": {
            "python": sys.version.split()[0],
            "required_package_present": package_status,
            "missing_required_packages": sorted(name for name, present in package_status.items() if not present),
        },
        "smoke": {
            "case_count": len(smoke),
            "population": "exact_current_manifest_intersection_non_frozen_multiline",
            "execution_started": False,
            "performance_gate": False,
            "test_evaluation_count": 0,
        },
        "gpu_execution_blocker": "existing M1 GPU job predates CAL audit; preserve current job and do not violate requested serial ordering",
    }
    write_json(output_dir / "source_audit.json", audit_payload)
    with (output_dir / "smoke_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in smoke:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return audit_payload


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
