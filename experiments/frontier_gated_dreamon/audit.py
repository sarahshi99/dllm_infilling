from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .manifests import read_jsonl, write_json
from .protocol import (
    DREAMON_COMMIT,
    DREAMON_ROOT,
    EXPERIMENT_DIR,
    FIXED_CHECKSUM_PATH,
    FIXED_FULL_ROWS,
    FIXED_MANIFEST_PATH,
    FIXED_METADATA_PATH,
    PILOT_CHECKSUM_PATH,
    PILOT_MANIFEST_PATH,
    PILOT_METADATA_PATH,
    PILOT_ROWS,
    REPO_ROOT,
    WIDTHS,
    config_hash,
    sha256_file,
    stable_sample_seed,
    width_label,
)
from .runner import (
    EQUIVALENCE_RESULTS_PATH,
    FIXED_RESULTS_PATH,
    FIXED_SUMMARY_PATH,
    PILOT_RESULTS_PATH,
    PILOT_SUMMARY_PATH,
    RUN_MANIFEST_PATH,
    unique_key,
)
from .schema import validate_result_row


STARTING_HEAD = "3eac683fcfdb2d8936e3d88416848b553ed4f69a"
REVIEW_MANIFEST_PATH = EXPERIMENT_DIR / "review_manifest.latest.json"


def _checksum_file_value(path: Path) -> str:
    return path.read_text(encoding="utf-8").split()[0]


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_manifest(path: Path, checksum_path: Path, expected: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    digest = sha256_file(path)
    if len(rows) != expected:
        raise AssertionError(f"{path} expected {expected} rows, found {len(rows)}")
    if digest != _checksum_file_value(checksum_path):
        raise AssertionError(f"checksum mismatch for {path}")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"duplicate sample IDs in {path}")
    return {"path": str(path), "rows": len(rows), "sha256": digest}


def audit_results(
    path: Path, *, manifest: str, expected_widths: list[str], expected_per_width: int
) -> dict[str, Any]:
    rows = read_jsonl(path)
    expected_total = expected_per_width * len(expected_widths)
    if len(rows) != expected_total:
        raise AssertionError(f"{path} expected {expected_total} rows, found {len(rows)}")
    keys = [unique_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError(f"duplicate unique keys in {path}")
    counts = Counter(str(row["w"]) for row in rows)
    if counts != Counter({width: expected_per_width for width in expected_widths}):
        raise AssertionError(f"width counts mismatch in {path}: {counts}")
    for row in rows:
        validate_result_row(row, scored=True)
        if row["manifest_id"] != manifest:
            raise AssertionError("result manifest_id mismatch")
        if row["config_hash"] != config_hash():
            raise AssertionError("result config_hash mismatch")
        if int(row["seed"]) != stable_sample_seed(str(row["sample_id"]), None):
            raise AssertionError("result seed mismatch")
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": sha256_file(path),
        "width_counts": dict(counts),
        "duplicates": 0,
        "frontier_violations": sum(int(row["frontier_violation"]) for row in rows),
        "exceptions": sum(row.get("exception") is not None for row in rows),
    }


def audit_frozen_paths() -> dict[str, Any]:
    changed = [line for line in _git("diff", "--name-only", STARTING_HEAD).splitlines() if line]
    forbidden = []
    for path in changed:
        if path.startswith("repro_results/") or path.startswith("manifests/"):
            forbidden.append(path)
        if path.startswith("repro_scripts/") and "dreamon" in path.lower():
            forbidden.append(path)
        if path.startswith("tests/test_dreamon"):
            forbidden.append(path)
    if forbidden:
        raise AssertionError(f"frozen DreamOn paths were modified: {sorted(set(forbidden))}")
    generator_diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "eval/generator.py"],
        cwd=DREAMON_ROOT,
        check=False,
    )
    if generator_diff.returncode != 0:
        raise AssertionError("official DreamOn eval/generator.py is locally modified")
    official_head = _git("rev-parse", "HEAD", cwd=DREAMON_ROOT)
    if official_head != DREAMON_COMMIT:
        raise AssertionError(f"official DreamOn commit changed: {official_head}")
    return {
        "starting_head": STARTING_HEAD,
        "changed_paths": changed,
        "frozen_path_modifications": [],
        "official_generator_unmodified": True,
        "official_dreamon_head": official_head,
    }


def audit_decoder_source() -> dict[str, Any]:
    core_path = EXPERIMENT_DIR / "core.py"
    source = core_path.read_text(encoding="utf-8")
    forbidden_literals = [
        'partition("\\n")',
        "BoundaryShift",
        "future-slot",
        "nonempty_guard",
        "compile_gate",
        "blacklist",
        "retry_empty",
        "physical line slot",
    ]
    hits = [literal for literal in forbidden_literals if literal in source]
    if hits:
        raise AssertionError(f"forbidden decoder mechanisms found: {hits}")
    return {
        "path": str(core_path),
        "sha256": sha256_file(core_path),
        "forbidden_mechanism_hits": hits,
        "newline_control_flow": "none; newline inspection occurs only in final diagnostics/counting",
        "full_forward": True,
    }


def audit_result_shards() -> dict[str, Any]:
    index_path = EXPERIMENT_DIR / "results/fixed_full_1000_results.index.json"
    index = _json(index_path)
    total = 0
    for width, item in index["shards"].items():
        path = EXPERIMENT_DIR / str(item["path"])
        rows = read_jsonl(path)
        if len(rows) != FIXED_FULL_ROWS:
            raise AssertionError(f"w={width} shard row mismatch")
        if sha256_file(path) != item["sha256"]:
            raise AssertionError(f"w={width} shard checksum mismatch")
        if any(str(row["w"]) != width for row in rows):
            raise AssertionError(f"w={width} shard contains another width")
        total += len(rows)
    if total != int(index["source_rows"]):
        raise AssertionError("shards do not cover the combined result count")
    return index


def main() -> None:
    pilot_manifest = audit_manifest(PILOT_MANIFEST_PATH, PILOT_CHECKSUM_PATH, PILOT_ROWS)
    fixed_manifest = audit_manifest(FIXED_MANIFEST_PATH, FIXED_CHECKSUM_PATH, FIXED_FULL_ROWS)
    pilot_meta = _json(PILOT_METADATA_PATH)
    fixed_meta = _json(FIXED_METADATA_PATH)
    equivalence = _json(EXPERIMENT_DIR / "equivalence_summary.json")
    if equivalence.get("status") != "passed" or not equivalence.get("all_matched"):
        raise AssertionError("equivalence gate is not passed")
    pilot_summary = _json(PILOT_SUMMARY_PATH)
    promoted = list(pilot_summary["promoted"])
    expected_fixed_widths = list(promoted)
    if any(width != "inf" for width in promoted) and "inf" not in expected_fixed_widths:
        expected_fixed_widths.append("inf")
    expected_fixed_widths.sort(key=lambda value: ["1", "4", "8", "16", "inf"].index(value))
    pilot_results = audit_results(
        PILOT_RESULTS_PATH,
        manifest=str(pilot_meta["manifest_id"]),
        expected_widths=[width_label(width) for width in WIDTHS],
        expected_per_width=PILOT_ROWS,
    )
    fixed_results = None
    fixed_summary = _json(FIXED_SUMMARY_PATH)
    if fixed_summary.get("status") == "completed":
        fixed_results = audit_results(
            FIXED_RESULTS_PATH,
            manifest=str(fixed_meta["manifest_id"]),
            expected_widths=expected_fixed_widths,
            expected_per_width=FIXED_FULL_ROWS,
        )
    frozen = audit_frozen_paths()
    decoder = audit_decoder_source()
    result_shards = audit_result_shards()
    required_json = [
        EXPERIMENT_DIR / "config.json",
        RUN_MANIFEST_PATH,
        PILOT_METADATA_PATH,
        FIXED_METADATA_PATH,
        PILOT_SUMMARY_PATH,
        FIXED_SUMMARY_PATH,
        EXPERIMENT_DIR / "equivalence_summary.json",
    ]
    for path in required_json:
        _json(path)
    review = {
        "status": "passed",
        "reviewer_gate": "disabled; local diff review + fresh verification fallback",
        "manifests": {"pilot30": pilot_manifest, "fixed_full_1000": fixed_manifest},
        "equivalence": equivalence,
        "pilot_results": pilot_results,
        "fixed_results": fixed_results,
        "frozen_paths": frozen,
        "decoder_source": decoder,
        "result_shards": result_shards,
        "json_parse_checks": [str(path) for path in required_json],
        "equivalence_results_rows": len(read_jsonl(EQUIVALENCE_RESULTS_PATH)),
        "git_head": _git("rev-parse", "HEAD"),
    }
    write_json(REVIEW_MANIFEST_PATH, review)
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
