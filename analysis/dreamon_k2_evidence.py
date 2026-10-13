"""Portable evidence for an existing K2 run. No model, evaluator or CUDA imports."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIELDS = (
    "policy", "arm", "task_id", "base_task_id", "line_id", "seed", "row_seed", "code_commit",
    "output", "passed", "failure_reason", "generation_sec", "evaluator_sec", "file_write_sec",
    "output_chars", "output_tokens", "completion_token_ids", "final_dynamic_length",
    "final_remaining_masks", "forward_count", "base_forward_count", "recompute_forward_count",
    "outer_round_count", "recompute_enabled_count", "structural_stop_count", "normal_token_count",
    "expand_count", "delete_count", "markov_enabled_count", "adjacent_opportunity_count",
    "nonadjacent_fallback_count", "corrected_structural_stop_count",
)
MISSING_OBSERVATIONS = (
    "case_start_utc", "case_end_utc", "evaluator_start_utc", "evaluator_end_utc",
    "evaluator_cpu_time_sec", "host_cpu_load", "evaluator_child_exitcode",
    "evaluator_exception_type", "evaluator_traceback", "evaluator_timeout_origin",
    "generated_action_sequence", "head_eligible_but_not_taken_reason",
)


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def case_key(row: dict) -> tuple[str, str, str]:
    return row["policy"], row["arm"], row["task_id"]


def validate_rows(rows: list[dict]) -> None:
    counts = Counter(case_key(r) for r in rows)
    if any(n != 1 for n in counts.values()):
        raise ValueError("duplicate (policy, arm, task_id) in source evidence")
    for row in rows:
        absent = set(FIELDS) - row.keys()
        if absent:
            raise ValueError(f"missing recorded fields for {case_key(row)}: {sorted(absent)}")
        if not isinstance(row["passed"], bool):
            raise ValueError("passed must be an original boolean")
        if len(row["completion_token_ids"]) != row["output_tokens"] or len(row["output"]) != row["output_chars"]:
            raise ValueError("output length disagrees with retained text/tokens")
        if row["base_forward_count"] + row["recompute_forward_count"] != row["forward_count"]:
            raise ValueError("forward accounting mismatch")
        if row["outer_round_count"] != row["base_forward_count"]:
            raise ValueError("round accounting mismatch")
        for name in ("generation_sec", "evaluator_sec", "file_write_sec"):
            if not isinstance(row[name], (float, int)) or row[name] < 0:
                raise ValueError(f"invalid {name}")


def source_rows(result_dir: Path) -> tuple[list[dict], dict]:
    db_path = result_dir / "per_case.sqlite"
    jsonl_path = result_dir / "per_case.jsonl"
    sources = {}
    if db_path.exists():
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            rows = []
            for policy, arm, task_id, payload in db.execute("SELECT policy,arm,task_id,payload FROM cases ORDER BY policy,arm,task_id"):
                row = json.loads(payload)
                if (policy, arm, task_id) != case_key(row):
                    raise ValueError("SQLite primary key and payload identity disagree")
                rows.append(row)
        sources["sqlite_read_only_rows"] = len(rows)
    elif jsonl_path.exists():
        rows = read_jsonl(jsonl_path)
    else:
        raise FileNotFoundError("original SQLite/per_case records missing; cannot export summaries as cases")
    validate_rows(rows)
    if jsonl_path.exists():
        other = read_jsonl(jsonl_path)
        validate_rows(other)
        if {case_key(r): r for r in rows} != {case_key(r): r for r in other}:
            raise ValueError("SQLite and original per_case JSONL disagree; keep both and investigate")
        sources["existing_jsonl_rows"] = len(other)
        sources["sqlite_jsonl_all_fields_equal"] = db_path.exists()
    return sorted(rows, key=case_key), sources


def anonymize_gpu(snapshot: dict, own_pid: int | None) -> dict:
    parts = [s.strip() for s in snapshot["gpus"].split(",")]
    own, other = [], []
    for line in snapshot["compute_processes"].splitlines():
        if not line.strip():
            continue
        fields = [s.strip() for s in line.split(",")]
        # Only PID comparison is used; names/paths/PIDs never leave this function.
        memory = None if fields[-1] == "[N/A]" else float(fields[-1])
        (own if own_pid is not None and int(fields[0]) == own_pid else other).append(memory)
    def total(values):
        return sum(values) if all(v is not None for v in values) else None
    return {"time_utc": snapshot["time_utc"], "gpu_index": int(parts[0]), "gpu_name": parts[2],
            "memory_used_mib": float(parts[3]), "memory_total_mib": float(parts[4]),
            "utilization_gpu_pct": float(parts[5]), "temperature_c": float(parts[6]),
            "own_task_count": len(own), "other_task_count": len(other),
            "own_memory_mib": total(own), "other_memory_mib": total(other)}


def load_bundle(evidence_dir: Path) -> tuple[list[dict], dict, set[str], dict, list[dict]]:
    meta = json.loads((evidence_dir / "evidence_manifest.json").read_text())
    def local(name):
        value = Path(meta["files"][name])
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("evidence files must be relative inside the bundle")
        return evidence_dir / value
    rows = read_jsonl(local("cases"))
    validate_rows(rows)
    population = json.loads(local("population").read_text())
    if len(population) != len(set(population)):
        raise ValueError("duplicate expected population ID")
    smoke = json.loads(local("smoke").read_text())
    failures = read_jsonl(local("failure_events"))
    return rows, meta["run_context"], set(population), smoke, failures


def export_evidence(result_dir: Path, *, source_head: str) -> dict:
    rows, sources = source_rows(result_dir)
    manifest = json.loads((result_dir / "run_manifest.json").read_text())
    target = result_dir / "evidence"
    target.mkdir(exist_ok=True)
    # Full original fields and float precision; gzip alone makes this compact.
    with (target / "per_case.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    # Already published fixed development IDs, not a fresh dataset/test opening.
    previous = REPO / "analysis_outputs/dreamon_markov_k2_full_20261011_v1/paired_deltas.csv"
    with previous.open() as handle:
        population = sorted({r["task_id"] for r in csv.DictReader(handle)})
    (target / "population.json").write_text(json.dumps(population, indent=2) + "\n")
    shutil.copyfile(result_dir / "smoke_checks.json", target / "smoke_checks.json")
    failures_path = result_dir / "failure_journal.jsonl"
    failures = read_jsonl(failures_path) if failures_path.exists() else []
    with (target / "failure_events.jsonl").open("w") as handle:
        for failure in failures:
            # Error/tracebacks may carry unrelated absolute paths; preserve only
            # identity and type needed for unrecovered-failure accounting.
            handle.write(json.dumps({k: failure.get(k) for k in ("policy", "arm", "task_id", "error_type")}) + "\n")
    launch = result_dir / "background_launch.json"
    own_pid = json.loads(launch.read_text())["python_pid"] if launch.exists() else None
    background = result_dir / "gpu_background.jsonl"
    gpu_rows = [anonymize_gpu(r, own_pid) for r in read_jsonl(background)] if background.exists() else []
    with (target / "gpu_background.csv").open("w", newline="") as handle:
        fields = list(gpu_rows[0]) if gpu_rows else ["time_utc", "other_task_count"]
        writer = csv.DictWriter(handle, fieldnames=["sample_index", *fields], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"sample_index": i, **r} for i, r in enumerate(gpu_rows))
    keep = ("code_commit", "branch", "seed", "smoke_task_ids", "dtype", "actual_device",
            "allow_shared_gpu", "status", "model_revision", "source_revision", "head_rank", "head_lambda",
            "generation_parameters", "right_sample_seed", "bootstrap", "gpu_session_sec", "evaluation_timeout_sec")
    context = {k: manifest[k] for k in keep}
    context["command"] = "python analysis/analyze_dreamon_markov_k2_diagnosis.py --result-dir RESULT_DIR --output-dir RECOMPUTED_DIR"
    context["evaluator"] = manifest["original_config"]["evaluator"]
    context["head"] = {k: manifest["head_registry_entry"][k] for k in ("head", "phase", "label", "sha256", "size")}
    meta = {"format_version": 1, "source_head_at_export": source_head,
            "source": "existing success payloads; SQLite read-only; no generation/evaluation",
            "files": {"cases": "per_case.jsonl.gz", "population": "population.json", "smoke": "smoke_checks.json",
                      "gpu": "gpu_background.csv", "failure_events": "failure_events.jsonl"},
            "run_context": context, "source_crosscheck": sources,
            "field_presence_counts": {key: sum(key in r for r in rows) for key in FIELDS},
            "missing_original_observations": list(MISSING_OBSERVATIONS),
            "failure_journal_present_at_export": failures_path.exists(),
            "failure_journal_note": "absent journal means zero observed events at export; cannot prove a historical file was never deleted",
            "gpu_samples": len(gpu_rows), "gpu_own_pid_available": own_pid is not None,
            "gpu_sampling_limit": "snapshots at row boundaries/preload/smoke, not continuous; no per-case/evaluator UTC times to align",
            "output_evidence": "complete original text AND token IDs; no synthesized digest or inferred labels",
            "file_write_timing_limit": "recorded first SQLite insert/commit only; following UPDATE commit and journal/progress writes excluded",
            "population_note": "1033 reused development/mechanism lines / 164 base tasks; seed42 conditional, no new sealed test"}
    (target / "evidence_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return meta


def redact_manifest_gpu_snapshots(result_dir: Path) -> list[str]:
    """Keep raw originals local, replace committed process paths by numeric load."""
    launch = result_dir / "background_launch.json"
    own_pid = json.loads(launch.read_text())["python_pid"] if launch.exists() else None
    changed = []
    for path in (result_dir / "run_manifest.json", result_dir / "before_shared_gpu_authorization/run_manifest.json"):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        altered = False
        for key in ("initial_gpu_snapshot", "gpu_snapshot"):
            if isinstance(data.get(key), dict) and "compute_processes" in data[key]:
                local = result_dir / ".private_original_manifests" / path.relative_to(result_dir)
                local.parent.mkdir(parents=True, exist_ok=True)
                if not local.exists():
                    shutil.copyfile(path, local)
                data[key] = anonymize_gpu(data[key], own_pid)
                altered = True
        if altered:
            data["gpu_snapshot_privacy"] = "process identities/paths removed; numeric load retained; original kept locally, not in portable bundle"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            changed.append(str(path.relative_to(result_dir)))
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    meta = export_evidence(args.result_dir.resolve(), source_head=args.source_head)
    print(json.dumps(meta["source_crosscheck"]))
