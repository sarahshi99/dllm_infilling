#!/usr/bin/env python3
"""One authorized A/B/C/D/E diagnosis, with resumable per-case transactions."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.dreamon_singleline_adapter import (
    CountingModel, OfficialHFTokenizerWrapper, derive_row_seed,
    ensure_modeling_rope_utils_available,
)
from experiments.dreamon_singleline_order_parallel import (
    GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER, K2_LEGACY_ACTIONS,
    K2_SEQUENTIAL_ACTIONS, decode_with_policy,
)
from experiments.run_dreamon_singleline_order_parallel import load_generator_module
from experiments.lrdllm_dreamcoder_adapter import (
    atomic_write_json, append_jsonl, load_dataset_rows, load_pinned_evaluator, selection_view,
)

POLICIES = {"global_confidence": GLOBAL_CONFIDENCE, "left_to_right_frontier": LEFT_TO_RIGHT_FRONTIER}
ARMS = tuple((policy, arm) for policy in POLICIES for arm in "ABCDE")
CODE_PATHS = (
    "experiments/run_dreamon_markov_k2_diagnosis.py",
    "experiments/dreamon_singleline_order_parallel.py",
    "experiments/dreamon_singleline_adapter.py",
    "experiments/run_dreamon_singleline_order_parallel.py",
    "experiments/lrdllm_dreamcoder_adapter.py",
    "experiments/dreamon_markov_head_training.py",
    "experiments/train_dreamon_markov_head.py",
    "analysis/analyze_dreamon_markov_k2_diagnosis.py",
)
COMPARE_FIELDS = (
    "completion", "completion_token_ids", "action_trace", "forward_count",
    "base_forward_count", "recompute_forward_count", "outer_round_count",
    "normal_token_count", "expand_count", "delete_count", "markov_enabled_count",
    "recompute_enabled_count", "adjacent_opportunity_count", "nonadjacent_fallback_count",
    "corrected_structural_stop_count", "structural_stop_count",
)


def stable_smoke(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: str(row["task_id"]))
    return [ordered[i * (len(ordered) - 1) // 15] for i in range(16)]


def rotated_arms(row_index: int) -> list[tuple[str, str]]:
    offset = row_index % len(ARMS)
    return list(ARMS[offset:] + ARMS[:offset])


def gpu_snapshot() -> dict:
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    return {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpus": query, "compute_processes": processes}


def require_idle(snapshot: dict, *, own_pid: int | None = None) -> None:
    gpus = snapshot["gpus"].splitlines()
    if len(gpus) != 1 or "H200" not in gpus[0]:
        raise RuntimeError("requires the single currently authorized H200")
    for line in snapshot["compute_processes"].splitlines():
        if line.strip() and int(line.split(",")[0]) != own_pid:
            raise RuntimeError("H200 is occupied by other compute processes; no GPU job launched/continued")


def open_store(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS cases (policy TEXT, arm TEXT, task_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(policy,arm,task_id))")
    db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.commit()
    return db


def save_case(db: sqlite3.Connection, row: dict) -> None:
    # One transaction is the success boundary. Failed cases never become completed.
    started = time.perf_counter()
    key = row["policy"], row["arm"], row["task_id"]
    with db:
        db.execute("INSERT INTO cases VALUES (?,?,?,?)", (*key, json.dumps(row, ensure_ascii=False)))
    row["file_write_sec"] = time.perf_counter() - started
    with db:
        db.execute("UPDATE cases SET payload=? WHERE policy=? AND arm=? AND task_id=?",
                   (json.dumps(row, ensure_ascii=False), *key))


def load_cases(db: sqlite3.Connection) -> list[dict]:
    return [json.loads(row[0]) for row in db.execute("SELECT payload FROM cases ORDER BY policy,arm,task_id")]


def seed_row(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not 0 < args.budget_gpu_hours <= 12:
        raise ValueError("GPU budget must be positive and at most 12 hours")
    original_path = Path(args.original_config).resolve()
    original = json.loads(original_path.read_text())
    protocol = original["official_protocol"]
    if (original["seed"], protocol["temperature"], protocol["top_p"], protocol["dtype"], protocol["batch_size"]) != (42, 0.2, 0.9, "bf16", 1):
        raise RuntimeError("original protocol differs from the authorized experiment")
    registry_path = REPO / "analysis_outputs/dreamon_markov_head_training_20260910_v2/checkpoint_registry.json"
    entry, = [r for r in json.loads(registry_path.read_text())
              if (r["head"], r["phase"], r["label"]) == ("kl", "full", "best")]
    checkpoint = Path(entry["path"])
    if not checkpoint.is_file() or checkpoint.stat().st_size != entry["size"]:
        raise RuntimeError("registered v2 KL full/best checkpoint missing or size differs")
    if str(checkpoint) != original["markov_checkpoint"] or original["markov_lambda"] != 1:
        raise RuntimeError("original selected head/lambda differs")
    rows = load_dataset_rows(Path(original["evaluator_root"]), "HumanEval-SingleLineInfilling")
    if len(rows) != 1033 or len({r["task_id"] for r in rows}) != 1033 or len({r["task_id"].split("/")[2] for r in rows}) != 164:
        raise RuntimeError("expected exact 1033 rows / 164 base tasks")
    # Compare ID membership to the reviewed compact paired table, not historical outcomes.
    import csv
    with (original_path.parent / "paired_deltas.csv").open() as handle:
        previous_ids = {r["task_id"] for r in csv.DictReader(handle)}
    if previous_ids != {r["task_id"] for r in rows}:
        raise RuntimeError("development population ID membership differs from reviewed run")
    source = Path(original["source_root"])
    source_revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    train_config = json.loads((registry_path.parent / "run_config.json").read_text())
    if source_revision != train_config["source_revision"] or Path(original["model_snapshot"]).name != train_config["model_revision"]:
        raise RuntimeError("model/source version mismatch")
    if not (Path(original["model_snapshot"]) / "config.json").is_file():
        raise RuntimeError("model snapshot missing")
    manifest = {
        "status": "prepared_not_run", "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "code_versions": {path: subprocess.check_output(["git", "rev-parse", "HEAD:" + path], cwd=REPO, text=True).strip() for path in CODE_PATHS},
        "original_config_path": str(original_path), "original_config": original,
        "source_revision": source_revision, "model_revision": train_config["model_revision"],
        "tokenizer_snapshot": original["model_snapshot"], "head_registry_path": str(registry_path),
        "head_registry_entry": entry, "head_version_verification": "registry version and size; existing digest reused, no new large-file hash scan",
        "head_rank": train_config["rank"], "head_lambda": 1.0,
        "population": "reused development/mechanism population, not independent sealed test",
        "population_count": 1033, "base_task_count": 164,
        "seed": 42, "row_seed_derivation": "existing derive_row_seed(42, task_id)",
        "right_sample_seed": "(row_seed + 1000003*step + 10007*left_absolute + right_absolute) % (2**63-1)",
        "rng_isolation": "fork_rng around head/recompute and right sampling; no additional generation seeds",
        "dtype": "bfloat16 model, float32 rank-256 heads (same as previous)", "batch_size": 1,
        "device": "cuda:0", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "budget_gpu_hours": args.budget_gpu_hours,
        "budget_accounting": "cumulative exclusive GPU session wall including loading/warmup/smoke/evaluator/writes; conservative upper bound",
        "arms": [{"policy": p, "arm": a, "k": 1 if a == "A" else 2,
                  "right_mode": {"A": "none", "B": "none", "C": "zero_output_same_head", "D": "v2_kl", "E": "fresh_model_fixed_right"}[a]}
                 for p, a in ARMS],
        "order": "per-row rotate ten configurations by row_index % 10",
        "smoke_task_ids": [r["task_id"] for r in stable_smoke(rows)],
        "smoke_selection": "16 evenly spaced lexicographically sorted stable IDs, no outcome selection",
        "generation_parameters": {k: protocol[k] for k in ("min_gen_len", "max_gen_len", "max_tokens", "steps", "temperature", "top_p", "top_k")},
        "expand_budget": protocol["max_gen_len"],
        "expand_budget_source": "pinned eval/generator.py: self.expand_budget = self.max_gen_len",
        "evaluation_timeout_sec": 3.0, "evaluator_args": [3.0, 0],
        "evaluation_args_source": "reviewed experiments/run_dreamon_singleline_order_parallel.py: check_correctness(source_row, completion, 3.0, 0)",
        "bootstrap": {"cluster": "base task", "replicates": 10000, "seed": 20261013,
                      "weights": ["line", "base_task_equal"], "scope": "conditional on single generation seed 42"},
        "command": shlex.join([sys.executable, *sys.argv]),
        "reviewer_gate_disabled": True,
    }
    path = out / "run_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("original_config", "head_registry_entry", "arms", "smoke_task_ids", "generation_parameters", "budget_gpu_hours", "code_versions"):
            if previous[key] != manifest[key]:
                raise RuntimeError(f"resume manifest differs: {key}; use a separate output directory for changed code/protocol")
        current_commit = manifest["code_commit"]
        manifest = previous
        manifest.setdefault("resume_commits", []).append(current_commit)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *CODE_PATHS], cwd=REPO, text=True).strip()
    if dirty:
        raise RuntimeError("commit diagnosis source changes before generating a versioned run")
    atomic_write_json(path, manifest)
    return manifest, rows


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_dir).resolve()
    manifest, rows = prepare(args)
    snapshot = gpu_snapshot()
    append_jsonl(out / "gpu_background.jsonl", snapshot)
    try:
        require_idle(snapshot)
    except RuntimeError as exc:
        manifest.update(status="blocked_gpu_busy", blocker=str(exc), gpu_snapshot=snapshot,
                        full_batch_estimate_sec=None, estimate_status="16-line GPU check not run: GPU busy")
        atomic_write_json(out / "run_manifest.json", manifest)
        print(str(exc), flush=True)
        return 3
    # Lock is per physical GPU, shared across invocations; never kill another process.
    lock = open("/tmp/dllm_infilling_h200_diagnosis.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require_idle(gpu_snapshot())
    db = open_store(out / "per_case.sqlite")
    prior_elapsed = float((db.execute("SELECT value FROM state WHERE key='gpu_session_sec'").fetchone() or [0])[0])
    session_started = time.perf_counter()
    def persist_budget() -> float:
        used = prior_elapsed + time.perf_counter() - session_started
        with db:
            db.execute("INSERT OR REPLACE INTO state VALUES ('gpu_session_sec',?)", (str(used),))
        return used
    def budget_check() -> None:
        if persist_budget() >= args.budget_gpu_hours * 3600:
            raise RuntimeError("12 GPU-hour cumulative execution budget reached")
    try:
        config = manifest["original_config"]
        source_module = load_generator_module(Path(config["source_root"]))
        ensure_modeling_rope_utils_available()
        from transformers import AutoModel, AutoTokenizer
        from experiments.train_dreamon_markov_head import make_head
        load_started = time.perf_counter()
        tokenizer = OfficialHFTokenizerWrapper(AutoTokenizer.from_pretrained(
            config["model_snapshot"], trust_remote_code=True, local_files_only=True))
        model = CountingModel(AutoModel.from_pretrained(
            config["model_snapshot"], trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16).to("cuda:0").eval()).eval()
        checkpoint = torch.load(manifest["head_registry_entry"]["path"], map_location="cpu", weights_only=False)
        trained_head = make_head(model.model.config, "cuda:0")
        trained_head.load_state_dict(checkpoint["head"])
        zero_head = copy.deepcopy(trained_head)
        with torch.no_grad():
            zero_head.output.weight.zero_()
        for head in (trained_head, zero_head):
            head.eval()
            for parameter in head.parameters():
                parameter.requires_grad_(False)
        del checkpoint
        torch.cuda.synchronize()
        loading_sec = time.perf_counter() - load_started
        check_correctness, evaluator_info = load_pinned_evaluator(Path(config["evaluator_root"]))
        if evaluator_info != config["evaluator"]:
            raise RuntimeError("evaluator version differs from original config")
        manifest.update(status="running_smoke", actual_device=torch.cuda.get_device_name(0),
                        torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        model_load_sec=loading_sec, actual_head_dtype=str(next(trained_head.parameters()).dtype))
        def decode(row: dict, policy: str, arm: str, *, diagnostics: bool, actions: bool = False) -> tuple[dict, float]:
            safe = selection_view(row)
            prefix = tokenizer.encode(safe["prefix"], add_bos=True, add_eos=False)
            suffix = tokenizer.encode(safe["suffix"], add_bos=False, add_eos=True)
            row_seed = derive_row_seed(42, row["task_id"])
            seed_row(row_seed)
            before = model.forward_calls
            torch.cuda.synchronize()
            started = time.perf_counter()
            decoded = decode_with_policy(
                model=model, tokenizer=tokenizer, sample_tokens_fn=source_module.sample_tokens,
                prefix_ids=prefix, suffix_ids=suffix, **manifest["generation_parameters"],
                expand_budget=manifest["expand_budget"], requested_k=1 if arm == "A" else 2,
                policy=POLICIES[policy], device="cuda:0",
                markov_head={"C": zero_head, "D": trained_head}.get(arm),
                markov_lambda=1.0 if arm in "CD" else 0.0,
                right_recompute=arm == "E", diagnostics=diagnostics, capture_actions=actions,
                k2_action_mode=K2_LEGACY_ACTIONS if arm == "A" else K2_SEQUENTIAL_ACTIONS,
                markov_sampling_seed=row_seed,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if model.forward_calls - before != decoded["forward_count"]:
                raise RuntimeError("actual model invocation count differs from decoder counters")
            return decoded, elapsed
        warm_started = time.perf_counter()
        # Warm every compute path outside measurement; deterministic row reset follows.
        for policy, arm in ARMS:
            budget_check()
            decode(stable_smoke(rows)[0], policy, arm, diagnostics=False)
        manifest["warmup_sec"] = time.perf_counter() - warm_started
        smoke_path = out / "smoke_checks.json"
        if not smoke_path.exists() or json.loads(smoke_path.read_text())["status"] != "passed":
            checks, sample_costs = [], []
            for index, row in enumerate(stable_smoke(rows)):
                background = gpu_snapshot()
                append_jsonl(out / "gpu_background.jsonl", background)
                require_idle(background, own_pid=os.getpid())
                for policy, arm in rotated_arms(index):
                    budget_check()
                    on, _ = decode(row, policy, arm, diagnostics=True, actions=True)
                    off, elapsed = decode(row, policy, arm, diagnostics=False, actions=True)
                    differences = [key for key in COMPARE_FIELDS if on[key] != off[key]]
                    if differences:
                        atomic_write_json(smoke_path, {"status": "failed", "task_id": row["task_id"],
                                                     "policy": policy, "arm": arm, "differences": differences})
                        raise RuntimeError("unexplained diagnostics on/off protocol difference")
                    eval_started = time.perf_counter()
                    check_correctness(row, off["completion"], 3.0, 0)
                    sample_costs.append(elapsed + time.perf_counter() - eval_started)
                    checks.append({"task_id": row["task_id"], "policy": policy, "arm": arm,
                                   "generation_sec": elapsed, "forward_count": off["forward_count"],
                                   "base_forward_count": off["base_forward_count"],
                                   "recompute_forward_count": off["recompute_forward_count"]})
            estimate = sum(sample_costs) / 16 * 1033
            atomic_write_json(smoke_path, {"status": "passed", "checks": checks, "count": len(checks),
                                          "full_batch_estimate_sec": estimate, "estimate_margin": 1.5})
        else:
            estimate = json.loads(smoke_path.read_text())["full_batch_estimate_sec"]
        completed_rows = load_cases(db)
        completed = {(r["policy"], r["arm"], r["task_id"]) for r in completed_rows}
        expected = {(p, a, r["task_id"]) for p, a in ARMS for r in rows}
        if not completed <= expected or any(r["row_seed"] != derive_row_seed(42, r["task_id"]) for r in completed_rows):
            raise RuntimeError("resume case identity/seed mismatch")
        remaining_estimate = estimate * (len(expected) - len(completed)) / len(expected)
        manifest.update(status="running_full", full_batch_estimate_sec=estimate,
                        remaining_estimate_with_margin_sec=1.5 * remaining_estimate)
        atomic_write_json(out / "run_manifest.json", manifest)
        if persist_budget() + 1.5 * remaining_estimate > args.budget_gpu_hours * 3600:
            raise RuntimeError("estimated remaining batch exceeds cumulative GPU-hour budget")
        for index, row in enumerate(rows):
            snapshot = gpu_snapshot()
            append_jsonl(out / "gpu_background.jsonl", snapshot)
            require_idle(snapshot, own_pid=os.getpid())
            for policy, arm in rotated_arms(index):
                if (policy, arm, row["task_id"]) in completed:
                    continue
                budget_check()
                try:
                    decoded, elapsed = decode(row, policy, arm, diagnostics=False)
                    completion = decoded.pop("completion")
                    for field in ("step_trace", "markov_transitions", "global_neighbor_observations", "replay_transitions", "replay_exclusion_counts", "action_trace"):
                        decoded.pop(field)
                    eval_started = time.perf_counter()
                    evaluation = check_correctness(row, completion, 3.0, 0)
                    evaluator_sec = time.perf_counter() - eval_started
                    save_case(db, {"policy": policy, "arm": arm, "task_id": row["task_id"],
                                   "base_task_id": row["task_id"].split("/")[2], "line_id": row["task_id"].split("/")[3],
                                   "seed": 42, "row_seed": derive_row_seed(42, row["task_id"]),
                                   "code_commit": manifest["code_commit"], "output": completion,
                                   "passed": bool(evaluation["passed"]), "failure_reason": evaluation["result"],
                                   "generation_sec": elapsed, "evaluator_sec": evaluator_sec,
                                   "output_chars": len(completion), "output_tokens": len(decoded["completion_token_ids"]),
                                   **decoded})
                    completed.add((policy, arm, row["task_id"]))
                except Exception as exc:
                    append_jsonl(out / "failure_journal.jsonl", {"policy": policy, "arm": arm, "task_id": row["task_id"],
                                                               "error": str(exc), "traceback": traceback.format_exc()})
                    raise
                atomic_write_json(out / "progress.json", {"status": "running", "completed": len(completed), "expected": 10330,
                                                         "gpu_session_sec": persist_budget()})
        require_idle(gpu_snapshot(), own_pid=os.getpid())
        manifest["status"] = "completed_generation"
        atomic_write_json(out / "run_manifest.json", manifest)
        from analysis.analyze_dreamon_markov_k2_diagnosis import analyze
        analyze(out)
        return 0
    except Exception as exc:
        manifest.update(status="blocked", blocker=str(exc))
        atomic_write_json(out / "run_manifest.json", manifest)
        append_jsonl(out / "runner_failures.jsonl", {"error": str(exc), "traceback": traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
        return 2
    finally:
        used = persist_budget()
        manifest["gpu_session_sec"] = used
        atomic_write_json(out / "run_manifest.json", manifest)
        db.close()
        lock.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--original-config", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--budget-gpu-hours", type=float, default=12)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
