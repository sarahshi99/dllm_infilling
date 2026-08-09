from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from human_eval_infilling.evaluation import check_correctness

from .core import decode_frontier
from .manifests import read_jsonl, write_json, write_jsonl
from .native_reference import (
    TokenizerAdapter,
    load_official_generator_module,
    run_unmodified_native,
)
from .protocol import (
    DREAMON_COMMIT,
    EVALUATION_TIMEOUT_SECONDS,
    EVALUATION_WORKERS,
    EXPERIMENT_DIR,
    FIXED_FULL_ROWS,
    FIXED_MANIFEST_PATH,
    FIXED_METADATA_PATH,
    MAX_FORWARDS,
    MODEL_PATH,
    MODEL_REVISION,
    PILOT_GATE_PASSES,
    PILOT_MANIFEST_PATH,
    PILOT_METADATA_PATH,
    PILOT_ROWS,
    RESULTS_DIR,
    WIDTHS,
    config_hash,
    generation_config,
    sha256_file,
    stable_sample_seed,
    width_label,
)
from .schema import validate_result_row


DEVICE = "cuda:0"
PILOT_RESULTS_PATH = RESULTS_DIR / "pilot30_results.jsonl"
FIXED_RESULTS_PATH = RESULTS_DIR / "fixed_full_1000_results.jsonl"
EQUIVALENCE_RESULTS_PATH = RESULTS_DIR / "equivalence_results.jsonl"
PROGRESS_PATH = EXPERIMENT_DIR / "progress.json"
RUN_MANIFEST_PATH = EXPERIMENT_DIR / "run_manifest.json"
PILOT_SUMMARY_PATH = EXPERIMENT_DIR / "pilot30_summary.json"
FIXED_SUMMARY_PATH = EXPERIMENT_DIR / "fixed_full_1000_summary.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=EXPERIMENT_DIR, text=True).strip()


def manifest_id(metadata_path: Path) -> str:
    return str(json.loads(metadata_path.read_text(encoding="utf-8"))["manifest_id"])


def unique_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(row["manifest_id"]),
        str(row["sample_id"]),
        str(row["w"]),
        int(row["seed"]),
        str(row["config_hash"]),
    )


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, rows)
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    keys = [unique_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate unique result keys in {path}")
    return rows


def _compile_result(row: Mapping[str, Any], completion: str) -> dict[str, Any]:
    source = str(row["prompt"]) + completion + str(row["suffix"])
    try:
        tree = ast.parse(source)
        compile(tree, f"<{row['sample_id']}>", "exec")
        return {"compiled": True, "compile_error": None}
    except Exception as exc:
        return {"compiled": False, "compile_error": f"{type(exc).__name__}: {exc}"}


def _base_result(
    *, manifest: str, sample: Mapping[str, Any], width: int | None, seed: int
) -> dict[str, Any]:
    return {
        "manifest_id": manifest,
        "manifest_role": sample["manifest_role"],
        "sample_id": sample["sample_id"],
        "task_id": sample["task_id"],
        "base_task_id": sample["base_task_id"],
        "group_id": sample["group_id"],
        "w": width_label(width),
        "seed": seed,
        "config_hash": config_hash(),
        "model_path": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "dreamon_repository_commit": DREAMON_COMMIT,
        "pass_at_1": None,
        "evaluator_result": None,
        "evaluator_wall_time_seconds": None,
        "evaluable": None,
        "unevaluable_reason": None,
    }


def generate_result(
    *,
    model: Any,
    tokenizer: TokenizerAdapter,
    official_module: Any,
    manifest: str,
    sample: Mapping[str, Any],
    width: int | None,
) -> dict[str, Any]:
    seed = stable_sample_seed(str(sample["sample_id"]), width)
    row = _base_result(manifest=manifest, sample=sample, width=width, seed=seed)
    try:
        prefix_ids = tokenizer.encode(str(sample["prompt"]), add_bos=True, add_eos=False)
        suffix_ids = tokenizer.encode(str(sample["suffix"]), add_bos=False, add_eos=True)
        decoded = decode_frontier(
            model=model,
            tokenizer=tokenizer,
            sample_tokens_fn=official_module.sample_tokens,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            width=width,
            seed=seed,
            device=DEVICE,
        )
        row.update(decoded)
        row.update(_compile_result(sample, str(decoded["completion"])))
        row["completed"] = decoded["status"] == "completed"
    except Exception as exc:
        row.update(
            {
                "initial_mask_count": 64,
                "initial_dynamic_length": 64,
                "final_dynamic_length": None,
                "completion_token_ids": [],
                "completion": "",
                "status": "exception",
                "completed": False,
                "stop_reason": "exception",
                "forward_count": 0,
                "forward_sequence_lengths": [],
                "token_forwards": 0,
                "wall_time_seconds": 0.0,
                "gpu_time_seconds": None,
                "peak_cuda_memory_bytes": None,
                "step_trace": [
                    {
                        "step": 0,
                        "kind": "initial_state",
                        "dynamic_length": 64,
                        "unresolved_masks": 64,
                        "continuous_middle": True,
                    }
                ],
                "normal_token_count": 0,
                "expand_count": 0,
                "delete_action_count": 0,
                "single_point_delete_count": 0,
                "broadcast_delete_count": 0,
                "broadcast_delete_tail_lengths": [],
                "mask_noop_count": 0,
                "newline_token_count": 0,
                "newline_commit_count": 0,
                "physical_line_count": 0,
                "peak_unresolved_masks": 64,
                "final_unresolved_masks": None,
                "exact_cycle_count": 0,
                "repeated_state_count": 0,
                "oscillation_detected": False,
                "frontier_violation": 0,
                "compiled": False,
                "compile_error": None,
                "exception": f"{type(exc).__name__}: {exc}",
            }
        )
    return row


def update_progress(
    *,
    stage: str,
    completed: int,
    expected: int,
    started_at: float,
    current: str | None,
    status: str = "running",
) -> None:
    elapsed = max(time.time() - started_at, 1e-9)
    rate = completed / elapsed
    remaining = expected - completed
    eta = remaining / rate if rate > 0 else None
    write_json(
        PROGRESS_PATH,
        {
            "updated_at": utc_now(),
            "stage": stage,
            "status": status,
            "completed": completed,
            "expected": expected,
            "remaining": remaining,
            "elapsed_seconds": elapsed,
            "rows_per_second": rate,
            "eta_seconds": eta,
            "current": current,
        },
    )


def score_pending(path: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = load_results(path)
    problems = {str(row["sample_id"]): dict(row) for row in manifest_rows}
    pending_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("pass_at_1") is None and row.get("exception") is None
    ]

    def score(index: int) -> tuple[int, dict[str, Any], float]:
        row = rows[index]
        start = time.perf_counter()
        result = check_correctness(
            problems[str(row["sample_id"])],
            str(row["completion"]),
            EVALUATION_TIMEOUT_SECONDS,
        )
        return index, result, time.perf_counter() - start

    with concurrent.futures.ThreadPoolExecutor(max_workers=EVALUATION_WORKERS) as executor:
        futures = [executor.submit(score, index) for index in pending_indices]
        for future in concurrent.futures.as_completed(futures):
            index, result, elapsed = future.result()
            rows[index]["pass_at_1"] = bool(result["passed"])
            rows[index]["evaluator_result"] = result["result"]
            rows[index]["evaluator_wall_time_seconds"] = elapsed
            rows[index]["evaluable"] = True
            rows[index]["unevaluable_reason"] = None
    for row in rows:
        if row.get("exception") is not None:
            row["evaluable"] = False
            row["unevaluable_reason"] = "generation_exception"
        validate_result_row(row, scored=True)
    _atomic_write_jsonl(path, rows)
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], expected_per_width: int) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for width in [width_label(item) for item in WIDTHS]:
        selected = [row for row in rows if row["w"] == width]
        if not selected:
            continue
        if len(selected) != expected_per_width:
            raise AssertionError(f"w={width} expected {expected_per_width} rows, found {len(selected)}")
        summaries[width] = {
            "denominator": len(selected),
            "passed": sum(bool(row.get("pass_at_1")) for row in selected),
            "pass_rate": sum(bool(row.get("pass_at_1")) for row in selected) / len(selected),
            "compiled": sum(bool(row.get("compiled")) for row in selected),
            "compile_rate": sum(bool(row.get("compiled")) for row in selected) / len(selected),
            "completed": sum(bool(row.get("completed")) for row in selected),
            "completion_rate": sum(bool(row.get("completed")) for row in selected) / len(selected),
            "mean_forwards": sum(float(row.get("forward_count") or 0) for row in selected) / len(selected),
            "mean_wall_time_seconds": sum(float(row.get("wall_time_seconds") or 0) for row in selected) / len(selected),
            "frontier_violations": sum(int(row.get("frontier_violation") or 0) for row in selected),
        }
    return summaries


def run_stage(
    *,
    stage: str,
    manifest_path: Path,
    metadata_path: Path,
    results_path: Path,
    widths: Sequence[int | None],
    required_rows: int,
    model: Any,
    tokenizer: TokenizerAdapter,
    official_module: Any,
) -> list[dict[str, Any]]:
    samples = read_jsonl(manifest_path)
    if len(samples) != required_rows:
        raise AssertionError(f"{stage} manifest must contain {required_rows}, found {len(samples)}")
    if stage == "fixed-full-1000" and len(samples) != FIXED_FULL_ROWS:
        raise AssertionError("hard stop: fixed-full stage may run only a 1000-row manifest")
    current_rows = load_results(results_path)
    completed_keys = {unique_key(row) for row in current_rows}
    manifest = manifest_id(metadata_path)
    expected = required_rows * len(widths)
    started = time.time()
    update_progress(
        stage=stage,
        completed=len(current_rows),
        expected=expected,
        started_at=started,
        current=None,
    )
    for width in widths:
        for sample in samples:
            seed = stable_sample_seed(str(sample["sample_id"]), width)
            key = (manifest, str(sample["sample_id"]), width_label(width), seed, config_hash())
            if key in completed_keys:
                continue
            current = f"{sample['sample_id']} w={width_label(width)}"
            row = generate_result(
                model=model,
                tokenizer=tokenizer,
                official_module=official_module,
                manifest=manifest,
                sample=sample,
                width=width,
            )
            if unique_key(row) != key:
                raise AssertionError("generated row unique key differs from requested key")
            validate_result_row(row, scored=False)
            _append_jsonl(results_path, row)
            current_rows.append(row)
            completed_keys.add(key)
            update_progress(
                stage=stage,
                completed=len(current_rows),
                expected=expected,
                started_at=started,
                current=current,
            )
            print(
                f"[{stage}] {len(current_rows)}/{expected} {current} "
                f"status={row['status']} forwards={row['forward_count']} wall={row['wall_time_seconds']:.3f}s",
                flush=True,
            )
    rows = score_pending(results_path, samples)
    update_progress(
        stage=stage,
        completed=len(rows),
        expected=expected,
        started_at=started,
        current=None,
        status="completed",
    )
    return rows


def _trace_signature(trace: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, str]]:
    return [
        (int(row["commit_position"]), int(row["proposal_token_id"]), str(row["action"]))
        for row in trace
        if row.get("kind") == "commit"
    ]


def run_equivalence(
    *, model: Any, tokenizer: TokenizerAdapter, official_module: Any
) -> dict[str, Any]:
    candidates = read_jsonl(PILOT_MANIFEST_PATH) + read_jsonl(FIXED_MANIFEST_PATH)[:70]
    coverage = {
        "normal": False,
        "newline": False,
        "newline_followed_by_commit": False,
        "expand": False,
        "delete": False,
    }
    comparisons: list[dict[str, Any]] = []
    for sample in candidates:
        seed = stable_sample_seed(str(sample["sample_id"]), None)
        prefix_ids = tokenizer.encode(str(sample["prompt"]), add_bos=True, add_eos=False)
        suffix_ids = tokenizer.encode(str(sample["suffix"]), add_bos=False, add_eos=True)
        native = run_unmodified_native(
            model=model,
            tokenizer=tokenizer,
            official_module=official_module,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            seed=seed,
            device=DEVICE,
        )
        frontier = decode_frontier(
            model=model,
            tokenizer=tokenizer,
            sample_tokens_fn=official_module.sample_tokens,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            width=None,
            seed=seed,
            device=DEVICE,
        )
        native_signature = _trace_signature(native["step_trace"])
        frontier_signature = _trace_signature(frontier["step_trace"])
        checks = {
            "final_tokens": native["completion_token_ids"] == frontier["completion_token_ids"],
            "final_text": native["completion"] == frontier["completion"],
            "step_signature": native_signature == frontier_signature,
            "stop_reason": native["stop_reason"] == frontier["stop_reason"],
            "initial_64_masks": frontier["step_trace"][0]["unresolved_masks"] == 64,
            "frontier_zero": frontier["frontier_violation"] == 0,
        }
        comparison = {
            "sample_id": sample["sample_id"],
            "seed": seed,
            "checks": checks,
            "matched": all(checks.values()),
            "native_stop_reason": native["stop_reason"],
            "frontier_stop_reason": frontier["stop_reason"],
            "native_final_tokens": native["completion_token_ids"],
            "frontier_final_tokens": frontier["completion_token_ids"],
            "native_trace_signature": native_signature,
            "frontier_trace_signature": frontier_signature,
        }
        comparisons.append(comparison)
        _atomic_write_jsonl(EQUIVALENCE_RESULTS_PATH, comparisons)
        if not comparison["matched"]:
            write_json(
                EXPERIMENT_DIR / "equivalence_summary.json",
                {"status": "failed", "comparisons": len(comparisons), "first_mismatch": comparison},
            )
            raise AssertionError(f"w=inf native mismatch on {sample['sample_id']}: {checks}")
        for action_index, (_, proposal, action) in enumerate(native_signature):
            if action in coverage:
                coverage[action] = True
            if action == "normal":
                decoded = tokenizer.decode([proposal], skip_special_tokens=False)
                if "\n" in decoded or "\r" in decoded:
                    coverage["newline"] = True
                    if action_index + 1 < len(native_signature):
                        coverage["newline_followed_by_commit"] = True
        print(
            f"[equivalence] {len(comparisons)} {sample['sample_id']} coverage={coverage}",
            flush=True,
        )
        if len(comparisons) >= 5 and all(coverage.values()):
            break
    if len(comparisons) < 5 or not all(coverage.values()):
        raise AssertionError(
            f"equivalence behavior coverage incomplete after {len(comparisons)} samples: {coverage}"
        )
    summary = {
        "status": "passed",
        "samples": len(comparisons),
        "all_matched": True,
        "behavior_coverage": coverage,
        "native_source": str(Path('/home/shx/projects/dllm_infilling/DreamOn/eval/generator.py')),
        "native_commit": DREAMON_COMMIT,
        "config_hash": config_hash(),
    }
    write_json(EXPERIMENT_DIR / "equivalence_summary.json", summary)
    return summary


def write_run_manifest() -> None:
    pilot_meta = json.loads(PILOT_METADATA_PATH.read_text(encoding="utf-8"))
    fixed_meta = json.loads(FIXED_METADATA_PATH.read_text(encoding="utf-8"))
    value = {
        "created_at": utc_now(),
        "experiment": "Frontier-Gated DreamOn V0",
        "branch": git_output("branch", "--show-current"),
        "starting_head": "3eac683fcfdb2d8936e3d88416848b553ed4f69a",
        "runner_head_at_launch": git_output("rev-parse", "HEAD"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "generation_config": generation_config(),
        "config_hash": config_hash(),
        "pilot_manifest": pilot_meta,
        "fixed_full_1000_manifest": fixed_meta,
        "commands": {
            "equivalence": "python -m experiments.frontier_gated_dreamon.runner equivalence",
            "pilot": "python -m experiments.frontier_gated_dreamon.runner pilot",
            "fixed_full_auto": "python -m experiments.frontier_gated_dreamon.runner fixed-full-auto",
        },
    }
    write_json(RUN_MANIFEST_PATH, value)


def load_model() -> tuple[Any, TokenizerAdapter, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked DreamOn experiment")
    official_module = load_official_generator_module()
    hf_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    tokenizer = TokenizerAdapter(hf_tokenizer)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    return model, tokenizer, official_module


def run_pilot(model: Any, tokenizer: TokenizerAdapter, official_module: Any) -> dict[str, Any]:
    rows = run_stage(
        stage="pilot30",
        manifest_path=PILOT_MANIFEST_PATH,
        metadata_path=PILOT_METADATA_PATH,
        results_path=PILOT_RESULTS_PATH,
        widths=WIDTHS,
        required_rows=PILOT_ROWS,
        model=model,
        tokenizer=tokenizer,
        official_module=official_module,
    )
    summaries = summarize(rows, PILOT_ROWS)
    promoted = [width for width, value in summaries.items() if value["passed"] >= PILOT_GATE_PASSES]
    summary = {
        "stage": "pilot30",
        "role": "development/mechanism population",
        "manifest_id": manifest_id(PILOT_METADATA_PATH),
        "manifest_sha256": sha256_file(PILOT_MANIFEST_PATH),
        "gate": f"Pass@1 >= {PILOT_GATE_PASSES}/{PILOT_ROWS}",
        "widths": summaries,
        "promoted": promoted,
        "rows": len(rows),
    }
    write_json(PILOT_SUMMARY_PATH, summary)
    return summary


def run_fixed_auto(model: Any, tokenizer: TokenizerAdapter, official_module: Any) -> dict[str, Any]:
    if not PILOT_SUMMARY_PATH.exists():
        raise RuntimeError("pilot30_summary.json is required before fixed-full-auto")
    pilot = json.loads(PILOT_SUMMARY_PATH.read_text(encoding="utf-8"))
    promoted_labels = list(pilot["promoted"])
    label_to_width = {width_label(width): width for width in WIDTHS}
    widths = [label_to_width[label] for label in promoted_labels]
    finite_promoted = [width for width in widths if width is not None]
    if finite_promoted and None not in widths:
        widths.append(None)
    widths = [width for width in WIDTHS if width in widths]
    if not widths:
        summary = {
            "stage": "fixed-full-1000",
            "status": "not_triggered",
            "reason": "No pilot configuration reached 16/30.",
            "promoted": [],
            "rows": 0,
        }
        write_json(FIXED_SUMMARY_PATH, summary)
        return summary
    rows = run_stage(
        stage="fixed-full-1000",
        manifest_path=FIXED_MANIFEST_PATH,
        metadata_path=FIXED_METADATA_PATH,
        results_path=FIXED_RESULTS_PATH,
        widths=widths,
        required_rows=FIXED_FULL_ROWS,
        model=model,
        tokenizer=tokenizer,
        official_module=official_module,
    )
    summary = {
        "stage": "fixed-full-1000",
        "status": "completed",
        "role": "fixed-full-1000 development/validation population",
        "manifest_id": manifest_id(FIXED_METADATA_PATH),
        "manifest_sha256": sha256_file(FIXED_MANIFEST_PATH),
        "pilot_promoted": promoted_labels,
        "executed_widths": [width_label(width) for width in widths],
        "widths": summarize(rows, FIXED_FULL_ROWS),
        "rows": len(rows),
    }
    write_json(FIXED_SUMMARY_PATH, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["equivalence", "pilot", "fixed-full-auto", "all"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_run_manifest()
    model, tokenizer, official_module = load_model()
    if args.mode in {"equivalence", "all"}:
        print(json.dumps(run_equivalence(model=model, tokenizer=tokenizer, official_module=official_module), indent=2))
    if args.mode in {"pilot", "all"}:
        if not (EXPERIMENT_DIR / "equivalence_summary.json").exists():
            raise RuntimeError("equivalence gate has not passed")
        equivalence = json.loads((EXPERIMENT_DIR / "equivalence_summary.json").read_text())
        if equivalence.get("status") != "passed":
            raise RuntimeError("equivalence gate is not passed")
        print(json.dumps(run_pilot(model, tokenizer, official_module), indent=2))
    if args.mode in {"fixed-full-auto", "all"}:
        print(json.dumps(run_fixed_auto(model, tokenizer, official_module), indent=2))


if __name__ == "__main__":
    main()
