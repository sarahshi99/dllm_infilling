#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.dreamon_singleline_adapter import (
    CountingModel,
    OfficialHFTokenizerWrapper,
    derive_row_seed,
    ensure_modeling_rope_utils_available,
    protocol_config,
    source_provenance,
)
from experiments.dreamon_singleline_order_parallel import (
    GLOBAL_CONFIDENCE,
    LEFT_TO_RIGHT_FRONTIER,
    decode_with_policy,
)
from experiments.lrdllm_dreamcoder_adapter import (
    append_jsonl,
    atomic_write_json,
    load_dataset_rows,
    load_pinned_evaluator,
    selection_view,
)


VARIANTS = {
    "C1": {
        "diagnostic_name_zh": "全局置信度逐词刷新诊断",
        "policy": GLOBAL_CONFIDENCE,
        "requested_k": 1,
        "scientific_purpose_zh": "服务于未来 global top-K 内连续短链 + Markov 设计",
    },
    "L1": {
        "diagnostic_name_zh": "固定左前沿逐词刷新诊断",
        "policy": LEFT_TO_RIGHT_FRONTIER,
        "requested_k": 1,
        "scientific_purpose_zh": "服务于未来强制固定左到右 + Markov 设计",
    },
}


def load_generator_module(source_root: Path) -> Any:
    if "omegaconf" not in sys.modules:
        shim = ModuleType("omegaconf")
        shim.OmegaConf = object()  # type: ignore[attr-defined]
        shim.__spec__ = importlib.machinery.ModuleSpec("omegaconf", loader=None)
        sys.modules["omegaconf"] = shim
    path = source_root / "eval" / "generator.py"
    spec = importlib.util.spec_from_file_location("dreamon_markov_premise_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned DreamOn generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def case_key(task_id: str, variant: str) -> str:
    return f"{task_id}|{variant}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rewrite_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def repair_resume_journals(
    cases_path: Path, traces_path: Path, transitions_path: Path
) -> tuple[list[dict[str, Any]], set[str]]:
    cases = read_jsonl(cases_path)
    keys = [str(row.get("case_key") or "") for row in cases]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise RuntimeError("resume case journal has blank or duplicate case keys")
    completed = set(keys)
    for path in (traces_path, transitions_path):
        if not path.exists():
            continue
        rows = read_jsonl(path)
        retained = [row for row in rows if str(row.get("case_key") or "") in completed]
        if len(retained) != len(rows):
            rewrite_jsonl(path, retained)
    return cases, completed


def trajectory_signature(trace: list[dict[str, Any]]) -> str:
    trajectory = [
        {
            "selected_positions": row["selected_positions"],
            "selected_token_ids": row["selected_token_ids"],
            "selected_token_types": row["selected_token_types"],
            "executed_expand_count": row["executed_expand_count"],
            "executed_delete_count": row["executed_delete_count"],
            "canvas_state_after": row["canvas_state_after"],
        }
        for row in trace
    ]
    payload = json.dumps(trajectory, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "per_case_results.jsonl"
    traces_path = output_dir / "per_step_trace.jsonl"
    transitions_path = output_dir / "per_transition_diagnostics.jsonl"
    progress_path = output_dir / "progress.json"
    cases, completed = repair_resume_journals(cases_path, traces_path, transitions_path)

    source_root = Path(args.source_root).resolve()
    evaluator_root = Path(args.evaluator_root).resolve()
    snapshot = Path(args.model_snapshot).resolve()
    rows = load_dataset_rows(evaluator_root, "HumanEval-SingleLineInfilling")
    if len(rows) != 1033 or len({str(row["task_id"]) for row in rows}) != 1033:
        raise RuntimeError("DreamOn official SingleLine loader must provide 1033 unique tasks")
    selected_rows = rows[: int(args.case_limit)] if args.case_limit else rows
    expected_keys = {
        case_key(str(row["task_id"]), variant) for row in selected_rows for variant in VARIANTS
    }
    extra_completed = completed - expected_keys
    if extra_completed:
        raise RuntimeError(f"resume output contains {len(extra_completed)} unexpected case keys")
    for row in cases:
        variant = str(row.get("variant"))
        if variant not in VARIANTS or int(row.get("seed", -1)) != int(args.seed):
            raise RuntimeError("resume output has a different variant or seed")

    check_correctness, evaluator_info = load_pinned_evaluator(evaluator_root)
    source_module = load_generator_module(source_root)
    ensure_modeling_rope_utils_available()
    from transformers import AutoModel, AutoTokenizer

    tokenizer = OfficialHFTokenizerWrapper(
        AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    )
    model = CountingModel(
        AutoModel.from_pretrained(
            snapshot,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        .to(args.device)
        .eval()
    ).eval()
    atomic_write_json(
        output_dir / "run_config.json",
        {
            "run_date_utc": "2026-08-30",
            "population": "official DreamOn HumanEval-Infilling SingleLine development/full-allowed",
            "population_rows": len(selected_rows),
            "full_population_rows": 1033,
            "task_group_count_expected": 164,
            "source_root": str(source_root),
            "source_provenance": source_provenance(source_root),
            "evaluator_root": str(evaluator_root),
            "evaluator": evaluator_info,
            "model_snapshot": str(snapshot),
            "device": str(args.device),
            "seed": int(args.seed),
            "variants": VARIANTS,
            "official_protocol": protocol_config(64),
            "reference_usage": "offline diagnostics only; never used by selection, sampling, actions, canvas, or output",
            "markov_head_training": False,
        },
    )

    started = time.perf_counter()
    expected = len(expected_keys)
    for row_index, source_row in enumerate(selected_rows):
        safe = selection_view(source_row)
        prefix = tokenizer.encode(safe["prefix"], add_bos=True, add_eos=False)
        suffix = tokenizer.encode(safe["suffix"], add_bos=False, add_eos=True)
        reference = tokenizer.encode(
            str(source_row["canonical_solution"]), add_bos=False, add_eos=False
        )
        task_id = str(source_row["task_id"])
        for variant, variant_config in VARIANTS.items():
            key = case_key(task_id, variant)
            if key in completed:
                continue
            row_seed = derive_row_seed(int(args.seed), task_id)
            torch.manual_seed(row_seed)
            torch.cuda.manual_seed_all(row_seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started_case = time.perf_counter()
            try:
                decoded = decode_with_policy(
                    model=model,
                    tokenizer=tokenizer,
                    sample_tokens_fn=source_module.sample_tokens,
                    prefix_ids=prefix,
                    suffix_ids=suffix,
                    reference_ids=reference,
                    min_gen_len=64,
                    max_gen_len=64,
                    max_tokens=2048,
                    steps=256,
                    expand_budget=64,
                    temperature=0.2,
                    top_p=0.9,
                    top_k=None,
                    requested_k=1,
                    policy=str(variant_config["policy"]),
                    device=str(args.device),
                )
                completion = str(decoded.pop("completion"))
                evaluation = check_correctness(source_row, completion, 3.0, 0)
                trace = decoded.pop("step_trace")
                transitions = decoded.pop("markov_transitions")
                neighbors = decoded.pop("global_neighbor_observations")
                common = {
                    "case_key": key,
                    "task_id": task_id,
                    "task_group": task_id.split("/")[2],
                    "variant": variant,
                    "diagnostic_name_zh": variant_config["diagnostic_name_zh"],
                    "seed": int(args.seed),
                }
                for step in trace:
                    append_jsonl(traces_path, {**common, **step})
                for transition in transitions:
                    append_jsonl(transitions_path, {**common, **transition})
                for observation in neighbors:
                    append_jsonl(transitions_path, {**common, **observation})
                eligible = [row for row in transitions if row.get("transition_eligible")]
                append_jsonl(
                    cases_path,
                    {
                        **common,
                        "row_seed": row_seed,
                        "output": completion,
                        "passed": bool(evaluation["passed"]),
                        "failure_reason": str(evaluation["result"]),
                        "wall_sec": time.perf_counter() - started_case,
                        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
                        if torch.cuda.is_available()
                        else 0,
                        "reference_token_count": len(reference),
                        "trajectory_sha256": trajectory_signature(trace),
                        "eligible_transition_count": len(eligible),
                        "reference_evaluable_transition_count": sum(
                            bool(row.get("reference_evaluable")) for row in eligible
                        ),
                        "global_neighbor_observation_count": len(neighbors),
                        **decoded,
                    },
                )
                completed.add(key)
            except Exception as exc:
                append_jsonl(
                    output_dir / "failure_journal.jsonl",
                    {
                        "case_key": key,
                        "task_id": task_id,
                        "variant": variant,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                atomic_write_json(
                    progress_path,
                    {
                        "status": "incomplete",
                        "completed": len(completed),
                        "expected": expected,
                        "last_error_case_key": key,
                    },
                )
                return 2
            atomic_write_json(
                progress_path,
                {
                    "status": "running",
                    "completed": len(completed),
                    "expected": expected,
                    "row_index": row_index,
                    "elapsed_sec": time.perf_counter() - started,
                },
            )
    atomic_write_json(
        progress_path,
        {
            "status": "generation_completed",
            "completed": len(completed),
            "expected": expected,
            "elapsed_sec": time.perf_counter() - started,
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--evaluator-root", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-limit", type=int)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
