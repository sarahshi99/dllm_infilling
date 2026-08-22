#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

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
)
from experiments.dreamon_singleline_order_parallel import (
    GLOBAL_CONFIDENCE,
    LEFT_TO_RIGHT_FRONTIER,
    decode_with_policy,
    validate_resume_rows,
)
from experiments.lrdllm_dreamcoder_adapter import (
    append_jsonl,
    atomic_write_json,
    load_dataset_rows,
    load_pinned_evaluator,
    selection_view,
)


VARIANTS = {
    "C1": (GLOBAL_CONFIDENCE, 1),
    "C2": (GLOBAL_CONFIDENCE, 2),
    "C4": (GLOBAL_CONFIDENCE, 4),
    "L1": (LEFT_TO_RIGHT_FRONTIER, 1),
    "L2": (LEFT_TO_RIGHT_FRONTIER, 2),
    "L4": (LEFT_TO_RIGHT_FRONTIER, 4),
}


def load_generator_module(source_root: Path) -> Any:
    if "omegaconf" not in sys.modules:
        shim = ModuleType("omegaconf")
        shim.OmegaConf = object()  # type: ignore[attr-defined]
        shim.__spec__ = importlib.machinery.ModuleSpec("omegaconf", loader=None)
        sys.modules["omegaconf"] = shim
    path = source_root / "eval" / "generator.py"
    spec = importlib.util.spec_from_file_location("dreamon_order_parallel_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load DreamOn generator")
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


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "per_case_results.jsonl"
    traces_path = output_dir / "per_step_trace.jsonl"
    progress_path = output_dir / "progress.json"
    existing = read_jsonl(cases_path)
    completed_by_variant = {
        variant: validate_resume_rows(
            [
                {"case_key": str(row["task_id"]), "variant": row["variant"], "seed": row["seed"]}
                for row in existing
                if row.get("variant") == variant
            ],
            variant=variant,
            seed=int(args.seed),
        )
        for variant in VARIANTS
    }
    evaluator_root = Path(args.evaluator_root).resolve()
    rows = load_dataset_rows(evaluator_root, "HumanEval-SingleLineInfilling")
    if len(rows) != 1033 or len({str(row["task_id"]) for row in rows}) != 1033:
        raise RuntimeError("DreamOn official SingleLine loader must provide 1033 unique tasks")
    selected_rows = rows[: int(args.case_limit)] if args.case_limit else rows
    check_correctness, evaluator_info = load_pinned_evaluator(evaluator_root)
    source_module = load_generator_module(Path(args.source_root).resolve())
    ensure_modeling_rope_utils_available()
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(args.model_snapshot).resolve()
    tokenizer = OfficialHFTokenizerWrapper(
        AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    )
    model = CountingModel(
        AutoModel.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16
        ).to(args.device).eval()
    ).eval()
    config = protocol_config(64)
    atomic_write_json(
        output_dir / "run_config.json",
        {
            "population": "HumanEval-Infilling SingleLine full 1033 development population",
            "source_root": str(Path(args.source_root).resolve()),
            "evaluator_root": str(evaluator_root),
            "model_snapshot": str(snapshot),
            "device": args.device,
            "seed": int(args.seed),
            "variants": {name: {"policy": policy, "requested_k": k} for name, (policy, k) in VARIANTS.items()},
            "official_protocol": config,
            "evaluator": evaluator_info,
        },
    )
    started = time.perf_counter()
    expected = len(selected_rows) * len(VARIANTS)
    for row_index, source_row in enumerate(selected_rows):
        safe = selection_view(source_row)
        prefix = tokenizer.encode(safe["prefix"], add_bos=True, add_eos=False)
        suffix = tokenizer.encode(safe["suffix"], add_bos=False, add_eos=True)
        for variant, (policy, requested_k) in VARIANTS.items():
            task_id = str(source_row["task_id"])
            if task_id in completed_by_variant[variant]:
                continue
            row_seed = derive_row_seed(int(args.seed), task_id)
            torch.manual_seed(row_seed)
            torch.cuda.manual_seed_all(row_seed)
            started_case = time.perf_counter()
            try:
                decoded = decode_with_policy(
                    model=model,
                    tokenizer=tokenizer,
                    sample_tokens_fn=source_module.sample_tokens,
                    prefix_ids=prefix,
                    suffix_ids=suffix,
                    min_gen_len=64,
                    max_gen_len=64,
                    max_tokens=2048,
                    steps=256,
                    expand_budget=64,
                    temperature=0.2,
                    top_p=0.9,
                    top_k=None,
                    requested_k=requested_k,
                    policy=policy,
                    device=args.device,
                )
                completion = str(decoded.pop("completion"))
                evaluation = check_correctness(source_row, completion, 3.0, 0)
                trace = decoded.pop("step_trace")
                append_jsonl(
                    cases_path,
                    {
                        "case_key": case_key(task_id, variant),
                        "task_id": task_id,
                        "task_group": task_id.split("/")[2],
                        "variant": variant,
                        "seed": int(args.seed),
                        "row_seed": row_seed,
                        "output": completion,
                        "passed": bool(evaluation["passed"]),
                        "failure_reason": str(evaluation["result"]),
                        "wall_sec": time.perf_counter() - started_case,
                        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                        **decoded,
                    },
                )
                for step in trace:
                    append_jsonl(traces_path, {"case_key": case_key(task_id, variant), "task_id": task_id, "variant": variant, "seed": int(args.seed), **step})
                completed_by_variant[variant].add(task_id)
            except Exception as exc:
                append_jsonl(output_dir / "failure_journal.jsonl", {"task_id": task_id, "variant": variant, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
                return 2
            completed = sum(len(keys) for keys in completed_by_variant.values())
            atomic_write_json(progress_path, {"status": "running", "completed": completed, "expected": expected, "row_index": row_index, "elapsed_sec": time.perf_counter() - started})
    atomic_write_json(progress_path, {"status": "completed", "completed": sum(len(keys) for keys in completed_by_variant.values()), "expected": expected, "elapsed_sec": time.perf_counter() - started})
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
