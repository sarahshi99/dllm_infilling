#!/usr/bin/env python3
"""Resumable adapter for pinned official LLaDA-CAL on the audited CAL-Rest set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.p1_official_cal_source_audit import (
    OFFICIAL_FIXED32_CONFIG,
    PRIMARY_CAL_CONFIG,
    PROJECT_FIXED64_CONFIG,
    EXPECTED_CAL_COMMIT,
    EXPECTED_HUMANEVAL_COMMIT,
    git_head,
    read_jsonl,
)
from expvision_dllm_clean.modeling import set_global_seed


MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
ARMS = ("official_cal_primary", "official_fixed32", "project_fixed64_internal")


class ForwardLedgerModel:
    """Transparent model proxy that records exact upstream decoder forwards."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.forward_input_tokens: list[int] = []

    @property
    def device(self) -> Any:
        return self._model.device

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        input_ids = args[0] if args else kwargs["input_ids"]
        self.forward_input_tokens.append(int(input_ids.shape[-1]))
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def arm_config(arm: str) -> dict[str, Any]:
    if arm == "official_cal_primary":
        return {**PRIMARY_CAL_CONFIG, "arm": arm}
    if arm == "official_fixed32":
        return dict(OFFICIAL_FIXED32_CONFIG)
    if arm == "project_fixed64_internal":
        return {
            "arm": arm,
            "initial_gen_length": 64,
            "span": 1,
            "dstep": -1,
            "max_gen_length": 64,
            "use_bias": False,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "oracle": False,
            "steps": 64,
            "block_length": 64,
            "comparison_boundary": PROJECT_FIXED64_CONFIG["arm"],
        }
    raise ValueError(f"unknown CAL arm {arm!r}")


def candidate_key(source_row_id: int, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return f"cal_rest_source_row={int(source_row_id)}|arm={arm}"


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_official_runtime(official_cal_root: Path, humaneval_root: Path) -> tuple[Any, Any]:
    if git_head(official_cal_root) != EXPECTED_CAL_COMMIT:
        raise RuntimeError("official CAL checkout is not pinned")
    if git_head(humaneval_root) != EXPECTED_HUMANEVAL_COMMIT:
        raise RuntimeError("HumanEval evaluator checkout is not pinned")
    for path in (str(official_cal_root), str(humaneval_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from llada_cal.llada_cal import generate  # type: ignore
    from human_eval_infilling.execution import check_correctness  # type: ignore

    return generate, check_correctness


def indexed_rows(dataset: Path) -> dict[int, dict[str, Any]]:
    return {index: row for index, row in enumerate(read_jsonl(dataset))}


def evaluator_problem_map(humaneval_root: Path) -> dict[str, dict[str, Any]]:
    import gzip

    path = humaneval_root / "data" / "HumanEval-MultiLineInfilling.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {str(row["task_id"]): row for row in (json.loads(line) for line in handle if line.strip())}


def decode_one(
    *,
    row: Mapping[str, Any],
    arm: str,
    tokenizer: Any,
    model: Any,
    generate: Any,
    check_correctness: Any,
    evaluator_problem: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    config = arm_config(arm)
    set_global_seed(seed)
    ledger = ForwardLedgerModel(model)
    prefix = str(row["prompt"])
    suffix = str(row["suffix"])
    encoded_prefix = tokenizer(prefix, add_special_tokens=False, padding=True, return_tensors="pt")
    encoded_suffix = tokenizer(suffix, add_special_tokens=False, padding=True, return_tensors="pt")
    prefix_ids = encoded_prefix["input_ids"].to(model.device)
    suffix_ids = encoded_suffix["input_ids"].to(model.device)
    prefix_mask = encoded_prefix["attention_mask"].to(model.device)
    suffix_mask = encoded_suffix["attention_mask"].to(model.device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output, search_forwards = generate(
        ledger,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        attention_mask=prefix_mask,
        suffix_attention_mask=suffix_mask,
        steps=config["steps"],
        gen_length=config["initial_gen_length"],
        block_length=config["block_length"],
        temperature=config["temperature"],
        cfg_scale=config["cfg_scale"],
        span=config["span"],
        max_gen_length=config["max_gen_length"],
        dstep=config["dstep"],
        use_bias=config["use_bias"],
    )
    decode_sec = time.perf_counter() - started
    full_ids = output[0]
    prefix_len, suffix_len = int(prefix_ids.shape[1]), int(suffix_ids.shape[1])
    middle_ids = full_ids[prefix_len : len(full_ids) - suffix_len] if suffix_len else full_ids[prefix_len:]
    completion = tokenizer.decode(middle_ids, skip_special_tokens=True)
    total_forwards = len(ledger.forward_input_tokens)
    formal_forwards = total_forwards - int(search_forwards)
    if formal_forwards < 0:
        raise RuntimeError("upstream CAL search forward count exceeds recorded forwards")
    evaluator_started = time.perf_counter()
    evaluation = check_correctness(evaluator_problem, completion, 3.0, 0)
    evaluator_sec = time.perf_counter() - evaluator_started
    return {
        "completion": completion,
        "selected_length": len(middle_ids),
        "search_forwards": int(search_forwards),
        "formal_decode_forwards": formal_forwards,
        "total_forwards": total_forwards,
        "search_token_budget": sum(ledger.forward_input_tokens[: int(search_forwards)]),
        "formal_decode_token_budget": sum(ledger.forward_input_tokens[int(search_forwards) :]),
        "token_budget": sum(ledger.forward_input_tokens),
        "wall_sec": decode_sec + evaluator_sec,
        "decode_sec": decode_sec,
        "evaluator_sec": evaluator_sec,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "passed": bool(evaluation["passed"]),
        "evaluator_result": evaluation["result"],
        "seed": int(seed),
        "determinism": "set_global_seed_before_each_case; official upstream has no seed CLI",
    }


def expected_keys(manifest: Sequence[Mapping[str, Any]], arm: str) -> set[str]:
    return {candidate_key(int(row["source_row_id"]), arm) for row in manifest}


def audit_rows(rows: Sequence[Mapping[str, Any]], expected: set[str], arm: str) -> dict[str, Any]:
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    errors = [row for row in rows if row.get("status") != "ok"]
    wrong_arm = [row for row in rows if row.get("arm") != arm]
    accounting_bad = [
        row
        for row in rows
        if row.get("status") == "ok"
        and int((row.get("metrics") or {}).get("total_forwards") or -1)
        != int((row.get("metrics") or {}).get("search_forwards") or 0)
        + int((row.get("metrics") or {}).get("formal_decode_forwards") or 0)
    ]
    return {
        "passed": observed == expected and all(value == 1 for value in counts.values()) and not errors and not wrong_arm and not accounting_bad,
        "expected_count": len(expected),
        "observed_count": len(rows),
        "missing_count": len(expected - observed),
        "extra_count": len(observed - expected),
        "duplicate_count": sum(value > 1 for value in counts.values()),
        "error_count": len(errors),
        "wrong_arm_count": len(wrong_arm),
        "forward_accounting_error_count": len(accounting_bad),
    }


def run(args: argparse.Namespace) -> int:
    official_cal_root = Path(args.official_cal_root).resolve()
    humaneval_root = Path(args.humaneval_root).resolve()
    current_dataset = Path(args.current_dataset).resolve()
    manifest_path = Path(args.manifest_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    arm = str(args.arm)
    manifest = read_jsonl(manifest_path)
    if args.auto_full:
        if len(manifest) != 4990:
            raise RuntimeError("official CAL auto-full requires the audited 4990-row CAL-Rest common manifest")
    elif len(manifest) != int(args.smoke_cases):
        raise RuntimeError("smoke manifest size does not match --smoke-cases")
    rows_by_source = indexed_rows(current_dataset)
    runtime_manifest = json.loads((manifest_path.parent / "source_audit.json").read_text(encoding="utf-8"))
    if runtime_manifest["population"]["project_nonfrozen_multiline_rest_common"] != 4990:
        raise RuntimeError("source audit does not certify corrected common population")
    generate, check_correctness = load_official_runtime(official_cal_root, humaneval_root)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    evaluator_problems = evaluator_problem_map(humaneval_root)
    raw_path = output_dir / f"{arm}_raw.jsonl"
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    counts = Counter(str(row.get("candidate_key") or "") for row in existing)
    if any(value > 1 for value in counts.values()):
        raise RuntimeError("official CAL resume refuses duplicate candidate keys")
    completed = set(counts)
    for item in manifest:
        source_row_id = int(item["source_row_id"])
        key = candidate_key(source_row_id, arm)
        if key in completed:
            continue
        source = rows_by_source[source_row_id]
        started = time.perf_counter()
        try:
            result = decode_one(
                row=source,
                arm=arm,
                tokenizer=tokenizer,
                model=model,
                generate=generate,
                check_correctness=check_correctness,
                evaluator_problem=evaluator_problems[str(source["task_id"])],
                seed=int(args.seed),
            )
            row = {
                "candidate_key": key,
                "arm": arm,
                "source_row_id": source_row_id,
                "task_id": str(source["task_id"]),
                "task_group": str(item["task_group"]),
                "status": "ok",
                "passed": result.pop("passed"),
                "completion": result.pop("completion"),
                "evaluator_result": result.pop("evaluator_result"),
                "metrics": result,
                "official_cal_commit": EXPECTED_CAL_COMMIT,
                "evaluator_commit": EXPECTED_HUMANEVAL_COMMIT,
                "config": arm_config(arm),
                "gpu": torch.cuda.get_device_name(),
            }
        except Exception as exc:
            row = {
                "candidate_key": key,
                "arm": arm,
                "source_row_id": source_row_id,
                "task_id": str(source.get("task_id") or ""),
                "task_group": str(item.get("task_group") or ""),
                "status": "error",
                "passed": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:240],
                "failure_traceback": traceback.format_exc(),
                "metrics": {"wall_sec": time.perf_counter() - started},
            }
        append_jsonl(raw_path, row)
        completed.add(key)
    rows = read_jsonl(raw_path)
    audit = audit_rows(rows, expected_keys(manifest, arm), arm)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if audit["passed"] else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--official-cal-root", required=True)
    root.add_argument("--humaneval-root", required=True)
    root.add_argument("--current-dataset", required=True)
    root.add_argument("--manifest-jsonl", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--arm", choices=ARMS, required=True)
    root.add_argument("--seed", type=int, default=42)
    root.add_argument("--smoke-cases", type=int, default=12)
    root.add_argument("--auto-full", action="store_true")
    return root


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
