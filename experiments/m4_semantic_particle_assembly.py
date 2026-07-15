#!/usr/bin/env python3
"""M4 Semantic Particle Assembly V0 from an existing eight-candidate bank."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.phase6_abductive_bridge_v1 import analyze_candidate, extract_backward_obligations
from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask
from expvision_dllm_clean.modeling import load_model_and_tokenizer, resolve_mask_token_id, set_global_seed
from expvision_dllm_clean.verifier import run_verifier_stack
from experiments.m2_constraint_homotopy import append_jsonl, load_frozen_groups, read_jsonl, sha256_text, write_csv, write_json
from experiments.phase6_remask import decode_fixed_canvas_state
from experiments.method_population_schedule import (
    RANDOMSPANLIGHT_ALLOWED_CASES,
    SMOKE_CASES,
    population_schedule,
    require_randomspanlight_full,
)


MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
SOURCE_CONFIG = "HumanEval-RandomSpanInfillingLight"
CANVAS_TOKENS = 64
REPAIR_STEPS = 64
SEED = 0
METHODS = ("m4_best_single_particle", "m4_assembly_without_repair", "m4_assembly_with_repair")
DEPLOYABLE_CANDIDATE_FIELDS = {"middle_text", "canvas_tokens", "seed", "final_token_confidences", "prefix_text", "suffix_text"}


def cfg_for() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = MODEL_PATH
    cfg.model.torch_dtype = "bfloat16"
    cfg.model.device_map = "auto"
    cfg.data.dataset_subset = SOURCE_CONFIG
    cfg.decode.mask_length_source = "fixed"
    cfg.decode.fixed_mask_length = CANVAS_TOKENS
    cfg.decode.total_steps = REPAIR_STEPS
    cfg.decode.seed = SEED
    cfg.decode.save_step_traces = False
    cfg.decode.save_full_text_per_step = False
    return cfg


def method_key(row_key: str, method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"Unknown M4 method {method!r}")
    return f"{row_key}|{method}|repair_steps={REPAIR_STEPS}|seed={SEED}"


def _source_offset(source: str, lineno: int, col: int) -> int:
    return sum(len(line) for line in source.splitlines(keepends=True)[: lineno - 1]) + col


def _defs_uses(node: ast.AST) -> tuple[set[str], set[str]]:
    definitions: set[str] = set()
    uses: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if isinstance(child.ctx, (ast.Store, ast.Del)):
                definitions.add(child.id)
            elif isinstance(child.ctx, ast.Load):
                uses.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.add(child.name)
        elif isinstance(child, ast.arg):
            definitions.add(child.arg)
    return definitions, uses - definitions


def candidate_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Strict view used by M4 assembly; outcomes/reference/IDs never enter."""
    return {
        "middle_text": str(row.get("middle_text") or ""),
        "canvas_tokens": int(row["canvas_tokens"]),
        "seed": int(row["seed"]),
        "final_token_confidences": list(row.get("final_token_confidences") or []),
        "prefix_text": str(row.get("prefix_text") or ""),
        "suffix_text": str(row.get("suffix_text") or ""),
    }


def candidate_fragments(candidate: Mapping[str, Any], ordinal: int) -> list[dict[str, Any]]:
    prefix, middle, suffix = str(candidate["prefix_text"]), str(candidate["middle_text"]), str(candidate["suffix_text"])
    source = prefix + middle + suffix
    begin, end = len(prefix), len(prefix) + len(middle)
    fragments: list[dict[str, Any]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        definitions = set(re.findall(r"(?m)^\s*([A-Za-z_]\w*)\s*(?::[^=\n]+)?=", middle))
        uses = set(re.findall(r"\b[A-Za-z_]\w*\b", middle)) - definitions
        return [{"kind": "basic_block", "candidate_ordinal": ordinal, "start": 0, "end": len(middle), "text": middle, "definitions": sorted(definitions), "uses": sorted(uses)}]
    statements: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt) or not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        left = _source_offset(source, int(node.lineno), int(node.col_offset))
        right = _source_offset(source, int(node.end_lineno), int(node.end_col_offset))
        if left < begin or right > end or right <= left:
            continue
        definitions, uses = _defs_uses(node)
        text = middle[left - begin:right - begin]
        item = {"kind": "statement", "candidate_ordinal": ordinal, "start": left - begin, "end": right - begin, "text": text, "definitions": sorted(definitions), "uses": sorted(uses)}
        statements.append(item)
        fragments.append(item)
        fragments.append({**item, "kind": "def_use"})
    if statements:
        definitions = sorted({name for item in statements for name in item["definitions"]})
        uses = sorted({name for item in statements for name in item["uses"]} - set(definitions))
        fragments.append({"kind": "basic_block", "candidate_ordinal": ordinal, "start": 0, "end": len(middle), "text": middle, "definitions": definitions, "uses": uses})
    elif middle:
        fragments.append({"kind": "basic_block", "candidate_ordinal": ordinal, "start": 0, "end": len(middle), "text": middle, "definitions": [], "uses": []})
    return fragments


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    diagnostic = analyze_candidate(str(candidate["prefix_text"]), str(candidate["middle_text"]), str(candidate["suffix_text"]))
    confidence = candidate.get("final_token_confidences") or []
    mean_confidence = sum(float(value) for value in confidence) / len(confidence) if confidence else 0.0
    return (*diagnostic.ranking_key, -mean_confidence, int(candidate["canvas_tokens"]), int(candidate["seed"]))


def assemble_fragments(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select AST statement/basic-block/def-use fragments using visible obligations."""
    if len(candidates) != 8:
        raise ValueError("M4 requires exactly eight deployable candidates")
    views = [candidate_view(row) for row in candidates]
    prefix, suffix = str(views[0]["prefix_text"]), str(views[0]["suffix_text"])
    if any(str(row["prefix_text"]) != prefix or str(row["suffix_text"]) != suffix for row in views):
        raise RuntimeError("M4 candidate bank has inconsistent prefix/suffix context")
    best_ordinal = min(range(len(views)), key=lambda index: _candidate_rank(views[index]))
    best = views[best_ordinal]
    obligation_names = set(extract_backward_obligations(prefix, suffix).dependency_names)
    statement_fragments = [
        fragment
        for ordinal, candidate in enumerate(views)
        for fragment in candidate_fragments(candidate, ordinal)
        if fragment["kind"] in {"statement", "def_use"}
    ]
    selected: list[dict[str, Any]] = []
    pending = sorted(obligation_names)
    resolved: set[str] = set()
    while pending:
        name = pending.pop(0)
        if name in resolved:
            continue
        providers = [fragment for fragment in statement_fragments if name in fragment["definitions"]]
        if not providers:
            continue
        provider = min(
            providers,
            key=lambda item: (
                0 if item["kind"] == "def_use" else 1,
                _candidate_rank(views[int(item["candidate_ordinal"])]),
                int(item["start"]),
                int(item["end"]),
            ),
        )
        if provider not in selected:
            selected.append(provider)
            pending.extend(name for name in provider["uses"] if name not in resolved)
        resolved.add(name)
    if not selected:
        selected = [fragment for fragment in candidate_fragments(best, best_ordinal) if fragment["kind"] == "basic_block"][:1]
    deduplicated: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for fragment in sorted(selected, key=lambda item: (int(item["candidate_ordinal"]), int(item["start"]), int(item["end"]))):
        text = str(fragment["text"])
        if text and text not in seen_text:
            deduplicated.append(fragment)
            seen_text.add(text)
    assembly = "".join(item["text"] if item["text"].endswith("\n") else item["text"] + "\n" for item in deduplicated)
    return {
        "prefix": prefix,
        "suffix": suffix,
        "best_middle_text": str(best["middle_text"]),
        "best_candidate_ordinal": best_ordinal,
        "assembly_text": assembly,
        "selected_fragments": deduplicated,
        "suffix_obligations": sorted(obligation_names),
        "resolved_obligations": sorted(resolved),
    }


def evaluator_task(prefix: str, suffix: str, source: Mapping[str, Any]) -> CodeTask:
    return CodeTask(
        task_id="m4_runtime_evaluator_task",
        prefix=prefix,
        suffix=suffix,
        full_prompt=prefix + "<FILL_ME>" + suffix,
        test_code=str(source["test"]),
        entry_point=str(source["entry_point"]),
        canonical_solution="",
        raw={},
    )


def evaluate_text(prefix: str, suffix: str, middle: str, source: Mapping[str, Any]) -> dict[str, Any]:
    task = evaluator_task(prefix, suffix, source)
    verification = run_verifier_stack(task=task, full_code=prefix + middle + suffix, completion_without_suffix=middle)
    tier3 = verification.get("tier3_unit_tests")
    return {"passed": bool(tier3.passed) if tier3 else False, "verification": {name: result.to_dict() for name, result in verification.items()}}


def connector_indices(tokenizer: Any, fragments: Sequence[Mapping[str, Any]], assembly: str) -> tuple[list[int], list[int]]:
    mask_id = int(resolve_mask_token_id(tokenizer))
    ids = tokenizer.encode(assembly, add_special_tokens=False)[:CANVAS_TOKENS]
    padded = [int(item) for item in ids] + [mask_id] * (CANVAS_TOKENS - len(ids))
    boundaries: list[int] = list(range(len(ids), CANVAS_TOKENS))
    cursor = 0
    for fragment in fragments[:-1]:
        cursor += len(tokenizer.encode(str(fragment["text"]), add_special_tokens=False))
        boundaries.extend(index for index in (cursor - 1, cursor) if 0 <= index < CANVAS_TOKENS)
    return padded, sorted(set(boundaries))


def repair_assembly(assembly: Mapping[str, Any], source: Mapping[str, Any], tokenizer: Any, model: Any) -> dict[str, Any]:
    middle_ids, indices = connector_indices(tokenizer, assembly["selected_fragments"], str(assembly["assembly_text"]))
    task = evaluator_task(str(assembly["prefix"]), str(assembly["suffix"]), source)
    set_global_seed(SEED)
    result = decode_fixed_canvas_state(
        task=task,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg_for(),
        canvas_tokens=CANVAS_TOKENS,
        total_steps=REPAIR_STEPS,
        phase_name="m4_fixed_budget_assembly_repair",
        initial_middle_ids=middle_ids,
        initial_mask_indices=indices,
        schedule_length=max(1, len(indices)),
    )
    result["connector_indices"] = indices
    return result


def build_manifest_from_bank(rows: Sequence[Mapping[str, Any]], frozen_groups: set[str]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("candidate_kind") == "deployable_grid":
            grouped[str(row["row_key"])].append(dict(row))
    manifest: list[dict[str, Any]] = []
    for row_key, candidates in sorted(grouped.items()):
        if len(candidates) != 8:
            raise RuntimeError("M4 bank requires exactly eight deployable candidates per row")
        context = candidates[0]
        if str(context["task_group"]) in frozen_groups:
            raise RuntimeError("Frozen task leaked into M4 candidate bank")
        manifest.append(
            {
                "row_key": row_key,
                "case_index": context["case_index"],
                "source_row_id": context["source_row_id"],
                "task_group": context["task_group"],
                "length_bucket": context["length_bucket"],
                "reference_middle_tokens": context["reference_middle_tokens"],
            }
        )
    if len(manifest) != 148 or len({row["task_group"] for row in manifest}) != 148:
        raise RuntimeError("M4 requires the existing 148-task RandomSpanLight bank")
    return manifest, grouped


def choose_smoke(manifest: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    buckets = ("short", "medium", "long", "extreme")
    per, remainder = divmod(count, len(buckets))
    chosen: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        candidates = sorted((dict(row) for row in manifest if row["length_bucket"] == bucket), key=lambda row: int(row["case_index"]))
        chosen.extend(candidates[: per + (1 if index < remainder else 0)])
    if len(chosen) != count:
        raise RuntimeError("M4 could not form requested smoke manifest")
    return sorted(chosen, key=lambda row: int(row["case_index"]))


def offline_rows(manifest: Sequence[Mapping[str, Any]], grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    best_rows: list[dict[str, Any]] = []
    assembly_rows: list[dict[str, Any]] = []
    for item in manifest:
        assembly = assemble_fragments(grouped[str(item["row_key"])])
        best_diagnostic = analyze_candidate(assembly["prefix"], assembly["best_middle_text"], assembly["suffix"])
        assembly_diagnostic = analyze_candidate(assembly["prefix"], assembly["assembly_text"], assembly["suffix"])
        common = {key: item[key] for key in ("row_key", "case_index", "source_row_id", "task_group", "length_bucket", "reference_middle_tokens")}
        best_rows.append({
            **common,
            "candidate_key": method_key(str(item["row_key"]), METHODS[0]),
            "candidate_kind": METHODS[0],
            "status": "ok",
            "offline_structural_valid": bool(best_diagnostic.full_parse_passed),
            "middle_text": assembly["best_middle_text"],
            "metrics": {"actual_forward_count": 0, "token_budget": 0},
            "assembly": {key: assembly[key] for key in ("best_candidate_ordinal", "suffix_obligations", "resolved_obligations")},
            "structural_diagnostics": {"full_parse_passed": best_diagnostic.full_parse_passed, "unsatisfied_obligation_count": len(best_diagnostic.unsatisfied_obligations), "def_use_conflict_count": len(best_diagnostic.def_use_conflicts)},
        })
        assembly_rows.append({
            **common,
            "candidate_key": method_key(str(item["row_key"]), METHODS[1]),
            "candidate_kind": METHODS[1],
            "status": "ok",
            "offline_structural_valid": bool(assembly_diagnostic.full_parse_passed),
            "middle_text": assembly["assembly_text"],
            "metrics": {"actual_forward_count": 0, "token_budget": 0},
            "assembly": {key: assembly[key] for key in ("selected_fragments", "suffix_obligations", "resolved_obligations")},
            "structural_diagnostics": {"full_parse_passed": assembly_diagnostic.full_parse_passed, "unsatisfied_obligation_count": len(assembly_diagnostic.unsatisfied_obligations), "def_use_conflict_count": len(assembly_diagnostic.def_use_conflicts)},
        })
    return best_rows, assembly_rows


def evaluated_offline_rows(
    manifest: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Post-selection evaluator rows for the best/assembly comparison arms."""
    best_rows: list[dict[str, Any]] = []
    assembly_rows: list[dict[str, Any]] = []
    for item in manifest:
        assembly = assemble_fragments(grouped[str(item["row_key"])])
        source = source_rows[int(item["source_row_id"])]
        best_eval = evaluate_text(assembly["prefix"], assembly["suffix"], assembly["best_middle_text"], source)
        assembly_eval = evaluate_text(assembly["prefix"], assembly["suffix"], assembly["assembly_text"], source)
        common = {key: item[key] for key in ("row_key", "case_index", "source_row_id", "task_group", "length_bucket", "reference_middle_tokens")}
        best_rows.append({
            **common,
            "candidate_key": method_key(str(item["row_key"]), METHODS[0]),
            "candidate_kind": METHODS[0],
            "status": "ok",
            "passed": best_eval["passed"],
            "middle_text": assembly["best_middle_text"],
            "metrics": {"actual_forward_count": 0, "token_budget": 0},
            "verification": best_eval["verification"],
        })
        assembly_rows.append({
            **common,
            "candidate_key": method_key(str(item["row_key"]), METHODS[1]),
            "candidate_kind": METHODS[1],
            "status": "ok",
            "passed": assembly_eval["passed"],
            "middle_text": assembly["assembly_text"],
            "metrics": {"actual_forward_count": 0, "token_budget": 0},
            "verification": assembly_eval["verification"],
        })
    return best_rows, assembly_rows


def write_replace_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_repair(manifest: Sequence[Mapping[str, Any]], grouped: Mapping[str, Sequence[Mapping[str, Any]]], source_rows: Sequence[Mapping[str, Any]], raw_path: Path, tokenizer: Any, model: Any) -> int:
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    counts = Counter(str(row.get("candidate_key") or "") for row in existing)
    if any(value > 1 for value in counts.values()):
        raise RuntimeError("M4 repair resume refuses duplicate keys")
    completed = set(counts)
    written = 0
    for item in manifest:
        key = method_key(str(item["row_key"]), METHODS[2])
        if key in completed:
            continue
        try:
            assembly = assemble_fragments(grouped[str(item["row_key"])])
            repair = repair_assembly(assembly, source_rows[int(item["source_row_id"])], tokenizer, model)
            middle = str(repair["middle_text"])
            row = {
                **item,
                "candidate_key": key,
                "candidate_kind": METHODS[2],
                "status": "ok",
                "passed": bool((repair.get("metrics") or {}).get("passed", False)),
                "middle_text": middle,
                "candidate_middle_sha256": sha256_text(middle),
                "metrics": repair.get("metrics") or {},
                "connector_indices": repair.get("connector_indices") or [],
                "assembly": {key: assembly[key] for key in ("selected_fragments", "suffix_obligations", "resolved_obligations")},
                "verification": repair.get("verification") or {},
            }
        except Exception as exc:
            row = {**item, "candidate_key": key, "candidate_kind": METHODS[2], "status": "error", "passed": False, "error_type": type(exc).__name__, "error_message": str(exc)[:240], "failure_traceback": traceback.format_exc(), "metrics": {}, "verification": {}}
        append_jsonl(raw_path, row)
        completed.add(key)
        written += 1
    return written


def audit(rows: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]], method: str, expected_forwards: int | None) -> dict[str, Any]:
    expected = {method_key(str(item["row_key"]), method) for item in manifest}
    counts = Counter(str(row.get("candidate_key") or "") for row in rows)
    observed = set(counts)
    bad_forward = [] if expected_forwards is None else [row for row in rows if row.get("status") == "ok" and int((row.get("metrics") or {}).get("actual_forward_count") or 0) != expected_forwards]
    return {"passed": observed == expected and all(value == 1 for value in counts.values()) and not any(row.get("status") != "ok" for row in rows) and not bad_forward, "missing_count": len(expected - observed), "extra_count": len(observed - expected), "duplicate_count": sum(value > 1 for value in counts.values()), "error_count": sum(row.get("status") != "ok" for row in rows), "bad_forward_count": len(bad_forward)}


def run(args: argparse.Namespace) -> int:
    source_rows = read_jsonl(Path(args.dataset_jsonl).resolve())
    bank_rows = read_jsonl(Path(args.candidate_bank_raw).resolve())
    frozen_groups, lock = load_frozen_groups()
    manifest, grouped = build_manifest_from_bank(bank_rows, frozen_groups)
    best_dir, assembly_dir, repair_dir, compact_dir = (Path(args.best_output_dir).resolve(), Path(args.assembly_output_dir).resolve(), Path(args.repair_output_dir).resolve(), Path(args.compact_dir).resolve())
    if len({best_dir, assembly_dir, repair_dir}) != 3:
        raise ValueError("M4 best, assembly, and repair outputs must be distinct")
    # Required first: offline assembly audit over the full existing 148-case bank.
    all_best, all_assembly = offline_rows(manifest, grouped)
    write_replace_jsonl(best_dir / "m4_best_single_raw.jsonl", all_best)
    write_replace_jsonl(assembly_dir / "m4_assembly_raw.jsonl", all_assembly)
    offline = {METHODS[0]: audit(all_best, manifest, METHODS[0], 0), METHODS[1]: audit(all_assembly, manifest, METHODS[1], 0)}
    if not all(item["passed"] for item in offline.values()):
        raise RuntimeError("M4 offline assembly audit failed")
    if not args.repair:
        write_json(compact_dir / "offline_summary.json", {"offline": offline, "case_count": len(manifest), "frozen_test_status": lock.get("test_status"), "test_evaluation_count": lock.get("test_evaluation_count")})
        return 0
    tokenizer, model = load_model_and_tokenizer(cfg_for().model)
    selected = choose_smoke(manifest, int(args.smoke_cases))
    evaluated_best, evaluated_assembly = evaluated_offline_rows(selected, grouped, source_rows)
    write_replace_jsonl(best_dir / "m4_best_evaluated_raw.jsonl", evaluated_best)
    write_replace_jsonl(assembly_dir / "m4_assembly_evaluated_raw.jsonl", evaluated_assembly)
    run_repair(selected, grouped, source_rows, repair_dir / "m4_repair_raw.jsonl", tokenizer, model)
    noops = run_repair(selected, grouped, source_rows, repair_dir / "m4_repair_raw.jsonl", tokenizer, model)
    smoke_rows = read_jsonl(repair_dir / "m4_repair_raw.jsonl")
    smoke = audit(smoke_rows, selected, METHODS[2], REPAIR_STEPS)
    best_smoke = audit(evaluated_best, selected, METHODS[0], 0)
    assembly_smoke = audit(evaluated_assembly, selected, METHODS[1], 0)
    smoke_gate = bool(smoke["passed"] and best_smoke["passed"] and assembly_smoke["passed"] and noops == 0 and lock.get("test_status") == "sealed" and int(lock.get("test_evaluation_count", -1)) == 0)
    full = None
    if smoke_gate and args.auto_full:
        require_randomspanlight_full(len(manifest))
        evaluated_best, evaluated_assembly = evaluated_offline_rows(manifest, grouped, source_rows)
        write_replace_jsonl(best_dir / "m4_best_evaluated_raw.jsonl", evaluated_best)
        write_replace_jsonl(assembly_dir / "m4_assembly_evaluated_raw.jsonl", evaluated_assembly)
        run_repair(manifest, grouped, source_rows, repair_dir / "m4_repair_raw.jsonl", tokenizer, model)
        full_noops = run_repair(manifest, grouped, source_rows, repair_dir / "m4_repair_raw.jsonl", tokenizer, model)
        full = {
            "best": audit(evaluated_best, manifest, METHODS[0], 0),
            "assembly": audit(evaluated_assembly, manifest, METHODS[1], 0),
            "repair": audit(read_jsonl(repair_dir / "m4_repair_raw.jsonl"), manifest, METHODS[2], REPAIR_STEPS),
            "resume_noop_writes": full_noops,
        }
    compact_dir.mkdir(parents=True, exist_ok=True)
    write_csv(compact_dir / "offline_method_audit.csv", [{"method": method, **value} for method, value in offline.items()])
    write_json(compact_dir / "run_manifest.json", {"offline": offline, "smoke": {"best": best_smoke, "assembly": assembly_smoke, "repair": smoke}, "smoke_gate_passed": smoke_gate, "full": full, "case_count": len(manifest), "methods": list(METHODS), "population_schedule": population_schedule(), "automatic_full_population": "randomspanlight_full", "test_evaluation_count": 0})
    return 0 if smoke_gate and (full is None or all(full[name]["passed"] for name in ("best", "assembly", "repair"))) else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="M4 Semantic Particle Assembly V0")
    root.add_argument("--dataset-jsonl", required=True)
    root.add_argument("--candidate-bank-raw", required=True)
    root.add_argument("--best-output-dir", required=True)
    root.add_argument("--assembly-output-dir", required=True)
    root.add_argument("--repair-output-dir", required=True)
    root.add_argument("--compact-dir", required=True)
    root.add_argument("--smoke-cases", type=int, default=SMOKE_CASES)
    root.add_argument("--repair", action="store_true")
    root.add_argument("--auto-full", action="store_true")
    return root


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
