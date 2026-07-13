#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import builtins
import csv
import json
import keyword
import math
import re
import sys
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BUILTIN_NAMES = set(dir(builtins))
IGNORED_NAMES = BUILTIN_NAMES | set(keyword.kwlist) | {"True", "False", "None"}
FORBIDDEN_SELECTION_FIELDS = {
    "reference_code",
    "reference_middle",
    "canonical_solution",
    "unit_test_result",
    "verification",
    "passed",
    "pass_fail",
    "error_type",
    "oracle_length",
    "oracle_mask_length",
    "task_id",
    "task_group",
    "split",
    "split_label",
    "supervised_score",
    "row_key",
    "source_row_id",
    "case_index",
    "reference_middle_tokens",
    "length_bucket",
}


@dataclass(frozen=True)
class BackwardObligations:
    required_definitions: tuple[str, ...]
    dependency_names: tuple[str, ...]
    control_requirements: tuple[str, ...]
    entry_indent: int
    starts_with_continuation: str
    requires_fallthrough: bool


@dataclass(frozen=True)
class CandidateAnalysis:
    full_parse_passed: bool
    boundary_violations: tuple[str, ...]
    control_contradictions: tuple[str, ...]
    unsatisfied_obligations: tuple[str, ...]
    def_use_conflicts: tuple[str, ...]
    undefined_uses: tuple[str, ...]
    satisfied_obligations: tuple[str, ...]
    restored_dependencies: tuple[str, ...]
    forward_definitions: tuple[str, ...]
    ranking_key: tuple[int, ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def validate_selection_fields(fields: Iterable[str]) -> None:
    forbidden = sorted(set(fields) & FORBIDDEN_SELECTION_FIELDS)
    if forbidden:
        raise ValueError(f"Forbidden selection field(s): {forbidden}")


def _nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _entry_indent(text: str) -> int:
    lines = _nonempty_lines(text)
    return _indent(lines[0]) if lines else 0


def _name_tokens(text: str) -> list[str]:
    try:
        return [
            token.string
            for token in tokenize.generate_tokens(StringIO(text).readline)
            if token.type == tokenize.NAME and token.string not in IGNORED_NAMES
        ]
    except (tokenize.TokenError, IndentationError):
        return [name for name in re.findall(r"\b[A-Za-z_]\w*\b", text) if name not in IGNORED_NAMES]


def _parameter_names(prefix: str) -> set[str]:
    names: set[str] = set()
    for match in re.finditer(r"\b(?:async\s+)?def\s+\w+\s*\((.*?)\)\s*(?:->[^:]*)?:", prefix, flags=re.S):
        fragment = match.group(1)
        for item in fragment.split(","):
            name = item.strip().lstrip("*").split(":", 1)[0].split("=", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", name):
                names.add(name)
    return names


def _lexical_definitions(text: str) -> set[str]:
    definitions = _parameter_names(text)
    patterns = [
        r"(?m)^\s*([A-Za-z_]\w*)\s*(?::[^=\n]+)?=",
        r"(?m)^\s*for\s+([A-Za-z_]\w*)\s+in\b",
        r"(?m)^\s*(?:with\s+.+?\s+as|except\s+.+?\s+as)\s+([A-Za-z_]\w*)\b",
        r"(?m)^\s*(?:def|class)\s+([A-Za-z_]\w*)\b",
    ]
    for pattern in patterns:
        definitions.update(re.findall(pattern, text))
    return definitions


def _fragment_defs_uses(text: str) -> tuple[set[str], set[str]]:
    definitions = _lexical_definitions(text)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return definitions, set(_name_tokens(text)) - definitions
    uses: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                definitions.add(node.id)
            elif isinstance(node.ctx, ast.Load) and node.id not in IGNORED_NAMES:
                uses.add(node.id)
        elif isinstance(node, ast.arg):
            definitions.add(node.arg)
    return definitions, uses


def _suffix_loads_before_definitions(suffix: str) -> set[str]:
    normalized = suffix
    lines = _nonempty_lines(suffix)
    if lines:
        minimum = min(_indent(line) for line in lines)
        normalized = "\n".join(line[minimum:] if len(line) >= minimum else line for line in suffix.splitlines()) + "\n"
    wrapped = "def __phase6_suffix__():\n" + "".join(f"    {line}\n" for line in normalized.splitlines())
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        definitions = _lexical_definitions(suffix)
        return set(_name_tokens(suffix)) - definitions
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    definitions: set[str] = set()
    required: set[str] = set()
    events: list[tuple[int, int, int, str]] = []
    for node in ast.walk(function):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            events.append((node.target.lineno, node.target.col_offset, 0, node.target.id))
        if isinstance(node, ast.Name):
            kind = 1 if isinstance(node.ctx, (ast.Store, ast.Del)) else 0
            events.append((node.lineno, node.col_offset, kind, node.id))
    for _, _, kind, name in sorted(events):
        if name in IGNORED_NAMES:
            continue
        if kind == 0 and name not in definitions:
            required.add(name)
        elif kind == 1:
            definitions.add(name)
    return required


def _control_requirements(suffix: str) -> tuple[str, ...]:
    requirements: set[str] = set()
    stripped = suffix.lstrip()
    if re.match(r"elif\b", stripped):
        requirements.add("open_if")
    elif re.match(r"else\s*:", stripped):
        requirements.add("open_if_loop_or_try")
    elif re.match(r"except\b|finally\s*:", stripped):
        requirements.add("open_try")
    if re.search(r"(?m)^\s*(?:break|continue)\b", suffix):
        requirements.add("active_loop")
    return tuple(sorted(requirements))


def extract_backward_obligations(prefix: str, suffix: str) -> BackwardObligations:
    prefix_defs, _ = _fragment_defs_uses(prefix)
    required = _suffix_loads_before_definitions(suffix) - prefix_defs - IGNORED_NAMES
    stripped = suffix.lstrip()
    continuation = ""
    match = re.match(r"(elif|else|except|finally)\b", stripped)
    if match:
        continuation = match.group(1)
    return BackwardObligations(
        required_definitions=tuple(sorted(required)),
        dependency_names=tuple(sorted(required)),
        control_requirements=_control_requirements(suffix),
        entry_indent=_entry_indent(suffix),
        starts_with_continuation=continuation,
        requires_fallthrough=bool(suffix.strip()),
    )


def _terminal_middle(middle: str) -> bool:
    lines = _nonempty_lines(middle)
    if not lines:
        return False
    return bool(re.match(r"\s*(return|raise)\b", lines[-1]))


def _control_obligation_satisfied(requirement: str, prefix_middle: str) -> bool:
    if requirement == "active_loop":
        return bool(re.search(r"(?m)^\s*(?:async\s+)?(?:for|while)\b", prefix_middle))
    if requirement == "open_if":
        return bool(re.search(r"(?m)^\s*if\b", prefix_middle))
    if requirement == "open_if_loop_or_try":
        return bool(re.search(r"(?m)^\s*(?:if|for|while|try)\b", prefix_middle))
    if requirement == "open_try":
        return bool(re.search(r"(?m)^\s*try\s*:", prefix_middle))
    return False


def _full_parse(prefix: str, middle: str, suffix: str) -> bool:
    try:
        ast.parse(prefix + middle + suffix)
        return True
    except SyntaxError:
        return False


def analyze_candidate(prefix: str, middle: str, suffix: str) -> CandidateAnalysis:
    obligations = extract_backward_obligations(prefix, suffix)
    prefix_defs, _ = _fragment_defs_uses(prefix)
    middle_defs, middle_uses = _fragment_defs_uses(middle)
    deleted = set(re.findall(r"(?m)^\s*del\s+([A-Za-z_]\w*)\b", middle))
    forward_defs = (prefix_defs | middle_defs) - deleted
    undefined = middle_uses - prefix_defs - middle_defs - IGNORED_NAMES
    unsatisfied = set(obligations.required_definitions) - forward_defs
    unsatisfied.update(
        f"control:{requirement}"
        for requirement in obligations.control_requirements
        if not _control_obligation_satisfied(requirement, prefix + middle)
    )
    satisfied = set(obligations.required_definitions) & forward_defs
    restored = set(obligations.dependency_names) & forward_defs

    def_use_conflicts = {f"use_before_definition:{name}" for name in undefined}
    def_use_conflicts.update(f"deletes_required:{name}" for name in deleted & set(obligations.required_definitions))

    control: set[str] = set()
    if re.search(r"(?m)^\s*break\b", middle) and not re.search(r"(?m)^\s*(for|while)\b", prefix + middle):
        control.add("break_outside_loop")
    if re.search(r"(?m)^\s*continue\b", middle) and not re.search(r"(?m)^\s*(for|while)\b", prefix + middle):
        control.add("continue_outside_loop")
    if obligations.requires_fallthrough and _terminal_middle(middle):
        control.add("premature_terminal_before_suffix")

    parse_passed = _full_parse(prefix, middle, suffix)
    boundary: set[str] = set()
    if middle and suffix and not middle.endswith("\n"):
        boundary.add("missing_middle_suffix_newline")
    if not parse_passed:
        boundary.add("full_program_syntax")

    ranking_key = (
        0 if parse_passed else 1,
        len(boundary),
        len(control),
        len(unsatisfied),
        len(def_use_conflicts),
        len(undefined),
        -len(restored),
        -len(satisfied),
    )
    return CandidateAnalysis(
        full_parse_passed=parse_passed,
        boundary_violations=tuple(sorted(boundary)),
        control_contradictions=tuple(sorted(control)),
        unsatisfied_obligations=tuple(sorted(unsatisfied)),
        def_use_conflicts=tuple(sorted(def_use_conflicts)),
        undefined_uses=tuple(sorted(undefined)),
        satisfied_obligations=tuple(sorted(satisfied)),
        restored_dependencies=tuple(sorted(restored)),
        forward_definitions=tuple(sorted(forward_defs)),
        ranking_key=ranking_key,
    )


def select_v1_candidate(
    prefix: str,
    suffix: str,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[int, CandidateAnalysis]]:
    """Select with state-only fields; opaque storage identifiers never enter ranking."""
    allowed_fields = {"candidate_ordinal", "middle_text", "canvas_tokens", "seed"}
    diagnostics: dict[int, CandidateAnalysis] = {}
    for row in candidates:
        validate_selection_fields(set(row) - allowed_fields)
        ordinal = int(row["candidate_ordinal"])
        if ordinal in diagnostics:
            raise ValueError(f"Duplicate candidate ordinal {ordinal}")
        diagnostics[ordinal] = analyze_candidate(prefix, str(row["middle_text"]), suffix)
    chosen = min(
        candidates,
        key=lambda row: (
            diagnostics[int(row["candidate_ordinal"])].ranking_key,
            int(row["canvas_tokens"]),
            int(row["seed"]),
        ),
    )
    return chosen, diagnostics


def selected_base_key(row: Mapping[str, Any]) -> str:
    """Stable state-only identifier used to attach a stage-two refinement."""
    key = row.get("selected_base_candidate_key")
    if not isinstance(key, str) or not key:
        raise ValueError("Refinement row is missing selected_base_candidate_key")
    return key


def _pairwise(rows: Sequence[Mapping[str, Any]], method: str, baseline: str) -> dict[str, int]:
    wins = sum(bool(row[f"{method}_passed"]) and not bool(row[f"{baseline}_passed"]) for row in rows)
    losses = sum(not bool(row[f"{method}_passed"]) and bool(row[f"{baseline}_passed"]) for row in rows)
    return {"wins": wins, "losses": losses, "net": wins - losses, "help": wins, "harm": losses}


def _row_budget(row: Mapping[str, Any]) -> dict[str, float]:
    metrics = row.get("metrics") or {}
    return {
        "forward_count": float(metrics.get("actual_forward_count") or metrics.get("total_steps") or 0.0),
        "token_budget": float(metrics.get("canvas_tokens") or row.get("canvas_tokens") or 0.0)
        * float(metrics.get("actual_forward_count") or metrics.get("total_steps") or 0.0),
        "wall_sec": float(metrics.get("total_sec_including_probe") or row.get("wall_sec") or 0.0),
    }


def _budget_summary(selections: Sequence[Mapping[str, Any]], method: str) -> dict[str, float]:
    budgets = [
        {
            "forward_count": float(row.get(f"{method}_forward_count") or 0.0),
            "token_budget": float(row.get(f"{method}_token_budget") or 0.0),
            "wall_sec": float(row.get(f"{method}_wall_sec") or 0.0),
        }
        for row in selections
    ]
    count = len(budgets)
    return {
        "mean_forward_count": sum(item["forward_count"] for item in budgets) / count if count else math.nan,
        "mean_token_budget": sum(item["token_budget"] for item in budgets) / count if count else math.nan,
        "mean_wall_sec": sum(item["wall_sec"] for item in budgets) / count if count else math.nan,
        "total_forward_count": sum(item["forward_count"] for item in budgets),
        "total_token_budget": sum(item["token_budget"] for item in budgets),
        "total_wall_sec": sum(item["wall_sec"] for item in budgets),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _task_macro_summary(selections: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selections:
        groups[str(row["task_group"])].append(row)
    group_accuracy = [
        _mean([float(bool(item[f"{method}_passed"])) for item in items])
        for _, items in sorted(groups.items())
    ]
    return {
        "task_group_count": len(group_accuracy),
        "equal_weight_base_task_macro_accuracy": _mean(group_accuracy),
        "span_micro_accuracy_descriptive": _mean([float(bool(row[f"{method}_passed"])) for row in selections]),
    }


def _write_frontier_rows(selections: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        macro = _task_macro_summary(selections, method)
        budget = _budget_summary(selections, method)
        rows.append(
            {
                "method": method,
                **macro,
                "mean_forward_count": budget["mean_forward_count"],
                "mean_token_budget": budget["mean_token_budget"],
                "mean_wall_sec": budget["mean_wall_sec"],
            }
        )
    return rows


def run_analysis(
    bank_dir: Path,
    generic_refinement_dir: Path,
    m1_refinement_dir: Path,
    compact_bank_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    all_raw = read_jsonl(bank_dir / "candidate_bank_raw.jsonl")
    refinement_raw = [
        *read_jsonl(generic_refinement_dir / "equal_compute_generic_remask_raw.jsonl"),
        *read_jsonl(m1_refinement_dir / "m1_dependency_cone_remask_raw.jsonl"),
    ]
    grid_rows = [row for row in all_raw if row.get("candidate_kind") == "deployable_grid"]
    oracle_rows = [row for row in all_raw if row.get("candidate_kind") == "oracle_sufficient_diagnostic_ceiling"]
    grouped_grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in grid_rows:
        grouped_grid[str(row["row_key"])].append(row)
    if not grouped_grid or any(len(rows) != 8 for rows in grouped_grid.values()):
        raise RuntimeError("Phase 6 analysis requires exactly eight stage-one deployable candidates per allowed row")
    oracle_by_key = {str(row["row_key"]): row for row in oracle_rows}
    refinement_by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in refinement_raw:
        method = str(row.get("candidate_kind") or "")
        if method in {"equal_compute_generic_remask", "m1_dependency_cone_remask"}:
            row_key = str(row["row_key"])
            if row_key in refinement_by_method[method]:
                raise RuntimeError(f"Duplicate refinement result for {method}/{row_key}")
            refinement_by_method[method][row_key] = row
    required = set(grouped_grid)
    if set(oracle_by_key) != required:
        raise RuntimeError("M1 analysis oracle ceiling rows do not match the stage-one task population")
    for method in ("equal_compute_generic_remask", "m1_dependency_cone_remask"):
        if set(refinement_by_method[method]) != required:
            raise RuntimeError(f"M1 analysis {method} rows do not match the stage-one task population")

    selections: list[dict[str, Any]] = []
    for row_key, rows in sorted(grouped_grid.items()):
        fixed64 = next(row for row in rows if int(row["canvas_tokens"]) == 64 and int(row["seed"]) == 0)
        confidence = max(
            rows,
            key=lambda row: (
                float((row.get("metrics") or {}).get("mean_final_confidence") or 0.0),
                -int(row["canvas_tokens"]),
                -int(row["seed"]),
            ),
        )
        selected_key = selected_base_key(refinement_by_method["m1_dependency_cone_remask"][row_key])
        selected_base = next(row for row in rows if str(row["candidate_key"]) == selected_key)
        chosen = {
            "fixed64": fixed64,
            "ordinary_confidence_best_of_grid": confidence,
            "equal_compute_generic_remask": refinement_by_method["equal_compute_generic_remask"][row_key],
            "m1_score_only": selected_base,
            "m1_full": refinement_by_method["m1_dependency_cone_remask"][row_key],
            "oracle_ceiling": oracle_by_key[row_key],
        }
        output: dict[str, Any] = {
            "row_key": row_key,
            "task_group": fixed64["task_group"],
            "length_bucket_offline_only": fixed64["length_bucket"],
        }
        for name, row in chosen.items():
            budget = _row_budget(row)
            output.update(
                {
                    f"{name}_candidate_key": row["candidate_key"],
                    f"{name}_canvas_tokens": row["canvas_tokens"],
                    f"{name}_seed": row["seed"],
                    f"{name}_passed": bool(row.get("passed")),
                    f"{name}_forward_count": budget["forward_count"],
                    f"{name}_token_budget": budget["token_budget"],
                    f"{name}_wall_sec": budget["wall_sec"],
                }
            )
        diagnostics = refinement_by_method["m1_dependency_cone_remask"][row_key].get("selection_diagnostics") or {}
        output.update({f"v1_{key}": value for key, value in diagnostics.items()})
        output["m1_full_fallback_to_fixed64"] = bool(refinement_by_method["m1_dependency_cone_remask"][row_key].get("fallback_to_fixed64"))
        output["m1_full_remasked_token_count"] = int(refinement_by_method["m1_dependency_cone_remask"][row_key].get("remasked_token_count") or 0)
        selections.append(output)

    methods = [
        "fixed64",
        "ordinary_confidence_best_of_grid",
        "equal_compute_generic_remask",
        "m1_score_only",
        "m1_full",
        "oracle_ceiling",
    ]
    method_summaries = {method: _task_macro_summary(selections, method) for method in methods}
    pairwise = {f"{method}_vs_fixed64": _pairwise(selections, method, "fixed64") for method in methods[1:]}
    pairwise.update({
        f"m1_full_vs_{method}": _pairwise(selections, "m1_full", method)
        for method in ("ordinary_confidence_best_of_grid", "equal_compute_generic_remask", "m1_score_only")
    })
    bucket_rows: list[dict[str, Any]] = []
    for bucket in ("short", "medium", "long", "extreme"):
        subset = [row for row in selections if row["length_bucket_offline_only"] == bucket]
        for method in methods:
            bucket_rows.append({"length_bucket_offline_only": bucket, "method": method, **_task_macro_summary(subset, method)})
    frontier_rows = _write_frontier_rows(selections, methods)
    bank_summary = json.loads((compact_bank_dir / "full_summary.json").read_text(encoding="utf-8"))
    summary = {
        "verdict": "phase6_m1_full_real_refinement_completed",
        "task_count": len(selections),
        "task_group_count": len({row["task_group"] for row in selections}),
        "stage1_grid_candidate_count": len(grid_rows),
        "oracle_ceiling_count": len(oracle_rows),
        "real_refinement_count": len(refinement_raw),
        "primary_estimand": "equal_weight_base_task_macro_accuracy",
        "span_micro_role": "descriptive_only",
        "methods": method_summaries,
        "pairwise": pairwise,
        "accuracy_cost_frontier": frontier_rows,
        "candidate_generation": {
            "wall_sec": bank_summary.get("wall_sec"),
            "summed_gpu_decode_sec": bank_summary.get("summed_gpu_decode_sec"),
            "summed_verification_sec": bank_summary.get("summed_verification_sec"),
            "mean_candidate_latency_sec": bank_summary.get("mean_candidate_latency_sec"),
            "peak_cuda_memory_bytes": bank_summary.get("peak_cuda_memory_bytes"),
            "candidate_rows": bank_summary.get("candidate_rows"),
        },
        "selection_policy": {
            "mechanism": "fixed_lexicographic_abductive_program_state_bridge_m1_1",
            "fitted_parameters": False,
            "reference_used_for_selection": False,
            "unit_tests_or_outcomes_used_for_selection": False,
            "oracle_length_used_for_selection": False,
            "task_identity_or_split_used_for_selection": False,
            "generic_remask": "actual 64-forward low-final-confidence refinement with the same initial remask cardinality as the dependency cone",
            "m1_full": "actual 64-forward AST/def-use dependency-cone remask; safe fallback is a fixed64 null refinement with its own 64 forwards",
        },
        "frozen_test_status": bank_summary.get("frozen_test_status"),
        "test_evaluation_count": bank_summary.get("test_evaluation_count"),
        "claim_boundary": "exploratory_development_not_heldout_sota",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "selection_results.csv", selections)
    write_csv(output_dir / "length_bucket_summary.csv", bucket_rows)
    write_csv(output_dir / "accuracy_cost_frontier.csv", frontier_rows)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "method_spec.json",
        {
            "name": "M1.1 Abductive Program-State Bridge",
            "ranking_order": [
                "full_program_parse",
                "boundary_violation_count",
                "control_contradiction_count",
                "unsatisfied_obligation_count",
                "def_use_conflict_count",
                "undefined_use_count",
                "restored_dependency_count_desc",
                "satisfied_obligation_count_desc",
                "smaller_canvas_tiebreak",
                "smaller_seed_tiebreak",
            ],
            "stage_two": {
                "equal_compute_generic_remask": "same selected stage-one candidate; low-confidence token remask matched to the M1 cone cardinality; 64 forward passes",
                "m1_dependency_cone_remask": "same selected stage-one candidate; AST/def-use suffix-obligation dependency cone; 64 forward passes",
                "safe_fallback": "fixed64 null refinement when the deployable dependency cone is absent or unmappable; still 64 forward passes",
            },
            "forbidden_selection_fields": sorted(FORBIDDEN_SELECTION_FIELDS),
        },
    )
    lines = [
        "# M1.1 Abductive Program-State Bridge",
        "",
        "Exploratory/development same-pool comparison with actual two-stage refinement; not held-out SOTA.",
        "",
        f"Spans/task groups: `{len(selections)}` / `{summary['task_group_count']}`.",
        f"Frozen test: `{summary['frozen_test_status']}`, `test_evaluation_count={summary['test_evaluation_count']}`.",
        "",
        "## Equal-weight task-macro accuracy",
        "",
    ]
    for method in methods:
        item = method_summaries[method]
        lines.append(
            f"- {method}: `{item['equal_weight_base_task_macro_accuracy']:.4%}` macro; "
            f"`{item['span_micro_accuracy_descriptive']:.4%}` span-micro descriptive"
        )
    lines.extend(["", "## M1 Full Pairwise", ""])
    for baseline in ("ordinary_confidence_best_of_grid", "equal_compute_generic_remask", "m1_score_only"):
        item = pairwise[f"m1_full_vs_{baseline}"]
        lines.append(f"- vs {baseline}: `{item['wins']}` wins / `{item['losses']}` losses / net `{item['net']}`")
    lines.extend(
        [
            "",
            "No reference, unit tests, pass/fail labels, oracle length, supervised score, task identity, split label, or frozen-test statistic entered selection or remasking.",
            "Generic and M1 full each execute an additional 64-forward refinement. M1 fallback is an explicit fixed64 null refinement, not a zero-forward row.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Phase 6 Abductive Program-State Bridge V1")
    root.add_argument("--bank-dir", required=True)
    root.add_argument("--generic-refinement-dir", required=True)
    root.add_argument("--m1-refinement-dir", required=True)
    root.add_argument("--compact-bank-dir", required=True)
    root.add_argument("--output-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(
        Path(args.bank_dir).resolve(),
        Path(args.generic_refinement_dir).resolve(),
        Path(args.m1_refinement_dir).resolve(),
        Path(args.compact_bank_dir).resolve(),
        Path(args.output_dir).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
