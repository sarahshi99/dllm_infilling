#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import builtins
import csv
import json
import keyword
import re
import sys
import time
import tokenize
from collections import defaultdict
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.phase5_premise_falsification import bridge_features, deployable_proxy_scores


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
) -> tuple[Mapping[str, Any], dict[str, CandidateAnalysis]]:
    allowed_fields = {"candidate_key", "middle_text", "canvas_tokens", "seed"}
    diagnostics: dict[str, CandidateAnalysis] = {}
    for row in candidates:
        validate_selection_fields(set(row) - allowed_fields)
        diagnostics[str(row["candidate_key"])] = analyze_candidate(prefix, str(row["middle_text"]), suffix)
    chosen = min(
        candidates,
        key=lambda row: (
            diagnostics[str(row["candidate_key"])].ranking_key,
            int(row["canvas_tokens"]),
            int(row["seed"]),
            str(row["candidate_key"]),
        ),
    )
    return chosen, diagnostics


def _max_score(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    return max(rows, key=lambda row: (float(row[key]), -int(row["canvas_tokens"]), -int(row["seed"])))


def _pairwise(rows: Sequence[Mapping[str, Any]], method: str, baseline: str) -> dict[str, int]:
    wins = sum(bool(row[f"{method}_passed"]) and not bool(row[f"{baseline}_passed"]) for row in rows)
    losses = sum(not bool(row[f"{method}_passed"]) and bool(row[f"{baseline}_passed"]) for row in rows)
    return {"wins": wins, "losses": losses, "net": wins - losses, "help": wins, "harm": losses}


def _method_summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    passed = sum(bool(row[f"{method}_passed"]) for row in rows)
    return {"pass_count": passed, "task_count": len(rows), "pass_at_1": passed / len(rows) if rows else 0.0}


def run_analysis(bank_dir: Path, compact_bank_dir: Path, output_dir: Path) -> dict[str, Any]:
    raw = [row for row in read_jsonl(bank_dir / "candidate_bank_raw.jsonl") if row.get("candidate_kind") == "deployable_grid"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        groups[str(row["row_key"])].append(row)
    if not groups or any(len(rows) != 8 for rows in groups.values()):
        raise RuntimeError("Phase 6 analysis requires exactly eight candidates per allowed row")

    selections: list[dict[str, Any]] = []
    selection_latency = defaultdict(float)
    for row_key, rows in sorted(groups.items()):
        prefix = str(rows[0]["prefix_text"])
        suffix = str(rows[0]["suffix_text"])
        scored: list[dict[str, Any]] = []
        for row in rows:
            features = bridge_features(prefix, str(row["middle_text"]), suffix)
            candidate_tokens = float(row.get("candidate_middle_tokens") or 0.0)
            canvas = float(row["canvas_tokens"])
            features.update(
                {
                    "candidate_canvas_fill_ratio": candidate_tokens / canvas if canvas else 0.0,
                    "ordinary_confidence": float((row.get("metrics") or {}).get("mean_final_confidence") or 0.0),
                }
            )
            scores = deployable_proxy_scores(features)
            scored.append({**row, **scores})

        fixed = next(row for row in scored if int(row["canvas_tokens"]) == 64 and int(row["seed"]) == 0)
        started = time.perf_counter(); confidence = _max_score(scored, "deployable_proxy_ordinary_confidence"); selection_latency["confidence"] += time.perf_counter() - started
        started = time.perf_counter(); prefix_only = _max_score(scored, "deployable_proxy_prefix_only"); selection_latency["prefix_only"] += time.perf_counter() - started
        started = time.perf_counter(); combined = _max_score(scored, "deployable_proxy_combined"); selection_latency["phase5_combined"] += time.perf_counter() - started
        v1_inputs = [
            {
                "candidate_key": row["candidate_key"],
                "middle_text": row.get("middle_text", ""),
                "canvas_tokens": row["canvas_tokens"],
                "seed": row["seed"],
            }
            for row in scored
        ]
        started = time.perf_counter(); v1_input, v1_diagnostics = select_v1_candidate(prefix, suffix, v1_inputs); selection_latency["abductive_bridge_v1"] += time.perf_counter() - started
        by_key = {str(row["candidate_key"]): row for row in scored}
        chosen = {
            "fixed64": fixed,
            "confidence": confidence,
            "prefix_only": prefix_only,
            "phase5_combined": combined,
            "abductive_bridge_v1": by_key[str(v1_input["candidate_key"])],
        }
        output: dict[str, Any] = {
            "row_key": row_key,
            "length_bucket_offline_only": rows[0]["length_bucket"],
        }
        for name, row in chosen.items():
            output[f"{name}_candidate_key"] = row["candidate_key"]
            output[f"{name}_canvas_tokens"] = row["canvas_tokens"]
            output[f"{name}_seed"] = row["seed"]
            output[f"{name}_passed"] = bool(row["passed"])
        diagnostic = v1_diagnostics[str(v1_input["candidate_key"])]
        output.update(
            {
                "v1_full_parse_passed": diagnostic.full_parse_passed,
                "v1_boundary_violation_count": len(diagnostic.boundary_violations),
                "v1_control_contradiction_count": len(diagnostic.control_contradictions),
                "v1_unsatisfied_obligation_count": len(diagnostic.unsatisfied_obligations),
                "v1_def_use_conflict_count": len(diagnostic.def_use_conflicts),
                "v1_undefined_use_count": len(diagnostic.undefined_uses),
                "v1_restored_dependency_count": len(diagnostic.restored_dependencies),
            }
        )
        selections.append(output)

    methods = ["fixed64", "confidence", "prefix_only", "phase5_combined", "abductive_bridge_v1"]
    method_summaries = {method: _method_summary(selections, method) for method in methods}
    pairwise: dict[str, Any] = {}
    for method in methods[1:]:
        pairwise[f"{method}_vs_fixed64"] = _pairwise(selections, method, "fixed64")
    for baseline in methods[:-1]:
        pairwise[f"abductive_bridge_v1_vs_{baseline}"] = _pairwise(selections, "abductive_bridge_v1", baseline)

    bucket_rows: list[dict[str, Any]] = []
    for bucket in ["short", "medium", "long", "extreme"]:
        subset = [row for row in selections if row["length_bucket_offline_only"] == bucket]
        for method in methods:
            summary = _method_summary(subset, method)
            bucket_rows.append({"length_bucket_offline_only": bucket, "method": method, **summary})

    bank_summary = json.loads((compact_bank_dir / "full_summary.json").read_text(encoding="utf-8"))
    summary = {
        "verdict": "phase6_exploratory_comparison_completed",
        "task_count": len(selections),
        "candidate_count": len(raw),
        "methods": method_summaries,
        "pairwise": pairwise,
        "selection_latency_sec_total": dict(selection_latency),
        "selection_latency_sec_mean_per_task": {name: value / len(selections) for name, value in selection_latency.items()},
        "candidate_generation": {
            "wall_sec": bank_summary.get("wall_sec"),
            "summed_gpu_decode_sec": bank_summary.get("summed_gpu_decode_sec"),
            "summed_verification_sec": bank_summary.get("summed_verification_sec"),
            "mean_candidate_latency_sec": bank_summary.get("mean_candidate_latency_sec"),
            "peak_cuda_memory_bytes": bank_summary.get("peak_cuda_memory_bytes"),
            "candidate_rows": bank_summary.get("candidate_rows"),
            "denoising_steps_per_candidate": bank_summary.get("denoising_steps_per_candidate"),
        },
        "selection_policy": {
            "mechanism": "fixed_lexicographic_abductive_program_state_bridge_v1",
            "fitted_parameters": False,
            "reference_used_for_selection": False,
            "unit_tests_or_outcomes_used_for_selection": False,
            "oracle_length_used_for_selection": False,
            "supervised_score_used_for_selection": False,
            "fusion_used": False,
        },
        "frozen_test_status": bank_summary.get("frozen_test_status"),
        "test_evaluation_count": bank_summary.get("test_evaluation_count"),
        "claim_boundary": "exploratory_development_not_heldout_sota",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "selection_results.csv", selections)
    write_csv(output_dir / "length_bucket_summary.csv", bucket_rows)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "method_spec.json",
        {
            "name": "Abductive Program-State Bridge V1",
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
            "forbidden_selection_fields": sorted(FORBIDDEN_SELECTION_FIELDS),
        },
    )
    lines = [
        "# Phase 6 Abductive Program-State Bridge V1",
        "",
        "Exploratory/development same-pool comparison; not held-out SOTA.",
        "",
        f"Tasks/candidates: `{len(selections)}` / `{len(raw)}`.",
        f"Frozen test: `{summary['frozen_test_status']}`, `test_evaluation_count={summary['test_evaluation_count']}`.",
        "",
        "## Pass@1",
        "",
    ]
    for method in methods:
        item = method_summaries[method]
        lines.append(f"- {method}: `{item['pass_count']}/{item['task_count']} = {item['pass_at_1']:.4%}`")
    lines.extend(["", "## V1 Pairwise", ""])
    for baseline in methods[:-1]:
        item = pairwise[f"abductive_bridge_v1_vs_{baseline}"]
        lines.append(f"- vs {baseline}: `{item['wins']}` wins / `{item['losses']}` losses / net `{item['net']}`")
    lines.extend(
        [
            "",
            "No reference, unit tests, pass/fail labels, oracle length, supervised score, task identity, or frozen-test statistic entered selection.",
            "No homotopy, birth–death, particle assembly, controller, cal-lite, or method fusion was used.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Phase 6 Abductive Program-State Bridge V1")
    root.add_argument("--bank-dir", required=True)
    root.add_argument("--compact-bank-dir", required=True)
    root.add_argument("--output-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(Path(args.bank_dir).resolve(), Path(args.compact_bank_dir).resolve(), Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
