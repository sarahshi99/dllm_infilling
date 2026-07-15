#!/usr/bin/env python3
"""State-capturing fixed-canvas decoding used by the Phase 6 M1 protocol.

This module deliberately contains no selection logic.  It executes a fixed
canvas decode from either an all-mask canvas or a supplied candidate state and
returns only the generated state plus post-generation evaluator results.
"""

from __future__ import annotations

import time
import ast
import builtins
import keyword
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from expvision_dllm_clean.dataset import CodeTask
from expvision_dllm_clean.decode import (
    linear_target_masks,
    prepare_model_inputs,
    select_low_confidence_mask_positions,
)
from expvision_dllm_clean.verifier import parse_compile_diagnostics, run_verifier_stack


def remask_count_for_canvas(canvas_tokens: int) -> int:
    """Use a small, fixed remask budget for both generic and M1 refinement."""
    return max(1, min(4, (int(canvas_tokens) + 9) // 10))


def _segments(tokenizer: Any, prepared: Mapping[str, Any], middle_ids: Sequence[int]) -> dict[str, str]:
    prefix = tokenizer.decode(prepared["prefix_ids"], skip_special_tokens=True)
    middle = tokenizer.decode(list(middle_ids), skip_special_tokens=True)
    suffix = tokenizer.decode(prepared["suffix_ids"], skip_special_tokens=True)
    return {
        "prefix_text": prefix,
        "middle_text": middle,
        "suffix_text": suffix,
        "full_text": prefix + middle + suffix,
    }


def _diagnostics(task: CodeTask, segments: Mapping[str, str]) -> dict[str, Any]:
    parsed = parse_compile_diagnostics(str(segments["full_text"]), compile_mode="exec")
    return {
        "task_prefix_equals_decoded_prefix": task.prefix == segments["prefix_text"],
        "task_suffix_equals_decoded_suffix": task.suffix == segments["suffix_text"],
        "final_full_code_parse_passed": parsed["parse_passed"],
        "final_full_code_compile_passed": parsed["compile_passed"],
        "final_full_code_parse_error": parsed["parse_error"],
        "final_full_code_compile_error": parsed["compile_error"],
    }


def decode_fixed_canvas_state(
    *,
    task: CodeTask,
    tokenizer: Any,
    model: Any,
    cfg: Any,
    canvas_tokens: int,
    total_steps: int,
    phase_name: str,
    initial_middle_ids: Sequence[int] | None = None,
    initial_mask_indices: Sequence[int] | None = None,
    schedule_length: int | None = None,
    evaluate_after_decode: Callable[[str, str, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a fixed 64-step decode and retain the final token state locally.

    ``initial_middle_ids`` makes stage two a real refinement: only the supplied
    positions are remasked, and all other first-stage tokens are preserved as
    the initial state.  In new candidate-method runners, ``task`` is a
    prefix/suffix-only visible task and ``evaluate_after_decode`` constructs
    test-bearing evaluator state only after decoding has completed.  The
    legacy fallback is retained for historical runners.
    """
    if int(canvas_tokens) <= 0 or int(total_steps) <= 0:
        raise ValueError("canvas_tokens and total_steps must be positive")
    prepared = prepare_model_inputs(task, tokenizer, int(canvas_tokens), cfg)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    x_t = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=device)
    middle_start = int(prepared["middle_start"])
    middle_end = int(prepared["middle_end"])
    mask_token_id = int(prepared["mask_token_id"])
    remask_indices = sorted({int(index) for index in initial_mask_indices or []})
    if any(index < 0 or index >= int(canvas_tokens) for index in remask_indices):
        raise ValueError("initial_mask_indices contain an out-of-canvas position")
    if initial_middle_ids is not None:
        if len(initial_middle_ids) != int(canvas_tokens):
            raise ValueError("initial_middle_ids length must equal canvas_tokens")
        x_t[0, middle_start:middle_end] = torch.tensor(list(initial_middle_ids), dtype=torch.long, device=device)
        if remask_indices:
            x_t[0, [middle_start + index for index in remask_indices]] = mask_token_id

    selected_schedule_length = int(schedule_length or canvas_tokens)
    if selected_schedule_length <= 0:
        raise ValueError("schedule_length must be positive")
    decode_started = time.perf_counter()
    total_token_changes = 0
    effective_update_steps = 0
    final_confidences: list[float] = [0.0] * int(canvas_tokens)
    step_trace: list[dict[str, Any]] = []

    for step in range(int(total_steps)):
        before_middle = x_t[0, middle_start:middle_end].detach().clone()
        with torch.no_grad():
            outputs = model(x_t)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probabilities = torch.softmax(logits, dim=-1)
        max_probs, predictions = torch.max(probabilities, dim=-1)
        current_mask_idx = x_t == mask_token_id
        middle_probs = max_probs[:, middle_start:middle_end]
        middle_mask_idx = current_mask_idx[:, middle_start:middle_end]
        target_masks = linear_target_masks(selected_schedule_length, int(total_steps), step)
        if target_masks > 0:
            low_confidence = select_low_confidence_mask_positions(middle_probs, middle_mask_idx, target_masks)
            x_t[current_mask_idx] = predictions[current_mask_idx]
            if low_confidence:
                x_t[0, [middle_start + index for index in low_confidence]] = mask_token_id
        else:
            low_confidence = []
            x_t[current_mask_idx] = predictions[current_mask_idx]
        after_middle = x_t[0, middle_start:middle_end].detach().clone()
        token_changes = int((after_middle != before_middle).sum().item())
        if token_changes:
            effective_update_steps += 1
            total_token_changes += token_changes
        final_confidences = [float(value) for value in middle_probs[0].detach().cpu().tolist()]
        step_trace.append(
            {
                "phase": phase_name,
                "step": step,
                "target_masks": target_masks,
                "remaining_masks_before": int(middle_mask_idx.sum().item()),
                "remaining_masks_after": int((after_middle == mask_token_id).sum().item()),
                "token_change_count": token_changes,
                "low_confidence_indices": [int(index) for index in low_confidence],
            }
        )

    decode_sec = time.perf_counter() - decode_started
    final_middle_ids = [int(value) for value in x_t[0, middle_start:middle_end].detach().cpu().tolist()]
    segments = _segments(tokenizer, prepared, final_middle_ids)
    if evaluate_after_decode is None:
        verification = run_verifier_stack(
            task=task,
            full_code=str(segments["full_text"]),
            completion_without_suffix=str(segments["middle_text"]),
        )
        verification_payload = {name: result.to_dict() for name, result in verification.items()}
        verification_sec = sum(item.duration_sec for item in verification.values())
        tier3 = verification.get("tier3_unit_tests")
        passed = bool(tier3.passed) if tier3 else False
    else:
        evaluated = dict(
            evaluate_after_decode(
                str(segments["prefix_text"]),
                str(segments["middle_text"]),
                str(segments["suffix_text"]),
            )
        )
        verification_payload = dict(evaluated.get("verification") or {})
        verification_sec = float(evaluated.get("verification_sec") or 0.0)
        passed = bool(evaluated.get("passed", False))
    return {
        "code": segments["full_text"],
        "prefix_text": segments["prefix_text"],
        "middle_text": segments["middle_text"],
        "suffix_text": segments["suffix_text"],
        "middle_token_ids": final_middle_ids,
        "final_token_confidences": final_confidences,
        "metrics": {
            "passed": passed,
            "decode_sec": decode_sec,
            "verification_sec": verification_sec,
            "total_sec": decode_sec + verification_sec,
            "total_sec_including_probe": decode_sec + verification_sec,
            "total_steps": int(total_steps),
            "actual_forward_count": int(total_steps),
            "effective_update_steps": effective_update_steps,
            "total_token_changes": total_token_changes,
            "mean_final_confidence": sum(final_confidences) / len(final_confidences) if final_confidences else 0.0,
            "canvas_tokens": int(canvas_tokens),
            "remasked_initial_token_count": len(remask_indices),
        },
        "verification": verification_payload,
        "diagnostics": _diagnostics(task, segments),
        "trajectory": {
            "phase": phase_name,
            "step_trace": step_trace,
        },
    }


def generic_low_confidence_indices(
    confidences: Sequence[float],
    canvas_tokens: int,
    remask_count: int | None = None,
) -> list[int]:
    """Choose generic targets with the same intervention cardinality as M1.

    ``remask_count`` is optional for the standalone helper, but the M1 runner
    supplies the dependency-cone cardinality. Both refinements already use the
    same canvas and 64 model forwards; matching initially masked positions
    keeps their edit opportunity comparable as well.
    """
    if len(confidences) != int(canvas_tokens):
        raise ValueError("generic remask requires one confidence value per canvas token")
    count = remask_count_for_canvas(int(canvas_tokens)) if remask_count is None else int(remask_count)
    if count < 0:
        raise ValueError("remask_count must not be negative")
    count = min(count, int(canvas_tokens))
    return sorted(
        sorted(range(int(canvas_tokens)), key=lambda index: (float(confidences[index]), index))[:count]
    )


def decoded_token_ranges(tokenizer: Any, token_ids: Sequence[int]) -> tuple[str, list[tuple[int, int]]]:
    """Map decoded character intervals back to token positions without source code."""
    previous = ""
    ranges: list[tuple[int, int]] = []
    ids = [int(token_id) for token_id in token_ids]
    for index in range(len(ids)):
        current = tokenizer.decode(ids[: index + 1], skip_special_tokens=True)
        if current.startswith(previous):
            ranges.append((len(previous), len(current)))
        else:
            # Tokenizer cleanup can rarely break prefix monotonicity.  Mark this
            # token as unmappable rather than inventing a target span.
            ranges.append((-1, -1))
        previous = current
    return previous, ranges


_IGNORED_NAMES = set(dir(builtins)) | set(keyword.kwlist) | {"True", "False", "None"}


@dataclass(frozen=True)
class DependencyConePlan:
    """AST/def-use remasking plan for one candidate state.

    The plan contains only candidate state and prefix/suffix program text. It
    never reads evaluator outcomes, reference code, task identity, split
    information, or oracle metadata.
    """

    token_indices: tuple[int, ...]
    statement_spans: tuple[tuple[int, int], ...]
    required_names: tuple[str, ...]
    selected_statement_count: int
    unresolved_names: tuple[str, ...]
    reason: str

    @property
    def executable(self) -> bool:
        return bool(self.token_indices) and not self.unresolved_names and not self.reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_indices": list(self.token_indices),
            "statement_spans": [list(span) for span in self.statement_spans],
            "required_names": list(self.required_names),
            "selected_statement_count": self.selected_statement_count,
            "unresolved_names": list(self.unresolved_names),
            "reason": self.reason,
            "executable": self.executable,
        }


def _source_offset(source: str, lineno: int, column: int) -> int:
    if lineno <= 0:
        return 0
    lines = source.splitlines(keepends=True)
    return sum(len(line) for line in lines[: lineno - 1]) + int(column)


def _node_span(source: str, node: ast.AST) -> tuple[int, int] | None:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    col = getattr(node, "col_offset", None)
    end_col = getattr(node, "end_col_offset", None)
    if None in (lineno, end_lineno, col, end_col):
        return None
    start = _source_offset(source, int(lineno), int(col))
    end = _source_offset(source, int(end_lineno), int(end_col))
    return (start, end) if start < end else None


def _bound_prefix_names(prefix: str) -> set[str]:
    """Small lexical fallback because an infilling prefix can be incomplete."""
    names: set[str] = set()
    for match in re.finditer(r"\b(?:async\s+)?def\s+\w+\s*\((.*?)\)\s*(?:->[^:]*)?:", prefix, flags=re.S):
        for item in match.group(1).split(","):
            name = item.strip().lstrip("*").split(":", 1)[0].split("=", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", name):
                names.add(name)
    for pattern in (
        r"(?m)^\s*([A-Za-z_]\w*)\s*(?::[^=\n]+)?=",
        r"(?m)^\s*for\s+([A-Za-z_]\w*)\s+in\b",
        r"(?m)^\s*(?:def|class)\s+([A-Za-z_]\w*)\b",
        r"(?m)^\s*(?:import|from)\s+([A-Za-z_]\w*)\b",
    ):
        names.update(re.findall(pattern, prefix))
    return names


def _statement_defs_uses(statement: ast.stmt) -> tuple[set[str], set[str]]:
    """Return direct runtime definitions and loads for one candidate statement."""
    definitions: set[str] = set()
    uses: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if node.id in _IGNORED_NAMES:
                return
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                definitions.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                uses.add(node.id)

        def visit_arg(self, node: ast.arg) -> None:
            definitions.add(node.arg)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            definitions.add(node.name)
            for decorator in node.decorator_list:
                self.visit(decorator)
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    self.visit(default)
            if node.returns is not None:
                self.visit(node.returns)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            definitions.add(node.name)
            for decorator in node.decorator_list:
                self.visit(decorator)
            for base in node.bases:
                self.visit(base)
            for keyword_item in node.keywords:
                self.visit(keyword_item.value)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                definitions.add(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                definitions.add(alias.asname or alias.name)

    Visitor().visit(statement)
    return definitions, uses - definitions


def _candidate_statements(prefix: str, candidate: str, suffix: str) -> tuple[list[tuple[int, int, set[str], set[str]]], str]:
    source = prefix + candidate + suffix
    candidate_start = len(prefix)
    candidate_end = candidate_start + len(candidate)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], "candidate_full_program_ast_unavailable"
    statements: list[tuple[int, int, set[str], set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        span = _node_span(source, node)
        if span is None:
            continue
        start, end = span
        if start < candidate_start or end > candidate_end:
            continue
        definitions, uses = _statement_defs_uses(node)
        statements.append((start - candidate_start, end - candidate_start, definitions, uses))
    return sorted(statements, key=lambda item: (item[0], item[1])), ""


def dependency_cone_plan(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    prefix: str,
    suffix: str,
    dependency_names: Sequence[str],
) -> DependencyConePlan:
    """Trace suffix obligations backwards through AST statement def-use edges.

    For each suffix-required name, select the latest preceding candidate
    statement that defines it, then recursively select statements defining the
    RHS names used by that statement. Entire statement spans—not matching
    identifier occurrences—are mapped back to candidate token indices.
    """
    decoded, token_ranges = decoded_token_ranges(tokenizer, token_ids)
    required = tuple(sorted({str(name) for name in dependency_names if str(name)}))
    if not required:
        return DependencyConePlan((), (), required, 0, (), "no_suffix_dependency_obligation")
    statements, parse_reason = _candidate_statements(prefix, decoded, suffix)
    if parse_reason:
        return DependencyConePlan((), (), required, 0, (), parse_reason)
    prefix_names = _bound_prefix_names(prefix)
    selected_indices: set[int] = set()
    unresolved: set[str] = set()
    pending: list[tuple[str, int, bool]] = [(name, len(decoded), True) for name in required]
    visited: set[tuple[str, int, bool]] = set()

    while pending:
        name, before, direct_obligation = pending.pop()
        state = (name, before, direct_obligation)
        if state in visited or name in _IGNORED_NAMES:
            continue
        visited.add(state)
        providers = [
            (index, statement)
            for index, statement in enumerate(statements)
            if statement[0] < before and name in statement[2]
        ]
        if not providers:
            if direct_obligation or name not in prefix_names:
                unresolved.add(name)
            continue
        index, statement = max(providers, key=lambda item: (item[1][0], item[1][1], item[0]))
        if index in selected_indices:
            continue
        selected_indices.add(index)
        statement_start, _, _, uses = statement
        for dependency in sorted(uses):
            if dependency not in _IGNORED_NAMES:
                pending.append((dependency, statement_start, False))

    if unresolved:
        return DependencyConePlan(
            (),
            (),
            required,
            len(selected_indices),
            tuple(sorted(unresolved)),
            "missing_required_definition_in_candidate:" + ",".join(sorted(unresolved)),
        )
    spans = tuple(sorted((statements[index][0], statements[index][1]) for index in selected_indices))
    mapped: list[int] = []
    for token_index, (left, right) in enumerate(token_ranges):
        if left < 0 or right <= left:
            continue
        if any(left < span_right and span_left < right for span_left, span_right in spans):
            mapped.append(token_index)
    if not mapped:
        return DependencyConePlan((), spans, required, len(selected_indices), (), "dependency_cone_statement_not_mappable_to_candidate_tokens")
    return DependencyConePlan(tuple(sorted(set(mapped))), spans, required, len(selected_indices), (), "")


def dependency_cone_token_indices(
    tokenizer: Any,
    token_ids: Sequence[int],
    dependency_names: Sequence[str],
    *,
    prefix: str,
    suffix: str,
) -> list[int]:
    """Compatibility wrapper around the AST/def-use dependency-cone plan."""
    return list(
        dependency_cone_plan(
            tokenizer,
            token_ids,
            prefix=prefix,
            suffix=suffix,
            dependency_names=dependency_names,
        ).token_indices
    )
