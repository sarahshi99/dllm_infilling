from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StructuralUnit:
    unit_type: str
    char_start: int
    char_end: int
    line_start: int
    line_end: int
    reason: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


STATEMENT_NODE_TYPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Expr,
)

EXPRESSION_NODE_TYPES = (
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.IfExp,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.Attribute,
    ast.Subscript,
    ast.Lambda,
    ast.Constant,
    ast.Name,
)

TINY_SEMANTIC_NODE_TYPES = (
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Store,
)

CONTROL_FLOW_HEADER_RE = re.compile(
    r"^\s*(for\b|if\b|elif\b|else\s*:|while\b|try\s*:|except\b|finally\s*:|with\b)"
)


def build_line_offsets(text: str) -> List[int]:
    offsets = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def line_col_to_char(text: str, line: int, col: int) -> int:
    if line <= 0:
        return 0
    offsets = build_line_offsets(text)
    if line > len(offsets):
        return len(text)
    return min(offsets[line - 1] + max(col, 0), len(text))


def char_to_line(text: str, char_index: int) -> int:
    char_index = min(max(char_index, 0), len(text))
    offsets = build_line_offsets(text)
    line = 1
    for idx, start in enumerate(offsets, start=1):
        if start > char_index:
            break
        line = idx
    return line


def node_to_char_span(text: str, node: ast.AST) -> Tuple[int, int, int, int]:
    line_start = getattr(node, "lineno", 1)
    col_start = getattr(node, "col_offset", 0)
    line_end = getattr(node, "end_lineno", line_start)
    col_end = getattr(node, "end_col_offset", col_start)
    char_start = line_col_to_char(text, line_start, col_start)
    char_end = line_col_to_char(text, line_end, col_end)
    if char_end <= char_start:
        char_end = min(len(text), char_start + 1)
    return char_start, char_end, line_start, line_end


def parse_python_ast(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def build_parent_map(tree: ast.AST) -> Dict[ast.AST, Optional[ast.AST]]:
    parent_map: Dict[ast.AST, Optional[ast.AST]] = {tree: None}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    return parent_map


def structural_unit_from_node(code: str, node: ast.AST, reason: str) -> StructuralUnit:
    char_start, char_end, line_start, line_end = node_to_char_span(code, node)
    return StructuralUnit(
        unit_type="statement" if isinstance(node, STATEMENT_NODE_TYPES) else "subtree",
        char_start=char_start,
        char_end=char_end,
        line_start=line_start,
        line_end=line_end,
        reason=reason,
        metadata={"ast_type": type(node).__name__, "selection_source": "ast"},
    )


def promote_tiny_semantic_node(
    code: str,
    node: ast.AST,
    parent_map: Dict[ast.AST, Optional[ast.AST]],
    min_char_span: int = 8,
) -> Tuple[ast.AST, List[str]]:
    promoted = node
    chain = [type(promoted).__name__]

    while True:
        char_start, char_end, _, _ = node_to_char_span(code, promoted)
        char_span = max(1, char_end - char_start)
        is_tiny_type = isinstance(promoted, TINY_SEMANTIC_NODE_TYPES)

        if not is_tiny_type and char_span >= min_char_span:
            break

        parent = parent_map.get(promoted)
        if parent is None:
            break
        if isinstance(parent, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            break

        promoted = parent
        chain.append(type(promoted).__name__)

        char_start, char_end, _, _ = node_to_char_span(code, promoted)
        char_span = max(1, char_end - char_start)
        if isinstance(promoted, EXPRESSION_NODE_TYPES + STATEMENT_NODE_TYPES) and char_span >= min_char_span:
            break

    return promoted, chain


# 旧实现（保留对照，不再直接使用）：
# 旧版只返回最小包围 AST 节点，不会额外处理“太小 subtree 要继续上提”，
# 对 suffix 边界附近的控制流片段也没有专门单元。
# def find_smallest_enclosing_node(code: str, target_line: int, prefer: str = "statement") -> Optional[StructuralUnit]:
#     tree = parse_python_ast(code)
#     if tree is None:
#         return None
#     ...
def find_smallest_enclosing_node(code: str, target_line: int, prefer: str = "statement") -> Optional[StructuralUnit]:
    tree = parse_python_ast(code)
    if tree is None:
        return None

    parent_map = build_parent_map(tree)

    if prefer == "statement":
        primary_types = STATEMENT_NODE_TYPES
        secondary_types = ()
    else:
        primary_types = EXPRESSION_NODE_TYPES
        secondary_types = STATEMENT_NODE_TYPES

    def collect_raw(node_types) -> List[Tuple[int, ast.AST]]:
        candidates: List[Tuple[int, ast.AST]] = []
        for node in ast.walk(tree):
            if not isinstance(node, node_types):
                continue
            if not hasattr(node, "lineno"):
                continue
            line_start = getattr(node, "lineno", 1)
            line_end = getattr(node, "end_lineno", line_start)
            if line_start <= target_line <= line_end:
                char_start, char_end, _, _ = node_to_char_span(code, node)
                span = max(1, char_end - char_start)
                candidates.append((span, node))
        candidates.sort(key=lambda x: x[0])
        return candidates

    raw_candidates = collect_raw(primary_types)
    if not raw_candidates and secondary_types:
        raw_candidates = collect_raw(secondary_types)
    if not raw_candidates:
        return None

    best_unit: Optional[StructuralUnit] = None
    best_span: Optional[int] = None
    for _, raw_node in raw_candidates:
        promoted_node = raw_node
        promotion_chain: List[str] = [type(raw_node).__name__]
        if prefer != "statement":
            promoted_node, promotion_chain = promote_tiny_semantic_node(
                code=code,
                node=raw_node,
                parent_map=parent_map,
                min_char_span=8,
            )

        unit = structural_unit_from_node(code, promoted_node, reason=f"ast_enclosing_line_{target_line}")
        unit.metadata = dict(unit.metadata or {})
        unit.metadata.update({
            "promotion_chain": promotion_chain,
            "requested_prefer": prefer,
        })
        span = max(1, unit.char_end - unit.char_start)
        if best_unit is None or span < (best_span or 10**9):
            best_unit = unit
            best_span = span

    return best_unit


def line_span(code: str, error_line: int, context_before: int = 0, context_after: int = 0) -> StructuralUnit:
    lines = code.splitlines(keepends=True)
    if not lines:
        return StructuralUnit("statement", 0, 0, 1, 1, "empty_code")

    idx = min(max(error_line - 1, 0), len(lines) - 1)
    start_idx = max(0, idx - context_before)
    end_idx = min(len(lines) - 1, idx + context_after)

    offsets = build_line_offsets(code)
    char_start = offsets[start_idx]
    char_end = offsets[end_idx + 1] if end_idx + 1 < len(offsets) else len(code)
    return StructuralUnit(
        unit_type="statement",
        char_start=char_start,
        char_end=max(char_start + 1, char_end),
        line_start=start_idx + 1,
        line_end=end_idx + 1,
        reason=f"line_span_line_{error_line}",
        metadata={
            "context_before": context_before,
            "context_after": context_after,
            "selection_source": "line_span",
        },
    )


def window_span(code: str, error_line: int) -> StructuralUnit:
    return line_span(code, error_line, context_before=0, context_after=1)


def indentation_block_span(code: str, error_line: int) -> StructuralUnit:
    lines = code.splitlines(keepends=True)
    if not lines:
        return StructuralUnit("statement", 0, 0, 1, 1, "empty_code")

    idx = min(max(error_line - 1, 0), len(lines) - 1)

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    target_indent = indent_of(lines[idx]) if lines[idx].strip() else 0
    start_idx = idx
    while start_idx > 0:
        prev = lines[start_idx - 1]
        if prev.strip() == "":
            start_idx -= 1
            continue
        if indent_of(prev) < target_indent:
            break
        start_idx -= 1

    end_idx = idx
    while end_idx + 1 < len(lines):
        nxt = lines[end_idx + 1]
        if nxt.strip() == "":
            end_idx += 1
            continue
        if indent_of(nxt) < target_indent:
            break
        end_idx += 1

    offsets = build_line_offsets(code)
    char_start = offsets[start_idx]
    char_end = offsets[end_idx + 1] if end_idx + 1 < len(offsets) else len(code)
    return StructuralUnit(
        unit_type="statement",
        char_start=char_start,
        char_end=char_end,
        line_start=start_idx + 1,
        line_end=end_idx + 1,
        reason=f"indentation_block_line_{error_line}",
        metadata={"indent_level": target_indent, "selection_source": "indentation_block"},
    )


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _control_flow_header_kind(line: str) -> Optional[str]:
    m = CONTROL_FLOW_HEADER_RE.match(line.rstrip("\n"))
    if not m:
        return None
    header = m.group(1)
    # 统一一下 else:/try:/finally: 这类格式，便于下游统计
    return header.replace(" :", "").replace(":", "")


def _is_control_flow_header(line: str) -> bool:
    return bool(CONTROL_FLOW_HEADER_RE.match(line.rstrip("\n")))


# 新增：为 suffix 边界附近的 for/if/else 小片段构造控制流修复单元。
# 目的不是替代 subtree，而是在 subtree 与整块 statement 之间增加一个更对症的中间粒度。
def control_flow_fragment_span(
    code: str,
    anchor_line: int,
    lookaround: int = 1,
    max_body_lines: int = 3,
) -> Optional[StructuralUnit]:
    lines = code.splitlines(keepends=True)
    if not lines:
        return None

    anchor_idx = min(max(anchor_line - 1, 0), len(lines) - 1)
    candidate_indices: List[Tuple[int, int]] = []
    for idx in range(max(0, anchor_idx - lookaround), min(len(lines), anchor_idx + lookaround + 1)):
        if _is_control_flow_header(lines[idx]):
            candidate_indices.append((abs(idx - anchor_idx), idx))

    if not candidate_indices:
        return None

    candidate_indices.sort(key=lambda x: x[0])
    header_idx = candidate_indices[0][1]
    base_indent = _indent_of(lines[header_idx])

    end_idx = header_idx
    body_nonblank = 0
    saw_body = False
    for idx in range(header_idx + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()

        if stripped == "":
            end_idx = idx
            continue

        curr_indent = _indent_of(line)
        if curr_indent <= base_indent and saw_body:
            break

        end_idx = idx

        if curr_indent > base_indent:
            saw_body = True
            body_nonblank += 1
            if body_nonblank >= max_body_lines:
                break
        elif curr_indent <= base_indent and not saw_body:
            break

    if end_idx == header_idx and header_idx + 1 < len(lines):
        end_idx = header_idx + 1

    offsets = build_line_offsets(code)
    char_start = offsets[header_idx]
    char_end = offsets[end_idx + 1] if end_idx + 1 < len(offsets) else len(code)

    header_text = lines[header_idx].rstrip("\n")
    header_keyword = _control_flow_header_kind(lines[header_idx])

    return StructuralUnit(
        unit_type="control_flow_fragment",
        char_start=char_start,
        char_end=max(char_start + 1, char_end),
        line_start=header_idx + 1,
        line_end=end_idx + 1,
        reason=f"boundary_control_flow_line_{header_idx + 1}",
        metadata={
            "selection_source": "control_flow_fragment",
            "anchor_line": anchor_line,
            "header_line": header_idx + 1,
            "header_keyword": header_keyword,
            "header_text": header_text,
            "max_body_lines": max_body_lines,
        },
    )


def choose_structural_unit_for_error(
    code: str,
    error_line: Optional[int],
    error_col: Optional[int],
    prefer: str,
    fallback_reason: str,
    fallback_mode: str = "block",
) -> StructuralUnit:
    if error_line is None:
        return StructuralUnit(
            unit_type="statement",
            char_start=0,
            char_end=len(code),
            line_start=1,
            line_end=max(1, code.count("\n") + 1),
            reason=fallback_reason,
            metadata={
                "selection_source": "full_code_fallback",
                "fallback_mode": fallback_mode,
                "requested_prefer": prefer,
                "error_col": error_col,
            },
        )

    unit = find_smallest_enclosing_node(code, target_line=error_line, prefer=prefer)
    if unit is not None:
        meta = dict(unit.metadata or {})
        meta.update({
            "requested_prefer": prefer,
            "fallback_mode": fallback_mode,
            "error_col": error_col,
        })
        unit.metadata = meta
        return unit

    if fallback_mode == "line":
        unit = line_span(code, error_line, context_before=0, context_after=0)
    elif fallback_mode == "window":
        unit = window_span(code, error_line)
    else:
        unit = indentation_block_span(code, error_line)

    meta = dict(unit.metadata or {})
    meta.update({
        "requested_prefer": prefer,
        "fallback_mode": fallback_mode,
        "fallback_reason": fallback_reason,
        "error_col": error_col,
        "ast_lookup": "miss",
    })
    unit.metadata = meta
    return unit


def diagnose_full_span_projection(prefix_text: str, middle_text: str, unit: StructuralUnit) -> Dict[str, Any]:
    middle_start = len(prefix_text)
    middle_end = middle_start + len(middle_text)
    overlap_start = max(unit.char_start, middle_start)
    overlap_end = min(unit.char_end, middle_end)
    if overlap_end <= overlap_start:
        if unit.char_end <= middle_start:
            case = "left_of_middle"
        elif unit.char_start >= middle_end:
            case = "right_of_middle"
        else:
            case = "no_overlap"
    else:
        case = "overlap"
    return {
        "middle_start_char": middle_start,
        "middle_end_char": middle_end,
        "unit_char_start": unit.char_start,
        "unit_char_end": unit.char_end,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "projection_case": case,
    }


def full_span_to_middle_relative(prefix_text: str, middle_text: str, unit: StructuralUnit) -> Optional[StructuralUnit]:
    diag = diagnose_full_span_projection(prefix_text, middle_text, unit)
    if diag["projection_case"] != "overlap":
        return None

    start = diag["overlap_start"]
    end = diag["overlap_end"]
    return StructuralUnit(
        unit_type=unit.unit_type,
        char_start=start - len(prefix_text),
        char_end=end - len(prefix_text),
        line_start=unit.line_start,
        line_end=unit.line_end,
        reason=unit.reason,
        metadata=dict(unit.metadata or {}, projection_case=diag["projection_case"]),
    )


def middle_char_span_to_token_indices(tokenizer, middle_text: str, middle_length_tokens: int, char_start: int, char_end: int) -> List[int]:
    if middle_length_tokens <= 0:
        return []
    if char_end <= char_start:
        char_end = min(len(middle_text), char_start + 1)

    try:
        encoded = tokenizer(middle_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        if offsets and len(offsets) == middle_length_tokens:
            hits: List[int] = []
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_end <= char_start or tok_start >= char_end:
                    continue
                hits.append(idx)
            if hits:
                return hits
    except Exception:
        pass

    text_len = max(1, len(middle_text))
    start_idx = max(0, min(middle_length_tokens - 1, int(math.floor(middle_length_tokens * char_start / text_len))))
    end_idx = max(start_idx + 1, int(math.ceil(middle_length_tokens * char_end / text_len)))
    end_idx = min(middle_length_tokens, end_idx)
    return list(range(start_idx, end_idx))