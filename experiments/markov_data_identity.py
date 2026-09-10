"""Conservative, inspectable identities for code-data leakage audits.

No new content hashes. Equality keys are normalized text / Python AST strings.
Matches identify possible leakage, not semantic equivalence of arbitrary code.
"""
from __future__ import annotations

import ast
import unicodedata
from collections import defaultdict
from typing import Any, Mapping, Sequence


def text_key(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    return "\n".join(x.rstrip() for x in value.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip("\n")


def response_code(record: Mapping[str, Any]) -> str:
    response = str(record.get("output") or "")
    marker = "```python" if "```python" in response else "```"
    if marker not in response:
        return str(record.get("code") or "")
    code = response.split(marker, 1)[1]
    if code.startswith("\n"):
        code = code[1:]
    return code.split("```", 1)[0]


class NormalizeCode(ast.NodeTransformer):
    def __init__(self, entry: str = ""):
        self.entry = entry

    def strip_doc(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    visit_Module = strip_doc
    visit_ClassDef = strip_doc

    def visit_FunctionDef(self, node):
        self.strip_doc(node)
        if self.entry and node.name == self.entry:
            node.name = "__entry__"
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node):
        if self.entry and node.id in {self.entry, "candidate"}:
            node.id = "__entry__"
        return node


def code_key(source: str, entry: str = "") -> str | None:
    try:
        tree = NormalizeCode(entry).visit(ast.parse(source))
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    return ast.dump(tree, include_attributes=False)


def assert_keys(source: str, entry: str) -> set[str]:
    try:
        tree = NormalizeCode(entry).visit(ast.parse(source))
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return set()
    return {ast.dump(n.test, include_attributes=False) for n in ast.walk(tree) if isinstance(n, ast.Assert)}


def human_base_task(row: Mapping[str, Any]) -> str:
    return "/".join(str(row["task_id"]).split("/")[:3])


def human_full_source(row: Mapping[str, Any]) -> str:
    """Reconstruct the complete program before any normalization."""
    return (
        str(row.get("prompt") or "")
        + str(row.get("canonical_solution") or "")
        + str(row.get("suffix") or "")
    )


def human_variant_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[human_base_task(row)].append(row)
    parse_failures = []
    inconsistent = []
    for task, variants in sorted(grouped.items()):
        keys = []
        for row in variants:
            key = code_key(human_full_source(row), str(row.get("entry_point") or ""))
            if key is None:
                parse_failures.append(str(row["task_id"]))
            else:
                keys.append((str(row["task_id"]), key))
        if len({key for _, key in keys}) > 1:
            inconsistent.append({"base_task": task, "variant_ids": [task_id for task_id, _ in keys]})
    return {
        "base_tasks": len(grouped),
        "variants": len(rows),
        "parse_failures": parse_failures,
        "inconsistent_base_tasks": inconsistent,
    }


def identity_keys(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Same instructions OR same code connect records despite changed tests.

    Code in the actual response is checked as well as the auxiliary code field.
    Tests connect only with entry point, avoiding generic assert over-grouping.
    """
    keys = []
    instruction = text_key(record.get("instruction"))
    if instruction:
        keys.append(("instruction", instruction))
    for source in {str(record.get("code") or ""), response_code(record)}:
        if text_key(source):
            keys.append(("code_text", text_key(source)))
            tree = code_key(source)
            if tree:
                keys.append(("code_ast_no_doc", tree))
    entry = str(record.get("entry_point") or "")
    tests = record.get("testcase") or []
    if isinstance(tests, str):
        tests = [tests]
    if tests:
        keys.append(("tests", entry + "\n" + "\n".join(sorted(text_key(t) for t in tests))))
    return keys


def connected_groups(records: Sequence[Mapping[str, Any]]) -> list[int]:
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, record in enumerate(records):
        for key in identity_keys(record):
            if key in owner:
                a, b = find(i), find(owner[key])
                parent[max(a, b)] = min(a, b)
            else:
                owner[key] = i
    return [find(i) for i in range(len(records))]


def human_index(rows: Sequence[Mapping[str, Any]]) -> dict:
    consistency = human_variant_consistency(rows)
    if consistency["parse_failures"] or consistency["inconsistent_base_tasks"]:
        raise ValueError(f"HumanEval variant reconstruction mismatch: {consistency}")
    codes, assertions = {}, {}
    seen = set()
    for row in rows:
        task = human_base_task(row)
        if task in seen:
            continue
        seen.add(task)
        entry = str(row.get("entry_point") or "")
        key = code_key(human_full_source(row), entry)
        codes.setdefault(key, set()).add(task)
        for key in assert_keys(str(row.get("test") or ""), entry):
            assertions.setdefault((entry, key), set()).add(task)
    return {
        "codes": codes,
        "assertions": assertions,
        "base_tasks": len(seen),
        "variant_consistency": consistency,
    }


def human_matches(record: Mapping[str, Any], index: Mapping) -> list[dict]:
    found = set()
    entry = str(record.get("entry_point") or "")
    for field, source in (("code", str(record.get("code") or "")), ("response_code", response_code(record))):
        key = code_key(source, entry)
        for task in index["codes"].get(key, ()):
            found.add((task, field + "_ast_no_doc_entry_normalized"))
    tests = record.get("testcase") or []
    if isinstance(tests, str):
        tests = [tests]
    for test in tests:
        for key in assert_keys(test, entry):
            for task in index["assertions"].get((entry, key), ()):
                found.add((task, "same_entry_normalized_assert_candidate"))
    return [{"task_id": task, "reason": reason} for task, reason in sorted(found)]
