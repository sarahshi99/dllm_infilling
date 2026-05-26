from __future__ import annotations

import ast
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .dataset import CodeTask


@dataclass
class VerificationResult:
    tier: str
    passed: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    traceback_text: Optional[str] = None
    duration_sec: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_traceback_line(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    patterns = [r"line (\d+)", r"File \"<string>\", line (\d+)", r"File \".*?\", line (\d+)"]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def parse_compile_diagnostics(code: str, compile_mode: str = "exec") -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "parse_passed": False,
        "compile_passed": False,
        "parse_error": None,
        "compile_error": None,
    }
    try:
        ast.parse(code)
        diagnostics["parse_passed"] = True
    except Exception as exc:
        diagnostics["parse_error"] = f"{type(exc).__name__}: {exc}"

    try:
        compile(code, "<string>", compile_mode)
        diagnostics["compile_passed"] = True
    except Exception as exc:
        diagnostics["compile_error"] = f"{type(exc).__name__}: {exc}"
    return diagnostics


def tier1_parse_and_compile(code: str, compile_mode: str = "exec") -> VerificationResult:
    start = time.perf_counter()
    try:
        ast.parse(code)
        compile(code, "<string>", compile_mode)
        return VerificationResult(tier="tier1_parse_compile", passed=True, duration_sec=time.perf_counter() - start)
    except SyntaxError as exc:
        return VerificationResult(
            tier="tier1_parse_compile",
            passed=False,
            error_type="SyntaxError",
            error_message=str(exc),
            line=getattr(exc, "lineno", None),
            col=getattr(exc, "offset", None),
            duration_sec=time.perf_counter() - start,
        )
    except Exception as exc:
        return VerificationResult(
            tier="tier1_parse_compile",
            passed=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            duration_sec=time.perf_counter() - start,
        )


def tier2_smoke_exec(code: str) -> VerificationResult:
    start = time.perf_counter()
    namespace: Dict[str, Any] = {}
    try:
        compiled = compile(code, "<string>", "exec")
        exec(compiled, namespace, namespace)
        return VerificationResult(tier="tier2_smoke_exec", passed=True, duration_sec=time.perf_counter() - start)
    except Exception as exc:
        tb = repr(exc)
        return VerificationResult(
            tier="tier2_smoke_exec",
            passed=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            line=_extract_traceback_line(tb),
            traceback_text=tb,
            duration_sec=time.perf_counter() - start,
        )


def tier3_unit_tests(task: CodeTask, completion_without_suffix: str, timeout: float = 3.0) -> VerificationResult:
    start = time.perf_counter()
    completion_with_suffix = completion_without_suffix + task.suffix

    try:
        from human_eval_infilling.execution import check_correctness as infilling_check_correctness
        infilling_available = True
    except Exception:
        infilling_check_correctness = None
        infilling_available = False

    from human_eval.execution import check_correctness as standard_check_correctness

    if infilling_available:
        harness_name = "human_eval_infilling.execution"
        problem = {
            "task_id": task.task_id,
            "prompt": task.prefix,
            "suffix": task.suffix,
            "test": task.test_code,
            "entry_point": task.entry_point,
        }
        result = infilling_check_correctness(problem, completion_without_suffix, timeout=timeout)
    else:
        harness_name = "human_eval.execution_fallback_with_suffix_wrapper"
        problem = {
            "task_id": task.task_id,
            "prompt": task.prefix,
            "test": task.test_code,
            "entry_point": task.entry_point,
        }
        result = standard_check_correctness(problem, completion_with_suffix, timeout=timeout)

    passed = bool(result.get("passed", False))
    raw_text = result.get("result")
    return VerificationResult(
        tier="tier3_unit_tests",
        passed=passed,
        error_type=None if passed else "UnitTestFailure",
        error_message=None if passed else str(raw_text),
        line=_extract_traceback_line(raw_text),
        traceback_text=None if passed else str(raw_text),
        duration_sec=time.perf_counter() - start,
        metadata={
            "harness_name": harness_name,
            "completion_without_suffix": completion_without_suffix,
            "completion_with_suffix": completion_with_suffix,
        },
    )


def run_verifier_stack(task: CodeTask, full_code: str, completion_without_suffix: str, timeout: float = 3.0) -> Dict[str, VerificationResult]:
    results: Dict[str, VerificationResult] = {}
    tier1 = tier1_parse_and_compile(full_code)
    results[tier1.tier] = tier1
    if not tier1.passed:
        return results

    tier2 = tier2_smoke_exec(full_code)
    results[tier2.tier] = tier2
    if not tier2.passed:
        return results

    tier3 = tier3_unit_tests(task, completion_without_suffix, timeout=timeout)
    results[tier3.tier] = tier3
    return results
