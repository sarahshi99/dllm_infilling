from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# =========================
# 旧实现（保留对照，不再直接使用）
# =========================
# 旧版只有这一句：
# from human_eval.execution import check_correctness
#
# 问题：
# 1. 这是普通 completion 的 harness，不是 infilling harness；
# 2. 对当前任务，最稳妥的是优先使用 human_eval_infilling.execution；
# 3. 如果环境里没有该包，再退回兼容模式。
#
# 删除原因：
# 不能继续把“普通 HumanEval harness”当成 infilling 的首选执行器。
#
# from human_eval.execution import check_correctness

# 新实现：
# 优先使用 infilling harness；如果环境里没有，则退回标准 harness。
try:
    from human_eval_infilling.execution import check_correctness as infilling_check_correctness
    _INFILLING_HARNESS_AVAILABLE = True
except Exception:
    infilling_check_correctness = None
    _INFILLING_HARNESS_AVAILABLE = False

from human_eval.execution import check_correctness as standard_check_correctness

from .data_loader import CodeTask


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
    patterns = [
        r"line (\d+)",
        r"File \"<string>\", line (\d+)",
        r"File \".*?\", line (\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))
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
    except Exception as e:
        diagnostics["parse_error"] = f"{type(e).__name__}: {e}"

    try:
        compile(code, "<string>", compile_mode)
        diagnostics["compile_passed"] = True
    except Exception as e:
        diagnostics["compile_error"] = f"{type(e).__name__}: {e}"

    return diagnostics


def tier1_parse_and_compile(code: str, compile_mode: str = "exec") -> VerificationResult:
    start = time.perf_counter()
    try:
        tree = ast.parse(code)
        compile(code, "<string>", compile_mode)
        return VerificationResult(
            tier="tier1_parse_compile",
            passed=True,
            duration_sec=time.perf_counter() - start,
            metadata={"ast_type": type(tree).__name__},
        )
    except SyntaxError as e:
        return VerificationResult(
            tier="tier1_parse_compile",
            passed=False,
            error_type="SyntaxError",
            error_message=str(e),
            line=getattr(e, "lineno", None),
            col=getattr(e, "offset", None),
            duration_sec=time.perf_counter() - start,
        )
    except Exception as e:
        return VerificationResult(
            tier="tier1_parse_compile",
            passed=False,
            error_type=type(e).__name__,
            error_message=str(e),
            duration_sec=time.perf_counter() - start,
        )


def tier2_smoke_exec(code: str) -> VerificationResult:
    start = time.perf_counter()
    namespace: Dict[str, Any] = {}
    try:
        compiled = compile(code, "<string>", "exec")
        exec(compiled, namespace, namespace)
        return VerificationResult(
            tier="tier2_smoke_exec",
            passed=True,
            duration_sec=time.perf_counter() - start,
        )
    except Exception as e:
        tb = repr(e)
        line = _extract_traceback_line(tb)
        return VerificationResult(
            tier="tier2_smoke_exec",
            passed=False,
            error_type=type(e).__name__,
            error_message=str(e),
            line=line,
            traceback_text=tb,
            duration_sec=time.perf_counter() - start,
        )


# =========================
# 旧实现（保留对照，不再直接使用）
# =========================
# 旧版思路：
# - 直接继续使用标准 human_eval.execution
# - 手工把 suffix 拼进 completion_with_suffix
#
# 问题：
# - 这只是兼容方案，不是最优；
# - 如果环境里已经有 human_eval_infilling.execution，就应该优先走 infilling harness。
#
# def tier3_unit_tests(task: CodeTask, completion_without_suffix: str, timeout: float = 3.0) -> VerificationResult:
#     start = time.perf_counter()
#     completion_with_suffix = completion_without_suffix + task.suffix
#     problem = {
#         "task_id": task.task_id,
#         "prompt": task.prefix,
#         "test": task.test_code,
#         "entry_point": task.entry_point,
#     }
#     approx_check_program = (
#         task.prefix
#         + completion_with_suffix
#         + "\n"
#         + task.test_code
#         + "\n"
#         + f"check({task.entry_point})"
#     )
#     approx_diag = parse_compile_diagnostics(approx_check_program, compile_mode="exec")
#     result = standard_check_correctness(problem, completion_with_suffix, timeout=timeout)
#     ...

def tier3_unit_tests(task: CodeTask, completion_without_suffix: str, timeout: float = 3.0) -> VerificationResult:
    """
    新实现：
    1. 如果 human_eval_infilling.execution 可用，优先使用它；
    2. 否则回退到标准 human_eval.execution，并显式把 suffix 拼进 completion；
    3. 无论走哪条路径，都把“近似执行程序”的 parse/compile 诊断写入 metadata。
    """
    start = time.perf_counter()

    completion_with_suffix = completion_without_suffix + task.suffix

    # 新增：
    # 不同 harness 下，近似 check_program 的构造方式不同，
    # 但都要显式落日志，便于人工检查。
    if _INFILLING_HARNESS_AVAILABLE:
        harness_name = "human_eval_infilling.execution"
        problem = {
            "task_id": task.task_id,
            "prompt": task.prefix,
            "suffix": task.suffix,
            "test": task.test_code,
            "entry_point": task.entry_point,
        }
        approx_check_program = (
            task.prefix
            + completion_without_suffix
            + task.suffix
            + "\n"
            + task.test_code
            + "\n"
            + f"check({task.entry_point})"
        )
        approx_diag = parse_compile_diagnostics(approx_check_program, compile_mode="exec")
        result = infilling_check_correctness(problem, completion_without_suffix, timeout=timeout)
    else:
        harness_name = "human_eval.execution_fallback_with_suffix_wrapper"
        problem = {
            "task_id": task.task_id,
            "prompt": task.prefix,
            "test": task.test_code,
            "entry_point": task.entry_point,
        }
        approx_check_program = (
            task.prefix
            + completion_with_suffix
            + "\n"
            + task.test_code
            + "\n"
            + f"check({task.entry_point})"
        )
        approx_diag = parse_compile_diagnostics(approx_check_program, compile_mode="exec")
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
            "raw_result": raw_text,
            "harness_name": harness_name,
            "prompt_used_for_harness": task.prefix,
            "suffix_used_for_harness": task.suffix,
            "completion_without_suffix": completion_without_suffix,
            "completion_with_suffix": completion_with_suffix,
            "entry_point": task.entry_point,
            "approx_check_program": approx_check_program,
            "approx_check_program_parse_passed": approx_diag["parse_passed"],
            "approx_check_program_compile_passed": approx_diag["compile_passed"],
            "approx_check_program_parse_error": approx_diag["parse_error"],
            "approx_check_program_compile_error": approx_diag["compile_error"],
        },
    )


# =========================
# 旧实现（保留对照，不再直接使用）
# =========================
# 旧版 conceptual 上已经拆成：
# - Tier1/Tier2 用 full_code
# - Tier3 用 completion_without_suffix
# 这点是对的，所以这里只保留旧注释，不做大改。
#
# def run_verifier_stack(
#     task: CodeTask,
#     full_code: str,
#     completion_without_suffix: str,
#     timeout: float = 3.0,
# ) -> Dict[str, VerificationResult]:
#     ...

def run_verifier_stack(
    task: CodeTask,
    full_code: str,
    completion_without_suffix: str,
    timeout: float = 3.0,
) -> Dict[str, VerificationResult]:
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