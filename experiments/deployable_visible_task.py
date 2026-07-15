"""Strict input boundary between deployable decoding and post-hoc evaluation.

Candidate methods may receive only the infilling prefix and suffix while they
select or decode a completion.  Test code and the evaluator entry point are
materialized only after a completion has been produced.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from expvision_dllm_clean.dataset import CodeTask
from expvision_dllm_clean.verifier import run_verifier_stack


def visible_task(prefix: str, suffix: str, *, method: str) -> CodeTask:
    """Build the only task object a deployable decoder may receive."""
    prefix, suffix = str(prefix), str(suffix)
    return CodeTask(
        task_id=f"{method}_visible_input",
        prefix=prefix,
        suffix=suffix,
        full_prompt=prefix + "<FILL_ME>" + suffix,
        test_code="",
        entry_point="",
        canonical_solution=None,
        raw={},
    )


def evaluator_task_after_decode(
    *, prefix: str, suffix: str, source_row: Mapping[str, Any], method: str
) -> CodeTask:
    """Materialize test-bearing state only after completion generation."""
    return CodeTask(
        task_id=f"{method}_post_decode_evaluator",
        prefix=str(prefix),
        suffix=str(suffix),
        full_prompt=str(prefix) + "<FILL_ME>" + str(suffix),
        test_code=str(source_row["test"]),
        entry_point=str(source_row["entry_point"]),
        canonical_solution=None,
        raw={},
    )


def evaluate_with_evaluator_task(
    *, task: CodeTask, prefix: str, suffix: str, middle: str
) -> dict[str, Any]:
    """Run the post-generation verifier for an already constructed task."""
    started = time.perf_counter()
    verification = run_verifier_stack(
        task=task,
        full_code=str(prefix) + str(middle) + str(suffix),
        completion_without_suffix=str(middle),
    )
    tier3 = verification.get("tier3_unit_tests")
    return {
        "passed": bool(tier3.passed) if tier3 else False,
        "verification_sec": time.perf_counter() - started,
        "verification": {name: value.to_dict() for name, value in verification.items()},
    }


def evaluate_completion_after_decode(
    *,
    prefix: str,
    suffix: str,
    middle: str,
    source_row: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    """Construct evaluator-only state after a candidate completion is fixed."""
    return evaluate_with_evaluator_task(
        task=evaluator_task_after_decode(
            prefix=prefix, suffix=suffix, source_row=source_row, method=method
        ),
        prefix=prefix,
        suffix=suffix,
        middle=middle,
    )
