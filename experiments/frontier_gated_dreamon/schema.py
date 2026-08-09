from __future__ import annotations

from typing import Any, Mapping


REQUIRED_RESULT_FIELDS = {
    "manifest_id",
    "manifest_role",
    "sample_id",
    "task_id",
    "base_task_id",
    "group_id",
    "w",
    "seed",
    "config_hash",
    "model_path",
    "model_revision",
    "initial_mask_count",
    "initial_dynamic_length",
    "final_dynamic_length",
    "completion_token_ids",
    "completion",
    "pass_at_1",
    "compiled",
    "completed",
    "status",
    "stop_reason",
    "forward_count",
    "wall_time_seconds",
    "gpu_time_seconds",
    "token_forwards",
    "step_trace",
    "normal_token_count",
    "expand_count",
    "delete_action_count",
    "single_point_delete_count",
    "broadcast_delete_count",
    "broadcast_delete_tail_lengths",
    "newline_token_count",
    "physical_line_count",
    "peak_unresolved_masks",
    "final_unresolved_masks",
    "exact_cycle_count",
    "oscillation_detected",
    "frontier_violation",
    "exception",
    "evaluable",
    "unevaluable_reason",
}


def validate_result_row(row: Mapping[str, Any], *, scored: bool) -> None:
    missing = sorted(REQUIRED_RESULT_FIELDS - set(row))
    if missing:
        raise ValueError(f"result row missing fields: {missing}")
    if int(row["initial_mask_count"]) != 64 or int(row["initial_dynamic_length"]) != 64:
        raise ValueError("every result must start from exactly 64 masks")
    trace = row["step_trace"]
    if not isinstance(trace, list) or not trace:
        raise ValueError("step_trace must be a non-empty list")
    step_zero = trace[0]
    if (
        step_zero.get("step") != 0
        or step_zero.get("kind") != "initial_state"
        or step_zero.get("dynamic_length") != 64
        or step_zero.get("unresolved_masks") != 64
    ):
        raise ValueError("trace step 0 does not prove one continuous 64-mask middle")
    violations = 0
    for step in trace[1:]:
        if step.get("kind") != "commit":
            raise ValueError("non-initial trace rows must be commits")
        eligible = step.get("eligible_positions")
        position = step.get("commit_position")
        if not isinstance(eligible, list) or position not in eligible:
            violations += 1
    if violations or int(row["frontier_violation"]) != 0:
        raise ValueError(f"frontier violation detected: trace={violations} row={row['frontier_violation']}")
    if row["w"] not in {"1", "4", "8", "16", "inf"}:
        raise ValueError(f"unexpected width: {row['w']}")
    if scored and row["exception"] is None and not isinstance(row["pass_at_1"], bool):
        raise ValueError("scored non-exception row must have boolean pass_at_1")
