#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "repro_results/dreamon_progressive_v3_hard_budgeted_protocol/protocol.json"
META = ROOT / "manifests/v3_pure_newline_blank30.meta.json"

METHODS = {
    "v3_c0_budgeted_oneshot_pure_newline_veto": {
        "protocol_name": "dreamon_v3_c0_budgeted_oneshot_pure_newline_veto",
        "mode": "oneshot_veto",
        "role": "targeted mechanism control: veto one exact pure-newline blank termination without canvas mutation",
        "output": ROOT / "repro_results/dreamon_v3_c0_pure_newline_veto_protocol/protocol.json",
    },
    "v3_c_budgeted_nonconsuming_blankline": {
        "protocol_name": "dreamon_v3_c_budgeted_nonconsuming_blankline",
        "mode": "nonconsuming_blankline",
        "role": "targeted mechanism method: insert one locked physical blank line and retain the content canvas",
        "output": ROOT / "repro_results/dreamon_v3_c_nonconsuming_blankline_protocol/protocol.json",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_protocol(
    method: str,
    values: Mapping[str, Any],
    parent: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = copy.deepcopy(parent)
    protocol.update(
        {
            "protocol_name": values["protocol_name"],
            "protocol_version": 4,
            "frozen_at_utc": "2026-08-03",
            "research_decision": "iterate_v3c_targeted",
            "method": method,
            "method_role": values["role"],
            "label": "post-hoc pure-newline targeted development/mechanism diagnostic; not held-out",
            "parent_method": "v3_hard_budgeted",
            "parent_result_commit": "2e731ede15b2e6e536c86ec24e0263433d6573f9",
        }
    )
    protocol.pop("cycle5_diagnostic", None)
    protocol["targeted_pure30"] = {
        "posthoc_mechanism_diagnostic": True,
        "selection_source": meta["selection_source"],
        "selection_predicate": meta["selection_predicate"],
        "population_path": meta["manifests"]["pure30"]["path"],
        "population_sha256": meta["manifests"]["pure30"]["sha256"],
        "rows": meta["rows"],
        "base_problems": meta["base_problems"],
        "slot_distribution": meta["slot_distribution"],
        "a_baseline": meta["a_baseline"],
        "resolved_right_token_rows": meta["resolved_right_token_rows"],
        "nonwhitespace_resolved_right_token_rows": meta[
            "nonwhitespace_resolved_right_token_rows"
        ],
        "pure_newline_token_id_observed": meta[
            "pure_newline_token_id_observed"
        ],
        "smoke_selection": "first five rows in frozen Pure30 order",
    }
    protocol["generation"].update(
        {
            "nonempty_guard": False,
            "pure_newline_guard_mode": values["mode"],
            "pure_newline_guard_budget_per_slot": 1,
            "pure_newline_guard_global_budget": 3,
            "future_slot_logit_diagnostics": "recorded on trigger forward only; observational and selection-inert",
        }
    )
    if values["mode"] == "oneshot_veto":
        protocol["generation"]["completion"] = (
            "Direct A-style region/token extraction. One exact pure-newline boundary may be vetoed without canvas mutation; no post-hoc truncation or repair."
        )
    else:
        protocol["generation"]["completion"] = (
            "Direct extraction of the three content slots, three fixed separators, and any inserted locked blank newlines. Four to six physical lines are allowed; no post-hoc truncation or repair."
        )
    protocol["decoder_revision"] = {
        "nonempty_guard": False,
        "pure_newline_guard_mode": values["mode"],
        "trigger": {
            "normalized_proposal": "\\n",
            "left_text": "",
            "right_text": "",
            "selected_position": "active region left edge",
            "hypothetical_a_boundary": "immediately completes an empty slot",
            "budget_required": True,
        },
        "budget": {
            "per_slot": 1,
            "global": 3,
            "second_request": "fall back to unchanged V3-A boundary behavior",
            "expand_budget_interaction": "none",
        },
        "c0_veto": (
            {
                "canvas_mutation": False,
                "pending_ban": "exact (pre_canvas_hash, selected_position, proposal_token_id, line_boundary)",
                "retry": "one normal next forward; exact candidate masked only on identical canvas",
                "no_finite_candidate": "clear pending ban and execute original A boundary",
            }
            if values["mode"] == "oneshot_veto"
            else None
        ),
        "c_blankline": (
            {
                "action": "insert_locked_blank_newline",
                "insert_before": "active content slot",
                "selected_and_right_content": "preserved",
                "active_slot": "remains active",
                "length_accounting": "counts global/context; not hard-slot or expand budget",
                "cap_fallback": "no insertion; execute original A boundary without truncation",
            }
            if values["mode"] == "nonconsuming_blankline"
            else None
        ),
        "unchanged_from_a": [
            "attention",
            "sequential activation",
            "entropy selection",
            "newline boundary outside the one targeted intervention",
            "region-local EOS",
            "official cumulative expand budget",
            "caps",
            "cycle detection",
            "model/tokenizer/scorer/population",
        ],
        "cycle_detection": {
            "full_state_additions": [
                "per-slot pure-newline guard remaining",
                "global pure-newline guard remaining",
                "pending exact C0 ban",
                "inserted blank-line count",
            ],
            "rule": "Canvas equality with changed guard state is not an exact cycle; identical full transition at stable guard and expand budgets remains terminal.",
        },
    }
    protocol["resume"].update(
        {
            "mid_row_checkpoint": False,
            "generator_state_checkpoint_schema": 1,
            "checkpoint_fields": [
                "canvas arrays",
                "active region",
                "expand budget",
                "pure-newline guard budgets",
                "pending C0 ban",
                "inserted blank-line count",
            ],
            "interrupted_row": "Deterministically rerun from initial expand budget 64 and fresh pure-newline budgets 1/slot, 3/global; completed rows resume only under identical method/config hash.",
            "cross_protocol_resume": False,
        }
    )
    protocol["stage_gates"] = {
        "v3c_smoke": {
            "rows": 5,
            "population": "first five frozen Pure30 rows",
            "save_step_trace": True,
            "engineering_only": True,
        },
        "pure30": {
            "rows": 30,
            "save_step_trace": True,
            "engineering_gate_required": True,
            "full_quality_authorization": "overall Pass@1 >= 6/30",
        },
        "pilot": {
            "rows": 30,
            "population": "unchanged first 30 rows of the frozen 642 population",
            "save_step_trace": True,
            "engineering_only": True,
            "pass_threshold": None,
        },
        "full": {
            "rows": 642,
            "requires": "Smoke5, Pure30, and Pilot engineering gates plus Pure30 Pass >= 6/30",
            "quality_early_stop": False,
            "run_authorization_field": "executed_after_preregistered_gate",
        },
    }
    protocol["diagnostics"].update(
        {
            "future_slot_logits": {
                "source": "same trigger forward; no extra model call",
                "selection_inert": True,
                "action_classes": [
                    "normal",
                    "expand",
                    "EOS",
                    "pure_newline",
                    "other_newline",
                    "special/sentinel",
                ],
                "interpretation": "observational mechanism evidence, not causal proof",
            },
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 42,
        }
    )
    protocol["forbidden"] = [
        "compile gate",
        "BoundaryShift or carry-and-remask",
        "AST repair",
        "general nonempty guard",
        "random sampling",
        "multiple retries",
        "OpenTail",
        "Joint-OpenTail",
        "reference/test access during generation",
        "post-hoc first-line truncation or string repair",
    ]
    return protocol


def main() -> None:
    parent = read_json(PARENT)
    meta = read_json(META)
    for method, values in METHODS.items():
        output = Path(values["output"])
        write_json(output, build_protocol(method, values, parent, meta))
        print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
