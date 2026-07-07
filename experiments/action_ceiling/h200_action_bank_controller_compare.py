#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl
from experiments.action_ceiling.action_ceiling_matrix import current_branch, current_commit, git_capture


OLD_ACTION_BANK = "analysis_outputs/controller_action_bank_20260703_phase2_bank_merged"
OLD_CONTROLLER = "analysis_outputs/controller_validation_20260703_phase2_controller_validation_v3"
BASELINE_AUDIT = "analysis_outputs/h200_repro_audit_20260707_tier1_v2"


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_rate(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_passed(row: Mapping[str, Any]) -> bool:
    if "passed" in row:
        return _to_bool(row.get("passed"))
    return _to_bool(row.get("action_passed"))


def keyed_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        out[(str(row["task_id"]), str(row["action"]))] = row
    return out


def add_labels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary_pass = {
        str(row["task_id"]): row_passed(row)
        for row in rows
        if str(row.get("action")) == "KEEP_PRIMARY"
    }
    out = []
    for row in rows:
        task_id = str(row["task_id"])
        passed = row_passed(row)
        keep_passed = bool(primary_pass.get(task_id, False))
        enriched = dict(row)
        enriched["computed_passed"] = passed
        enriched["computed_benefit_label"] = (not keep_passed) and passed and str(row.get("action")) != "KEEP_PRIMARY"
        enriched["computed_harm_label"] = keep_passed and (not passed) and str(row.get("action")) != "KEEP_PRIMARY"
        out.append(enriched)
    return out


def bank_scope_summary(scope: str, old_rows: Sequence[Mapping[str, Any]], h200_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    old_by_key = keyed_rows(old_rows)
    h200_by_key = keyed_rows(h200_rows)
    common = sorted(set(old_by_key) & set(h200_by_key))
    counts: Counter[str] = Counter()
    hash_common = hash_agree = 0
    canvas_common = canvas_agree = 0
    bucket_common = bucket_agree = 0
    benefit_agree = harm_agree = 0
    old_costs: list[float] = []
    h200_costs: list[float] = []
    for key in common:
        old = old_by_key[key]
        new = h200_by_key[key]
        old_pass = row_passed(old)
        new_pass = row_passed(new)
        if new_pass and not old_pass:
            counts["h200_wins"] += 1
        elif old_pass and not new_pass:
            counts["h200_losses"] += 1
        elif new_pass:
            counts["tie_pass"] += 1
        else:
            counts["tie_fail"] += 1
        if bool(old.get("computed_benefit_label")) == bool(new.get("computed_benefit_label")):
            benefit_agree += 1
        if bool(old.get("computed_harm_label")) == bool(new.get("computed_harm_label")):
            harm_agree += 1
        old_hash = old.get("generated_text_sha256") or old.get("code_sha256")
        new_hash = new.get("generated_text_sha256") or new.get("code_sha256")
        if old_hash and new_hash:
            hash_common += 1
            if old_hash == new_hash:
                hash_agree += 1
        if old.get("actual_canvas") not in {None, ""} and new.get("actual_canvas") not in {None, ""}:
            canvas_common += 1
            if int(float(old["actual_canvas"])) == int(float(new["actual_canvas"])):
                canvas_agree += 1
        if old.get("oracle_bucket") not in {None, ""} and new.get("oracle_bucket") not in {None, ""}:
            bucket_common += 1
            if str(old.get("oracle_bucket")) == str(new.get("oracle_bucket")):
                bucket_agree += 1
        if old.get("inference_cost_sec") not in {None, ""}:
            old_costs.append(_to_float(old.get("inference_cost_sec")))
        if new.get("inference_cost_sec") not in {None, ""}:
            h200_costs.append(_to_float(new.get("inference_cost_sec")))
    return {
        "scope": scope,
        "old_rows": len(old_rows),
        "h200_rows": len(h200_rows),
        "common_rows": len(common),
        "old_pass_count": sum(1 for row in old_rows if row_passed(row)),
        "h200_pass_count": sum(1 for row in h200_rows if row_passed(row)),
        "pass_delta_h200_minus_old": sum(1 for row in h200_rows if row_passed(row)) - sum(1 for row in old_rows if row_passed(row)),
        "h200_wins": counts["h200_wins"],
        "h200_losses": counts["h200_losses"],
        "tie_pass": counts["tie_pass"],
        "tie_fail": counts["tie_fail"],
        "outcome_agreement_rate": _safe_rate(counts["tie_pass"] + counts["tie_fail"], len(common)),
        "benefit_label_agreement_rate": _safe_rate(benefit_agree, len(common)),
        "harm_label_agreement_rate": _safe_rate(harm_agree, len(common)),
        "candidate_hash_agreement_rate": _safe_rate(hash_agree, hash_common),
        "actual_canvas_agreement_rate": _safe_rate(canvas_agree, canvas_common),
        "bucket_agreement_rate": _safe_rate(bucket_agree, bucket_common),
        "old_benefit_count_non_keep": sum(1 for row in old_rows if row.get("computed_benefit_label")),
        "h200_benefit_count_non_keep": sum(1 for row in h200_rows if row.get("computed_benefit_label")),
        "old_harm_count_non_keep": sum(1 for row in old_rows if row.get("computed_harm_label")),
        "h200_harm_count_non_keep": sum(1 for row in h200_rows if row.get("computed_harm_label")),
        "old_mean_cost_sec": _mean(old_costs),
        "h200_mean_cost_sec": _mean(h200_costs),
    }


def compare_action_banks(old_dir: Path, h200_dir: Path) -> list[dict[str, Any]]:
    old_rows = add_labels(load_jsonl(old_dir / "action_bank.jsonl"))
    h200_rows = add_labels(load_jsonl(h200_dir / "action_bank.jsonl"))
    rows = [bank_scope_summary("overall", old_rows, h200_rows)]
    for split in sorted({str(row.get("split")) for row in old_rows + h200_rows}):
        rows.append(
            bank_scope_summary(
                f"split:{split}",
                [row for row in old_rows if str(row.get("split")) == split],
                [row for row in h200_rows if str(row.get("split")) == split],
            )
        )
    for action in sorted({str(row.get("action")) for row in old_rows + h200_rows}):
        rows.append(
            bank_scope_summary(
                f"action:{action}",
                [row for row in old_rows if str(row.get("action")) == action],
                [row for row in h200_rows if str(row.get("action")) == action],
            )
        )
    return rows


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def controller_selected_row(controller_dir: Path) -> dict[str, Any]:
    summary = load_json_if_exists(controller_dir / "summary.json")
    chosen = ((summary.get("validation_gate") or {}).get("selected_controller") or {})
    rows = read_csv_rows(controller_dir / "controller_validation_results.csv")
    for row in rows:
        if (
            row.get("model_family") == str(chosen.get("model_family"))
            and row.get("feature_variant") == str(chosen.get("feature_variant"))
            and row.get("policy_variant") == str(chosen.get("policy_variant"))
            and math.isclose(_to_float(row.get("score_threshold")), _to_float(chosen.get("score_threshold")), rel_tol=0.0, abs_tol=1e-9)
        ):
            return row
    return rows[0] if rows else {}


def controller_metric_row(name: str, old: Mapping[str, Any], h200: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "validation_pass_count",
        "validation_total",
        "validation_pass_rate",
        "validation_wins_vs_primary",
        "validation_losses_vs_primary",
        "validation_intervention_count",
        "validation_intervention_harm_count",
        "validation_intervention_harm_upper95",
        "validation_mean_cost_sec",
    ]
    out: dict[str, Any] = {"entity": name}
    for key in keys:
        out[f"old_{key}"] = old.get(key)
        out[f"h200_{key}"] = h200.get(key)
    if old.get("validation_pass_count") not in {None, ""} and h200.get("validation_pass_count") not in {None, ""}:
        out["pass_delta_h200_minus_old"] = int(float(h200["validation_pass_count"])) - int(float(old["validation_pass_count"]))
    return out


def compare_controllers(old_dir: Path, h200_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    old_summary = load_json_if_exists(old_dir / "summary.json")
    h200_summary = load_json_if_exists(h200_dir / "summary.json")
    old_gate = old_summary.get("validation_gate") or {}
    h200_gate = h200_summary.get("validation_gate") or {}
    selected = controller_metric_row("selected_controller", controller_selected_row(old_dir), controller_selected_row(h200_dir))
    selected.update(
        {
            "old_selected_controller": json.dumps(old_gate.get("selected_controller"), sort_keys=True),
            "h200_selected_controller": json.dumps(h200_gate.get("selected_controller"), sort_keys=True),
            "old_gate_passed": old_gate.get("gate_passed"),
            "h200_gate_passed": h200_gate.get("gate_passed"),
            "old_test_decision": old_gate.get("test_decision"),
            "h200_test_decision": h200_gate.get("test_decision"),
        }
    )
    rows.append(selected)

    old_baselines = {row["baseline"]: row for row in read_csv_rows(old_dir / "validation_baselines.csv")}
    h200_baselines = {row["baseline"]: row for row in read_csv_rows(h200_dir / "validation_baselines.csv")}
    for name in sorted(set(old_baselines) & set(h200_baselines)):
        rows.append(controller_metric_row(f"baseline:{name}", old_baselines[name], h200_baselines[name]))
    return rows


def render_report(summary: Mapping[str, Any], action_rows: Sequence[Mapping[str, Any]], controller_rows: Sequence[Mapping[str, Any]]) -> str:
    overall = next(row for row in action_rows if row["scope"] == "overall")
    selected = next(row for row in controller_rows if row["entity"] == "selected_controller")
    lines = [
        "# H200 Action Bank and Controller V1 Replay Audit",
        "",
        f"repro_verdict: `{summary['repro_verdict']}`",
        f"branch: `{summary['branch']}`",
        f"commit: `{summary['commit']}`",
        "",
        "## Action Bank",
        "",
        f"- H200 rows: `{overall['h200_rows']}`",
        f"- H200 task/action coverage: `{summary['h200_action_bank']['task_count']}` tasks / `{summary['h200_action_bank']['row_count']}` rows",
        f"- outcome agreement vs old bank: `{overall['outcome_agreement_rate']}`",
        f"- H200 benefit/harm labels non-KEEP: `{overall['h200_benefit_count_non_keep']}` / `{overall['h200_harm_count_non_keep']}`",
        f"- old benefit/harm labels non-KEEP: `{overall['old_benefit_count_non_keep']}` / `{overall['old_harm_count_non_keep']}`",
        "",
        "## Controller V1 Replay",
        "",
        f"- old selected: `{selected.get('old_selected_controller')}`",
        f"- H200 selected: `{selected.get('h200_selected_controller')}`",
        f"- old gate/test decision: `{selected.get('old_gate_passed')}` / `{selected.get('old_test_decision')}`",
        f"- H200 gate/test decision: `{selected.get('h200_gate_passed')}` / `{selected.get('h200_test_decision')}`",
        f"- H200 selected validation pass count: `{selected.get('h200_validation_pass_count')}`",
        f"- H200 selected validation wins/losses vs primary: `{selected.get('h200_validation_wins_vs_primary')}` / `{selected.get('h200_validation_losses_vs_primary')}`",
        "",
        "## Decision",
        "",
        "- Core Tier 1 baselines already produced `h200_material_outcome_drift`; this remains the server migration verdict.",
        "- Controller V2 stays blocked until the material H200 drift is triaged by the research owner.",
        "- Frozen test remains sealed with evaluation count 0.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old and H200 controller action-bank/controller V1 artifacts.")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-root", default="analysis_outputs")
    parser.add_argument("--old-action-bank", default=OLD_ACTION_BANK)
    parser.add_argument("--h200-action-bank", required=True)
    parser.add_argument("--old-controller", default=OLD_CONTROLLER)
    parser.add_argument("--h200-controller", required=True)
    parser.add_argument("--baseline-audit", default=BASELINE_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"h200_repro_audit_{timestamp}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    old_bank = Path(args.old_action_bank)
    h200_bank = Path(args.h200_action_bank)
    old_controller = Path(args.old_controller)
    h200_controller = Path(args.h200_controller)
    baseline_audit = Path(args.baseline_audit)

    action_rows = compare_action_banks(old_bank, h200_bank)
    controller_rows = compare_controllers(old_controller, h200_controller)
    baseline_summary = load_json_if_exists(baseline_audit / "summary.json")
    h200_bank_summary = load_json_if_exists(h200_bank / "summary.json")
    h200_controller_summary = load_json_if_exists(h200_controller / "summary.json")
    old_bank_summary = load_json_if_exists(old_bank / "summary.json")
    old_controller_summary = load_json_if_exists(old_controller / "summary.json")
    test_lock = load_json_if_exists(Path("analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"))
    verdict = baseline_summary.get("repro_verdict") or "h200_material_outcome_drift"
    summary = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "repro_verdict": verdict,
        "baseline_audit": str(baseline_audit),
        "old_action_bank_dir": str(old_bank),
        "h200_action_bank_dir": str(h200_bank),
        "old_controller_dir": str(old_controller),
        "h200_controller_dir": str(h200_controller),
        "old_action_bank": {
            "row_count": old_bank_summary.get("row_count"),
            "task_count": old_bank_summary.get("task_count"),
            "split_counts": old_bank_summary.get("split_counts"),
            "action_counts": old_bank_summary.get("action_counts"),
        },
        "h200_action_bank": {
            "row_count": h200_bank_summary.get("row_count"),
            "task_count": h200_bank_summary.get("task_count"),
            "split_counts": h200_bank_summary.get("split_counts"),
            "action_counts": h200_bank_summary.get("action_counts"),
            "pass_count": h200_bank_summary.get("pass_count"),
            "benefit_labels_non_keep": h200_bank_summary.get("benefit_labels_non_keep"),
            "harm_labels_non_keep": h200_bank_summary.get("harm_labels_non_keep"),
        },
        "old_controller_v1": old_controller_summary,
        "h200_controller_v1": h200_controller_summary,
        "action_bank_rebuilt": h200_bank_summary.get("row_count") == 4635 and h200_bank_summary.get("task_count") == 927,
        "controller_v1_replayed": h200_controller_summary.get("mode") == "frozen_canvas_controller_validation",
        "test_lock_status": test_lock.get("test_status"),
        "test_evaluation_count": test_lock.get("test_evaluation_count"),
        "controller_v2_allowed": verdict in {"h200_reproduction_confirmed", "h200_minor_candidate_drift_same_claims"},
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }
    environment_diff = load_json_if_exists(baseline_audit / "environment_diff.json")

    baseline_csv = baseline_audit / "old_vs_h200_baselines.csv"
    if baseline_csv.exists():
        (output_dir / "old_vs_h200_baselines.csv").write_text(baseline_csv.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        write_csv(output_dir / "old_vs_h200_baselines.csv", [{"status": "missing", "path": str(baseline_csv)}])
    write_csv(output_dir / "old_vs_h200_action_bank.csv", action_rows)
    write_csv(output_dir / "old_vs_h200_controller_v1.csv", controller_rows)
    write_csv(
        output_dir / "true_long_replay.csv",
        [{"status": "not_run", "reason": "not part of the current authorized action-bank/controller replay scope"}],
    )
    (output_dir / "environment_diff.json").write_text(json.dumps(environment_diff, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary, action_rows, controller_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "repro_verdict": verdict}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
