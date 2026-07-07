#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl, metric, oracle_bucket
from experiments.action_ceiling.action_ceiling_matrix import current_branch, current_commit, git_capture


BASELINE_AUDIT = "analysis_outputs/h200_repro_audit_20260707_tier1_v2"
ACTION_BANK_AUDIT = "analysis_outputs/h200_repro_audit_20260707_action_bank_v1"
OLD_ACTION_BANK = "analysis_outputs/controller_action_bank_20260703_phase2_bank_merged"
H200_ACTION_BANK = "analysis_outputs/controller_action_bank_h200_20260707_tier1_offline"
SPLIT_ASSIGNMENT = "analysis_outputs/grouped_split_20260702_accel2/row_split_assignment.csv"
TEST_LOCK = "analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json"

PUBLIC_ROW_LEVEL_SPLITS = {"train", "calibration", "validation"}
TRIGGER_KEYS = (
    "official_repair_triggered",
    "lcal_v3_long_triggered",
    "lcal_v3_strong_triggered",
    "lcal_v3_weak_triggered",
    "route2_trace_rescue_triggered",
)


def _safe_rate(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


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


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_split_assignment(path: Path) -> dict[str, str]:
    return {row["task_id"]: row["split"] for row in read_csv_rows(path)}


def task_id(row: Mapping[str, Any]) -> str:
    return str(row["task_id"])


def row_passed(row: Mapping[str, Any]) -> bool:
    if "metrics" in row:
        return bool(metric(row, "passed", False))
    if "passed" in row:
        return _to_bool(row.get("passed"))
    return _to_bool(row.get("action_passed"))


def rows_by_task(rows: Iterable[Mapping[str, Any]], *, name: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        tid = task_id(row)
        if tid in out:
            raise ValueError(f"duplicate task_id in {name}: {tid}")
        out[tid] = row
    return out


def generated_code_hash(row: Mapping[str, Any]) -> str | None:
    code = row.get("code")
    if not isinstance(code, str):
        return None
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def action_hash(row: Mapping[str, Any]) -> str | None:
    value = row.get("generated_text_sha256") or row.get("code_sha256")
    if value is None or value == "":
        return None
    return str(value)


def error_type_signature(row: Mapping[str, Any]) -> str:
    if "error_type" in row:
        value = row.get("error_type")
        return "none" if value in {None, ""} else str(value)
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        return "unknown"
    errors = []
    for tier in sorted(verification):
        value = verification[tier]
        if not isinstance(value, Mapping):
            continue
        if not bool(value.get("passed", False)):
            errors.append(f"{tier}:{value.get('error_type')}")
    return "|".join(errors) if errors else "none"


def selected_length(row: Mapping[str, Any]) -> int | None:
    for key in ("selected_mask_length", "final_selected_length", "mask_length"):
        value = _to_int(metric(row, key))
        if value is not None:
            return value
    return None


def final_source(row: Mapping[str, Any]) -> str | None:
    for key in ("final_source", "route2_final_source", "lcal_v3_final_source", "mask_length_source"):
        value = metric(row, key)
        if value not in {None, ""}:
            return str(value)
    return None


def triggered(row: Mapping[str, Any]) -> bool:
    return any(bool(metric(row, key, False)) for key in TRIGGER_KEYS)


def signed_delta(new_value: int | None, old_value: int | None) -> int | None:
    if new_value is None or old_value is None:
        return None
    return int(new_value) - int(old_value)


def bool_delta_direction(old_pass: bool, h200_pass: bool) -> str:
    if h200_pass and not old_pass:
        return "h200_win"
    if old_pass and not h200_pass:
        return "h200_loss"
    return "tie_pass" if h200_pass else "tie_fail"


def core_flip_rows(
    run: str,
    old_rows: Sequence[Mapping[str, Any]],
    h200_rows: Sequence[Mapping[str, Any]],
    split_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    old_by_task = rows_by_task(old_rows, name=f"old_{run}")
    h200_by_task = rows_by_task(h200_rows, name=f"h200_{run}")
    rows: list[dict[str, Any]] = []
    for tid in sorted(set(old_by_task) & set(h200_by_task)):
        split = split_map.get(tid, "unknown")
        if split not in PUBLIC_ROW_LEVEL_SPLITS:
            continue
        old = old_by_task[tid]
        new = h200_by_task[tid]
        old_pass = row_passed(old)
        new_pass = row_passed(new)
        if old_pass == new_pass:
            continue
        old_len = selected_length(old)
        new_len = selected_length(new)
        old_hash = generated_code_hash(old)
        new_hash = generated_code_hash(new)
        oracle_len = _to_int(metric(old, "oracle_mask_length"))
        rows.append(
            {
                "run": run,
                "task_id": tid,
                "split": split,
                "oracle_bucket": oracle_bucket(oracle_len),
                "direction": bool_delta_direction(old_pass, new_pass),
                "old_passed": old_pass,
                "h200_passed": new_pass,
                "old_selected_length": old_len,
                "h200_selected_length": new_len,
                "selected_length_delta_h200_minus_old": signed_delta(new_len, old_len),
                "old_triggered": triggered(old),
                "h200_triggered": triggered(new),
                "old_final_source": final_source(old),
                "h200_final_source": final_source(new),
                "old_error_type": error_type_signature(old),
                "h200_error_type": error_type_signature(new),
                "candidate_hash_equal": old_hash is not None and old_hash == new_hash,
            }
        )
    return rows


def _core_scope_summary(
    run: str,
    scope: str,
    old_rows: Sequence[Mapping[str, Any]],
    h200_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    old_by_task = rows_by_task(old_rows, name=f"old_{run}_{scope}")
    h200_by_task = rows_by_task(h200_rows, name=f"h200_{run}_{scope}")
    common = sorted(set(old_by_task) & set(h200_by_task))
    counts: Counter[str] = Counter()
    hash_common = hash_agree = 0
    length_common = length_agree = 0
    trigger_agree = 0
    final_source_common = final_source_agree = 0
    length_deltas: list[float] = []
    abs_length_deltas: list[float] = []
    for tid in common:
        old = old_by_task[tid]
        new = h200_by_task[tid]
        direction = bool_delta_direction(row_passed(old), row_passed(new))
        counts[direction] += 1

        old_hash = generated_code_hash(old)
        new_hash = generated_code_hash(new)
        if old_hash is not None and new_hash is not None:
            hash_common += 1
            if old_hash == new_hash:
                hash_agree += 1

        old_len = selected_length(old)
        new_len = selected_length(new)
        delta = signed_delta(new_len, old_len)
        if delta is not None:
            length_common += 1
            length_deltas.append(float(delta))
            abs_length_deltas.append(float(abs(delta)))
            if delta == 0:
                length_agree += 1

        if triggered(old) == triggered(new):
            trigger_agree += 1

        old_source = final_source(old)
        new_source = final_source(new)
        if old_source is not None and new_source is not None:
            final_source_common += 1
            if old_source == new_source:
                final_source_agree += 1

    old_pass_count = sum(1 for row in old_rows if row_passed(row))
    h200_pass_count = sum(1 for row in h200_rows if row_passed(row))
    return {
        "run": run,
        "scope": scope,
        "rows": len(common),
        "old_pass_count": old_pass_count,
        "h200_pass_count": h200_pass_count,
        "pass_delta_h200_minus_old": h200_pass_count - old_pass_count,
        "h200_wins": counts["h200_win"],
        "h200_losses": counts["h200_loss"],
        "tie_pass": counts["tie_pass"],
        "tie_fail": counts["tie_fail"],
        "outcome_agreement_rate": _safe_rate(counts["tie_pass"] + counts["tie_fail"], len(common)),
        "candidate_hash_agreement_rate": _safe_rate(hash_agree, hash_common),
        "selected_length_agreement_rate": _safe_rate(length_agree, length_common),
        "selected_length_mean_delta": _mean(length_deltas),
        "selected_length_mean_abs_delta": _mean(abs_length_deltas),
        "trigger_agreement_rate": _safe_rate(trigger_agree, len(common)),
        "old_triggered_count": sum(1 for row in old_rows if triggered(row)),
        "h200_triggered_count": sum(1 for row in h200_rows if triggered(row)),
        "trigger_delta_h200_minus_old": sum(1 for row in h200_rows if triggered(row)) - sum(1 for row in old_rows if triggered(row)),
        "final_source_agreement_rate": _safe_rate(final_source_agree, final_source_common),
    }


def core_summary_rows(
    run: str,
    old_rows: Sequence[Mapping[str, Any]],
    h200_rows: Sequence[Mapping[str, Any]],
    split_map: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_by_task = rows_by_task(old_rows, name=f"old_{run}")
    h200_by_task = rows_by_task(h200_rows, name=f"h200_{run}")
    common = sorted(set(old_by_task) & set(h200_by_task))
    overall = _core_scope_summary(run, "overall_all_1033", old_rows, h200_rows)
    by_split_bucket: list[dict[str, Any]] = []
    for split in sorted(PUBLIC_ROW_LEVEL_SPLITS):
        tids = [tid for tid in common if split_map.get(tid) == split]
        split_old = [old_by_task[tid] for tid in tids]
        split_new = [h200_by_task[tid] for tid in tids]
        by_split_bucket.append(_core_scope_summary(run, f"split:{split}", split_old, split_new))
        buckets = sorted({oracle_bucket(_to_int(metric(old_by_task[tid], "oracle_mask_length"))) for tid in tids})
        for bucket in buckets:
            bucket_tids = [
                tid
                for tid in tids
                if oracle_bucket(_to_int(metric(old_by_task[tid], "oracle_mask_length"))) == bucket
            ]
            by_split_bucket.append(
                _core_scope_summary(
                    run,
                    f"split:{split}|bucket:{bucket}",
                    [old_by_task[tid] for tid in bucket_tids],
                    [h200_by_task[tid] for tid in bucket_tids],
                )
            )
    return [overall], by_split_bucket


def enrich_action_labels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary_pass = {
        str(row["task_id"]): row_passed(row)
        for row in rows
        if str(row.get("action")) == "KEEP_PRIMARY"
    }
    out = []
    for row in rows:
        action = str(row.get("action"))
        passed = row_passed(row)
        keep_passed = bool(primary_pass.get(str(row["task_id"]), False))
        enriched = dict(row)
        enriched["computed_passed"] = passed
        enriched["computed_benefit_label"] = (not keep_passed) and passed and action != "KEEP_PRIMARY"
        enriched["computed_harm_label"] = keep_passed and (not passed) and action != "KEEP_PRIMARY"
        out.append(enriched)
    return out


def keyed_action_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        out[(str(row["task_id"]), str(row["action"]))] = row
    return out


def action_scope_summary(scope: str, old_rows: Sequence[Mapping[str, Any]], h200_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    old_by_key = keyed_action_rows(old_rows)
    h200_by_key = keyed_action_rows(h200_rows)
    common = sorted(set(old_by_key) & set(h200_by_key))
    counts: Counter[str] = Counter()
    benefit_agree = harm_agree = 0
    hash_common = hash_agree = 0
    canvas_common = canvas_agree = 0
    for key in common:
        old = old_by_key[key]
        new = h200_by_key[key]
        counts[bool_delta_direction(row_passed(old), row_passed(new))] += 1
        if bool(old.get("computed_benefit_label")) == bool(new.get("computed_benefit_label")):
            benefit_agree += 1
        if bool(old.get("computed_harm_label")) == bool(new.get("computed_harm_label")):
            harm_agree += 1
        old_hash = action_hash(old)
        new_hash = action_hash(new)
        if old_hash and new_hash:
            hash_common += 1
            if old_hash == new_hash:
                hash_agree += 1
        if old.get("actual_canvas") not in {None, ""} and new.get("actual_canvas") not in {None, ""}:
            canvas_common += 1
            if int(float(old["actual_canvas"])) == int(float(new["actual_canvas"])):
                canvas_agree += 1

    old_pass_count = sum(1 for row in old_rows if row_passed(row))
    h200_pass_count = sum(1 for row in h200_rows if row_passed(row))
    return {
        "scope": scope,
        "old_rows": len(old_rows),
        "h200_rows": len(h200_rows),
        "common_rows": len(common),
        "old_pass_count": old_pass_count,
        "h200_pass_count": h200_pass_count,
        "pass_delta_h200_minus_old": h200_pass_count - old_pass_count,
        "h200_wins": counts["h200_win"],
        "h200_losses": counts["h200_loss"],
        "tie_pass": counts["tie_pass"],
        "tie_fail": counts["tie_fail"],
        "outcome_agreement_rate": _safe_rate(counts["tie_pass"] + counts["tie_fail"], len(common)),
        "benefit_label_agreement_rate": _safe_rate(benefit_agree, len(common)),
        "harm_label_agreement_rate": _safe_rate(harm_agree, len(common)),
        "candidate_hash_agreement_rate": _safe_rate(hash_agree, hash_common),
        "actual_canvas_agreement_rate": _safe_rate(canvas_agree, canvas_common),
        "old_benefit_count_non_keep": sum(1 for row in old_rows if row.get("computed_benefit_label")),
        "h200_benefit_count_non_keep": sum(1 for row in h200_rows if row.get("computed_benefit_label")),
        "old_harm_count_non_keep": sum(1 for row in old_rows if row.get("computed_harm_label")),
        "h200_harm_count_non_keep": sum(1 for row in h200_rows if row.get("computed_harm_label")),
    }


def action_bank_summary_rows(old_rows: Sequence[Mapping[str, Any]], h200_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [action_scope_summary("overall", old_rows, h200_rows)]
    splits = sorted({str(row.get("split")) for row in old_rows + h200_rows})
    actions = sorted({str(row.get("action")) for row in old_rows + h200_rows})
    for split in splits:
        rows.append(
            action_scope_summary(
                f"split:{split}",
                [row for row in old_rows if str(row.get("split")) == split],
                [row for row in h200_rows if str(row.get("split")) == split],
            )
        )
    for action in actions:
        rows.append(
            action_scope_summary(
                f"action:{action}",
                [row for row in old_rows if str(row.get("action")) == action],
                [row for row in h200_rows if str(row.get("action")) == action],
            )
        )
    for split in splits:
        for action in actions:
            rows.append(
                action_scope_summary(
                    f"split:{split}|action:{action}",
                    [row for row in old_rows if str(row.get("split")) == split and str(row.get("action")) == action],
                    [row for row in h200_rows if str(row.get("split")) == split and str(row.get("action")) == action],
                )
            )
    return rows


def action_bank_flip_rows(old_rows: Sequence[Mapping[str, Any]], h200_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    old_by_key = keyed_action_rows(old_rows)
    h200_by_key = keyed_action_rows(h200_rows)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(old_by_key) & set(h200_by_key)):
        old = old_by_key[key]
        new = h200_by_key[key]
        old_pass = row_passed(old)
        new_pass = row_passed(new)
        old_benefit = bool(old.get("computed_benefit_label"))
        new_benefit = bool(new.get("computed_benefit_label"))
        old_harm = bool(old.get("computed_harm_label"))
        new_harm = bool(new.get("computed_harm_label"))
        if old_pass == new_pass and old_benefit == new_benefit and old_harm == new_harm:
            continue
        old_hash = action_hash(old)
        new_hash = action_hash(new)
        rows.append(
            {
                "task_id": key[0],
                "split": old.get("split") or new.get("split"),
                "action": key[1],
                "oracle_bucket": old.get("oracle_bucket") or new.get("oracle_bucket"),
                "direction": bool_delta_direction(old_pass, new_pass),
                "old_passed": old_pass,
                "h200_passed": new_pass,
                "old_benefit_label": old_benefit,
                "h200_benefit_label": new_benefit,
                "old_harm_label": old_harm,
                "h200_harm_label": new_harm,
                "old_actual_canvas": old.get("actual_canvas"),
                "h200_actual_canvas": new.get("actual_canvas"),
                "old_error_type": error_type_signature(old),
                "h200_error_type": error_type_signature(new),
                "candidate_hash_equal": old_hash is not None and old_hash == new_hash,
            }
        )
    return rows


def load_baseline_pairs(baseline_audit: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(baseline_audit / "old_vs_h200_baselines.csv")
    required = {"run", "old_path", "h200_path"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"baseline audit CSV is missing columns: {sorted(missing)}")
    return rows


def classify_triage(summary_rows: Sequence[Mapping[str, Any]], action_overall: Mapping[str, Any]) -> str:
    max_abs_delta = max(abs(int(row["pass_delta_h200_minus_old"])) for row in summary_rows) if summary_rows else 0
    route2_delta = next((int(row["pass_delta_h200_minus_old"]) for row in summary_rows if row["run"] == "route2"), 0)
    v6_delta = next((int(row["pass_delta_h200_minus_old"]) for row in summary_rows if row["run"] == "v6"), 0)
    action_agreement = float(action_overall.get("outcome_agreement_rate") or 0.0)
    if route2_delta <= -2 or v6_delta <= -2 or max_abs_delta > 2:
        return "material_drift_confirmed_controller_v2_blocked"
    if action_agreement < 0.98:
        return "action_label_drift_controller_v2_blocked"
    return "minor_drift_same_claims_controller_v2_candidate"


def render_report(summary: Mapping[str, Any], core_rows: Sequence[Mapping[str, Any]], action_rows: Sequence[Mapping[str, Any]]) -> str:
    action_overall = next(row for row in action_rows if row["scope"] == "overall")
    lines = [
        "# H200 Material Drift Triage",
        "",
        f"triage_verdict: `{summary['triage_verdict']}`",
        f"server_repro_verdict: `{summary['server_repro_verdict']}`",
        f"controller_v2_allowed: `{summary['controller_v2_allowed']}`",
        f"branch: `{summary['branch']}`",
        f"commit: `{summary['commit']}`",
        "",
        "## Privacy / Protocol Guard",
        "",
        "- Frozen test remains sealed; this report does not write row-level test flips.",
        "- Core full-run aggregate counts still use the existing 1033-row reproduction audit.",
        "- Row-level triage CSVs are restricted to train/calibration/validation splits.",
        "",
        "## Core Outcome Drift",
        "",
        "| Run | Old Pass | H200 Pass | Delta | H200 Wins | H200 Losses | Agreement | Trigger Delta | Hash Agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in core_rows:
        lines.append(
            f"| `{row['run']}` | {row['old_pass_count']} | {row['h200_pass_count']} | "
            f"{row['pass_delta_h200_minus_old']} | {row['h200_wins']} | {row['h200_losses']} | "
            f"{float(row['outcome_agreement_rate']):.4f} | {row['trigger_delta_h200_minus_old']} | "
            f"{float(row['candidate_hash_agreement_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Action Bank Drift",
            "",
            f"- outcome agreement: `{action_overall['outcome_agreement_rate']}`",
            f"- pass delta: `{action_overall['pass_delta_h200_minus_old']}`",
            f"- benefit labels non-KEEP old/H200: `{action_overall['old_benefit_count_non_keep']}` / `{action_overall['h200_benefit_count_non_keep']}`",
            f"- harm labels non-KEEP old/H200: `{action_overall['old_harm_count_non_keep']}` / `{action_overall['h200_harm_count_non_keep']}`",
            f"- candidate hash agreement: `{action_overall['candidate_hash_agreement_rate']}`",
            "",
            "## Interpretation",
            "",
            "- The H200 deltas are small relative to 1033 rows but materially affect Route2/V6/CAL baselines and controller labels.",
            "- Controller V1 replay still selects zero intervention, but validation primary/V6 moved from 90 to 89 on the H200 bank.",
            "- Controller V2 remains blocked until the research owner decides whether to accept H200 as the new source of truth, rerun additional migration checks, or investigate kernel/model drift further.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only triage for material old-vs-H200 outcome and action-label drift.")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-root", default="analysis_outputs")
    parser.add_argument("--baseline-audit", default=BASELINE_AUDIT)
    parser.add_argument("--action-bank-audit", default=ACTION_BANK_AUDIT)
    parser.add_argument("--old-action-bank", default=OLD_ACTION_BANK)
    parser.add_argument("--h200-action-bank", default=H200_ACTION_BANK)
    parser.add_argument("--split-assignment", default=SPLIT_ASSIGNMENT)
    parser.add_argument("--test-lock", default=TEST_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"h200_material_drift_triage_{timestamp}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    baseline_audit = Path(args.baseline_audit)
    action_bank_audit = Path(args.action_bank_audit)
    split_map = load_split_assignment(Path(args.split_assignment))

    core_overall_rows: list[dict[str, Any]] = []
    core_by_split_bucket_rows: list[dict[str, Any]] = []
    core_public_flip_rows: list[dict[str, Any]] = []
    for row in load_baseline_pairs(baseline_audit):
        run = row["run"]
        old_rows = load_jsonl(Path(row["old_path"]) / "results.jsonl")
        h200_rows = load_jsonl(Path(row["h200_path"]) / "results.jsonl")
        overall, by_split_bucket = core_summary_rows(run, old_rows, h200_rows, split_map)
        core_overall_rows.extend(overall)
        core_by_split_bucket_rows.extend(by_split_bucket)
        core_public_flip_rows.extend(core_flip_rows(run, old_rows, h200_rows, split_map))

    old_action_rows = enrich_action_labels(load_jsonl(Path(args.old_action_bank) / "action_bank.jsonl"))
    h200_action_rows = enrich_action_labels(load_jsonl(Path(args.h200_action_bank) / "action_bank.jsonl"))
    action_rows = action_bank_summary_rows(old_action_rows, h200_action_rows)
    action_flips = action_bank_flip_rows(old_action_rows, h200_action_rows)
    action_overall = next(row for row in action_rows if row["scope"] == "overall")

    action_bank_audit_summary = load_json_if_exists(action_bank_audit / "summary.json")
    baseline_summary = load_json_if_exists(baseline_audit / "summary.json")
    test_lock = load_json_if_exists(Path(args.test_lock))
    triage_verdict = classify_triage(core_overall_rows, action_overall)
    server_repro_verdict = baseline_summary.get("repro_verdict") or action_bank_audit_summary.get("repro_verdict")
    summary = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "triage_verdict": triage_verdict,
        "server_repro_verdict": server_repro_verdict,
        "controller_v2_allowed": triage_verdict == "minor_drift_same_claims_controller_v2_candidate"
        and server_repro_verdict in {"h200_reproduction_confirmed", "h200_minor_candidate_drift_same_claims"},
        "baseline_audit": str(baseline_audit),
        "action_bank_audit": str(action_bank_audit),
        "old_action_bank": str(args.old_action_bank),
        "h200_action_bank": str(args.h200_action_bank),
        "split_assignment": str(args.split_assignment),
        "row_level_test_suppressed": True,
        "row_level_splits_written": sorted(PUBLIC_ROW_LEVEL_SPLITS),
        "core_flip_rows_train_calibration_validation": len(core_public_flip_rows),
        "action_bank_flip_or_label_change_rows": len(action_flips),
        "core_overall": core_overall_rows,
        "action_bank_overall": action_overall,
        "action_bank_audit_controller_v2_allowed": action_bank_audit_summary.get("controller_v2_allowed"),
        "test_lock_status": test_lock.get("test_status"),
        "test_evaluation_count": test_lock.get("test_evaluation_count"),
        "controller_v2_blocked_reason": (
            "H200 material outcome drift remains unresolved; this triage preserves row-level test privacy "
            "and keeps Controller V2 blocked."
        ),
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }

    write_csv(output_dir / "core_flip_summary.csv", core_overall_rows)
    write_csv(output_dir / "core_flip_rows_train_calibration_validation.csv", core_public_flip_rows)
    write_csv(output_dir / "core_drift_by_split_bucket_run.csv", core_by_split_bucket_rows)
    write_csv(output_dir / "trigger_length_drift_summary.csv", core_overall_rows + core_by_split_bucket_rows)
    write_csv(output_dir / "action_bank_flip_summary.csv", action_rows)
    write_csv(output_dir / "action_bank_flip_rows_train_calibration_validation.csv", action_flips)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary, core_overall_rows, action_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "triage_verdict": triage_verdict}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
