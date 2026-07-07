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


OLD_RUNS = {
    "control": "/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529",
    "midcons": "/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626",
    "route2": "/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516",
    "v6": "/home/shx/projects/dllm_infilling/git_workspace/outputs_clean/full_route2_v6_short_override_gpu2_20260620_124754",
    "cal": "/home/shx/projects/dllm_infilling/outputs_clean/full_official_cal_lcas_v3b_accel_gpus10_20260702_resumed_full_20260703_111615",
}

H200_DEFAULTS = {
    "control": "/home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_control_20260706_tier1_20260706_031441",
    "midcons": "/home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_midcons_20260706_tier1_20260706_042029",
    "route2": "/home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_route2_20260707_tier1_rerun_tmux_20260707_032537",
    "v6": "/home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_v6_20260707_tier1_tmux_20260707_044307",
}

OLD_EXPECTED_PASS_COUNTS = {
    "control": 787,
    "midcons": 795,
    "route2": 801,
    "v6": 802,
    "cal": 774,
}


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def row_passed(row: Mapping[str, Any]) -> bool:
    return bool(metric(row, "passed", False))


def task_id(row: Mapping[str, Any]) -> str:
    return str(row["task_id"])


def generated_code_hash(row: Mapping[str, Any]) -> str | None:
    code = row.get("code")
    if not isinstance(code, str):
        return None
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def verification_signature(row: Mapping[str, Any]) -> str:
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        return "unknown"
    parts = []
    for tier in sorted(verification):
        value = verification[tier]
        if not isinstance(value, Mapping):
            continue
        passed = value.get("passed")
        error_type = value.get("error_type")
        parts.append(f"{tier}:{passed}:{error_type}")
    return "|".join(parts) if parts else "unknown"


def error_type_signature(row: Mapping[str, Any]) -> str:
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


def rows_by_task(rows: Iterable[Mapping[str, Any]], *, name: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        tid = task_id(row)
        if tid in out:
            raise ValueError(f"duplicate task_id in {name}: {tid}")
        out[tid] = row
    return out


def summarize_run(name: str, run_dir: Path, gpu: str) -> dict[str, Any]:
    rows = load_jsonl(run_dir / "results.jsonl")
    pass_count = sum(1 for row in rows if row_passed(row))
    total_sec_values = [
        float(metric(row, "total_sec_including_probe"))
        for row in rows
        if metric(row, "total_sec_including_probe") is not None
    ]
    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run": name,
        "gpu": gpu,
        "path": str(run_dir),
        "rows": len(rows),
        "pass_count": pass_count,
        "pass_rate": _safe_rate(pass_count, len(rows)),
        "avg_total_sec_including_probe": _safe_rate(sum(total_sec_values), len(total_sec_values)),
        "summary_pass_rate": summary.get("pass_rate"),
        "trigger_count": summary.get("route2_trigger_count")
        or summary.get("lcal_v3_long_trigger_count")
        or summary.get("official_repair_trigger_count"),
    }


def compare_pair(name: str, old_rows: Sequence[Mapping[str, Any]], h200_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    old = rows_by_task(old_rows, name=f"old_{name}")
    new = rows_by_task(h200_rows, name=f"h200_{name}")
    common = sorted(set(old) & set(new))
    counts: Counter[str] = Counter()
    bucket_total: Counter[str] = Counter()
    bucket_agree: Counter[str] = Counter()
    code_hash_comparable = 0
    code_hash_agree = 0
    selected_len_comparable = 0
    selected_len_agree = 0
    verification_comparable = 0
    verification_agree = 0
    error_comparable = 0
    error_agree = 0
    bucket_comparable = 0
    bucket_agree_total = 0

    for tid in common:
        old_row = old[tid]
        new_row = new[tid]
        old_pass = row_passed(old_row)
        new_pass = row_passed(new_row)
        if new_pass and not old_pass:
            counts["h200_wins"] += 1
        elif (not new_pass) and old_pass:
            counts["h200_losses"] += 1
        elif new_pass and old_pass:
            counts["tie_pass"] += 1
        else:
            counts["tie_fail"] += 1

        old_bucket = oracle_bucket(metric(old_row, "oracle_mask_length"))
        new_bucket = oracle_bucket(metric(new_row, "oracle_mask_length"))
        if old_bucket == new_bucket:
            bucket_agree_total += 1
        bucket_comparable += 1
        bucket_total[old_bucket] += 1
        if old_pass == new_pass:
            bucket_agree[old_bucket] += 1

        old_hash = generated_code_hash(old_row)
        new_hash = generated_code_hash(new_row)
        if old_hash is not None and new_hash is not None:
            code_hash_comparable += 1
            if old_hash == new_hash:
                code_hash_agree += 1

        old_len = metric(old_row, "selected_mask_length", metric(old_row, "mask_length"))
        new_len = metric(new_row, "selected_mask_length", metric(new_row, "mask_length"))
        if old_len is not None and new_len is not None:
            selected_len_comparable += 1
            if int(old_len) == int(new_len):
                selected_len_agree += 1

        old_ver = verification_signature(old_row)
        new_ver = verification_signature(new_row)
        if old_ver != "unknown" and new_ver != "unknown":
            verification_comparable += 1
            if old_ver == new_ver:
                verification_agree += 1

        old_err = error_type_signature(old_row)
        new_err = error_type_signature(new_row)
        if old_err != "unknown" and new_err != "unknown":
            error_comparable += 1
            if old_err == new_err:
                error_agree += 1

    old_pass_count = sum(1 for row in old_rows if row_passed(row))
    h200_pass_count = sum(1 for row in h200_rows if row_passed(row))
    return {
        "run": name,
        "common": len(common),
        "old_rows": len(old_rows),
        "h200_rows": len(h200_rows),
        "old_pass_count": old_pass_count,
        "h200_pass_count": h200_pass_count,
        "pass_delta_h200_minus_old": h200_pass_count - old_pass_count,
        "h200_wins": counts.get("h200_wins", 0),
        "h200_losses": counts.get("h200_losses", 0),
        "tie_pass": counts.get("tie_pass", 0),
        "tie_fail": counts.get("tie_fail", 0),
        "outcome_agreement_rate": _safe_rate(counts.get("tie_pass", 0) + counts.get("tie_fail", 0), len(common)),
        "candidate_hash_agreement_rate": _safe_rate(code_hash_agree, code_hash_comparable),
        "candidate_hash_comparable": code_hash_comparable,
        "selected_length_agreement_rate": _safe_rate(selected_len_agree, selected_len_comparable),
        "selected_length_comparable": selected_len_comparable,
        "bucket_agreement_rate": _safe_rate(bucket_agree_total, bucket_comparable),
        "verification_signature_agreement_rate": _safe_rate(verification_agree, verification_comparable),
        "error_type_agreement_rate": _safe_rate(error_agree, error_comparable),
        "bucket_outcome_agreement": {
            bucket: {
                "rows": bucket_total[bucket],
                "outcome_agreement_rate": _safe_rate(bucket_agree[bucket], bucket_total[bucket]),
            }
            for bucket in sorted(bucket_total)
        },
    }


def classify_verdict(comparisons: Sequence[Mapping[str, Any]]) -> str:
    if not comparisons:
        return "h200_environment_invalid"
    max_abs_delta = max(abs(int(row["pass_delta_h200_minus_old"])) for row in comparisons)
    route2_delta = next((int(row["pass_delta_h200_minus_old"]) for row in comparisons if row["run"] == "route2"), 0)
    v6_delta = next((int(row["pass_delta_h200_minus_old"]) for row in comparisons if row["run"] == "v6"), 0)
    min_outcome_agreement = min(float(row.get("outcome_agreement_rate") or 0.0) for row in comparisons)
    if max_abs_delta == 0 and min_outcome_agreement >= 0.999:
        return "h200_reproduction_confirmed"
    if max_abs_delta <= 1 and min_outcome_agreement >= 0.99:
        return "h200_minor_candidate_drift_same_claims"
    if route2_delta <= -2 or v6_delta <= -2 or max_abs_delta > 2:
        return "h200_material_outcome_drift"
    return "h200_minor_candidate_drift_same_claims"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: Mapping[str, Any], baseline_rows: Sequence[Mapping[str, Any]], comparison_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# H200 Reproduction Audit",
        "",
        f"verdict: `{summary['repro_verdict']}`",
        f"branch: `{summary['branch']}`",
        f"commit: `{summary['commit']}`",
        "",
        "## Core Baselines",
        "",
        "| Run | Old GPU | Old Pass | H200 Pass | Delta | H200 Wins | H200 Losses | Outcome Agreement | Hash Agreement | Avg Sec Old | Avg Sec H200 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_run_gpu = {(row["run"], row["gpu"]): row for row in baseline_rows}
    by_run_cmp = {row["run"]: row for row in comparison_rows}
    for run in ("control", "midcons", "route2", "v6", "cal"):
        cmp_row = by_run_cmp.get(run)
        old_row = by_run_gpu.get((run, "old_a6000"))
        h200_row = by_run_gpu.get((run, "h200"))
        if cmp_row is None or old_row is None or h200_row is None:
            continue
        outcome = cmp_row.get("outcome_agreement_rate")
        hash_agree = cmp_row.get("candidate_hash_agreement_rate")
        old_sec = old_row.get("avg_total_sec_including_probe")
        h200_sec = h200_row.get("avg_total_sec_including_probe")
        lines.append(
            f"| `{run}` | A6000 | {cmp_row['old_pass_count']} | {cmp_row['h200_pass_count']} | "
            f"{cmp_row['pass_delta_h200_minus_old']} | {cmp_row['h200_wins']} | {cmp_row['h200_losses']} | "
            f"{float(outcome):.4f} | {float(hash_agree):.4f} | {float(old_sec):.3f} | {float(h200_sec):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Reproduction verdict is `{summary['repro_verdict']}`.",
            "- Candidate hash mismatch is recorded separately from pass/fail drift because deterministic GPU kernels can still produce text changes without changing outcome.",
            "- Controller V2 must not start automatically when verdict is `h200_material_outcome_drift`.",
        ]
    )
    return "\n".join(lines) + "\n"


def baseline_comparison_rows(
    run_summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_run_gpu = {(row["run"], row["gpu"]): row for row in run_summary_rows}
    out: list[dict[str, Any]] = []
    for comparison in comparison_rows:
        run = str(comparison["run"])
        old_row = by_run_gpu[(run, "old_a6000")]
        h200_row = by_run_gpu[(run, "h200")]
        out.append(
            {
                "run": run,
                "old_gpu": "NVIDIA RTX A6000",
                "h200_gpu": "NVIDIA H200 NVL",
                "old_rows": comparison["old_rows"],
                "h200_rows": comparison["h200_rows"],
                "old_pass_count": comparison["old_pass_count"],
                "h200_pass_count": comparison["h200_pass_count"],
                "pass_delta_h200_minus_old": comparison["pass_delta_h200_minus_old"],
                "h200_wins": comparison["h200_wins"],
                "h200_losses": comparison["h200_losses"],
                "tie_pass": comparison["tie_pass"],
                "tie_fail": comparison["tie_fail"],
                "outcome_agreement_rate": comparison["outcome_agreement_rate"],
                "candidate_hash_agreement_rate": comparison["candidate_hash_agreement_rate"],
                "selected_length_agreement_rate": comparison["selected_length_agreement_rate"],
                "bucket_agreement_rate": comparison["bucket_agreement_rate"],
                "error_type_agreement_rate": comparison["error_type_agreement_rate"],
                "old_avg_total_sec_including_probe": old_row["avg_total_sec_including_probe"],
                "h200_avg_total_sec_including_probe": h200_row["avg_total_sec_including_probe"],
                "old_trigger_count": old_row.get("trigger_count"),
                "h200_trigger_count": h200_row.get("trigger_count"),
                "old_path": old_row["path"],
                "h200_path": h200_row["path"],
            }
        )
    return out


def parse_overrides(values: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must be name=path, got {value!r}")
        name, path = value.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact old-vs-H200 reproduction audit artifacts.")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--output-root", default="analysis_outputs")
    parser.add_argument("--h200-run", action="append", default=[], help="Override H200 run dir as name=/path.")
    parser.add_argument("--old-run", action="append", default=[], help="Override old run dir as name=/path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"h200_repro_audit_{timestamp}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    old_runs = {**OLD_RUNS, **parse_overrides(args.old_run)}
    h200_runs = {**H200_DEFAULTS, **parse_overrides(args.h200_run)}
    missing_h200 = sorted(set(old_runs) - set(h200_runs))
    if missing_h200:
        raise ValueError(f"missing H200 run dirs: {missing_h200}")

    baseline_summary_rows = []
    comparison_rows = []
    for name in ("control", "midcons", "route2", "v6", "cal"):
        old_dir = Path(old_runs[name])
        h200_dir = Path(h200_runs[name])
        old_rows = load_jsonl(old_dir / "results.jsonl")
        h200_rows = load_jsonl(h200_dir / "results.jsonl")
        baseline_summary_rows.append(summarize_run(name, old_dir, "old_a6000"))
        baseline_summary_rows.append(summarize_run(name, h200_dir, "h200"))
        comparison = compare_pair(name, old_rows, h200_rows)
        comparison_rows.append(comparison)

    verdict = classify_verdict(comparison_rows)
    paired_baseline_rows = baseline_comparison_rows(baseline_summary_rows, comparison_rows)
    summary = {
        "branch": current_branch(),
        "commit": current_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "repro_verdict": verdict,
        "old_gpu": "NVIDIA RTX A6000",
        "new_gpu": "NVIDIA H200 NVL",
        "old_expected_pass_counts": OLD_EXPECTED_PASS_COUNTS,
        "baseline_summary": baseline_summary_rows,
        "comparisons": comparison_rows,
        "git_working_tree_status": git_capture("status", "--short", "--branch"),
    }
    environment_diff = {
        "old_gpu": "NVIDIA RTX A6000",
        "new_gpu": "NVIDIA H200 NVL",
        "old_torch_cuda_runtime": "12.1",
        "new_torch_cuda_runtime": "12.1",
        "new_driver_cuda": "13.0",
        "new_driver": "580.159.03",
        "notes": [
            "Old environment values are copied from prior run manifests.",
            "H200 host GPU is visible only through approved host-side commands in this Codex environment.",
        ],
    }

    write_csv(output_dir / "run_summary.csv", baseline_summary_rows)
    write_csv(output_dir / "old_vs_h200_baselines.csv", paired_baseline_rows)
    write_csv(
        output_dir / "old_vs_h200_action_bank.csv",
        [
            {
                "status": "not_run",
                "reason": "Tier 1 baseline material drift was detected before H200 action-bank rebuild.",
            }
        ],
    )
    write_csv(
        output_dir / "old_vs_h200_controller_v1.csv",
        [
            {
                "status": "not_run",
                "reason": "Controller V1 replay requires the H200 action bank.",
            }
        ],
    )
    write_csv(
        output_dir / "true_long_replay.csv",
        [
            {
                "status": "not_run",
                "reason": "True-long replay deferred until H200 action-label rebuild/drift triage.",
            }
        ],
    )
    (output_dir / "environment_diff.json").write_text(
        json.dumps(environment_diff, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_report(summary, baseline_summary_rows, comparison_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "repro_verdict": verdict}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
