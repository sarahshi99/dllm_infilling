#!/usr/bin/env python3
"""Shared grouped analysis for M2 Constraint-Homotopy V0."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.grouped_effects import write_grouped_effect_report


METHODS = ("m2_vanilla_fixed64", "m2_gradual_constraints", "m2_abrupt_constraints")
PAIR_LABELS = (
    (METHODS[1], METHODS[0], "gradual_vs_vanilla"),
    (METHODS[2], METHODS[0], "abrupt_vs_vanilla"),
    (METHODS[1], METHODS[2], "gradual_vs_abrupt"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * float(quantile)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "nonzero_count": sum(value != 0.0 for value in values),
        "min": min(values) if values else math.nan,
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "mean": sum(values) / len(values) if values else math.nan,
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else math.nan,
    }


def activation_audit(rows_by_method: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Summarize schedule activation using hashes and scalar dynamics only."""
    indexed = {
        method: {str(row["row_key"]): row for row in rows}
        for method, rows in rows_by_method.items()
    }
    base_keys = set(indexed[METHODS[0]])
    if any(set(rows) != base_keys for rows in indexed.values()):
        raise RuntimeError("M2 activation audit requires matched row keys")
    pairs: list[dict[str, Any]] = []
    for method_a, method_b, label in PAIR_LABELS:
        full_code_difference = sum(
            indexed[method_a][key].get("candidate_full_code_sha256")
            != indexed[method_b][key].get("candidate_full_code_sha256")
            for key in base_keys
        )
        middle_difference = sum(
            indexed[method_a][key].get("candidate_middle_sha256")
            != indexed[method_b][key].get("candidate_middle_sha256")
            for key in base_keys
        )
        pairs.append(
            {
                "comparison": label,
                "method_a": method_a,
                "method_b": method_b,
                "candidate_full_code_sha256_different_count": full_code_difference,
                "candidate_middle_sha256_different_count": middle_difference,
                "row_count": len(base_keys),
            }
        )
    dynamics = []
    for method, rows in rows_by_method.items():
        metrics = [dict(row.get("metrics") or {}) for row in rows]
        dynamics.append(
            {
                "method": method,
                "effective_update_steps": distribution(
                    [float(item.get("effective_update_steps") or 0.0) for item in metrics]
                ),
                "total_token_changes": distribution(
                    [float(item.get("total_token_changes") or 0.0) for item in metrics]
                ),
            }
        )
    changed = any(item["candidate_full_code_sha256_different_count"] > 0 for item in pairs)
    return {
        "raw_generated_code_included": False,
        "candidate_hash_fields": ["candidate_full_code_sha256", "candidate_middle_sha256"],
        "pairwise_hash_differences": pairs,
        "dynamics": dynamics,
        "schedule_activation": "changed_candidates_observed" if changed else "no_candidate_hash_change_observed",
    }


def scientific_judgment(summary: dict[str, Any], activation: dict[str, Any]) -> dict[str, str]:
    effects = {
        (str(item["method_a"]), str(item["method_b"])): item
        for item in summary["paired_effects"]
    }
    gradual_vanilla = effects[(METHODS[1], METHODS[0])]
    abrupt_vanilla = effects[(METHODS[2], METHODS[0])]
    gradual_abrupt = effects[(METHODS[1], METHODS[2])]
    if gradual_vanilla["task_macro_ci_low"] > 0 and gradual_abrupt["task_macro_ci_low"] > 0:
        return {"label": "gradual_homotopy_supported", "reason": "gradual has positive task-macro CIs against both vanilla and abrupt"}
    if gradual_vanilla["task_macro_ci_low"] > 0 and abrupt_vanilla["task_macro_ci_low"] > 0 and gradual_abrupt["task_macro_ci_low"] <= 0 <= gradual_abrupt["task_macro_ci_high"]:
        return {"label": "constraints_supported_homotopy_not_identified", "reason": "both constraint arms improve over vanilla but gradual-versus-abrupt is unresolved"}
    if gradual_abrupt["task_macro_ci_high"] < 0:
        return {"label": "gradual_schedule_falsified_by_abrupt", "reason": "abrupt has a positive task-macro advantage over gradual"}
    if activation["schedule_activation"] == "changed_candidates_observed":
        return {"label": "constraints_activated_no_reliable_grouped_advantage", "reason": "candidate hashes changed but no comparison meets the predeclared grouped support rule"}
    return {"label": "implementation_or_signal_noop_suspected", "reason": "candidate hashes did not change across schedules"}


def write_activation_artifacts(output_dir: Path, rows_by_method: dict[str, list[dict[str, Any]],], summary: dict[str, Any]) -> dict[str, Any]:
    activation = activation_audit(rows_by_method)
    judgment = scientific_judgment(summary, activation)
    payload = {**activation, "scientific_judgment": judgment}
    (output_dir / "activation_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# M2 activation audit",
        "",
        "This artifact contains candidate hashes, counts, and aggregate dynamics only; it contains no raw generated code.",
        "",
        f"Schedule activation: `{activation['schedule_activation']}`.",
        "",
        "| Comparison | Full-code SHA differences | Middle SHA differences | Rows |",
        "|---|---:|---:|---:|",
    ]
    for item in activation["pairwise_hash_differences"]:
        lines.append(
            f"| `{item['comparison']}` | `{item['candidate_full_code_sha256_different_count']}` | "
            f"`{item['candidate_middle_sha256_different_count']}` | `{item['row_count']}` |"
        )
    lines.extend(["", f"Scientific judgment: `{judgment['label']}` — {judgment['reason']}"])
    (output_dir / "activation_audit.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_analysis(
    vanilla_raw: Path,
    gradual_raw: Path,
    abrupt_raw: Path,
    output_dir: Path,
    *,
    run_manifest: Path | None = None,
) -> dict[str, Any]:
    rows_by_method = {
        METHODS[0]: read_jsonl(vanilla_raw),
        METHODS[1]: read_jsonl(gradual_raw),
        METHODS[2]: read_jsonl(abrupt_raw),
    }
    populations = [{str(row["row_key"]) for row in rows} for rows in rows_by_method.values()]
    if len({frozenset(keys) for keys in populations}) != 1:
        raise RuntimeError("M2 method populations differ")
    for method, rows in rows_by_method.items():
        if any(row.get("status") != "ok" for row in rows):
            raise RuntimeError(f"M2 {method} includes error rows")
        if any(int((row.get("metrics") or {}).get("actual_forward_count") or 0) != 64 for row in rows):
            raise RuntimeError(f"M2 {method} violates the 64-forward budget")
    summary = write_grouped_effect_report(
        output_dir=output_dir,
        title="M2 Constraint-Homotopy V0",
        rows_by_method=rows_by_method,
        comparisons=((METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[1], METHODS[2])),
    )
    if run_manifest is not None:
        manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        full = dict(manifest.get("full") or {})
        summary["run_peak_cuda_memory_bytes"] = full.get("peak_cuda_memory_bytes")
        summary["run_full_wall_sec"] = full.get("wall_sec")
    activation = write_activation_artifacts(output_dir, rows_by_method, summary)
    summary["activation_audit"] = activation
    summary["scientific_judgment"] = activation["scientific_judgment"]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--vanilla-raw", required=True)
    root.add_argument("--gradual-raw", required=True)
    root.add_argument("--abrupt-raw", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--run-manifest")
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(
        Path(args.vanilla_raw).resolve(),
        Path(args.gradual_raw).resolve(),
        Path(args.abrupt_raw).resolve(),
        Path(args.output_dir).resolve(),
        run_manifest=Path(args.run_manifest).resolve() if args.run_manifest else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
