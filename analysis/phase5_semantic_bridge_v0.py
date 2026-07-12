#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.phase5_premise_falsification import (
    DEPLOYABLE_PROXY_SCORE_KEYS,
    FORBIDDEN_DEPLOYABLE_FEATURES,
    grouped_bootstrap_metric,
    read_jsonl,
    validate_feature_names,
    write_csv,
    write_json,
)

MECHANISM_NAME = "AST/def-use bridge proxy V0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def boolish(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def floatish(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def hash_group(group: str) -> str:
    return hashlib.sha256(group.encode("utf-8")).hexdigest()[:16]


def bucket_from_reference_tokens(tokens: int) -> str:
    if tokens <= 8:
        return "short"
    if tokens <= 16:
        return "medium"
    if tokens <= 24:
        return "long"
    return "extreme"


def select_by_score(rows: Sequence[Mapping[str, Any]], score_key: str) -> Mapping[str, Any]:
    return max(
        rows,
        key=lambda row: (
            floatish(row.get(score_key)),
            -int(floatish(row.get("canvas_tokens"))),
            -int(floatish(row.get("seed"))),
        ),
    )


def validate_deployable_score_key(score_key: str) -> None:
    if score_key not in DEPLOYABLE_PROXY_SCORE_KEYS:
        raise ValueError(f"V0 rejects non-deployable or supervised score key: {score_key}")


def validate_proxy_score_schema(rows: Sequence[Mapping[str, Any]]) -> None:
    allowed = {"candidate_key", "row_key", "canvas_tokens", "seed", *DEPLOYABLE_PROXY_SCORE_KEYS}
    for row in rows:
        extra = set(row) - allowed
        if extra:
            raise ValueError(f"Deployable proxy score file contains forbidden/diagnostic columns: {sorted(extra)}")


def exact_binomial_two_sided(discordant_a: int, discordant_b: int) -> float:
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    probability = sum(math.comb(n, k) for k in range(lower + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def paired_summary(selections: Sequence[Mapping[str, Any]], method: str, baseline: str) -> dict[str, Any]:
    wins = sum(boolish(row[f"{method}_passed"]) and not boolish(row[f"{baseline}_passed"]) for row in selections)
    losses = sum(not boolish(row[f"{method}_passed"]) and boolish(row[f"{baseline}_passed"]) for row in selections)
    tie_pass = sum(boolish(row[f"{method}_passed"]) and boolish(row[f"{baseline}_passed"]) for row in selections)
    tie_fail = sum(not boolish(row[f"{method}_passed"]) and not boolish(row[f"{baseline}_passed"]) for row in selections)
    return {
        "baseline": baseline,
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
        "tie_pass": tie_pass,
        "tie_fail": tie_fail,
        "exact_paired_two_sided_p": exact_binomial_two_sided(wins, losses),
    }


def bootstrap_pass_rate(selections: Sequence[Mapping[str, Any]], key: str, replicates: int) -> dict[str, Any]:
    return grouped_bootstrap_metric(
        selections,
        "group_key",
        lambda sample: sum(boolish(row[key]) for row in sample) / len(sample) if sample else float("nan"),
        replicates=replicates,
    )


def render_killed(gate: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# AST/Def-Use Bridge Proxy V0",
            "",
            "Verdict: `killed_corrected_within_task_gate_failed`.",
            "",
            "AST/def-use bridge proxy V0 was not run as a deployable reranker because F3 did not meet the corrected within-task/cross-canvas gate.",
            f"Failed conditions: `{gate.get('failed_conditions', [])}`.",
            "The pass-trained supervised probe is diagnostic only and was not supplied to selection.",
            "No homotopy, birth–death, particle assembly, fusion, heuristic fallback, or additional generation was run.",
            "This proxy is not full Semantic Bridge Projection or Abductive Program-State Bridge.",
            "Frozen controller test remains sealed with `test_evaluation_count=0`.",
        ]
    ) + "\n"


def write_killed_outputs(output_dir: Path, gate: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": "killed_corrected_within_task_gate_failed",
        "mechanism_name": MECHANISM_NAME,
        "ast_def_use_bridge_proxy_v0_implemented": False,
        "gate": dict(gate),
        "supervised_probe_scores_used_for_selection": False,
        "frozen_test_status": "sealed",
        "test_evaluation_count": 0,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "gate_decision.json", summary)
    (output_dir / "report.md").write_text(render_killed(gate), encoding="utf-8")
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# AST/Def-Use Bridge Proxy V0",
        "",
        f"Verdict: `{summary['verdict']}`.",
        "",
        f"Pass@1: `{summary['pass_count']}/{summary['task_count']}` (`{summary['pass_rate']:.4f}`).",
        f"Fixed64 control: `{summary['fixed64_pass_count']}/{summary['task_count']}`.",
        f"Confidence reranking: `{summary['confidence_pass_count']}/{summary['task_count']}`.",
        f"Peak GPU memory inherited from shared bank: `{summary['peak_cuda_memory_bytes']}` bytes.",
        "No new GPU generation was run for reranking.",
        "",
        "## Paired Help/Harm",
        "",
        "| Baseline | Wins | Losses | Net | Exact paired p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["paired"]:
        lines.append(
            f"| `{row['baseline']}` | `{row['wins']}` | `{row['losses']}` | `{row['net']}` | `{row['exact_paired_two_sided_p']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| Score | Pass | Rate |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["ablations"]:
        lines.append(f"| `{row['score_family']}` | `{row['pass_count']}` | `{row['pass_rate']:.4f}` |")
    lines.extend(
        [
            "",
            "## Forbidden-Feature Audit",
            "",
            f"Passed: `{summary['forbidden_feature_audit']['passed']}`.",
            "Features exclude reference code, oracle length, unit-test outcomes, task IDs, split labels, errors, and test-derived statistics.",
            "Frozen controller test remains sealed with `test_evaluation_count=0`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    bank_dir = Path(args.bank_dir).resolve()
    premise_dir = Path(args.premise_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = json.loads((premise_dir / "f3_gate.json").read_text(encoding="utf-8"))
    if not bool(gate.get("passed", False)):
        write_killed_outputs(output_dir, gate)
        return 0

    prediction_rows = read_csv(premise_dir / "f3_deployable_proxy_scores.csv")
    validate_proxy_score_schema(prediction_rows)
    raw_rows = [
        row
        for row in read_jsonl(bank_dir / "candidate_bank_raw.jsonl")
        if row.get("candidate_kind") == "deployable_grid" and row.get("status") == "ok"
    ]
    prediction_by_key = {str(row["candidate_key"]): row for row in prediction_rows}
    bank_summary_path = Path(args.compact_bank_dir).resolve() / "full_summary.json" if args.compact_bank_dir else None
    bank_summary = json.loads(bank_summary_path.read_text(encoding="utf-8")) if bank_summary_path and bank_summary_path.exists() else {}
    raw_by_key = {str(row["candidate_key"]): row for row in raw_rows}
    if set(prediction_by_key) != set(raw_by_key):
        missing = sorted(set(raw_by_key) - set(prediction_by_key))
        extra = sorted(set(prediction_by_key) - set(raw_by_key))
        raise RuntimeError(f"Proxy/raw candidate-key mismatch: missing={missing[:5]} extra={extra[:5]}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in prediction_rows:
        groups[str(prediction["row_key"])].append(dict(prediction))

    ablation_scores = {
        "prefix_only": "deployable_proxy_prefix_only",
        "suffix_only": "deployable_proxy_suffix_only",
        "combined": "deployable_proxy_combined",
        "length_only": "deployable_proxy_token_canvas",
        "confidence": "deployable_proxy_ordinary_confidence",
    }
    for key in ablation_scores.values():
        validate_deployable_score_key(key)
    selections: list[dict[str, Any]] = []
    for row_key, rows in sorted(groups.items()):
        fixed = next(row for row in rows if int(row["canvas_tokens"]) == 64 and int(row["seed"]) == 0)
        selected = {name: select_by_score(rows, score) for name, score in ablation_scores.items()}
        fixed_raw = raw_by_key[str(fixed["candidate_key"])]
        reference_tokens = int(fixed_raw["reference_middle_tokens"])
        result: dict[str, Any] = {
            "row_key": row_key,
            "group_key": hash_group(str(fixed_raw["task_group"])),
            "length_bucket_offline_only": bucket_from_reference_tokens(reference_tokens),
            "fixed64_candidate_key": fixed["candidate_key"],
            "fixed64_passed": bool(fixed_raw["passed"]),
        }
        for name, row in selected.items():
            raw = raw_by_key[str(row["candidate_key"])]
            result[f"{name}_candidate_key"] = row["candidate_key"]
            result[f"{name}_canvas_tokens"] = row["canvas_tokens"]
            result[f"{name}_seed"] = row["seed"]
            result[f"{name}_score"] = row[ablation_scores[name]]
            result[f"{name}_passed"] = bool(raw["passed"])
        selections.append(result)

    combined_pass = sum(boolish(row["combined_passed"]) for row in selections)
    fixed_pass = sum(boolish(row["fixed64_passed"]) for row in selections)
    confidence_pass = sum(boolish(row["confidence_passed"]) for row in selections)
    ablations = []
    for name in ["prefix_only", "suffix_only", "combined", "length_only"]:
        count = sum(boolish(row[f"{name}_passed"]) for row in selections)
        ci = bootstrap_pass_rate(selections, f"{name}_passed", args.bootstrap_replicates)
        ablations.append(
            {
                "score_family": name,
                "pass_count": count,
                "pass_rate": count / len(selections),
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    bucket_rows = []
    for bucket in ["short", "medium", "long", "extreme"]:
        rows = [row for row in selections if row["length_bucket_offline_only"] == bucket]
        bucket_rows.append(
            {
                "bucket": bucket,
                "tasks": len(rows),
                "semantic_bridge_pass": sum(boolish(row["combined_passed"]) for row in rows),
                "fixed64_pass": sum(boolish(row["fixed64_passed"]) for row in rows),
                "confidence_pass": sum(boolish(row["confidence_passed"]) for row in rows),
                "semantic_bridge_vs_fixed_wins": sum(boolish(row["combined_passed"]) and not boolish(row["fixed64_passed"]) for row in rows),
                "semantic_bridge_vs_fixed_losses": sum(not boolish(row["combined_passed"]) and boolish(row["fixed64_passed"]) for row in rows),
            }
        )
    selected_keys = {str(row["combined_candidate_key"]) for row in selections}
    selected_raw = [row for row in raw_rows if str(row["candidate_key"]) in selected_keys]
    total_candidate_latency = sum(floatish((row.get("metrics") or {}).get("total_sec_including_probe"), floatish(row.get("wall_sec"))) for row in raw_rows)
    selected_latency = sum(floatish((row.get("metrics") or {}).get("total_sec_including_probe"), floatish(row.get("wall_sec"))) for row in selected_raw)
    forbidden_audit = {
        "passed": True,
        "forbidden_features": sorted(FORBIDDEN_DEPLOYABLE_FEATURES),
        "selection_score_column": "deployable_proxy_combined",
        "labels_used_for_offline_evaluation": True,
        "supervised_probe_diagnostic": {
            "outcome_labels_used_for_fit": True,
            "outcome_labels_used_for_selection": False,
            "deployable_authorization_role": "none",
        },
        "deployable_bridge_proxy": {
            "outcome_labels_used_for_fit": False,
            "reference_used_for_fit": False,
            "outcome_labels_used_for_selection": False,
            "reference_used_for_selection": False,
            "fitted_parameters": False,
            "outcome_labels_used_for_offline_gate_evaluation": True,
        },
    }
    paired = [
        paired_summary(selections, "combined", "fixed64"),
        paired_summary(selections, "combined", "confidence"),
    ]
    summary = {
        "verdict": "ast_def_use_bridge_proxy_v0_completed",
        "mechanism_name": MECHANISM_NAME,
        "ast_def_use_bridge_proxy_v0_implemented": True,
        "supervised_probe_scores_used_for_selection": False,
        "task_count": len(selections),
        "candidate_count_per_task": 8,
        "candidate_rows_scored": len(raw_rows),
        "pass_count": combined_pass,
        "pass_rate": combined_pass / len(selections),
        "pass_rate_grouped_bootstrap": bootstrap_pass_rate(selections, "combined_passed", args.bootstrap_replicates),
        "fixed64_pass_count": fixed_pass,
        "fixed64_pass_rate": fixed_pass / len(selections),
        "confidence_pass_count": confidence_pass,
        "confidence_pass_rate": confidence_pass / len(selections),
        "paired": paired,
        "ablations": ablations,
        "bucket_results": bucket_rows,
        "denoising_steps_per_candidate": 64,
        "gpu_generation_sec_shared_bank": floatish(bank_summary.get("summed_gpu_decode_sec"), total_candidate_latency),
        "verification_sec_shared_bank": floatish(bank_summary.get("summed_verification_sec"), 0.0),
        "wall_sec_shared_bank": floatish(bank_summary.get("wall_sec"), 0.0),
        "selected_candidate_latency_sec": selected_latency,
        "mean_selected_candidate_latency_sec": selected_latency / len(selected_raw) if selected_raw else 0.0,
        "peak_cuda_memory_bytes": int(floatish(bank_summary.get("peak_cuda_memory_bytes"), 0.0)),
        "new_gpu_generation_for_reranker": False,
        "gate": gate,
        "forbidden_feature_audit": forbidden_audit,
        "frozen_test_status": "sealed",
        "test_evaluation_count": 0,
    }
    write_csv(output_dir / "selections.csv", selections)
    write_csv(output_dir / "bucket_results.csv", bucket_rows)
    write_csv(output_dir / "ablations.csv", ablations)
    write_csv(output_dir / "paired_help_harm.csv", paired)
    write_json(output_dir / "forbidden_feature_audit.json", forbidden_audit)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Conditional deterministic AST/def-use bridge proxy V0 reranker")
    root.add_argument("--bank-dir", required=True)
    root.add_argument("--compact-bank-dir")
    root.add_argument("--premise-dir", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--bootstrap-replicates", type=int, default=1000)
    return root


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
