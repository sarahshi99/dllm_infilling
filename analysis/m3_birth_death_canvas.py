#!/usr/bin/env python3
"""Grouped equal-forward (not equal-token) analysis for M3."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.grouped_effects import DEFAULT_BOOTSTRAP_REPLICATES, write_grouped_effect_report


METHODS = ("m3_uniform_fixed_grid", "m3_birth_death_canvas")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _histogram(values: Sequence[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {key: int(counts[key]) for key in sorted(counts)}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _activation_audit(uniform: Sequence[Mapping[str, Any]], birth: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    uniform_by_key = {str(row["row_key"]): row for row in uniform}
    birth_by_key = {str(row["row_key"]): row for row in birth}
    if set(uniform_by_key) != set(birth_by_key):
        raise RuntimeError("M3 activation audit requires matched row keys")
    events = [event for row in birth for event in ((row.get("selection") or {}).get("birth_death_events") or [])]
    events_by_round: list[dict[str, Any]] = []
    for round_index in (15, 31, 47):
        at_round = [event for event in events if int(event.get("round", -1)) == round_index]
        events_by_round.append(
            {
                "round": round_index,
                "event_count": len(at_round),
                "dead_canvas_distribution": _histogram([event.get("dead_canvas") for event in at_round]),
                "born_canvas_distribution": _histogram([event.get("born_canvas") for event in at_round]),
            }
        )
    final_canvases = [canvas for row in birth for canvas in ((row.get("selection") or {}).get("particle_canvases_final") or [])]
    token_values = [int((row.get("metrics") or {}).get("actual_token_forward_budget") or 0) for row in birth]
    return {
        "contains_generated_code": False,
        "row_count": len(birth_by_key),
        "candidate_hash_difference_count": sum(
            str(uniform_by_key[key].get("candidate_middle_sha256") or "")
            != str(birth_by_key[key].get("candidate_middle_sha256") or "")
            for key in uniform_by_key
        ),
        "selected_canvas_distribution": _histogram([row.get("canvas_tokens") for row in birth]),
        "final_particle_canvas_distribution": _histogram(final_canvases),
        "birth_death_event_total": len(events),
        "birth_death_event_nonzero_row_count": sum(bool((row.get("selection") or {}).get("birth_death_events")) for row in birth),
        "population_changes_by_round": events_by_round,
        "birth_death_actual_token_forward": {
            "min": min(token_values) if token_values else 0,
            "mean": _mean([float(value) for value in token_values]),
            "max": max(token_values) if token_values else 0,
        },
        "uniform_birth_death_event_count": sum(len((row.get("selection") or {}).get("birth_death_events") or []) for row in uniform),
    }


def _write_frontier(output_dir: Path, summary: Mapping[str, Any]) -> None:
    rows = []
    for row in summary["method_summary"]:
        rows.append(
            {
                "method": row["method"],
                "equal_weight_task_macro_accuracy": row["equal_weight_task_macro_accuracy"],
                "span_micro_accuracy_descriptive": row["span_micro_accuracy_descriptive"],
                "actual_forwards": row["mean_standalone_forward_count"],
                "actual_token_forward_budget": row["mean_token_forward_budget"],
                "mean_wall_sec": row["mean_wall_sec"],
            }
        )
    for row in rows:
        row["pareto_non_dominated_accuracy_vs_token"] = not any(
            other["equal_weight_task_macro_accuracy"] >= row["equal_weight_task_macro_accuracy"]
            and other["actual_token_forward_budget"] <= row["actual_token_forward_budget"]
            and (
                other["equal_weight_task_macro_accuracy"] > row["equal_weight_task_macro_accuracy"]
                or other["actual_token_forward_budget"] < row["actual_token_forward_budget"]
            )
            for other in rows
        )
    with (output_dir / "accuracy_token_frontier.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(output_dir: Path, summary: Mapping[str, Any], activation: Mapping[str, Any]) -> None:
    effect = summary["paired_effects"][0]
    lines = [
        "# M3 Birth--Death Canvas Diffusion V0 — RandomSpanLight grouped result",
        "",
        "Primary estimand: equal-weight base-task macro accuracy over 148 task groups. Span-micro is descriptive only. The arms are equal-forward (256 forwards), not equal-token.",
        "",
        f"Paired task-macro delta (birth-death − uniform): `{effect['task_macro_delta']:.4f}` with 95% task-cluster CI [`{effect['task_macro_ci_low']:.4f}`, `{effect['task_macro_ci_high']:.4f}`]; wins/losses/ties=`{effect['wins']}/{effect['losses']}/{effect['ties']}`, help/harm=`{effect['span_help']}/{effect['span_harm']}`, group-aware label-swap p=`{effect['group_label_swap_two_sided_p']:.4f}`.",
        "",
        f"Activation audit: candidate hashes differ on `{activation['candidate_hash_difference_count']}/{activation['row_count']}` rows; events=`{activation['birth_death_event_total']}` across `{activation['birth_death_event_nonzero_row_count']}` rows; uniform events=`{activation['uniform_birth_death_event_count']}`. Birth-death token-forward min/mean/max=`{activation['birth_death_actual_token_forward']['min']}/{activation['birth_death_actual_token_forward']['mean']:.2f}/{activation['birth_death_actual_token_forward']['max']}`; uniform is 15,360 by contract.",
        "",
        "See `method_summary.csv`, `paired_effects.csv`, `length_bucket_summary.csv`, `activation_audit.json`, and `accuracy_token_frontier.csv` for paper-ready compact data. No raw generated code is emitted.",
    ]
    (output_dir / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    uniform_raw: Path,
    birth_raw: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    run_manifest: Path | None = None,
) -> dict[str, Any]:
    rows_by_method = {METHODS[0]: read_jsonl(uniform_raw), METHODS[1]: read_jsonl(birth_raw)}
    if {str(row["row_key"]) for row in rows_by_method[METHODS[0]]} != {str(row["row_key"]) for row in rows_by_method[METHODS[1]]}:
        raise RuntimeError("M3 populations differ")
    if any(len(rows) != 148 for rows in rows_by_method.values()):
        raise RuntimeError("M3 grouped analysis requires exactly 148 rows per arm")
    if any(len({str(row["task_group"]) for row in rows}) != 148 for rows in rows_by_method.values()):
        raise RuntimeError("M3 grouped analysis requires exactly 148 aligned base task groups")
    for method, rows in rows_by_method.items():
        if any(row.get("status") != "ok" for row in rows):
            raise RuntimeError(f"M3 {method} includes error rows")
        if any(int((row.get("metrics") or {}).get("actual_forward_count") or 0) != 256 for row in rows):
            raise RuntimeError(f"M3 {method} violates equal-forward accounting")
    summary = write_grouped_effect_report(
        output_dir=output_dir,
        title="M3 Birth-Death Canvas Diffusion V0",
        rows_by_method=rows_by_method,
        comparisons=((METHODS[1], METHODS[0]),),
        bootstrap_replicates=bootstrap_replicates,
    )
    summary["compute_claim"] = "equal_forward_only; actual token-forward totals are descriptive and may differ"
    summary["uniform_token_forward_contract"] = 15360
    summary["fixed_comparison"] = [METHODS[1], METHODS[0]]
    if run_manifest is not None:
        manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        full = dict(manifest.get("full") or {})
        summary["run_peak_cuda_memory_bytes"] = full.get("peak_cuda_memory_bytes")
        summary["run_full_wall_sec"] = full.get("wall_sec")
    activation = _activation_audit(rows_by_method[METHODS[0]], rows_by_method[METHODS[1]])
    (output_dir / "activation_audit.json").write_text(json.dumps(activation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_frontier(output_dir, summary)
    _write_report(output_dir, summary, activation)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--uniform-raw", required=True)
    root.add_argument("--birth-death-raw", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    root.add_argument("--run-manifest")
    return root


def main() -> int:
    args = parser().parse_args()
    run_analysis(
        Path(args.uniform_raw).resolve(),
        Path(args.birth_death_raw).resolve(),
        Path(args.output_dir).resolve(),
        bootstrap_replicates=int(args.bootstrap_replicates),
        run_manifest=Path(args.run_manifest).resolve() if args.run_manifest else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
