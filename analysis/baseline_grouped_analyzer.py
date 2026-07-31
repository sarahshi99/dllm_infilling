#!/usr/bin/env python3
"""Unified row- and base-function-grouped analyzer for external baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.grouped_effects import cluster_bootstrap


DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_SEED = 20260731
MANIFEST_FORBIDDEN_FIELDS = {
    "canonical_solution",
    "completion",
    "evaluator_outcome",
    "evaluator_result",
    "oracle_length",
    "passed",
    "reference_code",
    "test",
    "tests",
}


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric(row: Mapping[str, Any], *keys: str) -> Any:
    metrics = row.get("metrics") or {}
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
        if key in row and row[key] is not None:
            return row[key]
    return None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("manifest is empty")
    forbidden = sorted({field for row in rows for field in MANIFEST_FORBIDDEN_FIELDS if field in row})
    if forbidden:
        raise RuntimeError(f"manifest contains forbidden fields: {forbidden}")
    for field in ("task_id", "task_group"):
        values = [str(row.get(field) or "") for row in rows]
        if any(not value for value in values):
            raise RuntimeError(f"manifest has blank {field}")
        if field == "task_id" and len(values) != len(set(values)):
            raise RuntimeError(f"manifest has duplicate {field}")
    candidate_keys = [str(row.get("candidate_key") or row["task_id"]) for row in rows]
    if any(not value for value in candidate_keys):
        raise RuntimeError("manifest has blank normalized candidate_key")
    if len(candidate_keys) != len(set(candidate_keys)):
        raise RuntimeError("manifest has duplicate normalized candidate_key")


def normalize_rows(
    raw_rows: Sequence[Mapping[str, Any]], manifest_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validate_manifest(manifest_rows)
    manifest_by_task = {str(row["task_id"]): row for row in manifest_rows}
    raw_by_task: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise RuntimeError("raw row has blank task_id")
        if task_id in raw_by_task:
            raise RuntimeError(f"raw rows contain duplicate task_id {task_id}")
        raw_by_task[task_id] = row
    if set(raw_by_task) != set(manifest_by_task):
        missing = sorted(set(manifest_by_task) - set(raw_by_task))[:5]
        extra = sorted(set(raw_by_task) - set(manifest_by_task))[:5]
        raise RuntimeError(f"raw rows do not match manifest: missing={missing} extra={extra}")
    normalized: list[dict[str, Any]] = []
    for task_id in sorted(manifest_by_task):
        manifest_row = manifest_by_task[task_id]
        raw = raw_by_task[task_id]
        raw_group = str(raw.get("task_group") or manifest_row["task_group"])
        if raw_group != str(manifest_row["task_group"]):
            raise RuntimeError(f"task_group mismatch for {task_id}")
        normalized.append(
            {
                "candidate_key": str(manifest_row.get("candidate_key") or manifest_row["task_id"]),
                "task_id": task_id,
                "task_group": str(manifest_row["task_group"]),
                "status": str(raw.get("status") or ""),
                "passed": _bool(raw.get("passed")),
                "metrics": dict(raw.get("metrics") or {}),
                "raw_run_key": str(raw.get("candidate_key") or raw.get("run_key") or ""),
            }
        )
    return normalized


def _task_accuracies(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_group"])].append(float(_bool(row.get("passed"))))
    return {group: sum(values) / len(values) for group, values in grouped.items()}


def _cost_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    search: list[float] = []
    decode: list[float] = []
    total: list[float] = []
    tokens: list[float] = []
    walls: list[float] = []
    memories: list[float] = []
    for row in rows:
        search_value = _number(_metric(row, "search_forwards", "search_forward_calls"))
        decode_value = _number(
            _metric(row, "formal_decode_forwards", "decode_forwards", "decode_forward_calls")
        )
        total_value = _number(
            _metric(row, "total_forwards", "total_forward_calls", "actual_forward_count")
        )
        if search_value is not None and decode_value is not None and total_value is not None:
            if abs(total_value - search_value - decode_value) > 1e-9:
                raise RuntimeError("forward accounting does not satisfy total=search+decode")
        if search_value is not None:
            search.append(search_value)
        if decode_value is not None:
            decode.append(decode_value)
        if total_value is not None:
            total.append(total_value)
        value = _number(
            _metric(
                row,
                "token_budget",
                "token_forwards",
                "total_token_forwards",
                "actual_token_forward_budget",
                "standalone_token_budget",
            )
        )
        if value is not None:
            tokens.append(value)
        value = _number(_metric(row, "wall_sec", "total_sec", "total_sec_including_probe"))
        if value is not None:
            walls.append(value)
        value = _number(_metric(row, "peak_memory_bytes", "peak_gpu_memory_bytes"))
        if value is not None:
            memories.append(value)
    return {
        "total_search_forward_calls": sum(search) if search else None,
        "mean_search_forward_calls": _mean(search),
        "total_decode_forward_calls": sum(decode) if decode else None,
        "mean_decode_forward_calls": _mean(decode),
        "total_forward_calls": sum(total) if total else None,
        "mean_total_forward_calls": _mean(total),
        "total_token_forwards": sum(tokens) if tokens else None,
        "mean_token_forwards": _mean(tokens),
        "total_wall_sec": sum(walls) if walls else None,
        "mean_wall_sec": _mean(walls),
        "peak_gpu_memory_bytes": max(memories) if memories else None,
    }


def _dynamic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [
        value
        for row in rows
        if (
            value := _number(
                _metric(
                    row,
                    "selected_length",
                    "final_length",
                    "generated_length",
                    "final_generated_length",
                    "final_gen_length",
                )
            )
        )
        is not None
    ]
    expansions = [
        value
        for row in rows
        if (
            value := _number(
                _metric(
                    row,
                    "expansion_count",
                    "num_expansions",
                    "expansion_moves",
                    "net_expansion_tokens",
                )
            )
        )
        is not None
    ]
    contractions = [
        value
        for row in rows
        if (
            value := _number(
                _metric(
                    row,
                    "contraction_count",
                    "num_contractions",
                    "contraction_moves",
                    "net_contraction_tokens",
                )
            )
        )
        is not None
    ]
    termination = Counter(
        str(value)
        for row in rows
        if (value := _metric(row, "termination_reason", "stop_reason")) not in (None, "")
    )
    return {
        "selected_length_count": len(lengths),
        "selected_length_mean": _mean(lengths),
        "selected_length_min": min(lengths) if lengths else None,
        "selected_length_p50": _percentile(lengths, 0.50),
        "selected_length_p90": _percentile(lengths, 0.90),
        "selected_length_max": max(lengths) if lengths else None,
        "expansion_count_total": sum(expansions) if expansions else None,
        "expansion_positive_rows": sum(value > 0 for value in expansions),
        "contraction_count_total": sum(contractions) if contractions else None,
        "contraction_positive_rows": sum(value > 0 for value in contractions),
        "termination_reason_counts": dict(sorted(termination.items())),
    }


def summarize_method(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    expected_rows: int,
    expected_clusters: int,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if len(rows) != int(expected_rows):
        raise RuntimeError(f"{method} row count mismatch: {len(rows)} != {expected_rows}")
    if any(row.get("status") != "ok" for row in rows):
        raise RuntimeError(f"{method} contains non-ok rows")
    keys = [str(row["candidate_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{method} contains duplicate candidate keys")
    task_values = _task_accuracies(rows)
    if len(task_values) != int(expected_clusters):
        raise RuntimeError(
            f"{method} cluster count mismatch: {len(task_values)} != {expected_clusters}"
        )
    ci = cluster_bootstrap(
        list(task_values.values()), replicates=bootstrap_replicates, seed=seed
    )
    result = {
        "method": method,
        "row_count": len(rows),
        "cluster_count": len(task_values),
        "row_pass_at_1": sum(float(_bool(row.get("passed"))) for row in rows) / len(rows),
        "equal_weight_task_macro_pass_at_1": ci["estimate"],
        "task_macro_ci_low": ci["ci_low"],
        "task_macro_ci_high": ci["ci_high"],
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(seed),
        "dynamic": _dynamic_summary(rows),
    }
    result.update(_cost_summary(rows))
    return result


def paired_comparison(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    method_a: str,
    method_b: str,
    expected_clusters: int,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    left = {str(row["candidate_key"]): row for row in rows_a}
    right = {str(row["candidate_key"]): row for row in rows_b}
    if set(left) != set(right):
        raise RuntimeError(f"paired methods have different candidate keys: {method_a} vs {method_b}")
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    row_help = row_harm = 0
    for key in sorted(left):
        a, b = left[key], right[key]
        if str(a["task_group"]) != str(b["task_group"]):
            raise RuntimeError(f"paired task_group mismatch at {key}")
        grouped[str(a["task_group"])].append((a, b))
        row_help += int(_bool(a.get("passed")) and not _bool(b.get("passed")))
        row_harm += int(not _bool(a.get("passed")) and _bool(b.get("passed")))
    if len(grouped) != int(expected_clusters):
        raise RuntimeError(f"paired cluster count mismatch: {len(grouped)} != {expected_clusters}")
    deltas: list[float] = []
    cluster_help = cluster_harm = cluster_tie = 0
    for pairs in grouped.values():
        accuracy_a = sum(float(_bool(a.get("passed"))) for a, _ in pairs) / len(pairs)
        accuracy_b = sum(float(_bool(b.get("passed"))) for _, b in pairs) / len(pairs)
        delta = accuracy_a - accuracy_b
        deltas.append(delta)
        cluster_help += int(delta > 0)
        cluster_harm += int(delta < 0)
        cluster_tie += int(delta == 0)
    ci = cluster_bootstrap(deltas, replicates=bootstrap_replicates, seed=seed)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "paired_candidate_key_count": len(left),
        "cluster_count": len(grouped),
        "task_macro_delta": ci["estimate"],
        "task_macro_delta_ci_low": ci["ci_low"],
        "task_macro_delta_ci_high": ci["ci_high"],
        "row_help": row_help,
        "row_harm": row_harm,
        "cluster_help": cluster_help,
        "cluster_harm": cluster_harm,
        "cluster_tie": cluster_tie,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(seed),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key != "dynamic" and key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_analysis(
    *,
    output_dir: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    raw_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    comparisons: Sequence[tuple[str, str]],
    expected_rows: int,
    expected_clusters: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    normalized = {
        method: normalize_rows(rows, manifest_rows) for method, rows in raw_by_method.items()
    }
    summaries = [
        summarize_method(
            rows,
            method=method,
            expected_rows=expected_rows,
            expected_clusters=expected_clusters,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index,
        )
        for index, (method, rows) in enumerate(normalized.items())
    ]
    effects = [
        paired_comparison(
            normalized[method_a],
            normalized[method_b],
            method_a=method_a,
            method_b=method_b,
            expected_clusters=expected_clusters,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 100 + index,
        )
        for index, (method_a, method_b) in enumerate(comparisons)
    ]
    payload = {
        "schema_version": 1,
        "primary_estimand": "equal_weight_base_function_task_macro_pass_at_1",
        "row_metric": "row_level_pass_at_1",
        "inference_unit": "base_function_task_group",
        "row_count": int(expected_rows),
        "cluster_count": int(expected_clusters),
        "bootstrap_replicates": int(bootstrap_replicates),
        "methods": summaries,
        "paired_comparisons": effects,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "method_summary.csv", summaries)
    _write_csv(output_dir / "paired_comparisons.csv", effects)
    lines = [
        "# External Baseline Grouped Analysis",
        "",
        "Primary: equal-weight base-function task-macro Pass@1 with 10,000 cluster bootstrap replicates. Row Pass@1 is reported separately.",
        "",
        "| Method | Row Pass@1 | Task macro | 95% cluster CI | Total forwards | Token-forwards |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['method']}` | `{row['row_pass_at_1']:.6f}` | "
            f"`{row['equal_weight_task_macro_pass_at_1']:.6f}` | "
            f"`[{row['task_macro_ci_low']:.6f}, {row['task_macro_ci_high']:.6f}]` | "
            f"`{row['total_forward_calls']}` | `{row['total_token_forwards']}` |"
        )
    (output_dir / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--method", action="append", required=True, help="NAME=raw.jsonl")
    result.add_argument("--comparison", action="append", default=[], help="METHOD_A:METHOD_B")
    result.add_argument("--expected-rows", type=int, required=True)
    result.add_argument("--expected-clusters", type=int, required=True)
    result.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    raw_by_method: dict[str, list[dict[str, Any]]] = {}
    for specification in args.method:
        name, separator, path = specification.partition("=")
        if not separator or not name or not path:
            raise ValueError("--method must be NAME=raw.jsonl")
        raw_by_method[name] = load_jsonl(Path(path))
    comparisons: list[tuple[str, str]] = []
    for specification in args.comparison:
        left, separator, right = specification.partition(":")
        if not separator or left not in raw_by_method or right not in raw_by_method:
            raise ValueError("--comparison must reference two declared methods")
        comparisons.append((left, right))
    write_analysis(
        output_dir=args.output_dir,
        manifest_rows=load_jsonl(args.manifest),
        raw_by_method=raw_by_method,
        comparisons=comparisons,
        expected_rows=args.expected_rows,
        expected_clusters=args.expected_clusters,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
