#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.trace_long_rescue_features import load_jsonl, metric


JsonDict = Dict[str, Any]

ANCHOR_ID = "len32_s64"
SHORT_ID = "len24_s64"
SLOW_ID = "len32_s96"

FORBIDDEN_FEATURE_TOKENS = (
    "oracle",
    "passed",
    "pass",
    "label",
    "pairwise",
    "risk",
    "win",
    "loss",
    "failed",
    "failure",
    "verification",
    "offline",
    "task",
    "bucket",
)


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _bool(value: Any) -> bool:
    return bool(value)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _rows_by_task(rows: Iterable[Mapping[str, Any]], *, name: str) -> Dict[str, Mapping[str, Any]]:
    by_task: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id is None:
            raise ValueError(f"Missing task_id in {name}")
        by_task[str(task_id)] = row
    return by_task


def oracle_bucket(oracle_length: Optional[int]) -> str:
    if oracle_length is None:
        return "unknown"
    if oracle_length <= 8:
        return "<=8"
    if oracle_length <= 12:
        return "9-12"
    if oracle_length <= 16:
        return "13-16"
    if oracle_length <= 24:
        return "17-24"
    return "25+"


def pairwise(policy_passed: bool, reference_passed: bool) -> str:
    if policy_passed and not reference_passed:
        return "win"
    if (not policy_passed) and reference_passed:
        return "loss"
    if policy_passed and reference_passed:
        return "tie_pass"
    return "tie_fail"


def is_policy_feature(name: str) -> bool:
    lower = name.lower()
    return not any(token in lower for token in FORBIDDEN_FEATURE_TOKENS)


def candidate_by_id(v5_row: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    v5 = v5_row.get("route2_rescue_quality_v5") or {}
    candidates = v5.get("candidates") or []
    return {
        str(candidate.get("candidate_id")): candidate
        for candidate in candidates
        if candidate.get("candidate_id") is not None
    }


def candidate_pass(candidate: Optional[Mapping[str, Any]]) -> bool:
    return bool(candidate and candidate.get("offline_passed"))


def _text_features(candidate: Mapping[str, Any], *, prefix: str) -> JsonDict:
    middle = str(candidate.get("middle_text") or "")
    lines = middle.splitlines()
    nonempty = [line for line in lines if line.strip()]
    return {
        f"{prefix}_text_chars": float(len(middle)),
        f"{prefix}_text_lines": float(len(lines)),
        f"{prefix}_text_nonempty_lines": float(len(nonempty)),
        f"{prefix}_text_leading_spaces": float(len(middle) - len(middle.lstrip(" "))) if middle else 0.0,
    }


def _candidate_features(candidate: Mapping[str, Any], *, prefix: str) -> JsonDict:
    features: JsonDict = {
        f"{prefix}_requested_min_length": _num(candidate.get("requested_min_length")),
        f"{prefix}_steps": _num(candidate.get("steps")),
        f"{prefix}_selected_mask_length": _num(candidate.get("selected_mask_length")),
        f"{prefix}_actual_rescue_length": _num(candidate.get("actual_rescue_length")),
        f"{prefix}_total_sec_including_probe": _num(candidate.get("total_sec_including_probe")),
        f"{prefix}_syntax_parse_ok": 1.0 if (candidate.get("syntax") or {}).get("parse_passed") else 0.0,
        f"{prefix}_syntax_compile_ok": 1.0 if (candidate.get("syntax") or {}).get("compile_passed") else 0.0,
    }
    trace = candidate.get("trace_features") or {}
    for key, value in trace.items():
        parsed = _num(value)
        if parsed is not None:
            features[f"{prefix}_trace_{key}"] = parsed
    features.update(_text_features(candidate, prefix=prefix))
    return features


def build_policy_features(candidates: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    features: JsonDict = {}
    short = candidates.get(SHORT_ID)
    anchor = candidates.get(ANCHOR_ID)
    slow = candidates.get(SLOW_ID)
    if short:
        features.update(_candidate_features(short, prefix="short"))
    if anchor:
        features.update(_candidate_features(anchor, prefix="anchor"))
    if slow:
        features.update(_candidate_features(slow, prefix="slow"))
    if short and anchor:
        short_features = _candidate_features(short, prefix="short")
        anchor_features = _candidate_features(anchor, prefix="anchor")
        for short_key, short_value in short_features.items():
            suffix = short_key.removeprefix("short_")
            anchor_value = anchor_features.get(f"anchor_{suffix}")
            if _num(short_value) is not None and _num(anchor_value) is not None:
                features[f"delta_short_minus_anchor_{suffix}"] = float(short_value) - float(anchor_value)
        features["short_anchor_text_equal"] = 1.0 if short.get("middle_text") == anchor.get("middle_text") else 0.0
    if slow and anchor:
        slow_features = _candidate_features(slow, prefix="slow")
        anchor_features = _candidate_features(anchor, prefix="anchor")
        for slow_key, slow_value in slow_features.items():
            suffix = slow_key.removeprefix("slow_")
            anchor_value = anchor_features.get(f"anchor_{suffix}")
            if _num(slow_value) is not None and _num(anchor_value) is not None:
                features[f"delta_slow_minus_anchor_{suffix}"] = float(slow_value) - float(anchor_value)
    return {key: value for key, value in features.items() if is_policy_feature(key)}


def feature_columns(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    columns = sorted(
        {
            key
            for row in rows
            for key, value in (row.get("features") or {}).items()
            if is_policy_feature(key) and _num(value) is not None
        }
    )
    return columns


def build_triggered_table(
    v5_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    reference_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[List[JsonDict], JsonDict]:
    baseline = _rows_by_task(baseline_rows or [], name="baseline") if baseline_rows else {}
    reference = _rows_by_task(reference_rows or [], name="reference") if reference_rows else {}
    triggered: List[JsonDict] = []
    all_passed = 0
    pairwise_counts = Counter()
    for row in v5_rows:
        task_id = str(row.get("task_id"))
        passed = _bool(metric(row, "passed", False))
        all_passed += int(passed)
        ref_passed = _bool(metric(reference[task_id], "passed", False)) if task_id in reference else passed
        pairwise_counts[pairwise(passed, ref_passed)] += 1
        v5 = row.get("route2_rescue_quality_v5") or {}
        if not v5.get("triggered"):
            continue
        candidates = candidate_by_id(row)
        oracle_len = metric(row, "oracle_mask_length")
        oracle_int = int(oracle_len) if oracle_len is not None else None
        baseline_passed = _bool(metric(baseline[task_id], "passed", False)) if task_id in baseline else False
        anchor_pass = candidate_pass(candidates.get(ANCHOR_ID))
        short_pass = candidate_pass(candidates.get(SHORT_ID))
        slow_pass = candidate_pass(candidates.get(SLOW_ID))
        selected_id = str(v5.get("selected_candidate_id"))
        triggered.append(
            {
                "task_id": task_id,
                "oracle_length": oracle_int,
                "oracle_bucket": oracle_bucket(oracle_int),
                "baseline_passed": baseline_passed,
                "v5_selected_candidate_id": selected_id,
                "v5_passed": passed,
                "anchor_passed": anchor_pass,
                "short_passed": short_pass,
                "slow_passed": slow_pass,
                "any_candidate_passed": bool(anchor_pass or short_pass or slow_pass),
                "short_only_win": bool(short_pass and not anchor_pass),
                "anchor_only_win": bool(anchor_pass and not short_pass),
                "features": build_policy_features(candidates),
            }
        )
    summary = {
        "num_samples": len(v5_rows),
        "v5_pass_count": all_passed,
        "triggered_count": len(triggered),
        "candidate_upper_bound_count": sum(1 for row in triggered if row["any_candidate_passed"]),
        "selected_triggered_pass_count": sum(1 for row in triggered if row["v5_passed"]),
        "pairwise_vs_reference": {
            "win": int(pairwise_counts.get("win", 0)),
            "loss": int(pairwise_counts.get("loss", 0)),
            "tie_pass": int(pairwise_counts.get("tie_pass", 0)),
            "tie_fail": int(pairwise_counts.get("tie_fail", 0)),
        },
    }
    return triggered, summary


@dataclass(frozen=True)
class Rule:
    name: str
    clauses: Tuple[Tuple[str, str, float], ...]
    family: str

    def matches(self, row: Mapping[str, Any]) -> bool:
        features = row.get("features") or {}
        for feature, op, threshold in self.clauses:
            value = _num(features.get(feature))
            if value is None:
                return False
            if op == "<=":
                if not value <= threshold:
                    return False
            elif op == ">=":
                if not value >= threshold:
                    return False
            else:
                raise ValueError(f"Unknown op: {op}")
        return True


def _format_threshold(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "neg").replace(".", "p")


def candidate_thresholds(values: Sequence[float], *, max_thresholds: int = 16) -> List[float]:
    unique = sorted({float(value) for value in values if math.isfinite(float(value))})
    if len(unique) <= max_thresholds:
        return unique
    indexes = {
        round(i * (len(unique) - 1) / (max_thresholds - 1))
        for i in range(max_thresholds)
    }
    return [unique[index] for index in sorted(indexes)]


def generate_single_feature_rules(rows: Sequence[Mapping[str, Any]]) -> List[Rule]:
    rules: List[Rule] = []
    for column in feature_columns(rows):
        values = [_num((row.get("features") or {}).get(column)) for row in rows]
        numeric = [value for value in values if value is not None]
        if not numeric:
            continue
        for threshold in candidate_thresholds(numeric):
            for op in ("<=", ">="):
                rules.append(
                    Rule(
                        name=f"{column}_{'le' if op == '<=' else 'ge'}_{_format_threshold(threshold)}",
                        clauses=((column, op, threshold),),
                        family="single",
                    )
                )
    return rules


def generate_pair_rules(rows: Sequence[Mapping[str, Any]], *, max_pairs: int = 200) -> List[Rule]:
    singles = score_rules(generate_single_feature_rules(rows), rows)
    promising = [
        item
        for item in singles
        if item["captured_short_only_wins"] >= 1 and item["anchor_losses"] <= 3
    ][:24]
    rules: List[Rule] = []
    for left_index, left in enumerate(promising):
        for right in promising[left_index + 1 :]:
            left_rule = left["rule_obj"]
            right_rule = right["rule_obj"]
            clauses = tuple(dict.fromkeys([*left_rule.clauses, *right_rule.clauses]))
            if len(clauses) != 2:
                continue
            name = f"{left_rule.name}_AND_{right_rule.name}"
            rules.append(Rule(name=name, clauses=clauses, family="pair"))
            if len(rules) >= max_pairs:
                return rules
    return rules


def score_rule(rule: Rule, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    matched = [row for row in rows if rule.matches(row)]
    captured_short_only = [row for row in matched if row["short_only_win"]]
    anchor_losses = [row for row in matched if row["anchor_only_win"]]
    current_pass_losses = [
        row for row in matched if row["baseline_passed"] and row["anchor_passed"] and not row["short_passed"]
    ]
    short_risk = [row for row in matched if row["oracle_bucket"] == "<=8"]
    selected_pass = sum(1 for row in matched if row["short_passed"]) + sum(
        1 for row in rows if row not in matched and row["anchor_passed"]
    )
    anchor_pass = sum(1 for row in rows if row["anchor_passed"])
    utility = (
        100 * len(captured_short_only)
        - 1000 * len(anchor_losses)
        - 1000 * len(current_pass_losses)
        - 100 * len(short_risk)
        - 2 * len(matched)
        - 5 * len(rule.clauses)
    )
    decision = "reject"
    if (
        len(captured_short_only) >= 1
        and len(anchor_losses) == 0
        and len(current_pass_losses) == 0
        and len(short_risk) == 0
        and len(matched) <= 5
        and len(rule.clauses) <= 2
    ):
        decision = "policy_candidate"
    elif len(captured_short_only) >= 1 and len(anchor_losses) <= 1:
        decision = "diagnostic_only"
    return {
        "rule": rule.name,
        "family": rule.family,
        "clauses": " AND ".join(f"{feature} {op} {threshold:.6g}" for feature, op, threshold in rule.clauses),
        "override_count": len(matched),
        "captured_short_only_wins": len(captured_short_only),
        "anchor_losses": len(anchor_losses),
        "current_pass_losses": len(current_pass_losses),
        "short_risk_count": len(short_risk),
        "anchor_triggered_pass": anchor_pass,
        "rule_triggered_pass": selected_pass,
        "triggered_pass_delta_vs_anchor": selected_pass - anchor_pass,
        "captured_task_ids": ";".join(row["task_id"] for row in captured_short_only),
        "anchor_loss_task_ids": ";".join(row["task_id"] for row in anchor_losses),
        "override_task_ids": ";".join(row["task_id"] for row in matched),
        "utility": utility,
        "decision": decision,
        "rule_obj": rule,
    }


def score_rules(rules: Sequence[Rule], rows: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    scored = [score_rule(rule, rows) for rule in rules]
    scored.sort(
        key=lambda item: (
            item["decision"] == "policy_candidate",
            item["triggered_pass_delta_vs_anchor"],
            item["captured_short_only_wins"],
            -item["anchor_losses"],
            -item["short_risk_count"],
            -item["override_count"],
            item["utility"],
        ),
        reverse=True,
    )
    return scored


def strip_rule_objects(rows: Sequence[JsonDict]) -> List[JsonDict]:
    return [{key: value for key, value in row.items() if key != "rule_obj"} for row in rows]


def final_decision(scored_rules: Sequence[Mapping[str, Any]]) -> str:
    if any(row.get("decision") == "policy_candidate" for row in scored_rules):
        return "policy_candidate"
    if any(row.get("decision") == "diagnostic_only" for row in scored_rules):
        return "diagnostic_only"
    return "reject_selector_only"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def report_markdown(summary: Mapping[str, Any], rules: Sequence[Mapping[str, Any]], triggered: Sequence[Mapping[str, Any]]) -> str:
    top = list(rules[:10])
    short_only = [row for row in triggered if row["short_only_win"]]
    anchor_only = [row for row in triggered if row["anchor_only_win"]]
    lines = [
        "# Route2 V6 Short-Override Audit",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        "## Reproduced Accounting",
        "",
        f"- rows: `{summary['num_samples']}`",
        f"- V5.1 pass count: `{summary['v5_pass_count']}`",
        f"- triggered rows: `{summary['triggered_count']}`",
        f"- selected triggered pass: `{summary['selected_triggered_pass_count']}`",
        f"- candidate upper bound: `{summary['candidate_upper_bound_count']}`",
        f"- pairwise vs Route2 precision len32: `{summary['pairwise_vs_reference']}`",
        "",
        "## Upper-Bound-Only `len24_s64` Rows",
        "",
    ]
    if short_only:
        lines.extend(["| task_id | oracle bucket | oracle length |", "|---|---:|---:|"])
        for row in short_only:
            lines.append(f"| `{row['task_id']}` | `{row['oracle_bucket']}` | `{row['oracle_length']}` |")
    else:
        lines.append("None.")
    lines.extend(["", "## Anchor-Risk Rows", ""])
    if anchor_only:
        lines.extend(["| task_id | oracle bucket | oracle length |", "|---|---:|---:|"])
        for row in anchor_only:
            lines.append(f"| `{row['task_id']}` | `{row['oracle_bucket']}` | `{row['oracle_length']}` |")
    else:
        lines.append("None.")
    lines.extend(["", "## Top Rules", ""])
    if top:
        lines.extend(
            [
                "| Rule | Decision | Overrides | Captured | Anchor losses | Short risk | Delta |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in top:
            lines.append(
                f"| `{row['rule']}` | `{row['decision']}` | `{row['override_count']}` | "
                f"`{row['captured_short_only_wins']}` | `{row['anchor_losses']}` | "
                f"`{row['short_risk_count']}` | `{row['triggered_pass_delta_vs_anchor']}` |"
            )
    else:
        lines.append("No valid rules generated.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a CPU-only offline selector audit. It does not prove a new pass-rate claim.",
            "A rule is useful only if it captures `len24_s64`-only wins while preserving the `len32_s64` anchor.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, summary: JsonDict, triggered: Sequence[Mapping[str, Any]], rules: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_triggered = [
        {key: value for key, value in row.items() if key != "features"} | {
            f"feature_{key}": value for key, value in (row.get("features") or {}).items()
        }
        for row in triggered
    ]
    stripped_rules = strip_rule_objects(rules)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(report_markdown(summary, stripped_rules, triggered), encoding="utf-8")
    write_csv(output_dir / "triggered_candidate_table.csv", serializable_triggered)
    write_csv(output_dir / "rule_candidates.csv", stripped_rules)


def run_audit(
    *,
    v5_results: Path,
    output_dir: Path,
    baseline_results: Optional[Path] = None,
    reference_results: Optional[Path] = None,
) -> JsonDict:
    v5_rows = load_jsonl(str(v5_results))
    baseline_rows = load_jsonl(str(baseline_results)) if baseline_results else None
    reference_rows = load_jsonl(str(reference_results)) if reference_results else None
    triggered, base_summary = build_triggered_table(
        v5_rows,
        baseline_rows=baseline_rows,
        reference_rows=reference_rows,
    )
    rules = generate_single_feature_rules(triggered)
    rules.extend(generate_pair_rules(triggered))
    scored = score_rules(rules, triggered)
    decision = final_decision(scored)
    summary: JsonDict = {
        **base_summary,
        "decision": decision,
        "short_only_win_count": sum(1 for row in triggered if row["short_only_win"]),
        "anchor_only_win_count": sum(1 for row in triggered if row["anchor_only_win"]),
        "feature_count": len(feature_columns(triggered)),
        "rule_count": len(scored),
        "top_rules": strip_rule_objects(scored[:10]),
    }
    write_outputs(output_dir, summary, triggered, scored)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only Route2 V6 short-candidate override audit.")
    parser.add_argument("--v5-results", required=True, type=Path)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_audit(
        v5_results=args.v5_results,
        baseline_results=args.baseline_results,
        reference_results=args.reference_results,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReport: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
