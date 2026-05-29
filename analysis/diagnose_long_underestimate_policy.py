#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Rule:
    max_selected_len: int
    min_best_long_len: int
    min_gap: int
    min_long_ratio: float
    min_raw_long_ratio: float
    min_support_count: int
    allowed_sources: Tuple[str, ...]

    def label(self) -> str:
        sources = "+".join(self.allowed_sources)
        return (
            f"sel<={self.max_selected_len}|best>={self.min_best_long_len}|gap>={self.min_gap}|"
            f"ratio>={self.min_long_ratio:g}|raw>={self.min_raw_long_ratio:g}|"
            f"supp>={self.min_support_count}|src={sources}"
        )


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "task_id" not in row:
                raise ValueError(f"Missing task_id in {path} line {line_no}")
            rows.append(row)
    return rows


def metric(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    metrics = row.get("metrics") or {}
    return metrics.get(key, default) if isinstance(metrics, dict) else default


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def row_triggers(row: Dict[str, Any], rule: Rule) -> bool:
    selected = optional_int(metric(row, "selected_mask_length"))
    best_long = optional_int(metric(row, "best_long_len"))
    long_ratio = optional_float(metric(row, "long_ratio"))
    raw_long_ratio = optional_float(metric(row, "raw_long_ratio"))
    support = optional_int(metric(row, "support_count", 0))
    source = str(metric(row, "final_source", ""))

    if selected is None or best_long is None or long_ratio is None or raw_long_ratio is None:
        return False
    if source not in set(rule.allowed_sources):
        return False
    if selected > rule.max_selected_len:
        return False
    if best_long < rule.min_best_long_len:
        return False
    if best_long - selected < rule.min_gap:
        return False
    if long_ratio < rule.min_long_ratio:
        return False
    if raw_long_ratio < rule.min_raw_long_ratio:
        return False
    if support is None or support < rule.min_support_count:
        return False
    return True


def is_true_long(row: Dict[str, Any]) -> bool:
    oracle = optional_int(metric(row, "oracle_mask_length"))
    return oracle is not None and oracle >= 17


def is_short(row: Dict[str, Any]) -> bool:
    oracle = optional_int(metric(row, "oracle_mask_length"))
    return oracle is not None and oracle <= 8


def is_medium_or_short(row: Dict[str, Any]) -> bool:
    oracle = optional_int(metric(row, "oracle_mask_length"))
    return oracle is not None and oracle <= 12


def passed(row: Dict[str, Any]) -> bool:
    return bool(metric(row, "passed", False))


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def evaluate_rule(rows: Sequence[Dict[str, Any]], rule: Rule) -> Dict[str, Any]:
    triggered = [row for row in rows if row_triggers(row, rule)]
    failed_long_rows = [row for row in rows if is_true_long(row) and not passed(row)]
    true_long_count = sum(1 for row in triggered if is_true_long(row))
    short_risk_count = sum(1 for row in triggered if is_short(row))
    medium_risk_count = sum(1 for row in triggered if is_medium_or_short(row))
    failed_long_trigger_count = sum(1 for row in triggered if is_true_long(row) and not passed(row))
    current_pass_risk_count = sum(1 for row in triggered if passed(row))
    trigger_count = len(triggered)
    precision = rate(true_long_count, trigger_count)
    failed_recall = rate(failed_long_trigger_count, len(failed_long_rows))
    short_risk = rate(short_risk_count, trigger_count)
    pass_risk = rate(current_pass_risk_count, trigger_count)
    score = (
        (precision or 0.0)
        * (failed_recall or 0.0)
        * (1.0 - (short_risk or 0.0))
        * (1.0 - (pass_risk or 0.0))
    )

    payload: Dict[str, Any] = {
        "rule": rule.label(),
        **asdict(rule),
        "allowed_sources": ",".join(rule.allowed_sources),
        "trigger_count": trigger_count,
        "true_long_count": true_long_count,
        "short_risk_count": short_risk_count,
        "medium_risk_count": medium_risk_count,
        "failed_long_total": len(failed_long_rows),
        "failed_long_trigger_count": failed_long_trigger_count,
        "current_pass_risk_count": current_pass_risk_count,
        "true_long_precision": precision,
        "short_risk_rate": short_risk,
        "medium_risk_rate": rate(medium_risk_count, trigger_count),
        "failed_long_recall": failed_recall,
        "current_pass_risk_rate": pass_risk,
        "score": score,
    }
    return payload


def source_sets() -> Iterable[Tuple[str, ...]]:
    yield ("base",)
    yield ("base", "strong_correction", "weak_correction")


def sweep_rules(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for max_selected_len in (3, 5, 8, 10, 12, 16):
        for min_best_long_len in (13, 16, 20, 24):
            for min_gap in (4, 6, 8, 10, 12):
                for min_long_ratio in (0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
                    for min_raw_long_ratio in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
                        for min_support_count in (0, 1, 2):
                            for allowed_sources in source_sets():
                                rule = Rule(
                                    max_selected_len=max_selected_len,
                                    min_best_long_len=min_best_long_len,
                                    min_gap=min_gap,
                                    min_long_ratio=min_long_ratio,
                                    min_raw_long_ratio=min_raw_long_ratio,
                                    min_support_count=min_support_count,
                                    allowed_sources=allowed_sources,
                                )
                                result = evaluate_rule(rows, rule)
                                if result["trigger_count"] > 0:
                                    results.append(result)
    results.sort(
        key=lambda item: (
            item["score"],
            item["failed_long_trigger_count"],
            item["true_long_precision"] or 0.0,
            -(item["short_risk_rate"] or 0.0),
        ),
        reverse=True,
    )
    return results


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]], *, top_k: int = 30) -> None:
    lines = [
        "# Long Underestimate Detector Sweep",
        "",
        "| Rule | Triggers | True Long Precision | Failed Long Recall | Short Risk | Current Pass Risk |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:top_k]:
        lines.append(
            f"| `{row['rule']}` | {row['trigger_count']} | {pct(row['true_long_precision'])} | "
            f"{pct(row['failed_long_recall'])} | {pct(row['short_risk_rate'])} | "
            f"{pct(row['current_pass_risk_rate'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(rows: Sequence[Dict[str, Any]], output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "sweep.csv", rows)
    write_markdown(out_dir / "sweep.md", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep offline long-underestimation detector gates.")
    parser.add_argument("--results", required=True, help="Path to results.jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    sweep = sweep_rules(rows)
    write_outputs(sweep, args.output_dir)
    print(f"rows={len(rows)} rules={len(sweep)} output_dir={args.output_dir}")
    if sweep:
        best = sweep[0]
        print(
            "best "
            f"trigger_count={best['trigger_count']} "
            f"precision={pct(best['true_long_precision'])} "
            f"failed_long_recall={pct(best['failed_long_recall'])} "
            f"short_risk={pct(best['short_risk_rate'])} "
            f"pass_risk={pct(best['current_pass_risk_rate'])} "
            f"rule={best['rule']}"
        )


if __name__ == "__main__":
    main()
