#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class WideningPolicy:
    base_threshold: float
    slope: float
    min_threshold: float
    min_base_length: int
    max_expansion: float
    require_raw_confirm: bool
    raw_margin: float = 0.05
    min_raw_threshold: float = 0.75

    def label(self) -> str:
        raw = "raw" if self.require_raw_confirm else "noraw"
        return (
            f"base{self.base_threshold:g}_slope{self.slope:g}_min{self.min_threshold:g}_"
            f"minbase{self.min_base_length}_maxx{self.max_expansion:g}_{raw}"
        ).replace(".", "p")


def load_jsonl(path: str | Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
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


def metric(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    metrics = row.get("metrics") or {}
    return metrics.get(key, default) if isinstance(metrics, Mapping) else default


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def passed(row: Mapping[str, Any]) -> bool:
    return bool(metric(row, "passed", False))


def oracle_bucket(length: Any) -> str:
    value = optional_int(length)
    if value is None:
        return "unknown"
    if value <= 8:
        return "<=8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    if value <= 24:
        return "17-24"
    return "25+"


def candidate_score(candidate: Mapping[str, Any]) -> float:
    return float(candidate.get("score", candidate.get("adjusted_score", candidate.get("raw_score", 0.0))))


def candidate_raw_score(candidate: Mapping[str, Any]) -> float:
    return float(candidate.get("raw_score", candidate.get("mean_top1_prob", candidate_score(candidate))))


def candidate_length(candidate: Mapping[str, Any]) -> int:
    return int(candidate["mask_length"])


def candidate_scores(row: Mapping[str, Any]) -> List[JsonDict]:
    lcal = row.get("lcal_v3") or {}
    if isinstance(lcal, Mapping) and isinstance(lcal.get("base_candidate_scores"), list):
        return [dict(item) for item in lcal["base_candidate_scores"]]

    length_probe = row.get("length_probe") or {}
    if isinstance(length_probe, Mapping):
        if isinstance(length_probe.get("base_candidate_scores"), list):
            return [dict(item) for item in length_probe["base_candidate_scores"]]
        if isinstance(length_probe.get("candidate_scores"), list):
            return [dict(item) for item in length_probe["candidate_scores"]]

    return []


def best_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("Cannot select from empty candidates")
    return max(candidates, key=lambda item: (candidate_score(item), -candidate_length(item)))


def proportional_threshold(policy: WideningPolicy, *, base_length: int, candidate_length_value: int) -> float:
    if candidate_length_value <= base_length:
        return float(policy.base_threshold)
    ratio = float(candidate_length_value) / float(max(base_length, 1))
    return max(float(policy.min_threshold), float(policy.base_threshold) - float(policy.slope) * math.log(ratio))


def select_proportional_length(candidates: Sequence[Mapping[str, Any]], policy: WideningPolicy) -> Optional[int]:
    if not candidates:
        return None
    best = best_candidate(candidates)
    base_len = candidate_length(best)
    if base_len < int(policy.min_base_length):
        return base_len

    best_score = candidate_score(best)
    best_raw = max(candidate_raw_score(candidate) for candidate in candidates)
    max_len = int(math.ceil(float(base_len) * float(policy.max_expansion)))
    selected = base_len

    for candidate in candidates:
        cand_len = candidate_length(candidate)
        if cand_len < base_len or cand_len > max_len:
            continue
        threshold = proportional_threshold(policy, base_length=base_len, candidate_length_value=cand_len)
        if candidate_score(candidate) < threshold * best_score:
            continue
        if policy.require_raw_confirm:
            raw_threshold = max(float(policy.min_raw_threshold), threshold - float(policy.raw_margin))
            if candidate_raw_score(candidate) < raw_threshold * best_raw:
                continue
        selected = max(selected, cand_len)

    return selected


def policy_grid() -> List[WideningPolicy]:
    policies: List[WideningPolicy] = []
    for base_threshold in (0.95, 0.97, 0.985):
        for slope in (0.02, 0.04, 0.06, 0.08):
            for min_threshold in (0.85, 0.88, 0.90, 0.93):
                for min_base_length in (8, 9, 10, 12):
                    for max_expansion in (1.5, 2.0, 3.0):
                        for require_raw_confirm in (False, True):
                            policies.append(
                                WideningPolicy(
                                    base_threshold=base_threshold,
                                    slope=slope,
                                    min_threshold=min_threshold,
                                    min_base_length=min_base_length,
                                    max_expansion=max_expansion,
                                    require_raw_confirm=require_raw_confirm,
                                )
                            )
    return policies


def row_decision(row: Mapping[str, Any], policy: WideningPolicy) -> JsonDict:
    candidates = candidate_scores(row)
    current_len = optional_int(metric(row, "selected_mask_length", metric(row, "mask_length")))
    oracle_len = optional_int(metric(row, "oracle_mask_length"))
    prop_len = select_proportional_length(candidates, policy)

    if current_len is None or prop_len is None:
        new_len = current_len
        promoted = False
    else:
        new_len = max(int(current_len), int(prop_len))
        promoted = int(new_len) > int(current_len)

    current_abs_error = None
    new_abs_error = None
    delta_abs_error = None
    if current_len is not None and new_len is not None and oracle_len is not None:
        current_abs_error = abs(int(current_len) - int(oracle_len))
        new_abs_error = abs(int(new_len) - int(oracle_len))
        delta_abs_error = int(new_abs_error) - int(current_abs_error)

    return {
        "task_id": row.get("task_id"),
        "policy": policy.label(),
        "oracle_length": oracle_len,
        "oracle_bucket": oracle_bucket(oracle_len),
        "passed": passed(row),
        "current_selected_length": current_len,
        "proportional_selected_length": prop_len,
        "new_selected_length": new_len,
        "promoted": bool(promoted),
        "current_abs_error": current_abs_error,
        "new_abs_error": new_abs_error,
        "delta_abs_error": delta_abs_error,
        "improved_length_error": bool(delta_abs_error is not None and delta_abs_error < 0),
        "worsened_length_error": bool(delta_abs_error is not None and delta_abs_error > 0),
        "current_under_selected": bool(
            current_len is not None and oracle_len is not None and int(current_len) < int(oracle_len)
        ),
        "new_under_selected": bool(new_len is not None and oracle_len is not None and int(new_len) < int(oracle_len)),
        "current_over_selected": bool(
            current_len is not None and oracle_len is not None and int(current_len) > int(oracle_len)
        ),
        "new_over_selected": bool(new_len is not None and oracle_len is not None and int(new_len) > int(oracle_len)),
        "max_probe_length": max((candidate_length(candidate) for candidate in candidates), default=None),
        "candidate_count": len(candidates),
    }


def rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def summarize_decisions(policy: WideningPolicy, decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    valid = [row for row in decisions if row.get("new_selected_length") is not None and row.get("oracle_length") is not None]
    promoted = [row for row in valid if bool(row.get("promoted"))]
    improved = [row for row in valid if bool(row.get("improved_length_error"))]
    worsened = [row for row in valid if bool(row.get("worsened_length_error"))]
    true_long = [row for row in valid if str(row.get("oracle_bucket")) in {"17-24", "25+"}]
    failed_true_long = [row for row in true_long if not bool(row.get("passed"))]
    current_abs = sum(int(row["current_abs_error"]) for row in valid) / len(valid) if valid else None
    new_abs = sum(int(row["new_abs_error"]) for row in valid) / len(valid) if valid else None

    true_long_improved = [row for row in true_long if bool(row.get("improved_length_error"))]
    failed_long_improved = [row for row in failed_true_long if bool(row.get("improved_length_error"))]
    short_promoted = [row for row in promoted if row.get("oracle_bucket") == "<=8"]
    current_pass_promoted = [row for row in promoted if bool(row.get("passed"))]

    score = (
        4 * len(failed_long_improved)
        + 2 * len(true_long_improved)
        + len(improved)
        - 2 * len(worsened)
        - 3 * len(short_promoted)
        - len(current_pass_promoted)
    )

    payload: JsonDict = {
        **asdict(policy),
        "policy": policy.label(),
        "rows": len(valid),
        "promoted_count": len(promoted),
        "improved_count": len(improved),
        "worsened_count": len(worsened),
        "current_avg_abs_error": current_abs,
        "new_avg_abs_error": new_abs,
        "avg_abs_error_delta": None if current_abs is None or new_abs is None else new_abs - current_abs,
        "current_under_count": sum(bool(row.get("current_under_selected")) for row in valid),
        "new_under_count": sum(bool(row.get("new_under_selected")) for row in valid),
        "current_over_count": sum(bool(row.get("current_over_selected")) for row in valid),
        "new_over_count": sum(bool(row.get("new_over_selected")) for row in valid),
        "true_long_count": len(true_long),
        "true_long_improved_count": len(true_long_improved),
        "failed_true_long_count": len(failed_true_long),
        "failed_true_long_improved_count": len(failed_long_improved),
        "short_promoted_count": len(short_promoted),
        "current_pass_promoted_count": len(current_pass_promoted),
        "promoted_rate": rate(len(promoted), len(valid)),
        "improved_rate": rate(len(improved), len(valid)),
        "worsened_rate": rate(len(worsened), len(valid)),
        "score": score,
    }

    by_bucket: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in valid:
        bucket = str(row.get("oracle_bucket"))
        by_bucket[bucket]["n"] += 1
        by_bucket[bucket]["promoted"] += int(bool(row.get("promoted")))
        by_bucket[bucket]["improved"] += int(bool(row.get("improved_length_error")))
        by_bucket[bucket]["worsened"] += int(bool(row.get("worsened_length_error")))
    for bucket in ("<=8", "9-12", "13-16", "17-24", "25+"):
        counts = by_bucket[bucket]
        payload[f"bucket_{bucket}_n"] = counts["n"]
        payload[f"bucket_{bucket}_promoted"] = counts["promoted"]
        payload[f"bucket_{bucket}_improved"] = counts["improved"]
        payload[f"bucket_{bucket}_worsened"] = counts["worsened"]

    return payload


def sweep(rows: Sequence[Mapping[str, Any]]) -> tuple[List[JsonDict], List[JsonDict]]:
    summaries: List[JsonDict] = []
    best_decisions: List[JsonDict] = []

    best_summary: Optional[JsonDict] = None
    best_policy: Optional[WideningPolicy] = None
    for policy in policy_grid():
        decisions = [row_decision(row, policy) for row in rows]
        summary = summarize_decisions(policy, decisions)
        summaries.append(summary)
        if best_summary is None or (
            summary["score"],
            -summary["short_promoted_count"],
            -summary["worsened_count"],
            summary["failed_true_long_improved_count"],
        ) > (
            best_summary["score"],
            -best_summary["short_promoted_count"],
            -best_summary["worsened_count"],
            best_summary["failed_true_long_improved_count"],
        ):
            best_summary = summary
            best_policy = policy

    summaries.sort(
        key=lambda item: (
            item["score"],
            item["failed_true_long_improved_count"],
            item["true_long_improved_count"],
            -item["short_promoted_count"],
            -item["worsened_count"],
        ),
        reverse=True,
    )
    if best_policy is not None:
        best_decisions = [row_decision(row, best_policy) for row in rows]
    return summaries, best_decisions


def decision_label(rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "reject_existing_grid"
    max_probe_lengths = [optional_int(row.get("max_probe_length")) for row in rows]
    max_available = max((value for value in max_probe_lengths if value is not None), default=0)
    best = summaries[0] if summaries else {}
    best_failed_long = int(best.get("failed_true_long_improved_count") or 0)
    best_worsened = int(best.get("worsened_count") or 0)
    best_short = int(best.get("short_promoted_count") or 0)
    best_delta = optional_float(best.get("avg_abs_error_delta"))

    if best_failed_long >= 2 and best_short <= 2 and best_worsened <= 10 and (best_delta is not None and best_delta < 0):
        return "policy_candidate"
    if max_available <= 24:
        return "needs_expanded_grid_smoke"
    return "reject_existing_grid"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def write_report(path: Path, summary: Mapping[str, Any], policies: Sequence[Mapping[str, Any]]) -> None:
    best = policies[0] if policies else {}
    lines = [
        "# Proportional Length Widening Audit",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        "## Overview",
        "",
        f"- rows: `{summary['rows']}`",
        f"- max available probe length: `{summary['max_probe_length']}`",
        f"- baseline average absolute length error: `{summary['current_avg_abs_error']:.4f}`",
        "",
        "## Best Policy",
        "",
        "| Policy | Promoted | Improved | Worsened | Failed True-Long Improved | Short Promoted | Avg Abs Error Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if best:
        lines.append(
            f"| `{best['policy']}` | `{best['promoted_count']}` | `{best['improved_count']}` | "
            f"`{best['worsened_count']}` | `{best['failed_true_long_improved_count']}` | "
            f"`{best['short_promoted_count']}` | `{best['avg_abs_error_delta']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Top Policies",
            "",
            "| Policy | Score | Promoted | Improved | Worsened | 17-24 improved | 25+ improved | Short promoted |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in policies[:20]:
        lines.append(
            f"| `{row['policy']}` | `{row['score']}` | `{row['promoted_count']}` | "
            f"`{row['improved_count']}` | `{row['worsened_count']}` | "
            f"`{row['bucket_17-24_improved']}` | `{row['bucket_25+_improved']}` | "
            f"`{row['short_promoted_count']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit is CPU-only and replays stored probe scores. It does not prove a pass-rate gain.",
            "Oracle length and pass/fail are used only for offline accounting, not as policy inputs.",
        ]
    )
    if summary["decision"] == "needs_expanded_grid_smoke":
        lines.append(
            "The current stored probe grid is probably too narrow for the user's intended long-length widening idea; "
            "a small expanded-grid GPU smoke would be the next meaningful test."
        )
    elif summary["decision"] == "policy_candidate":
        lines.append("The best rule is conservative enough to justify a separate GPU smoke action brief.")
    else:
        lines.append("The current evidence does not justify a GPU run for this proportional widening variant.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def output_summary(rows: Sequence[Mapping[str, Any]], policies: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    max_probe_length = max((optional_int(row.get("max_probe_length")) or 0 for row in decisions), default=0)
    current_errors = [
        int(row["current_abs_error"])
        for row in decisions
        if row.get("current_abs_error") is not None
    ]
    summary: JsonDict = {
        "rows": len(rows),
        "decision": decision_label(decisions, policies),
        "max_probe_length": max_probe_length,
        "current_avg_abs_error": sum(current_errors) / len(current_errors) if current_errors else None,
        "best_policy": policies[0] if policies else None,
    }
    summary["probe_length_histogram"] = dict(
        sorted(Counter(str(row.get("max_probe_length")) for row in decisions).items())
    )
    return summary


def write_outputs(output_dir: str | Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policies, decisions = sweep(rows)
    summary = output_summary(rows, policies, decisions)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "policy_sweep.csv", policies)
    write_csv(out_dir / "row_decisions.csv", decisions)
    write_report(out_dir / "report.md", summary, policies)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only proportional length widening audit.")
    parser.add_argument("--results", required=True, help="Path to results.jsonl with stored probe candidate scores.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    summary = write_outputs(args.output_dir, rows)
    best = summary.get("best_policy") or {}
    print(f"rows={summary['rows']} decision={summary['decision']} output_dir={args.output_dir}")
    if best:
        print(
            "best "
            f"policy={best['policy']} "
            f"promoted={best['promoted_count']} "
            f"improved={best['improved_count']} "
            f"worsened={best['worsened_count']} "
            f"failed_true_long_improved={best['failed_true_long_improved_count']} "
            f"short_promoted={best['short_promoted_count']} "
            f"avg_abs_delta={best['avg_abs_error_delta']:.4f}"
        )


if __name__ == "__main__":
    main()
