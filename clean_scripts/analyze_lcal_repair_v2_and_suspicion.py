#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_map(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["task_id"]): row for row in rows}


def metric(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    return row.get("metrics", {}).get(key, default)


def length_probe_lcal(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("length_probe", {}).get("lcal_v3", {}) or row.get("lcal_v3", {}) or {}


def int_metric(row: Dict[str, Any], key: str, default: Optional[int] = None) -> Optional[int]:
    value = metric(row, key, default)
    if value is None:
        return default
    return int(value)


def passed(row: Dict[str, Any]) -> bool:
    return bool(metric(row, "passed", False))


def bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    value = int(length)
    if value <= 8:
        return "<=8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    if value <= 24:
        return "17-24"
    return "25+"


def rate(values: Iterable[bool]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(1 for value in vals if value) / len(vals)


def avg(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def hist(values: Iterable[Any]) -> Dict[str, int]:
    counter = Counter(str(value) for value in values)
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def official_len(row: Dict[str, Any]) -> Optional[int]:
    value = metric(row, "official_selected_length")
    if value is not None:
        return int(value)
    value = metric(row, "selected_mask_length")
    if value is not None:
        return int(value)
    value = row.get("official_cal", {}).get("selected_mask_length")
    return None if value is None else int(value)


def s3_len(row: Dict[str, Any]) -> int:
    value = metric(row, "s3_selected_length")
    if value is not None:
        return int(value)
    value = metric(row, "selected_mask_length")
    if value is not None:
        return int(value)
    raise KeyError(f"missing S3 selected length for {row.get('task_id')}")


def oracle_len(row: Dict[str, Any]) -> Optional[int]:
    value = metric(row, "oracle_mask_length")
    return None if value is None else int(value)


def lcal_signal(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    lcal = length_probe_lcal(row)
    value = metric(row, key)
    if value is not None:
        return value
    return lcal.get(key, default)


def repair_trigger(
    s3_selected: int,
    off_selected: Optional[int],
    s3_max: int,
    off_min: int,
    off_max: int,
    delta_min: int,
    delta_max: int,
) -> bool:
    if off_selected is None:
        return False
    delta = int(off_selected) - int(s3_selected)
    return (
        int(s3_selected) <= int(s3_max)
        and int(off_min) <= int(off_selected) <= int(off_max)
        and int(delta_min) <= int(delta) <= int(delta_max)
    )


def summarize_repair_policy(
    *,
    policy_name: str,
    task_ids: List[str],
    s3_rows: Dict[str, Dict[str, Any]],
    official_rows: Dict[str, Dict[str, Any]],
    bounded_rows: Dict[str, Dict[str, Any]],
    s3_max: int,
    off_min: int,
    off_max: int,
    delta_min: int,
    delta_max: int,
) -> Dict[str, Any]:
    triggered: List[str] = []
    mixed_passes: Dict[str, bool] = {}
    wins: List[str] = []
    losses: List[str] = []

    for task_id in task_ids:
        s3_row = s3_rows[task_id]
        off_row = official_rows[task_id]
        selected_s3 = s3_len(s3_row)
        selected_off = official_len(off_row)
        should_repair = repair_trigger(
            selected_s3,
            selected_off,
            s3_max=s3_max,
            off_min=off_min,
            off_max=off_max,
            delta_min=delta_min,
            delta_max=delta_max,
        )
        if should_repair:
            triggered.append(task_id)
        source_row = bounded_rows[task_id] if should_repair else s3_row
        mixed_pass = passed(source_row)
        mixed_passes[task_id] = mixed_pass
        if mixed_pass and not passed(s3_row):
            wins.append(task_id)
        elif passed(s3_row) and not mixed_pass:
            losses.append(task_id)

    triggered_rows = [s3_rows[task_id] for task_id in triggered]
    win_rows = [s3_rows[task_id] for task_id in wins]
    loss_rows = [s3_rows[task_id] for task_id in losses]
    pass_count = sum(1 for value in mixed_passes.values() if value)
    return {
        "policy": policy_name,
        "s3_max": s3_max,
        "official_min": off_min,
        "official_max": off_max,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "num_samples": len(task_ids),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(task_ids) if task_ids else None,
        "trigger_count": len(triggered),
        "trigger_rate": len(triggered) / len(task_ids) if task_ids else None,
        "wins_vs_s3": len(wins),
        "losses_vs_s3": len(losses),
        "net_vs_s3": len(wins) - len(losses),
        "trigger_win_rate": len(wins) / len(triggered) if triggered else None,
        "trigger_loss_rate": len(losses) / len(triggered) if triggered else None,
        "trigger_oracle_bucket_histogram": hist(bucket(oracle_len(row)) for row in triggered_rows),
        "win_oracle_bucket_histogram": hist(bucket(oracle_len(row)) for row in win_rows),
        "loss_oracle_bucket_histogram": hist(bucket(oracle_len(row)) for row in loss_rows),
        "short_wins": sum(1 for row in win_rows if (oracle_len(row) is not None and oracle_len(row) <= 8)),
        "short_losses": sum(1 for row in loss_rows if (oracle_len(row) is not None and oracle_len(row) <= 8)),
        "nonshort_trigger_precision": rate(
            oracle_len(row) is not None and oracle_len(row) >= 9 for row in triggered_rows
        ),
        "true_long_trigger_precision": rate(
            oracle_len(row) is not None and oracle_len(row) >= 17 for row in triggered_rows
        ),
        "triggered_task_ids": triggered,
        "wins_task_ids": wins,
        "losses_task_ids": losses,
    }


def suspicion_candidate(
    *,
    s3_row: Dict[str, Any],
    off_row: Dict[str, Any],
    s3_max: int,
    off_min: int,
    off_max: int,
    ratio_min: Optional[float],
    raw_ratio_min: Optional[float],
    support_min: int,
) -> bool:
    selected_s3 = s3_len(s3_row)
    selected_off = official_len(off_row)
    if selected_off is None:
        return False
    if selected_s3 > s3_max or selected_off < off_min or selected_off > off_max:
        return False
    ratio = lcal_signal(s3_row, "long_ratio")
    raw_ratio = lcal_signal(s3_row, "raw_long_ratio")
    support = int(lcal_signal(s3_row, "support_count", 0) or 0)
    if ratio_min is not None and (ratio is None or float(ratio) < float(ratio_min)):
        return False
    if raw_ratio_min is not None and (raw_ratio is None or float(raw_ratio) < float(raw_ratio_min)):
        return False
    if support < support_min:
        return False
    return True


def summarize_suspicion_policy(
    *,
    policy_name: str,
    task_ids: List[str],
    s3_rows: Dict[str, Dict[str, Any]],
    official_rows: Dict[str, Dict[str, Any]],
    s3_max: int,
    off_min: int,
    off_max: int,
    ratio_min: Optional[float],
    raw_ratio_min: Optional[float],
    support_min: int,
) -> Dict[str, Any]:
    suspects = [
        task_id
        for task_id in task_ids
        if suspicion_candidate(
            s3_row=s3_rows[task_id],
            off_row=official_rows[task_id],
            s3_max=s3_max,
            off_min=off_min,
            off_max=off_max,
            ratio_min=ratio_min,
            raw_ratio_min=raw_ratio_min,
            support_min=support_min,
        )
    ]
    rows = [s3_rows[task_id] for task_id in suspects]
    official_selected = [official_len(official_rows[task_id]) for task_id in suspects]
    s3_selected = [s3_len(s3_rows[task_id]) for task_id in suspects]
    s3_failed = [task_id for task_id in suspects if not passed(s3_rows[task_id])]
    official_would_win = [
        task_id
        for task_id in suspects
        if passed(official_rows[task_id]) and not passed(s3_rows[task_id])
    ]
    official_would_lose = [
        task_id
        for task_id in suspects
        if passed(s3_rows[task_id]) and not passed(official_rows[task_id])
    ]
    return {
        "policy": policy_name,
        "s3_max": s3_max,
        "official_min": off_min,
        "official_max": off_max,
        "ratio_min": ratio_min,
        "raw_ratio_min": raw_ratio_min,
        "support_min": support_min,
        "suspect_count": len(suspects),
        "suspect_rate": len(suspects) / len(task_ids) if task_ids else None,
        "true_long_precision": rate(oracle_len(row) is not None and oracle_len(row) >= 17 for row in rows),
        "nonshort_precision": rate(oracle_len(row) is not None and oracle_len(row) >= 9 for row in rows),
        "short_precision_bad": rate(oracle_len(row) is not None and oracle_len(row) <= 8 for row in rows),
        "s3_failed_count": len(s3_failed),
        "s3_failed_rate": len(s3_failed) / len(suspects) if suspects else None,
        "official_vs_s3_wins": len(official_would_win),
        "official_vs_s3_losses": len(official_would_lose),
        "official_vs_s3_net": len(official_would_win) - len(official_would_lose),
        "oracle_bucket_histogram": hist(bucket(oracle_len(row)) for row in rows),
        "official_len_histogram": hist(official_selected),
        "s3_len_histogram": hist(s3_selected),
        "avg_official_len": avg(official_selected),
        "avg_s3_len": avg(s3_selected),
        "suspect_task_ids": suspects,
        "official_wins_task_ids": official_would_win,
        "official_losses_task_ids": official_would_lose,
    }


def summarize_union_policy(
    *,
    policy_name: str,
    task_ids: List[str],
    s3_rows: Dict[str, Dict[str, Any]],
    official_rows: Dict[str, Dict[str, Any]],
    bounded_rows: Dict[str, Dict[str, Any]],
    s3_max: int,
    repair_off_min: int,
    repair_off_max: int,
    repair_delta_min: int,
    repair_delta_max: int,
    suspicion_off_min: int,
    suspicion_off_max: int,
    suspicion_delta_min: int,
) -> Dict[str, Any]:
    repair_triggered: List[str] = []
    suspicion_triggered: List[str] = []
    mixed_passes: Dict[str, bool] = {}
    wins: List[str] = []
    losses: List[str] = []

    for task_id in task_ids:
        s3_row = s3_rows[task_id]
        off_row = official_rows[task_id]
        selected_s3 = s3_len(s3_row)
        selected_off = official_len(off_row)
        should_repair = repair_trigger(
            selected_s3,
            selected_off,
            s3_max=s3_max,
            off_min=repair_off_min,
            off_max=repair_off_max,
            delta_min=repair_delta_min,
            delta_max=repair_delta_max,
        )
        should_suspect = (
            selected_off is not None
            and selected_s3 <= s3_max
            and suspicion_off_min <= selected_off <= suspicion_off_max
            and selected_off - selected_s3 >= suspicion_delta_min
        )
        if should_repair:
            source_row = bounded_rows[task_id]
            repair_triggered.append(task_id)
        elif should_suspect:
            source_row = off_row
            suspicion_triggered.append(task_id)
        else:
            source_row = s3_row

        mixed_pass = passed(source_row)
        mixed_passes[task_id] = mixed_pass
        if mixed_pass and not passed(s3_row):
            wins.append(task_id)
        elif passed(s3_row) and not mixed_pass:
            losses.append(task_id)

    triggered = repair_triggered + suspicion_triggered
    triggered_rows = [s3_rows[task_id] for task_id in triggered]
    suspicion_rows = [s3_rows[task_id] for task_id in suspicion_triggered]
    pass_count = sum(1 for value in mixed_passes.values() if value)
    return {
        "policy": policy_name,
        "s3_max": s3_max,
        "repair_official_min": repair_off_min,
        "repair_official_max": repair_off_max,
        "repair_delta_min": repair_delta_min,
        "repair_delta_max": repair_delta_max,
        "suspicion_official_min": suspicion_off_min,
        "suspicion_official_max": suspicion_off_max,
        "suspicion_delta_min": suspicion_delta_min,
        "num_samples": len(task_ids),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(task_ids) if task_ids else None,
        "trigger_count": len(triggered),
        "repair_trigger_count": len(repair_triggered),
        "suspicion_trigger_count": len(suspicion_triggered),
        "wins_vs_s3": len(wins),
        "losses_vs_s3": len(losses),
        "net_vs_s3": len(wins) - len(losses),
        "trigger_win_rate": len(wins) / len(triggered) if triggered else None,
        "trigger_loss_rate": len(losses) / len(triggered) if triggered else None,
        "trigger_oracle_bucket_histogram": hist(bucket(oracle_len(row)) for row in triggered_rows),
        "suspicion_oracle_bucket_histogram": hist(bucket(oracle_len(row)) for row in suspicion_rows),
        "suspicion_true_long_precision": rate(
            oracle_len(row) is not None and oracle_len(row) >= 17 for row in suspicion_rows
        ),
        "suspicion_nonshort_precision": rate(
            oracle_len(row) is not None and oracle_len(row) >= 9 for row in suspicion_rows
        ),
        "triggered_task_ids": triggered,
        "repair_task_ids": repair_triggered,
        "suspicion_task_ids": suspicion_triggered,
        "wins_task_ids": wins,
        "losses_task_ids": losses,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline repair-v2 replay and long-suspicion diagnostics for LCAL official repair."
    )
    parser.add_argument("--s3-results", required=True)
    parser.add_argument("--official-results", required=True)
    parser.add_argument("--bounded-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repair-s3-max-values", default="3,4,5")
    parser.add_argument("--repair-official-max-values", default="8,9,10,11")
    parser.add_argument("--repair-delta-max-values", default="3,4,5,8")
    parser.add_argument("--repair-official-min", type=int, default=6)
    parser.add_argument("--repair-delta-min", type=int, default=1)
    parser.add_argument("--suspicion-official-min-values", default="12,13,16,17,20")
    parser.add_argument("--suspicion-official-max", type=int, default=64)
    parser.add_argument("--suspicion-s3-max", type=int, default=5)
    parser.add_argument("--suspicion-ratio-values", default="none,0.90,0.95,0.97")
    parser.add_argument("--suspicion-raw-ratio-values", default="none,0.85,0.90,0.97")
    parser.add_argument("--suspicion-support-values", default="0,1,2")
    args = parser.parse_args()

    s3_rows = as_map(load_jsonl(args.s3_results))
    official_rows = as_map(load_jsonl(args.official_results))
    bounded_rows = as_map(load_jsonl(args.bounded_results))
    task_ids = sorted(set(s3_rows) & set(official_rows) & set(bounded_rows))
    if not task_ids:
        raise RuntimeError("No common task ids across input results")

    repair_s3_max_values = [int(value) for value in args.repair_s3_max_values.split(",") if value.strip()]
    repair_official_max_values = [
        int(value) for value in args.repair_official_max_values.split(",") if value.strip()
    ]
    repair_delta_max_values = [int(value) for value in args.repair_delta_max_values.split(",") if value.strip()]

    repair_rows: List[Dict[str, Any]] = []
    for s3_max in repair_s3_max_values:
        for off_max in repair_official_max_values:
            for delta_max in repair_delta_max_values:
                name = f"repair_s3le{s3_max}_off{args.repair_official_min}-{off_max}_delta{args.repair_delta_min}-{delta_max}"
                repair_rows.append(
                    summarize_repair_policy(
                        policy_name=name,
                        task_ids=task_ids,
                        s3_rows=s3_rows,
                        official_rows=official_rows,
                        bounded_rows=bounded_rows,
                        s3_max=s3_max,
                        off_min=args.repair_official_min,
                        off_max=off_max,
                        delta_min=args.repair_delta_min,
                        delta_max=delta_max,
                    )
                )

    def parse_optional_floats(text: str) -> List[Optional[float]]:
        values: List[Optional[float]] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            values.append(None if part.lower() == "none" else float(part))
        return values

    off_min_values = [int(value) for value in args.suspicion_official_min_values.split(",") if value.strip()]
    ratio_values = parse_optional_floats(args.suspicion_ratio_values)
    raw_ratio_values = parse_optional_floats(args.suspicion_raw_ratio_values)
    support_values = [int(value) for value in args.suspicion_support_values.split(",") if value.strip()]

    suspicion_rows: List[Dict[str, Any]] = []
    for off_min in off_min_values:
        for ratio_min in ratio_values:
            for raw_ratio_min in raw_ratio_values:
                for support_min in support_values:
                    name = (
                        f"susp_s3le{args.suspicion_s3_max}_offge{off_min}"
                        f"_ratio{ratio_min if ratio_min is not None else 'none'}"
                        f"_raw{raw_ratio_min if raw_ratio_min is not None else 'none'}"
                        f"_support{support_min}"
                    )
                    suspicion_rows.append(
                        summarize_suspicion_policy(
                            policy_name=name,
                            task_ids=task_ids,
                            s3_rows=s3_rows,
                            official_rows=official_rows,
                            s3_max=args.suspicion_s3_max,
                            off_min=off_min,
                            off_max=args.suspicion_official_max,
                            ratio_min=ratio_min,
                            raw_ratio_min=raw_ratio_min,
                            support_min=support_min,
                        )
                    )

    union_rows: List[Dict[str, Any]] = []
    for repair_off_max in repair_official_max_values:
        for suspicion_off_min in off_min_values:
            name = (
                f"union_s3le{args.suspicion_s3_max}"
                f"_repair{args.repair_official_min}-{repair_off_max}"
                f"_delta{args.repair_delta_min}-{max(repair_delta_max_values)}"
                f"_suspge{suspicion_off_min}"
            )
            union_rows.append(
                summarize_union_policy(
                    policy_name=name,
                    task_ids=task_ids,
                    s3_rows=s3_rows,
                    official_rows=official_rows,
                    bounded_rows=bounded_rows,
                    s3_max=args.suspicion_s3_max,
                    repair_off_min=args.repair_official_min,
                    repair_off_max=repair_off_max,
                    repair_delta_min=args.repair_delta_min,
                    repair_delta_max=max(repair_delta_max_values),
                    suspicion_off_min=suspicion_off_min,
                    suspicion_off_max=args.suspicion_official_max,
                    suspicion_delta_min=args.repair_delta_min,
                )
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repair_sort = sorted(
        repair_rows,
        key=lambda row: (
            row["net_vs_s3"],
            -row["losses_vs_s3"],
            row["pass_rate"] if row["pass_rate"] is not None else -1,
        ),
        reverse=True,
    )
    suspicion_sort = sorted(
        suspicion_rows,
        key=lambda row: (
            row["true_long_precision"] if row["true_long_precision"] is not None else -1,
            row["suspect_count"],
            row["official_vs_s3_net"],
        ),
        reverse=True,
    )
    union_sort = sorted(
        union_rows,
        key=lambda row: (
            row["pass_count"],
            row["net_vs_s3"],
            -row["losses_vs_s3"],
            -row["trigger_count"],
        ),
        reverse=True,
    )

    payload = {
        "inputs": {
            "s3_results": args.s3_results,
            "official_results": args.official_results,
            "bounded_results": args.bounded_results,
            "common_task_count": len(task_ids),
        },
        "s3_pass_count": sum(1 for task_id in task_ids if passed(s3_rows[task_id])),
        "official_pass_count": sum(1 for task_id in task_ids if passed(official_rows[task_id])),
        "bounded_pass_count": sum(1 for task_id in task_ids if passed(bounded_rows[task_id])),
        "repair_v2": repair_rows,
        "repair_v2_top10": repair_sort[:10],
        "long_suspicion": suspicion_rows,
        "long_suspicion_top10_by_true_long_precision": suspicion_sort[:10],
        "repair_suspicion_union": union_rows,
        "repair_suspicion_union_top10": union_sort[:10],
    }

    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    repair_fields = [
        "policy",
        "s3_max",
        "official_min",
        "official_max",
        "delta_min",
        "delta_max",
        "pass_count",
        "pass_rate",
        "trigger_count",
        "wins_vs_s3",
        "losses_vs_s3",
        "net_vs_s3",
        "short_wins",
        "short_losses",
        "trigger_win_rate",
        "trigger_loss_rate",
        "nonshort_trigger_precision",
        "true_long_trigger_precision",
    ]
    suspicion_fields = [
        "policy",
        "s3_max",
        "official_min",
        "official_max",
        "ratio_min",
        "raw_ratio_min",
        "support_min",
        "suspect_count",
        "suspect_rate",
        "true_long_precision",
        "nonshort_precision",
        "short_precision_bad",
        "s3_failed_count",
        "s3_failed_rate",
        "official_vs_s3_wins",
        "official_vs_s3_losses",
        "official_vs_s3_net",
        "avg_s3_len",
        "avg_official_len",
    ]
    union_fields = [
        "policy",
        "s3_max",
        "repair_official_min",
        "repair_official_max",
        "repair_delta_min",
        "repair_delta_max",
        "suspicion_official_min",
        "suspicion_official_max",
        "suspicion_delta_min",
        "pass_count",
        "pass_rate",
        "trigger_count",
        "repair_trigger_count",
        "suspicion_trigger_count",
        "wins_vs_s3",
        "losses_vs_s3",
        "net_vs_s3",
        "trigger_win_rate",
        "trigger_loss_rate",
        "suspicion_true_long_precision",
        "suspicion_nonshort_precision",
    ]
    write_csv(output_dir / "repair_v2_sweep.csv", repair_sort, repair_fields)
    write_csv(output_dir / "long_suspicion_diag.csv", suspicion_sort, suspicion_fields)
    write_csv(output_dir / "repair_suspicion_union.csv", union_sort, union_fields)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
