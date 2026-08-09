from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifests import read_jsonl, write_json
from .protocol import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPERIMENT_DIR,
    FIXED_FULL_ROWS,
    FIXED_METADATA_PATH,
    PILOT_GATE_PASSES,
    PILOT_ROWS,
    PILOT_METADATA_PATH,
)
from .runner import (
    FIXED_RESULTS_PATH,
    FIXED_SUMMARY_PATH,
    PILOT_RESULTS_PATH,
    PILOT_SUMMARY_PATH,
    summarize,
)


REPORT_PATH = EXPERIMENT_DIR / "report.zh.md"


def mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return statistics.fmean(float(row.get(field) or 0) for row in rows) if rows else 0.0


def totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "pass_at_1": sum(bool(row.get("pass_at_1")) for row in rows),
        "compiled": sum(bool(row.get("compiled")) for row in rows),
        "completed": sum(bool(row.get("completed")) for row in rows),
        "mean_forwards": mean(rows, "forward_count"),
        "mean_wall_time_seconds": mean(rows, "wall_time_seconds"),
        "mean_gpu_time_seconds": mean(rows, "gpu_time_seconds"),
        "total_wall_time_seconds": sum(float(row.get("wall_time_seconds") or 0) for row in rows),
        "total_gpu_time_seconds": sum(float(row.get("gpu_time_seconds") or 0) for row in rows),
        "total_token_forwards": sum(int(row.get("token_forwards") or 0) for row in rows),
        "normal_tokens": sum(int(row.get("normal_token_count") or 0) for row in rows),
        "expand": sum(int(row.get("expand_count") or 0) for row in rows),
        "delete_actions": sum(int(row.get("delete_action_count") or 0) for row in rows),
        "single_point_delete": sum(int(row.get("single_point_delete_count") or 0) for row in rows),
        "broadcast_delete": sum(int(row.get("broadcast_delete_count") or 0) for row in rows),
        "broadcast_delete_rows": sum(int(row.get("broadcast_delete_count") or 0) > 0 for row in rows),
        "incomplete": sum(not bool(row.get("completed")) for row in rows),
        "cycle_rows": sum(bool(row.get("oscillation_detected")) for row in rows),
        "frontier_violations": sum(int(row.get("frontier_violation") or 0) for row in rows),
    }


def failure_category(row: Mapping[str, Any]) -> str:
    if bool(row.get("pass_at_1")):
        return "passed"
    if row.get("exception"):
        return "generation_exception"
    if not bool(row.get("completed")) and bool(row.get("oscillation_detected")):
        return "incomplete_cycle_or_oscillation"
    if not bool(row.get("completed")):
        return "incomplete_unresolved_masks"
    if not bool(row.get("compiled")):
        return "compile_failure"
    return "functional_failure_after_compile"


def cluster_bootstrap_delta(
    candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline_by_id = {str(row["sample_id"]): row for row in baseline}
    clusters: dict[str, list[int]] = defaultdict(list)
    for row in candidate:
        other = baseline_by_id[str(row["sample_id"])]
        clusters[str(row["base_task_id"])].append(
            int(bool(row.get("pass_at_1"))) - int(bool(other.get("pass_at_1")))
        )
    cluster_values = [(sum(values), len(values)) for values in clusters.values()]
    observed = sum(value for value, _ in cluster_values) / sum(count for _, count in cluster_values)
    rng = random.Random(BOOTSTRAP_SEED)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [rng.choice(cluster_values) for _ in cluster_values]
        draws.append(sum(value for value, _ in sampled) / sum(count for _, count in sampled))
    draws.sort()
    lower = draws[int(0.025 * len(draws))]
    upper = draws[int(0.975 * len(draws))]
    return {
        "unit": "base_task_id cluster",
        "clusters": len(cluster_values),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "observed_delta": observed,
        "ci95": [lower, upper],
    }


def paired_comparison(
    candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline_by_id = {str(row["sample_id"]): row for row in baseline}
    help_count = harm_count = both_pass = both_fail = 0
    compile_help = compile_harm = 0
    broadcast_more = broadcast_less = broadcast_same = 0
    broadcast_early_harm = 0
    for row in candidate:
        other = baseline_by_id[str(row["sample_id"])]
        candidate_pass = bool(row.get("pass_at_1"))
        baseline_pass = bool(other.get("pass_at_1"))
        if candidate_pass and not baseline_pass:
            help_count += 1
        elif baseline_pass and not candidate_pass:
            harm_count += 1
        elif candidate_pass:
            both_pass += 1
        else:
            both_fail += 1
        if bool(row.get("compiled")) and not bool(other.get("compiled")):
            compile_help += 1
        elif bool(other.get("compiled")) and not bool(row.get("compiled")):
            compile_harm += 1
        candidate_broadcast = int(row.get("broadcast_delete_count") or 0)
        baseline_broadcast = int(other.get("broadcast_delete_count") or 0)
        if candidate_broadcast > baseline_broadcast:
            broadcast_more += 1
            if baseline_pass and not candidate_pass:
                broadcast_early_harm += 1
        elif candidate_broadcast < baseline_broadcast:
            broadcast_less += 1
        else:
            broadcast_same += 1
    denominator = len(candidate)
    return {
        "help": help_count,
        "harm": harm_count,
        "tie": both_pass + both_fail,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "pass_at_1_delta": (help_count - harm_count) / denominator,
        "compile_help": compile_help,
        "compile_harm": compile_harm,
        "broadcast_delete_more": broadcast_more,
        "broadcast_delete_less": broadcast_less,
        "broadcast_delete_same": broadcast_same,
        "broadcast_delete_more_and_pass_harm": broadcast_early_harm,
        "cluster_bootstrap": cluster_bootstrap_delta(candidate, baseline),
    }


def enrich_pilot() -> dict[str, Any]:
    rows = read_jsonl(PILOT_RESULTS_PATH)
    metadata = json.loads(PILOT_METADATA_PATH.read_text(encoding="utf-8"))
    by_width = summarize(rows, PILOT_ROWS)
    for width, summary in by_width.items():
        selected = [row for row in rows if row["w"] == width]
        summary.update(totals(selected))
        summary["failure_categories"] = dict(Counter(failure_category(row) for row in selected))
    promoted = [width for width, value in by_width.items() if value["passed"] >= PILOT_GATE_PASSES]
    value = {
        "stage": "pilot30",
        "role": "development/mechanism population",
        "manifest_id": metadata["manifest_id"],
        "manifest_sha256": metadata["manifest_sha256"],
        "gate": f"Pass@1 >= {PILOT_GATE_PASSES}/{PILOT_ROWS}",
        "widths": by_width,
        "promoted": promoted,
        "rows": len(rows),
    }
    write_json(PILOT_SUMMARY_PATH, value)
    return value


def enrich_fixed() -> dict[str, Any] | None:
    if not FIXED_RESULTS_PATH.exists():
        return None
    rows = read_jsonl(FIXED_RESULTS_PATH)
    metadata = json.loads(FIXED_METADATA_PATH.read_text(encoding="utf-8"))
    widths = sorted({str(row["w"]) for row in rows}, key=lambda value: ["1", "4", "8", "16", "inf"].index(value))
    if any(len([row for row in rows if row["w"] == width]) != FIXED_FULL_ROWS for width in widths):
        raise RuntimeError("fixed-full results are not complete")
    by_width_rows = {width: [row for row in rows if row["w"] == width] for width in widths}
    baseline = by_width_rows.get("inf")
    if baseline is None:
        raise RuntimeError("fixed-full comparison requires w=inf")
    summaries: dict[str, Any] = {}
    for width, selected in by_width_rows.items():
        summary = totals(selected)
        summary.update(
            {
                "pass_rate": summary["pass_at_1"] / len(selected),
                "compile_rate": summary["compiled"] / len(selected),
                "completion_rate": summary["completed"] / len(selected),
                "failure_categories": dict(Counter(failure_category(row) for row in selected)),
            }
        )
        if width != "inf":
            summary["paired_vs_inf"] = paired_comparison(selected, baseline)
        summaries[width] = summary
    value = {
        "stage": "fixed-full-1000",
        "status": "completed",
        "role": "fixed-full-1000 development/validation population",
        "manifest_id": metadata["manifest_id"],
        "manifest_sha256": metadata["manifest_sha256"],
        "not_official_5815_full": True,
        "not_frozen_test": True,
        "executed_widths": widths,
        "widths": summaries,
        "rows": len(rows),
    }
    write_json(FIXED_SUMMARY_PATH, value)
    return value


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_report(pilot: Mapping[str, Any], fixed: Mapping[str, Any] | None) -> str:
    lines = [
        "# Frontier-Gated DreamOn V0 结果报告",
        "",
        "本报告只覆盖 Pilot-30 development/mechanism population 与 fixed-full-1000 development/validation population；不是官方 5815 full，也不是 frozen test。旧 DreamOn 逐行 V1/V2/V3 路线保持冻结，其负向/混合证据不删除。",
        "",
        "## 实现与等价门",
        "",
        "- 初始动态中间区是单一连续的 64 个 `[MASK]`；`max_new_tokens=64`。",
        "- 唯一方法改动是 frontier 窗口内未解决 mask 的位置 eligibility；完整 prefix/middle/suffix 始终参加官方 DreamOn forward。",
        "- 换行是普通 token；没有 line slot、separator、截断、重试、nonempty、compile gate、blacklist、BoundaryShift 或 repair。",
        "- `w=∞` 与未经修改的官方 DreamOn 在 55 条真实样本上 final tokens/text、逐步位置/动作与停止原因完全一致，并覆盖 normal、换行后继续生成、expand、delete。",
        "",
        "## Pilot-30",
        "",
        "| w | Pass@1 | 编译 | 完成 | 平均 forwards | 平均生成耗时 | 晋级 |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for width in ["1", "4", "8", "16", "inf"]:
        row = pilot["widths"][width]
        lines.append(
            f"| {width} | {row['passed']}/{row['denominator']} ({percent(row['pass_rate'])}) | "
            f"{row['compiled']}/{row['denominator']} ({percent(row['compile_rate'])}) | "
            f"{row['completed']}/{row['denominator']} ({percent(row['completion_rate'])}) | "
            f"{row['mean_forwards']:.2f} | {row['mean_wall_time_seconds']:.3f}s | "
            f"{'是' if width in pilot['promoted'] else '否'} |"
        )
    lines.extend(
        [
            "",
            f"自动晋级：{', '.join('w=' + item for item in pilot['promoted']) or '无'}。门槛严格固定为至少 16/30。",
            "",
        ]
    )
    if fixed is None:
        lines.extend(["## Fixed-full-1000", "", "尚未完成或未触发。", ""])
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## Fixed-full-1000",
            "",
            "| w | Pass@1 | 编译率 | 完成率 | 平均 forwards | 平均 wall | GPU time 合计 | token-forwards |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for width in fixed["executed_widths"]:
        row = fixed["widths"][width]
        lines.append(
            f"| {width} | {row['pass_at_1']}/{row['rows']} ({percent(row['pass_rate'])}) | "
            f"{percent(row['compile_rate'])} | {percent(row['completion_rate'])} | "
            f"{row['mean_forwards']:.2f} | {row['mean_wall_time_seconds']:.3f}s | "
            f"{row['total_gpu_time_seconds']:.1f}s | {row['total_token_forwards']} |"
        )
    lines.extend(
        [
            "",
            "### 配对比较（有限窗口 vs w=∞）",
            "",
            "| w | help | harm | tie | Pass@1 差值 | base-task cluster bootstrap 95% CI | compile help/harm | broadcast 更多/更少/相同 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for width in [item for item in fixed["executed_widths"] if item != "inf"]:
        paired = fixed["widths"][width]["paired_vs_inf"]
        ci = paired["cluster_bootstrap"]["ci95"]
        lines.append(
            f"| {width} | {paired['help']} | {paired['harm']} | {paired['tie']} | "
            f"{100 * paired['pass_at_1_delta']:+.2f}pp | [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]pp | "
            f"{paired['compile_help']}/{paired['compile_harm']} | "
            f"{paired['broadcast_delete_more']}/{paired['broadcast_delete_less']}/{paired['broadcast_delete_same']} |"
        )
    lines.extend(["", "### 动作、未完成与主要失败类别", ""])
    for width in fixed["executed_widths"]:
        row = fixed["widths"][width]
        lines.append(
            f"- `w={width}`：expand {row['expand']}，delete action {row['delete_actions']}，"
            f"single delete {row['single_point_delete']}，broadcast delete {row['broadcast_delete']}（{row['broadcast_delete_rows']} rows），"
            f"未完成 {row['incomplete']}，cycle/oscillation rows {row['cycle_rows']}；失败类别 {json.dumps(row['failure_categories'], ensure_ascii=False)}。"
        )
    lines.extend(
        [
            "",
            "### 广播删除型提前终止检查",
            "",
        ]
    )
    for width in [item for item in fixed["executed_widths"] if item != "inf"]:
        paired = fixed["widths"][width]["paired_vs_inf"]
        lines.append(
            f"- `w={width}` 相对 `w=∞`：broadcast delete 次数更多的样本 {paired['broadcast_delete_more']}，"
            f"更少 {paired['broadcast_delete_less']}，相同 {paired['broadcast_delete_same']}；"
            f"其中“更多 broadcast 且由 baseline pass 变为 fail” {paired['broadcast_delete_more_and_pass_harm']}。"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    pilot = enrich_pilot()
    fixed = enrich_fixed()
    REPORT_PATH.write_text(render_report(pilot, fixed), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
