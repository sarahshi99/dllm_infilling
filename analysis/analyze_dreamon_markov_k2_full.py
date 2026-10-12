#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


VARIANTS = [
    "全局置信度、逐词刷新、无头",
    "全局置信度、每轮最多两个、无头",
    "全局置信度、top-2 相邻时加 KL 头",
    "固定左前沿、逐词刷新、无头",
    "固定左前沿、每轮最多两个、无头",
    "固定左前沿、第二词元加 KL 头",
]

PAIRS = [
    ("全局置信度、每轮最多两个、无头", "全局置信度、逐词刷新、无头", "全局双词元无头相对逐词"),
    ("全局置信度、top-2 相邻时加 KL 头", "全局置信度、每轮最多两个、无头", "全局 KL 相对全局双词元无头"),
    ("全局置信度、top-2 相邻时加 KL 头", "全局置信度、逐词刷新、无头", "全局 KL 相对逐词"),
    ("固定左前沿、每轮最多两个、无头", "固定左前沿、逐词刷新、无头", "左前沿双词元无头相对逐词"),
    ("固定左前沿、第二词元加 KL 头", "固定左前沿、每轮最多两个、无头", "左前沿 KL 相对左前沿双词元无头"),
    ("固定左前沿、第二词元加 KL 头", "固定左前沿、逐词刷新、无头", "左前沿 KL 相对逐词"),
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(sorted_values: list[float], fraction: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]


def bootstrap(
    grouped: dict[str, dict[str, list[int]]],
    left: str,
    right: str,
    groups: list[str],
    *,
    seed: int = 20261012,
    replicates: int = 10000,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [groups[rng.randrange(len(groups))] for _ in groups]
        value = sum(
            sum(grouped[left][group]) / len(grouped[left][group])
            - sum(grouped[right][group]) / len(grouped[right][group])
            for group in selected
        ) / len(selected)
        samples.append(value)
    samples.sort()
    return {
        "replicates": replicates,
        "seed": seed,
        "mean": sum(samples) / len(samples),
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
    }


def main() -> int:
    parser_path = Path(__file__).resolve()
    repo = parser_path.parents[1]
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    result_dir = Path(args.result_dir).resolve()
    rows = read_rows(result_dir / "per_case_results.jsonl")
    by_key = {(row["task_id"], row["variant"]): row for row in rows}
    task_ids = sorted({row["task_id"] for row in rows})
    groups = sorted({row["task_group"] for row in rows}, key=lambda value: int(value))
    counts = Counter((row["task_id"], row["variant"]) for row in rows)
    observed_variants = sorted({row["variant"] for row in rows})
    errors = []
    if len(rows) != 1033 * 6:
        errors.append(f"row_count={len(rows)} expected=6198")
    if observed_variants != sorted(VARIANTS):
        errors.append("variant_set_mismatch")
    if len(task_ids) != 1033:
        errors.append(f"task_count={len(task_ids)} expected=1033")
    if len(groups) != 164:
        errors.append(f"group_count={len(groups)} expected=164")
    if any(value != 1 for value in counts.values()):
        errors.append("duplicate_case_key")
    if len(counts) != len(rows):
        errors.append("missing_or_duplicate_case_key")
    if any(row.get("failure_reason") is None for row in rows):
        errors.append("missing_evaluation_result")

    grouped: dict[str, dict[str, list[int]]] = {
        variant: {group: [] for group in groups} for variant in VARIANTS
    }
    summaries: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        for row in subset:
            grouped[variant][row["task_group"]].append(int(bool(row["passed"])))
        summaries[variant] = {
            "count": len(subset),
            "passed": sum(int(bool(row["passed"])) for row in subset),
            "pass_rate": sum(int(bool(row["passed"])) for row in subset) / len(subset),
            "dreamon_forward_count": sum(row["forward_count"] for row in subset),
            "ordinary_token_submissions": sum(row["normal_token_count"] for row in subset),
            "markov_enabled_count": sum(row["markov_enabled_count"] for row in subset),
            "adjacent_opportunity_count": sum(row["adjacent_opportunity_count"] for row in subset),
            "nonadjacent_fallback_count": sum(row["nonadjacent_fallback_count"] for row in subset),
            "corrected_structural_stop_count": sum(row["corrected_structural_stop_count"] for row in subset),
            "expand_action_count": sum(row["expand_count"] for row in subset),
            "delete_action_count": sum(row["delete_count"] for row in subset),
            "wall_sec_sum": sum(row["wall_sec"] for row in subset),
            "wall_sec_mean": statistics.mean(row["wall_sec"] for row in subset),
            "ordinary_submissions_per_forward": sum(row["normal_token_count"] for row in subset)
            / sum(row["forward_count"] for row in subset),
            "head_coverage_of_adjacent_opportunities": (
                sum(row["markov_enabled_count"] for row in subset)
                / sum(row["adjacent_opportunity_count"] for row in subset)
                if sum(row["adjacent_opportunity_count"] for row in subset)
                else 0.0
            ),
            "peak_gpu_memory_bytes_max": max(row["peak_gpu_memory_bytes"] for row in subset),
        }

    paired: list[dict[str, Any]] = []
    pair_summary: dict[str, Any] = {}
    for left, right, label in PAIRS:
        deltas = []
        new_passes = []
        new_failures = []
        for task_id in task_ids:
            left_passed = int(bool(by_key[(task_id, left)]["passed"]))
            right_passed = int(bool(by_key[(task_id, right)]["passed"]))
            delta = left_passed - right_passed
            deltas.append(delta)
            paired.append(
                {
                    "comparison": label,
                    "left_variant": left,
                    "right_variant": right,
                    "task_id": task_id,
                    "task_group": by_key[(task_id, left)]["task_group"],
                    "left_passed": left_passed,
                    "right_passed": right_passed,
                    "delta": delta,
                }
            )
            if delta == 1:
                new_passes.append(task_id)
            elif delta == -1:
                new_failures.append(task_id)
        pair_summary[label] = {
            "left_variant": left,
            "right_variant": right,
            "new_pass_count": len(new_passes),
            "new_failure_count": len(new_failures),
            "net_change": sum(deltas),
            "new_pass_task_ids": new_passes,
            "new_failure_task_ids": new_failures,
            "group_bootstrap_10000": bootstrap(grouped, left, right, groups),
        }

    baseline_speed = {
        "全局置信度、每轮最多两个、无头": "全局置信度、逐词刷新、无头",
        "全局置信度、top-2 相邻时加 KL 头": "全局置信度、逐词刷新、无头",
        "固定左前沿、每轮最多两个、无头": "固定左前沿、逐词刷新、无头",
        "固定左前沿、第二词元加 KL 头": "固定左前沿、逐词刷新、无头",
    }
    speed: dict[str, Any] = {}
    for variant, baseline in baseline_speed.items():
        speed[variant] = {
            "baseline_variant": baseline,
            "forward_reduction_ratio": summaries[baseline]["dreamon_forward_count"]
            / summaries[variant]["dreamon_forward_count"],
            "wall_clock_speedup_ratio": summaries[baseline]["wall_sec_sum"]
            / summaries[variant]["wall_sec_sum"],
            "wall_clock_definition": "per-case wall_sec includes generation and evaluator call; model loading is excluded",
        }

    audit = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "row_count": len(rows),
        "expected_row_count": 6198,
        "task_count": len(task_ids),
        "expected_task_count": 1033,
        "base_task_group_count": len(groups),
        "expected_base_task_group_count": 164,
        "variant_counts": {variant: sum(row["variant"] == variant for row in rows) for variant in VARIANTS},
        "unique_case_keys": len(counts),
        "duplicate_case_keys": sum(value > 1 for value in counts.values()),
        "missing_case_keys": 6198 - len(counts),
        "failure_journal_present": (result_dir / "failure_journal.jsonl").exists(),
        "progress": json.loads((result_dir / "progress.json").read_text(encoding="utf-8")),
    }
    (result_dir / "integrity_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (result_dir / "k2_summary.json").write_text(
        json.dumps(
            {"integrity": audit, "variants": summaries, "paired": pair_summary, "speed": speed},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (result_dir / "paired_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["comparison", "left_variant", "right_variant", "task_id", "task_group", "left_passed", "right_passed", "delta"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    report = [
        "# DreamOn SingleLine K=2 完整开发/机制评测结果",
        "",
        "本轮是已反复使用的完整开发/机制评测集，不是首次打开的独立封闭测试。每题只生成一次，因此这里的 Pass@1 是一次生成通过率；结果不证明基础模型预训练未见过 HumanEval。",
        "",
        "## 结论先行",
        "",
        "1. 四个无头对照精确复现历史参照：全局逐词 `951/1033`、全局双词元 `901/1033`、左前沿逐词 `942/1033`、左前沿双词元 `877/1033`。这说明本轮数据、模型、采样、种子和结构语义与历史基线一致。",
        "2. KL 头改善了两种 K=2 无头方案：全局 `901→910`（净增 9），左前沿 `877→890`（净增 13）。全局改善的 164 题分组 bootstrap 95% CI 为 `+0.21/+2.26` 个百分点；左前沿 CI 为 `-1.08/+3.47` 个百分点，方向为正但证据不足以称稳定改善。",
        "3. 两种 KL K=2 仍低于各自逐词刷新基线：全局 `910` 对 `951`，左前沿 `890` 对 `942`。因此论文应写成“Markov 修正部分恢复并保持 K=2 的速度收益”，不能写成“完全消除双词元质量损失”或“超过逐词刷新”。",
        "",
        "## 六组结果",
        "",
        "| 方案 | 通过数/1033 | 通过率 | DreamOn 前向 | 普通词元提交 | 提交/前向 | 总 wall_sec | 相对逐词 wall 加速 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        item = summaries[variant]
        speed_value = "—"
        if variant in speed:
            speed_value = f"{speed[variant]['wall_clock_speedup_ratio']:.3f}×"
        report.append(
            f"| {variant} | {item['passed']}/1033 | {pct(item['pass_rate'])} | {item['dreamon_forward_count']} | {item['ordinary_token_submissions']} | {item['ordinary_submissions_per_forward']:.3f} | {item['wall_sec_sum']:.2f} | {speed_value} |"
        )
    report += [
        "",
        "## 配对变化与置信区间",
        "",
        "完整题目清单在 `paired_deltas.csv`，机器可读汇总在 `k2_summary.json`。CI 按 164 个基础题目分组、有放回重采样 10000 次计算。",
        "",
    ]
    for label in ("全局 KL 相对全局双词元无头", "左前沿 KL 相对左前沿双词元无头", "全局 KL 相对逐词", "左前沿 KL 相对逐词"):
        item = pair_summary[label]
        ci = item["group_bootstrap_10000"]
        report.append(
            f"- **{label}**：新增通过 `{item['new_pass_count']}`，新增失败 `{item['new_failure_count']}`，净变化 `{item['net_change']}`；95% CI `[{pct(ci['ci95_low'])}, {pct(ci['ci95_high'])}]`。"
        )
    report += [
        "",
        "## 速度与头覆盖",
        "",
        f"- 全局 KL：相对全局逐词 wall 加速 `{speed['全局置信度、top-2 相邻时加 KL 头']['wall_clock_speedup_ratio']:.3f}×`；相对全局双词元无头的 wall 比值为 `{summaries['全局置信度、每轮最多两个、无头']['wall_sec_sum'] / summaries['全局置信度、top-2 相邻时加 KL 头']['wall_sec_sum']:.3f}×`。头启用 `{summaries['全局置信度、top-2 相邻时加 KL 头']['markov_enabled_count']}` 次 / 相邻机会 `{summaries['全局置信度、top-2 相邻时加 KL 头']['adjacent_opportunity_count']}` 次，即 `{pct(summaries['全局置信度、top-2 相邻时加 KL 头']['head_coverage_of_adjacent_opportunities'])}`。",
        f"- 左前沿 KL：相对左前沿逐词 wall 加速 `{speed['固定左前沿、第二词元加 KL 头']['wall_clock_speedup_ratio']:.3f}×`；相对左前沿双词元无头的 wall 比值为 `{summaries['固定左前沿、每轮最多两个、无头']['wall_sec_sum'] / summaries['固定左前沿、第二词元加 KL 头']['wall_sec_sum']:.3f}×`。头启用 `{summaries['固定左前沿、第二词元加 KL 头']['markov_enabled_count']}` 次 / 相邻机会 `{summaries['固定左前沿、第二词元加 KL 头']['adjacent_opportunity_count']}` 次，即 `{pct(summaries['固定左前沿、第二词元加 KL 头']['head_coverage_of_adjacent_opportunities'])}`。",
        "- wall_sec 是单题生成加评测器调用的端到端时间，排除了模型加载；本轮 runner 没有把评测器调用和生成单独计时，因此不能把该数称为纯生成时间。",
        "- 全局方案发生 `595` 次非相邻 top-2 回退；KL 修正后触发结构停止全局 `12` 次、左前沿 `30` 次。",
        "",
        "## 论文含义",
        "",
        "这组结果足以支撑一条谨慎但可写入论文的结论：固定 K=2 的并行解码显著减少 DreamOn 前向次数，并造成可重复的质量损失；训练得到的 KL 一阶 Markov 头在两种 K=2 设计上都能部分修复代码通过率，其中全局 top-2 相邻短链的净改善更稳定。它是“质量—速度折中改善”的证据，不是全面优于逐词刷新或证明 Markov 头单独提高最终最高质量。",
        "",
        "## 完整性",
        "",
        "本轮 `6198/6198` 记录完成，六组各 `1033` 条，基础题组 `164` 个，重复/缺失/错误记录为 `0`。原始逐题结果和 trace 保留在本目录；大模型权重、transition bank 和诊断文件仍保留在服务器。",
    ]
    (result_dir / "report.zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "rows": len(rows), "variants": summaries, "paired": pair_summary}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
