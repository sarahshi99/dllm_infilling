#!/usr/bin/env python3
"""Paired single-seed attribution; all uncertainty resamples base tasks."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PAIRS = (("D", "C"), ("C", "B"), ("E", "C"), ("E", "D"),
         ("B", "A"), ("C", "A"), ("D", "A"), ("E", "A"))
POLICIES = ("global_confidence", "left_to_right_frontier")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else (["policy", "arm", "passed", "denominator", "generation_sec_total"]
                                           if path.name == "summary.csv" else ["policy", "comparison", "help", "harm", "net"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clustered_difference(rows: list[dict], left: str, right: str, *, replicates=10000, seed=20261013) -> dict:
    by_id = {(r["arm"], r["task_id"]): r for r in rows}
    grouped = defaultdict(list)
    for row in rows:
        if row["arm"] == left:
            grouped[str(row["base_task_id"])].append(int(row["passed"]) - int(by_id[right, row["task_id"]]["passed"]))
    groups = sorted(grouped)
    sums = np.array([sum(grouped[g]) for g in groups], dtype=float)
    sizes = np.array([len(grouped[g]) for g in groups], dtype=float)
    selected = np.random.default_rng(seed).integers(0, len(groups), size=(replicates, len(groups)))
    line_samples = sums[selected].sum(axis=1) / sizes[selected].sum(axis=1)
    macro_samples = (sums / sizes)[selected].mean(axis=1)
    result = {"line_delta": float(sums.sum() / sizes.sum()),
              "base_task_equal_delta": float((sums / sizes).mean()),
              "cluster_count": len(groups), "bootstrap_replicates": replicates, "bootstrap_seed": seed}
    for name, samples in (("line", line_samples), ("base_task_equal", macro_samples)):
        result[name + "_ci95_low"], result[name + "_ci95_high"] = map(float, np.quantile(samples, [0.025, 0.975]))
    return result


def integrity(rows: list[dict], manifest: dict, expected_ids: set[str]) -> dict:
    expected = {(p, a, task) for p in POLICIES for a in "ABCDE" for task in expected_ids}
    observed = Counter((r["policy"], r["arm"], r["task_id"]) for r in rows)
    errors = []
    if len(expected_ids) != 1033:
        errors.append("expected population is not 1033")
    if set(observed) != expected:
        errors.append("missing or unexpected case identities")
    if any(n != 1 for n in observed.values()):
        errors.append("duplicate case identities")
    from experiments.dreamon_singleline_adapter import derive_row_seed
    for row in rows:
        if row["base_task_id"] != row["task_id"].split("/")[2] or row["row_seed"] != derive_row_seed(42, row["task_id"]):
            errors.append("case grouping or row seed mismatch")
        if row["forward_count"] != row["base_forward_count"] + row["recompute_forward_count"]:
            errors.append("total forward accounting mismatch")
        if row["outer_round_count"] != row["base_forward_count"]:
            errors.append("outer round accounting mismatch")
        if row["arm"] != "E" and row["recompute_forward_count"]:
            errors.append("extra forward outside E")
        if row["arm"] == "E" and (row["markov_enabled_count"] or row["recompute_forward_count"] != row["recompute_enabled_count"]):
            errors.append("E branch accounting mismatch")
        if row["arm"] in "CD" and row["markov_enabled_count"] > row["adjacent_opportunity_count"]:
            errors.append("head opportunities denominator mismatch")
        if row["arm"] == "E" and row["recompute_enabled_count"] > row["adjacent_opportunity_count"]:
            errors.append("recompute opportunities denominator mismatch")
        if row["code_commit"] != manifest["code_commit"] or row["seed"] != 42:
            errors.append("case code or seed mismatch")
        if not isinstance(row["passed"], bool) or row.get("failure_reason") is None:
            errors.append("invalid evaluator record")
    return {"status": "passed" if not errors else "incomplete_or_invalid", "errors": sorted(set(errors)),
            "row_count": len(rows), "expected_row_count": 10330,
            "unique_case_count": len(observed), "duplicate_case_count": sum(n - 1 for n in observed.values()),
            "missing_case_count": len(expected - set(observed)), "unexpected_case_count": len(set(observed) - expected),
            "base_task_count": len({r["base_task_id"] for r in rows}),
            "counts": {f"{p}/{a}": sum(r["policy"] == p and r["arm"] == a for r in rows) for p in POLICIES for a in "ABCDE"}}


def analyze(out: Path) -> dict:
    manifest = json.loads((out / "run_manifest.json").read_text())
    db_path = out / "per_case.sqlite"
    rows = []
    if db_path.exists():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            rows = [json.loads(r[0]) for r in db.execute("SELECT payload FROM cases ORDER BY policy,arm,task_id")]
    with (Path(manifest["original_config_path"]).parent / "paired_deltas.csv").open() as handle:
        expected_ids = {r["task_id"] for r in csv.DictReader(handle)}
    audit = integrity(rows, manifest, expected_ids)
    smoke_path = out / "smoke_checks.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {"status": "not_run"}
    audit["smoke_status"] = smoke["status"]
    smoke_identities = {(r["policy"], r["arm"], r["task_id"]) for r in smoke.get("checks", [])}
    expected_smoke = {(p, a, t) for p in POLICIES for a in "ABCDE" for t in manifest["smoke_task_ids"]}
    if smoke["status"] != "passed" or smoke_identities != expected_smoke or len(smoke.get("checks", [])) != 160:
        audit["status"] = "incomplete_or_invalid"
        audit["errors"].append("16-line real-model diagnostics equivalence not passed")
    failures_path = out / "failure_journal.jsonl"
    failures = [json.loads(s) for s in failures_path.read_text().splitlines() if s] if failures_path.exists() else []
    successful = {(r["policy"], r["arm"], r["task_id"]) for r in rows}
    audit["historical_failure_events"] = len(failures)
    audit["unrecovered_failure_keys"] = len({(r["policy"], r["arm"], r["task_id"]) for r in failures} - successful)
    audit["no_new_sealed_test_access"] = True
    write_json(out / "integrity_audit.json", audit)
    # Compact per-case export is local evidence, not model/diagnostic tensors.
    with (out / "per_case.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if audit["status"] != "passed":
        write_csv(out / "summary.csv", [])
        write_csv(out / "paired_comparisons.csv", [])
        (out / "report.zh.md").write_text(
            "# DreamOn + Markov K2 诊断：未完成\n\n"
            f"状态：`{manifest['status']}`。阻塞：{manifest.get('blocker', '详见完整性审计')}。\n\n"
            f"已完成 `{len(rows)}/10330` 行；16 行真实模型检查 `{smoke['status']}`。"
            "未产生完整实验结论；空表表示未测量，不能解读为零收益。\n\n"
            "训练出的头究竟有没有帮忙？尚不能归因；旧 D−B 包含路径变化，必须测 D−C。\n\n"
            "公平计时后到底快多少？尚未测量；旧 wall 含评测和不对称诊断，不能复用。\n\n"
            "让模型重新算，能不能明显救回来？尚未测量。E 实现和 CPU 检查不能替代真实模型结果。\n\n"
            "下一轮最值得改哪一件事，为什么？先在空闲 H200 上完成本轮固定协议，"
            "获得归因和机制证据后再选算法改动。不得凭缺失结果选新方法。\n\n"
            "人口为反复使用的 1033 行/164 基础题开发/机制诊断人口，单生成种子 42。"
            "没有访问新的封存测试集、训练或调参。预算 12 GPU 小时，"
            "完整批次估算必须等待 16 行真实模型检查。\n\n"
            f"恢复命令：\n\n```bash\n{manifest['command']}\n```\n\n"
            f"代码：`{manifest['branch']}@{manifest['code_commit']}`。"
            "旧结果与旧判断保留。本轮不自动扩展到新方法。\n")
        return audit
    summaries = []
    totals = {}
    for policy in POLICIES:
        for arm in "ABCDE":
            subset = [r for r in rows if r["policy"] == policy and r["arm"] == arm]
            grouped = defaultdict(list)
            for row in subset:
                grouped[row["base_task_id"]].append(int(row["passed"]))
            result = {"policy": policy, "arm": arm, "passed": sum(r["passed"] for r in subset),
                      "denominator": len(subset), "pass_rate": np.mean([r["passed"] for r in subset]),
                      "base_task_equal_pass_rate": np.mean([np.mean(v) for v in grouped.values()]),
                      "generation_sec_total": sum(r["generation_sec"] for r in subset),
                      "generation_sec_median": float(np.median([r["generation_sec"] for r in subset])),
                      "generation_sec_p95": float(np.quantile([r["generation_sec"] for r in subset], .95)),
                      "evaluator_sec_total": sum(r["evaluator_sec"] for r in subset),
                      "file_write_sec_total": sum(r["file_write_sec"] for r in subset),
                      "output_tokens_total": sum(r["output_tokens"] for r in subset),
                      "output_tokens_mean": float(np.mean([r["output_tokens"] for r in subset])),
                      "output_chars_total": sum(r["output_chars"] for r in subset)}
            for key in ("forward_count", "base_forward_count", "recompute_forward_count", "outer_round_count",
                        "normal_token_count", "expand_count", "delete_count", "adjacent_opportunity_count",
                        "markov_enabled_count", "recompute_enabled_count", "structural_stop_count",
                        "corrected_structural_stop_count", "nonadjacent_fallback_count"):
                result[key] = sum(r[key] for r in subset)
            result["ordinary_commits_per_total_forward"] = result["normal_token_count"] / result["forward_count"]
            totals[policy, arm] = result
            summaries.append(result)
        for arm in "ABCDE":
            result = totals[policy, arm]
            speed = totals[policy, "A"]["generation_sec_total"] / result["generation_sec_total"]
            result["speed_ratio_vs_A"] = speed
            result["time_reduction_fraction_vs_A"] = 1 - 1 / speed
            result["speed_interpretation"] = "diagnostic_cost_only" if arm == "E" else "pure_generation"
    paired = []
    for policy in POLICIES:
        subset = [r for r in rows if r["policy"] == policy]
        by_id = {(r["arm"], r["task_id"]): r for r in subset}
        damaged = {task for task in expected_ids if by_id["A", task]["passed"] and not by_id["B", task]["passed"]}
        for left, right in PAIRS:
            helps = sum(by_id[left, t]["passed"] and not by_id[right, t]["passed"] for t in expected_ids)
            harms = sum(by_id[right, t]["passed"] and not by_id[left, t]["passed"] for t in expected_ids)
            result = {"policy": policy, "comparison": f"{left}-{right}", "help": helps, "harm": harms,
                      "net": helps - harms, **clustered_difference(subset, left, right)}
            paired.append(result)
        for arm in "CDE":
            rescued = sum(by_id[arm, t]["passed"] for t in damaged)
            other = expected_ids - damaged
            harm = sum(by_id["B", t]["passed"] and not by_id[arm, t]["passed"] for t in other)
            paired.append({"policy": policy, "comparison": f"{arm} rescue on A-pass/B-fail",
                           "help": rescued, "harm": harm, "net": "", "line_delta": "",
                           "base_task_equal_delta": "", "cluster_count": "", "bootstrap_replicates": "",
                           "bootstrap_seed": "", "line_ci95_low": "", "line_ci95_high": "",
                           "base_task_equal_ci95_low": "", "base_task_equal_ci95_high": "",
                           "rescue_denominator": len(damaged), "other_population_denominator": len(other),
                           "other_harm_vs": "B"})
    all_keys = list(dict.fromkeys(k for r in paired for k in r))
    write_csv(out / "paired_comparisons.csv", [{k: r.get(k, "") for k in all_keys} for r in paired])
    write_csv(out / "summary.csv", summaries)
    lines = ["# DreamOn + Markov K2 归因与机制诊断", "",
             "完整 10330/10330 行；1033 行/164 基础题是反复使用的开发/机制诊断人口，"
             "不是新的独立测试。单生成种子 42，95% 区间仅表达这一生成种子条件下的基础题重采样不确定性。", "",
             "A=K1 无头，B=K2 无头，C=同路径零输出头 λ=1，D=v2 KL λ=1，E=固定右位只提交左词后重算。", "",
             "|策略|组|通过/1033|通过率|生成总秒|中位/P95秒|输出词元总计|基础/额外/总前向|普通提交/总前向|相对A速度|", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summaries:
        lines.append(f"|{r['policy']}|{r['arm']}|{r['passed']}/1033|{100*r['pass_rate']:.2f}%|{r['generation_sec_total']:.2f}|{r['generation_sec_median']:.4f}/{r['generation_sec_p95']:.4f}|{r['output_tokens_total']}|{r['base_forward_count']}/{r['recompute_forward_count']}/{r['forward_count']}|{r['ordinary_commits_per_total_forward']:.3f}|{r['speed_ratio_vs_A']:.3f}×|")
    lines += ["", "E 的速度只表示诊断成本；E 不是验证器、数学上界或 K1 的等价实现，global 没有重选右位置。", "",
              "|策略|配对|帮助/伤害|净行数|行加权差及95%CI(pp)|基础题等权差及95%CI(pp)|", "|---|---|---|---|---|---|"]
    comparisons = {(r["policy"], r["comparison"]): r for r in paired}
    for r in paired:
        if r["net"] == "":
            continue
        lines.append(f"|{r['policy']}|{r['comparison']}|{r['help']}/{r['harm']}|{r['net']:+d}|{100*r['line_delta']:+.2f} [{100*r['line_ci95_low']:+.2f},{100*r['line_ci95_high']:+.2f}]|{100*r['base_task_equal_delta']:+.2f} [{100*r['base_task_equal_ci95_low']:+.2f},{100*r['base_task_equal_ci95_high']:+.2f}]|")
    lines += ["", "训练出的头究竟有没有帮忙？"]
    for policy in POLICIES:
        trained = comparisons[policy, "D-C"]
        path = comparisons[policy, "C-B"]
        certainty = "方向为正，但区间包含零，不能称稳定额外贡献" if trained["net"] > 0 and trained["line_ci95_low"] <= 0 else (
            "支持训练修正有额外贡献" if trained["line_ci95_low"] > 0 else "本轮未支持稳定的正向额外贡献")
        lines.append(f"{policy}：D−C 净 {trained['net']:+d}，C−B 净 {path['net']:+d}；{certainty}。D−B 的变化不能全归训练修正，也不能据此说过去所有收益都是假的。")
    lines += ["", "公平计时后到底快多少？"]
    for policy in POLICIES:
        for arm in "BCD":
            r = totals[policy, arm]
            lines.append(f"{policy}/{arm}：相对 A {r['speed_ratio_vs_A']:.3f}×，纯生成时间减少 {100*r['time_reduction_fraction_vs_A']:.2f}%。")
    lines += ["", "让模型重新算，能不能明显救回来？"]
    for policy in POLICIES:
        ec, ed = comparisons[policy, "E-C"], comparisons[policy, "E-D"]
        clear = ec["line_ci95_low"] > 0 and ed["line_ci95_low"] > 0
        lines.append(f"{policy}：E−C 净 {ec['net']:+d}，E−D 净 {ed['net']:+d}；" + ("这两项行加权区间均在零以上，支持本轮覆盖右位仍有恢复空间。" if clear else "尚不能确认明显恢复；效果小或区间跨零时应保留不确定性。"))
    lines += ["", "|策略|方法|A正确/B错误人口救回|其他人口新增伤害（相对B）|", "|---|---|---|---|"]
    for r in paired:
        if "rescue_denominator" in r:
            lines.append(f"|{r['policy']}|{r['comparison'][0]}|{r['help']}/{r['rescue_denominator']}|{r['harm']}/{r['other_population_denominator']}|")
    lines += ["", "下一轮最值得改哪一件事，为什么？"]
    space = any(comparisons[p, "E-D"]["line_ci95_low"] > 0 and comparisons[p, "E-C"]["line_ci95_low"] > 0 for p in POLICIES)
    lines.append("优先改进头对提交左词后右位变化的拟合：真实重算在本轮覆盖位置给出额外恢复证据。下一轮需另行授权。" if space else "优先定位 A正确/B错误样本的首个分歧属于位置选择、结构动作、未覆盖触发还是此前错误积累；本轮重算证据不足以支持直接扩大头或验证算法，也不否定所有验证方法。")
    if any(totals[p, "D"]["speed_ratio_vs_A"] < 1.1 for p in POLICIES):
        lines.append("关闭诊断后 D 的速度优势至少在一种策略小于 10%；质量改善之外仍有推理成本问题。")
    lines += ["", "所有相邻机会、头调用、重算调用、结构停止、扩展/删除、非相邻回退分母分列于 summary.csv。"
              "普通提交计数与输出长度分开；总前向包含 E 的额外前向。bootstrap 按基础题整组抽样 10000 次，种子 20261013；不追加生成种子或调参。", "",
              f"实际设备/精度：{manifest.get('actual_device')} / {manifest['dtype']}。加载和预热在计时外，decode 前后 CUDA 同步；评测与写盘分开计时。GPU 背景记录见 gpu_background.jsonl。", "",
              f"代码 `{manifest['branch']}@{manifest['code_commit']}`，完整性 `{audit['status']}`；旧报告和被替代判断保留。", "",
              f"运行命令：\n\n```bash\n{manifest['command']}\n```", ""]
    (out / "report.zh.md").write_text("\n".join(lines))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    analyze(Path(args.result_dir).resolve())
