#!/usr/bin/env python3
"""Independent arithmetic and an isolated portable rerun of existing K2 evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analysis.dreamon_k2_evidence import load_bundle, case_key

POLICIES = ("global_confidence", "left_to_right_frontier")
PAIRS = (("D", "C"), ("C", "B"), ("E", "C"), ("E", "D"),
         ("B", "A"), ("C", "A"), ("D", "A"), ("E", "A"), ("D", "B"))
COUNTS = ("forward_count", "base_forward_count", "recompute_forward_count", "outer_round_count",
          "normal_token_count", "expand_count", "delete_count", "adjacent_opportunity_count",
          "markov_enabled_count", "recompute_enabled_count", "structural_stop_count",
          "corrected_structural_stop_count", "nonadjacent_fallback_count")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_statistics(rows, left, right, *, replicates=10000):
    by_key = {(r["arm"], r["task_id"]): r for r in rows}
    groups = defaultdict(list)
    help_count = harm_count = 0
    for (arm, task_id), row in by_key.items():
        if arm != left:
            continue
        delta = int(row["passed"]) - int(by_key[right, task_id]["passed"])
        help_count += delta == 1
        harm_count += delta == -1
        groups[row["base_task_id"]].append(delta)
    ordered = sorted(groups)
    sizes = np.array([len(groups[g]) for g in ordered], dtype=np.int64)
    net = np.array([sum(groups[g]) for g in ordered], dtype=np.int64)
    # Reproduce the registered PCG64 integer cluster draws, lexicographic group
    # order and quantile rule. Each selected base task brings ALL its lines.
    rng = np.random.default_rng(20261013)
    draws = rng.integers(0, len(ordered), (replicates, len(ordered)))
    micro = net[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    macro = (net / sizes)[draws].mean(axis=1)
    result = {"comparison": f"{left}-{right}", "help": help_count, "harm": harm_count,
              "net": help_count - harm_count, "line_delta": sum(net) / sum(sizes),
              "base_task_equal_delta": statistics.fmean(sum(groups[g]) / len(groups[g]) for g in ordered),
              "cluster_count": len(ordered), "bootstrap_replicates": replicates, "bootstrap_seed": 20261013}
    for label, values in (("line", micro), ("base_task_equal", macro)):
        low, high = np.quantile(values, (.025, .975), method="linear")
        result[label + "_ci95_low"] = float(low)
        result[label + "_ci95_high"] = float(high)
    # Convert numpy scalars, preserving original float precision in CSV/JSON.
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in result.items()}


def independent_tables(rows):
    summaries, paired = [], []
    for policy in POLICIES:
        selected = [r for r in rows if r["policy"] == policy]
        local = []
        for arm in "ABCDE":
            group = [r for r in selected if r["arm"] == arm]
            by_base = defaultdict(list)
            for row in group:
                by_base[row["base_task_id"]].append(row["passed"])
            times = [r["generation_sec"] for r in group]
            result = {"policy": policy, "arm": arm, "passed": sum(r["passed"] for r in group),
                      "denominator": len(group), "pass_rate": statistics.fmean(r["passed"] for r in group),
                      "base_task_equal_pass_rate": statistics.fmean(statistics.fmean(v) for v in by_base.values()),
                      "generation_sec_total": math.fsum(times), "generation_sec_median": statistics.median(times),
                      "generation_sec_p95": float(np.quantile(times, .95, method="linear")),
                      "evaluator_sec_total": math.fsum(r["evaluator_sec"] for r in group),
                      "file_write_sec_total": math.fsum(r["file_write_sec"] for r in group),
                      "output_tokens_total": sum(r["output_tokens"] for r in group),
                      "output_tokens_mean": statistics.fmean(r["output_tokens"] for r in group),
                      "output_chars_total": sum(r["output_chars"] for r in group)}
            result.update({key: sum(r[key] for r in group) for key in COUNTS})
            result["ordinary_commits_per_total_forward"] = result["normal_token_count"] / result["forward_count"]
            local.append(result)
        for result in local:
            speed = local[0]["generation_sec_total"] / result["generation_sec_total"]
            result.update(speed_ratio_vs_A=speed, time_reduction_fraction_vs_A=1 - 1 / speed,
                          speed_interpretation="diagnostic_cost_only" if result["arm"] == "E" else "pure_generation")
        summaries.extend(local)
        for left, right in PAIRS:
            paired.append({"policy": policy, **paired_statistics(selected, left, right)})
        keyed = {(r["arm"], r["task_id"]): r for r in selected}
        ids = {r["task_id"] for r in selected}
        damaged = {t for t in ids if keyed["A", t]["passed"] and not keyed["B", t]["passed"]}
        for arm in "CDE":
            paired.append({"policy": policy, "comparison": f"{arm} rescue on A-pass/B-fail",
                "help": sum(keyed[arm, t]["passed"] for t in damaged),
                "harm": sum(keyed["B", t]["passed"] and not keyed[arm, t]["passed"] for t in ids - damaged),
                "rescue_denominator": len(damaged), "other_population_denominator": len(ids - damaged), "other_harm_vs": "B"})
    fields = list(dict.fromkeys(k for r in paired for k in r))
    paired = [{k: r.get(k, "") for k in fields} for r in paired]
    return summaries, paired


def compare_table(published, recomputed, keys, table):
    def indexed(rows):
        index = {tuple(str(r[k]) for k in keys): r for r in rows}
        if len(index) != len(rows):
            raise ValueError("duplicate comparison table key")
        return index
    old, new = indexed(published), indexed(recomputed)
    results = []
    for identity, original in old.items():
        current = new.get(identity, {})
        for field, expected in original.items():
            actual = current.get(field, "")
            difference = ""
            try:
                left, right = float(expected), float(actual)
                difference = right - left
                # Exact integer counts; float reduction order may differ at a few
                # ulps. CI endpoints also require the same UNROUNDED sign/zero.
                tolerance = 1e-14 * max(1, abs(left))
                equal = math.isfinite(right) and abs(difference) <= tolerance
                if "ci95" in field:
                    equal = equal and ((left > 0) - (left < 0) == (right > 0) - (right < 0))
                if left.is_integer() and field not in ("line_ci95_low", "line_ci95_high", "base_task_equal_ci95_low", "base_task_equal_ci95_high"):
                    equal = equal and left == right
            except (ValueError, TypeError):
                equal = str(expected) == str(actual)
            results.append({"table": table, "key": "|".join(identity), "field": field,
                            "published": expected, "recomputed": actual, "difference": difference,
                            "status": "matched" if equal else "DIFFERENT"})
    return results


def compare_report_tables(text, summaries, paired):
    groups = {(r["policy"], r["arm"]): r for r in summaries}
    pairs = {(r["policy"], r["comparison"]): r for r in paired}
    result = []
    for line in text.splitlines():
        cells = line.strip("|").split("|")
        if not cells or cells[0] not in POLICIES:
            continue
        policy = cells[0]
        if len(cells) == 10:
            r = groups[policy, cells[1]]
            expected = [policy, r["arm"], f"{r['passed']}/1033", f"{100*r['pass_rate']:.2f}%",
                        f"{r['generation_sec_total']:.2f}", f"{r['generation_sec_median']:.4f}/{r['generation_sec_p95']:.4f}",
                        str(r["output_tokens_total"]), f"{r['base_forward_count']}/{r['recompute_forward_count']}/{r['forward_count']}",
                        f"{r['ordinary_commits_per_total_forward']:.3f}", f"{r['speed_ratio_vs_A']:.3f}×"]
        elif len(cells) == 6:
            r = pairs[policy, cells[1]]
            expected = [policy, r["comparison"], f"{r['help']}/{r['harm']}", f"{r['net']:+d}",
                        f"{100*r['line_delta']:+.2f} [{100*r['line_ci95_low']:+.2f},{100*r['line_ci95_high']:+.2f}]",
                        f"{100*r['base_task_equal_delta']:+.2f} [{100*r['base_task_equal_ci95_low']:+.2f},{100*r['base_task_equal_ci95_high']:+.2f}]"]
        elif len(cells) == 4:
            r = pairs[policy, f"{cells[1]} rescue on A-pass/B-fail"]
            expected = [policy, cells[1], f"{r['help']}/{r['rescue_denominator']}", f"{r['harm']}/{r['other_population_denominator']}"]
        else:
            raise ValueError("unexpected published Markdown table shape")
        for index, (old, new) in enumerate(zip(cells, expected)):
            result.append({"table": "published_report_table", "key": policy + "|" + cells[1],
                           "field": str(index), "published": old, "recomputed": new, "difference": "",
                           "status": "matched" if old == new else "DIFFERENT"})
    if len(result) != 220:
        raise ValueError("published report table coverage differs from 10 group / 16 pair / 6 subset rows")
    return result


def isolated_rerun(result_dir, destination):
    with tempfile.TemporaryDirectory(prefix="k2_cpu_review_") as temp:
        root = Path(temp)
        code = root / "code/analysis"
        code.mkdir(parents=True)
        for name in ("analyze_dreamon_markov_k2_diagnosis.py", "dreamon_k2_evidence.py"):
            shutil.copyfile(REPO / "analysis" / name, code / name)
        bundle = root / "bundle"
        shutil.copytree(result_dir / "evidence", bundle)
        cwd = root / "unrelated_cwd"
        cwd.mkdir()
        script = code / "analyze_dreamon_markov_k2_diagnosis.py"
        wrapper = (
            "import builtins,runpy,sqlite3,sys; "
            "sqlite3.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('SQLite forbidden')); "
            "original=builtins.__import__; "
            "exec(\"def guarded(name,*a,**k):\\n if name.split('.')[0] in ('torch','transformers'): raise AssertionError('GPU libraries forbidden')\\n return original(name,*a,**k)\"); "
            "builtins.__import__=guarded; "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        command = [sys.executable, "-c", wrapper, str(script), "--evidence-dir", str(bundle),
                   "--output-dir", str(root / "output")]
        environment = {**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
        process = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError("isolated evidence reproduction failed: " + process.stderr)
        for name in ("summary.csv", "paired_comparisons.csv", "integrity_audit.json"):
            shutil.copyfile(root / "output" / name, destination / ("portable_" + name))
        return {"status": "passed", "copied": "two CPU analysis source files + evidence directory only",
                "cwd": "unrelated temporary directory", "sqlite_connect_forbidden": True,
                "torch_transformers_import_forbidden": True, "author_dataset_or_model_access": False,
                "temporary_directory_cleaned": True}


def run(result_dir, destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows, context, expected_ids, smoke, failures = load_bundle(result_dir / "evidence")
    observed = Counter(case_key(r) for r in rows)
    expected = {(p, a, t) for p in POLICIES for a in "ABCDE" for t in expected_ids}
    successful = set(observed)
    audit = {"row_count": len(rows), "unique_case_count": len(observed),
             "base_task_count": len({r["base_task_id"] for r in rows}),
             "missing_case_count": len(expected - successful), "unexpected_case_count": len(successful - expected),
             "duplicate_case_count": sum(n - 1 for n in observed.values()),
             "unrecovered_failure_count": len({case_key(r) for r in failures} - successful),
             "base_task_group_sizes": dict(sorted(Counter(r["base_task_id"] for r in rows if r["policy"] == POLICIES[0] and r["arm"] == "A").items()))}
    if (audit["row_count"], audit["base_task_count"], len(expected_ids)) != (10330, 164, 1033) or any(audit[k] for k in ("missing_case_count", "unexpected_case_count", "duplicate_case_count", "unrecovered_failure_count")):
        raise ValueError("independent integrity check failed")
    audit["isolated_rerun"] = isolated_rerun(result_dir, destination)
    summaries, paired = independent_tables(rows)
    write_csv(destination / "independent_summary.csv", summaries)
    write_csv(destination / "independent_paired.csv", paired)
    comparisons = []
    for filename, key, rebuilt in (("summary.csv", ("policy", "arm"), summaries),
                                   ("paired_comparisons.csv", ("policy", "comparison"), paired)):
        with (result_dir / filename).open() as handle:
            old = list(csv.DictReader(handle))
        comparisons += compare_table(old, rebuilt, key, filename + ":independent")
        with (destination / ("portable_" + filename)).open() as handle:
            portable = list(csv.DictReader(handle))
        comparisons += compare_table(old, portable, key, filename + ":isolated_portable")
    markdown = compare_report_tables((result_dir / "report.zh.md").read_text(), summaries, paired)
    write_csv(destination / "published_report_table_comparison.csv", markdown)
    comparisons.extend(markdown)
    write_csv(destination / "all_cell_comparison.csv", comparisons)
    differences = [r for r in comparisons if r["status"] != "matched"]
    audit.update(status="passed" if not differences else "differences_found", compared_cells=len(comparisons),
                 different_cells=len(differences), comparison_precision="full float, <=1e-14*max(1,abs(reference)); CI sign and exact zero checked separately",
                 bootstrap_seed=20261013, bootstrap_replicates=10000,
                 numpy_version=np.__version__, source_generation_seed=42)
    write_json(destination / "reproduction_audit.json", audit)
    evaluation = []
    timeouts = []
    reasons = []
    for policy in POLICIES:
        for arm in "ABCDE":
            group = [r for r in rows if r["policy"] == policy and r["arm"] == arm]
            count = Counter(r["failure_reason"] for r in group)
            durations = [r["evaluator_sec"] for r in group]
            evaluation.append({"policy": policy, "arm": arm, "records": len(group),
                "passed": sum(r["passed"] for r in group), "failed": sum(not r["passed"] for r in group),
                "timed_out": count["timed out"], "failed_message": sum(n for k, n in count.items() if k.startswith("failed:")),
                "unknown_result": sum(n for k, n in count.items() if k not in ("passed", "timed out") and not k.startswith("failed:")),
                "evaluator_sec_median": statistics.median(durations), "evaluator_sec_p95": float(np.quantile(durations, .95)),
                "evaluator_sec_max": max(durations)})
            reasons.extend({"policy": policy, "arm": arm, "failure_reason": k, "count": n} for k, n in sorted(count.items()))
    for row in rows:
        if row["failure_reason"] != "timed out":
            continue
        same = [r for r in rows if r["task_id"] == row["task_id"] and r["output"] == row["output"]]
        timeouts.append({k: row[k] for k in ("policy", "arm", "task_id", "base_task_id", "line_id", "evaluator_sec")}
                        | {"identical_text_records": len(same), "identical_text_passes": sum(r["passed"] for r in same)})
    write_csv(destination / "evaluation_groups.csv", evaluation)
    write_csv(destination / "evaluation_reasons.csv", reasons)
    write_csv(destination / "timeout_cases.csv", timeouts)
    timeout_counts = {(r["policy"], r["arm"]): r["timed_out"] for r in evaluation}
    sensitivity = []
    for r in paired:
        if " rescue " in r["comparison"]:
            continue
        left, right = r["comparison"].split("-")
        sensitivity.append({"policy": r["policy"], "comparison": r["comparison"], "observed_net": r["net"],
                            "net_lower_if_only_recorded_timeouts_misclassified": r["net"] - timeout_counts[r["policy"], right],
                            "net_upper_if_only_recorded_timeouts_misclassified": r["net"] + timeout_counts[r["policy"], left],
                            "interpretation": "algebraic uncertainty bounds only; original pass labels unchanged, no re-evaluation"})
    write_csv(destination / "timeout_sensitivity.csv", sensitivity)
    with (result_dir / "evidence/gpu_background.csv").open() as handle:
        gpu = list(csv.DictReader(handle))
    gpu_summary = {"samples": len(gpu), "first_sample_utc": gpu[0]["time_utc"], "last_sample_utc": gpu[-1]["time_utc"],
                   "other_task_count_range": [min(int(r["other_task_count"]) for r in gpu), max(int(r["other_task_count"]) for r in gpu)],
                   "all_samples_have_other_tasks": all(int(r["other_task_count"]) > 0 for r in gpu),
                   "utilization_pct_min_median_max": [min(float(r["utilization_gpu_pct"]) for r in gpu), statistics.median(float(r["utilization_gpu_pct"]) for r in gpu), max(float(r["utilization_gpu_pct"]) for r in gpu)],
                   "memory_used_mib_min_max": [min(float(r["memory_used_mib"]) for r in gpu), max(float(r["memory_used_mib"]) for r in gpu)],
                   "time_alignment_limit": "no case/evaluator UTC timestamps or host CPU load; row-boundary snapshots cannot establish timeout cause"}
    write_json(destination / "gpu_background_summary.json", gpu_summary)
    extra = []
    for policy in POLICIES:
        summary = {r["arm"]: r for r in summaries if r["policy"] == policy}
        net_gain = summary["D"]["passed"] - summary["B"]["passed"]
        net_loss = summary["A"]["passed"] - summary["B"]["passed"]
        extra.append({"policy": policy, "D_minus_B_net": net_gain, "A_minus_B_net": net_loss,
                      "parallel_net_loss_recovered_fraction": net_gain / net_loss,
                      "denominator_warning": "net pass gap, not A-pass/B-fail subset size"})
    write_csv(destination / "net_loss_recovery.csv", extra)
    print(json.dumps({k: v for k, v in audit.items() if k != "base_task_group_sizes"}, ensure_ascii=False))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.result_dir.resolve(), args.output_dir.resolve())["status"] == "passed" else 2)
