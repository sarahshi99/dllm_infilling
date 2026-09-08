#!/usr/bin/env python3
"""Rebuild tables from EXISTING diagnostics, without loading a model/test bank."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.markov_head_metrics import no_head_summary, summarize


def read_json(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def extended(rows):
    result = summarize(rows, bootstrap_reps=10000)
    aligned = [r for r in rows if r["reference_aligned"]]
    for prefix in ("baseline", "head"):
        values = [r[prefix + "_reference_rank"] for r in aligned if prefix + "_reference_rank" in r]
        result[prefix + "_reference_rank"] = float(np.mean(values)) if values else None
    for action in ("enter", "exit"):
        values = [r["top_p_support_" + action + "_count"] for r in rows if "top_p_support_" + action + "_count" in r]
        result["top_p_support_" + action + "_count_mean"] = float(np.mean(values)) if values else None
    result["recovered_count"] = sum(r["mismatch_recovery"] for r in rows)
    result["corrupted_count"] = sum(r["stable_corruption"] for r in rows)
    return result


def comparison(rows_by_head, statuses, *, include_subgroups):
    output = []
    scopes = [("overall", lambda r: True)]
    if include_subgroups:
        scopes += [(p, lambda r, p=p: r["trajectory_policy"] == p)
                   for p in ("confidence_global", "left_to_right_frontier")]
    for scope, select in scopes:
        for aligned_only in ((False, True) if include_subgroups else (False,)):
            name = scope + ("_aligned" if aligned_only else "")
            for kind in ("tv", "kl"):
                subset = [r for r in rows_by_head[kind] if select(r) and (not aligned_only or r["reference_aligned"])]
                if not subset:
                    continue
                summary = extended(subset)
                if kind == "tv":
                    baseline = no_head_summary(summary)
                    baseline.update(recovered_count=0, corrupted_count=0)
                    output.append({"method": "no_head", "scope": name, "lambda": 0.0, **baseline})
                output.append({"method": kind + "_head", "scope": name,
                               "lambda": statuses[kind]["chosen_lambda"], **summary})
    return output


def run(args):
    root = Path(args.result_dir)
    excluded = set(read_json(root / "sensitivity_excluded_problem_groups.json"))
    group_map = read_json(root / "sensitivity_group_map.json.gz")
    statuses = {k: read_json(root / (k + "_training_status.json")) for k in ("tv", "kl")}
    audit = {"status": "completed_with_protocol_deviations", "reanalysis_only": True,
             "models_or_checkpoints_loaded": False, "new_test_inference_count": 0,
             "historically_reported_test_open_count": 1,
             "lifetime_test_open_count_independently_verified": False, "diagnostic_files": {}}
    sensitivity_rows = []
    phase_rows = []
    for phase, prefix, expected in (("validation", "validation", 20000), ("external_test", "external_test", 20000)):
        rows_by_head = {}
        for kind in ("tv", "kl"):
            path = root / f"{prefix}_diagnostics_{kind}.jsonl.gz"
            with gzip.open(path, "rt") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            keys = [r["sample_key"] for r in rows]
            if len(rows) != expected or len(set(keys)) != expected:
                raise ValueError(f"wrong count/duplicate: {path.name}")
            for row in rows:
                if not np.isclose(row["tv_improvement"], row["baseline_tv"] - row["head_tv"], atol=1e-7):
                    raise ValueError("inconsistent per-row TV")
                if bool(row["mismatch_recovery"]) != (not row["baseline_top1_agreement"] and row["head_top1_agreement"]):
                    raise ValueError("inconsistent recovery")
                if bool(row["stable_corruption"]) != (row["baseline_top1_agreement"] and not row["head_top1_agreement"]):
                    raise ValueError("inconsistent corruption")
            rows_by_head[kind] = rows
            audit["diagnostic_files"][path.name] = {"rows": len(rows), "unique_keys": len(set(keys))}
        a = {r["sample_key"]: r for r in rows_by_head["tv"]}
        b = {r["sample_key"]: r for r in rows_by_head["kl"]}
        if a.keys() != b.keys() or any(a[k]["baseline_tv"] != b[k]["baseline_tv"] or a[k]["problem_group_id"] != b[k]["problem_group_id"] for k in a):
            raise ValueError("heads not evaluated on identical records/baselines")
        if phase == "external_test":
            write_csv(root / "external_test_comparison.csv", comparison(rows_by_head, statuses, include_subgroups=True))
        clean = {kind: [{**r, "problem_group_id": group_map[r["problem_group_id"]]}
                        for r in rows if r["problem_group_id"] not in excluded]
                 for kind, rows in rows_by_head.items()}
        for row in comparison(clean, statuses, include_subgroups=False):
            sensitivity_rows.append({"phase": phase, "evidence_type": "post_hoc_conservative_sensitivity", **row})
        audit[phase + "_retained_transitions"] = len(clean["tv"])
        for kind, rows in rows_by_head.items():
            for stage in ("early", "middle", "late"):
                subset = [r for r in rows if r["generation_stage"] == stage]
                phase_rows.append({"phase": phase, "head": kind, "stage": stage, **extended(subset)})
    write_csv(root / "sensitivity_comparison.csv", sensitivity_rows)
    write_csv(root / "generation_stage_comparison.csv", phase_rows)
    required = ["tv_training_status.json", "kl_training_status.json", "external_test_comparison.csv",
                "validation_comparison.csv", "split_manifest.jsonl.zst", "data_overlap_audit.json",
                "sensitivity_comparison.csv", "transition_bank_summary.json"]
    audit.update({"missing_required_files": [n for n in required if not (root / n).is_file()],
                  "tv_status_present": (root / "tv_training_status.json").is_file(),
                  "kl_status_present": (root / "kl_training_status.json").is_file(),
                  "winner": "kl", "winner_basis": "original validation only; sensitivity did not reselect",
                  "data_isolation_status": "cross_split_source_overlap_confirmed; actual bank membership pending",
                  "loss_normalization_of_existing_weights": "microbatch_aligned_mean_v1",
                  "corrected_training_code_retrained": False})
    write_json(root / "completeness_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    run(p.parse_args())
