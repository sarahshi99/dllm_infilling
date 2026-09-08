#!/usr/bin/env python3
"""Server-side read-only join of the saved source audit to sampled training rows."""
import argparse
import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def run(args):
    root = Path(args.result_dir)
    groups = json.loads(gzip.decompress((root / "sensitivity_group_map.json.gz").read_bytes()))
    candidates = json.loads((root / "humaneval_overlap_candidates.json").read_text())
    connection = sqlite3.connect(f"file:{Path(args.bank_db).resolve()}?mode=ro", uri=True)
    records = defaultdict(Counter)
    components = defaultdict(Counter)
    counts = Counter()
    samples = defaultdict(set)
    for split, record, group, key in connection.execute("SELECT split,record_id,problem_group_id,sample_key FROM samples"):
        if group not in groups:
            raise ValueError("Bank group missing from pinned manifest audit")
        records[record][split] += 1
        components[groups[group]][split] += 1
        counts[split] += 1
        samples[split].add(key)
    connection.close()
    if dict(counts) != {"train": 200000, "validation": 20000, "external_test": 20000}:
        raise ValueError(f"Unexpected saved bank counts: {dict(counts)}")
    for split in ("validation", "external_test"):
        for kind in ("tv", "kl"):
            with gzip.open(root / f"{split}_diagnostics_{kind}.jsonl.gz", "rt") as f:
                keys = {json.loads(line)["sample_key"] for line in f}
            if keys != samples[split]:
                raise ValueError("Saved bank differs from evaluated sample IDs")
    matches = [{**candidate, "sampled_transitions": dict(records.get(candidate["record_id"], {}))}
               for candidate in candidates if candidate["record_id"] in records]
    strong_train = [r for r in matches if r["sampled_transitions"].get("train")
                    and any("ast_no_doc" in m["reason"] for m in r["matches"])]
    candidate_train = [r for r in matches if r["sampled_transitions"].get("train")]
    cross = [dict(component=k, sampled_transitions=dict(v)) for k, v in components.items() if len(v) > 1]
    output = {
        "status": "completed_read_only_membership_audit", "sample_counts": dict(counts),
        "evaluation_sample_keys_match_saved_diagnostics": True,
        "humaneval_sampled_candidates": matches,
        "humaneval_training_candidate_records": len(candidate_train),
        "humaneval_training_code_ast_match_records": len(strong_train),
        "cross_split_conservative_components": cross,
        "training_bank_modified": False,
        "requires_manual_review_before_humaneval": bool(candidate_train),
        "scope": "Confirms membership in the saved bank; actual run used all full train keys per training code. Single-assert matches require review; component overlap is conservative.",
    }
    (root / "actual_bank_membership_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in {"humaneval_sampled_candidates", "cross_split_conservative_components"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--bank-db", required=True)
    run(parser.parse_args())
