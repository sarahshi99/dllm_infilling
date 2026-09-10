#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sqlite3
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.dreamon_markov_head_training import freeze_module
from experiments.dreamon_singleline_adapter import (
    CountingModel,
    OfficialHFTokenizerWrapper,
    source_provenance,
)
from experiments.dreamon_singleline_order_parallel import (
    GLOBAL_CONFIDENCE,
    LEFT_TO_RIGHT_FRONTIER,
    decode_with_policy,
)
from experiments.run_dreamon_markov_premise import load_generator_module


POLICIES = (GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER)
STAGES = ("early", "middle", "late")
TARGETS = {"train": 200000, "validation": 20000, "external_test": 20000}
MODEL_REVISION = "8ccc74750e43177327f29dab9e91882ba759e194"
TOKENIZER_REVISION = MODEL_REVISION
SOURCE_REVISION = "8a0a54918412eda9402a327646f7f067f7160ec8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def capacities(total: int) -> dict[tuple[str, str], int]:
    per_policy = total // 2
    base, remainder = divmod(per_policy, len(STAGES))
    output = {}
    for policy in POLICIES:
        for index, stage in enumerate(STAGES):
            output[(policy, stage)] = base + (1 if index < remainder else 0)
    return output


class ReservoirBank:
    def __init__(self, path: Path, *, per_problem_cap: int, oversample: float) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples (
                split TEXT NOT NULL,
                policy TEXT NOT NULL,
                stage TEXT NOT NULL,
                slot INTEGER NOT NULL,
                sample_key TEXT NOT NULL UNIQUE,
                problem_group_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                payload BLOB NOT NULL,
                PRIMARY KEY (split, policy, stage, slot)
            );
            CREATE TABLE IF NOT EXISTS counters (
                split TEXT NOT NULL,
                policy TEXT NOT NULL,
                stage TEXT NOT NULL,
                seen INTEGER NOT NULL,
                PRIMARY KEY (split, policy, stage)
            );
            CREATE TABLE IF NOT EXISTS progress (
                split TEXT PRIMARY KEY,
                next_record_index INTEGER NOT NULL
            );
            """
        )
        sample_count = int(self.connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
        progress_count = int(self.connection.execute("SELECT COUNT(*) FROM progress").fetchone()[0])
        if sample_count and not progress_count:
            raise RuntimeError(
                "existing partial bank has no exact-resume progress; use a fresh versioned database"
            )
        self.per_problem_cap = int(per_problem_cap)
        self.oversample = float(oversample)
        self.seen: dict[tuple[str, str, str], int] = defaultdict(int)
        self.selected_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        for split, policy, stage, seen in self.connection.execute(
            "SELECT split, policy, stage, seen FROM counters"
        ):
            self.seen[(split, policy, stage)] = int(seen)
        for split, policy, problem, count in self.connection.execute(
            "SELECT split, policy, problem_group_id, COUNT(*) FROM samples GROUP BY split, policy, problem_group_id"
        ):
            self.selected_counts[(split, policy, problem)] = int(count)

    def _deterministic_slot(self, sample_key: str, seen: int) -> int:
        value = int.from_bytes(hashlib.sha256(sample_key.encode("utf-8")).digest()[:8], "big")
        return value % seen

    def add(self, row: Mapping[str, Any], capacity: int) -> bool:
        split = str(row["split"])
        policy = str(row["trajectory_policy"])
        stage = str(row["generation_stage"])
        stratum = (split, policy, stage)
        self.seen[stratum] += 1
        seen = self.seen[stratum]
        current = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM samples WHERE split=? AND policy=? AND stage=?",
                stratum,
            ).fetchone()[0]
        )
        problem_key = (split, policy, str(row["problem_group_id"]))
        sample_key = str(row["sample_key"])
        if current < capacity:
            if self.selected_counts[problem_key] >= self.per_problem_cap:
                return False
            slot = current
            old_problem = None
        else:
            slot = self._deterministic_slot(sample_key, seen)
            if slot >= capacity:
                return False
            old = self.connection.execute(
                "SELECT problem_group_id FROM samples WHERE split=? AND policy=? AND stage=? AND slot=?",
                (split, policy, stage, slot),
            ).fetchone()
            old_problem = str(old[0]) if old else None
            if old_problem != problem_key[2] and self.selected_counts[problem_key] >= self.per_problem_cap:
                return False
        payload = zlib.compress(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            ),
            level=6,
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO samples(split,policy,stage,slot,sample_key,problem_group_id,record_id,payload) VALUES(?,?,?,?,?,?,?,?)",
            (
                split,
                policy,
                stage,
                slot,
                sample_key,
                problem_key[2],
                str(row["record_id"]),
                payload,
            ),
        )
        if old_problem is not None and old_problem != problem_key[2]:
            self.selected_counts[(split, policy, old_problem)] -= 1
        if old_problem != problem_key[2]:
            self.selected_counts[problem_key] += 1
        return True

    def checkpoint(self, progress_split: str, next_record_index: int) -> None:
        for (split, policy, stage), seen in self.seen.items():
            self.connection.execute(
                "INSERT OR REPLACE INTO counters(split,policy,stage,seen) VALUES(?,?,?,?)",
                (split, policy, stage, int(seen)),
            )
        self.connection.execute(
            "INSERT OR REPLACE INTO progress(split,next_record_index) VALUES(?,?)",
            (str(progress_split), int(next_record_index)),
        )
        self.connection.commit()

    def next_record_index(self, split: str) -> int:
        row = self.connection.execute(
            "SELECT next_record_index FROM progress WHERE split=?", (str(split),)
        ).fetchone()
        return int(row[0]) if row else 0

    def complete(self, split: str, split_capacities: Mapping[tuple[str, str], int]) -> bool:
        for (policy, stage), capacity in split_capacities.items():
            selected = int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM samples WHERE split=? AND policy=? AND stage=?",
                    (split, policy, stage),
                ).fetchone()[0]
            )
            seen = self.seen[(split, policy, stage)]
            if selected < capacity or seen < math.ceil(capacity * self.oversample):
                return False
        return True

    def counts(self) -> list[dict[str, Any]]:
        rows = []
        for split, policy, stage, selected in self.connection.execute(
            "SELECT split,policy,stage,COUNT(*) FROM samples GROUP BY split,policy,stage ORDER BY split,policy,stage"
        ):
            rows.append(
                {
                    "split": split,
                    "policy": policy,
                    "generation_stage": stage,
                    "selected": int(selected),
                    "seen": self.seen[(split, policy, stage)],
                }
            )
        return rows

    def close(self) -> None:
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def load_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[str(row["split"])].append(row)
    for split, rows in output.items():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"20260901:{split}:{row['record_id']}".encode("utf-8")
            ).hexdigest()
        )
    return output


def sample_key(row: Mapping[str, Any], transition: Mapping[str, Any]) -> str:
    material = (
        f"{row['split']}:{transition['trajectory_policy']}:{row['record_id']}:"
        f"{transition['source_step_index']}:{transition['target_position']}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def run(args: argparse.Namespace) -> int:
    started = time.time()
    records_by_split = load_records(Path(args.prepared_records).resolve())
    source_root = Path(args.source_root).resolve()
    snapshot = Path(args.model_snapshot).resolve()
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    tokenizer_base = AutoTokenizer.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    tokenizer_base.expand_token_id = int(config.expand_token_id)
    maximum_token_id = max(tokenizer_base.get_vocab().values())
    if int(config.vocab_size) != 152064:
        raise RuntimeError(f"unexpected DreamOn config vocabulary: {config.vocab_size}")
    if maximum_token_id >= int(config.vocab_size):
        raise RuntimeError("tokenizer emits an id outside the model vocabulary")
    tokenizer = OfficialHFTokenizerWrapper(tokenizer_base)
    source_module = load_generator_module(source_root)
    base_model = AutoModel.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    freeze_module(base_model)
    model = CountingModel(base_model).eval()
    versions_before = [parameter._version for parameter in base_model.parameters()]
    database = Path(args.bank_db).resolve()
    bank = ReservoirBank(
        database,
        per_problem_cap=int(args.per_problem_cap),
        oversample=float(args.oversample),
    )
    processed = Counter()
    exclusions = Counter()
    candidate_counts = Counter()
    selected_writes = 0
    targets = {
        "train": int(args.train_target),
        "validation": int(args.validation_target),
        "external_test": int(args.external_test_target),
    }
    if any(target <= 0 or target % 2 for target in targets.values()):
        raise ValueError("all split targets must be positive even numbers")
    split_caps = {split: capacities(target) for split, target in targets.items()}
    for split in ("train", "validation", "external_test"):
        start_index = bank.next_record_index(split)
        processed[split] = start_index
        for record_index, record in enumerate(
            records_by_split[split][start_index:], start=start_index
        ):
            if bank.complete(split, split_caps[split]):
                break
            processed[split] += 1
            for policy in POLICIES:
                torch.manual_seed(int(record["row_seed"]))
                torch.cuda.manual_seed_all(int(record["row_seed"]))
                decoded = decode_with_policy(
                    model=model,
                    tokenizer=tokenizer,
                    sample_tokens_fn=source_module.sample_tokens,
                    prefix_ids=record["prefix_ids"],
                    suffix_ids=record["suffix_ids"],
                    reference_ids=record["middle_ids"],
                    min_gen_len=64,
                    max_gen_len=64,
                    max_tokens=1024,
                    steps=256,
                    expand_budget=64,
                    temperature=0.2,
                    top_p=0.9,
                    top_k=None,
                    requested_k=1,
                    policy=policy,
                    device=str(args.device),
                )
                for reason, count in decoded["replay_exclusion_counts"].items():
                    exclusions[(split, policy, reason)] += int(count)
                for transition in decoded["replay_transitions"]:
                    if int(transition["target_position"]) != int(transition["target_local_position"]) + len(
                        record["prefix_ids"]
                    ):
                        raise RuntimeError("target coordinate mismatch")
                    if len(transition["stale_input_ids"]) != len(
                        transition["fresh_input_ids"]
                    ):
                        raise RuntimeError("structural length change entered replay bank")
                    if transition["fresh_input_ids"][transition["target_position"]] != tokenizer.mask_id:
                        raise RuntimeError("fresh target is not an active mask")
                    row = {
                        **transition,
                        "sample_key": sample_key(record, transition),
                        "record_id": record["record_id"],
                        "problem_group_id": record["problem_group_id"],
                        "source_seq_id": record["source_seq_id"],
                        "split": split,
                        "row_seed": record["row_seed"],
                        "model_revision": MODEL_REVISION,
                        "tokenizer_revision": TOKENIZER_REVISION,
                        "source_revision": SOURCE_REVISION,
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "offset": 1,
                    }
                    candidate_counts[(split, policy, transition["generation_stage"])] += 1
                    capacity = split_caps[split][(policy, transition["generation_stage"])]
                    selected_writes += int(bank.add(row, capacity))
            bank.checkpoint(split, record_index + 1)
            if processed[split] % 100 == 0:
                print(
                    json.dumps(
                        {
                            "split": split,
                            "records_processed": processed[split],
                            "model_forwards": model.forward_calls,
                            "selected_writes": selected_writes,
                            "strata": bank.counts(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if not bank.complete(split, split_caps[split]):
            print(json.dumps({"split_shortfall": split, "strata": bank.counts()}), flush=True)
    bank.close()
    model_forwards = model.forward_calls
    versions_after = [parameter._version for parameter in base_model.parameters()]
    if versions_before != versions_after:
        raise RuntimeError("DreamOn parameter version changed during bank generation")
    del model, base_model
    torch.cuda.empty_cache()

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    selected_counts = [
        {
            "split": split,
            "policy": policy,
            "generation_stage": stage,
            "count": int(count),
        }
        for split, policy, stage, count in connection.execute(
            "SELECT split,policy,stage,COUNT(*) FROM samples GROUP BY split,policy,stage ORDER BY split,policy,stage"
        )
    ]
    problem_contribution = [
        int(row[0])
        for row in connection.execute(
            "SELECT COUNT(*) FROM samples GROUP BY split,policy,problem_group_id"
        )
    ]
    samples = []
    for (payload,) in connection.execute(
        "SELECT payload FROM samples ORDER BY split,policy,stage,slot LIMIT 12"
    ):
        row = json.loads(zlib.decompress(payload))
        samples.append(
            {
                key: row[key]
                for key in (
                    "sample_key",
                    "record_id",
                    "problem_group_id",
                    "split",
                    "trajectory_policy",
                    "generation_stage",
                    "source_step_index",
                    "previous_token_id",
                    "target_position",
                    "reference_token_id",
                    "reference_prefix_aligned_through_predecessor",
                    "active_mask_count",
                    "canvas_length",
                )
            }
        )
    total_selected = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    connection.close()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "completed" if total_selected == sum(targets.values()) else "shortfall",
        "targets": targets,
        "total_selected": total_selected,
        "selected_counts": selected_counts,
        "candidate_counts": [
            {
                "split": key[0],
                "policy": key[1],
                "generation_stage": key[2],
                "count": value,
            }
            for key, value in sorted(candidate_counts.items())
        ],
        "records_processed": dict(processed),
        "exclusion_counts": [
            {"split": key[0], "policy": key[1], "reason": key[2], "count": value}
            for key, value in sorted(exclusions.items())
        ],
        "per_problem_cap": int(args.per_problem_cap),
        "maximum_selected_per_problem_policy": max(problem_contribution, default=0),
        "oversample_factor": float(args.oversample),
        "bank_db": str(database),
        "bank_db_size": database.stat().st_size,
        "bank_db_sha256": sha256(database),
        "no_full_vocabulary_logits_serialized": True,
        "dreamon_requires_grad_false": True,
        "dreamon_parameter_versions_unchanged": True,
        "model_forwards": model_forwards,
        "elapsed_seconds": time.time() - started,
        "vocabulary": {
            "config_vocab_size": int(config.vocab_size),
            "tokenizer_length": len(tokenizer_base),
            "tokenizer_max_id": maximum_token_id,
            "reserved_model_rows": int(config.vocab_size) - len(tokenizer_base),
            "mask_token_id": int(config.mask_token_id),
            "expand_token_id": int(config.expand_token_id),
            "eos_token_id": int(config.eos_token_id),
            "expected_rank256_parameter_count": int(config.vocab_size) * 256 * 2,
        },
        "source_provenance": source_provenance(source_root),
    }
    atomic_json(result_dir / "transition_bank_summary.json", summary)
    atomic_json(result_dir / "transition_bank_audit_samples.json", samples)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if summary["status"] == "completed" else 2


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-records", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--bank-db", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-problem-cap", type=int, default=16)
    parser.add_argument("--oversample", type=float, default=1.25)
    parser.add_argument("--train-target", type=int, default=TARGETS["train"])
    parser.add_argument("--validation-target", type=int, default=TARGETS["validation"])
    parser.add_argument("--external-test-target", type=int, default=TARGETS["external_test"])
    return parser


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
