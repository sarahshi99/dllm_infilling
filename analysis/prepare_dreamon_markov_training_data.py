#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.dreamon_markov_head_training import (
    SPLIT_SEED,
    deduplicate_and_split_records,
    deterministic_line_split,
)
from experiments.lrdllm_dreamcoder_adapter import load_dataset_rows


DATASET_ID = "OpenCoder-LLM/opc-sft-stage2"
DATASET_CONFIG = "educational_instruct"
DATASET_REVISION = "7d28f40d579edd7c24402d17d0c7639f991e6f8d"
DATASET_LICENSE = "mit"
EXPECTED_SOURCE_ROWS = 118278
INITIAL_MASKS = 64
MAX_CONTEXT = 1024


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_code_block(response: str, fallback_code: str) -> tuple[str, str, str]:
    marker = "```python"
    start = response.find(marker)
    if start < 0:
        marker = "```"
        start = response.find(marker)
    if start < 0:
        return "", fallback_code, ""
    content_start = start + len(marker)
    if content_start < len(response) and response[content_start] == "\n":
        content_start += 1
    end = response.find("```", content_start)
    if end < 0:
        return response[:content_start], response[content_start:], ""
    return response[:content_start], response[content_start:end], response[end:]


def row_seed(record_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{SPLIT_SEED}:{record_id}".encode("utf-8")).digest()[:8], "big"
    ) % (2**31)


def tokenize_record(record: Mapping[str, Any], tokenizer: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response_prefix, code, response_suffix = extract_code_block(
            str(record["output"]), str(record["code"])
        )
        code_prefix, middle, code_suffix = deterministic_line_split(
            code, row_seed=row_seed(str(record["record_id"]))
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(record["instruction"])}],
            add_generation_prompt=True,
            tokenize=True,
        )
        response_prefix_ids = tokenizer.encode(
            response_prefix + code_prefix, add_special_tokens=False
        )
        middle_ids = tokenizer.encode(middle, add_special_tokens=False)
        response_suffix_ids = tokenizer.encode(
            code_suffix + response_suffix + tokenizer.eos_token,
            add_special_tokens=False,
        )
    except (ValueError, TypeError, IndexError) as error:
        return None, f"split_or_tokenize:{type(error).__name__}"
    structural = {
        int(tokenizer.mask_token_id),
        int(tokenizer.eos_token_id),
        int(getattr(tokenizer, "expand_token_id", 151667)),
    }
    ordinary_middle = [token for token in middle_ids if int(token) not in structural]
    if len(ordinary_middle) < 2:
        return None, "middle_fewer_than_two_ordinary_tokens"
    prefix_ids = [int(token) for token in prompt_ids + response_prefix_ids]
    suffix_ids = [int(token) for token in response_suffix_ids]
    full_reference_length = len(prefix_ids) + len(middle_ids) + len(suffix_ids)
    initial_canvas_length = len(prefix_ids) + INITIAL_MASKS + len(suffix_ids)
    if full_reference_length > MAX_CONTEXT or initial_canvas_length > MAX_CONTEXT:
        return None, "context_over_1024"
    return {
        "record_id": record["record_id"],
        "problem_group_id": record["problem_group_id"],
        "source_seq_id": record["source_seq_id"],
        "split": record["split"],
        "entry_point": record["entry_point"],
        "row_seed": row_seed(str(record["record_id"])),
        "prefix_ids": prefix_ids,
        "middle_ids": [int(token) for token in middle_ids],
        "suffix_ids": suffix_ids,
        "prefix_token_count": len(prefix_ids),
        "middle_token_count": len(middle_ids),
        "suffix_token_count": len(suffix_ids),
        "full_reference_length": full_reference_length,
        "initial_canvas_length": initial_canvas_length,
    }, None


def run(args: argparse.Namespace) -> int:
    result_dir = Path(args.result_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    if (result_dir / "split_manifest.jsonl.zst").exists() or (cache_dir / "prepared_records.jsonl.gz").exists():
        raise RuntimeError("Existing data are immutable; use new result/cache directories for corrected grouping.")
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        revision=DATASET_REVISION,
    )
    source_rows = [dict(dataset[index]) for index in range(len(dataset))]
    if len(source_rows) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"OpenCoder row count changed: {len(source_rows)} != {EXPECTED_SOURCE_ROWS}"
        )
    human_eval_rows = load_dataset_rows(
        Path(args.evaluator_root).resolve(), "HumanEval-SingleLineInfilling"
    )
    records, audit = deduplicate_and_split_records(
        source_rows,
        human_eval_rows=human_eval_rows,
        split_seed=int(args.split_seed),
    )
    exclusion_details = audit.pop("humaneval_exclusion_details")
    atomic_json(
        result_dir / "humaneval_exclusion_audit.json",
        {
            "matching_policy": "existing_conservative_text_ast_and_same_entry_assert_candidates",
            "excluded_groups": len(exclusion_details),
            "excluded_group_records": sum(len(row["member_record_ids"]) for row in exclusion_details),
            "groups": exclusion_details,
            "scope_note": "Conservative benchmark exclusion; candidate membership is not a claim of copying.",
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        Path(args.model_snapshot).resolve(),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.expand_token_id = int(getattr(tokenizer, "expand_token_id", 151667))
    model_config = AutoConfig.from_pretrained(
        Path(args.model_snapshot).resolve(), trust_remote_code=True, local_files_only=True
    )
    prepared: list[dict[str, Any]] = []
    excluded = Counter()
    for index, record in enumerate(records, start=1):
        item, reason = tokenize_record(record, tokenizer)
        if item is None:
            excluded[str(reason)] += 1
        else:
            prepared.append(item)
        if index % 10000 == 0:
            print(json.dumps({"prepared_progress": index, "kept": len(prepared)}), flush=True)
    prepared.sort(key=lambda row: (row["split"], row["problem_group_id"], row["record_id"]))
    split_counts = Counter(str(row["split"]) for row in prepared)
    group_splits: dict[str, set[str]] = {}
    for row in prepared:
        group_splits.setdefault(str(row["problem_group_id"]), set()).add(str(row["split"]))
    cross_split = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if cross_split:
        raise RuntimeError(f"problem groups cross splits: {len(cross_split)}")

    local_records = cache_dir / "prepared_records.jsonl.gz"
    temporary_gzip = local_records.with_suffix(local_records.suffix + ".tmp")
    with gzip.open(temporary_gzip, "wt", encoding="utf-8") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_gzip.replace(local_records)

    manifest_rows = [
        {
            "record_id": row["record_id"],
            "problem_group_id": row["problem_group_id"],
            "source_seq_id": row["source_seq_id"],
            "split": row["split"],
            "row_seed": row["row_seed"],
            "prefix_token_count": row["prefix_token_count"],
            "middle_token_count": row["middle_token_count"],
            "suffix_token_count": row["suffix_token_count"],
            "full_reference_length": row["full_reference_length"],
            "initial_canvas_length": row["initial_canvas_length"],
        }
        for row in prepared
    ]
    plain_manifest = cache_dir / "split_manifest.jsonl"
    write_jsonl(plain_manifest, manifest_rows)
    split_manifest = result_dir / "split_manifest.jsonl.zst"
    subprocess.run(
        ["zstd", "-q", "-f", "-10", str(plain_manifest), "-o", str(split_manifest)],
        check=True,
    )
    test_plain = cache_dir / "external_test_manifest.jsonl"
    write_jsonl(
        test_plain,
        (row for row in manifest_rows if row["split"] == "external_test"),
    )
    test_manifest = result_dir / "external_test_manifest.jsonl.zst"
    subprocess.run(
        ["zstd", "-q", "-f", "-10", str(test_plain), "-o", str(test_manifest)],
        check=True,
    )
    audit.update(
        {
            "tokenization_exclusions": dict(excluded),
            "tokenized_available_rows": len(prepared),
            "tokenized_split_counts": dict(split_counts),
            "cross_split_problem_groups": len(cross_split),
            "model_snapshot": str(Path(args.model_snapshot).resolve()),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_length": len(tokenizer),
            "config_vocab_size": int(model_config.vocab_size),
            "mask_token_id": int(tokenizer.mask_token_id),
            "expand_token_id": int(tokenizer.expand_token_id),
            "eos_token_id": int(tokenizer.eos_token_id),
        }
    )
    atomic_json(result_dir / "split_and_dedup_audit.json", audit)
    summary = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "fields": ["seq_id", "instruction", "output", "code", "entry_point", "testcase"],
        "advertised_rows": EXPECTED_SOURCE_ROWS,
        "actual_source_rows": len(source_rows),
        "available_rows_after_normalize_dedup_decontam_tokenize": len(prepared),
        "split_seed": int(args.split_seed),
        "split_counts": dict(split_counts),
        "split_manifest": split_manifest.name,
        "split_manifest_sha256": sha256(split_manifest),
        "external_test_manifest": test_manifest.name,
        "external_test_manifest_sha256": sha256(test_manifest),
        "external_test_frozen_before_training": True,
        "local_prepared_records": str(local_records),
        "local_prepared_records_size": local_records.stat().st_size,
        "human_eval_usage": "strict decontamination only; no labels, results, or tuning",
        "human_eval_exclusion_audit": "humaneval_exclusion_audit.json",
    }
    atomic_json(result_dir / "data_manifest_summary.json", summary)
    atomic_json(
        result_dir / "data_audit_samples.json",
        [
            {
                "record_id": row["record_id"],
                "problem_group_id": row["problem_group_id"],
                "split": row["split"],
                "token_counts": {
                    "prefix": row["prefix_token_count"],
                    "middle": row["middle_token_count"],
                    "suffix": row["suffix_token_count"],
                },
            }
            for row in prepared[:12]
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--evaluator-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
