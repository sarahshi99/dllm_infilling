"""CPU-only Dream Base tokenizer audit; no model or head is constructed here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL_ID = "Dream-org/Dream-v0-Base-7B"
MODEL_REVISION = "6572adb5535263e4d1a337b56942ba48b6dee2a9"
REPRESENTATIVE_TEXTS = (
    "def f(x):\n    return x + 1\n",
    "    # 中文 café\n    s = 'Hello world'\n",
    "\treturn [i**2 for i in range(10)]\r\n",
    "Hello world", " Hello world", "e\u0301 ≠ é\n", "<|beginoftext|><|mask|><|endoftext|>",
)


def audit(source: Path, target: Path) -> dict:
    source_vocab = json.loads((source / "vocab.json").read_text())
    target_vocab = json.loads((target / "vocab.json").read_text())
    source_tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    source_slow = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True, use_fast=False)
    target_tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=True, local_files_only=True)
    source_map, target_map = source_tokenizer.get_vocab(), target_tokenizer.get_vocab()
    common = source_map.keys() & target_map.keys()
    common_mismatches = [token for token in common if source_map[token] != target_map[token]]
    examples = []
    for text in REPRESENTATIVE_TEXTS:
        left = source_tokenizer.encode(text, add_special_tokens=False)
        right = target_tokenizer.encode(text, add_special_tokens=False)
        examples.append({"text": text, "source_ids": left, "target_ids": right, "equal": left == right})
    rules = {name: (source / name).read_bytes() == (target / name).read_bytes()
             for name in ("merges.txt", "tokenization_dream.py")}
    runtime_rules = {
        "pretokenization_regex_equal": source_slow.pat.pattern == target_tokenizer.pat.pattern,
        "bpe_ranks_equal": source_slow.bpe_ranks == target_tokenizer.bpe_ranks,
        "byte_encoder_equal": source_slow.byte_encoder == target_tokenizer.byte_encoder,
        "byte_decoder_equal": source_slow.byte_decoder == target_tokenizer.byte_decoder,
        "errors_equal": source_slow.errors == target_tokenizer.errors,
    }
    if source_tokenizer.is_fast:
        backend = json.loads(source_tokenizer.backend_tokenizer.to_str())
        runtime_rules["source_fast_regex_equal"] = (
            backend["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == target_tokenizer.pat.pattern)
        runtime_rules["source_fast_nfc_matches_target_code"] = (
            backend["normalizer"] == {"type": "NFC"}
            and 'unicodedata.normalize("NFC", text)' in (target / "tokenization_dream.py").read_text())
    source_config = json.loads((source / "config.json").read_text())
    target_config = json.loads((target / "config.json").read_text())
    source_only = {token: source_map[token] for token in source_map.keys() - target_map.keys()}
    target_only = {token: target_map[token] for token in target_map.keys() - source_map.keys()}
    compatible = (source_vocab == target_vocab and not common_mismatches and all(rules.values())
                  and all(runtime_rules.values())
                  and all(row["equal"] for row in examples)
                  and source_only == {"<|expand|>": 151667} and not target_only
                  and source_config["vocab_size"] == target_config["vocab_size"])
    weight_index = json.loads((target / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(weight_index["weight_map"].values()))
    return {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "target_snapshot": str(target),
        "source_snapshot": str(source), "ordinary_vocab_mapping_equal": source_vocab == target_vocab,
        "ordinary_vocab_count": len(source_vocab), "common_mapping_count": len(common),
        "common_mapping_mismatch_count": len(common_mismatches), "rules_byte_equal": rules,
        "runtime_rules": runtime_rules,
        "tokenizer_classes": [type(source_tokenizer).__name__, type(target_tokenizer).__name__],
        "source_only_tokens": source_only, "target_only_tokens": target_only,
        "source_special_ids": source_tokenizer.all_special_ids,
        "target_special_ids": target_tokenizer.all_special_ids,
        "tokenizer_bos_ids": [source_tokenizer.bos_token_id, target_tokenizer.bos_token_id],
        "config_bos_ids": [source_config["bos_token_id"], target_config["bos_token_id"]],
        "model_output_widths": [source_config["vocab_size"], target_config["vocab_size"]],
        "representative_tokenizations": examples,
        "tokenizer_status": "compatible_for_fixed_canvas_with_original_head_mask" if compatible else "incompatible_stop",
        "output_column_semantics": {
            "151643": "EOS/pad in both; fixed canvas never deletes EOS",
            "151665": "tokenizer BOS in both; use tokenizer BOS, not stale config BOS",
            "151666": "mask in both",
            "151667": "source expand; target unassigned. Preserve original head output_mask zero, no expand action",
            "151668_to_152063": "unassigned output columns in both; invalid-token checks remain required",
        },
        "weight_shards": [{"name": name, "exists": (target / name).is_file()} for name in shard_names],
        "weights_complete": all((target / name).is_file() for name in shard_names),
        "gpu_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source.resolve(), args.target.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["tokenizer_status"])
    if result["tokenizer_status"] == "incompatible_stop":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
