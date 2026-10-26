"""Reproduce CPU evidence. Only tokenizer metadata and optional old head are read.

No model instantiation, backbone weights, datasets, GPU APIs, training or sampling
of HumanEval answers. A requested manifest hash is for cross-worktree handoff.
"""
import argparse
import hashlib
import io
import json
import os
import platform
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from .accounting import accounting
from .contract import HeadConfig
from .heads import ContextHead
from .test_cpu import fixture

SNAPSHOT = Path("/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194")
CHECKPOINT = Path("/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/checkpoints/kl/full/best.pt")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def verify_legacy(checkpoint):
    if not checkpoint.is_file():
        return dict(status="missing", path=str(checkpoint), action="no download attempted")
    # Trusted local repository-produced checkpoint. mmap avoids duplicating tables;
    # its optimizer tensors are mapped, never used or restored.
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    old = state["head"]
    v, r = old["embedding.weight"].shape
    special = tuple((old["output_mask"] == 0).nonzero().flatten().tolist())
    with torch.device("meta"):
        head = ContextHead("H0-static", HeadConfig(v, 7, special))
    head.load_legacy_static(state)
    batch = fixture(b=1, h=7, v=v)
    # Exclude structural tokens without needing to allocate another large model.
    batch = replace(batch, token_ids=torch.tensor([[3, 21, 80, 500]]))
    with torch.no_grad():
        actual = head(batch).delta_logits
        # Same GEMM row count: bitwise comparison of the literal legacy formula.
        previous = torch.cat([batch.token_ids[:, :1], batch.token_ids[:, :-1]], 1)
        expected = torch.nn.functional.linear(
            torch.nn.functional.embedding(previous, old["embedding.weight"]),
            old["output.weight"]) * old["output_mask"]
        expected[:, 0] = 0  # new interface leaves the fixed first anchor unchanged
        three_rows = torch.nn.functional.linear(
            torch.nn.functional.embedding(batch.token_ids[:, :-1], old["embedding.weight"]),
            old["output.weight"]) * old["output_mask"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 1:], three_rows, rtol=1e-5, atol=1e-6)
    return dict(status="passed", path=str(checkpoint), sha256=sha(checkpoint),
                compared_queries=3, V=v, rank=r, structural_mask_ids=list(special),
                max_absolute_difference=(actual-expected).abs().max().item(),
                different_GEMM_rows_max_absolute_difference=(actual[:, 1:]-three_rows).abs().max().item(),
                equality="same-shape CPU float32 legacy formula bitwise; differing GEMM rows atol=1e-6 rtol=1e-5; meta+assign+mmap, one legacy head",
                use="compatibility check only; no candidate initialized from trained weights")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"artifacts")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--legacy-checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("set CUDA_VISIBLE_DEVICES='' for this CPU-only verification")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    missing = []
    tokenizer = None
    if (args.snapshot/"tokenizer_config.json").is_file():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), trust_remote_code=True, local_files_only=True)
        model_config = json.loads((args.snapshot/"config.json").read_text())
        config = HeadConfig(model_config["vocab_size"], model_config["hidden_size"],
                            tuple(sorted(tokenizer.all_special_ids)), tokenizer_size=len(tokenizer))
        metadata = dict(snapshot=str(args.snapshot), revision=args.snapshot.name,
                        tokenizer_class=type(tokenizer).__name__, tokenizer_size=len(tokenizer),
                        tokenizer_base_vocab_size=tokenizer.vocab_size,
                        tokenizer_max_id=max(tokenizer.get_vocab().values()),
                        model_logit_V=config.vocab_size, hidden_H=config.hidden_size,
                        tokenizer_special_ids=list(config.special_token_ids),
                        files={f: sha(args.snapshot/f) for f in ("config.json", "tokenizer_config.json", "vocab.json", "added_tokens.json", "merges.txt")})
    else:
        missing.append("local tokenizer/config unavailable; dimensions below are declared estimates")
        config = HeadConfig(152064, 3584, (151643, 151665, 151666, 151667), tokenizer_size=151668)
        metadata = dict(status="missing", snapshot=str(args.snapshot))
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("experiments.context_heads_v1.test_cpu")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.output/"tests.log").write_text(stream.getvalue())
    legacy = verify_legacy(args.legacy_checkpoint)
    if legacy["status"] != "passed":
        missing.append("actual old checkpoint unavailable; synthetic mapping test only")
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)
    write(args.output/"accounting.json", accounting(config))
    if config.tokenizer_size is not None and config.tokenizer_size != config.vocab_size:
        # Explicit second support table: actual tokenizer vocabulary versus the
        # larger padded model logit dimension. No model is allocated for either.
        write(args.output/"accounting_tokenizer_support.json", accounting(
            replace(config, vocab_size=config.tokenizer_size)))
    write(args.output/"dreamon_metadata.json", metadata)
    write(args.output/"legacy_compatibility.json", legacy)
    write(args.output/"verification.json", dict(status="passed", tests_run=result.testsRun,
          failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
          python=sys.version, executable=sys.executable, torch=torch.__version__,
          transformers=__import__("transformers").__version__ if tokenizer is not None else None,
          platform=platform.platform(), cpu_threads=torch.get_num_threads(),
          CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
          gpu_used=False, training_started=False, backbone_loaded=False, human_eval_answers_generated=False,
          missing=missing, seeds=dict(head=20261015, fixture=61, deterministic_test_seeds=[17,43,71]),
          tolerance="float32: exact zeros/masks/legacy formulas and token equality; step-vs-batch atol=1e-6 rtol=1e-5; attention sums atol=rtol=1e-6. No near-tie equivalence claim across kernels/hardware."))
    print(f"passed {result.testsRun} CPU tests; legacy={legacy['status']}; artifacts={args.output}")


if __name__ == "__main__":
    main()
