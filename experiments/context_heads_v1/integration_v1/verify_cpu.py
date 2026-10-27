"""Bounded CPU interop audit. Reads only existing contract fields from <=12 states.

No backbone weights, answer/test/pass fields, teacher arrays, training or decode.
All output is new; the first-batch B manifest/artifacts/ready are preserved.
"""
import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from ..contract import HeadBatch, validate
from .adapter import config_from_a_schema

MODULE = Path(__file__).resolve().parent
ROOT = MODULE.parents[2]
FIELDS = ("old_hidden", "position_ids", "chain_ids", "selected_mask", "query_indices",
          "old_logits", "token_ids", "token_state", "history_mask")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def check_original_payload():
    manifest_path = MODULE.parent/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["data"]:
        if sha(ROOT/entry["path"]) != entry["sha256"]:
            raise RuntimeError("first-batch B payload changed: "+entry["path"])
    return dict(manifest_sha256=sha(manifest_path), entries_unchanged=len(manifest["data"]),
                source_commit="ec8b159ce276c7a919a924c1b6991ffb64a5b8bf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-root", type=Path, default=ROOT.parent/"dreamon-context-foundation-20261015-v1")
    parser.add_argument("--output", type=Path, default=MODULE/"artifacts")
    parser.add_argument("--state-limit", type=int, default=12)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("set CUDA_VISIBLE_DEVICES='' for CPU-only verification")
    if not 1 <= args.state_limit <= 12:
        raise ValueError("this technical audit is bounded to 1..12 existing states")
    torch.set_num_threads(1)
    original = check_original_payload()
    common = Path(git(ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"))/"research_runs/dreamon_context_20261015_v1"
    b_ready_hash = sha(common/"B_done.json")
    a_root = args.a_root.resolve()
    a_output = a_root/"analysis_outputs/dreamon_context_foundation_20261015_v1"
    a_commit = git(a_root, "rev-parse", "HEAD")
    adapter_path = a_root/"experiments/context_foundation/adapter.py"
    schema_path = a_output/"protocol_schema.json"
    # Pin exactly the published source. No merge or file writes in A's worktree.
    for path in (adapter_path, schema_path):
        committed = subprocess.check_output(["git", "show", a_commit+":"+str(path.relative_to(a_root))], cwd=a_root)
        if hashlib.sha256(committed).hexdigest() != sha(path):
            raise RuntimeError("A source has local edits; choose the intended dependency explicitly")
    dependency = "experiments/dreamon_singleline_order_parallel.py"
    if sha(a_root/dependency) != sha(ROOT/dependency):
        raise RuntimeError("A sampler import's decoder dependency differs from B's base")
    a_ready = json.loads((common/"A_done.json").read_text())
    if a_ready["source_commit"] != a_commit or sha(Path(a_ready["manifest_absolute_path"])) != a_ready["manifest_sha256"]:
        raise RuntimeError("A source does not match its published ready")
    schema = json.loads(schema_path.read_text())
    # Use previously verified local tokenizer metadata; do not load/download one.
    metadata = json.loads((MODULE.parent/"artifacts/dreamon_metadata.json").read_text())
    config = config_from_a_schema(schema, tokenizer_size=metadata["tokenizer_size"])
    index = json.loads((a_output/"tensor_manifest.json").read_text())
    runtime = json.loads((a_output/"runtime_paths.local.json").read_text())
    cache = Path(runtime["cache_root"])
    entries = [e for e in index["data"] if e["path"].startswith("states/")][:args.state_limit]
    if len(entries) != args.state_limit:
        raise RuntimeError("not enough existing committed state packages")
    states = []
    for entry in entries:
        path = cache/entry["path"]
        if sha(path) != entry["sha256"]:
            raise RuntimeError("A cached state differs from published index")
        # Lazy NPZ access: intentionally never read x0, teacher, answers or labels.
        with np.load(path, allow_pickle=False) as package:
            fields = {name: torch.from_numpy(package[name]) for name in FIELDS}
        batch = HeadBatch(**fields)
        validate(batch, config)
        states.append(dict(state_id=entry["state_id"], sha256=entry["sha256"],
                           shapes={name: list(value.shape) for name, value in fields.items()},
                           dtypes={name: str(value.dtype) for name, value in fields.items()},
                           history_lengths=batch.history_mask.sum(-1).tolist(),
                           validation="passed; committed ordinary prefix; no hidden alignment transform"))
    os.environ["CONTEXT_HEADS_A_ADAPTER"] = str(adapter_path)
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromName("experiments.context_heads_v1.test_cpu"),
        unittest.defaultTestLoader.loadTestsFromName("experiments.context_heads_v1.integration_v1.test_cpu")])
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"tests.txt").write_text(log.getvalue())
    if not result.wasSuccessful() or result.skipped:
        print(log.getvalue())
        raise SystemExit("interop verification incomplete")
    write(args.output/"state_contract_checks.json", dict(checked=len(states), fields_read=list(FIELDS), states=states))
    evidence = dict(status="passed", tests_run=result.testsRun, failures=len(result.failures),
                    errors=len(result.errors), skipped=len(result.skipped), original_payload=original,
                    B_done_sha256_unchanged=b_ready_hash, a_commit=a_commit, a_status=a_ready["status"],
                    a_adapter_sha256=sha(adapter_path), a_schema_sha256=sha(schema_path),
                    a_manifest_sha256=a_ready["manifest_sha256"],
                    a_H=config.hidden_size, a_V=config.vocab_size, tokenizer_size=config.tokenizer_size,
                    a_special_count=len(config.special_token_ids),
                    padded_rows_excluded=dict(start=config.tokenizer_size, stop_exclusive=config.vocab_size),
                    dependency_sha256=sha(ROOT/dependency), existing_states_checked=len(states),
                    actual_A_functions_tested=["sample_key", "uniform", "distribution", "inverse_cdf",
                                               "FrozenHeadAdapter.forward", "Engine.correction"],
                    actual_A_sampler=dict(temperature=.2, top_p=.9, fixed_u=True, ordinary_support_only=True),
                    not_tested=["Engine.__init__/decode or any real generation", "GPU execution/performance",
                                "provisional expand/EOS scheduling and pre-barrier sequential equivalence",
                                "training, teacher quality or answer correctness"],
                    gpu_used=False, backbone_loaded=False, training_started=False,
                    human_eval_answers_generated=False, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    platform=platform.platform(), cpu_threads=torch.get_num_threads(),
                    tolerance="same-shape adapter formulas/zero/token results exact; inherited CPU float32 rules retained")
    write(args.output/"verification.json", evidence)
    if sha(common/"B_done.json") != b_ready_hash:
        raise RuntimeError("B_done changed during audit")
    check_original_payload()
    print(f"passed {result.testsRun} CPU tests; validated {len(states)} existing A states; original B payload preserved")


if __name__ == "__main__":
    main()
