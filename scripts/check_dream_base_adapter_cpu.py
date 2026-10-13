"""Check adapter vs pinned native Base source with a small deterministic CPU model."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load("experiments.dreamcoder_fixed_transfer_adapter", args.common_adapter)
    from experiments import dream_base_zero_training_adapter as adapter
    native = load("dream_base_pinned_generation_cpu", Path(adapter.PROFILE["local_path"]) / "generation_utils.py")

    class FakeModel(torch.nn.Module, native.DreamGenerationMixin):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.device = torch.device("cpu")

        def forward(self, x, attention_mask, tok_idx):
            position = torch.arange(x.shape[1]).float()[None, :, None]
            token = torch.arange(16).float()[None, None, :]
            logits = torch.sin(position * 0.17 + token * 0.41 + x.sum() * 0.013) * 2
            logits[..., :3] = -100.0
            return SimpleNamespace(logits=logits)

    class FakeTokenizer:
        mask_token_id, bos_token_id, eos_token_id, pad_token_id = 2, 0, 1, 1
        all_special_ids = [0, 1, 2]
        eos_token = pad_token = "<eos>"

        def get_vocab(self):
            return {str(i): i for i in range(16)}

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids if not skip_special_tokens or i not in self.all_special_ids)

    checks = []
    model, tokenizer = FakeModel(), FakeTokenizer()
    for policy in ("confidence_global", "left_to_right_frontier"):
        for k in (1, 2):
            kwargs = dict(model=model, tokenizer=tokenizer, prefix_ids=[0, 3], suffix_ids=[4, 1],
                          sample_tokens_fn=native.sample_tokens, requested_k=k, policy=policy,
                          row_seed=42, eos_penalty=0.0)
            actual = adapter.decode_fixed_transfer(**kwargs)
            reference = adapter.native_fixed_reference(**kwargs)
            equal = actual["completion_token_ids"] == reference["completion_token_ids"]
            checks.append({"policy": policy, "k": k, "token_ids_equal": equal,
                           "forward_count": actual["forward_count"], "remaining_masks": actual["final_remaining_masks"]})
            if not equal:
                raise RuntimeError(f"CPU native reference differs: {policy}, K{k}")
    args.output.write_text(json.dumps({"status": "passed", "scope": "CPU fake model only, no actual Base weight inference",
                                     "common_adapter": str(args.common_adapter), "profile": adapter.PROFILE,
                                     "checks": checks, "gpu_job_hours": 0}, indent=2) + "\n")
    print("CPU fake model: 4/4 adapter/native-reference checks passed")


if __name__ == "__main__":
    main()
