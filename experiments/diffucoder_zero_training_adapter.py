"""Pinned original DiffuCoder profile and unmodified native fixed-canvas reference.

This module does not load CUDA, train, choose a length, or alter the shared decoder.
The native reference exposes the released floating timestep schedule; it is not
claimed to implement exactly K transfers per forward. See the accompanying audit.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

MODEL_ID = "apple/DiffuCoder-7B-Base"
MODEL_REVISION = "a2c33054cbe99b9ab7af074f09b1d13943157ff1"
MODEL_PATH = Path("/home/shx/.cache/dllm_infilling/model_snapshots/apple--DiffuCoder-7B-Base") / MODEL_REVISION
PROFILE = {
    "id": MODEL_ID,
    "revision": MODEL_REVISION,
    "local_path": str(MODEL_PATH),
    "weight_kind": "original_base_fixed_length_auxiliary",
    "direct_transfer_status": "passed",
    "validation_scope": "CPU full ordinary-token mapping, tokenizer rules, special columns, raw-logit alignment; GPU 16 pending",
    "canvas_length": 64,
    "model_logits_shifted": False,
    "decoder_shift_count": 1,
    "head_source_structural_columns": [151643, 151666, 151667],
    "head_source_structural_meanings": ["EOS/deletion in DreamOn; EOS only in Base", "mask", "expand in DreamOn; unnamed in Base"],
    "base_has_expand_delete_actions": False,
}


def native_generation_module(model_path: Path = MODEL_PATH):
    """Load the generation reference from this exact model snapshot."""
    spec = importlib.util.spec_from_file_location("pinned_diffucoder_generation", model_path / "generation_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_canvas(tokenizer, prefix: str, suffix: str) -> tuple[list[int], int, int]:
    """Use tokenizer BOS, 64 masks, suffix, EOS; never inspect a test answer."""
    prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    start = len(prefix_ids)
    return prefix_ids + [tokenizer.mask_token_id] * 64 + suffix_ids, start, start + 64


@torch.inference_mode()
def released_native_reference(model, tokenizer, prefix: str, suffix: str, *, steps: int = 64, model_path: Path = MODEL_PATH) -> dict:
    """Run released `_sample` on exactly the supplied canvas, without right padding.

    A caller must acquire the controller GPU mutex and seed derive_row_seed first.
    This is a reference for the released schedule, not a hidden CAL length search.
    """
    native = native_generation_module(model_path)
    ids, start, end = fixed_canvas(tokenizer, prefix, suffix)
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    config = native.DreamGenerationConfig(
        max_length=len(ids), steps=steps, temperature=0.2, top_p=0.9,
        top_k=None, alg="entropy", alg_temp=0.0,
        mask_token_id=tokenizer.mask_token_id,
        return_dict_in_generate=True, output_history=False,
    )
    remaining = []

    def token_hook(step, x, logits):
        if step is not None:
            remaining.append(int((x[:, start:end] == tokenizer.mask_token_id).sum().item()))
        return x

    result = native.DreamGenerationMixin._sample(
        model, input_ids, None, config, token_hook, lambda step, x, logits: logits,
    )
    middle = result.sequences[0, start:end].tolist()
    return {"middle_token_ids": middle,
            "raw_middle": tokenizer.decode(middle, skip_special_tokens=False),
            "specials_removed_middle": tokenizer.decode(middle, skip_special_tokens=True),
            "remaining_masks_by_step": remaining,
            "reference_schedule": "released int(num_mask * (1 - s/t)); not exact K"}
