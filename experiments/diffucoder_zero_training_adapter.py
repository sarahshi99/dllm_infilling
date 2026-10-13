"""Pinned DiffuCoder Base profile, fixed-K adapter and independent native reference.

This module does not load CUDA, train, choose a length, or alter the shared decoder.
The released reference remains available for the floating-schedule audit. The
fixed reference patches K/order in this release only; it has no EOS penalty.
"""
from __future__ import annotations

import importlib.util
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

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
    "eos_penalty": 0.0,
    "native_eps": 1e-12,
    "native_selection_marker_count": 1,
    "native_fixed_reference_patch": "fixed K and optional frontier positions only",
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


def decode_fixed_transfer(*, eos_penalty: float = 0.0, **kwargs) -> dict:
    """Reuse the ordinary fixed decoder with this release's absent EOS penalty."""
    if eos_penalty != 0.0:
        raise ValueError("pinned DiffuCoder native generation has no EOS penalty")
    from experiments.dreamcoder_fixed_transfer_adapter import decode_fixed_transfer as decode
    kwargs.setdefault("eps", 1e-12)
    return decode(eos_penalty=0.0, **kwargs)


@torch.inference_mode()
def native_fixed_reference(
    *, model, tokenizer, prefix_ids: list[int], suffix_ids: list[int],
    requested_k: int, policy: str, row_seed: int, temperature: float = 0.2,
    top_p: float | None = 0.9, top_k: int | None = None,
    eos_penalty: float = 0.0, eps: float = 1e-12, steps: int = 64,
    sample_tokens_fn=None,
) -> dict:
    """Patch only K/order in this exact release's independent native `_sample`.

    Unlike DreamCoder's native implementation, this release has one selection
    marker and no EOS penalty. It retains its native hard-coded eps=1e-12.
    No global file or model method is overwritten.
    """
    from experiments.dreamcoder_fixed_transfer_adapter import _canvas, _finish, _seed
    from experiments.dreamon_singleline_order_parallel import GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER
    if requested_k not in (1, 2) or policy not in (GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER) or steps != 64:
        raise ValueError("reference requires frozen K1/K2, order and 64-step settings")
    if eos_penalty != 0.0:
        raise ValueError("pinned DiffuCoder native generation has no EOS penalty")
    native_model = model if hasattr(model, "_sample") else model.model
    original = type(native_model)._sample
    source = textwrap.dedent(inspect.getsource(original))
    schedule = "number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)"
    marker = "x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id"
    if source.count(schedule) != 1 or source.count(marker) != 1 or "eos_penalty" in source or "eps = 1e-12" not in source:
        raise RuntimeError("DiffuCoder native reference differs from the audited exact release")
    source = source.replace(schedule, f"number_transfer_tokens = min({requested_k}, int(num_mask_token))")
    if policy == LEFT_TO_RIGHT_FRONTIER:
        index = source.index(marker)
        source = source[:index] + "transfer_index = mask_index[0].nonzero().flatten()[:number_transfer_tokens].unsqueeze(0)\n                " + source[index:]
    namespace = dict(original.__globals__)
    exec(compile(source, "<diffucoder-native-fixed-K-reference>", "exec"), namespace)
    x, start = _canvas(model, tokenizer, prefix_ids, suffix_ids)
    config = SimpleNamespace(output_history=False, return_dict_in_generate=True,
                             max_length=x.shape[1], mask_token_id=int(tokenizer.mask_token_id),
                             steps=steps, alg="entropy", alg_temp=0.0,
                             temperature=temperature, top_p=top_p, top_k=top_k)
    _seed(row_seed)
    output = namespace[original.__name__](native_model, x, None, config,
                                         lambda step, x, logits: x,
                                         lambda step, x, logits: logits)
    return _finish(output.sequences, start, tokenizer)


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
