"""Dream Base fixed-canvas auxiliary adapter; never interprets expand/delete."""
from __future__ import annotations

import inspect
import textwrap
from types import SimpleNamespace


PROFILE = {
    "model_id": "Dream-org/Dream-v0-Base-7B",
    "revision": "6572adb5535263e4d1a337b56942ba48b6dee2a9",
    "local_path": "/home/shx/.cache/dllm_infilling/zero_training_transfer_20261013/dream_base/6572adb5535263e4d1a337b56942ba48b6dee2a9",
    "weight_kind": "original_base_fixed_length_auxiliary",
    "canvas_length": 64, "steps": 64, "temperature": 0.2, "top_p": 0.9,
    "alg": "entropy", "eos_penalty": 0.0,
    "alg_scope": "supported native entropy; existing project Base setting, not native default",
    "eos_scope": "native Base has no EOS penalty; public decode binds zero",
}


def canvas_inputs(tokenizer, prefix: str, suffix: str) -> tuple[list[int], list[int]]:
    return ([int(tokenizer.bos_token_id)] + tokenizer.encode(prefix, add_special_tokens=False),
            tokenizer.encode(suffix, add_special_tokens=False) + [int(tokenizer.eos_token_id)])


def decode_fixed_transfer(**kwargs) -> dict:
    from experiments.dreamcoder_fixed_transfer_adapter import decode_fixed_transfer as common_decode
    if float(kwargs.get("eos_penalty", 0.0)) != 0.0:
        raise ValueError("Dream Base native sampler has no EOS penalty")
    kwargs["eos_penalty"] = 0.0
    return common_decode(**kwargs)


def _patch_native_source(source: str, requested_k: int, policy: str) -> str:
    """Only patch fixed-K count and optional frontier selection in Base release."""
    if requested_k not in (1, 2) or policy not in ("confidence_global", "left_to_right_frontier"):
        raise ValueError("only frozen K1/K2 and original orders are supported")
    source = textwrap.dedent(source)
    schedule = "number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)"
    marker = "x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id"
    if source.count(schedule) != 1 or source.count(marker) != 1 or "eos_penalty" in source:
        raise RuntimeError("Dream Base native source differs from audited release")
    source = source.replace(schedule, f"number_transfer_tokens = min({requested_k}, int(num_mask_token))")
    if policy == "left_to_right_frontier":
        lines = source.splitlines()
        index = next(i for i, line in enumerate(lines) if marker in line)
        indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
        lines.insert(index, indent + "transfer_index = mask_index[0].nonzero().flatten()[:number_transfer_tokens].unsqueeze(0)")
        source = "\n".join(lines) + "\n"
    return source


def native_fixed_reference(
    *, model, tokenizer, prefix_ids, suffix_ids, requested_k, policy, row_seed,
    temperature=0.2, top_p=0.9, top_k=None, eos_penalty=0.0, eps=1e-3, steps=64,
    sample_tokens_fn=None,
) -> dict:
    from experiments.dreamcoder_fixed_transfer_adapter import _canvas, _finish, _seed
    if eos_penalty != 0.0 or steps != 64:
        raise ValueError("Dream Base reference requires no EOS penalty and 64 steps")
    native_model = model if hasattr(model, "_sample") else model.model
    original = type(native_model)._sample
    namespace = dict(original.__globals__)
    source = _patch_native_source(inspect.getsource(original), requested_k, policy)
    exec(compile(source, "<Dream-Base-native-fixed-K-reference>", "exec"), namespace)
    x, start = _canvas(model, tokenizer, prefix_ids, suffix_ids)
    config = SimpleNamespace(output_history=False, return_dict_in_generate=True,
                             max_length=x.shape[1], mask_token_id=int(tokenizer.mask_token_id),
                             pad_token_id=int(tokenizer.pad_token_id), steps=steps, eps=eps,
                             alg="entropy", alg_temp=0.0, temperature=temperature,
                             top_p=top_p, top_k=top_k)
    _seed(row_seed)
    output = namespace[original.__name__](native_model, x, None, config,
                                         lambda step, x, logits: x,
                                         lambda step, x, logits: logits)
    return _finish(output.sequences, start, tokenizer)
