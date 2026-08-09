from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

import torch

from .core import action_name, set_seed
from .protocol import (
    DREAMON_ROOT,
    INITIAL_MASK_COUNT,
    MAX_CONTEXT_TOKENS,
    MAX_FORWARDS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)


def load_official_generator_module() -> Any:
    path = DREAMON_ROOT / "eval/generator.py"
    spec = importlib.util.spec_from_file_location("frontier_gate_official_dreamon_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official DreamOn generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.bos_id = int(tokenizer.bos_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.mask_id = int(tokenizer.mask_token_id)
        self.expand_id = 151667

    def encode(self, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)


class ModelProbe:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, x: torch.Tensor, attention_mask: torch.Tensor, tok_idx: torch.Tensor) -> Any:
        self.calls.append(
            {
                "x": x.detach().cpu().clone(),
                "attention_mask": attention_mask.detach().cpu().clone(),
                "tok_idx": tok_idx.detach().cpu().clone(),
            }
        )
        return self.model(x, attention_mask, tok_idx)


def _native_trace(
    *,
    model_calls: list[dict[str, torch.Tensor]],
    sample_calls: list[dict[str, torch.Tensor]],
    prefix_length: int,
    suffix_length: int,
    mask_id: int,
    expand_id: int,
    eos_id: int,
) -> list[dict[str, Any]]:
    if len(model_calls) != len(sample_calls):
        raise AssertionError(
            f"native probe mismatch: model_calls={len(model_calls)} sample_calls={len(sample_calls)}"
        )
    trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "kind": "initial_state",
            "dynamic_length": INITIAL_MASK_COUNT,
            "unresolved_masks": INITIAL_MASK_COUNT,
            "mask_token_id": mask_id,
            "continuous_middle": True,
        }
    ]
    for index, (model_call, sample_call) in enumerate(zip(model_calls, sample_calls), start=1):
        x = model_call["x"]
        attention = model_call["attention_mask"]
        visible = torch.diagonal(attention[0, 0], dim1=-2, dim2=-1).bool()
        real_length = int(visible.sum().item())
        dynamic_length = real_length - prefix_length - suffix_length
        mask_positions_abs = ((x[0] == mask_id) & visible).nonzero(as_tuple=False).squeeze(1)
        local_positions = (mask_positions_abs - prefix_length).tolist()
        proposals = sample_call["x0"]
        selected_rank = int(sample_call["selected_rank"].item())
        selected_position = int(local_positions[selected_rank])
        proposal_token_id = int(proposals[selected_rank].item())
        trace.append(
            {
                "step": index,
                "kind": "commit",
                "frontier": min(local_positions),
                "window": "inf",
                "eligible_positions": local_positions,
                "commit_position": selected_position,
                "proposal_token_id": proposal_token_id,
                "action": action_name(proposal_token_id, mask_id, expand_id, eos_id),
                "dynamic_length_before": dynamic_length,
            }
        )
    return trace


@torch.inference_mode()
def run_unmodified_native(
    *,
    model: Any,
    tokenizer: TokenizerAdapter,
    official_module: Any,
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    seed: int,
    device: str,
) -> dict[str, Any]:
    set_seed(seed)
    probe = ModelProbe(model)
    cfg = official_module.MDMGeneratorArgs(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        show_progress=False,
        dtype="bf16",
        device=device,
        max_tokens=MAX_CONTEXT_TOKENS,
        min_gen_len=INITIAL_MASK_COUNT,
        max_prompt_len=MAX_CONTEXT_TOKENS,
        max_gen_len=MAX_NEW_TOKENS,
        pad_to_max_len=False,
        pad_eos_to_right=True,
        batch_size=1,
        steps=MAX_FORWARDS,
        alg="entropy",
        alg_temp=0.0,
    )
    cfg.delete_eos_token = True
    generator = official_module.MDMGenerator(cfg, probe, tokenizer)
    input_ids = list(prefix_ids) + [tokenizer.mask_id] * INITIAL_MASK_COUNT + list(suffix_ids)
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    sample_calls: list[dict[str, torch.Tensor]] = []
    original_sample_tokens = official_module.sample_tokens

    def sample_probe(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        confidence, x0 = original_sample_tokens(*args, **kwargs)
        sample_calls.append(
            {
                "confidence": confidence.detach().cpu().clone(),
                "x0": x0.detach().cpu().clone(),
                "selected_rank": torch.topk(confidence, 1).indices.detach().cpu().clone(),
            }
        )
        return confidence, x0

    official_module.sample_tokens = sample_probe
    try:
        response, response_length = generator.batch_generate_with_expand_as_token(tensor)
    finally:
        official_module.sample_tokens = original_sample_tokens
    final_ids = response[
        0, len(prefix_ids) : len(prefix_ids) + int(response_length)
    ].detach().cpu().tolist()
    trace = _native_trace(
        model_calls=probe.calls,
        sample_calls=sample_calls,
        prefix_length=len(prefix_ids),
        suffix_length=len(suffix_ids),
        mask_id=tokenizer.mask_id,
        expand_id=tokenizer.expand_id,
        eos_id=tokenizer.eos_id,
    )
    unresolved = sum(token_id == tokenizer.mask_id for token_id in final_ids)
    stop_reason = "no_unresolved_masks" if unresolved == 0 else "forward_cap"
    return {
        "completion_token_ids": final_ids,
        "completion": tokenizer.decode(final_ids, skip_special_tokens=True),
        "step_trace": trace,
        "stop_reason": stop_reason,
        "forward_count": len(sample_calls),
        "final_unresolved_masks": unresolved,
    }
