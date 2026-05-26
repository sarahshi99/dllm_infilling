from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import ModelConfig


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def load_model_and_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=get_torch_dtype(cfg.torch_dtype),
        device_map=cfg.device_map,
    )
    model.eval()
    return tokenizer, model


def resolve_mask_token_id(tokenizer) -> int:
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return tokenizer.mask_token_id

    candidates = ["<|mdm_mask|>", "<|mask|>", "[MASK]", "<mask>", "<mask_1>"]
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for cand in candidates:
        token_id = tokenizer.convert_tokens_to_ids(cand)
        if token_id is not None and token_id != unk_id:
            return int(token_id)

    raise ValueError("Unable to resolve mask token id for tokenizer.")
