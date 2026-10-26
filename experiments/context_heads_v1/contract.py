"""Tensor contract. Hidden states are target-aligned BY THE CALLER, never shifted here.

B=batch, K=selected slots (normally <=4), Q=queries, H=frozen hidden, V=logit
support, r=256. Slot order may be arbitrary; spatial order is position_ids.
No reference answers, tests, scores, or future realized tokens are accepted.
"""
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
from torch import Tensor

CONTRACT_ID = "dreamon-context-v1-20261015"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HeadConfig:
    vocab_size: int
    hidden_size: int
    special_token_ids: Tuple[int, ...] = ()
    rank: int = 256
    max_slots: int = 4
    position_scale: float = 1024.0
    phase: float = 1.0  # fixed round metadata; never inferred from token values
    seed: int = 20261015
    memory_mode: str = "full-memory"
    attention_mode: str = "learned-attention"
    allow_gaps: bool = False  # reserved cross-gap protocol, OFF in this batch
    tokenizer_size: Optional[int] = None  # may be smaller than padded logit V

    def __post_init__(self):
        if self.rank != 256 or self.vocab_size < 1 or self.hidden_size < 1:
            raise ValueError("rank must be 256; H and V must be positive")
        if self.max_slots < 1 or self.position_scale <= 0:
            raise ValueError("invalid capacity/position scale")
        if self.tokenizer_size is not None and not 1 <= self.tokenizer_size <= self.vocab_size:
            raise ValueError("tokenizer_size must be in [1,V]")
        if any(t < 0 or t >= self.vocab_size for t in self.special_token_ids):
            raise ValueError("special token outside vocabulary")
        if self.memory_mode not in ("full-memory", "reset-every-step", "last-only"):
            raise ValueError("unknown GRU memory mode")
        if self.attention_mode not in ("learned-attention", "uniform-mean", "last-only"):
            raise ValueError("unknown attention mode")


@dataclass(frozen=True)
class HeadBatch:
    old_hidden: Tensor       # float[B,K,H], already aligned to target prediction
    position_ids: Tensor     # int64[B,K], ORIGINAL X0 canvas coordinates
    chain_ids: Tensor        # int64[B,K], reset at gaps/barriers, -1 for padding
    selected_mask: Tensor    # bool[B,K]
    query_indices: Tensor    # int64[B,Q], selected-set slot indices
    token_ids: Tensor        # int64[B,K], meaningful only on explicit visible slots
    token_state: Tensor      # int64[B,K]: 0 unknown, 1 committed, 2 provisional
    history_mask: Tensor     # bool[B,Q,K], explicit query-specific visibility
    old_logits: Optional[Tensor] = None  # float[B,Q,V], only needed by adapter/loss

    def queries(self, indices: Tensor, mask: Tensor, logits=None):
        return replace(self, query_indices=indices, history_mask=mask, old_logits=logits)


@dataclass
class HeadOutput:
    latent: Tensor                         # float[B,Q,256]
    delta_logits: Optional[Tensor] = None   # float[B,Q,V], lambda=1
    attention_weights: Optional[Tensor] = None  # float[B,Q,4,K], zero on invisibles


def gather_slots(values: Tensor, indices: Tensor) -> Tensor:
    return values[torch.arange(values.shape[0], device=values.device)[:, None], indices]


def history_for(batch: HeadBatch, queries: Tensor) -> Tensor:
    """Geometric possible history; caller must still supply actual visibility/state.

    This does NOT infer visibility from token ids and does not bypass barriers.
    """
    pos = gather_slots(batch.position_ids, queries)
    chain = gather_slots(batch.chain_ids, queries)
    valid = gather_slots(batch.selected_mask, queries)
    return (batch.selected_mask[:, None, :] & valid[:, :, None]
            & (batch.chain_ids[:, None, :] == chain[:, :, None])
            & (batch.position_ids[:, None, :] < pos[:, :, None]))


def split_chains(position_ids: Tensor, selected_mask: Tensor,
                 barrier_mask: Optional[Tensor] = None, *, allow_gaps=False) -> Tensor:
    """Split spatially contiguous ordinary selected slots; barriers are excluded.

    A expands/EOS/other structural action belongs to A's scheduler. It must set a
    barrier, discard temporary structural candidates, and start a fresh prefix.
    allow_gaps is an explicit future interface, disabled by default.
    """
    barriers = torch.zeros_like(selected_mask) if barrier_mask is None else barrier_mask
    result = torch.full_like(position_ids, -1)
    for b in range(position_ids.shape[0]):
        chain, previous = -1, None
        for j in torch.argsort(position_ids[b]).tolist():
            if not bool(selected_mask[b, j]):
                continue
            p = int(position_ids[b, j])
            if bool(barriers[b, j]):
                previous = None
                continue
            if previous is None or (not allow_gaps and p != previous + 1):
                chain += 1
            result[b, j] = chain
            previous = p
    return result


def validate(batch: HeadBatch, cfg: HeadConfig, *, allow_provisional=False):
    b, k, h = batch.old_hidden.shape
    q = batch.query_indices.shape[1]
    if not (1 <= k <= cfg.max_slots) or h != cfg.hidden_size:
        raise ValueError("selected-set shape exceeds configured capacity")
    for name in ("position_ids", "chain_ids", "selected_mask", "token_ids", "token_state"):
        if getattr(batch, name).shape != (b, k):
            raise ValueError(f"{name}: expected [B,K]")
    if batch.query_indices.shape != (b, q) or batch.history_mask.shape != (b, q, k):
        raise ValueError("query/history shapes must be [B,Q] / [B,Q,K]")
    if batch.selected_mask.dtype != torch.bool or batch.history_mask.dtype != torch.bool:
        raise ValueError("masks must be explicit booleans")
    for name in ("position_ids", "chain_ids", "token_ids", "token_state", "query_indices"):
        if getattr(batch, name).dtype != torch.long:
            raise ValueError(f"{name} must be int64")
    if bool(((batch.query_indices < 0) | (batch.query_indices >= k)).any()):
        raise ValueError("query index outside selected set")
    if batch.old_logits is not None and batch.old_logits.shape != (b, q, cfg.vocab_size):
        raise ValueError("old_logits must be [B,Q,V]")
    if bool((batch.selected_mask & (batch.chain_ids < 0)).any()):
        raise ValueError("valid ordinary slots require chain ids")
    for row in range(b):
        p = batch.position_ids[row, batch.selected_mask[row]]
        if len(torch.unique(p)) != len(p):
            raise ValueError("duplicate original coordinates")
        for c in torch.unique(batch.chain_ids[row, batch.selected_mask[row]]):
            coords = torch.sort(batch.position_ids[row, batch.selected_mask[row]
                                & (batch.chain_ids[row] == c)]).values
            if not cfg.allow_gaps and bool((coords[1:] - coords[:-1] != 1).any()):
                raise ValueError("cross-gap chains require explicit opt-in")
    if bool((batch.history_mask & ~history_for(batch, batch.query_indices)).any()):
        raise ValueError("history must be valid, strictly earlier and same chain")
    visible = batch.history_mask.any(dim=1)
    legal_state = (batch.token_state == 1)
    if allow_provisional:
        legal_state |= batch.token_state == 2
    if bool((visible & ~legal_state).any()):
        raise ValueError("visible history requires committed ordinary tokens (or explicit Jacobi)")
    token_limit = cfg.tokenizer_size or cfg.vocab_size
    if bool((visible & ((batch.token_ids < 0) | (batch.token_ids >= token_limit))).any()):
        raise ValueError("visible token outside vocabulary")
    for t in cfg.special_token_ids:
        if bool((visible & (batch.token_ids == t)).any()):
            raise ValueError("structural/special candidate is a barrier, not ordinary memory")
