"""Ordinary-token CPU reference executors, separate from head architecture.

Not A's sampler/structure scheduler. The caller supplies a deterministic function
sample(scores[B,Q,V], fixed_u[B,Q])->token_ids[B,Q]. Never redraw u per iteration.
Jacobi tokens remain provisional; this module cannot authorize GPU submission.
"""
from dataclasses import dataclass, replace
from typing import Callable, List

import torch
from torch import Tensor

from .contract import HeadBatch, history_for, validate

Sampler = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class BarrierEvent:
    chain_id: int
    reason: str  # expand, EOS, deletion, or other structural edit

    def apply(self, head, batch):
        if not self.reason:
            raise ValueError("barrier reason required")
        return head.reset(batch, chain_id=self.chain_id)


@dataclass
class ExecutionResult:
    token_ids: Tensor          # [B,K], padding unchanged
    token_state: Tensor        # sequential mock:1; Jacobi:2 except fixed anchors
    positions_per_call: List[int]  # actual full-vocab projected rows, includes all rounds
    sampling_positions: int

    @property
    def projection_calls(self):
        return len(self.positions_per_call)

    @property
    def projected_positions(self):
        return sum(self.positions_per_call)


def inverse_cdf_mock(scores: Tensor, fixed_u: Tensor) -> Tensor:
    """CPU T=1 mock, ascending token-id CDF. This is NOT A's T=.2/top-p=.9 sampler."""
    cdf = scores.float().softmax(-1).cumsum(-1)
    return (cdf < fixed_u[..., None]).sum(-1).clamp_max(scores.shape[-1] - 1)


def greedy_mock(scores: Tensor, fixed_u: Tensor) -> Tensor:
    return scores.argmax(-1)  # PyTorch's first-index tie rule is deterministic


def _row(batch, b):
    return HeadBatch(**{key: (None if value is None else value[b:b+1])
                        for key, value in vars(batch).items()})


@torch.no_grad()
def execute(head, batch: HeadBatch, initial_tokens: Tensor, fixed_u: Tensor,
            sampler: Sampler, *, mode="sequential", rounds=3) -> ExecutionResult:
    """K4: sequential versus 1/2/3 synchronous Jacobi updates, same head weights.

    Requires canonical query_indices=arange(K), old_logits[B,K,V], full possible
    history_mask. Slots may be spatially unordered. Every chain's first token is
    already sampled/fixed by A; initial_tokens supplies these anchors. Ordinary
    support only, no structural actions, immutable X0 and selected set.

    Bound: longest chain length L needs <= L-1 rounds; 3 is NOT a long-chain bound.
    No early stopping, so cost accounting reports the requested full passes.
    """
    if mode not in ("sequential", "jacobi") or rounds < 1:
        raise ValueError("invalid executor mode/round count")
    if mode == "jacobi" and head.name not in ("H3-short-attention", "H4-causal-mixer"):
        raise ValueError("Jacobi reserved for H3/H4")
    bsz, k = batch.selected_mask.shape
    canonical = torch.arange(k, device=batch.query_indices.device)[None].expand(bsz, k)
    if not torch.equal(batch.query_indices, canonical) or batch.old_logits is None:
        raise ValueError("executor requires full canonical [B,K,V] old logits")
    if initial_tokens.shape != (bsz, k) or initial_tokens.dtype != torch.long:
        raise ValueError("initial_tokens must be int64[B,K]")
    if fixed_u.shape != (bsz, k) or bool(((fixed_u <= 0) | (fixed_u >= 1)).any()):
        raise ValueError("fixed_u must be [B,K] in (0,1)")
    geometry = history_for(batch, canonical)
    if not torch.equal(batch.history_mask, geometry):
        raise ValueError("reference executor requires full possible per-chain prefix")
    anchors = batch.selected_mask & ~geometry.any(-1)
    tokens = initial_tokens.clone()
    states = torch.zeros_like(tokens)
    states[anchors] = 1
    # Validate geometry using temporary legal ordinary tokens, independent of seeds.
    token_limit = head.config.tokenizer_size or head.config.vocab_size
    legal_ids = [t for t in range(token_limit) if t not in head.config.special_token_ids]
    if not legal_ids:
        raise ValueError("ordinary vocabulary is empty")
    check = replace(batch, token_ids=torch.full_like(tokens, legal_ids[0]),
                    token_state=torch.ones_like(tokens))
    validate(check, head.config)
    if mode == "jacobi":
        states[batch.selected_mask & ~anchors] = 2
    for special in head.config.special_token_ids:
        if bool((batch.selected_mask & (tokens == special)).any()):
            raise ValueError("initial structural guesses require A's structure scheduler")
    if bool((batch.selected_mask & ((tokens < 0) | (tokens >= token_limit))).any()):
        raise ValueError("initial ordinary token outside vocabulary")
    positions_per_call = []
    total_sampling = 0
    for _ in range(1 if mode == "sequential" else rounds):
        previous = tokens.clone()
        for b in range(bsz):
            row = _row(batch, b)
            ordered = torch.argsort(row.position_ids[0]).tolist()
            targets = [j for j in ordered if bool(batch.selected_mask[b, j] & ~anchors[b, j])]
            groups = [[j] for j in targets] if mode == "sequential" else [targets]
            for group in groups:
                if not group:
                    continue
                idx = torch.tensor([group], dtype=torch.long, device=tokens.device)
                readable_tokens = tokens if mode == "sequential" else previous
                prefix = replace(row, token_ids=readable_tokens[b:b+1], token_state=states[b:b+1],
                                 query_indices=idx, history_mask=geometry[b:b+1, group],
                                 old_logits=batch.old_logits[b:b+1, group])
                out = head(prefix, allow_provisional=mode == "jacobi")
                scores = prefix.old_logits.detach() + out.delta_logits
                # Restrict mock equivalence to ordinary support. Delta=0 on
                # specials alone would NOT remove their softmax probability.
                scores = scores.clone()
                scores[..., list(head.config.special_token_ids)] = -torch.inf
                scores[..., token_limit:] = -torch.inf
                sampled = sampler(scores, fixed_u[b:b+1, group])
                if sampled.shape != idx.shape or sampled.dtype != torch.long:
                    raise ValueError("sampler must return int64[B,Q]")
                if bool(((sampled < 0) | (sampled >= token_limit)).any()):
                    raise ValueError("sampler returned out-of-range token")
                for special in head.config.special_token_ids:
                    if bool((sampled == special).any()):
                        raise ValueError("structural output requires A's barrier scheduling")
                tokens[b, group] = sampled[0]
                if mode == "sequential":
                    states[b, group] = 1
                positions_per_call.append(len(group))
                total_sampling += len(group)
    return ExecutionResult(tokens, states, positions_per_call, total_sampling)
