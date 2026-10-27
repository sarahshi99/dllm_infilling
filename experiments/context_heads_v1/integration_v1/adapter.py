"""B-owned adapter for A f521bd3's keyword/dict correction boundary.

No alignment, sampling, position selection, canvas mutation, or hidden case state.
The optional legacy head exists ONLY for A Engine's existing zero-control path.
"""
from typing import Optional

from torch import nn

from ..contract import CONTRACT_ID, SCHEMA_VERSION, HeadBatch, HeadConfig


def config_from_a_schema(schema, *, tokenizer_size: Optional[int] = None,
                         seed=20261015) -> HeadConfig:
    """Use A's actual H/V/special set verbatim, including padded vocabulary rows.

    tokenizer_size, when supplied, is an independently verified len(tokenizer).
    Never infer it by subtracting special-token count or by examining token ids.
    """
    if schema["contract_id"] != CONTRACT_ID or schema["schema_version"] != SCHEMA_VERSION:
        raise ValueError("A schema does not match the existing common contract")
    if schema["rank"] != 256:
        raise ValueError("common rank must be 256")
    return HeadConfig(vocab_size=schema["V"], hidden_size=schema["H"],
                      special_token_ids=tuple(schema["special_token_ids"]),
                      max_slots=schema["max_k"], seed=seed, tokenizer_size=tokenizer_size)


class KeywordHeadAdapter(nn.Module):
    """A-compatible kwargs -> B HeadBatch -> dict(latent, delta_logits).

    calls counts query rows, like A's FrozenHeadAdapter. The model itself remains
    stateless: every prefix/geometry tensor comes from the caller on each call.

    A's Engine.__init__ creates its zero arm from head.head and assumes that raw
    legacy module has embedding/output/output_mask. For Hctx this cannot be the
    candidate (which has no token embedding). Supply A's already-loaded OLD raw
    static head explicitly as legacy_zero_head; this does not initialize the new
    candidate from trained weights. No new legacy weights are downloaded/loaded.
    """
    def __init__(self, context_head, *, legacy_zero_head=None, zero_override=False):
        super().__init__()
        self.context_head = context_head
        self.legacy_zero_head = legacy_zero_head
        self.zero_override = zero_override
        self.calls = 0

    @property
    def head(self):
        if self.legacy_zero_head is None:
            raise AttributeError("A Engine zero arm needs its explicitly supplied raw legacy static head")
        return self.legacy_zero_head

    def forward(self, *, old_hidden, position_ids, chain_ids, selected_mask,
                query_indices, old_logits, token_ids, token_state, history_mask):
        batch = HeadBatch(old_hidden=old_hidden, position_ids=position_ids, chain_ids=chain_ids,
                          selected_mask=selected_mask, query_indices=query_indices,
                          old_logits=old_logits, token_ids=token_ids, token_state=token_state,
                          history_mask=history_mask)
        # A's first-batch real generation has committed prefixes only. This
        # wrapper deliberately does not enable the separate Jacobi executor.
        out = self.context_head(batch, zero_override=self.zero_override)
        self.calls += int(query_indices.numel())
        return {"latent": out.latent, "delta_logits": out.delta_logits}
