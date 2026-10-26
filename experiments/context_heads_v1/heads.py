"""Seven additive heads. All share r=256, lambda=1, no full-logit input.

forward: teacher-force any explicit legal prefix, returns [B,Q,r] (+ [B,Q,V]).
step: same calculation for Q=1; caller supplies the current selected-set prefix.
reset: returns cleared explicit prefix; modules keep NO case/round/chain cache.
H2 rebuilds its <=3-record state for each query (a cached GPU path is not claimed).
"""
from dataclasses import replace

import torch
from torch import nn

from .contract import (HeadBatch, HeadConfig, HeadOutput, gather_slots, history_for,
                       validate)

MODEL_NAMES = ("H0-static", "Hctx-old-context-only", "Hgated-author-family",
               "H1-context-prev", "H2-GRU", "H3-short-attention", "H4-causal-mixer")


def mlp(n, r):
    return nn.Sequential(nn.Linear(n, r), nn.SiLU(), nn.Linear(r, r))


class ContextHead(nn.Module):
    def __init__(self, name: str, config: HeadConfig):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(name)
        self.name, self.config = name, config
        r, h, v = config.rank, config.hidden_size, config.vocab_size
        # Preserve the caller's RNG; common E/W tensors match across all token heads.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.output = nn.Linear(r, v, bias=False)
            nn.init.normal_(self.output.weight, std=0.02)
            if name != "Hctx-old-context-only":
                self.embedding = nn.Embedding(v, r)
                nn.init.normal_(self.embedding.weight, std=0.02)
            if name not in ("H0-static", "Hgated-author-family"):
                self.hidden_proj = nn.Linear(h, r)
            if name == "Hctx-old-context-only":
                self.context_mlp = mlp(r + 3, r)
            elif name == "Hgated-author-family":
                self.gate = nn.Linear(h + r, r)  # includes author's bias
            elif name == "H1-context-prev":
                self.gamma = mlp(2 * r + 2, r)
                nn.init.zeros_(self.gamma[-1].weight)
                nn.init.zeros_(self.gamma[-1].bias)
            elif name in ("H2-GRU", "H3-short-attention"):
                self.record = nn.Linear(2 * r + 2, r)
                self.readout = mlp(2 * r + 3, r)
                if name == "H2-GRU":
                    self.gru = nn.GRUCell(r, r)  # one layer, no hidden module cache
                else:
                    self.q_proj = nn.Linear(r, r)
                    self.k_proj = nn.Linear(r, r)
                    self.v_proj = nn.Linear(r, r)
                    # Per-head positive distance penalty via softplus.
                    self.distance_raw = nn.Parameter(torch.zeros(4))
            elif name == "H4-causal-mixer":
                self.summary_proj = nn.Linear(h, r)
                self.position_proj = nn.Linear(3, r)
                self.fuse_query = nn.Linear(3 * r, r)
                self.fuse_record = nn.Linear(3 * r, r)
                # Stored full tensor, strictly lower-triangular effective use.
                self.mixer = nn.Parameter(torch.empty(config.max_slots, config.max_slots, r))
                nn.init.normal_(self.mixer, mean=0.1, std=0.02)
                self.transform = mlp(r, r)
        mask = torch.ones(v)
        if config.special_token_ids:
            mask[list(config.special_token_ids)] = 0
        self.register_buffer("output_mask", mask)

    def reset(self, batch: HeadBatch, *, chain_id=None) -> HeadBatch:
        """Round reset (all) or committed structure barrier (one chain).

        Clears memory, not X0 geometry. A must provide new geometry after a
        structural edit; this method does not commit expand/EOS or guessed tokens.
        """
        clear = batch.selected_mask if chain_id is None else batch.chain_ids == chain_id
        return replace(batch, token_state=batch.token_state.masked_fill(clear, 0),
                       token_ids=batch.token_ids.masked_fill(clear, 0),
                       history_mask=batch.history_mask & ~clear[:, None, :])

    def step(self, batch: HeadBatch, **kwargs) -> HeadOutput:
        if batch.query_indices.shape[1] != 1:
            raise ValueError("step requires Q=1; use forward for teacher forcing")
        return self.forward(batch, **kwargs)

    def forward(self, batch: HeadBatch, *, project=True, zero_override=False,
                allow_provisional=False) -> HeadOutput:
        if allow_provisional and self.name not in ("H3-short-attention", "H4-causal-mixer"):
            raise ValueError("only H3/H4 accept explicit provisional Jacobi prefixes")
        validate(batch, self.config, allow_provisional=allow_provisional)
        cfg, qi, mask = self.config, batch.query_indices, batch.history_mask
        b, k, _ = batch.old_hidden.shape
        q, r = qi.shape[1], cfg.rank
        # Frozen old information; padded h (including NaN) must contribute nothing.
        h = torch.where(batch.selected_mask[..., None], batch.old_hidden.detach(), 0)
        ht = gather_slots(h, qi)
        pos = batch.position_ids
        qp = gather_slots(pos, qi)
        possible = history_for(batch, qi)
        order = possible.sum(-1)
        selected_order = ((pos[:, :, None] > pos[:, None, :])
                          & (batch.chain_ids[:, :, None] == batch.chain_ids[:, None, :])
                          & batch.selected_mask[:, None, :]).sum(-1)
        # Coordinates and generation order are separate features; phase is fixed.
        fixed = torch.stack([qp / cfg.position_scale, order / cfg.max_slots,
                             torch.full_like(qp, cfg.phase, dtype=h.dtype)], -1).to(h.dtype)
        distance = qp[:, :, None] - pos[:, None, :]
        rel = torch.stack([distance / cfg.position_scale,
                           (order[:, :, None] - selected_order[:, None, :]) / cfg.max_slots], -1).to(h.dtype)
        last = pos[:, None, :].expand(b, q, k).masked_fill(~mask, torch.iinfo(pos.dtype).min).argmax(-1)
        last_h = gather_slots(h, last)
        visible_tokens = mask.any(1)
        weights = None
        if self.name != "Hctx-old-context-only":
            safe_tokens = torch.where(visible_tokens, batch.token_ids, 0)
            emb = self.embedding(safe_tokens)
            emb = torch.where(visible_tokens[..., None], emb, 0)
            prev = gather_slots(emb, last)
        if self.name == "H0-static":
            z = prev
        elif self.name == "Hctx-old-context-only":
            z = self.context_mlp(torch.cat([self.hidden_proj(ht), fixed], -1))
        elif self.name == "Hgated-author-family":
            z = prev * torch.sigmoid(self.gate(torch.cat([ht, prev], -1)))
        elif self.name == "H1-context-prev":
            prev_rel = rel.gather(2, last[..., None, None].expand(b, q, 1, 2)).squeeze(2)
            gamma = 1 + torch.tanh(self.gamma(torch.cat([
                self.hidden_proj(ht), self.hidden_proj(last_h), prev_rel], -1)))
            z = prev * gamma
        elif self.name in ("H2-GRU", "H3-short-attention"):
            # [B,Q,K,r]; no new-token backbone encoding. History old h is X0's h.
            hp = self.hidden_proj(h)
            target = self.hidden_proj(ht)
            records = self.record(torch.cat([
                emb[:, None].expand(b, q, k, r),
                hp[:, None].expand(b, q, k, r), rel], -1))
            records = torch.where(mask[..., None], records, 0)
            if self.name == "H2-GRU":
                state = h.new_zeros(b, q, r)
                spatial = torch.argsort(pos, -1)
                for n in range(k):
                    idx = spatial[:, n][:, None].expand(b, q)
                    rec = records.gather(2, idx[..., None, None].expand(b, q, 1, r)).squeeze(2)
                    active = mask.gather(2, idx[..., None]).squeeze(2)
                    if cfg.memory_mode == "last-only":
                        active &= idx == last
                    old_state = state if cfg.memory_mode == "full-memory" else torch.zeros_like(state)
                    updated = self.gru(rec.reshape(-1, r), old_state.reshape(-1, r)).reshape(b, q, r)
                    state = torch.where(active[..., None], updated, state)
                read = state
            else:
                d = r // 4
                query = self.q_proj(target).reshape(b, q, 4, d)
                keys = self.k_proj(records).reshape(b, q, k, 4, d).transpose(2, 3)
                values = self.v_proj(records).reshape(b, q, k, 4, d).transpose(2, 3)
                score = (query[..., None, :] * keys).sum(-1) / d ** 0.5
                score -= torch.nn.functional.softplus(self.distance_raw)[None, None, :, None] * distance.abs()[:, :, None] / cfg.position_scale
                visible = mask[:, :, None, :].expand(b, q, 4, k)
                if cfg.attention_mode == "learned-attention":
                    score = score.masked_fill(~visible, -torch.inf)
                    score = torch.where(visible.any(-1, keepdim=True), score, 0)
                    weights = torch.softmax(score, -1).masked_fill(~visible, 0)
                elif cfg.attention_mode == "uniform-mean":
                    weights = visible.to(h.dtype) / visible.sum(-1, keepdim=True).clamp_min(1)
                else:
                    weights = torch.zeros_like(score).scatter_(-1, last[:, :, None, None].expand(b, q, 4, 1), 1)
                    weights = weights.masked_fill(~visible, 0)
                read = (weights[..., None] * values).sum(-2).reshape(b, q, r)
            z = self.readout(torch.cat([read, target, fixed], -1))
        else:  # H4: xPress-inspired strict causal per-channel linear mix
            summary = h.sum(1) / batch.selected_mask.sum(1).clamp_min(1)[:, None]
            summary = self.summary_proj(summary)
            base = self.fuse_query(torch.cat([
                self.hidden_proj(ht), summary[:, None].expand(b, q, r),
                self.position_proj(fixed)], -1))
            records = self.fuse_record(torch.cat([
                self.hidden_proj(h), summary[:, None].expand(b, k, r), emb], -1))
            qo = order.clamp_max(cfg.max_slots - 1)
            ko = selected_order.clamp_max(cfg.max_slots - 1)
            coeff = self.mixer[qo[:, :, None], ko[:, None, :]]
            coeff = torch.where(mask[..., None], coeff, 0)
            mixed = base + (coeff * records[:, None]).sum(2)
            z = mixed + self.transform(mixed)
        # No-history anchors, padded queries, and zero override: exactly zero,
        # including all learned biases. Output projection is bias-free as in H0.
        active = mask.any(-1) & gather_slots(batch.selected_mask, qi)
        if zero_override:
            active = torch.zeros_like(active)
        z = torch.where(active[..., None], z, 0)
        delta = self.output(z) * self.output_mask if project else None
        return HeadOutput(z, delta, weights)

    def load_legacy_static(self, state):
        """Map old MarkovHead state (or checkpoint['head']); no optimizer import.

        old embedding.weight -> embedding.weight, output.weight -> output.weight,
        output_mask -> output_mask. Also supports author's markov_w1/markov_w2.
        Use a meta-constructed H0 + assign=True here to avoid a second full table.
        """
        if self.name != "H0-static":
            raise ValueError("legacy mapping is only for H0")
        state = state.get("head", state)
        mapped = {
            "embedding.weight": state.get("embedding.weight", state.get("markov_w1.weight")),
            "output.weight": state.get("output.weight", state.get("markov_w2.weight")),
            "output_mask": state.get("output_mask", torch.ones(self.config.vocab_size)),
        }
        if mapped["embedding.weight"] is None or mapped["output.weight"] is None:
            raise ValueError("missing legacy low-rank weights")
        expected = torch.ones(self.config.vocab_size)
        expected[list(self.config.special_token_ids)] = 0
        if not torch.equal(mapped["output_mask"].cpu(), expected):
            raise ValueError("legacy special-token mask differs from configuration")
        self.load_state_dict(mapped, strict=True, assign=True)
