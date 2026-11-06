"""Independent common-u decoder. Historical sampler/decoder are never patched.

FP32 distribution arithmetic; stable descending score sort (token ID ascending
on ties) for nucleus selection, include the first token crossing p. CDF is
FP64, in ascending token ID order. Keys exclude arm/head/iteration names.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import torch
from experiments.dreamon_singleline_order_parallel import (
    GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER, select_positions,
    _apply_one_expand, _apply_one_delete,
)


def sample_key(case, round_index, position, purpose='token'):
    return ["dreamon-context-v1-20261015", 42, str(case), int(round_index), int(position), purpose]


def uniform(key):
    d = hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(',', ':')).encode()).digest()
    # Midpoints of 2**52 bins, strictly between zero and one in binary64.
    return (int.from_bytes(d[:8], 'big') >> 12) * 2.0**-52 + 2.0**-53


def distribution(logits, temperature=0.2, top_p=0.9):
    scores = logits.float() / float(temperature)
    if top_p is not None and top_p < 1:
        values, ids = torch.sort(scores, dim=-1, descending=True, stable=True)
        cumulative = values.softmax(-1).double().cumsum(-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = torch.zeros_like(remove).scatter(-1, ids, remove)
        scores = scores.masked_fill(remove, -torch.inf)
    return torch.log_softmax(scores, -1)


def inverse_cdf(logp, u):
    cdf = logp.exp().double().cumsum(-1)
    # Renormalize FP32 summation residue in FP64; last CDF is exactly one.
    cdf = cdf / cdf[..., -1:]
    value = torch.as_tensor(u, device=logp.device, dtype=torch.float64)
    return torch.searchsorted(cdf.contiguous(), value.unsqueeze(-1), right=False).squeeze(-1)


def draw(logits, key):
    return int(inverse_cdf(distribution(logits), uniform(key)).item())


def aligned_hidden(hidden, positions):
    # Released DreamOn duplicates h[0] at target0, then target t reads h[t-1].
    return hidden[:, (positions - 1).clamp_min(0), :]


def chain_ids(positions):
    ids = []
    for i, p in enumerate(positions):
        ids.append(0 if i == 0 else ids[-1] + int(p != positions[i-1] + 1))
    return ids


def history_mask(positions, chains, selected_mask, token_state, queries):
    qpos = positions.gather(1, queries)
    qchain = chains.gather(1, queries)
    return (selected_mask[:, None, :] & (chains[:, None, :] == qchain[:, :, None]) &
            (positions[:, None, :] < qpos[:, :, None]) & (token_state[:, None, :] == 1))


class FrozenHeadAdapter(torch.nn.Module):
    """Contract-compatible wrapper around the frozen latest-token rank256 head.

    It only consumes the latest visible committed slot; H0 remains part of the
    common input. The wrapper does not select targets, sample, or see labels.
    """
    def __init__(self, head, zero_override=False):
        super().__init__()
        self.head = head
        self.zero_override = zero_override
        self.calls = 0

    def forward(self, *, old_hidden, position_ids, chain_ids, selected_mask,
                query_indices, old_logits, token_ids, token_state, history_mask):
        self.calls += int(query_indices.numel())
        legal = globals()['history_mask'](position_ids, chain_ids, selected_mask, token_state, query_indices)
        if bool((history_mask & ~legal).any()):
            raise ValueError('history mask exposes future/uncommitted/other-chain slot')
        if not bool(history_mask.any(-1).all()):
            raise ValueError('anchors must not call correction interface')
        latest = position_ids[:, None, :].expand_as(history_mask).masked_fill(~history_mask, -1).argmax(-1)
        ids = token_ids.gather(1, latest)
        latent = self.head.embedding(ids)
        delta = self.head.output(latent) * self.head.output_mask
        # A zero head really executes embedding/projection through this interface.
        if self.zero_override:
            latent = torch.zeros_like(latent)
            delta = torch.zeros_like(delta)
        return dict(latent=latent, delta_logits=delta)


class Engine:
    def __init__(self, model, tokenizer, head, special_ids, check=lambda: None):
        self.model, self.tokenizer, self.head = model, tokenizer, head
        self.zero = FrozenHeadAdapter(head.head, zero_override=True)
        self.special = set(special_ids)
        self.check = check
        self.nfe = 0
        self.device = next(model.parameters()).device
        self._hidden = None
        # Capture the actual final normalized H from the same and only X0 call.
        self.hook = model.model.norm.register_forward_hook(self._capture)

    def _capture(self, _module, _args, output):
        self._hidden = output.detach()

    def forward(self, canvas, attention, positions):
        self.check()
        output = self.model(canvas, attention, positions, use_cache=False, return_dict=True)
        self.nfe += 1
        return output.logits, self._hidden

    def fresh(self, state, tokens=None):
        """Only this chain's legal committed prefix is written into an X0 copy."""
        x = state['x0'].clone()
        visible = state['history_mask'][0, 0].nonzero().flatten()
        ids = state['token_ids'] if tokens is None else tokens
        for slot in visible.tolist():
            token = int(ids[0, slot])
            if token in self.special:
                raise ValueError('nonordinary token in teacher history')
            x[0, state['position_ids'][0, slot]] = token
        q = int(state['query_indices'][0, 0])
        target = int(state['position_ids'][0, q])
        logits, _ = self.forward(x, state['attention'], state['time_ids'])
        return self.action_mask(logits[0, max(0, target-1)].float(), state['expand_masked'])

    def action_mask(self, logits, expand_masked):
        logits = logits.clone()
        if expand_masked:
            logits[..., self.tokenizer.expand_id] -= 1e9
        return logits

    def correction(self, state, zero=False):
        adapter = self.zero if zero else self.head
        fields = ['old_hidden', 'position_ids', 'chain_ids', 'selected_mask', 'query_indices',
                  'old_logits', 'token_ids', 'token_state', 'history_mask']
        return adapter(**{k: state[k] for k in fields})['delta_logits'][0, 0]

    @torch.inference_mode()
    def decode(self, prefix_ids, suffix_ids, case, arm, order='global', observer=None, detailed=False):
        p = len(prefix_ids)
        mask, eos, expand = self.tokenizer.mask_id, self.tokenizer.eos_id, self.tokenizer.expand_id
        maximum = min(2048, p + 64 + len(suffix_ids))
        ids = list(prefix_ids) + [mask]*64 + list(suffix_ids)
        if len(ids) > maximum:
            raise ValueError('input exceeds frozen canvas maximum')
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        dynamic, expansion_budget = 64, 64
        start_nfe = self.nfe
        head_calls = self.head.calls + self.zero.calls
        traces, total_normal, total_expand, total_delete = [], 0, 0, 0
        key_digest = hashlib.sha256()
        for step in range(256):
            self.check()
            real = p + dynamic + len(suffix_ids)
            active = torch.arange(maximum, device=x.device)[None, :] < real
            mask_index = (x == mask) & active
            allpos = mask_index[0].nonzero().flatten()
            if not allpos.numel():
                break
            time_ids = active.long().cumsum(-1)-1
            time_ids.masked_fill_(~active, 1)
            attention = active[:, None, None, :] & active[:, None, :, None]
            raw, hidden = self.forward(x, attention, time_ids)
            full_old = self.action_mask(raw[0, (allpos-1).clamp_min(0)].float(), real == maximum or expansion_budget == 0)
            deploy = distribution(full_old)
            probability = deploy.exp()
            confidence = (probability * torch.log(probability + 1e-10)).sum(-1)
            selected_positions = select_positions(allpos.tolist(), confidence.tolist(), requested_k=1 if arm == 'K1' else 4,
                policy=GLOBAL_CONFIDENCE if order == 'global' else LEFT_TO_RIGHT_FRONTIER)
            selected_positions.sort()
            ranks = [allpos.tolist().index(v) for v in selected_positions]
            pos = torch.tensor([selected_positions], device=x.device)
            chains = torch.tensor([chain_ids(selected_positions)], device=x.device)
            selected = torch.ones_like(pos, dtype=torch.bool)
            old_hidden = aligned_hidden(hidden, pos[0])
            old_all = full_old[ranks].unsqueeze(0)
            tokens = torch.zeros_like(pos)
            states = torch.zeros_like(pos)
            branch_logp = torch.full_like(old_all, -torch.inf)
            x0 = x.clone()
            committed, events = [], []
            barrier = None
            memory_floor = -1
            # This distribution chooses coverage from X0 and remains fixed for the round.
            for slot, target in enumerate(selected_positions):
                key = sample_key(case, step, target)
                key_digest.update(json.dumps(key, separators=(',', ':')).encode())
                base = int(inverse_cdf(deploy[ranks[slot]], uniform(key)))
                query = torch.tensor([[slot]], device=x.device)
                hm = history_mask(pos, chains, selected, states, query)
                hm &= torch.arange(pos.shape[1], device=x.device)[None, None, :] > memory_floor
                state = dict(x0=x0, attention=attention, time_ids=time_ids, old_hidden=old_hidden,
                    position_ids=pos, chain_ids=chains, selected_mask=selected, query_indices=query,
                    old_logits=old_all[:, slot:slot+1], selected_old_logits=old_all,
                    token_ids=tokens.clone(), token_state=states.clone(), history_mask=hm,
                    branch_logp=branch_logp.clone(), expand_masked=real == maximum or expansion_budget == 0,
                    round_index=step, case=case, order=order, arm=arm, key=key, prefix_length=p,
                    dynamic_length=dynamic, maximum=maximum)
                if observer is not None:
                    observer('query', state)
                visible = int(hm.sum())
                final, used = base, 'anchor' if visible == 0 else 'stale'
                if base in (expand, eos):
                    barrier = dict(position=target, token=base, reason='base_structure')
                    if base == expand and (real == maximum or expansion_budget == 0):
                        final = mask
                        barrier = None
                    else:
                        committed.append((target, base))
                        events.append(dict(slot=slot, position=target, chain=int(chains[0, slot]), raw=base, final=base,
                            visible_slots=hm[0, 0].nonzero().flatten().tolist(), key=key, mode='base_structure'))
                        break
                elif visible and base not in self.special:
                    if arm in ('Z4', 'H4-old'):
                        corrected = old_all[0, slot] + self.correction(state, zero=arm == 'Z4')
                        final, used = draw(corrected, key), 'zero' if arm == 'Z4' else 'old_head'
                    elif arm == 'F4-chain':
                        corrected = self.fresh(state)
                        final, used = draw(corrected, key), 'chain_local_fresh'
                    else:
                        corrected = old_all[0, slot]
                    if arm in ('Z4', 'H4-old', 'F4-chain') and final in (expand, eos):
                        barrier = dict(position=target, token=final, reason='corrected_structure_refresh_only')
                        events.append(dict(slot=slot, position=target, chain=int(chains[0, slot]), raw=base, final=final,
                            visible_slots=hm[0, 0].nonzero().flatten().tolist(), key=key, mode=used, committed=False))
                        break
                else:
                    corrected = old_all[0, slot]
                if used in ('anchor', 'stale'):
                    corrected = old_all[0, slot]
                branch_logp[0, slot] = distribution(corrected)
                tokens[0, slot] = final
                # All tokenizer special/unassigned tokens are excluded from ordinary memory.
                states[0, slot] = int(final not in self.special)
                if final in self.special:
                    memory_floor = slot
                if final != mask:
                    committed.append((target, final))
                events.append(dict(slot=slot, position=target, chain=int(chains[0, slot]), raw=base, final=final,
                    visible_slots=hm[0, 0].nonzero().flatten().tolist(), key=key, mode=used, committed=final != mask))
            # Commit once; teacher subcalls never mutate the actual canvas.
            for target, token in committed:
                x[0, target] = token
            normal = sum(t not in self.special for _, t in committed)
            expand_indices, delete_indices = [], []
            if barrier and barrier['reason'] == 'base_structure':
                target, token = barrier['position'], barrier['token']
                if token == expand:
                    x = _apply_one_expand(x, target, mask)[:, :maximum]
                    dynamic += 1
                    expansion_budget -= 1
                    expand_indices = [target]
                elif token == eos:
                    # Same global EOS broadcast and frozen-mask deletion provenance as v2.
                    first_eos = int((x[0] == eos).nonzero()[0])
                    tail = torch.arange(maximum, device=x.device) >= first_eos
                    x[0].masked_fill_(tail & mask_index[0], eos)
                    delete_indices = ((x == eos) & mask_index)[0].nonzero().flatten().tolist()
                    for index in reversed(delete_indices):
                        x = _apply_one_delete(x, index, mask)
                        dynamic -= 1
            total_normal += normal
            total_expand += len(expand_indices)
            total_delete += len(delete_indices)
            trace = dict(round=step, selected=selected_positions, chains=chains[0].tolist(),
                chain_lengths=[chain_ids(selected_positions).count(c) for c in sorted(set(chain_ids(selected_positions)))],
                events=events, submitted=committed, actual_submitted_count=len(committed), barrier=barrier,
                expand=expand_indices, delete=delete_indices, dynamic_length=dynamic,
                gaps=[b-a for a,b in zip(selected_positions, selected_positions[1:]) if b-a != 1])
            if detailed:
                trace.update(x0=x0[0].tolist(), x_after=x[0].tolist(), time_ids=time_ids[0].tolist())
            traces.append(trace)
            if observer is not None:
                observer('round', trace)
        completion = x[0, p:p+dynamic].tolist()
        remaining = completion.count(mask)
        return dict(output=self.tokenizer.decode(completion, skip_special_tokens=True), completion_token_ids=completion,
            forward_count=self.nfe-start_nfe, head_calls=self.head.calls+self.zero.calls-head_calls,
            outer_rounds=len(traces), actual_submitted_count=sum(t['actual_submitted_count'] for t in traces),
            normal_token_count=total_normal, expand_count=total_expand, delete_count=total_delete,
            final_remaining_masks=remaining, stop_reason='complete' if not remaining else 'step_budget',
            sampling_key_sha256=key_digest.hexdigest(), action_trace=traces)
