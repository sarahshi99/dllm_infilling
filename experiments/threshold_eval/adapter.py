"""Threshold decoder derived from the audited common-u decoder; history stays immutable."""
from collections import Counter
import hashlib
import json
import torch
from experiments.context_foundation.adapter import (
    Engine as ReferenceEngine, FrozenHeadAdapter, aligned_hidden, chain_ids,
    distribution, draw, history_mask, inverse_cdf, sample_key, uniform,
)
from experiments.dreamon_singleline_order_parallel import (
    GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER, select_positions,
    _apply_one_expand, _apply_one_delete,
)


class Engine(ReferenceEngine):
    def __init__(self, model, tokenizer, head, special_ids, check=lambda: None, max_gen_len=64):
        if max_gen_len not in (64, 128):
            raise ValueError('max_gen_len must be 64 or 128')
        self.max_gen_len = max_gen_len
        self.model, self.tokenizer, self.head = model, tokenizer, head
        self.zero = FrozenHeadAdapter(head.head, zero_override=True) if head is not None else None
        self.special = set(special_ids)
        self.check, self.nfe = check, 0
        self.device = next(model.parameters()).device
        self._hidden = None
        self.hook = model.model.norm.register_forward_hook(self._capture)

    def call_count(self):
        return (self.head.calls if self.head is not None else 0) + (self.zero.calls if self.zero is not None else 0)

    @torch.inference_mode()
    def decode(self, prefix_ids, suffix_ids, case, method, order='global', requested_k=4,
               threshold=None, observer=None, detailed=False, record_trace=True):
        if requested_k not in (1, 2, 4):
            raise ValueError('requested_k must be 1, 2 or 4')
        if method not in ('nohead', 'trained', 'zero'):
            raise ValueError('unknown method')
        if threshold not in (None, .80, .90, .95):
            raise ValueError('unknown threshold')
        tracing = record_trace or observer is not None
        arm = {'nohead':'nohead', 'trained':'H4-old', 'zero':'Z4'}[method]
        p = len(prefix_ids)
        mask, eos, expand = self.tokenizer.mask_id, self.tokenizer.eos_id, self.tokenizer.expand_id
        maximum = min(2048, p + self.max_gen_len + len(suffix_ids))
        ids = list(prefix_ids) + [mask]*64 + list(suffix_ids)
        if len(ids) > maximum:
            raise ValueError('input exceeds frozen canvas maximum')
        x = torch.tensor([ids + [mask]*(maximum-len(ids))], dtype=torch.long, device=self.device)
        dynamic, expansion_budget = 64, self.max_gen_len
        start_nfe = self.nfe
        head_calls = self.call_count()
        traces, total_normal, total_expand, total_delete = [], 0, 0, 0
        key_digest = hashlib.sha256()
        stats = Counter({k: 0 for k in ('selected_candidates','evaluated_candidates','chain_skipped','extra_ordinary_attempted','extra_ordinary_accepted','extra_ordinary_rejected','mask_noop','special_nonstructural','base_structure_rounds','corrected_structure_rounds')})
        submission_histogram = Counter()
        round_count = 0
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
            selected_positions = select_positions(allpos.tolist(), confidence.tolist(), requested_k=requested_k,
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
            branch_logp = torch.full_like(old_all, -torch.inf) if observer is not None else None
            x0 = x.clone() if detailed or observer is not None else x
            committed, events = [], []
            barrier = None
            memory_floor = -1
            rejected_chain = None
            stats["selected_candidates"] += len(selected_positions)
            # This distribution chooses coverage from X0 and remains fixed for the round.
            for slot, target in enumerate(selected_positions):
                if rejected_chain == int(chains[0, slot]):
                    stats['chain_skipped'] += 1
                    continue
                key = sample_key(case, step, target)
                if record_trace:
                    key_digest.update(json.dumps(key, separators=(',', ':')).encode())
                base = int(inverse_cdf(deploy[ranks[slot]], uniform(key)))
                stats['evaluated_candidates'] += 1
                query = torch.tensor([[slot]], device=x.device)
                hm = history_mask(pos, chains, selected, states, query)
                hm &= torch.arange(pos.shape[1], device=x.device)[None, None, :] > memory_floor
                state = dict(x0=x0, attention=attention, time_ids=time_ids, old_hidden=old_hidden,
                    position_ids=pos, chain_ids=chains, selected_mask=selected, query_indices=query,
                    old_logits=old_all[:, slot:slot+1], selected_old_logits=old_all,
                    token_ids=tokens, token_state=states, history_mask=hm,
                    branch_logp=branch_logp, expand_masked=real == maximum or expansion_budget == 0,
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
                        if tracing:
                            events.append(dict(slot=slot, position=target, chain=int(chains[0, slot]), raw=base, final=base,
                                visible_slots=hm[0, 0].nonzero().flatten().tolist(), key=key, mode='base_structure'))
                        break
                elif visible and base not in self.special:
                    if arm in ('Z4', 'H4-old'):
                        corrected = old_all[0, slot] + self.correction(state, zero=arm == 'Z4')
                        final, used = draw(corrected, key), 'zero' if arm == 'Z4' else 'old_head'
                    else:
                        corrected = old_all[0, slot]
                    if arm in ('Z4', 'H4-old') and final in (expand, eos):
                        barrier = dict(position=target, token=final, reason='corrected_structure_refresh_only')
                        if tracing:
                            events.append(dict(slot=slot, position=target, chain=int(chains[0, slot]), raw=base, final=final,
                                visible_slots=hm[0, 0].nonzero().flatten().tolist(), key=key, mode=used, committed=False))
                        break
                else:
                    corrected = old_all[0, slot]
                if used in ('anchor', 'stale'):
                    corrected = old_all[0, slot]
                # T=1, after legal action masking, before deployment T/top-p.
                # Full vocabulary includes all still legal special tokens.
                candidate_probability = None
                if slot > 0 and final not in self.special:
                    stats['extra_ordinary_attempted'] += 1
                    if threshold is not None:
                        candidate_probability = float(torch.softmax(corrected.float(), -1)[final])
                    if threshold is not None and candidate_probability < threshold:
                        stats['extra_ordinary_rejected'] += 1
                        rejected_chain = int(chains[0, slot])
                        if tracing:
                            events.append(dict(slot=slot, position=target, chain=rejected_chain, raw=base,
                                final=final, visible_slots=hm[0, 0].nonzero().flatten().tolist(),
                                key=key, mode=used, committed=False, rejected=True,
                                candidate_probability=candidate_probability))
                        continue
                    stats['extra_ordinary_accepted'] += 1
                # No distribution retention on the deployment/timing path.
                if observer is not None:
                    branch_logp[0, slot] = distribution(corrected)
                stats['mask_noop'] += int(final == mask)
                stats['special_nonstructural'] += int(final in self.special and final not in (mask,eos,expand))
                tokens[0, slot] = final
                # All tokenizer special/unassigned tokens are excluded from ordinary memory.
                states[0, slot] = int(final not in self.special)
                if final in self.special:
                    memory_floor = slot
                if final != mask:
                    committed.append((target, final))
                if tracing:
                    events.append(dict(slot=slot, position=target, chain=int(chains[0, slot]), raw=base, final=final,
                        visible_slots=hm[0, 0].nonzero().flatten().tolist(), key=key, mode=used, committed=final != mask,
                        candidate_probability=candidate_probability))
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
            stats['base_structure_rounds'] += int(barrier is not None and barrier['reason'] == 'base_structure')
            stats['corrected_structure_rounds'] += int(barrier is not None and barrier['reason'] == 'corrected_structure_refresh_only')
            submission_histogram[normal] += 1
            round_count += 1
            total_normal += normal
            total_expand += len(expand_indices)
            total_delete += len(delete_indices)
            if tracing:
                trace = dict(round=step, selected=selected_positions, chains=chains[0].tolist(),
                    chain_lengths=[chain_ids(selected_positions).count(c) for c in sorted(set(chain_ids(selected_positions)))],
                    events=events, submitted=committed, actual_submitted_count=len(committed), barrier=barrier,
                    expand=expand_indices, delete=delete_indices, dynamic_length=dynamic,
                    gaps=[b-a for a,b in zip(selected_positions, selected_positions[1:]) if b-a != 1])
                if detailed:
                    trace.update(x0=x0[0].tolist(), x_after=x[0].tolist(), time_ids=time_ids[0].tolist())
                if record_trace:
                    traces.append(trace)
                if observer is not None:
                    observer('round', trace)
        completion = x[0, p:p+dynamic].tolist()
        remaining = completion.count(mask)
        return dict(output=self.tokenizer.decode(completion, skip_special_tokens=True), completion_token_ids=completion,
            forward_count=self.nfe-start_nfe, head_calls=self.call_count()-head_calls,
            outer_rounds=round_count, actual_submitted_count=total_normal+stats['special_nonstructural']+stats['base_structure_rounds'],
            statistics=dict(stats), normal_submissions_histogram=dict(submission_histogram),
            normal_token_count=total_normal, expand_count=total_expand, delete_count=total_delete,
            final_remaining_masks=remaining, stop_reason='complete' if not remaining else 'step_budget',
            sampling_key_sha256=key_digest.hexdigest(), action_trace=traces)
