"""Independent instrumented copy of 660f296 threshold decoder.

The historical entry point is unchanged. Snapshots are taken before candidate
processing, including fixed old logits and in-round committed memory.
"""
from collections import Counter
import copy
import hashlib
import json
import torch
from experiments.threshold_eval.adapter import Engine as Historical
from experiments.context_foundation.adapter import (
    aligned_hidden, chain_ids, distribution, draw, history_mask,
    inverse_cdf, sample_key, uniform,
)
from experiments.dreamon_singleline_order_parallel import (
    select_positions, GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER,
    _apply_one_expand, _apply_one_delete,
)


def clone_state(value, device='cpu'):
    if torch.is_tensor(value): return value.detach().to(device).clone()
    if isinstance(value, dict): return type(value)({k:clone_state(v,device) for k,v in value.items()})
    if isinstance(value, list): return [clone_state(v,device) for v in value]
    if isinstance(value, tuple): return tuple(clone_state(v,device) for v in value)
    if isinstance(value, set): return set(value)
    return copy.deepcopy(value)


def key_for(case, step, target, seed):
    key=sample_key(case,step,target)
    key[1]=seed
    return key


def query_state(s):
    slot=s['slot']
    query=torch.tensor([[slot]],device=s['x'].device)
    hm=history_mask(s['pos'],s['chains'],s['selected'],s['states'],query)
    hm &= torch.arange(s['pos'].shape[1],device=s['x'].device)[None,None,:]>s['memory_floor']
    return dict(x0=s['x0'],attention=s['attention'],time_ids=s['time_ids'],old_hidden=s['old_hidden'],
        position_ids=s['pos'],chain_ids=s['chains'],selected_mask=s['selected'],query_indices=query,
        old_logits=s['old_all'][:,slot:slot+1],selected_old_logits=s['old_all'],
        token_ids=s['tokens'],token_state=s['states'],history_mask=hm,
        expand_masked=s['expand_masked'])


def integrity(r):
    st=r['statistics']
    assert st['selected_candidates']==st['evaluated_candidates']+st['chain_skipped']+r['barrier_unreached']
    assert st['extra_ordinary_attempted']==st['extra_ordinary_accepted']+st['extra_ordinary_rejected']
    assert r['dynamic_length']==64+r['expand_count']-r['delete_count']
    assert r['expansion_budget']==64-r['expand_count']
    h={int(k):v for k,v in r['normal_submissions_histogram'].items()}
    assert r['outer_rounds']==sum(h.values())
    assert r['outer_rounds']==r['normal_token_count']-sum(max(k-1,0)*v for k,v in h.items())+h.get(0,0)


class Engine(Historical):
    @torch.inference_mode()
    def decode(self,prefix_ids=None,suffix_ids=None,case=None,method=None,order='global',requested_k=4,
               threshold=None,seed=42,capture=None,snapshot=None,intervention=None,continuation_seed=None):
        mask,eos,expand=self.tokenizer.mask_id,self.tokenizer.eos_id,self.tokenizer.expand_id
        if snapshot is None:
            p=len(prefix_ids); maximum=min(2048,p+64+len(suffix_ids))
            ids=list(prefix_ids)+[mask]*64+list(suffix_ids)
            if len(ids)>maximum: raise ValueError('input exceeds frozen canvas maximum')
            s=dict(x=torch.tensor([ids],device=self.device),p=p,maximum=maximum,suffix_length=len(suffix_ids),
                dynamic=64,budget=64,case=case,method=method,order=order,k=requested_k,threshold=threshold,
                seed=seed,step=0,cursor=None,forward_count=0,head_calls=0,normal=0,expands=0,deletes=0,
                barrier_unreached=0,traces=[],keys=[],histogram=Counter(),lineage=list(range(maximum)),next_lineage=maximum,
                stats=Counter({k:0 for k in ('selected_candidates','evaluated_candidates','chain_skipped',
                    'extra_ordinary_attempted','extra_ordinary_accepted','extra_ordinary_rejected','mask_noop',
                    'special_nonstructural','base_structure_rounds','corrected_structure_rounds')}))
        else:
            s=clone_state(snapshot,self.device)
        first=True
        intervention=intervention or {}
        while s['step']<256:
            self.check()
            if s['cursor'] is None:
                real=s['p']+s['dynamic']+s['suffix_length']
                active=torch.arange(s['maximum'],device=self.device)[None,:]<real
                mi=(s['x']==mask)&active
                allpos=mi[0].nonzero().flatten()
                if not allpos.numel(): break
                time_ids=active.long().cumsum(-1)-1; time_ids.masked_fill_(~active,1)
                attention=active[:,None,None,:]&active[:,None,:,None]
                raw,hidden=self.forward(s['x'],attention,time_ids);s['forward_count']+=1
                full_old=self.action_mask(raw[0,(allpos-1).clamp_min(0)].float(),real==s['maximum'] or s['budget']==0)
                deploy=distribution(full_old); probability=deploy.exp()
                confidence=(probability*torch.log(probability+1e-10)).sum(-1)
                policy=GLOBAL_CONFIDENCE if s['order']=='global' else LEFT_TO_RIGHT_FRONTIER
                positions=select_positions(allpos.tolist(),confidence.tolist(),requested_k=s['k'],policy=policy)
                positions.sort()
                ranks=[allpos.tolist().index(v) for v in positions]
                pos=torch.tensor([positions],device=self.device)
                s.update(real=real,active=active,mask_index=mi,time_ids=time_ids,attention=attention,
                    positions=positions,pos=pos,chains=torch.tensor([chain_ids(positions)],device=self.device),
                    selected=torch.ones_like(pos,dtype=torch.bool),old_hidden=aligned_hidden(hidden,pos[0]),
                    old_all=full_old[ranks].unsqueeze(0),deploy=deploy[ranks],tokens=torch.zeros_like(pos),states=torch.zeros_like(pos),
                    x0=s['x'].clone(),committed=[],events=[],barrier=None,memory_floor=-1,rejected_chains=set(),
                    cursor=0,traversal=list(range(len(positions))),exempt=0,
                    scores=[float(confidence[r]) for r in ranks],
                    s2=sorted(select_positions(allpos.tolist(),confidence.tolist(),requested_k=2,policy=policy)),
                    expand_masked=real==s['maximum'] or s['budget']==0,
                    round_dynamic=s['dynamic'],round_budget=s['budget'])
                s['stats']['selected_candidates']+=len(positions)
            if first and intervention.get('kind')=='order':
                a2=min(s['s2']); slot=s['positions'].index(a2)
                if intervention['a2_first']: s['traversal']=[slot]+[i for i in s['traversal'] if i!=slot]
                if intervention['a2_exempt']: s['exempt']=slot
            while s['cursor']<len(s['traversal']):
                slot=s['traversal'][s['cursor']];s['slot']=slot
                target=s['positions'][slot];chain=int(s['chains'][0,slot])
                if chain in s['rejected_chains']:
                    s['stats']['chain_skipped']+=1
                    s['events'].append(dict(slot=slot,position=target,chain=chain,reason='chain_skipped',processed=False,lineage=s['lineage'][target]))
                    s['cursor']+=1
                    continue
                # capture must deep-copy tensors immediately, before any memory mutation.
                if capture is not None: capture(s)
                event_seed=s['seed']
                if not first and continuation_seed is not None:event_seed=continuation_seed
                key=key_for(s['case'],s['step'],target,event_seed);u=uniform(key)
                s['keys'].append(key)
                base=int(inverse_cdf(s['deploy'][slot],u));original_base=base
                s['stats']['evaluated_candidates']+=1
                qs=query_state(s);visible=qs['history_mask'][0,0].nonzero().flatten().tolist()
                eligible=bool(visible) and base not in self.special
                old=s['old_all'][0,slot]
                base_probability=float(torch.softmax(old.float(),-1)[base])
                corrected=old;final=base;used='anchor' if not visible else 'stale'
                called=False;forced=False;defer=False;veto=False
                this=intervention if first else {}
                first=False
                if this.get('kind')=='structure':
                    if this['action']=='defer':defer=True
                    elif this['action']=='veto':
                        veto=True; alternative=old.clone();alternative[base]=-torch.inf
                        base=int(inverse_cdf(distribution(alternative),u)); final=base
                        eligible=bool(visible) and base not in self.special
                if this.get('kind')=='head':
                    if not eligible: raise ValueError('head intervention requires factual ordinary candidate and visible predecessor')
                    forced=True
                    final=int(this['candidate']);corrected=this['score_logits'].to(self.device)
                    called=this['candidate_source']=='q' or this['score_source']=='q'
                    used='old_head' if called else ('anchor' if not visible else 'stale')
                    s['head_calls']+=int(called)
                elif not defer and base not in (expand,eos) and eligible and s['method'] in ('trained','zero'):
                    corrected=old+self.correction(qs,zero=s['method']=='zero');called=True;s['head_calls']+=1
                    final=draw(corrected,key);used='zero' if s['method']=='zero' else 'old_head'
                ev=dict(slot=slot,position=target,lineage=s['lineage'][target],chain=chain,raw=original_base,final=final,resampled_base=base if veto else None,
                    visible_slots=visible,predecessor_slot=max(visible,key=lambda i:s['positions'][i]) if visible else None,
                    predecessor_token=int(s['tokens'][0,max(visible,key=lambda i:s['positions'][i])]) if visible else None,
                    key=key,u=u,mode=used,head_eligible=eligible,head_called=called,raw_probability=base_probability,
                    final_probability=float(torch.softmax(corrected.float(),-1)[final]),processed=True,committed=False,
                    exempt=slot==s['exempt'],candidate_probability=None,veto=veto)
                if defer:
                    ev['reason']='structure_deferred';s['events'].append(ev)
                    s['barrier']=dict(position=target,token=base,reason='structure_deferred');break
                if not forced and base in (expand,eos):
                    s['barrier']=dict(position=target,token=base,reason='base_structure')
                    if base==expand and s['expand_masked']:
                        final=mask;ev['final']=mask;s['barrier']=None
                    else:
                        s['committed'].append((target,base));ev.update(mode='base_structure',reason='base_structure',committed=True)
                        s['events'].append(ev);break
                elif called and final in (expand,eos):
                    s['barrier']=dict(position=target,token=final,reason='corrected_structure_refresh_only')
                    ev['reason']='corrected_structure_refresh_only';s['events'].append(ev);break
                if slot!=s['exempt'] and final not in self.special:
                    s['stats']['extra_ordinary_attempted']+=1
                    cp=float(torch.softmax(corrected.float(),-1)[final]) if s['threshold'] is not None else None
                    ev['candidate_probability']=cp
                    if s['threshold'] is not None and cp<s['threshold']:
                        s['stats']['extra_ordinary_rejected']+=1
                        if this.get('kind')!='no_propagation':s['rejected_chains'].add(chain)
                        ev.update(rejected=True,reason='threshold_rejected');s['events'].append(ev);s['cursor']+=1;continue
                    s['stats']['extra_ordinary_accepted']+=1
                s['stats']['mask_noop']+=int(final==mask)
                s['stats']['special_nonstructural']+=int(final in self.special and final not in (mask,eos,expand))
                s['tokens'][0,slot]=final;s['states'][0,slot]=int(final not in self.special)
                if final in self.special:s['memory_floor']=slot
                if final!=mask:s['committed'].append((target,final))
                ev.update(committed=final!=mask,reason='exempt' if slot==s['exempt'] and final not in self.special else 'ordinary_accepted' if final not in self.special else 'mask_noop' if final==mask else 'other_special')
                s['events'].append(ev);s['cursor']+=1
            if s['barrier']:
                for pending in s['traversal'][s['cursor']+1:]:
                    target=s['positions'][pending]
                    s['events'].append(dict(slot=pending,position=target,lineage=s['lineage'][target],chain=int(s['chains'][0,pending]),reason='barrier_unreached',processed=False))
                    s['barrier_unreached']+=1
            for target,token in s['committed']:s['x'][0,target]=token
            normal=sum(t not in self.special for _,t in s['committed'])
            expand_indices=[];delete_indices=[]
            barrier=s['barrier']
            if barrier and barrier['reason']=='base_structure':
                target,token=barrier['position'],barrier['token']
                if token==expand:
                    s['x']=_apply_one_expand(s['x'],target,mask)[:,:s['maximum']]
                    s['dynamic']+=1;s['budget']-=1;expand_indices=[target]
                    # original expand replaces target by two masks.
                    s['lineage'].insert(target+1,s['next_lineage']);s['next_lineage']+=1;s['lineage']=s['lineage'][:s['maximum']]
                elif token==eos:
                    first_eos=int((s['x'][0]==eos).nonzero()[0])
                    tail=torch.arange(s['maximum'],device=self.device)>=first_eos
                    s['x'][0].masked_fill_(tail&s['mask_index'][0],eos)
                    delete_indices=((s['x']==eos)&s['mask_index'])[0].nonzero().flatten().tolist()
                    for index in reversed(delete_indices):
                        s['x']=_apply_one_delete(s['x'],index,mask);s['dynamic']-=1
                        s['lineage'].pop(index);s['lineage'].append(s['next_lineage']);s['next_lineage']+=1
            s['stats']['base_structure_rounds']+=int(barrier is not None and barrier['reason']=='base_structure')
            s['stats']['corrected_structure_rounds']+=int(barrier is not None and barrier['reason']=='corrected_structure_refresh_only')
            s['histogram'][normal]+=1;s['normal']+=normal;s['expands']+=len(expand_indices);s['deletes']+=len(delete_indices)
            s['traces'].append(dict(round=s['step'],selected=s['positions'],chains=s['chains'][0].tolist(),
                scores=s['scores'],s2=s['s2'],processing_order=[s['positions'][i] for i in s['traversal']],
                exempt_position=s['positions'][s['exempt']],exempt_reached=any(e.get('processed') and e['slot']==s['exempt'] for e in s['events']),
                events=s['events'],submitted=s['committed'],barrier=barrier,expand=expand_indices,delete=delete_indices,
                dynamic_length=s['dynamic'],expansion_budget=s['budget'],dynamic_before=s['round_dynamic'],budget_before=s['round_budget'],
                x0=s['x0'][0].tolist(),x_after=s['x'][0].tolist(),active_length=s['real'],
                normal_submissions=normal,round_end=barrier['reason'] if barrier else 'selected_exhausted'))
            s['step']+=1;s['cursor']=None
        completion=s['x'][0,s['p']:s['p']+s['dynamic']].tolist();remaining=completion.count(mask)
        digest=hashlib.sha256()
        for key in s['keys']:digest.update(json.dumps(key,separators=(',',':')).encode())
        result=dict(output=self.tokenizer.decode(completion,skip_special_tokens=True),completion_token_ids=completion,
            forward_count=s['forward_count'],head_calls=s['head_calls'],outer_rounds=s['step'],statistics=dict(s['stats']),
            normal_submissions_histogram=dict(s['histogram']),normal_token_count=s['normal'],expand_count=s['expands'],delete_count=s['deletes'],
            final_remaining_masks=remaining,stop_reason='complete' if not remaining else 'step_budget',sampling_key_sha256=digest.hexdigest(),
            action_trace=s['traces'],barrier_unreached=s['barrier_unreached'],dynamic_length=s['dynamic'],expansion_budget=s['budget'])
        integrity(result)
        return result
