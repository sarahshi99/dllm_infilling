"""CPU-only discrete canvas replay; never samples or runs a model/test program."""
from dataclasses import dataclass

@dataclass
class Canvas:
    x: list
    length: int
    budget: int
    prefix_length: int
    suffix_length: int
    maximum: int

    def copy(self):
        return Canvas(list(self.x),self.length,self.budget,self.prefix_length,self.suffix_length,self.maximum)


def initial_canvas(prefix,suffix,mask_id):
    values=list(prefix)+[mask_id]*64+list(suffix)
    maximum=min(2048,len(values))
    return Canvas(values[:maximum]+[mask_id]*max(0,maximum-len(values)),64,64,len(prefix),len(suffix),maximum)


def submissions(trace,state,*,policy,mask_id,eos_id,expand_id,resolved=None):
    """Restore old partial frontier records via persistent final-token constraints."""
    resolved=resolved or {}
    if 'submitted_absolute' in trace:return [list(pair) for pair in trace['submitted_absolute']]
    if 'submitted' in trace:return [list(pair) for pair in trace['submitted']]
    raw={int(pos):token for pos,token in trace['selected_raw_relative']}
    positions=[int(pos) for pos in trace['legacy_trace']['selected_positions']]
    barrier=trace['barrier']
    result=[]
    for pos in positions:
        if policy=='left_to_right_frontier' and barrier is not None and pos>barrier:continue
        token=raw[pos]
        if pos==barrier:
            token=expand_id if trace['expand'] else eos_id
        if token is None:
            symbol=f"preserved-normal:{trace['step']}:{pos}"
            token=resolved.get(symbol,symbol)
        if token!=mask_id:result.append([state.prefix_length+pos,token])
    return result


def apply_round(state,trace,*,stage,arm,policy,mask_id,eos_id,expand_id,fixed,resolved=None):
    state=state.copy();x=state.x
    real=state.prefix_length+state.length+state.suffix_length
    masks=[i for i,t in enumerate(x) if i<real and t==mask_id]
    legacy=trace.get('legacy_trace')
    if legacy is not None:
        assert [i-state.prefix_length for i in masks]==legacy['active_mask_positions'],'legacy mask coordinates differ'
    submitted=submissions(trace,state,policy=policy,mask_id=mask_id,eos_id=eos_id,expand_id=expand_id,resolved=resolved)
    if stage=='B':
        for pos,token in submitted:
            assert pos in masks
            x[pos]=token
        return state,dict(expand=0,delete=0,delete_indices=[],submitted=submitted,masks=masks)
    for pos in masks:x[pos]=mask_id
    for pos,token in submitted:
        assert pos in masks, 'submitted position outside active mask'
        x[pos]=token
    raw=trace.get('selected_raw_absolute',trace.get('selected_raw_relative',[]))
    selected=[int(p) for p,t in raw]
    sequential_pair=stage=='C' and arm!='A' and len(selected)==2 and abs(selected[0]-selected[1])==1
    k4_chain=stage=='A' and arm!='A'
    native_global=policy=='global_confidence' and not sequential_pair and not k4_chain
    barrier=trace.get('barrier')
    barrier_pos=None if barrier is None else (barrier if isinstance(barrier,int) else barrier[0])
    # Historical frontier barrier and current traces both store a relative position.
    barrier_token=None
    if barrier_pos is not None:
        absolute=barrier_pos+state.prefix_length
        barrier_token=next((tok for pos,tok in submitted if pos==absolute),None)
    broadcast=native_global or barrier_token==eos_id
    if broadcast:
        first=next((i for i,t in enumerate(x) if t==eos_id),None)
        if first is not None:
            for pos in masks:
                if pos>=first:x[pos]=eos_id
    mask_set=set(masks)
    eligible=[i in mask_set and tok==eos_id for i,tok in enumerate(x)] if broadcast else [False]*len(x)
    if native_global:
        expands=[i for i,t in enumerate(x) if t==expand_id]
    elif barrier_token==expand_id:
        expands=[barrier_pos+state.prefix_length]
    else:expands=[]
    for pos in sorted(expands,reverse=True):
        x[pos:pos+1]=[mask_id,mask_id]
        del x[state.maximum:]
        eligible[pos:pos+1]=[False,False]
        del eligible[state.maximum:]
        state.length+=1;state.budget-=1
    if broadcast:
        deletes=[i for i in range(len(x)) if eligible[i]] if fixed else [i for i in masks if x[i]==eos_id]
    else:deletes=[]
    for pos in sorted(deletes,reverse=True):
        assert x[pos]==eos_id
        x.pop(pos);x.append(mask_id);state.length-=1
    return state,dict(expand=len(expands),delete=len(deletes),delete_indices=deletes,submitted=submitted,masks=masks)


def replay_old(row,start,*,mask_id,eos_id,expand_id,resolved=None):
    state=start;counts={'expand':0,'delete':0}
    for step,trace in enumerate(row['action_trace']):
        assert trace['step']==step
        state,actions=apply_round(state,trace,stage=row['stage'],arm=row['arm'],policy=row['policy'],
                                  mask_id=mask_id,eos_id=eos_id,expand_id=expand_id,fixed=False,resolved=resolved)
        for key in counts:
            counts[key]+=actions[key]
            if row['stage']!='B':assert actions[key]==trace[key],f'{key} count differs at {step}'
        if 'legacy_trace' in trace:
            old=trace['legacy_trace'];assert state.length==old['canvas_length']
            assert sum(t==mask_id for t in state.x[state.prefix_length:state.prefix_length+state.length])==old['unresolved_after']
    assert state.length==row['final_dynamic_length']
    assert len(row['action_trace'])==row['base_forward_count']
    assert counts['expand']==row['expand_count'] and counts['delete']==row['delete_count']
    return state


def compare_row(row,prefix,suffix,*,mask_id,eos_id,expand_id):
    start=initial_canvas(prefix,suffix,mask_id)
    old=replay_old(row,start,mask_id=mask_id,eos_id=eos_id,expand_id=expand_id)
    final=old.x[len(prefix):len(prefix)+old.length];expected=row['completion_token_ids']
    assert len(final)==len(expected)
    resolved={}
    for token,actual in zip(final,expected):
        if isinstance(token,str):
            assert token not in resolved or resolved[token]==actual
            assert actual not in [mask_id,eos_id,expand_id], 'inferred token was not ordinary'
            resolved[token]=actual
        else:assert token==actual, 'old final token ID mismatch'
    if resolved:
        old=replay_old(row,start,mask_id=mask_id,eos_id=eos_id,expand_id=expand_id,resolved=resolved)
    assert old.x[len(prefix):len(prefix)+old.length]==expected,'old full final token mismatch'
    assert old.x[:len(prefix)]==prefix,'prefix unexpectedly changed'
    assert old.x[len(prefix)+old.length:len(prefix)+old.length+len(suffix)]==suffix,'suffix unexpectedly changed'
    state=start
    for step,trace in enumerate(row['action_trace']):
        prior=state
        state,old_actions=apply_round(prior,trace,stage=row['stage'],arm=row['arm'],policy=row['policy'],mask_id=mask_id,eos_id=eos_id,expand_id=expand_id,fixed=False,resolved=resolved)
        new,new_actions=apply_round(prior,trace,stage=row['stage'],arm=row['arm'],policy=row['policy'],mask_id=mask_id,eos_id=eos_id,expand_id=expand_id,fixed=True,resolved=resolved)
        if state.x!=new.x or state.length!=new.length:
            differences=[i for i,(a,b) in enumerate(zip(state.x,new.x)) if a!=b]
            center=differences[0] if differences else len(prefix)+min(state.length,new.length)
            left=max(0,center-6);right=min(len(state.x),center+10)
            return dict(status='first_divergence',first_step=step,reconstructed_ordinary_tokens=len(resolved),old_replay='all rounds/counts/final token IDs/prefix/suffix verified',
                        difference=dict(window=[left,right],before=prior.x[left:right],old=state.x[left:right],fixed=new.x[left:right],old_length=state.length,fixed_length=new.length,
                                        old_actions=old_actions,fixed_actions=new_actions),reason='pre-expand EOS deletion provenance moved with insertion; stale old mask coordinate differed')
    return dict(status='proved_equivalent',first_step=None,reconstructed_ordinary_tokens=len(resolved),old_replay='all rounds/counts/final token IDs/prefix/suffix verified',
                reason='every round full canvas and dynamic length identical under old/fixed mutation; same inputs/sampling/RNG imply complete generation/official-score reuse')
