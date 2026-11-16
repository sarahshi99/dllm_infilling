"""Predeclared state selection and finite one-decision branch experiments."""
from collections import defaultdict
import gzip
import hashlib
import json
import time
import torch
from experiments.context_foundation.adapter import distribution, inverse_cdf, uniform
from experiments.context_foundation.common import atomic_json
from experiments.threshold_trace_fork.engine import clone_state, query_state, key_for
from experiments.threshold_trace_fork.run import (
    OUT,LOCAL,FOCUS,IDS,CONFIGS,generate,trace_read,write_gz,trace_path,compact,status,compare,
)


def stable(*parts):return hashlib.sha256(json.dumps([20261112,*parts],separators=(',',':')).encode()).hexdigest()


def loc(config,case,t,e):
    return dict(config=config,case=case,base_task_id=case.split('/')[2],round=t['round'],slot=e['slot'],position=e['position'])


def identity(v):return '::'.join(str(v[k]) for k in ('config','case','round','slot'))


def events(v):
    for t in v['action_trace']:
        for e in t['events']:
            if e.get('processed'):yield t,e


def action(e):
    if not e.get('processed',True):return ('unprocessed',e.get('reason'))
    if e.get('rejected'):return ('reject',e['chain'])
    if e.get('reason')=='corrected_structure_refresh_only':return ('head_refresh',)
    if e.get('reason')=='base_structure':return ('native_structure',e['final'])
    if e.get('reason')=='mask_noop':return ('mask_noop',)
    if e.get('committed'):return ('commit',e['final'])
    return ('none',)


def first_action_difference(a,b):
    for t,u in zip(a['action_trace'],b['action_trace']):
        if t['x0']!=u['x0'] or t['selected']!=u['selected']:return None
        aa={e['slot']:e for e in t['events']};bb={e['slot']:e for e in u['events']}
        for slot in range(len(t['selected'])):
            e,f=aa.get(slot),bb.get(slot)
            if e is None or f is None:return None
            if action(e)!=action(f):return t,e,f
    return None


def choose_panel(db):
    path=OUT/'state_panel.json'
    if path.exists():return json.loads(path.read_text())['selected']
    selected=[];pool=[];strata=[]
    for threshold in ('090','095'):
        for method in ('nohead','trained'):
            cid=f'global_k4_{method}_{threshold}';groups=defaultdict(lambda:defaultdict(list))
            for r in db.rows('replay'):
                if r['config']!=cid:continue
                v=trace_read(cid,r['task_id'])
                for t,e in events(v):
                    if e['head_eligible']:
                        item=loc(cid,r['task_id'],t,e);pool.append(item)
                        groups[item['base_task_id']][item['case']].append(item)
            # One state from each base task before any task is reused. Nested
            # ordering is fixed before outcomes; no event-count weighting.
            choices=[]
            for base in sorted(groups,key=lambda b:stable(cid,'base',b)):
                case=sorted(groups[base],key=lambda c:stable(cid,'case',c))[0]
                choices.append(sorted(groups[base][case],key=lambda e:stable(cid,'event',identity(e)))[0])
            chosen=choices[:16]
            selected.extend(dict(x,sample_group='panel') for x in chosen)
            strata.append(dict(config=cid,pool_events=sum(len(es) for cs in groups.values() for es in cs.values()),eligible_base_tasks=len(groups),selected=len(chosen),shortfall=16-len(chosen),shortfall_reason=None if len(chosen)==16 else 'insufficient eligible base tasks'))
    write_gz(OUT/'state_candidate_pool.json.gz',pool)
    atomic_json(path,dict(selection_seed=20261112,rule='threshold x K4 host; sort base-task, then case, then eligible event by stable seeded ordering; one per base task; select before forks',strata=strata,candidate_pool='state_candidate_pool.json.gz',selected=selected))
    assert len(selected)==64
    return selected


def choose_special(db):
    head=[];decoder=[];na=[];common_checks=[]
    case=FOCUS[0];trained=trace_read('global_k4_trained_095',case);nohead=trace_read('global_k4_nohead_095',case)
    for t,e in events(trained):
        if e['head_called']:head.append(dict(loc('global_k4_trained_095',case,t,e),sample_group='139_factual_head'))
    diff=first_action_difference(nohead,trained)
    if diff:
        t,e,f=diff
        if e['head_eligible']:
            a=loc('global_k4_nohead_095',case,t,e);b=loc('global_k4_trained_095',case,t,f)
            head.append(dict(a,sample_group='139_common_reverse'));common_checks.append(dict(a=a,b=b,group='139_common_reverse'))
        else:na.append(dict(group='139_common_reverse',reason='first action difference is not eligible'))
    else:na.append(dict(group='139_common_reverse',reason='no common pre-action difference located'))
    before=[]
    for t,e in events(nohead):
        if e['reason']=='base_structure' and e['final']==nohead.get('expand_token_id',-1):break
        if t['expand']: # keep eligible events strictly before the native expand slot
            native=next(z for z in t['events'] if z.get('reason')=='base_structure')
            if e['slot']>=native['slot']:break
        if e['head_eligible']:before.append(loc('global_k4_nohead_095',case,t,e))
        if t['expand'] and e==t['events'][-1]:break
    # Stop above at the first expand; traces explicitly include native expand indices.
    first_expand=next((t['round'] for t in nohead['action_trace'] if t['expand']),None)
    before=[v for v in before if first_expand is not None and v['round']<=first_expand]
    if before:head.append(dict(before[-1],sample_group='139_pre_expand'))
    else:na.append(dict(group='139_pre_expand',reason='no eligible state before first native expand'))
    paired={r['config']+'::'+r['task_id']:r for r in db.rows('replay')}
    flips=[]
    for r in db.rows('replay'):
        if r['config']!='global_k4_nohead_090':continue
        h=paired['global_k4_trained_090::'+r['task_id']]
        if h['passed']==r['passed']:continue
        case=r['task_id'];a=trace_read(r['config'],case);b=trace_read(h['config'],case)
        flips.append(dict(case=case,nohead_passed=r['passed'],trained_passed=h['passed']))
        write_gz(LOCAL/('flip_traces__'+case.replace('/','_')+'.json.gz'),dict(nohead=a,trained=b))
        diff=first_action_difference(a,b)
        if diff and diff[2].get('head_eligible'):
            t,e,f=diff;aa=loc(r['config'],case,t,e);bb=loc(h['config'],case,t,f)
            head.append(dict(bb,sample_group='quality_flip'));common_checks.append(dict(a=aa,b=bb,group='quality_flip'))
        else:na.append(dict(group='quality_flip',case=case,reason='no first common eligible head event'))
    for case in FOCUS:
        for threshold in ('090','095'):
            cid=f'global_k4_nohead_{threshold}';v=trace_read(cid,case)
            order=next((t for t in v['action_trace'] if min(t['s2'])!=min(t['selected']) and min(t['s2']) in t['selected']),None)
            if order:
                decoder.append(dict(loc(cid,case,order,order['events'][0]),sample_group='decoder',kind='order'))
            else:na.append(dict(config=cid,case=case,kind='order',reason='no a2 != a4 with a2 in S4'))
            reject=next(((t,e) for t,e in events(v) if e.get('rejected') and any(z['slot']>e['slot'] and z['chain']==e['chain'] for z in t['events'])),None)
            if reject:decoder.append(dict(loc(cid,case,*reject),sample_group='decoder',kind='no_propagation'))
            else:na.append(dict(config=cid,case=case,kind='no_propagation',reason='no rejected ordinary with a later same-chain selected slot'))
            for kind in ('expand','delete'):
                t=next((t for t in v['action_trace'] if t[kind]),None)
                if t:
                    e=next(e for e in t['events'] if e.get('reason')=='base_structure')
                    decoder.append(dict(loc(cid,case,t,e),sample_group='decoder',kind=kind))
                else:na.append(dict(config=cid,case=case,kind=kind,reason='native action absent'))
    atomic_json(OUT/'special_state_manifest.json',dict(head=head,decoder=decoder,not_applicable=na,common_state_checks=common_checks,quality_flips=flips))
    return head,decoder,common_checks


def snapshot_file(v):return LOCAL/('snapshot__'+v['config']+'__'+v['case'].replace('/','_')+'__'+str(v['round'])+'_'+str(v['slot'])+'.pt')


def capture_states(e,byid,db,items):
    groups=defaultdict(dict)
    for v in items:
        if not snapshot_file(v).exists():groups[(v['config'],v['case'])][(v['round'],v['slot'])]=v
    for (cid,case),wanted in groups.items():
        def capture(s):
            key=(s['step'],s['slot'])
            if key in wanted:
                v=wanted[key];torch.save(clone_state(s),snapshot_file(v))
        nfe=e.nfe;calls=e.call_count();start=time.perf_counter()
        result=generate(e,byid[case],CONFIGS[cid],capture=capture)
        compare(trace_read(cid,case),result)
        assert all(snapshot_file(v).exists() for v in wanted.values())
        for v in wanted.values():
            factual_snapshot=torch.load(snapshot_file(v),map_location='cpu',weights_only=False)
            compare(result,e.decode(snapshot=factual_snapshot))
        key=cid+'::'+case
        if not db.get('capture_cost',key):db.put('capture_cost',key,dict(config=cid,case=case,prefix_replay_forwards=e.nfe-nfe,prefix_replay_head_calls=e.call_count()-calls,elapsed_sec=time.perf_counter()-start,snapshots=len(wanted)))
    return [str(snapshot_file(v)) for v in items]


def same_state(a,b):
    # Exclude method label and historical cost counters; include every variable
    # that controls the upcoming action or later evolution, including logits/H.
    fields=['x','x0','mask_index','attention','time_ids','old_all','old_hidden','deploy','pos','chains','selected','tokens','states',
        'p','maximum','suffix_length','dynamic','budget','case','order','k','threshold','seed','step','slot','cursor','traversal','exempt',
        'committed','memory_floor','rejected_chains','barrier','expand_masked','lineage']
    return all(torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k] for k in fields)


def distributions(e,s):
    s=clone_state(s,e.device);qs=query_state(s)
    calls=e.call_count();q=s['old_all'][0,s['slot']]+e.correction(qs)
    shadow=e.call_count()-calls
    # Shadow head must not contaminate the deployed adapters' counters.
    e.head.calls-=shadow
    e.shadow_head_calls_total=getattr(e,'shadow_head_calls_total',0)+shadow
    p=s['old_all'][0,s['slot']]
    key=key_for(s['case'],s['step'],s['positions'][s['slot']],s['seed'])
    yp=int(inverse_cdf(distribution(p),uniform(key)));yq=int(inverse_cdf(distribution(q),uniform(key)))
    return p.detach().cpu().clone(),q.detach().cpu().clone(),yp,yq


def remaining(result,s,special):
    hist={int(k):v for k,v in result['normal_submissions_histogram'].items()}
    zero=hist.get(0,0)-s['histogram'].get(0,0)
    return dict(remaining_forward_count=result['forward_count']-s['forward_count'],
        remaining_head_calls=result['head_calls']-s['head_calls'],
        remaining_normal_submissions=result['normal_token_count']-s['normal']-sum(t not in special for _,t in s['committed']),remaining_zero_rounds=zero,
        remaining_expand_count=result['expand_count']-s['expands'],remaining_delete_positions=result['delete_count']-s['deletes'],
        remaining_delete_actions=sum(bool(t['delete']) for t in result['action_trace'][s['step']:]),
        remaining_structure_refreshes=result['statistics']['corrected_structure_rounds']-s['stats']['corrected_structure_rounds'])


def branch(e,byid,db,v,s,label,seed,intervention,extra=None):
    key=v['sample_group']+'::'+v.get('kind','head')+'::'+identity(v)+'::'+label+'::'+str(seed)
    cached=db.get('fork',key)
    if cached:return cached
    nfe=e.nfe;calls=e.call_count();start=time.perf_counter()
    result=e.decode(snapshot=s,intervention=intervention,continuation_seed=seed)
    # Every seed42 host-policy arm must reproduce the complete factual trace.
    factual=(label=='original' or label=='PP' and s['method']=='nohead' or label=='QQ' and s['method']=='trained')
    if seed==42 and factual:compare(trace_read(v['config'],v['case']),result)
    execution=key
    write_gz(LOCAL/('fork__'+hashlib.sha256(key.encode()).hexdigest()[:24]+'.json.gz'),result)
    row=dict(compact(result),**remaining(result,s,e.special),state=identity(v),config=v['config'],task_id=v['case'],base_task_id=v['base_task_id'],
        sample_group=v['sample_group'],kind=v.get('kind','head'),intervention_round=v['round'],intervention_slot=v['slot'],position=v['position'],
        intervention=label,continuation_seed=seed,actual_execution=execution,reused_execution=False,
        actual_forward_count=e.nfe-nfe,actual_head_calls=e.call_count()-calls,elapsed_sec=time.perf_counter()-start,
        snapshot=str(snapshot_file(v)),**(extra or {}))
    row.update(db.score(byid[v['case']],result));db.put('fork',key,row)
    return row


def head_experiment(e,byid,db,v,s):
    p,q,yp,yq=distributions(e,s)
    evidence=dict(y_p=yp,y_q=yq,p_yp=float(p.softmax(-1)[yp]),q_yp=float(q.softmax(-1)[yp]),p_yq=float(p.softmax(-1)[yq]),q_yq=float(q.softmax(-1)[yq]),
        candidate_changed=yp!=yq,threshold=s['threshold'],unique_exempt=s['slot']==s['exempt'])
    key=identity(v)
    if not db.get('shadow_cost',key):db.put('shadow_cost',key,dict(shadow_head_calls=1))
    for seed in range(42,50):
        for label in ('PP','PQ','QP','QQ'):
            cs,ss=label.lower()
            intervention=dict(kind='head',candidate=yp if cs=='p' else yq,score_logits=p if ss=='p' else q,candidate_source=cs,score_source=ss)
            branch(e,byid,db,v,s,label,seed,intervention,dict(evidence,candidate_source=cs,score_source=ss))


def teacher(e,db,v,s):
    key=identity(v)
    if db.get('teacher',key):return
    p,q,yp,yq=distributions(e,s);p=p.softmax(-1);q=q.softmax(-1)
    nfe=e.nfe;calls=e.call_count()
    with torch.inference_mode():fresh=e.fresh(query_state(clone_state(s,e.device))).float().softmax(-1).cpu()
    e.teacher_forwards_total=getattr(e,'teacher_forwards_total',0)+e.nfe-nfe
    e.nfe=nfe # Teacher accounting is completely separate from decoding cost.
    eps=1e-30
    special=[e.tokenizer.expand_id,e.tokenizer.eos_id]
    row=dict(state=key,config=v['config'],case=v['case'],base_task_id=v['base_task_id'],teacher_forwards=1,shadow_head_calls=1,
        kl_fresh_p=float((fresh*(fresh.clamp_min(eps).log()-p.clamp_min(eps).log())).sum()),
        kl_fresh_q=float((fresh*(fresh.clamp_min(eps).log()-q.clamp_min(eps).log())).sum()),
        tv_fresh_p=float((fresh-p).abs().sum()/2),tv_fresh_q=float((fresh-q).abs().sum()/2),
        y_p=yp,y_q=yq,probabilities={name:dict(y_p=float(d[yp]),y_q=float(d[yq]),structural_mass=float(d[special].sum())) for name,d in [('p',p),('q',q),('fresh',fresh)]})
    assert e.call_count()==calls
    db.put('teacher',key,row)


def run_forks(e,byid,db):
    # Replay and fork selection are separate stages. The decoder blob stays
    # unchanged; pin the selection/orchestration revision before any fork.
    from experiments.threshold_eval.common import commit
    from experiments.threshold_trace_fork.run import verify_phase_versions
    verify_phase_versions(db,commit())
    if db.get('meta','fork_execution_version') is None:
        db.put('meta','fork_execution_version',dict(execution_commit=commit()))
    atomic_json(OUT/'phase_versions.json',dict(replay=db.get('meta','execution_version'),forks=db.get('meta','fork_execution_version'),decoder_unchanged_from_replay=True))
    panel=choose_panel(db)
    manifest=OUT/'special_state_manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text());heads,decoder,common=m['head'],m['decoder'],m['common_state_checks']
    else:heads,decoder,common=choose_special(db)
    items=panel+heads+decoder+[v for c in common for v in (c['a'],c['b'])]
    # Store snapshots on disk; load only the current state's tensors to bound RAM.
    capture_states(e,byid,db,items)
    checks=[];invalid=set()
    for c in common:
        a=torch.load(snapshot_file(c['a']),map_location='cpu',weights_only=False);b=torch.load(snapshot_file(c['b']),map_location='cpu',weights_only=False)
        equal=same_state(a,b);checks.append(dict(c,complete_pre_action_state_equal=equal))
        if not equal:
            invalid.add((c['group'],identity(c['b'])))
            invalid.add((c['group'],identity(c['a'])))
    atomic_json(OUT/'common_state_integrity.json',checks)
    for i,v in enumerate(panel+heads):
        if (v['sample_group'],identity(v)) in invalid:continue
        s=torch.load(snapshot_file(v),map_location='cpu',weights_only=False)
        if v['sample_group']=='panel':teacher(e,db,v,s)
        head_experiment(e,byid,db,v,s)
        status('head_forks',states_completed=i+1,states_expected=len(panel+heads),logical_branches=len(db.rows('fork')))
    for i,v in enumerate(decoder):
        s=torch.load(snapshot_file(v),map_location='cpu',weights_only=False)
        if v['kind']=='order':
            choices=[('original',{}),('a2_order',dict(kind='order',a2_first=True,a2_exempt=False)),('a2_exempt',dict(kind='order',a2_first=False,a2_exempt=True)),('a2_both',dict(kind='order',a2_first=True,a2_exempt=True))]
        elif v['kind']=='no_propagation':choices=[('original',{}),('no_propagation',dict(kind='no_propagation'))]
        else:choices=[('original',{}),('defer',dict(kind='structure',action='defer')),('veto',dict(kind='structure',action='veto'))]
        for seed in range(42,50):
            for label,intervention in choices:branch(e,byid,db,v,s,label,seed,intervention)
        status('decoder_forks',states_completed=i+1,states_expected=len(decoder),logical_branches=len(db.rows('fork')))
    for case in FOCUS:
        for cid,c in CONFIGS.items():
            for seed in (43,44,45):
                key=cid+'::'+case+'::'+str(seed)
                if db.get('multiseed',key):continue
                result=generate(e,byid[case],c,seed=seed)
                write_gz(trace_path(cid,case,seed),result)
                row=dict(compact(result),config=cid,task_id=case,base_task_id=case.split('/')[2],seed=seed)
                row.update(db.score(byid[case],result));db.put('multiseed',key,row)
            status('from_scratch_seeds',completed=len(db.rows('multiseed')),expected=54,case=case,config=cid)
