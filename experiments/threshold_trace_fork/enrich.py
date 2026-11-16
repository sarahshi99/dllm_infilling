"""Additional descriptive summaries from completed, immutable local evidence."""
from collections import Counter,defaultdict
import json
import hashlib
from experiments.context_foundation.common import atomic_json
from experiments.threshold_trace_fork.run import OUT,LOCAL,Ledger,trace_read,IDS
from experiments.threshold_trace_fork.forks import events
from experiments.threshold_trace_fork.analyze import clustered,export,fork_pairs


def enrich(db):
    replay=db.rows('replay');forks=db.rows('fork')
    assert len(replay)==5168
    export('raw_trace_index.jsonl.gz',[dict(actual_execution=r['actual_execution'],state=r['state'],sample_group=r['sample_group'],kind=r['kind'],intervention=r['intervention'],continuation_seed=r['continuation_seed'],raw_trace=str(LOCAL/('fork__'+hashlib.sha256(r['actual_execution'].encode()).hexdigest()[:24]+'.json.gz')),snapshot=r['snapshot'],reused_execution=r['reused_execution']) for r in forks])
    absolute=[]
    for config in IDS:
        rr=[r for r in replay if r['config']==config]
        h=Counter()
        for r in rr:h.update({int(k):v for k,v in r['normal_submissions_histogram'].items()})
        absolute.append(dict(config=config,rows=len(rr),base_tasks=len({r['base_task_id'] for r in rr}),passed=sum(r['passed'] for r in rr),quality=clustered(rr,'passed'),forward_count=clustered(rr,'forward_count'),total_forwards=sum(r['forward_count'] for r in rr),total_head_calls=sum(r['head_calls'] for r in rr),O=sum(r['normal_token_count'] for r in rr),S=sum(max(k-1,0)*v for k,v in h.items()),Z=h.get(0,0),expand_actions=sum(r['expand_count'] for r in rr),delete_positions=sum(r['delete_count'] for r in rr),delete_actions=sum(r['statistics']['base_structure_rounds']-r['expand_count'] for r in rr),head_structure_refreshes=sum(r['statistics']['corrected_structure_rounds'] for r in rr),incomplete_cases=sum(r['final_remaining_masks']>0 for r in rr),stop_reasons=dict(Counter(r['stop_reason'] for r in rr))))
    atomic_json(OUT/'replay_absolute_summary.json',absolute)
    paired_tables=[]
    for ca,cb in [('global_k2_nohead_095','global_k4_nohead_095'),('global_k4_nohead_090','global_k4_trained_090'),('global_k4_nohead_095','global_k4_trained_095')]:
        aa={r['task_id']:r for r in replay if r['config']==ca};bb={r['task_id']:r for r in replay if r['config']==cb}
        tables={}
        for equality,field in [('tokens','completion_token_ids'),('text','output')]:
            table=Counter();passed_complete=Counter()
            for case,x in aa.items():
                y=bb[case];d=y['forward_count']-x['forward_count']
                key=('same' if x[field]==y[field] else 'different')+'_'+('decrease' if d<0 else 'increase' if d>0 else 'equal')
                table[key]+=1
                if x['passed'] and y['passed'] and not x['final_remaining_masks'] and not y['final_remaining_masks']:passed_complete[key]+=1
            tables[equality]=dict(all_pairs=dict(table),both_passed_complete=dict(passed_complete))
        paired_tables.append(dict(baseline=ca,comparison=cb,tables=tables))
    atomic_json(OUT/'complete_paired_tables.json',paired_tables)
    zero_types=defaultdict(Counter)
    config_cases=Counter()
    for row in replay:
        config=row['config'];config_cases[config]+=1
        for t in trace_read(config,row['task_id'])['action_trace']:
            if t['normal_submissions']!=0:continue
            if t['expand']:kind='native_expand'
            elif t['delete']:kind='native_delete'
            elif t['barrier'] and t['barrier']['reason']=='corrected_structure_refresh_only':kind='head_structure_refresh'
            elif any(e.get('reason')=='mask_noop' for e in t['events']):kind='mask_noop'
            elif any(e.get('reason')=='other_special' for e in t['events']):kind='other_special'
            else:kind='other'
            zero_types[config][kind]+=1
    atomic_json(OUT/'zero_round_types.json',[dict(config=k,cases=config_cases[k],zero_round_types=dict(v),zero_rounds=sum(v.values())) for k,v in zero_types.items()])
    head_summary=[]
    for suffix in ('090','095'):
        config='global_k4_trained_'+suffix;counts=Counter();case_counts=Counter()
        for r in replay:
            if r['config']!=config:continue
            c=Counter()
            for t,e in events(trace_read(config,r['task_id'])):
                if not e['head_called']:continue
                c['calls']+=1;c['candidate_changed' if e['raw']!=e['final'] else 'candidate_unchanged']+=1
                c['destination_'+e['reason']]+=1
                if e['raw']==e['final']:
                    before=e['raw_probability']>=float('.'+suffix[1:]);after=e['final_probability']>=float('.'+suffix[1:])
                    c['same_word_'+('promoted' if not before and after else 'demoted' if before and not after else 'both_pass' if before else 'both_reject')]+=1
            counts.update(c)
            for k,v in c.items():case_counts[k]+=int(v>0)
        head_summary.append(dict(config=config,event_counts=dict(counts),case_counts=dict(case_counts),population='full repeated-development 1033 cases',
            caveat='same-word probability crossings are descriptive events; four-arm branches are needed for path effects'))
    atomic_json(OUT/'head_event_summary.json',head_summary)
    grouped=defaultdict(dict)
    for r in forks:
        if r['kind']=='head':grouped[(r['sample_group'],r['state'],r['continuation_seed'])][r['intervention']]=r
    interactions=[]
    for key,g in grouped.items():
        if set(g)!=set(('PP','PQ','QP','QQ')):continue
        n={k:v['remaining_forward_count'] for k,v in g.items()};q={k:int(v['passed']) for k,v in g.items()}
        interactions.append(dict(sample_group=key[0],state=key[1],continuation_seed=key[2],task_id=g['PP']['task_id'],base_task_id=g['PP']['base_task_id'],
            nfe_interaction=n['QQ']-n['QP']-n['PQ']+n['PP'],pass_interaction=q['QQ']-q['QP']-q['PQ']+q['PP'],
            candidate_changed=g['PP']['candidate_changed'],same_candidate_threshold_crossed=(g['PP']['p_yp']>=g['PP']['threshold'])!=(g['PP']['q_yp']>=g['PP']['threshold'])))
    export('head_interactions_per_state_seed.jsonl.gz',interactions)
    summaries=[]
    for group in sorted({r['sample_group'] for r in interactions}):
        rows=[r for r in interactions if r['sample_group']==group]
        summaries.append(dict(sample_group=group,logical_state_seeds=len(rows),states=len({r['state'] for r in rows}),
            nfe_interaction=clustered(rows,'nfe_interaction'),pass_interaction=clustered(rows,'pass_interaction'),
            definition='QQ - QP - PQ + PP; conditional path total effects under paired seeds, not additive factor shares'))
    atomic_json(OUT/'head_interactions.json',summaries)
    paired=fork_pairs(forks)
    source_by_state={r['state']:r['config'] for r in forks}
    strata=[]
    for config in sorted({r['config'] for r in forks if r['sample_group']=='panel'}):
        for a,b in [('PP','PQ'),('PP','QP'),('PP','QQ'),('PQ','QQ'),('QP','QQ')]:
            rr=[r for r in paired if r['sample_group']=='panel' and source_by_state[r['state']]==config and r['baseline']==a and r['comparison']==b]
            strata.append(dict(config=config,baseline=a,comparison=b,logical_pairs=len(rr),quality=clustered(rr,'delta_pass'),efficiency=clustered(rr,'delta_nfe'),changed_tokens=sum(r['tokens_changed'] for r in rr)))
    atomic_json(OUT/'head_effects_by_stratum.json',strata)
    numeric=db.rows('teacher_logspace_validation')
    assert len(numeric) in (0,64)
    teacher_rows=numeric if numeric else db.rows('teacher')
    if numeric:
        pp=[r for r in forks if r['sample_group']=='panel' and r['intervention']=='PP' and r['continuation_seed']==42]
        atomic_json(OUT/'conditional_distribution_logspace_summary.json',dict(states=len(numeric),kl_closer=sum(r['kl_fresh_q']<r['kl_fresh_p'] for r in numeric),tv_closer=sum(r['tv_fresh_q']<r['tv_fresh_p'] for r in numeric),mean_kl_p=sum(r['kl_fresh_p'] for r in numeric)/64,mean_kl_q=sum(r['kl_fresh_q'] for r in numeric)/64,mean_tv_p=sum(r['tv_fresh_p'] for r in numeric)/64,mean_tv_q=sum(r['tv_fresh_q'] for r in numeric)/64,candidate_changed=sum(r['candidate_changed'] for r in pp),fixed_yp_score_crossed=sum((r['p_yp']>=r['threshold'])!=(r['q_yp']>=r['threshold']) for r in pp),method='FP32 full legal log_softmax; no probability floor',original_numeric_evidence_retained='teacher_per_state.jsonl.gz',primary_numeric_evidence='teacher_logspace_per_state.jsonl.gz'))
    teacher_strata=[]
    for config in sorted({r['config'] for r in teacher_rows}):
        rr=[r for r in teacher_rows if r['config']==config]
        teacher_strata.append(dict(config=config,states=len(rr),kl_closer=sum(r['kl_fresh_q']<r['kl_fresh_p'] for r in rr),tv_closer=sum(r['tv_fresh_q']<r['tv_fresh_p'] for r in rr),mean_kl_change=sum(r['kl_fresh_q']-r['kl_fresh_p'] for r in rr)/len(rr),mean_tv_change=sum(r['tv_fresh_q']-r['tv_fresh_p'] for r in rr)/len(rr)))
    atomic_json(OUT/'teacher_by_stratum.json',teacher_strata)
    scratch=db.rows('multiseed');checks=db.rows('smoke');captures=db.rows('capture_cost');original=db.rows('original_full_validation')
    smoke_forwards=sum(3*r['forward_count']-(r['snapshot_round']+1)+(r['forward_count'] if r['forward_count']==1 else 0) for r in checks)
    cost=dict(replay_forwards=sum(r['forward_count'] for r in replay),scratch_forwards=sum(r['forward_count'] for r in scratch),
        branch_actual_forwards=sum(r['actual_forward_count'] for r in forks),prefix_rebuild_and_restore_validation_forwards=sum(r['prefix_replay_forwards'] for r in captures),
        smoke_original_logged_restore_and_fallback_forwards=smoke_forwards,teacher_forwards=len(db.rows('teacher')),teacher_numeric_validation_forwards=sum(r['teacher_forwards'] for r in numeric),
        extra_original_validation_forwards=sum(r['original_forward_count'] for r in original),original_validation_rows=len(original),
        replay_execution_version=db.get('meta','execution_version'),fork_execution_version=db.get('meta','fork_execution_version'),logical_fork_count=len(forks),scratch_rows=len(scratch))
    cost['all_recorded_forwards']=sum(cost[k] for k in ('replay_forwards','scratch_forwards','branch_actual_forwards','prefix_rebuild_and_restore_validation_forwards','smoke_original_logged_restore_and_fallback_forwards','teacher_forwards','teacher_numeric_validation_forwards','extra_original_validation_forwards'))
    atomic_json(OUT/'complete_forward_accounting.json',cost)
    return head_summary,summaries,cost

if __name__=='__main__':enrich(Ledger())
