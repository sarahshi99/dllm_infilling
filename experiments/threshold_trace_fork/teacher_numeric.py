"""Offline log-space check on the same 64 teacher states; never a decision input."""
import gzip
import json
import time
import torch


def distances(log_fresh,log_p,log_q):
    fresh=log_fresh.exp()
    def kl(other):
        return float(torch.where(fresh>0,fresh*(log_fresh-other),torch.zeros_like(fresh)).sum())
    return dict(kl_fresh_p=kl(log_p),kl_fresh_q=kl(log_q),
        tv_fresh_p=float((fresh-log_p.exp()).abs().sum()/2),
        tv_fresh_q=float((fresh-log_q.exp()).abs().sum()/2))


@torch.inference_mode()
def recheck(e,db):
    from experiments.threshold_trace_fork.run import OUT,status
    from experiments.threshold_trace_fork.forks import snapshot_file,identity
    from experiments.threshold_trace_fork.engine import clone_state,query_state,key_for
    from experiments.context_foundation.adapter import distribution,inverse_cdf,uniform
    from experiments.context_foundation.common import atomic_json
    panel=json.loads((OUT/'state_panel.json').read_text())['selected']
    assert len(panel)==64
    for i,v in enumerate(panel):
        key=identity(v)
        if db.get('teacher_logspace_validation',key):continue
        initial=db.get('teacher',key);assert initial is not None
        s=clone_state(torch.load(snapshot_file(v),map_location='cpu',weights_only=False),e.device)
        qs=query_state(s);p=s['old_all'][0,s['slot']]
        calls=e.call_count();q=p+e.correction(qs);shadow=e.call_count()-calls
        e.head.calls-=shadow
        u=uniform(key_for(s['case'],s['step'],s['positions'][s['slot']],s['seed']))
        assert int(inverse_cdf(distribution(p),u))==initial['y_p']
        assert int(inverse_cdf(distribution(q),u))==initial['y_q']
        begin=time.perf_counter();nfe=e.nfe
        fresh_logits=e.fresh(qs).float();actual=e.nfe-nfe;e.nfe=nfe
        lp,lq,lf=p.log_softmax(-1),q.log_softmax(-1),fresh_logits.log_softmax(-1)
        metric=distances(lf,lp,lq)
        yp,yq=initial['y_p'],initial['y_q'];special=[e.tokenizer.expand_id,e.tokenizer.eos_id]
        row=dict(initial,**metric)
        row.update(teacher_forwards=actual,shadow_head_calls=shadow,numeric_method='FP32 log_softmax, no probability floor; zero fresh mass contributes zero',
            generation_cost=False,validation_elapsed_sec=time.perf_counter()-begin,
            probabilities={name:dict(y_p=float(d.exp()[yp]),y_q=float(d.exp()[yq]),structural_mass=float(d.exp()[special].sum())) for name,d in [('p',lp),('q',lq),('fresh',lf)]},
            initial_clipped_metrics={k:initial[k] for k in metric},numeric_differences={k:metric[k]-initial[k] for k in metric})
        assert actual==1 and shadow==1
        db.put('teacher_logspace_validation',key,row)
        if (i+1)%8==0:status('teacher_logspace_validation',completed=i+1,expected=64)
    rows=db.rows('teacher_logspace_validation');assert len(rows)==64
    with gzip.open(OUT/'teacher_logspace_per_state.jsonl.gz','wt') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    initial={r['state']:r for r in db.rows('teacher')}
    atomic_json(OUT/'teacher_numeric_integrity.json',dict(states=64,extra_teacher_forwards=sum(r['teacher_forwards'] for r in rows),
        extra_shadow_head_calls=sum(r['shadow_head_calls'] for r in rows),new_states=0,used_in_branch_decisions=False,
        max_abs_difference={k:max(abs(r['numeric_differences'][k]) for r in rows) for k in ('kl_fresh_p','kl_fresh_q','tv_fresh_p','tv_fresh_q')},
        kl_closeness_flags_changed=sum((r['kl_fresh_q']<r['kl_fresh_p'])!=(initial[r['state']]['kl_fresh_q']<initial[r['state']]['kl_fresh_p']) for r in rows),
        tv_closeness_flags_changed=sum((r['tv_fresh_q']<r['tv_fresh_p'])!=(initial[r['state']]['tv_fresh_q']<initial[r['state']]['tv_fresh_p']) for r in rows)))
