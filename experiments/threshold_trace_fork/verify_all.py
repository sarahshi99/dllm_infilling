"""Validate previously logged detailed fields by running the unchanged original.

Historical per-case rows did not contain detailed canvases. This is a present
original-entry-point comparison, never labeled as a historical trace archive.
Run only after the primary single-model process has exited.
"""
import json
import gzip
import subprocess
import time
from experiments.threshold_eval.adapter import Engine as Original
from experiments.context_foundation.common import atomic_json
from experiments.threshold_eval.common import commit
from experiments.threshold_trace_fork.run import (Ledger,OUT,LOCAL,CONFIGS,IDS,FOCUS,load,compare,trace_read,status)


def main():
    db=Ledger();start=time.perf_counter();version=commit()
    prior=db.get('meta','original_validation_version')
    if prior is None:db.put('meta','original_validation_version',dict(execution_commit=version))
    elif prior['execution_commit']!=version:
        paths=['experiments/threshold_trace_fork/teacher_numeric.py','experiments/threshold_trace_fork/verify_all.py','experiments/threshold_trace_fork/run.py','experiments/threshold_eval','experiments/context_foundation','experiments/dreamon_singleline_order_parallel.py','experiments/dreamon_singleline_adapter.py','experiments/k4_batch_common.py']
        if subprocess.check_output(['git','diff',prior['execution_commit'],version,'--',*paths],text=True):
            raise RuntimeError('cannot mix original-validation versions')
    status('original_full_validation_loading',execution_commit=version)
    old,byid=load();old.hook.remove()
    e=Original(old.model,old.tokenizer,old.head,old.special)
    try:
        for cid in IDS+['global_k2_nohead_090']:
            cases=sorted(byid) if cid in IDS else FOCUS;c=CONFIGS[cid]
            for case in cases:
                key=cid+'::'+case
                if db.get('original_full_validation',key):continue
                r=byid[case];begin=time.perf_counter()
                reference=e.decode(r['prefix_ids'],r['suffix_ids'],case,c['method'],c['order'],c['k'],c['threshold'],detailed=True)
                instrumented=trace_read(cid,case)
                compare(reference,instrumented)
                assert reference['output']==instrumented['output']
                for t,u in zip(reference['action_trace'],instrumented['action_trace']):
                    actual=[ev for ev in u['events'] if ev.get('processed')]
                    assert len(t['events'])==len(actual),(cid,case,t['round'],'candidate count')
                    for a,b in zip(t['events'],actual):
                        for field in ('slot','position','chain','raw','final','visible_slots','key','mode','candidate_probability'):
                            assert a.get(field)==b.get(field),(cid,case,t['round'],a['slot'],field,a.get(field),b.get(field))
                        assert bool(a.get('rejected'))==bool(b.get('rejected'))
                scored=db.get('replay',key)
                db.put('original_full_validation',key,dict(config=cid,case=case,matched=True,original_forward_count=reference['forward_count'],original_head_calls=reference['head_calls'],elapsed_sec=time.perf_counter()-begin,passed=scored['passed'],score_reused_from_replay=True,score_identity=scored['score_identity']))
                if db.count('original_full_validation')%25==0:status('original_full_validation',completed=db.count('original_full_validation'),expected=5168)
        with gzip.open(OUT/'original_validation_per_case.jsonl.gz','wt') as stream:
            for row in db.rows('original_full_validation'):stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        original_elapsed=time.perf_counter()-start
        from experiments.threshold_trace_fork.teacher_numeric import recheck
        recheck(e,db)
        atomic_json(OUT/'original_full_validation.json',dict(status='passed',execution_version=db.get('meta','original_validation_version'),completed=db.count('original_full_validation'),expected=5168,
            provenance='present unchanged-original run vs instrumented archive; not historical detailed traces',
            fields=['tokens','round canvases','selected','submitted','barrier','expand/delete','counts','histograms','sampling digest','stop reason'],
            elapsed_sec=time.perf_counter()-start,original_validation_elapsed_sec=original_elapsed,recorded_case_elapsed_sec=sum(r['elapsed_sec'] for r in db.rows('original_full_validation')),numeric_validation_states=64,extra_numeric_teacher_forwards=64,actual_original_forwards=sum(r['original_forward_count'] for r in db.rows('original_full_validation')),
            actual_original_head_calls=sum(r['original_head_calls'] for r in db.rows('original_full_validation'))))
        status('original_full_validation_passed',completed=5168,teacher_numeric_states=64,elapsed_sec=time.perf_counter()-start)
    finally:e.hook.remove()

if __name__=='__main__':main()
