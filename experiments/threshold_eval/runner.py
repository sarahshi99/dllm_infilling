"""Persistent single-model role workers and post-generation unified timing."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

from experiments.threshold_eval.common import *


def set_status(role, state, **fields):
    atomic_json(SHARED/f'{role}.status.json',dict(task_id=TASK,role=role,status=state,pid=os.getpid(),at=utc(),**fields))
    print(utc(),role,state,json.dumps(fields,ensure_ascii=False),flush=True)


def load_engine(with_head):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from experiments.dreamon_singleline_adapter import OfficialHFTokenizerWrapper, ensure_modeling_rope_utils_available
    from experiments.threshold_eval.adapter import Engine, FrozenHeadAdapter
    from experiments.train_dreamon_markov_head import make_head
    ensure_modeling_rope_utils_available()
    frozen=spec()
    tokenizer=OfficialHFTokenizerWrapper(AutoTokenizer.from_pretrained(frozen['snapshot'],trust_remote_code=True,local_files_only=True))
    model=AutoModel.from_pretrained(frozen['snapshot'],trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False)
    assert next(model.parameters()).dtype==torch.bfloat16
    h=None
    if with_head:
        assert sha(frozen['head']['path'])==frozen['head']['sha256']
        head=make_head(model.config,'cpu')
        ckpt=torch.load(frozen['head']['path'],map_location='cpu',weights_only=False)
        head.load_state_dict(ckpt['head'],strict=True)
        del ckpt
        head=head.to('cuda:0').eval().requires_grad_(False)
        assert next(head.parameters()).dtype==torch.float32
        h=FrozenHeadAdapter(head)
    special=json.loads((REPO/'analysis_outputs/dreamon_context_foundation_20261015_v1/protocol_schema.json').read_text())['special_token_ids']
    e=Engine(model,tokenizer,h,special,max_gen_len=frozen['protocol']['max_gen_len'])
    properties=torch.cuda.get_device_properties(0)
    atomic_json(SHARED/f'model_{os.getpid()}.json',dict(pid=os.getpid(),at=utc(),cuda_logical_device=0,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device_name=properties.name,
        device_uuid=str(getattr(properties,'uuid','not_exposed')),observation=gpu_observation(),head_loaded=with_head,
        torch_version=torch.__version__,cuda_version=torch.version.cuda))
    return e


def data(engine):
    from experiments.k4_batch_common import population
    rows=sorted(population(spec().get('dataset','single-line')),key=lambda r:r['task_id'])
    assert [r['task_id'] for r in rows]==spec()['cases']
    for r in rows:
        r['prefix_ids']=engine.tokenizer.encode(r['prompt'],add_bos=True,add_eos=False)
        r['suffix_ids']=engine.tokenizer.encode(r['suffix'],add_bos=False,add_eos=True)
    return rows


def generate(engine,r,c,trace=False,detailed=False):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    result=engine.decode(r['prefix_ids'],r['suffix_ids'],r['task_id'],c['method'],c['order'],c['k'],c['threshold'],
                         detailed=detailed,record_trace=trace)
    torch.cuda.synchronize()
    result.update(generation_sec=time.perf_counter()-begin,peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),task_id=r['task_id'],
        **case_metadata(r['task_id']),config=c['id'],execution_commit=(os.environ.get('THRESHOLD_EXECUTION_COMMIT') or commit()),
        seed=42,method=c['method'],order=c['order'],k=c['k'],threshold=c['threshold'],max_gen_len=engine.max_gen_len)
    if not trace:
        result.pop('action_trace',None)
        result.pop('sampling_key_sha256',None)
    return result


def heartbeat(role,stop,progress):
    while not stop.wait(30):
        atomic_json(SHARED/f'{role}.heartbeat.json',dict(at=utc(),pid=os.getpid(),**progress))
        with (SHARED/f'{role}.gpu.jsonl').open('a') as f:
            f.write(json.dumps(gpu_observation())+'\n')


def smoke():
    import torch
    from experiments.context_foundation.adapter import Engine as Reference
    e=load_engine(True)
    byid={r['task_id']:r for r in data(e)}
    checks=[];measurement=[]
    raw=SHARED/f'smoke_traces.{os.getpid()}.jsonl.gz'
    with gzip.open(raw,'wt') as traces:
        for i,key in enumerate(spec()['smoke_cases']):
            r=byid[key];order=('global','left')[i%2];k=(2,4)[(i//2)%2]
            ref=Reference(e.model,e.tokenizer,e.head,e.special,max_gen_len=e.max_gen_len)
            try:
                values={}
                for kk,method,arm in [(1,'nohead','K1'),(4,'nohead','nohead'),(4,'trained','H4-old')]:
                    c=dict(id='smoke',method=method,order=order,k=kk,threshold=None)
                    v=generate(e,r,c,trace=True,detailed=True)
                    w=ref.decode(r['prefix_ids'],r['suffix_ids'],key,arm,order,detailed=True)
                    for field in ('completion_token_ids','forward_count','head_calls','normal_token_count','expand_count','delete_count'):
                        assert v[field]==w[field],(key,method,kk,field)
                    assert len(v['action_trace'])==len(w['action_trace'])
                    for t,u in zip(v['action_trace'],w['action_trace']):
                        for field in ('selected','submitted','expand','delete','x_after','barrier'):
                            assert t[field]==u[field],(key,method,kk,field)
                    compact=generate(e,r,c,trace=False)
                    for field in ('completion_token_ids','forward_count','head_calls','normal_token_count','expand_count','delete_count','statistics','normal_submissions_histogram'):
                        assert compact[field]==v[field],(key,'compact_off',field)
                    values[f'{method}_{kk}_off']=v
                for method in ('nohead','trained','zero'):
                    c=dict(id='smoke',method=method,order=order,k=k,threshold=.90)
                    values[method]=generate(e,r,c,trace=True,detailed=True)
                    if method!='zero':measurement.append(dict(config=c,case_key=key,span_bucket=case_metadata(key)['span_bucket'],generation_sec=values[method]['generation_sec']))
                    assert max(len(t['selected']) for t in values[method]['action_trace'])<=k
                for method in ('nohead','trained'):
                    c=dict(id='smoke',method=method,order=order,k=k,threshold=.90)
                    compact=generate(e,r,c,trace=False)
                    for field in ('completion_token_ids','forward_count','head_calls','normal_token_count','expand_count','delete_count','statistics','normal_submissions_histogram'):
                        assert compact[field]==values[method][field],(key,'compact_threshold',field)
                a,b=values['nohead'],values['zero']
                assert a['completion_token_ids']==b['completion_token_ids']
                assert [t['x_after'] for t in a['action_trace']]==[t['x_after'] for t in b['action_trace']]
                assert a['head_calls']==0
                checks.append(dict(case_key=key,order=order,k=k,off_matches_reference=True,zero_matches_nohead=True,
                    nohead_calls=a['head_calls'],zero_calls=b['head_calls'],trained_calls=values['trained']['head_calls'],
                    **case_metadata(key),delete_count=sum(v['delete_count'] for v in values.values()),expand_count=sum(v['expand_count'] for v in values.values())))
                traces.write(json.dumps(dict(case_key=key,values=values),ensure_ascii=False)+'\n')
                set_status('smoke','running',completed=i+1,expected=16)
            finally:
                ref.hook.remove()
    assert sum(c['zero_calls'] for c in checks)>0 and sum(c['trained_calls'] for c in checks)>0
    assert sum(c['delete_count'] for c in checks)>0
    # Parameterization must also preserve real old SingleLine trajectories.
    from experiments.k4_batch_common import population
    old_keys=json.loads((REPO/'analysis_outputs/dreamon_threshold_20261104_v1/run_spec.json').read_text())['smoke_cases']
    old_rows={r['task_id']:r for r in population('single-line')}
    regressions=[]
    if MULTILINE:
        budget=e.max_gen_len
        e.max_gen_len=64
        ref=Reference(e.model,e.tokenizer,e.head,e.special)
        try:
            for i,key in enumerate(old_keys[:4]):
                r=old_rows[key]
                r['prefix_ids']=e.tokenizer.encode(r['prompt'],add_bos=True,add_eos=False)
                r['suffix_ids']=e.tokenizer.encode(r['suffix'],add_bos=False,add_eos=True)
                order=('global','left')[i%2];method=('nohead','trained')[(i//2)%2]
                v=generate(e,r,dict(id='singleline_regression',method=method,order=order,k=4,threshold=None),trace=True,detailed=True)
                w=ref.decode(r['prefix_ids'],r['suffix_ids'],key,'nohead' if method=='nohead' else 'H4-old',order,detailed=True)
                assert v['completion_token_ids']==w['completion_token_ids']
                assert [[t[f] for f in ('x_after','submitted','expand','delete','barrier')] for t in v['action_trace']]==[[t[f] for f in ('x_after','submitted','expand','delete','barrier')] for t in w['action_trace']]
                regressions.append(dict(case_key=key,method=method,order=order,passed=True))
        finally:
            ref.hook.remove()
            e.max_gen_len=budget
    # Evaluate known canonical answer with the official evaluator; quality outcomes do not gate smoke.
    from experiments.run_k4_multiline_stage import official_evaluate
    sample=byid[spec()['smoke_cases'][0]]
    evaluator_check=official_evaluate(sample,sample['canonical_solution'])
    assert evaluator_check['passed'],evaluator_check
    atomic_json(OUT/'smoke.json',dict(status='passed',checks=checks,evaluator_check=evaluator_check,
        raw_traces=str(raw),old_singleline_regressions=regressions,measurements=measurement,
        rough_quality_gpu_hours=sum(m['generation_sec'] for m in measurement)/len(measurement)*spec()['expected_quality']/3600,
        estimate_limitation='fixed stratified smoke mixture; config and length costs vary; shared contention not a completion deadline',quality_selection=False,fixed_cases=spec()['smoke_cases']))
    e.hook.remove()
    set_status('smoke','passed',completed=16,model_released_on_process_exit=True)


def quality(role):
    import torch
    version=json.loads((SHARED/'READY.json').read_text())['execution_commit']
    verify_execution(version)
    os.environ['THRESHOLD_EXECUTION_COMMIT']=version
    db=Ledger(role)
    assigned=[c for c in spec()['configs'] if c['method']==role]
    # Resume never combines executable versions.
    assert all(r['execution_commit']==version for r in db.rows('quality'))
    expected=len(assigned)*len(spec()['cases'])
    if db.count('quality')==expected:
        set_status(role,'generation_complete',completed=expected,expected=expected)
        return
    progress=dict(completed=db.count('quality'),expected=expected,execution_commit=version)
    stop=threading.Event();thread=threading.Thread(target=heartbeat,args=(role,stop,progress),daemon=True);thread.start()
    try:
        set_status(role,'loading_model',**progress)
        e=load_engine(role=='trained')
        rows=data(e)
        for c in assigned:
            for r in rows:
                if db.has('quality',c['id'],r['task_id']):continue
                progress.update(config=c['id'],case_key=r['task_id'])
                result=generate(e,r,c)
                db.put('quality',c['id'],r['task_id'],result)
                progress.update(completed=progress['completed']+1,config=c['id'],case_key=r['task_id'])
                if progress['completed']%10==0:set_status(role,'generating',**progress)
            set_status(role,'config_complete',**progress)
        assert db.count('quality')==expected
        e.hook.remove()
        set_status(role,'generation_complete',**progress)
    except torch.cuda.OutOfMemoryError:
        db.attempt('oom',progress.get('config','load')+'::'+progress.get('case_key',''),traceback.format_exc())
        set_status(role,'oom_process_exit',**progress)
        return 42
    except Exception as error:
        db.attempt('generation_infrastructure' if isinstance(error,(OSError,TimeoutError,subprocess.TimeoutExpired)) else 'generation_code_error',progress.get('config','load')+'::'+progress.get('case_key',''),traceback.format_exc())
        set_status(role,'generation_failed',error=traceback.format_exc(),**progress)
        return 75 if isinstance(error,(OSError,TimeoutError,subprocess.TimeoutExpired)) else 2
    finally:
        stop.set();thread.join(timeout=3);db.db.close()
    return 0


def timing():
    import torch
    role='timing';db=Ledger(role)
    version=json.loads((SHARED/'READY.json').read_text())['execution_commit']
    verify_execution(version)
    os.environ['THRESHOLD_EXECUTION_COMMIT']=version
    expected=spec()['expected_timing']
    if db.count('timing')==expected:return 0
    stop=threading.Event();progress=dict(completed=db.count('timing'),expected=expected)
    thread=threading.Thread(target=heartbeat,args=(role,stop,progress),daemon=True);thread.start()
    try:
        set_status(role,'loading_model',**progress)
        e=load_engine(True)
        byid={r['task_id']:r for r in data(e)}
        conf=spec()['configs']
        # Warm up each configuration on the same fixed case; never enter quality ledger.
        warm=byid[spec()['timing_cases'][0]]
        for c in conf:generate(e,warm,c)
        for rep in range(3):
            for i,key in enumerate(spec()['timing_cases']):
                offset=(rep+i)%len(conf)
                order=conf[offset:]+conf[:offset]
                for c in order:
                    if db.has('timing',c['id'],key,rep):continue
                    result=generate(e,byid[key],c)
                    result.update(repetition=rep,population='timing_per_base_task_mixed_crossline_and_single_fallback' if MULTILINE else 'timing_164_per_base_task',schedule_offset=offset)
                    db.put('timing',c['id'],key,result,rep)
                    progress.update(completed=progress['completed']+1,repetition=rep,case_key=key)
                set_status(role,'timing',**progress)
        e.hook.remove()
        assert db.count('timing')==expected
        set_status(role,'timing_complete',**progress)
    except torch.cuda.OutOfMemoryError:
        db.attempt('oom','timing',traceback.format_exc());set_status(role,'oom_process_exit',**progress);return 42
    except Exception as error:
        db.attempt('timing_infrastructure' if isinstance(error,(OSError,TimeoutError,subprocess.TimeoutExpired)) else 'timing_code_error','timing',traceback.format_exc());set_status(role,'timing_failed',error=traceback.format_exc(),**progress);return 75 if isinstance(error,(OSError,TimeoutError,subprocess.TimeoutExpired)) else 2
    finally:
        stop.set();thread.join(timeout=3);db.db.close()
    return 0


def role_running(role):
    path=SHARED/f'{role}.lock'
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(f,fcntl.LOCK_UN)
    return False


def pipeline():
    """No model here; persistent pipeline starts timing only after own role processes exit."""
    with (SHARED/'pipeline.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 0
        # Attach to existing roles; start only missing unfinished roles without a recorded failure.
        for role in ('nohead','trained'):
            ledger=Ledger(role);count=ledger.count('quality');ledger.db.close()
            expected=sum(c['method']==role for c in spec()['configs'])*len(spec()['cases'])
            exit_path=SHARED/f'{role}.exit.json'
            failed=exit_path.exists() and json.loads(exit_path.read_text())['exit_code']!=0
            if count<expected and not role_running(role) and not failed:
                command=f'bash {SHARED}/start_worker.sh --role {role}'
                subprocess.run(['tmux','new-session','-d','-s',f'{TASK}_{role}','-c',str(REPO),command],check=True)
        while True:
            counts={}
            for role in ('nohead','trained'):
                ledger=Ledger(role);counts[role]=ledger.count('quality');ledger.db.close()
            atomic_json(SHARED/'status.json',dict(task_id=TASK,status='quality_generation',quality_completed=sum(counts.values()),quality_by_role=counts,
                timing_completed=0,at=utc(),pid=os.getpid()))
            if counts=={r:sum(c['method']==r for c in spec()['configs'])*len(spec()['cases']) for r in counts} and not any(role_running(r) for r in counts):break
            failures=[]
            for role in counts:
                p=SHARED/f'{role}.exit.json'
                if p.exists() and json.loads(p.read_text())['exit_code']!=0 and not role_running(role):failures.append(role)
            if failures:
                atomic_json(SHARED/'status.json',dict(status='blocked_generation',failed_roles=failures,quality_by_role=counts,at=utc()))
                # Leave the surviving role running to natural exit before committing any report.
                if not any(role_running(r) for r in counts):
                    subprocess.run([PYTHON,'-m','experiments.threshold_eval.analyze','--partial','--publish'],cwd=REPO)
                    return 2
            time.sleep(15)
        atomic_json(SHARED/'status.json',dict(status='official_scoring',quality_completed=spec()['expected_quality'],at=utc()))
        code=subprocess.call([PYTHON,'-m','experiments.threshold_eval.analyze','--score-only'],cwd=REPO)
        if code:
            subprocess.call([PYTHON,'-m','experiments.threshold_eval.analyze','--partial','--publish'],cwd=REPO)
            return code
        atomic_json(SHARED/'status.json',dict(status='unified_timing',quality_completed=spec()['expected_quality'],at=utc()))
        code=subprocess.call(['bash',str(SHARED/'start_worker.sh'),'--role','timing'],cwd=REPO)
        # Even failed timing yields honest partial audit/report.
        analysis_code=subprocess.call([PYTHON,'-m','experiments.threshold_eval.analyze']+(['--partial','--publish'] if code else ['--publish']),cwd=REPO)
        return code or analysis_code


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','quality','timing','pipeline'],required=True);p.add_argument('--role',choices=['nohead','trained'])
    a=p.parse_args()
    if a.phase=='smoke':smoke();return 0
    if a.phase=='quality':return quality(a.role)
    if a.phase=='timing':return timing()
    return pipeline()


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(2)
