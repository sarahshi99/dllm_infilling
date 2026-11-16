"""Single-model finite diagnostic runner; transactional resume by logical identity."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import traceback
import torch
from experiments.threshold_eval.common import configs, spec, commit, gpu_observation
from experiments.context_foundation.common import atomic_json, utc
from experiments.threshold_trace_fork.engine import Engine, clone_state, integrity, query_state

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis_outputs/dreamon_threshold_trace_fork_20261112_v1'
LOCAL=OUT/'local'
FOCUS=['SingleLineInfilling/HumanEval/139/L5','SingleLineInfilling/HumanEval/126/L0','SingleLineInfilling/HumanEval/80/L5']
IDS=['global_k2_nohead_095','global_k4_nohead_095','global_k4_trained_095','global_k4_nohead_090','global_k4_trained_090']
CONFIGS={c['id']:c for c in configs() if c['id'] in IDS+['global_k2_nohead_090']}
FIELDS=['completion_token_ids','forward_count','head_calls','outer_rounds','statistics','normal_submissions_histogram','normal_token_count','expand_count','delete_count','stop_reason','final_remaining_masks','sampling_key_sha256']
TRACE_FIELDS=['selected','submitted','barrier','expand','delete','x0','x_after']


def status(phase,**kw):
    v=dict(at=utc(),phase=phase,pid=os.getpid(),**kw)
    atomic_json(OUT/'status.json',v)
    print(json.dumps(v,ensure_ascii=False),flush=True)


class Ledger:
    def __init__(self):
        LOCAL.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(LOCAL/'results.sqlite',timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS results(kind TEXT,key TEXT,payload TEXT,PRIMARY KEY(kind,key))')
        self.db.execute('CREATE TABLE IF NOT EXISTS scores(case_id TEXT,output TEXT,payload TEXT,PRIMARY KEY(case_id,output))')
        self.db.commit()
    def get(self,kind,key):
        r=self.db.execute('SELECT payload FROM results WHERE kind=? AND key=?',(kind,key)).fetchone()
        return json.loads(r[0]) if r else None
    def put(self,kind,key,value):
        with self.db:self.db.execute('INSERT INTO results VALUES(?,?,?)',(kind,key,json.dumps(value,ensure_ascii=False,allow_nan=False)))
    def count(self,kind):return self.db.execute('SELECT COUNT(*) FROM results WHERE kind=?',(kind,)).fetchone()[0]
    def rows(self,kind):return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM results WHERE kind=? ORDER BY key',(kind,))]
    def score(self,problem,result):
        key=(problem['task_id'],result['output'])
        row=self.db.execute('SELECT payload FROM scores WHERE case_id=? AND output=?',key).fetchone()
        if row:return dict(json.loads(row[0]),score_reused=True,score_identity=list(key))
        from experiments.run_k4_multiline_stage import official_evaluate
        score=official_evaluate(problem,result['output'])
        with self.db:self.db.execute('INSERT INTO scores VALUES(?,?,?)',(*key,json.dumps(score)))
        return dict(score,score_reused=False,score_identity=list(key))


def write_gz(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp,'wt') as f:json.dump(value,f,ensure_ascii=False,separators=(',',':'))
    temp.replace(path)


def trace_path(config,case,seed=42):
    return LOCAL/(config+'__'+case.replace('/','_')+'__'+str(seed)+'.json.gz')


def trace_read(config,case):
    with gzip.open(trace_path(config,case),'rt') as f:return json.load(f)


def generate(e,r,c,**kwargs):
    return e.decode(r['prefix_ids'],r['suffix_ids'],r['task_id'],c['method'],c['order'],c['k'],c['threshold'],**kwargs)


def historical_rows():
    path=ROOT/'analysis_outputs/dreamon_threshold_20261104_v1/per_case.jsonl.gz'
    result={}
    with gzip.open(path,'rt') as f:
        for line in f:
            r=json.loads(line)
            if r['config'] in CONFIGS:result[(r['config'],r['task_id'])]=r
    return result


def load():
    # Reuse the original model loader, redirect only its diagnostic metadata output.
    from experiments.threshold_eval import runner as historical
    historical.SHARED=LOCAL
    old=historical.load_engine(True)
    old.hook.remove()
    e=Engine(old.model,old.tokenizer,old.head,old.special)
    rows=historical.data(e)
    return e,{r['task_id']:r for r in rows}


def compare(a,b):
    for f in FIELDS:
        av,bv=a[f],b[f]
        if f=='normal_submissions_histogram':av={int(k):v for k,v in av.items()};bv={int(k):v for k,v in bv.items()}
        assert av==bv,(f,av,bv)
    assert len(a['action_trace'])==len(b['action_trace'])
    for t,u in zip(a['action_trace'],b['action_trace']):
        for f in TRACE_FIELDS:
            # JSON round trips convert submitted tuples to lists.
            assert json.dumps(t[f],sort_keys=True)==json.dumps(u[f],sort_keys=True),(t['round'],f)


def smoke(e,byid,db):
    existing=OUT/'smoke_integrity.json'
    if existing.exists() and json.loads(existing.read_text()).get('status')=='passed':return
    from experiments.threshold_eval.adapter import Engine as Original
    checks=[];start=time.perf_counter()
    for case in list(dict.fromkeys(spec()['smoke_cases']+FOCUS)):
        for cid,c in CONFIGS.items():
            key=cid+'::'+case
            done=db.get('smoke',key)
            if done:checks.append(done);continue
            r=byid[case];saved=[]
            def capture(s):
                # First available candidate in the second round, or very short
                # structural trajectory in round zero. Includes the in-round cursor.
                if not saved and (s['step']>=1 or int(s['x0'].eq(e.tokenizer.mask_id).sum())<2):saved.append(clone_state(s))
            original=Original(e.model,e.tokenizer,e.head,e.special)
            try:
                a=original.decode(r['prefix_ids'],r['suffix_ids'],case,c['method'],c['order'],c['k'],c['threshold'],detailed=True)
                b=generate(e,r,c,capture=capture);compare(a,b)
                if not saved:
                    generate(e,r,c,capture=lambda s:saved.append(clone_state(s)) if not saved else None)
                resumed=e.decode(snapshot=saved[0]);compare(b,resumed)
                assert b==resumed,'all diagnostic fields must match on factual restore'
                checked=dict(config=cid,case=case,original_matches=True,restore_matches=True,
                    forward_count=b['forward_count'],head_calls=b['head_calls'],expand_count=b['expand_count'],delete_count=b['delete_count'],snapshot_round=saved[0]['step'],snapshot_cursor=saved[0]['cursor'])
                db.put('smoke',key,checked);checks.append(checked)
                write_gz(LOCAL/('smoke__'+cid+'__'+case.replace('/','_')+'.json.gz'),dict(original=a,instrumented=b))
                status('smoke',completed=len(checks),expected=(len(spec()['smoke_cases'])+3)*6,case=case,config=cid)
            finally:original.hook.remove()
    evaluator_check=db.score(byid[FOCUS[0]],dict(output=byid[FOCUS[0]]['canonical_solution']))
    assert evaluator_check['passed']
    elapsed=time.perf_counter()-start
    atomic_json(OUT/'smoke_integrity.json',dict(status='passed',checks=checks,evaluator=evaluator_check,elapsed_sec=elapsed,
        estimated_replay_sec=elapsed/max(1,len(checks))/3*5165,estimate_caveat='smoke has original/logged/restored paths; shared GPU; variable lengths'))
    status('smoke_passed',checks=len(checks),elapsed_sec=elapsed)


def compact(result):return {k:v for k,v in result.items() if k!='action_trace'}


def replay(e,byid,db):
    history=historical_rows()
    for cid in IDS+['global_k2_nohead_090']:
        cases=sorted(byid) if cid in IDS else FOCUS
        c=CONFIGS[cid]
        for case in cases:
            key=cid+'::'+case
            if db.get('replay',key):continue
            start=time.perf_counter();v=generate(e,byid[case],c)
            old=history[(cid,case)]
            differences={f:dict(historical=old.get(f),replayed=v.get(f)) for f in FIELDS if f!='sampling_key_sha256' and json.dumps(old.get(f),sort_keys=True)!=json.dumps(v.get(f),sort_keys=True)}
            write_gz(trace_path(cid,case),v)
            row=dict(compact(v),config=cid,task_id=case,base_task_id=case.split('/')[2],seed=42,
                historical_differences=differences,historical_passed=old['passed'],generation_sec=time.perf_counter()-start)
            row.update(db.score(byid[case],v))
            db.put('replay',key,row)
            count=db.count('replay')
            if count%10==0:status('replay',completed=count,expected=5168,config=cid,case=case)
    allrows=db.rows('replay');audit={}
    for cid in CONFIGS:
        rows=[r for r in allrows if r['config']==cid]
        audit[cid]=dict(expected=1033 if cid in IDS else 3,completed=len(rows),historical_mismatches=[dict(case=r['task_id'],fields=r['historical_differences']) for r in rows if r['historical_differences']],
            score_mismatches=[r['task_id'] for r in rows if r['passed']!=r['historical_passed']])
    atomic_json(OUT/'replay_integrity.json',audit)
    with gzip.open(OUT/'replay_per_case.jsonl.gz','wt') as f:
        for r in allrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    if any(a['historical_mismatches'] for a in audit.values()):raise RuntimeError('historical replay drift: do not run dependent forks')
    status('replay_passed',completed=len(allrows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','all'],default='all');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(exist_ok=True)
    db=Ledger();version=commit();old=db.get('meta','execution_version' if args.phase=='all' else 'development_version')
    if old is None:db.put('meta','execution_version' if args.phase=='all' else 'development_version',dict(execution_commit=version))
    elif old['execution_commit']!=version:
        changed=subprocess.check_output(['git','diff',old['execution_commit'],version,'--','experiments/threshold_trace_fork','experiments/threshold_eval','experiments/context_foundation'],cwd=ROOT,text=True)
        if changed:raise RuntimeError('resume cannot mix executable versions')
    start=time.perf_counter()
    atomic_json(LOCAL/'environment.json',dict(at=utc(),commit=version,devices=gpu_observation(),torch=torch.__version__,cuda=torch.version.cuda))
    try:
        status('loading',execution_commit=version)
        e,byid=load();smoke(e,byid,db)
        if args.phase=='smoke':return
        replay(e,byid,db)
        from experiments.threshold_trace_fork.forks import run_forks
        run_forks(e,byid,db)
        e.hook.remove()
        from experiments.threshold_trace_fork.analyze import analyze
        analyze(db)
        atomic_json(OUT/'actual_diagnostic_cost.json',dict(shadow_head_calls=getattr(e,'shadow_head_calls_total',0),teacher_forwards=getattr(e,'teacher_forwards_total',0),elapsed_sec=time.perf_counter()-start,execution_commit=version))
        status('completed',elapsed_sec=time.perf_counter()-start)
    except torch.cuda.OutOfMemoryError:
        status('oom_process_exit',error=traceback.format_exc(),elapsed_sec=time.perf_counter()-start)
        raise SystemExit(42)
    except (AssertionError,ValueError):
        status('scientific_validity_blocked',error=traceback.format_exc(),elapsed_sec=time.perf_counter()-start)
        raise SystemExit(3)
    except Exception as exc:
        status('blocked',error=traceback.format_exc(),elapsed_sec=time.perf_counter()-start)
        raise SystemExit(3 if 'drift' in str(exc) or 'version' in str(exc) else 2)

if __name__=='__main__':main()
