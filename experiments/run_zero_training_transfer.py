#!/usr/bin/env python3
"""Controller-only, serial fixed-canvas transfer; no training and no dynamic actions."""
from __future__ import annotations
import argparse
import copy
import fcntl
import gzip
import importlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import time
import traceback

import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.dreamon_singleline_adapter import CountingModel, derive_row_seed, ensure_modeling_rope_utils_available
from experiments.dreamon_singleline_order_parallel import GLOBAL_CONFIDENCE, LEFT_TO_RIGHT_FRONTIER
from experiments.lrdllm_dreamcoder_adapter import atomic_write_json, append_jsonl, load_dataset_rows, load_pinned_evaluator, selection_view
from experiments.run_dreamon_markov_k2_diagnosis import gpu_snapshot, has_start_capacity, stable_smoke

POLICIES = {'global_confidence': GLOBAL_CONFIDENCE, 'left_to_right_frontier': LEFT_TO_RIGHT_FRONTIER}
ARMS = tuple((p, a) for p in POLICIES for a in 'ABCD')
DIAG = REPO / 'analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/run_manifest.json'
CONTROLLER = REPO / 'analysis_outputs/zero_training_transfer_20261013/controller'


def open_store(path):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS cases(target TEXT,policy TEXT,arm TEXT,task_id TEXT,seed INTEGER,payload TEXT NOT NULL,PRIMARY KEY(target,policy,arm,task_id,seed))')
    db.commit()
    return db


def save_case(db, row):
    with db:
        db.execute('INSERT INTO cases VALUES(?,?,?,?,?,?)', (row['target'], row['policy'], row['arm'], row['task_id'], row['seed'], json.dumps(row, ensure_ascii=False)))


def read_cases(db):
    return [json.loads(r[0]) for r in db.execute('SELECT payload FROM cases ORDER BY target,policy,arm,task_id,seed')]


def charge(ledger, session_id, started, target, *, status='running'):
    data = json.loads(ledger.read_text())
    sessions = data['sessions']
    entry = next((s for s in sessions if s['session_id'] == session_id), None)
    if entry is None:
        entry = {'session_id': session_id, 'target': target, 'pid': os.getpid(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        sessions.append(entry)
    entry.update(elapsed_sec=time.perf_counter()-started, status=status)
    data['used_gpu_job_sec'] = sum(s['elapsed_sec'] for s in sessions)
    atomic_write_json(ledger, data)
    return data['limit_gpu_job_sec'] - data['used_gpu_job_sec']


def run(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads(DIAG.read_text())
    registered = json.loads(Path(args.target_registry).read_text())
    target = registered[args.target]
    if target['direct_transfer_status'] != 'passed' or target['weight_kind'] != 'original_base_fixed_length_auxiliary':
        raise RuntimeError('target has not passed direct transfer compatibility or is not a Base auxiliary')
    adapter_name=target.get('adapter_module','experiments.dreamcoder_fixed_transfer_adapter')
    adapter=importlib.import_module(adapter_name)
    decode_fixed_transfer=adapter.decode_fixed_transfer
    native_fixed_reference=adapter.native_fixed_reference
    snapshot_path = Path(target['local_path'])
    if not snapshot_path.is_dir() or snapshot_path.name != target['revision']:
        raise RuntimeError('exact target snapshot missing')
    rows = load_dataset_rows(Path(original['original_config']['evaluator_root']), 'HumanEval-SingleLineInfilling')
    import csv
    with (Path(original['original_config_path']).parent / 'paired_deltas.csv').open() as f:
        old_ids = {r['task_id'] for r in csv.DictReader(f)}
    if len(rows) != 1033 or old_ids != {r['task_id'] for r in rows} or len({r['task_id'].split('/')[2] for r in rows}) != 164:
        raise RuntimeError('development population mismatch')
    smoke_ids = [r['task_id'] for r in stable_smoke(rows)]
    if smoke_ids != original['smoke_task_ids']:
        raise RuntimeError('fixed smoke IDs mismatch')
    code_paths = ['experiments/run_zero_training_transfer.py', 'experiments/dreamcoder_fixed_transfer_adapter.py', 'experiments/dreamon_singleline_order_parallel.py', 'experiments/dreamon_markov_head_training.py', 'experiments/train_dreamon_markov_head.py']
    adapter_path=adapter_name.replace('.','/')+'.py'
    if adapter_path not in code_paths: code_paths.append(adapter_path)
    manifest = {'target': args.target, 'target_registry_entry': target, 'head_registry_entry': original['head_registry_entry'],
                'source_model_snapshot': original['original_config']['model_snapshot'], 'head_rank':256, 'head_lambda':1,
                'seed':42, 'native_reference_eps':target.get('native_eps',1e-3), 'smoke_task_ids':smoke_ids, 'population_count':1033, 'base_task_count':164,
                'population':'reused development population, single generation seed; no sealed test access',
                'arms':[{'policy':p,'arm':a,'k':1 if a=='A' else 2} for p,a in ARMS],
                'parameters':{'canvas_length':64,'steps':64,'eps':1e-3,'temperature':.2,'top_p':.9,'top_k':None,'eos_penalty':target.get('native_eos_penalty',3.0)},
                'differences_from_dreamon':f"fixed canvas64, no expand/delete; native Base EOS penalty {target.get('native_eos_penalty',3.0)}; target-specific native reference plus fixed K/order patch; steps64 safety bound vs DreamOn256; same rules for all arms",
                'timing':'shared H200 cost only; no formal speed ratio', 'command':shlex.join([sys.executable,*sys.argv]),
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'code_versions':{p:subprocess.check_output(['git','rev-parse','HEAD:'+p],cwd=REPO,text=True).strip() for p in code_paths}}
    dirty = subprocess.check_output(['git','status','--porcelain','--',*code_paths],cwd=REPO,text=True).strip()
    if dirty:
        raise RuntimeError('commit evaluated source before run')
    mpath = out/'run_manifest.json'
    if mpath.exists():
        previous = json.loads(mpath.read_text())
        for key in ('target_registry_entry','head_registry_entry','parameters','code_versions','smoke_task_ids','arms'):
            if previous[key] != manifest[key]:
                raise RuntimeError('resume protocol mismatch: '+key)
        manifest = previous
    manifest['status']='prepared'
    atomic_write_json(mpath,manifest)
    # A single physical GPU mutex and ledger cover every target/session in this round.
    with open('/tmp/dllm_infilling_h200_diagnosis.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        snap = gpu_snapshot()
        append_jsonl(out/'gpu_background.jsonl',snap)
        if len(snap['gpus'].splitlines()) != 1 or 'H200' not in snap['gpus'] or not has_start_capacity(snap,min_free_gib=32):
            manifest.update(status='blocked_resource',blocker='single H200 with at least32GiB free required; no process interrupted')
            atomic_write_json(mpath,manifest)
            return 3
        ledger = CONTROLLER/'gpu_budget.json'
        data = json.loads(ledger.read_text())
        if data['used_gpu_job_sec'] >= data['limit_gpu_job_sec']:
            raise RuntimeError('global round budget exhausted')
        started=time.perf_counter()
        session_id=f'{args.target}:{time.time_ns()}:{os.getpid()}'
        db=open_store(out/'per_case.sqlite')
        def budget_check():
            remaining=charge(ledger,session_id,started,args.target)
            if remaining <= 0:
                raise RuntimeError('global cumulative GPU job budget exhausted')
            return remaining
        try:
            budget_check()
            ensure_modeling_rope_utils_available()
            from transformers import AutoModel, AutoTokenizer, AutoConfig
            from experiments.train_dreamon_markov_head import make_head
            tokenizer=AutoTokenizer.from_pretrained(snapshot_path,trust_remote_code=True,local_files_only=True)
            source_config=AutoConfig.from_pretrained(manifest['source_model_snapshot'],trust_remote_code=True,local_files_only=True)
            trained=make_head(source_config,'cpu')
            checkpoint_path=Path(manifest['head_registry_entry']['path'])
            if checkpoint_path.stat().st_size != manifest['head_registry_entry']['size']:
                raise RuntimeError('registered checkpoint size mismatch')
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            trained.load_state_dict(checkpoint['head'],strict=True)
            del checkpoint
            zero=copy.deepcopy(trained)
            with torch.no_grad(): zero.output.weight.zero_()
            for head in (trained,zero):
                head.to('cuda:0').eval().requires_grad_(False)
            model=CountingModel(AutoModel.from_pretrained(snapshot_path,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda:0').eval())
            if model.model.config.vocab_size != trained.vocab_size:
                raise RuntimeError('model output vocabulary differs from source head')
            # Use this target's native sampler, not DreamOn's dynamic generation code.
            sample_tokens_fn=type(model.model)._sample.__globals__['sample_tokens']
            check_correctness,evaluator_info=load_pinned_evaluator(Path(original['original_config']['evaluator_root']))
            if evaluator_info != original['original_config']['evaluator']:
                raise RuntimeError('pinned evaluator mismatch')
            legal_ids=set(tokenizer.get_vocab().values())
            def decode(row,p,a,reference=False):
                safe=selection_view(row)
                kwargs=dict(model=model,tokenizer=tokenizer,sample_tokens_fn=sample_tokens_fn,
                            prefix_ids=[tokenizer.bos_token_id]+tokenizer.encode(safe['prefix'],add_special_tokens=False),
                            suffix_ids=tokenizer.encode(safe['suffix'],add_special_tokens=False)+[tokenizer.eos_token_id],
                            requested_k=1 if a=='A' else 2,policy=POLICIES[p],row_seed=derive_row_seed(42,row['task_id']),
                            **{k:v for k,v in manifest['parameters'].items() if k!='canvas_length'})
                torch.manual_seed(kwargs['row_seed']); torch.cuda.manual_seed_all(kwargs['row_seed'])
                torch.cuda.synchronize(); t=time.perf_counter()
                before=model.forward_calls
                if reference:
                    result=native_fixed_reference(**kwargs)
                else:
                    result=decode_fixed_transfer(**kwargs,markov_head={'C':zero,'D':trained}.get(a))
                torch.cuda.synchronize(); elapsed=time.perf_counter()-t
                if any(int(v) not in legal_ids or int(v)==tokenizer.mask_token_id for v in result['completion_token_ids']):
                    raise RuntimeError('unregistered or residual mask token generated')
                if not reference and model.forward_calls-before != result['forward_count']:
                    raise RuntimeError('forward counter mismatch')
                return result,elapsed
            manifest.update(status='running_smoke',actual_device=torch.cuda.get_device_name(0),pid=os.getpid(),strict_head_load='all parameters and persistent output_mask; no missing/unexpected keys')
            atomic_write_json(mpath,manifest)
            checks=[]; costs=[]
            # Always redo technical checks on a resumed session; no outcome-based choice.
            for row in stable_smoke(rows):
                for p in POLICIES:
                    for a in 'ABCD':
                        budget_check()
                        result,elapsed=decode(row,p,a)
                        if a in 'AB':
                            reference,_=decode(row,p,a,reference=True)
                            if reference['completion_token_ids'] != result['completion_token_ids']:
                                raise RuntimeError(f'native fixed-K reference mismatch: {row["task_id"]}/{p}/{a}')
                        if a in 'CD' and result['markov_enabled_count'] > result['adjacent_opportunity_count']:
                            raise RuntimeError('invalid correction opportunity denominator')
                        t=time.perf_counter(); evaluation=check_correctness(row,result['completion'],3.0,0)
                        costs.append(elapsed+time.perf_counter()-t)
                        checks.append({'task_id':row['task_id'],'policy':p,'arm':a,'reference_checked':a in 'AB','generation_sec':elapsed})
                atomic_write_json(out/'smoke_progress.json',{'checked':len(checks),'expected':128})
            estimate=sum(costs)/16*1033
            atomic_write_json(out/'smoke_checks.json',{'status':'passed','checks':checks,'full_batch_estimate_sec':estimate,'reference':'official native fixed canvas with disclosed fixed-K/policy patch'})
            prior=read_cases(db)
            expected={(args.target,p,a,r['task_id'],42) for p,a in ARMS for r in rows}
            completed={(r['target'],r['policy'],r['arm'],r['task_id'],r['seed']) for r in prior}
            if not completed <= expected or any(r['row_seed']!=derive_row_seed(42,r['task_id']) for r in prior):
                raise RuntimeError('resume identity/seed mismatch')
            remaining_estimate=estimate*(len(expected)-len(completed))/len(expected)*1.5
            manifest.update(status='running_full',full_batch_estimate_sec=estimate,remaining_estimate_with_margin_sec=remaining_estimate)
            atomic_write_json(mpath,manifest)
            if remaining_estimate > budget_check():
                raise RuntimeError('estimated batch exceeds remaining shared round budget')
            for index,row in enumerate(rows):
                append_jsonl(out/'gpu_background.jsonl',gpu_snapshot())
                offset=index%len(ARMS)
                for p,a in ARMS[offset:]+ARMS[:offset]:
                    key=(args.target,p,a,row['task_id'],42)
                    if key in completed: continue
                    budget_check()
                    try:
                        result,elapsed=decode(row,p,a)
                        completion=result.pop('completion')
                        result.pop('action_trace',None)
                        t=time.perf_counter(); evaluation=check_correctness(row,completion,3.0,0)
                        save_case(db,{'target':args.target,'policy':p,'arm':a,'task_id':row['task_id'],'base_task_id':row['task_id'].split('/')[2],
                                      'seed':42,'row_seed':derive_row_seed(42,row['task_id']),'code_commit':manifest['code_commit'],
                                      'passed':bool(evaluation['passed']),'failure_reason':evaluation['result'],'output':completion,
                                      'generation_sec':elapsed,'evaluator_sec':time.perf_counter()-t,**result})
                        completed.add(key)
                    except Exception as exc:
                        append_jsonl(out/'failure_journal.jsonl',{'target':args.target,'policy':p,'arm':a,'task_id':row['task_id'],'seed':42,'error':str(exc),'traceback':traceback.format_exc()})
                        raise
                    atomic_write_json(out/'progress.json',{'status':'running','completed':len(completed),'expected':8264,'global_budget_remaining_sec':budget_check()})
            manifest['status']='completed_generation'
            return 0
        except Exception as exc:
            manifest.update(status='blocked',blocker=str(exc))
            append_jsonl(out/'runner_failures.jsonl',{'error':str(exc),'traceback':traceback.format_exc()})
            print(traceback.format_exc(),flush=True)
            return 2
        finally:
            charge(ledger,session_id,started,args.target,status=manifest['status'])
            atomic_write_json(mpath,manifest)
            all_rows=read_cases(db)
            with gzip.open(out/'per_case.jsonl.gz','wt',encoding='utf-8') as f:
                for row in all_rows: f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
            db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',required=True)
    parser.add_argument('--target-registry',required=True)
    parser.add_argument('--output-dir',required=True)
    raise SystemExit(run(parser.parse_args()))
