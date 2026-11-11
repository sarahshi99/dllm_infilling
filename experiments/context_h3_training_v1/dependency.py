"""Read H1's published bank in place. Waiting never initializes CUDA."""
import collections
import json
import sqlite3
import time
from pathlib import Path

from experiments.context_foundation.common import Store, REPO, atomic_json, utc
from experiments.context_h1_training_v1.bank import RUN as PUBLIC, Bank, cells, cell, config

RUN=Path('/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1')
OUT=REPO/'analysis_outputs/dreamon_context_h3_training_20261111_v1'
RESUME=f'cd {REPO} && bash experiments/context_h3_training_v1/launch.sh'


class ReadOnlyStore(Store):
    def __init__(self,path):
        self.db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    def put(self,*args):
        raise RuntimeError('public source is read only')


def progress(stage,**values):
    import os
    row=dict(stage=stage,at=utc(),pid=os.getpid(),**values)
    atomic_json(RUN/'progress.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def ready():
    coverage=PUBLIC/'coverage.json'
    if not coverage.exists(): return False,'H1 has not published coverage'
    c=json.loads(coverage.read_text())
    if not c.get('complete'): return False,'H1 bank not complete'
    if c.get('counts')!=cells() or any(c.get('gaps',{}).values()):
        raise ValueError('H1 complete marker conflicts with agreed 32-cell quotas')
    # Initial full validation is published only after collection has returned and
    # all immutable shards were read. H1 may still be training; do not wait for it.
    if not (PUBLIC/'validation_epoch_0.json').exists(): return False,'await H1 full initial validation / collection closure'
    a=json.loads((PUBLIC/'source_audit.json').read_text())
    t=json.loads((PUBLIC/'technical_checks.json').read_text())
    if a['status']!='passed' or a['group_overlap']!=0 or a['external_payload_opened'] or t['status']!='passed':
        raise ValueError('H1 source or technical audit failed')
    p=json.loads((PUBLIC/'plan.json').read_text())
    expected=dict(seed=20261107,optimizer='AdamW',lr=3e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clip_norm=1,effective_batch=128,epochs=5,head_dtype='FP32',train_states=200000,validation_states=20000,
        loss='full-vocab T1 mean KL(fresh||old+delta)',initialization='W_out=0; gamma=1; remaining internal parameters random')
    differences={k:dict(expected=v,actual=p.get(k)) for k,v in expected.items() if p.get(k)!=v}
    if p['config']!=json.loads(json.dumps(vars(config()))): differences['head_config']=p['config']
    if differences:
        atomic_json(RUN/'protocol_differences.json',differences)
        raise ValueError('substantive H1 configuration differences; see protocol_differences.json')
    return True,'H1 complete bank, full initial validation and audits published'


def wait_for_public():
    path=RUN/'dependency_wait.json'
    state=json.loads(path.read_text()) if path.exists() else dict(started=time.time(),deadline=time.time()+72*3600,poll_seconds=60)
    atomic_json(path,state)
    while True:
        yes,reason=ready()
        if yes: return
        exit_path=PUBLIC/'exit.json'
        if exit_path.exists():
            e=json.loads(exit_path.read_text())
            if e.get('status') not in ('complete','probe_complete','ready_for_background'):
                raise RuntimeError(f'H1 failed before public bank release: {e}; {reason}')
        if time.time()>=state['deadline']: raise TimeoutError('72 hour public data wait expired')
        progress('waiting_public_data',reason=reason,deadline_epoch=state['deadline'],cuda_initialized=False)
        time.sleep(min(60,max(0,state['deadline']-time.time())))


def open_bank():
    source=ReadOnlyStore(PUBLIC/'queue.sqlite')
    if source.db.execute('PRAGMA quick_check').fetchone()[0]!='ok': raise ValueError('public sqlite corrupt')
    bank=Bank(source,config(),root=PUBLIC)
    rows=bank.rows
    counts=dict(collections.Counter(cell(r) for r in rows))
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ('train','validation')}
    if counts!=cells() or groups['train']&groups['validation']: raise ValueError('public quotas or source split mismatch')
    if len({r['state_id'] for r in rows})!=220000: raise ValueError('duplicate public states')
    caps=collections.Counter((r['group'],cell(r)) for r in rows)
    if max(caps.values())>4: raise ValueError('public per-source contribution cap violated')
    shards={r['shard'] for r in rows}
    missing=[s for s in shards if not (PUBLIC/s).is_file() or (PUBLIC/s).stat().st_size==0]
    if missing: raise ValueError(f'missing public shards: {missing[:3]}')
    latest_shard_mtime=max((PUBLIC/s).stat().st_mtime for s in shards)
    if latest_shard_mtime>(PUBLIC/'validation_epoch_0.json').stat().st_mtime:
        raise ValueError('public shard modified after published collection closure / initial validation')
    # H1's complete validation has read all validation shards; its completed
    # training (when available) additionally read every training state five times.
    # Each H3 batch still opens the original arrays and validates the head contract.
    summary=dict(version='H1 formal 20261107 / 9aff455',public_root=str(PUBLIC),metadata='queue.sqlite:state ordered by key',
        train_states=200000,validation_states=20000,source_groups={s:len(g) for s,g in groups.items()},group_overlap=0,
        counts=counts,max_source_cell_contribution=max(caps.values()),shards=len(shards),
        shard_bytes=sum((PUBLIC/s).stat().st_size for s in shards),
        integrity_reused=['coverage.json','source_audit.json','technical_checks.json','validation_epoch_0.json'],
        producer_training_complete=(PUBLIC/'training_summary.json').exists(),
        latest_shard_mtime=latest_shard_mtime,closure_validation_mtime=(PUBLIC/'validation_epoch_0.json').stat().st_mtime,
        collection_closed=True,cache_copied=False,public_access='SQLite mode=ro; original NPZ shards read only',
        teacher='A/H1 X0 + current-chain committed ordinary prefix only; full-vocabulary FP32',
        source_gap_records=source.count('source_gap'),stage_counts=dict(collections.Counter(r['stage'] for r in rows)))
    atomic_json(RUN/'data_summary.json',summary)
    source.db.close()
    return bank
