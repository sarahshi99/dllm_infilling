"""Replay missing raw tensors from frozen sources within the same remaining budget.

Invoked by the queue with its inherited A lock. Completed generation is untouched.
Hash mismatches block restoration rather than silently changing evidence.
"""
from __future__ import annotations

import gc
import threading
import traceback

import torch

from experiments.context_foundation.common import *
from experiments.context_foundation.evidence import *
from experiments.context_foundation.adapter import Engine,FrozenHeadAdapter
from experiments.context_foundation.worker import wait_gpu,frozen_special_ids


def valid(meta,pathkey='raw_path',hashkey='raw_sha256'):
    p=CACHE/meta[pathkey]
    return p.exists() and sha(p)==meta[hashkey]


def main():
    store=Store()
    selected=list(read_jsonl(OUT/'blind_selected_states.jsonl.gz')) if (OUT/'blind_selected_states.jsonl.gz').exists() else []
    missing=[r for r in selected if not valid(r)]
    teachers=[r for r in store.rows('state') if not valid(r,'teacher_path','teacher_sha256')]
    pairs=[r for r in store.rows('pair') if r['status']=='complete' and not valid(r)]
    dcd=[r for r in store.rows('dcd') if not valid(r)]
    if not (missing or teachers or pairs or dcd):
        print('All original raw shards verified; no GPU restoration needed.',flush=True)
        return
    lock=wait_gpu()
    if lock is None:
        raise RuntimeError('finite restoration GPU wait exhausted')
    budget=Budget()
    budget.switch('restoration')
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(1):
            budget.save()
    thread=threading.Thread(target=heartbeat,daemon=True)
    thread.start()
    engine=model=head=None
    try:
        budget.check()
        from experiments.dreamon_singleline_adapter import OfficialHFTokenizerWrapper,ensure_modeling_rope_utils_available
        ensure_modeling_rope_utils_available()
        from transformers import AutoModel,AutoTokenizer
        from experiments.train_dreamon_markov_head import make_head
        model=AutoModel.from_pretrained(SNAPSHOT,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False)
        tokenizer=OfficialHFTokenizerWrapper(AutoTokenizer.from_pretrained(SNAPSHOT,trust_remote_code=True,local_files_only=True))
        head=make_head(model.config,'cpu')
        checkpoint=torch.load(HEAD,map_location='cpu',weights_only=False)
        head.load_state_dict(checkpoint['head'],strict=True)
        del checkpoint
        head=head.to('cuda:0').eval().requires_grad_(False)
        engine=Engine(model,tokenizer,FrozenHeadAdapter(head),frozen_special_ids(),budget.check)
        trajectories={t['trajectory_id']:t for t in read_jsonl(OUT/'blind_trajectories.jsonl.gz')}
        by_trajectory={}
        for m in missing:
            by_trajectory.setdefault(m['trajectory_id'],{})[m['state_id']]=m
        budget.switch('mechanism')
        for tid,wanted in by_trajectory.items():
            t=trajectories[tid]
            def observe(kind,s):
                if kind!='query' or int(s['history_mask'].sum()) not in (1,2,3):
                    return
                sid=state_id(s)
                if sid in wanted:
                    original=wanted[sid]
                    restored=save_state(s,original)
                    assert restored['raw_sha256']==original['raw_sha256'],'replay numerical or encoding mismatch'
            engine.decode(t['prefix_ids'],t['suffix_ids'],t['case'],t['arm'],t['order'],observer=observe)
            assert all(valid(m) for m in wanted.values()),'frozen state not reproduced by original trajectory'
        for r in teachers:
            restored=measure_state(engine,r)
            assert restored['teacher_sha256']==r['teacher_sha256']
        state_index={r['state_id']:r for r in store.rows('state')}
        for r in pairs:
            restored=counterfactual(engine,state_index[r['state_id']])
            assert restored['raw_sha256']==r['raw_sha256']
        meta_index={r['state_id']:r for r in selected}
        budget.switch('dcd')
        for r in dcd:
            restored=dcd_one(engine,meta_index[r['state_id']])
            assert restored['raw_sha256']==r['raw_sha256']
        atomic_json(OUT/'restoration_audit.json',dict(status='verified',at=utc(),states=len(missing),teachers=len(teachers),pairs=len(pairs),dcd=len(dcd)))
    finally:
        if engine is not None:
            engine.hook.remove()
        del engine,model,head
        gc.collect()
        torch.cuda.empty_cache()
        stop.set()
        thread.join(timeout=5)
        budget.close()
        lock.close()


if __name__=='__main__':
    main()
