#!/usr/bin/env python3
"""The only GPU worker: smoke -> A1/A2 -> A3 -> global four-arm generation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import collections
import gc
import json
import signal
import subprocess
import threading
import time
import traceback

import numpy as np
import torch

from experiments.context_foundation.common import *
from experiments.context_foundation.adapter import *
from experiments.context_foundation.evidence import *


class StopRequested(Exception):
    pass


def wait_gpu():
    waiting_path = OUT/'waiting.local.json'
    waiting = json.loads(waiting_path.read_text()) if waiting_path.exists() else dict(waited_sec=0)
    started = time.monotonic()
    while True:
        waited = waiting['waited_sec']+time.monotonic()-started
        if waited >= 86400:
            atomic_json(waiting_path,dict(status='wait_limit_24h',waited_sec=waited,at=utc()))
            return None
        lock = None
        try:
            lock = acquire('/tmp/dllm_infilling_h200_diagnosis.lock')
            fields = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).strip().split(',')
            assert len(fields)==4 and 'H200' in fields[0], 'requires single H200'
            observation = dict(at=utc(),gpu=fields[0].strip(),free_mib=float(fields[1]),used_mib=float(fields[2]),util_pct=float(fields[3]))
            with (OUT/'gpu_observations.local.jsonl').open('a') as f:
                f.write(json.dumps(observation)+'\n')
            if float(fields[1]) >= 32768:
                atomic_json(waiting_path,dict(status='acquired',waited_sec=waited,at=utc(),observation=observation))
                return lock
        except BlockingIOError:
            pass
        finally:
            if lock is not None and ('fields' not in locals() or float(fields[1]) < 32768):
                lock.close()
        atomic_json(waiting_path,dict(status='waiting_gpu_lock_or_32GiB',waited_sec=waited,at=utc()))
        print(f'{utc()} waiting for cooperative GPU lock and >=32GiB free; waited={waited:.0f}s',flush=True)
        # Finite wait; existing external processes are never signalled.
        time.sleep(min(30,86400-waited))


def progress(stage, **values):
    atomic_json(OUT/'gpu_progress.local.json',dict(stage=stage,at=utc(),pid=os.getpid(),**values))
    print(utc(),stage,json.dumps(values,ensure_ascii=False),flush=True)


@torch.inference_mode()
def smoke(engine,store):
    rows = list(read_jsonl(REPO/'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz'))
    chosen = sorted(rows,key=lambda r:stable(['smoke',r['task_id']]))[:2]
    checks = []
    for case_index,r in enumerate(chosen):
        order = 'global' if case_index == 0 else 'frontier'
        prefix = engine.tokenizer.encode(r['prompt'],add_bos=True,add_eos=False)
        suffix = engine.tokenizer.encode(r['suffix'],add_bos=False,add_eos=True)
        states = []
        def observe(kind,s):
            if kind=='query' and int(s['history_mask'].sum()) and not states:
                states.append({k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in s.items()})
        timing = {}
        values = {}
        for arm in ['nohead','Z4','H4-old']:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            values[arm] = engine.decode(prefix,suffix,r['task_id'],arm,order=order,observer=observe,detailed=True)
            torch.cuda.synchronize()
            timing[arm] = time.perf_counter()-begin
        a,b = values['nohead'],values['Z4']
        assert a['completion_token_ids']==b['completion_token_ids']
        assert [t['x_after'] for t in a['action_trace']]==[t['x_after'] for t in b['action_trace']]
        assert a['head_calls']==0
        if states:
            s = states[0]
            raw,h = engine.forward(s['x0'],s['attention'],s['time_ids'])
            pos = s['position_ids'][0]
            projected = engine.model.lm_head(aligned_hidden(h,pos).to(torch.bfloat16))[0].float()
            reference = raw[0,(pos-1).clamp_min(0)].float()
            # GEMM shape can change last-bit BF16 rounding; compare exact target logits
            # to h[t-1] using same full-shape projection and separately report small-shape error.
            full_projected = engine.model.lm_head(h).float()
            assert torch.equal(full_projected[0,(pos-1).clamp_min(0)],reference)
            self_test = dict(full_shape_exact=True,selected_projection_max_error=float((projected-reference).abs().max()))
            before = s['x0'].clone()
            fresh = engine.fresh(s)
            assert torch.equal(before,s['x0'])
            future = ~s['history_mask'][0,0]
            altered = s['token_ids'].clone()
            altered[0,future] = 0
            assert torch.equal(fresh,engine.fresh(s,altered))
            zero = engine.correction(s,zero=True)
            assert torch.count_nonzero(zero)==0 and b['head_calls']>0
        else:
            self_test = dict(no_head_opportunity_in_this_case=True)
        checks.append(dict(task_id=r['task_id'],order=order,zero_nohead_tokens=True,zero_nohead_every_canvas=True,
            no_future_leak=True,teacher_x0_immutable=True,alignment=self_test,
            genuine_nohead_calls=a['head_calls'],zero_head_calls=b['head_calls'],old_head_calls=values['H4-old']['head_calls'],
            observed_shared_gpu_sec=timing,trace=values))
    assert sum(c['zero_head_calls'] for c in checks)>0, 'no genuine zero-head opportunity in fixed smoke'
    assert sum(c['old_head_calls'] for c in checks)>0, 'no genuine old-head opportunity in fixed smoke'
    atomic_json(OUT/'gpu_smoke.json',dict(status='passed',checks=checks,
        boundary='target0 -> h0 and target1 -> h0 verified in CPU tests; selected real targets use t-1',
        quality='technical checks only; no scores used to select or tune'))
    store.put('phase','smoke',dict(status='passed'))


def mechanism(engine,store,budget):
    budget.switch('mechanism')
    trajectories = list(read_jsonl(OUT/'blind_trajectories.jsonl.gz'))
    existing = store.get('phase','candidates_frozen')
    if existing is None:
        limit = 7200
        for i,t in enumerate(trajectories):
            budget.check()
            elapsed = budget.snapshot()['stage_sec'].get('mechanism',0)
            if elapsed >= limit:
                break
            if store.get('trajectory',t['trajectory_id']):
                continue
            histogram = collections.Counter()
            omissions = collections.Counter()
            def observe(kind,s):
                if kind=='query':
                    length = int(s['history_mask'].sum())
                    omissions['eligible_prefix_'+str(length)] += 1
                    outcome=reserve_candidate(store,s,t)
                    omissions['candidate_'+outcome] += 1
                else:
                    histogram.update(s['chain_lengths'])
                    if s['gaps']:
                        omissions['round_with_gaps'] += 1
                    if s['barrier']:
                        omissions[s['barrier']['reason']] += 1
            try:
                result = engine.decode(t['prefix_ids'],t['suffix_ids'],t['case'],t['arm'],t['order'],observer=observe)
            except BudgetExceeded:
                break
            store.put('trajectory',t['trajectory_id'],dict(trajectory_id=t['trajectory_id'],group=t['group'],split=t['split'],
                order=t['order'],span=t['span'],arm=t['arm'],rounds=result['outer_rounds'],nfe=result['forward_count'],
                chain_length_counts=dict(histogram),opportunities=dict(omissions)))
            if i%16==0:
                progress('A1_blind_trajectory_collection',ordinal=i,planned=len(trajectories),
                    candidate_count=sum(len(v) for v in store.rows('reservoir')),gpu_job_sec=budget.snapshot()['used_sec'])
        states = sorted([s for bucket in store.rows('reservoir') for s in bucket],key=lambda r:r['state_id'])
        assert len(states)==len({s['state_id'] for s in states}), 'duplicate semantic mechanism states'
        # Freeze blind selection before computing any fresh teacher distributions.
        write_jsonl(OUT/'blind_selected_states.jsonl.gz',states)
        store.put('phase','candidates_frozen',dict(count=len(states),sha256=sha(OUT/'blind_selected_states.jsonl.gz')))
    states = list(read_jsonl(OUT/'blind_selected_states.jsonl.gz'))
    for i,meta in enumerate(states):
        budget.check()
        if store.get('state',meta['state_id']):
            continue
        record = measure_state(engine,meta)
        store.put('state',meta['state_id'],record)
        if i%32==0:
            progress('A1_teacher',ordinal=i,total=len(states))
    parents = store.rows('state')
    pairs = select_derived(parents,[2,3],64)
    write_jsonl(OUT/'blind_pair_parents.jsonl.gz',pairs)
    for i,meta in enumerate(pairs):
        budget.check()
        if store.get('pair',meta['state_id']):
            continue
        store.put('pair',meta['state_id'],counterfactual(engine,meta))
        if i%32==0:
            progress('A2_counterfactual',ordinal=i,total=len(pairs))
    noise = sorted(parents,key=lambda r:stable(['noise32',r['state_id']]))[:32]
    for meta in noise:
        budget.check()
        if store.get('noise',meta['state_id']):
            continue
        s = load_state(meta,engine.device)
        a,b = engine.fresh(s),engine.fresh(s)
        store.put('noise',meta['state_id'],dict(state_id=meta['state_id'],group=meta['group'],
            repeat=pair_metrics(a,b,engine.special),max_logit_abs=float((a-b).abs().max())))
    store.put('phase','mechanism',dict(status='finished',states=len(store.rows('state')),pairs=len(store.rows('pair'))))


def dcd(engine,store,budget):
    budget.switch('dcd')
    states = store.rows('state')
    # A1 teacher cap can leave selected but unmeasured states. DCD only needs X0/H0.
    if (OUT/'blind_selected_states.jsonl.gz').exists():
        available = {r['state_id']:r for r in states}
        for r in read_jsonl(OUT/'blind_selected_states.jsonl.gz'):
            available.setdefault(r['state_id'],r)
        states = list(available.values())
    parents = select_derived(states,[1],64,48)
    write_jsonl(OUT/'blind_dcd_parents.jsonl.gz',parents)
    try:
        for i,meta in enumerate(parents):
            budget.check()
            if store.get('dcd',meta['state_id']):
                continue
            store.put('dcd',meta['state_id'],dcd_one(engine,meta))
            progress('A3_DCD',completed=len(store.rows('dcd')),expected=256)
    except BudgetExceeded as exc:
        store.put('phase','dcd_budget',dict(reason=str(exc)))
    status = export_dcd(store)
    store.put('phase','dcd',dict(status=status,count=len(store.rows('dcd'))))


def generation(engine,store,budget):
    budget.switch('generation')
    rows = list(read_jsonl(REPO/'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz'))
    assert len(rows)==1033 and len({r['task_id'] for r in rows})==1033
    detailed = {r['task_id'] for r in sorted(rows,key=lambda r:stable(['trace32',r['task_id']]))[:32]}
    rows.sort(key=lambda r:stable(['generation_order',r['task_id']]))
    for i,r in enumerate(rows):
        # Rotate arm order independently of scores to limit shared-load confounding.
        arms = ['K1','Z4','H4-old','F4-chain']
        arms = arms[i%4:]+arms[:i%4]
        for arm in arms:
            key = arm+'::'+r['task_id']
            if store.get('generated',key):
                continue
            budget.check()
            prefix = engine.tokenizer.encode(r['prompt'],add_bos=True,add_eos=False)
            suffix = engine.tokenizer.encode(r['suffix'],add_bos=False,add_eos=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = engine.decode(prefix,suffix,r['task_id'],arm,detailed=r['task_id'] in detailed)
            torch.cuda.synchronize()
            result.update(arm=arm,task_id=r['task_id'],base_task_id=r['task_id'].split('/')[2],
                seed=42,order='global',generation_shared_gpu_sec=time.perf_counter()-start,
                detailed_trace=r['task_id'] in detailed,source_commit=head_commit())
            store.put('generated',key,result)
        progress('global_generation',cases_visited=i+1,logical_generated=store.count('generated'),
                 expected=4132,gpu_job_sec=budget.snapshot()['used_sec'])
    store.put('phase','generation',dict(status='complete',count=4132))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    store = Store()
    prior=OUT/'budget.local.json'
    if prior.exists() and json.loads(prior.read_text()).get('used_sec',0)>=43200:
        atomic_json(OUT/'gpu_exit.json',dict(status='original_total_budget_exhausted',at=utc(),generated=store.count('generated')))
        return 2
    lock = wait_gpu()
    if lock is None:
        atomic_json(OUT/'gpu_exit.json',dict(status='waiting_budget_exhausted',at=utc(),generated=len(store.rows('generated'))))
        return 3
    budget = Budget()
    stop = threading.Event()
    def heartbeat():
        ticks = 0
        while not stop.wait(1):
            budget.save()
            ticks += 1
            if ticks % 30 == 0:
                fields = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).strip().split(',')
                with (OUT/'gpu_observations.local.jsonl').open('a') as f:
                    f.write(json.dumps(dict(at=utc(),gpu=fields[0].strip(),free_mib=float(fields[1]),
                        used_mib=float(fields[2]),util_pct=float(fields[3]),resident=True))+'\n')
    budget.save()
    thread = threading.Thread(target=heartbeat,daemon=True)
    thread.start()
    def interrupted(_signum,_frame):
        raise StopRequested('external_interrupt_safe_idempotent_checkpoint')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    engine = model = head = None
    status = 'technical_failure'
    try:
        budget.check()
        frozen = json.loads((OUT/'frozen_protocol.json').read_text())
        assert sha(HEAD)==frozen['head']['sha256']
        from experiments.dreamon_singleline_adapter import OfficialHFTokenizerWrapper, ensure_modeling_rope_utils_available
        ensure_modeling_rope_utils_available()
        from transformers import AutoModel,AutoTokenizer
        from experiments.train_dreamon_markov_head import make_head
        tokenizer = OfficialHFTokenizerWrapper(AutoTokenizer.from_pretrained(SNAPSHOT,trust_remote_code=True,local_files_only=True))
        model = AutoModel.from_pretrained(SNAPSHOT,trust_remote_code=True,local_files_only=True,
            torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False)
        assert model.config.hidden_size==3584 and model.config.vocab_size==152064
        head = make_head(model.config,'cpu')
        checkpoint = torch.load(HEAD,map_location='cpu',weights_only=False)
        head.load_state_dict(checkpoint['head'],strict=True)
        del checkpoint
        head = head.to('cuda:0').eval().requires_grad_(False)
        assert next(model.parameters()).dtype==torch.bfloat16
        assert next(head.parameters()).dtype==torch.float32
        engine = Engine(model,tokenizer,FrozenHeadAdapter(head),frozen_special_ids(),budget.check)
        engine.zero = FrozenHeadAdapter(head,zero_override=True)
        progress('model_loaded',gpu_job_sec=budget.snapshot()['used_sec'],model_dtype=str(next(model.parameters()).dtype),
            head_dtype=str(next(head.parameters()).dtype),torch_version=torch.__version__)
        if store.get('phase','smoke') is None:
            smoke(engine,store)
        # Estimate from actual smoke without consulting any quality scores.
        timing = json.loads((OUT/'gpu_smoke.json').read_text())['checks']
        median_zero = float(np.median([x['observed_shared_gpu_sec']['Z4'] for x in timing]))
        atomic_json(OUT/'timing_estimate.json',dict(smoke_zero_median_sec=median_zero,
            rough_generation_sec=1033*median_zero*8,mechanism_max_sec=14400,dcd_max_sec=7200,
            forecast='rough conservative factor8 includes K1/F4 relative to chain Z4; shared load and prompt lengths uncertain',
            plan_unchanged=True))
        try:
            if store.get('phase','mechanism') is None:
                mechanism(engine,store,budget)
        except BudgetExceeded as exc:
            store.put('phase','mechanism',dict(status='partial',reason=str(exc),states=len(store.rows('state')),pairs=len(store.rows('pair'))))
        if store.get('phase','dcd') is None:
            dcd(engine,store,budget)
        generation(engine,store,budget)
        status = 'gpu_phases_finished'
    except (BudgetExceeded,StopRequested) as exc:
        status = 'budget_or_interrupt_checkpoint'
        store.put('phase','gpu_stop',dict(reason=str(exc),at=utc()))
    except Exception:
        atomic_json(OUT/'gpu_failure.local.json',dict(at=utc(),traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        # CPU phase starts only after this process releases CUDA and cooperative lock.
        if engine is not None:
            engine.hook.remove()
        del engine,model,head
        gc.collect()
        torch.cuda.empty_cache()
        stop.set()
        thread.join(timeout=5)
        budget.close()
        lock.close()
        atomic_json(OUT/'gpu_exit.json',dict(status=status,at=utc(),budget=json.loads((OUT/'budget.local.json').read_text()),
            generated=len(store.rows('generated')),states=len(store.rows('state')),pairs=len(store.rows('pair')),dcd=len(store.rows('dcd'))))
        store.db.close()
    return 0 if status=='gpu_phases_finished' else 2


def frozen_special_ids():
    return json.loads((OUT/'protocol_schema.json').read_text())['special_token_ids']


if __name__=='__main__':
    raise SystemExit(run())
