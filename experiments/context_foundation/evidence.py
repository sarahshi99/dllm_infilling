"""Paired distribution evidence; no generation scoring or sample selection by effect."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch

from experiments.context_foundation.adapter import distribution, inverse_cdf, uniform
from experiments.context_foundation.common import *

TENSOR_FIELDS = ['x0','attention','time_ids','old_hidden','position_ids','chain_ids','selected_mask',
                 'query_indices','old_logits','selected_old_logits','token_ids','token_state',
                 'history_mask','branch_logp']


def state_id(s):
    # Arm/order labels cannot turn the same mechanism state into distinct samples.
    return stable(['state',s['case'],s['dynamic_length'],s['time_ids'].tolist(),
                   s['position_ids'].tolist(),s['query_indices'].tolist(),s['token_ids'].tolist(),
                   s['token_state'].tolist(),s['history_mask'].tolist(),s['x0'].tolist()])


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    assert all(np.asarray(v).dtype != object for v in arrays.values())
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def save_state(s, meta):
    sid = meta['state_id']
    path = CACHE / 'states' / (sid+'.npz')
    arrays = {k:s[k].detach().cpu().float().numpy() if s[k].dtype == torch.bfloat16
              else s[k].detach().cpu().numpy() for k in TENSOR_FIELDS}
    save_npz(path, **arrays)
    meta = dict(meta, raw_path='states/'+sid+'.npz', raw_sha256=sha(path), raw_bytes=path.stat().st_size,
                expand_masked=s['expand_masked'], round_index=s['round_index'], sampling_key=s['key'],
                prefix_length=s['prefix_length'], maximum=s['maximum'], dynamic_length=s['dynamic_length'])
    return meta


def load_state(meta, device):
    p = CACHE / meta['raw_path']
    assert sha(p) == meta['raw_sha256'], 'raw shard hash mismatch: '+meta['state_id']
    with np.load(p, allow_pickle=False) as arr:
        s = {k:torch.from_numpy(arr[k].copy()).to(device) for k in TENSOR_FIELDS}
    return dict(s, **{k:meta[k] for k in ['expand_masked','round_index','maximum','dynamic_length','prefix_length']})


def candidate_cell(meta):
    return '|'.join(map(str,[meta['order'],meta['history_length'],meta['span'],meta['split'],meta['arm']]))


def reserve_candidate(store, s, trajectory):
    length = int(s['history_mask'].sum())
    if length not in (1,2,3):
        return 'ineligible_history_length'
    # Collection is before target commit; prefixes must start at the chain anchor.
    query = int(s['query_indices'][0,0])
    slots = (s['chain_ids'][0] == s['chain_ids'][0,query]).nonzero().flatten().tolist()
    earlier = [v for v in slots if v < query]
    visible = s['history_mask'][0,0].nonzero().flatten().tolist()
    if visible != earlier or any(int(s['token_ids'][0,v]) == 0 and int(s['token_state'][0,v]) != 1 for v in visible):
        return 'not_legal_chain_anchor_prefix'
    meta = {k:trajectory[k] for k in ['trajectory_id','case','record_id','source_seq_id','group','split','span','order','arm','hole_start_line','hole_end_line']}
    meta.update(state_id=state_id(s), history_length=length, target_slot=query,
                source_policy=trajectory['arm'], stage='early' if s['round_index'] < 6 else 'middle' if s['round_index'] < 16 else 'late',
                selected_positions=s['position_ids'][0].tolist(), chain_ids=s['chain_ids'][0].tolist())
    cell = candidate_cell(meta)
    owner = store.get('state_owner',meta['state_id'])
    if owner is not None and owner['cell'] != cell:
        return 'duplicate_other_origin_or_order'
    current = store.get('reservoir',cell) or []
    if any(r['state_id'] == meta['state_id'] for r in current):
        return 'duplicate_same_cell'
    candidates = sorted(current+[meta], key=lambda r:r['state_id'])
    # Four/group/branch ensures <=8/group in the final cell and exact 50/50 origin.
    groups, selected = collections.Counter(), []
    cap = 96 if meta['split'] == 'train' else 32
    for r in candidates:
        if groups[r['group']] < 4 and len(selected) < cap:
            selected.append(r)
            groups[r['group']] += 1
    if not any(r['state_id'] == meta['state_id'] for r in selected):
        return 'hash_or_group_quota_not_retained'
    saved = save_state(s,meta)
    selected = [saved if r['state_id'] == meta['state_id'] else r for r in selected]
    store.put('reservoir',cell,selected)
    store.put('state_owner',meta['state_id'],dict(cell=cell))
    return 'retained_by_blind_hash'


def pair_metrics(fresh, estimate, special):
    # Primary full-vocabulary teacher distribution. No epsilon is added to KL.
    lp, lq = torch.log_softmax(fresh.float(),-1), torch.log_softmax(estimate.float(),-1)
    p,q = lp.exp(),lq.exp()
    pd,qd = distribution(fresh),distribution(estimate)
    sp,sq = torch.isfinite(pd),torch.isfinite(qd)
    kl_inf = bool((sp & ~sq).any())
    kl_deploy = 'infinite' if kl_inf else float((pd.exp()[sp]*(pd[sp]-qd[sp])).sum())
    tracked = [i for i in [151643,151666,151667] if i in special and i < p.numel()]
    structural_probs = {str(i):dict(teacher_T1=float(p[i]),estimate_T1=float(q[i]),
        teacher_deploy=float(pd[i].exp()),estimate_deploy=float(qd[i].exp())) for i in tracked}
    special = torch.as_tensor(sorted(special),device=p.device)
    return dict(kl_T1=float((p*(lp-lq)).sum()),tv_T1=float((p-q).abs().sum()/2),
        tv_deploy=float((pd.exp()-qd.exp()).abs().sum()/2),kl_deploy=kl_deploy,
        support_teacher_only=int((sp & ~sq).sum()),support_estimate_only=int((sq & ~sp).sum()),
        support_teacher_only_mass=float(pd.exp()[sp & ~sq].sum()),support_estimate_only_mass=float(qd.exp()[sq & ~sp].sum()),
        support_teacher_size=int(sp.sum()),support_estimate_size=int(sq.sum()),
        structural_token_probabilities=structural_probs,
        special_teacher_T1=float(p[special].sum()),special_estimate_T1=float(q[special].sum()),
        special_teacher_deploy=float(pd.exp()[special].sum()),special_estimate_deploy=float(qd.exp()[special].sum()))


@torch.inference_mode()
def measure_state(engine, meta):
    s = load_state(meta, engine.device)
    fresh = engine.fresh(s)
    stale = s['old_logits'][0,0]
    head = stale + engine.correction(s)
    path = CACHE/'teachers'/(meta['state_id']+'.npz')
    save_npz(path, fresh_logits=fresh.cpu().numpy())
    return dict(meta, teacher_path='teachers/'+path.name, teacher_sha256=sha(path),
        storage='FP32 logits, lossless npz; exact BF16 hidden converted to FP32; no quantization',
        stale=pair_metrics(fresh,stale,engine.special),head=pair_metrics(fresh,head,engine.special))


@torch.inference_mode()
def counterfactual(engine, meta):
    s = load_state(meta, engine.device)
    visible = s['history_mask'][0,0].nonzero().flatten().tolist()
    intervention_slot = visible[0]
    original = int(s['token_ids'][0,intervention_slot])
    branch_lp = s['branch_logp'][0,intervention_slot]
    valid = torch.isfinite(branch_lp)
    valid[list(engine.special)] = False
    valid[original] = False
    if not bool(valid.any()):
        return dict(state_id=meta['state_id'],status='no_alternative_ordinary_support')
    alternative_lp = branch_lp.masked_fill(~valid,-torch.inf).log_softmax(-1)
    intervention_key = [CONTRACT,42,meta['state_id'],'counterfactual_earliest_token']
    alternative = int(inverse_cdf(alternative_lp,uniform(intervention_key)))
    changed = s['token_ids'].clone()
    changed[0,intervention_slot] = alternative
    with np.load(CACHE/meta['teacher_path'],allow_pickle=False) as t:
        left = torch.from_numpy(t['fresh_logits'].copy()).to(engine.device)
    right = engine.fresh(s,changed)
    # Recompute left as well: comparison/noise is not cached vs newly rounded data.
    left_repeat = engine.fresh(s)
    stale = s['old_logits'][0,0]
    head_left = stale+engine.correction(s)
    s_changed = dict(s, token_ids=changed)
    head_right = stale+engine.correction(s_changed)
    path = CACHE/'pairs'/(meta['state_id']+'.npz')
    save_npz(path, fresh_left=left_repeat.cpu().numpy(),fresh_right=right.cpu().numpy())
    return dict(state_id=meta['state_id'],status='complete',group=meta['group'],split=meta['split'],
        order=meta['order'],span=meta['span'],history_length=meta['history_length'],
        source_policy=meta['arm'], intervention_slot=intervention_slot,
        intervention_position=int(s['position_ids'][0,intervention_slot]),original_token=original,
        alternative_token=alternative,original_branch_logp=float(branch_lp[original]),
        alternative_branch_logp=float(branch_lp[alternative]),alternative_draw_logp=float(alternative_lp[alternative]),
        intervention_key=intervention_key, latest_token=int(s['token_ids'][0,visible[-1]]),
        latest_position=int(s['position_ids'][0,visible[-1]]),
        teacher_change=pair_metrics(left_repeat,right,engine.special),
        repeat_noise=pair_metrics(left,left_repeat,engine.special),
        left_stale=pair_metrics(left_repeat,stale,engine.special),right_stale=pair_metrics(right,stale,engine.special),
        left_head=pair_metrics(left_repeat,head_left,engine.special),right_head=pair_metrics(right,head_right,engine.special),
        head_branch_tv_T1=pair_metrics(head_left,head_right,engine.special)['tv_T1'],
        raw_path='pairs/'+path.name,raw_sha256=sha(path),
        interpretation='intervention, not two natural independent trajectories; latest decision fixed; no correctness labels')


def select_derived(states, history, per_cell, train_target=None):
    chosen = []
    for order in ['global','frontier']:
        for span in ['single','crossline']:
            for length in history:
                candidates = sorted((s for s in states if s['order']==order and s['span']==span and s['history_length']==length),
                                    key=lambda s:stable(['derived',history,s['state_id']]))
                if train_target is None:
                    chosen += candidates[:per_cell]
                else:
                    first = ([s for s in candidates if s['split']=='train'][:train_target] +
                             [s for s in candidates if s['split']=='validation'][:per_cell-train_target])
                    seen = {s['state_id'] for s in first}
                    first += [s for s in candidates if s['state_id'] not in seen][:per_cell-len(first)]
                    chosen += first
    return chosen


@torch.inference_mode()
def dcd_one(engine, meta):
    s = load_state(meta,engine.device)
    second = int(s['query_indices'][0,0])
    first = s['history_mask'][0,0].nonzero().flatten().tolist()[0]
    old = s['selected_old_logits'][0]
    ordinary = torch.ones(old.shape[-1],dtype=torch.bool,device=old.device)
    ordinary[list(engine.special)] = False
    def candidates(logits):
        return torch.argsort(logits.masked_fill(~ordinary,-torch.inf),descending=True,stable=True)[:16]
    a,b = candidates(old[first]),candidates(old[second])
    l1,l2 = old[first].log_softmax(-1),old[second].log_softmax(-1)
    values = dict(candidate_ids_1=a.cpu().numpy(),candidate_ids_2=b.cpu().numpy(),
        old_p1=l1[a].exp().cpu().numpy(),old_p2=l2[b].exp().cpu().numpy(),
        old_logp1=l1[a].cpu().numpy(),old_logp2=l2[b].cpu().numpy(),
        old_candidate_mass1=np.array(float(l1[a].exp().sum())),old_candidate_mass2=np.array(float(l2[b].exp().sum())),
        old_special_mass1=np.array(float(l1[list(engine.special)].exp().sum())),
        old_special_mass2=np.array(float(l2[list(engine.special)].exp().sum())))
    matrices = collections.defaultdict(list)
    for token in a.tolist():
        s['token_ids'][0,first] = token
        fresh = engine.fresh(s).log_softmax(-1)
        head = (old[second]+engine.correction(s)).log_softmax(-1)
        for label, lp in [('head',head),('fresh',fresh)]:
            matrices[label+'_logcond2'].append(lp[b].cpu().numpy())
            matrices[label+'_cond2'].append(lp[b].exp().cpu().numpy())
            matrices[label+'_candidate_mass'].append(float(lp[b].exp().sum()))
            matrices[label+'_special_mass'].append(float(lp[list(engine.special)].exp().sum()))
    values.update({k:np.asarray(v) for k,v in matrices.items()})
    values['state_id'] = np.array(meta['state_id'],dtype='U64')
    path = CACHE/'dcd'/(meta['state_id']+'.npz')
    save_npz(path,**values)
    return dict(state_id=meta['state_id'],source_state_id=meta['state_id'],order=meta['order'],span=meta['span'],
        group=meta['group'],split=meta['split'],coordinates=[int(s['position_ids'][0,first]),int(s['position_ids'][0,second])],
        raw_path='dcd/'+path.name,raw_sha256=sha(path))


def export_dcd(store):
    rows = store.rows('dcd')
    tensors = collections.defaultdict(list)
    for r in rows:
        assert sha(CACHE/r['raw_path']) == r['raw_sha256']
        with np.load(CACHE/r['raw_path'], allow_pickle=False) as arr:
            for name in arr.files:
                tensors[name].append(arr[name].copy())
    if rows:
        arrays = {k:np.stack(v) for k,v in tensors.items()}
    else:
        arrays = dict(state_id=np.empty((0,),dtype='U64'),candidate_ids_1=np.empty((0,16),dtype=np.int64),
            candidate_ids_2=np.empty((0,16),dtype=np.int64))
        for k in ['old_p1','old_p2','old_logp1','old_logp2']:
            arrays[k] = np.empty((0,16),dtype=np.float32)
        for k in ['head_cond2','fresh_cond2','head_logcond2','fresh_logcond2']:
            arrays[k] = np.empty((0,16,16),dtype=np.float32)
        for k in ['head_candidate_mass','fresh_candidate_mass','head_special_mass','fresh_special_mass']:
            arrays[k] = np.empty((0,16),dtype=np.float32)
        for k in ['old_candidate_mass1','old_candidate_mass2','old_special_mass1','old_special_mass2']:
            arrays[k] = np.empty((0,),dtype=np.float32)
    save_npz(OUT/'dcd_states.npz',**arrays)
    frozen = json.loads((OUT/'frozen_protocol.json').read_text())
    write_jsonl(OUT/'dcd_metadata.jsonl', [{k:v for k,v in r.items() if k not in ['raw_path','raw_sha256']} |
        dict(model_revision=frozen['model']['revision'],head_sha256=frozen['head']['sha256'],
             adapter_sha256=sha(REPO/'experiments/context_foundation/adapter.py')) for r in rows])
    status = 'complete' if len(rows)==256 else 'partial'
    with np.load(OUT/'dcd_states.npz',allow_pickle=False) as z:
        assert len(z['state_id'])==len(rows)
        assert all(z[k].dtype != object for k in z.files)
        for k in ['old_logp1','old_logp2','head_logcond2','fresh_logcond2']:
            assert np.isfinite(z[k]).all()
    atomic_json(OUT/'dcd_manifest.json',dict(contract_id=CONTRACT,schema_version=1,status=status,
        source_commit=head_commit(),producer_branch=BRANCH,count=len(rows),expected=256,coverage_gap=256-len(rows),
        T=1, probability_authority='full-vocabulary log_softmax FP32 before extraction; log fields are numerical authority',
        causal_anchor='first position distribution is old p1',
        interpretation='truncated K2-style diagnostic derived from K4 A1; not full-vocabulary joint or actual K2 generation; no candidate renormalization',
        model=frozen['model'],model_shards_sha256=frozen['model_shards'],head=frozen['head'],
        tokenizer_sha256=frozen['tokenizer_sha256'],adapter_sha256=sha(REPO/'experiments/context_foundation/adapter.py'),
        data=manifest_entries([OUT/'dcd_states.npz',OUT/'dcd_metadata.jsonl'])))
    paths=[str((OUT/n).relative_to(REPO)) for n in ['dcd_states.npz','dcd_metadata.jsonl','dcd_manifest.json']]
    if (OUT/'blind_dcd_parents.jsonl.gz').exists():
        paths.append(str((OUT/'blind_dcd_parents.jsonl.gz').relative_to(REPO)))
    print('DCD staged files:',paths,flush=True)
    subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=REPO).returncode:
        subprocess.run(['git','commit','--only','-m','results: publish chain-local DCD matrices for CPU analysis','--',*paths],cwd=REPO,check=True)
    publish_ready('dcd_ready.json', OUT/'dcd_manifest.json',status)
    return status
