"""Deterministic source iterator and immutable FP32 shards shared by all heads."""
import collections
import gzip
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from experiments.context_foundation.common import (REPO, PREPARED, SNAPSHOT, HEAD,
    Store, atomic_json, read_jsonl, stable, sha, utc)
from experiments.context_foundation.evidence import state_id
from experiments.context_heads_v1.contract import HeadBatch
from experiments.context_heads_v1.training_v1.data import FIELDS, shapes
from experiments.context_heads_v1.integration_v1.adapter import config_from_a_schema

RUN = Path('/home/shx/.cache/dllm_infilling/context_h1_training_20261107_v1')
OUT = REPO/'analysis_outputs/dreamon_context_h1_training_20261107_v1'
AOUT = REPO/'analysis_outputs/dreamon_context_foundation_20261015_v1'
SEED = 20261107
TARGET = {'train':6250, 'validation':625}


def config():
    schema = json.loads((AOUT/'protocol_schema.json').read_text())
    return config_from_a_schema(schema, seed=SEED)


def cell(meta):
    return '|'.join(str(meta[k]) for k in ('split','K','history_length','order','span','arm'))


def cells():
    return {f'{s}|{k}|{h}|{o}|{sp}|{a}': n for s,n in TARGET.items()
            for k,h in ((2,1),(4,1),(4,2),(4,3)) for o in ('global','frontier')
            for sp in ('single','crossline') for a in ('Z4','H4-old')}


def progress(stage, **values):
    data = dict(stage=stage, at=utc(), pid=os.getpid(), **values)
    atomic_json(RUN/'progress.json',data)
    print(json.dumps(data,ensure_ascii=False),flush=True)


def sources():
    """Verify existing audits; never decode external-test records or read test labels."""
    path = RUN/'sources.jsonl.gz'
    if path.exists():
        return list(read_jsonl(path))
    old = REPO/'analysis_outputs/dreamon_markov_head_training_20260910_v2'
    audit = json.loads((AOUT/'source_audit.json').read_text())
    if sha(PREPARED) != audit['source']['prepared_sha256']:
        raise ValueError('prepared public source changed')
    split_path = old/'split_manifest.jsonl.zst'
    if sha(split_path) != audit['prepared_split_manifest_sha256']:
        raise ValueError('source split manifest changed')
    excluded = json.loads((old/'humaneval_exclusion_audit.json').read_text())
    excluded_groups = {g['problem_group_id'] for g in excluded['groups']}
    excluded_ids = {r for g in excluded['groups'] for r in g['member_record_ids']}
    with subprocess.Popen(['zstd','-dc',str(split_path)],stdout=subprocess.PIPE,text=True) as proc:
        members = {r['record_id']:r for r in map(json.loads,proc.stdout)}
        if proc.wait():
            raise RuntimeError('split manifest decompression failed')
    chosen, group_split = {}, {}
    with gzip.open(PREPARED,'rt') as f:
        for line in f:
            # Avoid even deserializing sealed prepared token payloads.
            if '"external_test"' in line:
                continue
            r = json.loads(line)
            if r['split'] not in TARGET:
                raise ValueError('unexpected split')
            rid,g = r['record_id'],r['problem_group_id']
            if rid in excluded_ids or g in excluded_groups:
                raise ValueError('decontamination violation')
            for key in ('split','source_seq_id','problem_group_id'):
                if r[key] != members[rid][key]:
                    raise ValueError('original membership mismatch')
            if group_split.setdefault(g,r['split']) != r['split']:
                raise ValueError('group split leak')
            meta = {k:r[k] for k in ('record_id','source_seq_id','problem_group_id','split')}
            if g not in chosen or stable(['source',rid]) < stable(['source',chosen[g]['record_id']]):
                chosen[g] = meta
    result = sorted(chosen.values(),key=lambda r:stable(['source_group',42,r['problem_group_id']]))
    from experiments.context_foundation.common import write_jsonl
    write_jsonl(path,result)
    atomic_json(RUN/'source_audit.json',dict(status='passed',original_audit=audit,
        selected_source_groups=dict(collections.Counter(r['split'] for r in result)),
        group_overlap=0,external_payload_opened=False,source_selection='one record/group, A source hash; fixed seed42 group order'))
    progress('source_prepared',groups=len(result))
    return result


def trajectories(tokenizer, store=None):
    rows = sources()
    from datasets import Dataset
    from experiments.dreamon_markov_head_training import _normalize_record,_canonical_json,_sha256_text
    from analysis.prepare_dreamon_markov_training_data import extract_code_block
    raw_root = Path('/home/shx/.cache/huggingface/datasets/OpenCoder-LLM___opc-sft-stage2/educational_instruct/0.0.0/7d28f40d579edd7c24402d17d0c7639f991e6f8d')
    if sha(raw_root/'opc-sft-stage2-train.arrow') != json.loads((AOUT/'source_audit.json').read_text())['source']['arrow_sha256']:
        raise ValueError('raw OpenCoder revision mismatch')
    ds = Dataset.from_file(str(raw_root/'opc-sft-stage2-train.arrow'))
    # Read only identity column first, select permitted source payloads thereafter.
    lookup = {seq:i for i,seq in enumerate(ds['seq_id'])}
    for meta in rows:
        raw = _normalize_record(ds[lookup[meta['source_seq_id']]])
        rid = _sha256_text(_canonical_json({k:raw[k] for k in ('instruction','code','entry_point','testcase')}))
        if rid != meta['record_id']:
            raise ValueError('source record content mismatch')
        response_prefix,code,response_suffix = extract_code_block(raw['output'],raw['code'])
        lines = code.strip('\n').splitlines(keepends=True)
        for span,width in (('single',1),('crossline',2)):
            holes = [(s,s+width) for s in range(1,len(lines)-width) if len(''.join(lines[s:s+width]).split()) >= 2]
            if not holes:
                if store is not None: store.put('source_gap',rid+'|'+span,dict(record_id=rid,split=meta['split'],span=span,reason='no_legal_source_hole'))
                continue
            start,end = min(holes,key=lambda h:stable(['hole',rid,span,h]))
            prefix = tokenizer.apply_chat_template([{'role':'user','content':raw['instruction']}],add_generation_prompt=True,tokenize=True)
            prefix += tokenizer.encode(response_prefix+''.join(lines[:start]),add_special_tokens=False)
            suffix = tokenizer.encode(''.join(lines[end:])+response_suffix+tokenizer.eos_token,add_special_tokens=False)
            if len(prefix)+64+len(suffix) > 1024:
                if store is not None: store.put('source_gap',rid+'|'+span,dict(record_id=rid,split=meta['split'],span=span,reason='frozen_source_context_over_1024'))
                continue
            for k in (2,4):
                for order in ('global','frontier'):
                    for arm in ('Z4','H4-old'):
                        yield dict(meta,group=meta['problem_group_id'],K=k,span=span,order=order,arm=arm,
                            case=rid+':'+span,prefix_ids=prefix,suffix_ids=suffix,hole_start_line=start,hole_end_line=end,
                            trajectory_id=stable(['formal_trajectory',42,rid,span,k,order,arm]))


class Bank:
    def __init__(self, store, cfg, root=RUN):
        self.root = Path(root)
        self.rows = store.rows('state')
        self.config = cfg
        self.cached_path, self.cached = None, None

    def indices(self, split):
        return np.array([i for i,r in enumerate(self.rows) if r['split']==split],dtype=np.int64)

    def batch(self, indices, device):
        values = collections.defaultdict(list)
        # Read each referenced shard once per batch, even under global shuffling.
        bypath = collections.defaultdict(list)
        for j,i in enumerate(indices):
            r=self.rows[int(i)]
            bypath[r['shard']].append((j,r['offset']))
        ordered = {}
        for path,items in bypath.items():
            with np.load(self.root/path,allow_pickle=False) as z:
                for name in (*FIELDS,'teacher_logits'):
                    if name not in ordered:
                        ordered[name] = np.empty((len(indices),*z[name].shape[1:]),dtype=z[name].dtype)
                    # One extraction per array, rather than once per row.
                    a=z[name]
                    for j,offset in items:
                        ordered[name][j]=a[offset]
        tensors={k:torch.from_numpy(v).to(device) for k,v in ordered.items()}
        teacher=tensors.pop('teacher_logits')
        return HeadBatch(**tensors),teacher


def shard_write(store, trajectory, picked, engine):
    if not picked:
        return 0,0.,0.,0
    cfg=config()
    arrays={}
    for name,shape in shapes(cfg).items():
        dtype=np.bool_ if name in ('selected_mask','history_mask') else np.float32 if name in ('old_hidden','old_logits','teacher_logits') else np.int64
        arrays[name]=np.full((len(picked),*shape),-1 if name=='chain_ids' else 0,dtype=dtype)
    # X0 / fixed time inputs are audit-only and are never passed to a head.
    arrays['audit_x0']=np.stack([s['x0'][0].cpu().numpy() for _,s in picked])
    arrays['audit_time_ids']=np.stack([s['time_ids'][0].cpu().numpy() for _,s in picked])
    metas=[]
    teacher_seconds=0.
    reused=0
    old_index=getattr(engine,'reuse_index',{})
    old_cache=Path('/home/shx/.cache/dllm_infilling/dreamon_context_foundation_20261015_v1')
    for i,(meta,s) in enumerate(picked):
        k=s['old_hidden'].shape[1]
        for name in FIELDS:
            a=s[name].detach().cpu().numpy()
            if name=='old_hidden': arrays[name][i,:k]=a[0]
            elif name=='history_mask': arrays[name][i,:,:k]=a[0]
            elif name in ('query_indices','old_logits'): arrays[name][i]=a[0]
            else: arrays[name][i,:k]=a[0]
        torch.cuda.synchronize()
        start=time.monotonic()
        prior=old_index.get(meta['state_id'])
        fresh=None
        if prior and (old_cache/prior['teacher_path']).exists() and sha(old_cache/prior['teacher_path'])==prior['teacher_sha256']:
            with np.load(old_cache/prior['raw_path'],allow_pickle=False) as z:
                matches=all(np.array_equal(s[n].cpu().numpy(),z[n]) for n in (*FIELDS,'x0','time_ids','attention'))
            if matches:
                with np.load(old_cache/prior['teacher_path'],allow_pickle=False) as z:
                    fresh=z['fresh_logits'].copy()
                reused+=1
        if fresh is None:
            fresh=engine.fresh(s).float().cpu().numpy()
        torch.cuda.synchronize()
        teacher_seconds+=time.monotonic()-start
        if fresh.dtype != np.float32 or not np.isfinite(fresh).all():
            raise FloatingPointError('teacher must be finite full vocabulary FP32')
        arrays['teacher_logits'][i,0]=fresh
        metas.append(dict(meta,offset=i,expand_masked=s['expand_masked'],sampling_key=s['key'],
                          round_index=s['round_index'],stage='early' if s['round_index']<6 else 'middle' if s['round_index']<16 else 'late'))
    path=RUN/'data'/('trajectory-'+trajectory['trajectory_id']+'.npz')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    start=time.monotonic()
    with tmp.open('wb') as f:
        np.savez(f,**arrays)
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)
    write_seconds=time.monotonic()-start
    # Shard is durable before metadata. Crash can only leave an unreferenced shard.
    with store.db:
        for m in metas:
            m['shard']=str(path.relative_to(RUN))
            store.db.execute('INSERT INTO rows VALUES(?,?,?)',('state',m['state_id'],json.dumps(m)))
    return path.stat().st_size,teacher_seconds,write_seconds,reused


def capacity(store):
    counts=collections.Counter(cell(r) for r in store.rows('state'))
    return counts,{k:max(0,n-counts[k]) for k,n in cells().items()}


def collect(engine, store, probe=False):
    counts,gaps=capacity(store)
    if not any(gaps.values()):
        return
    engine.reuse_index={r['state_id']:r for r in read_jsonl(AOUT/'state_metrics.jsonl.gz')}
    initial_rows=store.rows('state')
    existing={r['state_id'] for r in initial_rows}
    group_cells=collections.Counter((r['group'],cell(r)) for r in initial_rows)
    group_sets={sp:{r['group'] for r in initial_rows if r['split']==sp} for sp in TARGET}
    stats=store.get('phase','collection_stats') or dict(trajectories=0,states=0,bytes=0,teacher_sec=0.,write_sec=0.,decode_sec=0.,reused=0)
    start=time.monotonic()
    for t in trajectories(engine.tokenizer.tokenizer,store):
        if store.get('trajectory',t['trajectory_id']):
            continue
        histories=(1,) if t['K']==2 else (1,2,3)
        wanted={h for h in histories if counts[cell(dict(t,history_length=h))]<TARGET[t['split']]
                and group_cells[(t['group'],cell(dict(t,history_length=h)))]<4}
        if not wanted:
            continue
        candidates={h:{} for h in wanted}
        def observe(kind,s):
            if kind!='query': return
            h=int(s['history_mask'].sum())
            if h not in wanted: return
            q=int(s['query_indices'][0,0])
            earlier=((s['chain_ids'][0]==s['chain_ids'][0,q]) & (s['position_ids'][0]<s['position_ids'][0,q])).nonzero().flatten()
            if not torch.equal(earlier,s['history_mask'][0,0].nonzero().flatten()): return
            sid=state_id(s)
            if sid in existing or sid in candidates[h]: return
            cap=4-group_cells[(t['group'],cell(dict(t,history_length=h)))]
            if len(candidates[h])>=cap and sid>=max(candidates[h]): return
            meta={key:t[key] for key in ('record_id','source_seq_id','group','split','K','span','order','arm','trajectory_id','hole_start_line','hole_end_line','case')}
            meta.update(state_id=sid,history_length=h)
            copied={key:s[key].detach().clone() for key in (*FIELDS,'x0','attention','time_ids')}
            copied.update({key:s[key] for key in ('expand_masked','round_index','key')})
            copied['old_hidden']=copied['old_hidden'].float()
            candidates[h][sid]=(meta,copied)
            if len(candidates[h])>cap: del candidates[h][max(candidates[h])]
        torch.cuda.synchronize();ts=time.monotonic()
        result=engine.decode(t['prefix_ids'],t['suffix_ids'],t['case'],t['arm'],t['order'],observer=observe,requested_k=t['K'])
        torch.cuda.synchronize();decode_sec=time.monotonic()-ts
        picked=[]
        for h in sorted(wanted):
            key=cell(dict(t,history_length=h))
            picked.extend([candidates[h][sid] for sid in sorted(candidates[h])][:TARGET[t['split']]-counts[key]])
        if picked:
            estimated=(stats['bytes']/stats['states'] if stats['states'] else 4*(2*152064+4*3584)+16384)
            remaining=sum(gaps.values())
            required=int(estimated*remaining+20*2**30)
            free=shutil.disk_usage(RUN).free
            atomic_json(RUN/'capacity.json',dict(free_bytes=free,required_remaining_bytes=required,shortfall_bytes=max(0,required-free),estimated_bytes_per_state=estimated))
            if free<required:
                raise OSError(f'disk shortfall {required-free} bytes; no precision/quota reduction')
        size,teacher_sec,write_sec,reused=shard_write(store,t,picked,engine)
        for meta,_ in picked:
            counts[cell(meta)]+=1;existing.add(meta['state_id'])
            group_cells[(meta['group'],cell(meta))]+=1
            group_sets[meta['split']].add(meta['group'])
        stats=dict(trajectories=stats['trajectories']+1,states=stats['states']+len(picked),bytes=stats['bytes']+size,
            teacher_sec=stats['teacher_sec']+teacher_sec,write_sec=stats['write_sec']+write_sec,
            decode_sec=stats['decode_sec']+decode_sec,reused=stats['reused']+reused)
        # State metadata is sufficient to reconstruct counters after a crash.
        store.put('trajectory',t['trajectory_id'],dict(trajectory_id=t['trajectory_id'],states=len(picked),forward_count=result['forward_count'],decode_shared_sec=decode_sec))
        store.put('phase','collection_stats',stats)
        gaps={key:n-counts[key] for key,n in cells().items()}
        summary=dict(stats,counts=dict(counts),gaps=gaps,total=sum(counts.values()),target=220000,
            groups={s:len(g) for s,g in group_sets.items()})
        atomic_json(RUN/'data_summary.json',summary)
        seconds_per_state=(stats['teacher_sec']+stats['write_sec']+stats['decode_sec'])/max(1,stats['states'])
        progress('collection',completed=sum(counts.values()),target=220000,trajectories=stats['trajectories'],
            last_trajectory_states=len(picked),last_decode_shared_sec=decode_sec,last_teacher_shared_sec=teacher_sec,
            observed_seconds_per_state=seconds_per_state,rough_remaining_shared_hours=seconds_per_state*sum(gaps.values())/3600)
        if stats['states'] and not (RUN/'pilot_measurement.json').exists():
            path=RUN/picked[0][0].get('shard','data/trajectory-'+t['trajectory_id']+'.npz')
            ts=time.monotonic()
            with np.load(path,allow_pickle=False) as z:
                for key in z.files: _=z[key]
            read_sec=time.monotonic()-ts
            atomic_json(RUN/'pilot_measurement.json',dict(states=len(picked),bytes=size,read_sec=read_sec,
                write_sec=write_sec,teacher_shared_sec=teacher_sec,decode_shared_sec=decode_sec,
                projected_cache_bytes=int(size/len(picked)*220000),projected_collection_shared_hours=seconds_per_state*220000/3600,
                caveat='small fixed first trajectory; shared GPU load, lengths and bucket scarcity vary'))
        if not any(gaps.values()):
            break
        if probe and stats['trajectories']>=4 and stats['states']:
            return
    counts,gaps=capacity(store)
    atomic_json(RUN/'coverage.json',dict(counts=dict(counts),gaps=gaps,complete=not any(gaps.values())))
    if any(gaps.values()):
        raise RuntimeError('source groups exhausted: exact bucket gaps saved; formal training not started')
