"""One serial GPU process: public collection, H1 training, fixed generation, archive."""
import argparse
import collections
import gc
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch

from experiments.context_foundation.common import (REPO,SNAPSHOT,HEAD,Store,acquire,atomic_json,
    read_jsonl,write_jsonl,sha,utc,head_commit)
from experiments.context_foundation.adapter import Engine,FrozenHeadAdapter
from experiments.context_heads_v1.integration_v1.adapter import KeywordHeadAdapter
from .bank import RUN,OUT,AOUT,config,collect,progress,capacity,cell,Bank
from .train import initialize,train


def load_engine(*,progress_fn=progress):
    progress=progress_fn
    free,total=torch.cuda.mem_get_info()
    if free<18*2**30:
        raise torch.cuda.OutOfMemoryError('shared GPU has less than minimum unquantized model budget; no exclusive waiting')
    torch.cuda.set_per_process_memory_fraction(min(.90,(free-8*2**30)/total))
    progress('loading_model',free_bytes=free,reserve_bytes=8*2**30)
    from experiments.dreamon_singleline_adapter import OfficialHFTokenizerWrapper,ensure_modeling_rope_utils_available
    ensure_modeling_rope_utils_available()
    from transformers import AutoModel,AutoTokenizer
    from experiments.train_dreamon_markov_head import make_head
    frozen=json.loads((AOUT/'frozen_protocol.json').read_text())
    if sha(HEAD)!=frozen['head']['sha256']: raise ValueError('frozen static head changed')
    for name,value in frozen['model_runtime_sha256'].items():
        if sha(SNAPSHOT/name)!=value: raise ValueError('DreamOn runtime changed '+name)
    model=AutoModel.from_pretrained(SNAPSHOT,trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False)
    tokenizer=OfficialHFTokenizerWrapper(AutoTokenizer.from_pretrained(SNAPSHOT,trust_remote_code=True,local_files_only=True))
    head=make_head(model.config,'cpu')
    checkpoint=torch.load(HEAD,map_location='cpu',weights_only=False)
    head.load_state_dict(checkpoint['head'],strict=True);del checkpoint
    head=head.float().to('cuda:0').eval().requires_grad_(False)
    special=json.loads((AOUT/'protocol_schema.json').read_text())['special_token_ids']
    return Engine(model,tokenizer,FrozenHeadAdapter(head),special)


def release(engine):
    engine.hook.remove()
    # Caller must drop its reference before gc; hook itself must not retain model.


def technical_checks(engine,store):
    if (RUN/'technical_checks.json').exists(): return
    from experiments.context_heads_v1.loss import teacher_kl
    bank=Bank(store,config())
    batch,teacher=bank.batch(list(range(min(4,len(bank.rows)))),'cuda:0')
    head=initialize(config()).to('cuda:0')
    delta=head(batch).delta_logits
    if torch.count_nonzero(delta): raise AssertionError('formal zero residual mismatch')
    opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=0)
    norms=[]
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss=teacher_kl(batch.old_logits,head(batch).delta_logits,teacher,batch.history_mask.any(-1))
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(loss) or float(norm)<=0: raise AssertionError('no finite learning signal')
        norms.append(float(norm));opt.step()
    from experiments.context_heads_v1.contract import HeadBatch
    with torch.no_grad():
        whole=head(batch).delta_logits
        for i in range(whole.shape[0]):
            single=HeadBatch(**{k:v[i:i+1] if v is not None else None for k,v in vars(batch).items()})
            torch.testing.assert_close(head.step(single).delta_logits,whole[i:i+1],atol=2e-6,rtol=2e-5)
    row=bank.rows[0]
    with np.load(RUN/row['shard'],allow_pickle=False) as z:
        x=torch.from_numpy(z['audit_x0'][row['offset']:row['offset']+1]).to('cuda:0')
        time_ids=torch.from_numpy(z['audit_time_ids'][row['offset']:row['offset']+1]).to('cuda:0')
    active=time_ids==torch.arange(x.shape[1],device='cuda:0')[None]
    attention=active[:,None,None,:]&active[:,None,:,None]
    with torch.inference_mode():
        raw,h=engine.forward(x,attention,time_ids)
        valid=batch.selected_mask[0]
        pos=batch.position_ids[0,valid]
        aligned=h[:,(pos-1).clamp_min(0)].float()
        if not torch.equal(aligned,batch.old_hidden[:1,valid]): raise AssertionError('old hidden original target mapping mismatch')
        projected=engine.model.lm_head(h).float()
        if not torch.equal(projected[0,(pos-1).clamp_min(0)],raw[0,(pos-1).clamp_min(0)].float()): raise AssertionError('DreamOn logit alignment mismatch')
        q=int(batch.query_indices[0,0]);target=int(batch.position_ids[0,q])
        scores=engine.action_mask(raw[0,max(0,target-1)].float(),row['expand_masked'])
        if not torch.equal(scores,batch.old_logits[0,0]): raise AssertionError('old target logits mismatch')
    atomic_json(RUN/'technical_checks.json',dict(status='passed',real_fp32_full_vocab=True,zero_residual_exact=True,
        finite_nonzero_gradient_norms=norms,batch_step_consistent=True,original_target_alignment_exact=True,
        future_chain_barrier_checks='CPU unittest evidence; head and A teacher tests',
        smoke_optimizer_discarded=True,validation_improvement_gate=False))
    del head,opt,batch,teacher,delta
    gc.collect();torch.cuda.empty_cache()


def controls_reuse(store,*,run=RUN):
    """A uses precisely this model, population, sampler, structure and evaluator."""
    RUN=run
    audit_path=RUN/'control_reuse.json'
    if audit_path.exists(): return json.loads(audit_path.read_text())['reused']
    reference='43a52fd5fb43392e2ede7b6be659d3261dd6818a'
    files=('experiments/dreamon_singleline_order_parallel.py','experiments/dreamon_singleline_adapter.py',
        'experiments/run_k4_multiline_stage.py','experiments/k4_official_eval_worker.py',
        'experiments/lrdllm_dreamcoder_adapter.py','experiments/k4_batch_common.py',
        'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz',
        'analysis_outputs/dreamon_context_foundation_20261015_v1/frozen_protocol.json')
    differences=[]
    for name in files:
        if subprocess.check_output(['git','show',reference+':'+name],cwd=REPO)!=(REPO/name).read_bytes(): differences.append(name)
    old_adapter=subprocess.check_output(['git','show',reference+':experiments/context_foundation/adapter.py'],cwd=REPO,text=True)
    current=(REPO/'experiments/context_foundation/adapter.py').read_text()
    # Verify the entire historical default decoder, not merely selected helpers.
    restored=current.replace(", requested_k=None):", "):")
    restored=restored.replace("        if requested_k is not None and requested_k not in (1, 2, 4):\n            raise ValueError('requested_k must be 1, 2 or 4')\n",'')
    restored=restored.replace("requested_k=requested_k or (1 if arm == 'K1' else 4)","requested_k=1 if arm == 'K1' else 4")
    if restored!=old_adapter: differences.append('adapter semantics differ beyond optional K2 argument')
    frozen=json.loads((AOUT/'frozen_protocol.json').read_text())
    for name,value in frozen['model_shards'].items():
        if sha(SNAPSHOT/name)!=value: differences.append('model shard '+name)
    controls=[r for r in read_jsonl(AOUT/'per_case.jsonl.gz') if r['arm'] in ('K1','Z4','H4-old')]
    for arm in ('K1','Z4','H4-old'):
        rs=[r for r in controls if r['arm']==arm]
        if len(rs)!=1033 or len({r['task_id'] for r in rs})!=1033: differences.append(arm+' incomplete/duplicate')
    reused=not differences
    if reused:
        for r in controls:
            store.put('scored',r['arm']+'::'+r['task_id'],dict(r,reused_from='A 43a52fd'))
    result=dict(reused=reused,differences=differences,arms=['K1','Z4','H4-old'],rows=len(controls),
        model_revision=frozen['model']['revision'],head_sha256=frozen['head']['sha256'],
        semantics='A common-u FP32 sampler, identical source population/scoring/default K4 decoder; only optional K2 argument added',F4_rerun=False)
    atomic_json(audit_path,result)
    return reused


def generate(store):
    rows=list(read_jsonl(REPO/'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz'))
    if len(rows)!=1033 or len({r['task_id'] for r in rows})!=1033 or len({r['task_id'].split('/')[2] for r in rows})!=164: raise ValueError('population mismatch')
    reused=controls_reuse(store)
    arms=['H1'] if reused else ['H1','Z4','H4-old','K1']
    pending=[(r,a) for r in rows for a in arms if not store.get('generated',a+'::'+r['task_id'])]
    if not pending: return
    engine=load_engine()
    try:
        ck=torch.load(RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        candidate=initialize(config());candidate.load_state_dict(ck['head']);del ck
        candidate=candidate.to('cuda:0').eval().requires_grad_(False)
        old=engine.head
        new=KeywordHeadAdapter(candidate,legacy_zero_head=old.head)
        for r,arm in pending:
            key=arm+'::'+r['task_id']
            if store.get('generation_failure',key): continue
            engine.head=new if arm=='H1' else old
            prefix=engine.tokenizer.encode(r['prompt'],add_bos=True,add_eos=False)
            suffix=engine.tokenizer.encode(r['suffix'],add_bos=False,add_eos=True)
            torch.cuda.synchronize();ts=time.monotonic()
            try:
                result=engine.decode(prefix,suffix,r['task_id'],'H4-old' if arm=='H1' else arm,order='global')
                torch.cuda.synchronize()
                result.update(arm=arm,task_id=r['task_id'],base_task_id=r['task_id'].split('/')[2],seed=42,order='global',
                    generation_shared_gpu_sec=time.monotonic()-ts,source_commit=head_commit())
                store.put('generated',key,result)
            except torch.cuda.OutOfMemoryError:
                store.put('generation_failure',key,dict(arm=arm,task_id=r['task_id'],reason='OOM batch1; no automatic regeneration'))
                raise
            except Exception as exc:
                store.put('generation_failure',key,dict(arm=arm,task_id=r['task_id'],reason=str(exc)))
            progress('generation',completed=store.count('generated'),expected=1033*len(arms),arm=arm,task_id=r['task_id'])
    finally:
        release(engine)


def score(store,*,progress_fn=progress):
    progress=progress_fn
    from experiments.run_k4_multiline_stage import official_evaluate
    contexts={r['task_id']:r for r in read_jsonl(REPO/'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz')}
    for r in store.rows('generated'):
        key=r['arm']+'::'+r['task_id']
        if store.get('scored',key) or store.get('score_failure',key): continue
        try:
            result=official_evaluate(contexts[r['task_id']],r['output'])
            store.put('scored',key,dict(r,**result))
        except Exception as exc:
            store.put('score_failure',key,dict(key=key,error=str(exc),technical=True))
        if store.count('scored')%64==0: progress('scoring',completed=store.count('scored'),expected=4132)


def archive(store,status):
    from analysis.analyze_context_foundation import paired_quality,grouped_bootstrap,interval
    OUT.mkdir(parents=True,exist_ok=True)
    counts,gaps=capacity(store)
    states=store.rows('state');scored=store.rows('scored')
    groups={s:len({r['group'] for r in states if r['split']==s}) for s in ('train','validation')}
    data=dict(total=len(states),target=220000,counts=dict(counts),gaps=gaps,source_groups=groups,
        group_overlap=len({r['group'] for r in states if r['split']=='train'} & {r['group'] for r in states if r['split']=='validation'}),
        stage_counts=dict(collections.Counter(r['stage'] for r in states)),source_hole_gaps=len(store.rows('source_gap')))
    atomic_json(OUT/'data_summary.json',data)
    table=[]
    for arm in ('H1','Z4','H4-old','K1'):
        rs=[r for r in scored if r['arm']==arm]
        passed=sum(r['passed'] for r in rs)
        table.append(dict(arm=arm,N=len(rs),passed=passed,rate=passed/len(rs) if rs else None,missing=1033-len(rs),duplicates=len(rs)-len({r['task_id'] for r in rs}),
            forward_count=sum(r['forward_count'] for r in rs),actual_submitted_count=sum(r['actual_submitted_count'] for r in rs),head_calls=sum(r['head_calls'] for r in rs),
            expand_count=sum(r['expand_count'] for r in rs),delete_count=sum(r['delete_count'] for r in rs),shared_gpu_sec=sum(r['generation_shared_gpu_sec'] for r in rs)))
    comparisons=[paired_quality(scored,'H1',arm) for arm in ('Z4','H4-old','K1')]
    arms={a:{r['task_id']:r for r in scored if r['arm']==a} for a in ('H1','Z4','K1')}
    common=sorted(set.intersection(*(set(v) for v in arms.values())))
    num=sum(int(arms['H1'][k]['passed'])-int(arms['Z4'][k]['passed']) for k in common)
    den=sum(int(arms['K1'][k]['passed'])-int(arms['Z4'][k]['passed']) for k in common)
    base_groups=sorted({arms['H1'][k]['base_task_id'] for k in common})
    recovery=dict(matched=len(common),numerator=num,denominator=den,value=num/den if den>0 else None,reason='defined' if den>0 else 'nonpositive denominator')
    if base_groups and den>0:
        nums=np.array([sum(int(arms['H1'][k]['passed'])-int(arms['Z4'][k]['passed']) for k in common if arms['H1'][k]['base_task_id']==g) for g in base_groups])
        dens=np.array([sum(int(arms['K1'][k]['passed'])-int(arms['Z4'][k]['passed']) for k in common if arms['H1'][k]['base_task_id']==g) for g in base_groups])
        w=np.random.default_rng(20261015).multinomial(len(base_groups),np.full(len(base_groups),1/len(base_groups)),size=10000)
        d=w@dens;valid=d>0
        recovery.update(ci95_positive_denominator=interval((w@nums)[valid]/d[valid]),groups=len(base_groups),bootstrap_valid=int(valid.sum()))
    result=dict(status=status,table=table,comparisons=comparisons,pure_learning_recovery=recovery,
        generation_technical_failures=store.rows('generation_failure'),score_technical_failures=store.rows('score_failure'),
        expected_base_groups=164,paired_interval='10000 base-task cluster bootstrap; 1033 lines are not independent tasks',
        caveat='Only H1 retrained; no same-data retrained H0/Hctx/Hgated comparison. Shared GPU times are observations, not fair speedups.')
    atomic_json(OUT/'generation_comparison.json',result)
    write_jsonl(RUN/'generated_rows.jsonl.gz',store.rows('generated'))
    write_jsonl(RUN/'per_case.jsonl.gz',scored)
    compact=('training_summary.json','curve.json','pilot_measurement.json','capacity.json','control_reuse.json','exit.json','environment.json','source_audit.json','coverage.json','technical_checks.json','plan.json','shell_exit.json')
    for name in compact:
        if (RUN/name).exists(): shutil.copyfile(RUN/name,OUT/name)
    for p in RUN.glob('validation_epoch_*.json'): shutil.copyfile(p,OUT/p.name)
    # Full outputs and large FP32 bank/checkpoints remain server-local.
    atomic_json(OUT/'evidence_index.json',dict(run=str(RUN),code_commit=head_commit(),
        bank='queue.sqlite rows.kind=state -> data/*.npz offsets; full selected-set FP32 tensors',
        checkpoints=str(RUN/'checkpoints'),generated=str(RUN/'generated_rows.jsonl.gz'),scored=str(RUN/'per_case.jsonl.gz'),
        logs=str(RUN/'pipeline.log'),progress=str(RUN/'progress.json'),exit=str(RUN/'exit.json'),
        disk_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),resume='bash experiments/context_h1_training_v1/launch.sh'))
    text=['# H1正式训练（2026-11-07）',f'状态：{status}。数据实际{len(states)}/220000；源程序组train/validation={groups}。',
        f'缺口总计{sum(gaps.values())}，详见data_summary.json；保留真实K2/K4及32桶，原分组隔离，external_test未打开。',
        'FP32全词表KL，rank256，强度1；seed20261107，W_out=0、gamma=1；AdamW lr3e-4，有效batch128，最多5轮。',
        '','|组|通过/N|通过率|遗漏|','|---|---:|---:|---:|']
    for r in table: text.append(f"|{r['arm']}|{r['passed']}/{r['N']}|{r['rate'] if r['rate'] is not None else 'N/A'}|{r['missing']}|")
    if (RUN/'training_summary.json').exists():
        t=json.loads((RUN/'training_summary.json').read_text())
        text += [f"best epoch={t['best_epoch']}；last epoch=5。best为未训练初始权重={t['best_is_untrained_initial']}。第5轮仍明显改善={t['not_fully_converged_at_epoch5']}，不追加轮次。"]
    for r in comparisons: text.append(f"H1−{r['b']}：配对{r['matched']}，救回{r['help']}、改错{r['harm']}；差值百分点评估区间{r['ci95']}。")
    text += [f'纯学习恢复比例：{recovery}。','',
        f'大缓存、生成、评分、失败日志及优化器只在服务器 {RUN}；精确位置见evidence_index.json。',
        '共享唯一cuda:0，单GPU进程，阶段释放DreamOn，不等待独占；时间仅观测值，不称公平加速。',
        '其余结构可直接用bank.Bank读取同一queue.sqlite，不重采teacher；训练日程/初始化规则见train.py及README.zh.md。',
        '本轮没有训练H0/Hctx/Hgated/H2/H3/H4，没有自身轨迹补训、超参搜索或多seed。',
        f'代码提交：{head_commit()}；恢复：`bash experiments/context_h1_training_v1/launch.sh`。']
    (OUT/'report.zh.md').write_text('\n\n'.join(text)+'\n')
    ledger=REPO/'docs/paper_agent/activity_ledger.zh.md'
    marker='H1 formal 20261107 archive '+status
    if marker not in ledger.read_text():
        with ledger.open('a') as f: f.write(f'\n\n## 2026-11-07 {marker}\n\n数据{len(states)}/220000；结果analysis_outputs/dreamon_context_h1_training_20261107_v1/；只H1；共享cuda:0，其他路线不启动。\n')


def publish():
    paths=['experiments/context_h1_training_v1','experiments/context_foundation/adapter.py',
        'analysis_outputs/dreamon_context_h1_training_20261107_v1','docs/paper_agent/activity_ledger.zh.md']
    subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode:
        subprocess.run(['git','commit','-m','results: archive finite H1 formal training pipeline'],cwd=REPO,check=True)
    r=subprocess.run(['git','push','-u','origin','HEAD'],cwd=REPO,timeout=90,capture_output=True,text=True)
    atomic_json(RUN/'publication.json',dict(returncode=r.returncode,stdout=r.stdout[-2000:],stderr=r.stderr[-2000:],local_commit=head_commit()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    lock=acquire(RUN/'worker.lock')
    if (RUN/'exit.json').exists() and json.loads((RUN/'exit.json').read_text()).get('status')=='complete':
        print('Formal pipeline already complete; no rerun',flush=True);return
    torch.set_num_threads(4)
    if torch.cuda.device_count()!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!='0': raise RuntimeError('use unique visible GPU0')
    store=Store(RUN/'queue.sqlite')
    atomic_json(RUN/'environment.json',dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),code_commit=head_commit(),gpu=subprocess.check_output(['nvidia-smi'],text=True),shared_gpu=True,workers=1))
    stopped=threading.Event()
    def observe():
        while not stopped.wait(30):
            try:
                value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                with (RUN/'gpu_observations.jsonl').open('a') as f: f.write(json.dumps(dict(at=utc(),observed=value))+'\n')
            except Exception: pass
    thread=threading.Thread(target=observe,daemon=True);thread.start()
    def interrupted(*_): raise InterruptedError('interrupted; resume from durable phase cursor')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    engine=None;status='technical_failure';rc=1
    try:
        counts,gaps=capacity(store)
        if any(gaps.values()):
            engine=load_engine()
            collect(engine,store,probe=args.probe)
            technical_checks(engine,store)
            release(engine);del engine;engine=None;gc.collect();torch.cuda.empty_cache()
        if args.probe:
            progress('probe_complete',completed=store.count('state'));status='probe_complete';rc=0;return
        train(store)
        gc.collect();torch.cuda.empty_cache()
        generate(store)
        gc.collect();torch.cuda.empty_cache()
        score(store)
        complete=all(sum(r['arm']==a for r in store.rows('scored'))==1033 for a in ('H1','Z4','H4-old','K1'))
        status='complete' if complete else 'partial_technical_failures';rc=0 if complete else 2
    except torch.cuda.OutOfMemoryError:
        status='blocked_shared_gpu_memory';traceback.print_exc()
    except OSError:
        status='blocked_storage_or_io';traceback.print_exc()
    except InterruptedError:
        status='interrupted_resumable';traceback.print_exc()
    except Exception:
        traceback.print_exc()
    finally:
        if engine is not None: release(engine)
        stopped.set();thread.join(timeout=2)
        if not args.probe:
            atomic_json(RUN/'exit.json',dict(status=status,exit_code=rc,at=utc()))
            try: archive(store,status);publish()
            except Exception: traceback.print_exc()
        lock.close()
    sys.exit(rc)


if __name__=='__main__': main()
