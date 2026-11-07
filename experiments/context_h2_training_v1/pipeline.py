"""Finite H2 pipeline: public dependency -> shared training -> fixed best eval -> archive."""
import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import threading

import torch
from experiments.context_foundation.common import (REPO,Store,acquire,atomic_json,read_jsonl,write_jsonl,utc,head_commit)
from experiments.context_h1_training_v1 import pipeline as shared
from experiments.context_h1_training_v1.train import initialize,train
from experiments.context_heads_v1.integration_v1.adapter import KeywordHeadAdapter
from .dependency import RUN,OUT,PUBLIC,RESUME,progress,wait_for_public,open_bank
from .checks import technical


class CountedAdapter(KeywordHeadAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gru_calls=0;self.gru_rows=0;self.active_updates=0
        def count(_module,args,_result):
            self.gru_calls+=1;self.gru_rows+=args[0].shape[0]
        self.handle=self.context_head.gru.register_forward_hook(count)
    def forward(self,**values):
        self.active_updates+=int(values['history_mask'].sum())
        return super().forward(**values)


def generate(store,bank):
    contexts=list(read_jsonl(REPO/'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz'))
    if len(contexts)!=1033 or len({r['task_id'] for r in contexts})!=1033 or len({r['task_id'].split('/')[2] for r in contexts})!=164:
        raise ValueError('1033 line / 164 source population mismatch')
    pending=[r for r in contexts if not store.get('generated','H2::'+r['task_id']) and not store.get('generation_failure','H2::'+r['task_id'])]
    if not pending: return
    engine=None;candidate=None;adapter=None
    try:
        engine=shared.load_engine(progress_fn=progress)
        ck=torch.load(RUN/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        candidate=initialize(bank.config,'H2-GRU');candidate.load_state_dict(ck['head']);del ck
        candidate=candidate.to('cuda:0').eval().requires_grad_(False)
        adapter=CountedAdapter(candidate,legacy_zero_head=engine.head.head)
        engine.head=adapter
        for r in pending:
            key='H2::'+r['task_id']
            # Do not choose among repeated stochastic attempts after interruption.
            if store.get('generation_started',key) and not store.get('oom_deferred',key):
                store.put('generation_failure',key,dict(arm='H2',task_id=r['task_id'],reason='interrupted in-flight attempt; no automatic regeneration'))
                continue
            prefix=engine.tokenizer.encode(r['prompt'],add_bos=True,add_eos=False)
            suffix=engine.tokenizer.encode(r['suffix'],add_bos=False,add_eos=True)
            before=(adapter.gru_calls,adapter.gru_rows,adapter.active_updates)
            store.put('generation_started',key,dict(at=utc(),best=str(RUN/'checkpoints/best.pt')))
            torch.cuda.synchronize();ts=time.monotonic()
            try:
                result=engine.decode(prefix,suffix,r['task_id'],'H4-old',order='global')
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                store.put('oom_deferred',key,dict(at=utc(),reason='batch1 memory; resume same checkpoint after memory available'))
                raise
            except InterruptedError: raise
            except Exception as exc:
                store.put('generation_failure',key,dict(arm='H2',task_id=r['task_id'],reason=str(exc),traceback=traceback.format_exc()))
                raise
            result.update(arm='H2',task_id=r['task_id'],base_task_id=r['task_id'].split('/')[2],seed=42,order='global',
                generation_shared_gpu_sec=time.monotonic()-ts,source_commit=head_commit(),
                gru_cell_calls=adapter.gru_calls-before[0],gru_executed_rows=adapter.gru_rows-before[1],gru_active_record_updates=adapter.active_updates-before[2])
            store.put('generated',key,result)
            progress('generation',completed=store.count('generated'),expected=1033,task_id=r['task_id'],last_shared_sec=result['generation_shared_gpu_sec'])
    finally:
        if adapter is not None: adapter.handle.remove()
        if engine is not None: shared.release(engine)
        del engine,candidate,adapter
        gc.collect();torch.cuda.empty_cache()


def reuse_controls(store):
    # The same audited reuse function as H1 checks model files and complete
    # default decoder semantics against A. It writes only this run's audit/store.
    if not shared.controls_reuse(store,run=RUN):
        progress('controls_pending',reason='protocol mismatch; see control_reuse.json; H2 generation proceeds first')


def h1_comparison(store):
    p=PUBLIC/'per_case.jsonl.gz'
    summary=PUBLIC/'training_summary.json'
    if not p.exists() or not summary.exists(): return dict(status='pending',reason='H1 final scored result not published')
    rows=[r for r in read_jsonl(p) if r['arm']=='H1']
    if len(rows)!=1033 or len({r['task_id'] for r in rows})!=1033:
        return dict(status='pending',reason=f'H1 has {len(rows)}/1033 scored lines; H1 generation dtype failures require separate H1 repair')
    h1_plan=json.loads((PUBLIC/'plan.json').read_text());ours=json.loads((RUN/'plan.json').read_text())
    differences=[k for k in ('seed','config','optimizer','lr','betas','eps','weight_decay','clip_norm','effective_batch','epochs','head_dtype','loss','train_states','validation_states','best') if h1_plan[k]!=ours[k]]
    if differences: return dict(status='pending',reason='H1 schedule differences',differences=differences)
    from analysis.analyze_context_foundation import paired_quality
    return dict(status='complete',result=paired_quality(store.rows('scored')+rows,'H2','H1'))


def archive(store,status):
    from analysis.analyze_context_foundation import paired_quality,learning_recovery
    OUT.mkdir(parents=True,exist_ok=True)
    scored=store.rows('scored');generated=store.rows('generated')
    table=[]
    for arm in ('H2','Z4','H4-old','K1'):
        rs=[r for r in scored if r['arm']==arm]
        table.append(dict(arm=arm,N=len(rs),passed=sum(bool(r['passed']) for r in rs),
            rate=sum(bool(r['passed']) for r in rs)/len(rs) if rs else None,missing=1033-len(rs),
            duplicates=len(rs)-len({r['task_id'] for r in rs}),base_groups=len({r['base_task_id'] for r in rs}),
            **{key:sum(r.get(key,0) for r in rs) for key in ('forward_count','actual_submitted_count','head_calls','gru_cell_calls','gru_executed_rows','gru_active_record_updates','expand_count','delete_count','generation_shared_gpu_sec')}))
    comparisons=[paired_quality(scored,'H2',arm) for arm in ('Z4','H4-old','K1')]
    # Reuse A's exact grouped-ratio calculation by relabeling only its input arm.
    triples=[dict(r,arm='H4-old') if r['arm']=='H2' else r for r in scored if r['arm'] in ('H2','Z4','K1')]
    recovery=learning_recovery(triples)
    complete=all(r['N']==1033 and r['duplicates']==0 and r['base_groups']==164 for r in table)
    result=dict(status=status,complete_population=complete,table=table,comparisons=comparisons,pure_learning_recovery=recovery,
        h1=h1_comparison(store),paired_interval='10000 paired cluster bootstrap by 164 base tasks',
        generation_technical_failures=store.rows('generation_failure'),score_technical_failures=store.rows('score_failure'),
        caveat='No same-data retrained Hctx control: H2 improvements alone do not establish multi-token memory causality. Shared GPU times are observations only.')
    atomic_json(OUT/'generation_comparison.json',result)
    write_jsonl(RUN/'generated_rows.jsonl.gz',generated);write_jsonl(RUN/'per_case.jsonl.gz',scored)
    for name in ('data_summary.json','technical_checks.json','plan.json','training_summary.json','curve.json','control_reuse.json','exit.json','environment.json','dependency_wait.json','protocol_differences.json','cpu_checks.txt','pilot_measurement.json','decoder_smoke.json'):
        if (RUN/name).exists(): shutil.copyfile(RUN/name,OUT/name)
    for path in RUN.glob('validation_epoch_*.json'): shutil.copyfile(path,OUT/path.name)
    atomic_json(OUT/'evidence_index.json',dict(code_commit=head_commit(),public_root=str(PUBLIC),run=str(RUN),
        checkpoints=str(RUN/'checkpoints'),raw_generations=str(RUN/'generated_rows.jsonl.gz'),scored=str(RUN/'per_case.jsonl.gz'),
        sqlite=str(RUN/'queue.sqlite'),resources=str(RUN/'resource_observations.jsonl'),log=str(RUN/'pipeline.log'),
        progress=str(RUN/'progress.json'),exit=str(RUN/'exit.json'),shell_exit=str(RUN/'shell_exit.json'),resume=RESUME,
        public_code_base='9aff4553c3ec955b96704e8f404a0cd785a5a272',implementation='B fca5775 heads.py H2-GRU unchanged',
        gpu='single worker, shared cuda:0',incomplete_results_are_not_1033_final=True))
    text=['# H2 上下文＋GRU多词记忆头正式训练',f'状态：{status}；完整1033评测：{complete}。',
        '公共库：H1 formal 20261107（9aff455），原地只读，不采集、不复制。实际规模、源组及32桶见data_summary.json。',
        '实现：B fca5775 的 heads.py 原样；单层GRU宽256、full-memory，每目标重读至多3历史词，h0=0，无跨案例/链/轮缓存。',
        '初始化seed20261107，只有W_out置零，内部随机；不继承任何旧权重/优化器。',
        '共享H1 train.py：FP32完整词表T1 KL(fresh||old+delta)，rank256、强度1，AdamW lr3e-4/betas(.9,.999)/eps1e-8/wd0/clip1，固定有效批128，最多5轮。',
        '顺序default_rng(20261107+epoch)，每轮尾批64仍按实际样本数归一化；精度、损失、teacher、源划分和H1一致。',
        '修复：KeywordHeadAdapter将DreamOn的BF16 old_hidden转FP32，匹配缓存训练；H1工作区未改。',
        '固定best：SingleLine/global/K4/1033行/seed42/T=.2/top_p=.9/batch1，完全复用A采样与结构动作。',
        '|组|通过/N|通过率|缺失|','|---|---:|---:|---:|']
    for r in table: text.append(f"|{r['arm']}|{r['passed']}/{r['N']}|{r['rate']}|{r['missing']}|")
    if (RUN/'training_summary.json').exists():
        t=json.loads((RUN/'training_summary.json').read_text())
        text.append(f"best epoch={t['best_epoch']}，初始权重={t['best_is_untrained_initial']}；last epoch=5；第5轮仍明显改善={t['not_fully_converged_at_epoch5']}，未加轮。曲线见curve.json；完整分桶、源等权、T1 KL/TV、部署TV及固定随机采样变化见validation_epoch_*.json。")
    for r in comparisons: text.append(f"H2−{r['b']}：配对{r['matched']}；救回{r['help']}、改错{r['harm']}，净差{r['net']}；百分点区间{r['ci95']}。")
    text.extend([f'纯学习恢复比例：{recovery}（分母非正不报告）。',f"H1配对：{result['h1']}。",
        '资源计数在generation_comparison.json；GRU cell调用、执行行及有效历史更新分别报告，包含padding计算开销，不宣称增量缓存或公平加速。',
        result['caveat'],f'代码：{head_commit()}；服务器证据：{RUN}；精确路径见evidence_index.json。',
        f'恢复（保留checkpoint及数据游标，不重训已完成阶段）：`{RESUME}`。'])
    (OUT/'report.zh.md').write_text('\n\n'.join(text)+'\n')


def publish():
    paths=['experiments/context_h2_training_v1','analysis_outputs/dreamon_context_h2_training_20261107_v1']
    subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode:
        print('Publishing staged files:',flush=True)
        subprocess.run(['git','diff','--cached','--name-only'],cwd=REPO,check=True)
        subprocess.run(['git','commit','-m','results: archive H2 formal training pipeline'],cwd=REPO,check=True)
    try:
        r=subprocess.run(['git','push','-u','origin','HEAD'],cwd=REPO,timeout=90,capture_output=True,text=True)
        atomic_json(RUN/'publication.json',dict(returncode=r.returncode,stdout=r.stdout[-2000:],stderr=r.stderr[-2000:],local_commit=head_commit()))
    except Exception as exc: atomic_json(RUN/'publication.json',dict(error=str(exc),local_commit=head_commit()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    lock=acquire(RUN/'worker.lock');store=Store(RUN/'queue.sqlite')
    if (RUN/'exit.json').exists() and json.loads((RUN/'exit.json').read_text())['status']=='complete':
        print('H2 pipeline complete; no repeated training or generation',flush=True);return
    status='technical_failure';reason=None;rc=1;bank=None
    stop=threading.Event();observer=None;phase='dependency'
    def interrupted(*_): raise InterruptedError('resume from durable phase/optimizer cursor')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        progress('checking_public_dependency',cuda_initialized=torch.cuda.is_initialized())
        wait_for_public();bank=open_bank()
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1: raise RuntimeError('use the unique visible cuda:0')
        torch.set_num_threads(4)
        atomic_json(RUN/'environment.json',dict(python=sys.version,torch=torch.__version__,code_commit=head_commit(),gpu=subprocess.check_output(['nvidia-smi'],text=True),shared_gpu=True,gpu_workers=1))
        def observe():
            while not stop.wait(30):
                try:
                    value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                    with (RUN/'resource_observations.jsonl').open('a') as f: f.write(json.dumps(dict(at=utc(),observation=value))+'\n')
                except Exception: traceback.print_exc()
        observer=threading.Thread(target=observe,daemon=True);observer.start()
        phase='technical'
        if not (RUN/'technical_checks.json').exists(): technical(bank)
        if args.check_only:
            progress('check_only_complete');return
        phase='training'
        free,total=torch.cuda.mem_get_info()
        if free<=4*2**30: raise torch.cuda.OutOfMemoryError('insufficient shared memory after 4 GiB reserve')
        torch.cuda.set_per_process_memory_fraction(min(.90,(free-4*2**30)/total))
        progress('training_start',free_bytes=free,reserve_bytes=4*2**30,dreamon_loaded=False)
        train(store,run=RUN,head_name='H2-GRU',bank=bank,progress_fn=progress)
        gc.collect();torch.cuda.empty_cache()
        phase='evaluation';reuse_controls(store);generate(store,bank)
        shared.score(store,progress_fn=progress)
        counts={a:sum(r['arm']==a for r in store.rows('scored')) for a in ('H2','Z4','H4-old','K1')}
        status='complete' if all(v==1033 for v in counts.values()) else 'partial_technical_failures'
        rc=0 if status=='complete' else 2
    except torch.cuda.OutOfMemoryError as exc:
        status='training_complete_evaluation_blocked' if (RUN/'training_summary.json').exists() else 'blocked_shared_gpu_memory';reason=str(exc);traceback.print_exc()
    except InterruptedError as exc: status='interrupted_resumable';reason=str(exc);traceback.print_exc()
    except Exception as exc:
        status='blocked_public_dependency' if phase=='dependency' else 'blocked_storage_or_io' if isinstance(exc,OSError) else 'technical_failure'
        reason=str(exc);traceback.print_exc()
    finally:
        stop.set()
        if observer: observer.join(timeout=2)
        if not args.check_only:
            atomic_json(RUN/'exit.json',dict(status=status,reason=reason,phase=phase,exit_code=rc,at=utc(),resume=RESUME,training_complete=(RUN/'training_summary.json').exists()))
            try: archive(store,status);publish()
            except Exception: traceback.print_exc()
        lock.close()
    sys.exit(rc)


if __name__=='__main__': main()
