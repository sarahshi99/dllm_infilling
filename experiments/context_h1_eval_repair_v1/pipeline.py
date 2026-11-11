"""Evaluation-only repair. Never calls train or mutates original H1 artifacts."""
import json
import shutil
import signal
import subprocess
import sys
import traceback
import torch
from experiments.context_foundation.common import REPO,Store,acquire,atomic_json,write_jsonl,read_jsonl,utc,head_commit
from experiments.context_h1_training_v1 import pipeline as shared
from experiments.context_h2_training_v1.dependency import ReadOnlyStore
from .pipeline_state import RUN,SOURCE,OUT,RESUME,progress
from .generate import generate


def prepare(store):
    summary=json.loads((SOURCE/'training_summary.json').read_text())
    assert summary['status']=='complete' and summary['best_epoch']==5
    source=ReadOnlyStore(SOURCE/'queue.sqlite')
    good=source.rows('generated');failures=source.rows('generation_failure')
    assert len(good)==6 and len(failures)==1027
    assert all('BFloat16 and Float' in r['reason'] for r in failures)
    for row in good:
        assert row['arm']=='H1' and row['head_calls']==0
        key='H1::'+row['task_id']
        if not store.get('generated',key):
            store.put('generated',key,dict(row,reused_from=str(SOURCE),reuse_reason='successful unchanged no-head-call path'))
    source.db.close()
    if not shared.controls_reuse(store,run=RUN):
        progress('controls_pending',reason='A baseline protocol audit did not match')
    atomic_json(RUN/'repair_provenance.json',dict(source_run=str(SOURCE),checkpoint=summary['best'],best_epoch=5,
        original_successes_reused=6,original_dtype_failures=1027,old_results_preserved=True,training_started=False,
        correction='published H2 KeywordHeadAdapter promotes cached/live old_hidden to FP32',
        protocol='A SingleLine/global/K4/1033/seed42/T=.2/top_p=.9/batch1; no semantic changes',
        model_memory_reserve_gib=4,checkpoint_unchanged=True))


def summarize(store,status):
    from analysis.analyze_context_foundation import paired_quality,learning_recovery
    OUT.mkdir(parents=True,exist_ok=True)
    scored=store.rows('scored');generated=store.rows('generated')
    own=[r for r in scored if r['arm']=='H1']
    expected={r['task_id'] for r in read_jsonl(REPO/'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz')}
    generated_ids=[r['task_id'] for r in generated]
    scored_ids=[r['task_id'] for r in own]
    complete=set(scored_ids)==expected and len(scored_ids)==1033 and len({r['base_task_id'] for r in own})==164
    table=[]
    peer_root=RUN.parent/'context_h2_training_20261107_v1'
    peer=[r for r in read_jsonl(peer_root/'per_case.jsonl.gz') if r['arm']=='H2']
    a=json.loads((SOURCE/'plan.json').read_text());b=json.loads((peer_root/'plan.json').read_text())
    common=('seed','config','optimizer','lr','betas','eps','weight_decay','clip_norm','effective_batch','epochs','head_dtype','loss','train_states','validation_states','best')
    assert all(a[k]==b[k] for k in common)
    assert json.loads((peer_root/'control_reuse.json').read_text())['reused']
    scored+=peer
    for arm in ('H1','H2','Z4','H4-old','K1'):
        rows=[r for r in scored if r['arm']==arm]
        passed=sum(bool(r['passed']) for r in rows)
        table.append(dict(arm=arm,N=len(rows),passed=passed,rate=passed/len(rows) if rows else None,
            **{k:sum(r.get(k,0) for r in rows) for k in ('forward_count','actual_submitted_count','head_calls','expand_count','delete_count','generation_shared_gpu_sec')}))
    comparisons=[paired_quality(scored,'H1',arm) for arm in ('Z4','H4-old','K1','H2')]
    triples=[dict(r,arm='H4-old') if r['arm']=='H1' else r for r in scored if r['arm'] in ('H1','Z4','K1')]
    result=dict(status=status,complete_population=complete,table=table,comparisons=comparisons,pure_learning_recovery=learning_recovery(triples),
        paired_interval='10000 paired bootstrap replicates grouped by 164 base tasks',
        missing_generated=sorted(expected-set(generated_ids)),missing_scored=sorted(expected-set(scored_ids)),
        duplicate_generated=len(generated_ids)-len(set(generated_ids)),duplicate_scored=len(scored_ids)-len(set(scored_ids)),
        generation_technical_failures=store.rows('generation_failure'),score_technical_failures=store.rows('score_failure'))
    atomic_json(OUT/'generation_comparison.json',result)
    write_jsonl(RUN/'generated_rows.jsonl.gz',generated);write_jsonl(RUN/'per_case.jsonl.gz',store.rows('scored'))
    for name in ('repair_provenance.json','exit.json','environment.json','control_reuse.json','cpu_checks.txt'):
        if (RUN/name).exists(): shutil.copyfile(RUN/name,OUT/name)
    atomic_json(OUT/'evidence_index.json',dict(code_commit=head_commit(),checkpoint=str(SOURCE/'checkpoints/best.pt'),
        source_training=str(SOURCE/'training_summary.json'),run=str(RUN),raw_generations=str(RUN/'generated_rows.jsonl.gz'),
        per_case=str(RUN/'per_case.jsonl.gz'),sqlite=str(RUN/'queue.sqlite'),log=str(RUN/'pipeline.log'),resume=RESUME))
    text=['# H1正式best完整评测补跑',f'状态：{status}；1033行/164题完整：{complete}。',
        '复用H1 epoch5正式best，不训练、不修改权重。修复BF16→FP32接口，原1027条技术失败重跑，6条成功路径复用；原始失败记录和输出完整保留。',
        'SingleLine/global/K4，seed42，T=.2，top_p=.9，batch1；A公共基线3099条直接复用，模型、随机输入、FP32采样、选位、结构屏障和官方评分不变。',
        '|组|通过/N|通过率|','|---|---:|---:|']
    for row in table: text.append(f"|{row['arm']}|{row['passed']}/{row['N']}|{100*row['rate']:.2f}%|" if row['N'] else f"|{row['arm']}|0/0|待补|")
    text.append('')
    for row in comparisons: text.append(f"H1−{row['b']}：救回{row['help']}、改错{row['harm']}、净差{row['net']}；百分点区间{row['ci95']}。")
    text.extend(['',f"纯学习恢复比例：{result['pure_learning_recovery']}。",'配对区间以164基础题分组计算；共享GPU耗时仅作观测。',
        'H2最佳验证KL从0.29015降至0.11201（下降61.4%），部署采样与teacher不一致率从7.63%降至4.585%；但1033行程序通过率仅82.38%。KL是预测分布目标，不保证程序通过。H2相对零头救回56/改错25，净增31；相对旧静态头救回38/改错29，净增9，区间跨0；相对K1救回8/改错106，净少98，恢复比例24.0%。现有证据只能说明有限学习收益，不能确定缺口成因，不能宣称多词信息因果有效。',
        f'证据与恢复：`{RESUME}`；日志 `{RUN}/pipeline.log`；逐行结果 `{RUN}/per_case.jsonl.gz`。'])
    (OUT/'report.zh.md').write_text('\n'.join(text)+'\n')
    return complete


def publish():
    paths=['experiments/context_h1_eval_repair_v1','analysis_outputs/dreamon_context_h1_eval_repair_20261111_v1']
    subprocess.run(['git','add','--',*paths],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:
        subprocess.run(['git','diff','--cached','--name-only'],check=True)
        subprocess.run(['git','commit','-m','results: archive fixed-checkpoint H1 evaluation repair'],check=True)
    try:
        result=subprocess.run(['git','push','-u','origin','HEAD'],capture_output=True,text=True,timeout=90)
        atomic_json(RUN/'publication.json',dict(returncode=result.returncode,stdout=result.stdout[-2000:],stderr=result.stderr[-2000:],commit=head_commit()))
    except Exception as exc: atomic_json(RUN/'publication.json',dict(error=str(exc),commit=head_commit()))


def main():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    lock=acquire(RUN/'worker.lock');store=Store(RUN/'queue.sqlite')
    if (RUN/'exit.json').exists() and json.loads((RUN/'exit.json').read_text())['status']=='complete': return
    status='technical_failure';reason=None;rc=1;phase='prepare'
    def interrupt(*_): raise InterruptedError('resume from durable generation/score records')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        progress('prepare',training_started=False)
        torch.set_num_threads(4)
        prepare(store)
        atomic_json(RUN/'environment.json',dict(code_commit=head_commit(),gpu=subprocess.check_output(['nvidia-smi'],text=True),shared_gpu=True,gpu_workers=1))
        phase='generation';generate(store)
        phase='scoring';progress('scoring');shared.score(store,progress_fn=progress)
        n=sum(r['arm']=='H1' for r in store.rows('scored'))
        status='complete' if n==1033 else 'partial_technical_failures';rc=0 if n==1033 else 2
    except torch.cuda.OutOfMemoryError as exc: status='blocked_evaluation_memory';reason=str(exc);traceback.print_exc()
    except InterruptedError as exc: status='interrupted_resumable';reason=str(exc);traceback.print_exc()
    except Exception as exc: reason=str(exc);traceback.print_exc()
    finally:
        atomic_json(RUN/'exit.json',dict(status=status,phase=phase,reason=reason,at=utc(),exit_code=rc,resume=RESUME,training_started=False))
        try:
            complete=summarize(store,status)
            if complete:
                atomic_json(SOURCE/'evaluation_supplement_20261111_v1.json',dict(status='complete',run=str(RUN),per_case=str(RUN/'per_case.jsonl.gz'),report=str(OUT/'report.zh.md'),original_artifacts_preserved=True))
            publish()
        except Exception:
            traceback.print_exc();rc=1
        lock.close()
    sys.exit(rc)


if __name__=='__main__': main()
