"""Official sequential CPU scoring, grouped paired statistics and audit/report."""
import argparse
import collections
import csv
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
import numpy as np

from experiments.threshold_eval.common import *


def write_csv(name,rows):
    tmp=OUT/(name+'.tmp')
    with tmp.open('w') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    tmp.replace(OUT/name)


def score():
    from experiments.k4_batch_common import population
    from experiments.run_k4_multiline_stage import official_evaluate
    contexts={r['task_id']:r for r in population(spec().get('dataset','single-line'))}
    for role in ('nohead','trained'):
        db=Ledger(role)
        for r in db.rows('quality'):
            c,key=r['config'],r['task_id']
            if db.has('score',c,key):continue
            # Up to 3 infrastructure retries in addition to the first attempt.
            for attempt in range(4):
                try:
                    value=official_evaluate(contexts[key],r['output'])
                    value.update(task_id=key,config=c,execution_commit=r['execution_commit'],attempt=attempt+1)
                    db.put('score',c,key,value)
                    break
                except Exception:
                    db.attempt('score_infrastructure',f'{c}::{key}::{attempt+1}',traceback.format_exc())
                    if attempt==3:
                        set_score_status('blocked_scoring',role,db,error=traceback.format_exc());raise
            if db.count('score')%25==0:set_score_status('scoring',role,db)
        set_score_status('role_scored',role,db)
        db.db.close()


def set_score_status(state,role,db,**kw):
    value=dict(status=state,role=role,completed=db.count('score'),generated=db.count('quality'),at=utc(),**kw)
    atomic_json(SHARED/'scoring.status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def bounds(a):
    return [float(x) for x in np.quantile(a,[.025,.975])]


_bootstrap_weights={}


def grouped(values,keys,weights=None):
    groups=sorted({k.split('/')[2] for k in keys},key=int)
    if not groups:return dict(line_mean=None,task_mean=None,line_ci95=None,task_ci95=None)
    index={g:i for i,g in enumerate(groups)};n=len(groups)
    sums=np.zeros(n);counts=np.zeros(n)
    for value,key in zip(values,keys):
        i=index[key.split('/')[2]];sums[i]+=value;counts[i]+=1
    if weights is None or weights.shape[1]!=n:
        if n not in _bootstrap_weights:
            _bootstrap_weights[n]=np.random.default_rng(20261104).multinomial(n,np.full(n,1/n),size=10000)
        weights=_bootstrap_weights[n]
    line=(weights@sums)/(weights@counts)
    task=weights@(sums/counts)/n
    return dict(line_mean=float(sums.sum()/counts.sum()),task_mean=float(np.mean(sums/counts)),line_ci95=bounds(line),task_ci95=bounds(task))


def audit(quality,timing):
    sp=spec();ready_path=SHARED/'READY.json'
    version=json.loads(ready_path.read_text())['execution_commit'] if ready_path.exists() else None
    expected={(c['id'],key) for c in sp['configs'] for key in sp['cases']}
    found=[(r['config'],r['task_id']) for r in quality]
    timed_expected={(c['id'],key,rep) for c in sp['configs'] for key in sp['timing_cases'] for rep in range(3)}
    timed_found=[(r['config'],r['task_id'],r['repetition']) for r in timing]
    versions=sorted({r['execution_commit'] for r in quality+timing})
    quality_bykey={(r['config'],r['task_id']):r for r in quality}
    invalid=[]
    configuration={c['id']:c for c in sp['configs']}
    for r in quality:
        if not isinstance(r.get('completion_token_ids'),list) or r.get('forward_count',0)<=0:invalid.append([r['config'],r['task_id'],'generation_schema'])
        if 'passed' in r and type(r['passed']) is not bool:invalid.append([r['config'],r['task_id'],'score_schema'])
        if r['method']=='nohead' and r['head_calls']!=0:invalid.append([r['config'],r['task_id'],'nohead_called_head'])
        hist={int(k):v for k,v in r['normal_submissions_histogram'].items()}
        if sum(hist.values())!=r['forward_count'] or sum(k*v for k,v in hist.items())!=r['normal_token_count']:
            invalid.append([r['config'],r['task_id'],'histogram'])
        c=configuration.get(r['config'])
        if c and (r['method']!=c['method']):invalid.append([r['config'],r['task_id'],'role_config_mismatch'])
        st=r['statistics']
        if st['extra_ordinary_attempted']!=st['extra_ordinary_accepted']+st['extra_ordinary_rejected']:
            invalid.append([r['config'],r['task_id'],'threshold_accounting'])
    timing_divergences=[]
    for r in timing:
        if not isinstance(r.get('completion_token_ids'),list) or r.get('generation_sec',0)<=0:
            invalid.append([r['config'],r['task_id'],'timing_schema'])
        q=quality_bykey.get((r['config'],r['task_id']))
        if q and r['completion_token_ids']!=q['completion_token_ids']:
            timing_divergences.append([r['config'],r['task_id'],r['repetition']])
    result=dict(quality_expected=len(expected),quality_actual=len(quality),scored=sum('passed' in r for r in quality),
        quality_unique=len(set(found)),quality_missing_count=len(expected-set(found)),quality_missing=sorted(expected-set(found))[:100],quality_unexpected=sorted(set(found)-expected),
        quality_duplicates=len(found)-len(set(found)),timing_expected=len(timed_expected),timing_actual=len(timing),timing_unique=len(set(timed_found)),
        timing_missing_count=len(timed_expected-set(timed_found)),timing_missing=sorted(timed_expected-set(timed_found))[:100],timing_unexpected=sorted(set(timed_found)-timed_expected),timing_duplicates=len(timed_found)-len(set(timed_found)),
        execution_versions=versions,expected_execution_version=version,version_mixing=bool(versions and versions!=[version]),invalid=invalid,
        timing_output_divergences=timing_divergences,timing_population_separate=True,sealed_external_test_opened=False,
        per_config={c['id']:dict(generated=sum(r['config']==c['id'] for r in quality),scored=sum(r['config']==c['id'] and 'passed' in r for r in quality),timed=sum(r['config']==c['id'] for r in timing)) for c in sp['configs']})
    result['complete']=bool(len(set(found))==sp.get('expected_quality',35122) and result['scored']==sp.get('expected_quality',35122) and len(set(timed_found))==sp.get('expected_timing',16728) and not any((result['quality_duplicates'],result['timing_duplicates'],result['version_mixing'],invalid,result['quality_unexpected'],result['timing_unexpected'],timing_divergences)))
    result['quality_by_role']={role:sum(r['method']==role for r in quality) for role in ('nohead','trained')}
    result['crossline_quality_actual']=sum(case_metadata(r['task_id'])['crossline'] for r in quality)
    result['crossline_timing_actual']=sum(case_metadata(r['task_id'])['crossline'] for r in timing)
    result['population']=sp.get('population')
    result['missing_list_note']='First 100 missing keys shown; full missing set reproducible from run_spec configs/cases/timing_cases minus per_case keys.'
    for c in sp['configs']:
        cid=c['id'];q=[r for r in quality if r['config']==cid];t=[r for r in timing if r['config']==cid]
        result['per_config'][cid].update(crossline_generated=sum(case_metadata(r['task_id'])['crossline'] for r in q),
            crossline_timed=sum(case_metadata(r['task_id'])['crossline'] for r in t),
            span_counts={b:sum(case_metadata(r['task_id'])['span_bucket']==b for r in q) for b in ('1','2-3','4-7','8+')})
    atomic_json(OUT/'integrity_audit.json',result)
    return result


def load_results():
    quality=[]
    for role in ('nohead','trained'):
        db=Ledger(role);scored={(r['config'],r['task_id']):r for r in db.rows('score')}
        for r in db.rows('quality'):
            value=scored.get((r['config'],r['task_id']),{})
            if value:
                assert value['execution_commit']==r['execution_commit'],'score/generation version mismatch'
            quality.append(dict(r,**value))
        db.db.close()
    db=Ledger('timing');timing=db.rows('timing');db.db.close()
    return quality,timing


def gpu_contention():
    own=set()
    for p in SHARED.glob('model_*.json'):
        own.add(json.loads(p.read_text())['pid'])
    external=set();observations=0
    path=SHARED/'timing.gpu.jsonl'
    if path.exists():
        for r in read_jsonl(path):
            observations+=1
            for line in r.get('compute_processes',[]):
                try:pid=int(line.split(',')[0])
                except ValueError:continue
                if pid not in own:external.add(pid)
    return dict(shared_environment=True,external_pids=sorted(external),timing_observations=observations,
        limitation='Nonexclusive H200; 30s load snapshots do not eliminate short external contention. Exploratory speed flag cannot establish definitive attainment.')


def population_summary(quality,timing,case_keys,timing_keys,label):
    expected=len(case_keys);timing_expected=len(timing_keys)*3
    qset=set(case_keys);tset=set(timing_keys)
    quality=[r for r in quality if r['task_id'] in qset]
    timing=[r for r in timing if r['task_id'] in tset]
    indexed={c['id']:{r['task_id']:r for r in quality if r['config']==c['id'] and 'passed' in r} for c in configs()}
    summary=[];timed=[];comparisons=[];histograms={}
    mean=lambda a:statistics.mean(a) if a else None
    for c in configs():
        cid=c['id'];rows=[r for r in quality if r['config']==cid];scored=list(indexed[cid].values())
        q=grouped([100*int(r['passed']) for r in scored],[r['task_id'] for r in scored])
        nfe=sum(r['forward_count'] for r in rows);normal=sum(r['normal_token_count'] for r in rows)
        st=collections.Counter();hist=collections.Counter()
        for r in rows:
            st.update(r['statistics']);hist.update({int(k):v for k,v in r['normal_submissions_histogram'].items()})
        histograms[cid]=dict(sorted(hist.items()))
        tr=[r for r in timing if r['config']==cid];secs=[r['generation_sec'] for r in tr]
        t=dict(config=cid,population=label,n=len(tr),expected=timing_expected,base_tasks=len(tset),
            mean_sec=mean(secs),median_sec=statistics.median(secs) if secs else None,
            p90_sec=float(np.quantile(secs,.9)) if secs else None,total_sec=sum(secs),shared_environment=True)
        timed.append(t)
        attempted=st['extra_ordinary_attempted'];evaluated=st['evaluated_candidates']
        success=sum(r['passed'] for r in scored)
        # Partial scored means are explicitly auxiliary; official denominator rate only when all scored.
        rate=100*success/expected if len(scored)==expected and expected else None
        s=dict(config=cid,population=label,base_tasks=len({k.split('/')[2] for k in case_keys}),order=c['order'],k=c['k'],method=c['method'],threshold='off' if c['threshold'] is None else c['threshold'],
            generated=len(rows),scored=len(scored),expected=expected,success=success,line_pass_pct=rate,
            observed_scored_pass_pct=q['line_mean'],task_equal_pass_pct=q['task_mean'],line_ci95=json.dumps(q['line_ci95']),task_ci95=json.dumps(q['task_ci95']),
            forward_count=nfe,normal_commits=normal,normal_commits_per_forward=normal/nfe if nfe else None,
            extra_attempted=attempted,extra_accepted=st['extra_ordinary_accepted'],extra_rejected=st['extra_ordinary_rejected'],
            extra_accept_rate=st['extra_ordinary_accepted']/attempted if attempted else None,
            extra_reject_rate=st['extra_ordinary_rejected']/attempted if attempted else None,
            selected_candidates=st['selected_candidates'],chain_skipped=st['chain_skipped'],head_calls=sum(r['head_calls'] for r in rows),
            expand_count=sum(r['expand_count'] for r in rows),delete_count=sum(r['delete_count'] for r in rows),
            base_structure_rounds=st['base_structure_rounds'],corrected_structure_rounds=st['corrected_structure_rounds'],
            structure_proportion=(st['base_structure_rounds']+st['corrected_structure_rounds'])/evaluated if evaluated else None,
            noop_count=st['mask_noop'],noop_proportion=st['mask_noop']/evaluated if evaluated else None,
            mean_generated_length=mean([len(r['completion_token_ids']) for r in rows]),
            remaining_masks=sum(r['final_remaining_masks'] for r in rows),step_budget_exhausted=sum(r.get('stop_reason')=='step_budget' for r in rows),
            unfinished_rate=mean([r['final_remaining_masks']>0 for r in rows]),
            observed_quality_mean_sec=mean([r['generation_sec'] for r in rows]),
            timing_n=len(tr),timing_expected=timing_expected,timing_mean_sec=t['mean_sec'],timing_median_sec=t['median_sec'],timing_p90_sec=t['p90_sec'],timing_total_sec=t['total_sec'],
            peak_allocated_bytes=max([r['peak_memory_allocated_bytes'] for r in rows+tr],default=0),peak_reserved_bytes=max([r['peak_memory_reserved_bytes'] for r in rows+tr],default=0))
        summary.append(s)
    for c in configs():
        cid=c['id'];base=f"{c['order']}_k1_nohead_off";off=f"{c['order']}_k{c['k']}_{c['method']}_off"
        other=cid.replace('_nohead_','_trained_') if c['method']=='nohead' else cid.replace('_trained_','_nohead_')
        for relation,bid in [('vs_k1',base),('vs_same_method_off',off),('vs_other_method_same_threshold',other)]:
            if bid==cid or bid not in indexed:continue
            a,b=indexed[cid],indexed[bid];keys=sorted(a.keys()&b.keys())
            q=grouped([100*(int(a[k]['passed'])-int(b[k]['passed'])) for k in keys],keys)
            rescued=sum(a[k]['passed'] and not b[k]['passed'] for k in keys);harmed=sum(not a[k]['passed'] and b[k]['passed'] for k in keys)
            comparisons.append(dict(config=cid,population=label,reference=bid,relation=relation,expected=expected,matched=len(keys),
                base_tasks=len({k.split('/')[2] for k in keys}),rescued=rescued,harmed=harmed,net=rescued-harmed,
                line_delta_pp=q['line_mean'],task_equal_delta_pp=q['task_mean'],line_ci95=json.dumps(q['line_ci95']),task_ci95=json.dumps(q['task_ci95']),
                bootstrap_unit='present_base_tasks',bootstrap_repetitions=10000,bootstrap_seed=20261104))
    smap={s['config']:s for s in summary};cmap={(r['config'],r['relation']):r for r in comparisons}
    for s,t in zip(summary,timed):
        base=smap[f"{s['order']}_k1_nohead_off"]
        for relation in ('vs_k1','vs_same_method_off','vs_other_method_same_threshold'):
            pair=cmap.get((s['config'],relation))
            s[relation+'_line_delta_pp']=pair['line_delta_pp'] if pair else (0 if relation!='vs_other_method_same_threshold' else None)
            s[relation+'_task_delta_pp']=pair['task_equal_delta_pp'] if pair else (0 if relation!='vs_other_method_same_threshold' else None)
        fulltime=timing_expected>0 and s['timing_n']==base['timing_n']==timing_expected
        ratio=s['timing_total_sec']/base['timing_total_sec'] if fulltime and base['timing_total_sec'] else None
        s['timing_total_ratio_vs_k1']=ratio;t['total_ratio_vs_k1']=ratio
        for field,key in [('exploratory_engineering_flag','task'),('exploratory_case_engineering_flag','line')]:
            delta=s['vs_k1_'+key+'_delta_pp']
            s[field]=bool(s['scored']==expected and base['scored']==expected and delta is not None and delta>=-1 and ratio is not None and ratio<=.8)
        s['definitive_engineering_attainment']='undetermined_shared_environment'
    nearest=[]
    for s in summary:
        candidates=[b for b in summary if b['order']==s['order'] and b['method']!=s['method'] and b['line_pass_pct'] is not None]
        if s['line_pass_pct'] is None or not candidates:continue
        b=min(candidates,key=lambda b:(abs(s['line_pass_pct']-b['line_pass_pct']),b['config']))
        nearest.append(dict(config=s['config'],population=label,other=b['config'],case_quality_gap_pp=s['line_pass_pct']-b['line_pass_pct'],
            timing_ratio=s['timing_mean_sec']/b['timing_mean_sec'] if s['timing_n']==b['timing_n']==timing_expected and b['timing_mean_sec'] else None,
            selection='nearest measured case pass rate, descriptive posthoc; confidence scales not assumed calibrated'))
    return summary,timed,comparisons,nearest,histograms


def analyze(partial=False):
    OUT.mkdir(parents=True,exist_ok=True)
    quality,timing=load_results();integrity=audit(quality,timing);sp=spec()
    for name,rows in [('per_case.jsonl.gz',quality),('timing_per_case.jsonl.gz',timing)]:
        tmp=OUT/(name+'.tmp')
        with gzip.open(tmp,'wt') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        tmp.replace(OUT/name)
    scopes=[('all',sp['cases'],sp['timing_cases'])]
    if sp.get('dataset','single-line')=='multi-line':
        scopes += [('crossline',[k for k in sp['cases'] if case_metadata(k)['crossline']],
                    [k for k in sp['timing_cases'] if case_metadata(k)['crossline']])]
        scopes += [('span_'+b,[k for k in sp['cases'] if case_metadata(k)['span_bucket']==b],
                    [k for k in sp['timing_cases'] if case_metadata(k)['span_bucket']==b]) for b in ('1','2-3','4-7','8+')]
    all_summary=[];all_timed=[];all_pairs=[];all_nearest=[];histograms={}
    for label,keys,tkeys in scopes:
        summary,timed,pairs,nearest,hist=population_summary(quality,timing,keys,tkeys,label)
        all_summary.extend(summary);all_timed.extend(timed);all_pairs.extend(pairs);all_nearest.extend(nearest);histograms[label]=hist
        if label in ('all','crossline'):
            write_csv('summary.csv' if label=='all' else 'crossline_summary.csv',summary)
            charts(summary,label)
    write_csv('span_summary.csv',[s for s in all_summary if s['population'].startswith('span_')])
    write_csv('paired_comparisons.csv',all_pairs);write_csv('timing.csv',all_timed);write_csv('nearest_quality.csv',all_nearest)
    atomic_json(OUT/'submission_distributions.json',histograms);atomic_json(OUT/'timing_environment.json',gpu_contention())
    attempts=[]
    for role in ('nohead','trained','timing'):
        db=Ledger(role)
        attempts.extend(dict(role=role,at=r[0],phase=r[1],key=r[2],error=r[3]) for r in db.db.execute('SELECT at,phase,key,error FROM attempts ORDER BY id'))
        db.db.close()
    atomic_json(OUT/'attempts.json',attempts)
    report(all_summary,all_pairs,integrity,all_nearest)
    state='completed' if integrity['complete'] else 'incomplete'
    atomic_json(SHARED/'status.json',dict(task_id=TASK,status=state,quality_completed=len(quality),quality_scored=sum('passed' in r for r in quality),timing_completed=len(timing),at=utc(),integrity=str(OUT/'integrity_audit.json')))
    atomic_json(OUT/'status.json',json.loads((SHARED/'status.json').read_text()))
    (OUT/'commands.txt').write_text((SHARED/'commands.txt').read_text())
    update_checkpoint(integrity,state)
    if not partial and not integrity['complete']:raise RuntimeError('Full completion audit failed; see integrity_audit.json')


def charts(summary,label='all'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for axis,name in [('timing_mean_sec','quality_time'),('normal_commits_per_forward','quality_parallelism')]:
        fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,order in zip(axs,('global','left')):
            for method,marker in [('nohead','o'),('trained','s')]:
                for k,color in [(2,'#2878b5'),(4,'#d95f02')]:
                    rows=[s for s in summary if s['order']==order and s['method']==method and s['k']==k and s[axis] is not None and s['line_pass_pct'] is not None]
                    rows.sort(key=lambda r:(r['threshold']!='off',str(r['threshold'])))
                    ax.plot([r[axis] for r in rows],[r['line_pass_pct'] for r in rows],marker=marker,color=color,linestyle='-' if method=='nohead' else '--',label=f'{method} K{k}')
                    for r in rows:ax.annotate(str(r['threshold']),(r[axis],r['line_pass_pct']),fontsize=7)
            b=next(s for s in summary if s['order']==order and s['k']==1)
            if b[axis] is not None and b['line_pass_pct'] is not None:ax.scatter([b[axis]],[b['line_pass_pct']],c='black',marker='*',s=90,label='K1')
            ax.set_title(order+' / '+label);ax.set_xlabel('Mean generation seconds (shared GPU)' if axis=='timing_mean_sec' else 'Ordinary commits / forward');ax.grid(alpha=.25);ax.legend(fontsize=8)
        axs[0].set_ylabel('Case pass rate (%)');fig.tight_layout();fig.savefig(OUT/(label+'_'+name+'.png'),dpi=180);plt.close(fig)


def report(summary,comparisons,integrity,nearest):
    def fmt(v):return '未完成' if v is None else f'{v:.2f}'
    runtime={}
    for role in ('nohead','trained','timing'):
        path=SHARED/f'{role}.status.json'
        if path.exists():runtime[role]=json.loads(path.read_text())
    atomic_json(OUT/'execution_status.json',runtime)
    lines=[f'# {TASK} 官方阈值评测','',
        f"状态：{'完整完成' if integrity['complete'] else '未完成，不能作完整实验结论'}。质量生成 {integrity['quality_actual']}/{integrity.get('quality_expected',35122)}，评分 {integrity['scored']}/{integrity.get('quality_expected',35122)}；计时 {integrity['timing_actual']}/{integrity['timing_expected']}。",
        '',f"实际生成版本 `{integrity['expected_execution_version']}`。34配置，DreamOn revision8ccc747，bf16；旧rank256 v2 KL/full/best头float32、λ=1。seed42、T=.2、top-p=.9、batch1、entropy/alg_temp0、初始mask64、steps256、原扩删/EOS规则；不复用旧single-line生成或评分。",
        '', '阈值使用实际抽到的普通词在合法完整词表T=1 FP32概率；全轮首个空间候选唯一豁免，拒绝不重抽、不写前驱并停止当前连续链。共同随机键沿用原contract/seed/真实case/round/original_position/purpose。']
    for role,state in runtime.items():
        lines.append(f"角色{role}：{state.get('status')}，{state.get('completed',0)}/{state.get('expected','?')}，PID {state.get('pid')}，UTC {state.get('at')}。")
        if state.get('error'):lines.append('具体阻塞：`'+state['error'].splitlines()[-1][:1200]+'`')
    labels=[('all','完整官方人口'),('crossline','真正跨行人口（至少2行）')]
    for label,title in labels:
        rows=[s for s in summary if s['population']==label]
        if not rows:continue
        n=rows[0]['expected'];groups=rows[0]['base_tasks']
        lines+=['',f'## {title}：{n}例 / {groups}基础题','',
            '|配置|成功数/分母|逐例通过率%|相对K1 pp|计时均值秒|中位秒|p90秒|时间比|普通词/NFE|额外接受率|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            accept='N/A' if r['k']==1 else fmt(r['extra_accept_rate'])
            lines.append(f"|{r['config']}|{r['success']}/{n}（已评分{r['scored']}）|{fmt(r['line_pass_pct'])}|{fmt(r['vs_k1_line_delta_pp'])}|{fmt(r['timing_mean_sec'])}|{fmt(r['timing_median_sec'])}|{fmt(r['timing_p90_sec'])}|{fmt(r['timing_total_ratio_vs_k1'])}|{fmt(r['normal_commits_per_forward'])}|{accept}|")
        lines+=['',f"时间来自该人口固定子集，每配置预期{rows[0]['timing_expected']}次，不能称为全{n}例时延。跨行使用相同127题/相同案例K1配对；全计时包含37题单行回退。质量不完整时逐例率留空，成功数不得解释为全量结果。"]
        for order in ('global','left'):
            for k in (2,4):
                candidates=[r for r in rows if r['order']==order and r['k']==k and r['method']=='nohead' and r['threshold']!='off' and r['line_pass_pct'] is not None]
                off=next(r for r in rows if r['config']==f'{order}_k{k}_nohead_off')
                if candidates and off['line_pass_pct'] is not None:
                    best=max(candidates,key=lambda r:r['line_pass_pct'])
                    lines.append(f"- {order} K{k}无头已测阈值最高点{best['threshold']}：{best['success']}/{n}；较off {best['line_pass_pct']-off['line_pass_pct']:+.2f}pp，普通词/NFE {fmt(best['normal_commits_per_forward'])}。仅本人口内事后描述，所有负结果保留。")
        for p in comparisons:
            if p['population']==label and p['relation']=='vs_other_method_same_threshold' and '_trained_' in p['config'] and p['matched']==n:
                lines.append(f"- {p['config']}相对无头同阈值：救回{p['rescued']}、改错{p['harmed']}、净{p['net']:+d}；逐例差{fmt(p['line_delta_pp'])}pp，整题bootstrap95%区间{p['line_ci95']}。")
        head_pairs=[p for p in comparisons if p['population']==label and p['relation']=='vs_other_method_same_threshold' and '_trained_' in p['config'] and p['matched']==n]
        if head_pairs:
            positive=sum(p['net']>0 for p in head_pairs);negative=sum(p['net']<0 for p in head_pairs)
            lines.append(f"头的额外价值：同阈值配对共{len(head_pairs)}点，净正向{positive}点、净负向{negative}点、持平{len(head_pairs)-positive-negative}点；逐点区间与时间开销见表和CSV，不把任一最好点外推到全部跨度。")
        else:lines.append('本人口尚无完整同阈值配对，不能判断无头阈值有效性或头的额外价值。')
        for field,title_flag in [('exploratory_engineering_flag','历史题等权'),('exploratory_case_engineering_flag','逐例')]:
            flags=[r['config'] for r in rows if r[field]]
            lines.append(f"{title_flag}预设标记（下降≤1pp且减时≥20%）：{', '.join(flags) or '暂无完整实测点满足'}。共享争用下不作确定达标判断。")
    span=[s for s in summary if s['population'].startswith('span_')]
    if span:
        lines+=['','## 原始孔洞跨度分层','','|配置|跨度|基础题|成功数/分母|已评分|通过率%|','|---|---|---:|---:|---:|---:|']
        for r in span:lines.append(f"|{r['config']}|{r['population'][5:]}|{r['base_tasks']}|{r['success']}/{r['expected']}|{r['scored']}|{fmt(r['line_pass_pct'])}|")
    lines+=['', '## 解释边界与证据', '',
        'MultiLine名称不等于实际跨行；跨度由task_id原始Lstart_Lend端点定义。全5815=1033+4782；跨行分1511/1870/1401。37基础题无跨行。总体最高点不自动是跨行最高点。single-line与multi-line共享164基础题及部分输入，本轮是任务形态迁移，不是独立新题泛化，不能拼接独立样本。未打开封存external test或旧test lock。',
        '', '固定64预算可能限制长孔洞。summary与span_summary保存生成长度、剩余mask、step-budget耗尽、未完成率、NFE、普通提交分布、额外接受/拒绝、链跳过、结构/no-op、头调用和峰值显存；不按答案或早期通过率改预算、阈值或删样本。K1额外接受率N/A。',
        '', '配对救回/改错/净变化见paired_comparisons.csv，覆盖对应K1、同方法off、同阈值另一方法及全部跨度层。10000次整题bootstrap只重采样当前人口真实存在基础题（164/127/164/127/108/60），缺席题不补0。逐例通过率为主，题等权仅辅助。nearest_quality.csv按已测逐例率寻找另一方法相近点；不假定头与无头置信度校准。',
        '', '统一计时164固定例×34×3=16728；127跨行题12954次，37单行回退3774次。每次恢复预热、CUDA同步、稳定轮换配置；包括模型/头/阈值/选位/采样/结构，排除加载、测试、写盘和重型轨迹。质量双worker阶段时延仅负载观测。H200非独占，timing_environment与GPU日志记外部争用，不能宣称确定速度达标。',
        '', '可复算证据per_case.jsonl.gz、timing_per_case.jsonl.gz；attempts.json保存基础设施尝试。普通失败和3秒超时不重抽。完整性、缺失、重复、无效、版本和计时token分歧见integrity_audit.json。reviewer_gate_disabled：本地diff review + fresh tests；原始日志、权重和重型trace不进Git。',
        '', '![全人口质量—时间](all_quality_time.png)', '', '![跨行质量—时间](crossline_quality_time.png)', '', '![跨行质量—实际并行度](crossline_quality_parallelism.png)', '', '恢复入口：resume.md与checkpoint.md。']
    (OUT/'report.zh.md').write_text('\n'.join(lines)+'\n')


def update_checkpoint(integrity,state):
    text=f'''# {TASK} checkpoint

状态：{state}；质量生成{integrity['quality_actual']}/{integrity.get('quality_expected',35122)}、评分{integrity['scored']}/{integrity.get('quality_expected',35122)}、统一计时{integrity['timing_actual']}/16728。
执行版本：{integrity['expected_execution_version']}；原始账本与心跳：{SHARED}。
报告：analysis_outputs/{TASK}/report.zh.md；审计：同目录integrity_audit.json。
恢复：bash {SHARED}/start_worker.sh --role nohead / trained；pipeline见resume.md。正常在跑时只读状态，禁止重复模型实例。共享GPU，不排队、不等待外部任务退出。
本轮单seed开发评测，历史结果不混算；未完整时不声称完成。
'''
    (OUT/'checkpoint.md').write_text(text)
    for filename in ('pause_checkpoint.current.md','paper_agent_dashboard.zh.md'):
        p=REPO/'docs/paper_agent'/filename
        old=p.read_text();marker=f'<!-- {TASK} -->'
        end_marker=f'<!-- /{TASK} -->'
        block=marker+'\n'+text+'\n'+end_marker+'\n'
        while marker in old:
            start=old.index(marker)
            if end_marker in old[start:]:
                end=old.index(end_marker,start)+len(end_marker)
                old=old[:start]+old[end:].lstrip('\n')
            else:
                # Upgrade the initial compact block written before an end marker existed.
                next_header=old.find('\n# ',start+len(marker))
                next_header=old.find('\n# ',next_header+1)
                if next_header<0:
                    raise ValueError('checkpoint legacy block lacks historical header')
                old=old[:start]+old[next_header+1:]
        p.write_text(block+'\n'+old)


def publish():
    for suite in ('tests/threshold_eval','tests/context_foundation'):
        subprocess.run([PYTHON,'-m','unittest','discover','-s',suite],cwd=REPO,check=True)
    # This dedicated branch has only task-owned edits. Stage explicit paths.
    paths=['experiments/threshold_eval','tests/threshold_eval',f'analysis_outputs/{TASK}',
           'docs/paper_agent/pause_checkpoint.current.md','docs/paper_agent/paper_agent_dashboard.zh.md']
    subprocess.run(['git','diff','--check'],cwd=REPO,check=True)
    staged=subprocess.run(['git','add','--',*paths],cwd=REPO,capture_output=True,text=True)
    if staged.returncode:raise RuntimeError(staged.stderr)
    names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True)
    print('Explicit staged files:\n'+names,flush=True)
    with (SHARED/'publication.log').open('a') as log:
        log.write('Staged files:\n'+names);log.flush()
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode:
            subprocess.run(['git','commit','-m','results: report DreamOn multiline threshold matrix and timing audit' if MULTILINE else 'results: report DreamOn threshold matrix and timing audit'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
        push_and_verify(branch,log)


def push_and_verify(branch,log):
    for attempt in range(4):
        try:
            subprocess.run(['git','push','-u','origin',branch],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
            remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True,timeout=60).split()[0]
            if remote!=commit():raise RuntimeError('remote commit mismatch')
            atomic_json(SHARED/'publication.status.json',dict(status='verified',commit=commit(),remote_commit=remote,attempts=attempt+1,at=utc()))
            return
        except (subprocess.SubprocessError,OSError,RuntimeError,IndexError) as error:
            log.write(f'{utc()} publication attempt {attempt+1}: {error}\n');log.flush()
            atomic_json(SHARED/'publication.status.json',dict(status='retrying' if attempt<3 else 'push_failed',commit=commit(),attempts=attempt+1,error=str(error),at=utc()))
    status=json.loads((SHARED/'status.json').read_text()) if (SHARED/'status.json').exists() else {}
    status.update(status='publication_blocked',at=utc())
    atomic_json(SHARED/'status.json',status)
    raise RuntimeError('Publication failed after initial attempt plus 3 infrastructure retries; see publication.log')



def main():
    p=argparse.ArgumentParser();p.add_argument('--score-only',action='store_true');p.add_argument('--partial',action='store_true');p.add_argument('--publish',action='store_true');a=p.parse_args()
    if a.score_only:score();return
    analyze(a.partial)
    if a.publish:publish()


if __name__=='__main__':main()
