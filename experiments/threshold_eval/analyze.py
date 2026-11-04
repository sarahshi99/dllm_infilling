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
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(OUT/name)


def score():
    from experiments.k4_batch_common import population
    from experiments.run_k4_multiline_stage import official_evaluate
    contexts={r['task_id']:r for r in population('single-line')}
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


def grouped(values,keys,weights):
    group_sums=np.zeros(164);group_counts=np.zeros(164)
    for value,key in zip(values,keys):
        g=int(key.split('/')[2]);group_sums[g]+=value;group_counts[g]+=1
    present=group_counts>0
    if not present.any():return dict(line_mean=None,task_mean=None,line_ci95=None,task_ci95=None)
    if not present.all():
        # Partial results never get a full-population interval.
        return dict(line_mean=float(group_sums.sum()/group_counts.sum()),task_mean=float(np.mean(group_sums[present]/group_counts[present])),line_ci95=None,task_ci95=None)
    line=(weights@group_sums)/(weights@group_counts)
    task=weights@(group_sums/group_counts)/164
    return dict(line_mean=float(group_sums.sum()/group_counts.sum()),task_mean=float(np.mean(group_sums/group_counts)),line_ci95=bounds(line),task_ci95=bounds(task))


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
    for r in quality:
        if not isinstance(r.get('completion_token_ids'),list) or r.get('forward_count',0)<=0:invalid.append([r['config'],r['task_id'],'generation_schema'])
        if 'passed' in r and type(r['passed']) is not bool:invalid.append([r['config'],r['task_id'],'score_schema'])
        if r['method']=='nohead' and r['head_calls']!=0:invalid.append([r['config'],r['task_id'],'nohead_called_head'])
        hist={int(k):v for k,v in r['normal_submissions_histogram'].items()}
        if sum(hist.values())!=r['forward_count'] or sum(k*v for k,v in hist.items())!=r['normal_token_count']:
            invalid.append([r['config'],r['task_id'],'histogram'])
        st=r['statistics']
        if st['extra_ordinary_attempted']!=st['extra_ordinary_accepted']+st['extra_ordinary_rejected']:
            invalid.append([r['config'],r['task_id'],'threshold_accounting'])
    timing_divergences=[]
    for r in timing:
        q=quality_bykey.get((r['config'],r['task_id']))
        if q and r['completion_token_ids']!=q['completion_token_ids']:
            timing_divergences.append([r['config'],r['task_id'],r['repetition']])
    result=dict(quality_expected=len(expected),quality_actual=len(quality),scored=sum('passed' in r for r in quality),
        quality_unique=len(set(found)),quality_missing=sorted(expected-set(found)),quality_unexpected=sorted(set(found)-expected),
        quality_duplicates=len(found)-len(set(found)),timing_expected=len(timed_expected),timing_actual=len(timing),timing_unique=len(set(timed_found)),
        timing_missing=sorted(timed_expected-set(timed_found)),timing_unexpected=sorted(set(timed_found)-timed_expected),timing_duplicates=len(timed_found)-len(set(timed_found)),
        execution_versions=versions,expected_execution_version=version,version_mixing=bool(versions and versions!=[version]),invalid=invalid,
        timing_output_divergences=timing_divergences,timing_population_separate=True,sealed_external_test_opened=False,
        per_config={c['id']:dict(generated=sum(r['config']==c['id'] for r in quality),scored=sum(r['config']==c['id'] and 'passed' in r for r in quality),timed=sum(r['config']==c['id'] for r in timing)) for c in sp['configs']})
    result['complete']=bool(len(set(found))==35122 and result['scored']==35122 and len(set(timed_found))==16728 and not any((result['quality_duplicates'],result['timing_duplicates'],result['version_mixing'],invalid,result['quality_unexpected'],result['timing_unexpected'])))
    atomic_json(OUT/'integrity_audit.json',result)
    return result


def load_results():
    quality=[]
    for role in ('nohead','trained'):
        db=Ledger(role);scored={(r['config'],r['task_id']):r for r in db.rows('score')}
        quality.extend(dict(r,**scored.get((r['config'],r['task_id']),{})) for r in db.rows('quality'))
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


def analyze(partial=False):
    OUT.mkdir(parents=True,exist_ok=True)
    quality,timing=load_results();integrity=audit(quality,timing)
    # Portable complete evidence, with quality and timing kept separate.
    for name,rows in [('per_case.jsonl.gz',quality),('timing_per_case.jsonl.gz',timing)]:
        tmp=OUT/(name+'.tmp')
        with gzip.open(tmp,'wt') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        tmp.replace(OUT/name)
    weights=np.random.default_rng(20261104).multinomial(164,np.full(164,1/164),size=10000)
    indexed={c['id']:{r['task_id']:r for r in quality if r['config']==c['id'] and 'passed' in r} for c in configs()}
    summary=[];timing_summary=[];histograms={};attempts=[]
    for c in configs():
        cid=c['id'];rows=[r for r in quality if r['config']==cid];scored=list(indexed[cid].values())
        keys=[r['task_id'] for r in scored]
        q=grouped([100*int(r['passed']) for r in scored],keys,weights)
        nfe=sum(r['forward_count'] for r in rows);normal=sum(r['normal_token_count'] for r in rows)
        st=collections.Counter();hist=collections.Counter()
        for r in rows:
            st.update(r['statistics']);hist.update({int(k):v for k,v in r['normal_submissions_histogram'].items()})
        histograms[cid]=dict(sorted(hist.items()))
        tr=[r for r in timing if r['config']==cid];secs=[r['generation_sec'] for r in tr]
        sharedsecs=[r['generation_sec'] for r in rows]
        def metric_mean(values):return statistics.mean(values) if values else None
        t=dict(config=cid,n=len(tr),expected=492,mean_sec=metric_mean(secs),median_sec=statistics.median(secs) if secs else None,
            p90_sec=float(np.quantile(secs,.9)) if secs else None,total_sec=sum(secs),shared_environment=True,
            peak_allocated_bytes=max([r['peak_memory_allocated_bytes'] for r in tr],default=0),peak_reserved_bytes=max([r['peak_memory_reserved_bytes'] for r in tr],default=0))
        timing_summary.append(t)
        attempted=st['extra_ordinary_attempted'];evaluated=st['evaluated_candidates']
        s=dict(config=cid,order=c['order'],k=c['k'],method=c['method'],threshold='off' if c['threshold'] is None else c['threshold'],
            generated=len(rows),scored=len(scored),expected=1033,success=sum(r['passed'] for r in scored),
            line_pass_pct=q['line_mean'],task_equal_pass_pct=q['task_mean'],line_ci95=json.dumps(q['line_ci95']),task_ci95=json.dumps(q['task_ci95']),
            forward_count=nfe,normal_commits=normal,normal_commits_per_forward=normal/nfe if nfe else None,
            extra_attempted=attempted,extra_accepted=st['extra_ordinary_accepted'],extra_rejected=st['extra_ordinary_rejected'],
            extra_accept_rate=st['extra_ordinary_accepted']/attempted if attempted else None,extra_reject_rate=st['extra_ordinary_rejected']/attempted if attempted else None,
            selected_candidates=st['selected_candidates'],chain_skipped=st['chain_skipped'],head_calls=sum(r['head_calls'] for r in rows),
            expand_count=sum(r['expand_count'] for r in rows),delete_count=sum(r['delete_count'] for r in rows),base_structure_rounds=st['base_structure_rounds'],
            corrected_structure_rounds=st['corrected_structure_rounds'],structure_proportion=(st['base_structure_rounds']+st['corrected_structure_rounds'])/evaluated if evaluated else None,
            noop_count=st['mask_noop'],noop_proportion=st['mask_noop']/evaluated if evaluated else None,
            mean_generated_length=metric_mean([len(r['completion_token_ids']) for r in rows]),unfinished_rate=metric_mean([r['final_remaining_masks']>0 for r in rows]),
            observed_quality_mean_sec=metric_mean(sharedsecs),observed_quality_median_sec=statistics.median(sharedsecs) if sharedsecs else None,
            observed_quality_p90_sec=float(np.quantile(sharedsecs,.9)) if sharedsecs else None,observed_quality_total_sec=sum(sharedsecs),
            timing_n=t['n'],timing_mean_sec=t['mean_sec'],timing_median_sec=t['median_sec'],timing_p90_sec=t['p90_sec'],timing_total_sec=t['total_sec'],
            peak_allocated_bytes=max([r['peak_memory_allocated_bytes'] for r in rows+tr],default=0),peak_reserved_bytes=max([r['peak_memory_reserved_bytes'] for r in rows+tr],default=0))
        summary.append(s)
    comparisons=[]
    for c in configs():
        cid=c['id'];base=f"{c['order']}_k1_nohead_off"
        off=f"{c['order']}_k{c['k']}_{c['method']}_off"
        other=cid.replace('_nohead_','_trained_') if c['method']=='nohead' else cid.replace('_trained_','_nohead_')
        for relation,bid in [('vs_k1',base),('vs_same_method_off',off),('vs_other_method_same_threshold',other)]:
            if bid==cid or bid not in indexed:continue
            a,b=indexed[cid],indexed[bid];keys=sorted(a.keys()&b.keys())
            delta=[100*(int(a[k]['passed'])-int(b[k]['passed'])) for k in keys]
            q=grouped(delta,keys,weights)
            help_count=sum(a[k]['passed'] and not b[k]['passed'] for k in keys)
            harm=sum(not a[k]['passed'] and b[k]['passed'] for k in keys)
            comparisons.append(dict(config=cid,reference=bid,relation=relation,matched=len(keys),rescued=help_count,harmed=harm,net=help_count-harm,
                line_delta_pp=q['line_mean'],task_equal_delta_pp=q['task_mean'],line_ci95=json.dumps(q['line_ci95']),task_ci95=json.dumps(q['task_ci95']),
                bootstrap_unit='base_task',bootstrap_repetitions=10000,bootstrap_seed=20261104))
    smap={s['config']:s for s in summary};tmap={t['config']:t for t in timing_summary}
    cmap={(r['config'],r['relation']):r for r in comparisons}
    for s in summary:
        base=smap[f"{s['order']}_k1_nohead_off"]
        for relation in ['vs_k1','vs_same_method_off','vs_other_method_same_threshold']:
            pair=cmap.get((s['config'],relation))
            s[relation+'_line_delta_pp']=pair['line_delta_pp'] if pair else (0 if relation!='vs_other_method_same_threshold' else None)
            s[relation+'_task_delta_pp']=pair['task_equal_delta_pp'] if pair else (0 if relation!='vs_other_method_same_threshold' else None)
        fulltime=s['timing_n']==492 and base['timing_n']==492
        ratio=s['timing_total_sec']/base['timing_total_sec'] if fulltime and base['timing_total_sec'] else None
        s['timing_total_ratio_vs_k1']=ratio
        s['exploratory_engineering_flag']=bool(s['generated']==1033 and s['scored']==1033 and s['vs_k1_task_delta_pp'] is not None and s['vs_k1_task_delta_pp']>=-1 and ratio is not None and ratio<=.80)
        s['definitive_engineering_attainment']='undetermined_shared_environment'
        tmap[s['config']]['total_ratio_vs_k1']=ratio
    # Nearest measured quality point of the other method, selected only for descriptive reporting.
    nearest=[]
    for s in summary:
        candidates=[b for b in summary if b['order']==s['order'] and b['method']!=s['method'] and b['task_equal_pass_pct'] is not None]
        if s['task_equal_pass_pct'] is None or not candidates:continue
        b=min(candidates,key=lambda b:(abs(s['task_equal_pass_pct']-b['task_equal_pass_pct']),b['config']))
        nearest.append(dict(config=s['config'],other=b['config'],task_quality_gap_pp=s['task_equal_pass_pct']-b['task_equal_pass_pct'],
            timing_ratio=s['timing_mean_sec']/b['timing_mean_sec'] if s['timing_n']==b['timing_n']==492 else None,
            selection='nearest measured task-mean quality, descriptive posthoc; confidence scales not assumed calibrated'))
    write_csv('summary.csv',summary);write_csv('paired_comparisons.csv',comparisons);write_csv('timing.csv',timing_summary);write_csv('nearest_quality.csv',nearest)
    atomic_json(OUT/'submission_distributions.json',histograms)
    atomic_json(OUT/'timing_environment.json',gpu_contention())
    for role in ('nohead','trained','timing'):
        db=Ledger(role)
        attempts.extend(dict(role=role,at=r[0],phase=r[1],key=r[2],error=r[3]) for r in db.db.execute('SELECT at,phase,key,error FROM attempts ORDER BY id'))
        db.db.close()
    atomic_json(OUT/'attempts.json',attempts)
    charts(summary)
    report(summary,comparisons,integrity,nearest)
    state='completed' if integrity['complete'] else 'incomplete'
    atomic_json(SHARED/'status.json',dict(task_id=TASK,status=state,quality_completed=len(quality),quality_scored=sum('passed' in r for r in quality),timing_completed=len(timing),at=utc(),integrity=str(OUT/'integrity_audit.json')))
    atomic_json(OUT/'status.json',json.loads((SHARED/'status.json').read_text()))
    (OUT/'commands.txt').write_text((SHARED/'commands.txt').read_text())
    update_checkpoint(integrity,state)
    if not partial and not integrity['complete']:
        raise RuntimeError('Full completion audit failed; see integrity_audit.json')


def charts(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for axis,name,xlabel in [('timing_mean_sec','quality_time','统一计时均值（秒） / shared GPU'),('normal_commits_per_forward','quality_parallelism','Ordinary commits / forward')]:
        fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,order in zip(axs,('global','left')):
            for method,marker in [('nohead','o'),('trained','s')]:
                for k,color in [(2,'#2878b5'),(4,'#d95f02')]:
                    rows=[s for s in summary if s['order']==order and s['method']==method and s['k']==k and s[axis] is not None and s['task_equal_pass_pct'] is not None]
                    rows.sort(key=lambda r:(r['threshold']!='off',str(r['threshold'])))
                    ax.plot([r[axis] for r in rows],[r['task_equal_pass_pct'] for r in rows],marker=marker,color=color,linestyle='-' if method=='nohead' else '--',label=f'{method} K{k}')
                    for r in rows:ax.annotate(str(r['threshold']),(r[axis],r['task_equal_pass_pct']),fontsize=7)
            b=next(s for s in summary if s['order']==order and s['k']==1)
            if b[axis] is not None and b['task_equal_pass_pct'] is not None:ax.scatter([b[axis]],[b['task_equal_pass_pct']],c='black',marker='*',s=90,label='K1')
            ax.set_title(order);ax.set_xlabel(xlabel if axis!='timing_mean_sec' else 'Mean unified generation seconds (shared GPU)');ax.grid(alpha=.25);ax.legend(fontsize=8)
        axs[0].set_ylabel('Base-task equal pass rate (%)');fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)


def report(summary,comparisons,integrity,nearest):
    def fmt(v):return '未完成' if v is None else f'{v:.2f}'
    lines=['# DreamOn 阈值完整矩阵评测', '',
        f"状态：{'完整完成' if integrity['complete'] else '未完成，不能作完整实验结论'}。质量生成 {integrity['quality_actual']}/35122，评分 {integrity['scored']}/35122；计时 {integrity['timing_actual']}/16728。",
        '',f"起点 43a52fd；实际执行提交 `{integrity['expected_execution_version']}`。34配置、每配置1033唯一案例、164基础题；固定seed42、T=.2、top-p=.9、batch1、entropy/alg_temp0、mask64、steps256。DreamOn revision8ccc747，bf16；冻结rank256 v2 KL/full/best头float32、λ=1，sha256见run_spec。",
        '', '阈值按完整合法词表T=1 FP32概率，对实际抽到的额外普通词判定；全轮首个空间候选唯一豁免。拒绝不重抽、不写入前驱，停止当前连续链；独立链首仍判阈值。原始扩删一次结束轮次，修正结构词保持屏障。共同随机键完全继承旧协议，不含本轮任务ID/方法/阈值/worker；分叉后不保证同坐标语义相同。',
        '', '|配置|通过/1033|行均值%|基础题等权%|相对K1任务pp|普通提交/NFE|额外接受率|统一秒均值|K1时间比|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"|{s['config']}|{s['success']}/{s['expected']}|{fmt(s['line_pass_pct'])}|{fmt(s['task_equal_pass_pct'])}|{fmt(s['vs_k1_task_delta_pp'])}|{fmt(s['normal_commits_per_forward'])}|{fmt(s['extra_accept_rate'])}|{fmt(s['timing_mean_sec'])}|{fmt(s['timing_total_ratio_vs_k1'])}|")
    lines+=['','普通提交统计排除结构与no-op。部分状态的通过率只描述已评分人口；不得当成1033全量率。逐例输出、token、NFE、结构计数、额外词接受拒绝、提交分布、显存与耗时见per_case.jsonl.gz。计时重跑另见timing_per_case.jsonl.gz，不计新质量样本。',
        '', '配对比较：summary.csv列出每配置相对对应K1、同方法off、同阈值另一方法的行和题均值差。paired_comparisons.csv保存逐例救回/改错/净变化和整道基础题bootstrap95%区间（10000次，seed20261104）；每次重采样164题并保留题内全部行。不能把1033行当独立题。', '', '## 预设问题的实测判断']
    smap={s['config']:s for s in summary}
    for order in ('global','left'):
        for k in (2,4):
            no=smap[f'{order}_k{k}_nohead_off'];base=smap[f'{order}_k1_nohead_off']
            candidates=[s for s in summary if s['order']==order and s['k']==k and s['method']=='nohead' and s['threshold']!='off' and s['scored']==1033]
            if candidates:
                best=max(candidates,key=lambda s:s['task_equal_pass_pct'])
                gap=base['task_equal_pass_pct']-no['task_equal_pass_pct']
                recovery=(best['task_equal_pass_pct']-no['task_equal_pass_pct'])/gap if gap>0 else None
                lines.append(f"- {order} K{k} 无头：已测阈值中题均值最高点为{best['threshold']}，相对off {fmt(best['task_equal_pass_pct']-no['task_equal_pass_pct'])}pp，恢复K1差距比例{fmt(recovery)}，普通提交/NFE从{fmt(no['normal_commits_per_forward'])}降至{fmt(best['normal_commits_per_forward'])}；其质量相对K1 {fmt(best['vs_k1_task_delta_pp'])}pp。此为事后描述最佳点，完整负结果仍在表中。")
    for order in ('global','left'):
        pairs=[p for p in comparisons if p['relation']=='vs_other_method_same_threshold' and p['config'].startswith(order) and '_trained_' in p['config'] and p['matched']==1033]
        for p in pairs:
            lines.append(f"- {p['config']} − 无头同阈值：救回{p['rescued']}、改错{p['harmed']}、净{p['net']:+d}，题均值{fmt(p['task_equal_delta_pp'])}pp，95%区间{p['task_ci95']}。")
    flags=[s['config'] for s in summary if s['exploratory_engineering_flag']]
    lines += ['',f"预设工程标记（题均值相对K1下降≤1pp且统一总生成时间减少≥20%）的观察候选：{', '.join(flags) if flags else '暂无满足的完整实测点'}。当前为非独占共享H200，存在争用时不作确定达标判断；无候选时也不以未完成计时代替否定结论。",
        '', '相近质量的跨方法时间比较见nearest_quality.csv：同选位内寻找已测题均值距离最小的另一方法配置，报告实际质量差和时间比；属于事后描述，不假定置信度已校准。',
        '', '计时为164基础题各固定1例×34配置×3次，配置按case/repetition稳定轮换。每次恢复先对所有配置预热；cuda同步边界包含主模型、头、阈值、选位、共同采样及结构操作，排除模型加载、官方代码测试、写盘与重型轨迹。质量阶段两个模型并发的耗时仅作负载观测。外部GPU进程和显存每30秒记录，摘要见timing_environment.json；不声称独占公平测速。',
        '', '本轮为单seed、反复使用开发题的探索评测。没有打开封存external test，没有训练或换头，没有教师重算/KV缓存/多行/K8/新阈值；历史951/817/831等结果不混入配对。不能宣称独立验证、正确率保证或方法新颖性。',
        '', '完整性、缺失、重复、异常、版本混用和计时输出分歧见integrity_audit.json；所有基础设施尝试见attempts.json。普通失败/3秒超时原样保留；无按结果补抽。reviewer_gate_disabled：本地diff review + fresh CPU/GPU技术检查，未有独立reviewer。',
        '', '![质量—耗时](quality_time.png)', '', '![质量—实际并行度](quality_parallelism.png)', '', '恢复入口：见resume.md；共享状态和角色锁位于run_spec指定目录；大日志和权重不进普通git。']
    (OUT/'report.zh.md').write_text('\n'.join(lines)+'\n')


def update_checkpoint(integrity,state):
    text=f'''# {TASK} checkpoint

状态：{state}；质量生成{integrity['quality_actual']}/35122、评分{integrity['scored']}/35122、统一计时{integrity['timing_actual']}/16728。
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
            subprocess.run(['git','commit','-m','results: report DreamOn threshold matrix and timing audit'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
        pushed=subprocess.run(['git','push','-u','origin',branch],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if pushed.returncode:
            atomic_json(SHARED/'publication.status.json',dict(status='push_failed',commit=commit(),at=utc()));return
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True,timeout=60).split()[0]
        atomic_json(SHARED/'publication.status.json',dict(status='verified' if remote==commit() else 'remote_mismatch',commit=commit(),remote_commit=remote,at=utc()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--score-only',action='store_true');p.add_argument('--partial',action='store_true');p.add_argument('--publish',action='store_true');a=p.parse_args()
    if a.score_only:score();return
    analyze(a.partial)
    if a.publish:publish()


if __name__=='__main__':main()
