#!/usr/bin/env python3
"""GPU-free scoring, grouped statistics, partial coverage and portable delivery."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import collections
import csv
import json
import os
import statistics
import time
import traceback

import numpy as np

from experiments.context_foundation.common import *


def csv_file(name, rows):
    path = OUT/name
    with path.open('w') as f:
        if rows:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def interval(values):
    a = np.asarray(values,dtype=float)
    a = a[np.isfinite(a)]
    return [float(v) for v in np.quantile(a,[.025,.975])] if len(a) else None


def grouped_bootstrap(rows, value_key, group_key='group', seed=20261015):
    if not rows:
        return dict(n=0,groups=0,mean=None,ci95=None)
    groups = sorted({r[group_key] for r in rows})
    sums,counts = [],[]
    for g in groups:
        vals = [r[value_key] for r in rows if r[group_key]==g]
        sums.append(sum(vals)); counts.append(len(vals))
    sums,counts = np.asarray(sums),np.asarray(counts)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
    estimates = (w@sums)/(w@counts)
    equal = w@(sums/counts)/len(groups)
    return dict(n=len(rows),groups=len(groups),mean=float(sums.sum()/counts.sum()),ci95=interval(estimates),
        equal_group_mean=float(np.mean(sums/counts)),equal_group_ci95=interval(equal),replicates=10000,seed=seed)


def paired_quality(scored, a, b):
    ra = {r['task_id']:r for r in scored if r['arm']==a}
    rb = {r['task_id']:r for r in scored if r['arm']==b}
    common = sorted(ra.keys() & rb.keys())
    rows = [dict(group=ra[k]['base_task_id'],delta=100*(int(ra[k]['passed'])-int(rb[k]['passed']))) for k in common]
    stats = grouped_bootstrap(rows,'delta')
    help_count = sum(ra[k]['passed'] and not rb[k]['passed'] for k in common)
    harm = sum(not ra[k]['passed'] and rb[k]['passed'] for k in common)
    return dict(a=a,b=b,matched=len(common),help=help_count,harm=harm,net=help_count-harm,**stats)


def learning_recovery(scored):
    arms = {a:{r['task_id']:r for r in scored if r['arm']==a} for a in ['K1','Z4','H4-old']}
    common = sorted(set.intersection(*(set(v) for v in arms.values())))
    if not common:
        return dict(matched=0,value=None,reason='no_matched_triples')
    numerator = sum(int(arms['H4-old'][k]['passed'])-int(arms['Z4'][k]['passed']) for k in common)
    denominator = sum(int(arms['K1'][k]['passed'])-int(arms['Z4'][k]['passed']) for k in common)
    groups = sorted({arms['K1'][k]['base_task_id'] for k in common})
    num = np.array([sum(int(arms['H4-old'][k]['passed'])-int(arms['Z4'][k]['passed']) for k in common if arms['K1'][k]['base_task_id']==g) for g in groups])
    den = np.array([sum(int(arms['K1'][k]['passed'])-int(arms['Z4'][k]['passed']) for k in common if arms['K1'][k]['base_task_id']==g) for g in groups])
    w = np.random.default_rng(20261015).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
    d = w@den
    valid = d>0
    ratio = (w@num)[valid]/d[valid]
    return dict(matched=len(common),numerator=numerator,denominator=denominator,
        value=float(numerator/denominator) if denominator>0 else None,
        reason='defined' if denominator>0 else 'nonpositive_denominator_NA',
        ci95_positive_denominator=interval(ratio),bootstrap_valid=int(valid.sum()),bootstrap_total=10000)


def score(store):
    from experiments.run_k4_multiline_stage import official_evaluate
    contexts = {r['task_id']:r for r in read_jsonl(REPO/'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz')}
    for r in store.rows('generated'):
        key = r['arm']+'::'+r['task_id']
        if store.get('scored',key):
            continue
        # Only infrastructure errors retry. Official timeout/assertion/syntax is final.
        for attempt in range(2):
            try:
                evaluation = official_evaluate(contexts[r['task_id']],r['output'])
                store.put('scored',key,dict(r,**evaluation))
                break
            except Exception as exc:
                store.put('score_infra',key+'::'+str(attempt),dict(key=key,attempt=attempt+1,error=str(exc),at=utc()))
        print(f'scored {store.count("scored")}/{store.count("generated")} {key}',flush=True)


def export(store):
    states,pairs,noise = store.rows('state'),store.rows('pair'),store.rows('noise')
    trajectories = store.rows('trajectory')
    scored,generated = store.rows('scored'),store.rows('generated')
    for kind,name in [('state','state_metrics.jsonl.gz'),('pair','counterfactual_pairs.jsonl.gz'),('noise','teacher_noise32.jsonl'),
                      ('trajectory','trajectory_coverage.jsonl.gz'),('generated','generated_rows.jsonl.gz'),
                      ('scored','per_case.jsonl.gz'),('score_infra','score_infrastructure_retries.jsonl'),('phase','phase_checkpoints.jsonl')]:
        write_jsonl(OUT/name,store.rows(kind))
    raw = []
    # Raw cache is separate from the Git portable evidence package. No absolute paths.
    seen = set()
    selected = list(read_jsonl(OUT/'blind_selected_states.jsonl.gz')) if (OUT/'blind_selected_states.jsonl.gz').exists() else []
    for r in selected+states+pairs+store.rows('dcd'):
        for pathkey,hashkey in [('raw_path','raw_sha256'),('teacher_path','teacher_sha256')]:
            if pathkey not in r or r[pathkey] in seen:
                continue
            p = CACHE/r[pathkey]
            valid = p.exists() and sha(p)==r[hashkey]
            raw.append(dict(path=r[pathkey],sha256=r[hashkey],rows=1,exists_verified=valid,state_id=r['state_id']))
            seen.add(r[pathkey])
    atomic_json(OUT/'tensor_manifest.json',dict(contract_id=CONTRACT,schema_version=1,
        storage='external lossless npz cache, logical paths relative to cache root; local root only in runtime_paths.local.json',
        precision='old hidden exact BF16->FP32; old/fresh logits and online metrics FP32; no half-precision teacher quantization',
        restore_command='bash scripts/launch_context_foundation_queue.sh --restore-missing',
        restore_scope='original source/trajectory/key replay within remaining original GPU budget; never reset budget',
        data=raw,missing_or_corrupt=sum(not r['exists_verified'] for r in raw)))
    atomic_json(OUT/'runtime_paths.local.json',dict(cache_root=str(CACHE),output_root=str(OUT),common_dir=str(common_dir())))
    coverage = []
    for order in ['global','frontier']:
        for length in [1,2,3]:
            for span in ['single','crossline']:
                cell = [r for r in states if r['order']==order and r['history_length']==length and r['span']==span]
                counts = collections.Counter(r['split'] for r in cell)
                branches = collections.Counter(r['arm'] for r in cell)
                groups = collections.Counter(r['group'] for r in cell)
                coverage.append(dict(order=order,history_length=length,span=span,target=256,actual=len(cell),
                    train=counts['train'],validation=counts['validation'],train_target=192,validation_target=64,
                    zero_origin=branches['Z4'],old_head_origin=branches['H4-old'],max_per_group=max(groups.values(),default=0),gap=256-len(cell)))
    csv_file('A1_coverage.csv',coverage)
    coverage_pairs = []
    for order in ['global','frontier']:
        for length in [2,3]:
            for span in ['single','crossline']:
                cell = [r for r in pairs if r.get('status')=='complete' and r['order']==order and r['history_length']==length and r['span']==span]
                coverage_pairs.append(dict(order=order,history_length=length,span=span,target=64,actual=len(cell),gap=64-len(cell)))
    csv_file('A2_coverage.csv',coverage_pairs)
    chain_summary = {}
    for order in ['global','frontier']:
        hist = collections.Counter()
        eligible = collections.Counter()
        for r in trajectories:
            if r['order']==order:
                hist.update({int(k):v for k,v in r['chain_length_counts'].items()})
                eligible.update(r['opportunities'])
        selected_count = sum(k*v for k,v in hist.items())
        chain_summary[order] = dict(trajectory_count=sum(r['order']==order for r in trajectories),chain_length_counts=dict(hist),
            all_selected_slots=selected_count,slots_in_chains_at_least2=sum(k*v for k,v in hist.items() if k>=2),
            potential_correction_slots=sum((k-1)*v for k,v in hist.items()),actually_visited_prefix_opportunities=dict(eligible),
            scope='all encountered rounds, including barriers/gaps; distinct from selected long-chain subset')
    atomic_json(OUT/'chain_coverage_summary.json',chain_summary)
    mechanism = {}
    for label in ['stale','head']:
        for metric in ['kl_T1','tv_T1','tv_deploy']:
            data = [dict(group=r['group'],value=r[label][metric]) for r in states]
            mechanism[label+'_'+metric] = grouped_bootstrap(data,'value')
    if states:
        mechanism['head_minus_stale_tv_T1'] = grouped_bootstrap([dict(group=r['group'],value=r['head']['tv_T1']-r['stale']['tv_T1']) for r in states],'value')
    complete_pairs = [r for r in pairs if r['status']=='complete']
    noise_values = [r['repeat']['tv_T1'] for r in noise]
    max_noise = max(noise_values,default=None)
    cf = grouped_bootstrap([dict(group=r['group'],value=r['teacher_change']['tv_T1']) for r in complete_pairs],'value')
    cf.update(noise_states=len(noise),noise_max_tv_T1=max_noise,
        above_observed_noise=sum(r['teacher_change']['tv_T1']>max_noise for r in complete_pairs) if max_noise is not None else None,
        paired_states=len(complete_pairs),interpretation='interventions nested within parent states/groups; tiny nonzero changes alone do not establish a need for complex heads')
    atomic_json(OUT/'mechanism_statistics.json',dict(distributions=mechanism,counterfactual=cf))
    stratified=[]
    cells=collections.defaultdict(list)
    for r in states:
        cells[(r['order'],r['history_length'],r['span'],r['split'],r['arm'])].append(r)
    for (order,length,span,split,origin), rs in sorted(cells.items()):
        delta=grouped_bootstrap([dict(group=r['group'],value=r['head']['tv_T1']-r['stale']['tv_T1']) for r in rs],'value')
        summary=dict(order=order,history_length=length,span=span,split=split,origin=origin,n=len(rs),groups=delta['groups'],
            head_minus_stale_T1_TV=delta['mean'],delta_ci_low=delta['ci95'][0],delta_ci_high=delta['ci95'][1])
        for label in ['stale','head']:
            for metric in ['kl_T1','tv_T1','tv_deploy','special_teacher_T1','special_estimate_T1']:
                summary[label+'_'+metric]=float(np.mean([r[label][metric] for r in rs]))
            summary[label+'_deploy_KL_infinite_count']=sum(r[label]['kl_deploy']=='infinite' for r in rs)
        stratified.append(summary)
    csv_file('mechanism_stratified.csv',stratified)
    cf_strata={}
    for order in ['global','frontier']:
        for length in [2,3]:
            for span in ['single','crossline']:
                data=[dict(group=r['group'],value=r['teacher_change']['tv_T1']) for r in complete_pairs
                      if r['order']==order and r['history_length']==length and r['span']==span]
                cf_strata[f'{order}|{length}|{span}']=grouped_bootstrap(data,'value')
    atomic_json(OUT/'counterfactual_stratified_bootstrap.json',cf_strata)
    # Examples are fixed hashes, never picked for a successful result or a large effect.
    examples = sorted(states,key=lambda r:stable(['example',r['state_id']]))[:8]
    atomic_json(OUT/'mechanism_examples.json',examples)
    table = []
    for arm in ['K1','Z4','H4-old','F4-chain']:
        rs = [r for r in scored if r['arm']==arm]
        table.append(dict(arm=arm,expected=1033,generated=sum(r['arm']==arm for r in generated),scored=len(rs),
            passed=sum(r['passed'] for r in rs),pass_pct=100*sum(r['passed'] for r in rs)/len(rs) if rs else None,
            status='complete' if len(rs)==1033 else 'partial',
            actual_submitted=sum(r['actual_submitted_count'] for r in rs),NFE=sum(r['forward_count'] for r in rs),
            head_calls=sum(r['head_calls'] for r in rs)))
    csv_file('generation_summary.csv',table)
    comparisons = [paired_quality(scored,a,b) for a,b in [('H4-old','Z4'),('F4-chain','Z4'),('F4-chain','H4-old'),
        ('Z4','K1'),('H4-old','K1'),('F4-chain','K1')]]
    atomic_json(OUT/'paired_comparisons.json',dict(comparisons=comparisons,pure_learning_recovery=learning_recovery(scored),
        total_recovery=None,total_recovery_reason='no native K4 arm under common-u; historical recovery not mixed',
        population='1033 official rows,164 base tasks,single seed42,repeated development population',
        bootstrap='10000 whole base-task clusters, line-weighted and equal-base-task estimates; paired common population only'))
    error_types = []
    f4 = {r['task_id']:r for r in scored if r['arm']=='F4-chain'}
    h4 = {r['task_id']:r for r in scored if r['arm']=='H4-old'}
    for key in sorted(f4.keys() & h4.keys()):
        if h4[key]['passed'] and not f4[key]['passed']:
            error_types.append(dict(task_id=key,base_task_id=f4[key]['base_task_id'],failure_reason=f4[key]['failure_reason'],
                F4_output=f4[key]['output'],old_head_output=h4[key]['output']))
    write_jsonl(OUT/'fresh_harms_vs_old_head.jsonl.gz',error_types)
    expected_keys = {a+'::'+r['task_id'] for a in ['K1','Z4','H4-old','F4-chain']
        for r in read_jsonl(REPO/'analysis_outputs/k4_multiline_20261014_v1/single-line_contexts.jsonl.gz')}
    done = {r['arm']+'::'+r['task_id'] for r in scored}
    gen = {r['arm']+'::'+r['task_id'] for r in generated}
    write_jsonl(OUT/'generation_retry_matrix.jsonl.gz',[dict(key=k,status='scored' if k in done else 'generated_needs_score' if k in gen else 'not_generated',
        retry='none' if k in done else 'same_seed_same_key_no_resampling') for k in sorted(expected_keys)])
    budget_path = OUT/'budget.local.json'
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else dict(used_sec=0,stage_sec={})
    waiting = json.loads((OUT/'waiting.local.json').read_text()) if (OUT/'waiting.local.json').exists() else {}
    atomic_json(OUT/'shared_gpu_timing.json',dict(gpu_job_sec=budget['used_sec'],stage_sec=budget['stage_sec'],
        gpu_wait_sec=waiting.get('waited_sec',0),gpu='single H200, GPU0',policy='cooperative existing global flock plus dedicated A lock; external processes need not honor either',
        inference='shared GPU observations only; no F4 speedup or fair-hardware acceleration claim'))
    if (OUT/'gpu_observations.local.jsonl').exists():
        write_jsonl(OUT/'shared_gpu_observations.jsonl',read_jsonl(OUT/'gpu_observations.local.jsonl'))
    phases = {r['state_id'] for r in states}
    matrix = dict(contract_id=CONTRACT,schema_version=1,
        A1=dict(target=3072,actual=len(states),gap=3072-len(states)),
        A2=dict(target=512,actual=len(complete_pairs),gap=512-len(complete_pairs)),
        teacher_noise=dict(target=32,actual=len(noise)),DCD=dict(target=256,actual=store.count('dcd')),
        generation=dict(target=4132,generated=len(generated),scored=len(scored),missing=len(expected_keys-done),duplicates=len(scored)-len(done)),
        cpu_technical_passed='OK' in (OUT/'cpu_tests.txt').read_text(),
        gpu_technical_passed=(OUT/'gpu_smoke.json').exists() and json.loads((OUT/'gpu_smoke.json').read_text())['status']=='passed',
        source_audit_passed=json.loads((OUT/'source_audit.json').read_text())['status']=='passed',
        raw_shards_verified=all(r['exists_verified'] for r in raw),gpu_job_sec=budget['used_sec'],
        infrastructure_retry_events=store.count('score_infra'),scope='A only; no training, second batch, native K4, frontier/multiline full runs or mindmap change')
    complete = (len(states)==3072 and len(complete_pairs)==512 and len(noise)==32 and store.count('dcd')==256 and len(scored)==4132)
    matrix['status'] = 'complete' if complete and matrix['gpu_technical_passed'] and matrix['raw_shards_verified'] else 'partial'
    if not matrix['gpu_technical_passed']:
        matrix['status']='blocked_technical_or_resource'
    atomic_json(OUT/'completion_matrix.json',matrix)
    return matrix,table,comparisons,cf,chain_summary


def report(matrix,table,comparisons,cf,chains):
    lines = ['# 第一批 A：共同采样、共享状态和 K4 链局部真实重算', '',
        f"状态：**{matrix['status']}**。A1={matrix['A1']['actual']}/3072；反事实={matrix['A2']['actual']}/512；DCD={matrix['DCD']['actual']}/256；评分={matrix['generation']['scored']}/4132。",'',
        '冻结 DreamOn 8ccc747、旧 v2 KL/full/best rank256，seed42、T=.2/top-p=.9、batch1；本批 common-u 和 FP32 运算差异见 frozen_protocol.json。起点 d39a51d，坐标修补560e98a已包含；旧结果不变、不混算逐例救回。', '',
        '|组|生成|评分|通过|通过率|', '|---|---:|---:|---:|---:|']
    for r in table:
        rate = f"{r['pass_pct']:.2f}%" if r['pass_pct'] is not None else 'N/A'
        lines.append(f"|{r['arm']}|{r['generated']}|{r['scored']}|{r['passed']}|{rate}|")
    lines += ['', '未完成组的比率只描述实际已评分行，不代表完整1033人口。完整输出、逐行官方3秒评分、实际提交/NFE/头调用、扩删屏障和key摘要见 per_case.jsonl.gz；固定hash32案例留逐轮画布。', '',
        '配对（先列帮助/伤害；百分点与基础题bootstrap见paired_comparisons.json）：']
    for r in comparisons:
        lines.append(f"- {r['a']} − {r['b']}：共同{r['matched']}行，救回{r['help']}、改错{r['harm']}、净{r['net']:+d}；差={r['mean'] if r['mean'] is not None else 'N/A'}pp，95%区间={r['ci95']}。")
    lines += ['', '纯学习恢复=(H4-old−Z4)/(K1−Z4)，仅在分母为正时定义；数值/有效bootstrap次数见配对JSON。未跑common-u原生K4，不报告同协议总体恢复。F4差值是有符号参考差，不是上界。', '',
        f"更早决定的影响：完成{cf['paired_states']}个干预对，teacher T1 TV均值={cf['mean']}、按源题组95%区间={cf['ci95']}；重复teacher={cf['noise_states']}/32，观察到的最大数值噪声TV={cf['noise_max_tv_T1']}，超过该阈值的对数={cf['above_observed_noise']}。"]
    if cf['paired_states']==0 or cf['noise_states']<32:
        lines.append('当前干预或32状态重复计算不足，无法作完整的“超出数值噪声”判断。')
    else:
        lines.append('这一结果回答固定最近词时更早干预是否改变条件分布；任何很小的非零值都不单独证明需要复杂头。干预不是两条自然轨迹，母/子状态不计为独立题。')
    lines += ['', f"global覆盖：所有已遇到轨迹的链长分布={chains['global']['chain_length_counts']}；选中槽位={chains['global']['all_selected_slots']}，位于至少2长度链={chains['global']['slots_in_chains_at_least2']}，潜在修正槽位={chains['global']['potential_correction_slots']}。分母包含全部轮次/断链，不只筛出的长链。源孔洞跨度定义single/crossline；early/middle/late仅记账。",'',
        'F4是否有实际净收益及造成的错误：以F4−H4-old的有符号配对和fresh_harms_vs_old_head.jsonl.gz为准；该文件保存所有旧头通过而F4失败的输出/原因，不隐去负结果。分布变近不等于代码通过。', '',
        f"共同随机输入及零头：CPU技术={'通过' if matrix['cpu_technical_passed'] else '未通过'}；真实模型逐token/逐画布与h/logit对齐={'通过' if matrix['gpu_technical_passed'] else '未完成或失败'}。零头真实走接口，其耗时不冒充无头。不同轨迹的位置含义不保证相同。", '',
        'DCD是K4母状态派生的两位置截断诊断，首位为旧p1；候选只看X0，各取16普通词，完整词表归一化后抽取。稳定log字段为数值权威，候选外/特殊概率保留；C无需模型、GPU或SQLite即可读取小包。', '',
        '第二批候选：仅当反事实影响超过数值噪声且F4配对有净收益时，才有理由继续评估读更长同链上下文的头；若global长链覆盖不足，应先判断其适用人口。复杂头、跨间隔、Jacobi、融合及公平测速的证据均尚不足；本批不自动启动。', '',
        f"GPU作业时间={matrix['gpu_job_sec']/3600:.4f}h/12h；包括模型驻留、写盘和教师；等待与共享GPU墙时见shared_gpu_timing.json。单H200单worker，CPU隔离评分在GPU进程退出后运行。没有训练新头或打开封闭external test。", '',
        '恢复：在本分支执行 `bash scripts/launch_context_foundation_queue.sh`，按事务原seed/key补缺，不重抽普通失败；耗尽的预算不重置。原始lossless FP32张量缓存的相对索引/hash和恢复入口见tensor_manifest.json。可移植结果无需本机绝对路径；本机路径仅在*.local.json和公共git目录ready文件。', '',
        '技术/范围限制：共享负载、单seed、重复开发164基础题；机制状态使用原train/validation分组，不使用1033/5815开发评测作状态训练库。teacher只读本链实际前缀，不是完整实际画布刷新、不等于K1、不是正确答案或通过率上界。', '']
    (OUT/'report.zh.md').write_text('\n'.join(lines))


def main():
    store=Store()
    score(store)
    matrix,table,comparisons,cf,chains=export(store)
    report(matrix,table,comparisons,cf,chains)
    store.db.close()
    print('CPU analysis and portable exports finished:',matrix['status'],flush=True)


if __name__=='__main__':
    main()
