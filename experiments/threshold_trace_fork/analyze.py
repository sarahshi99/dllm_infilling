"""Compact paired diagnostics; task-cluster bootstrap, panel kept separate."""
from collections import Counter,defaultdict
import gzip
import json
import numpy as np
from experiments.context_foundation.common import atomic_json
from experiments.threshold_trace_fork.run import OUT,LOCAL,FOCUS,IDS,trace_read
from experiments.threshold_trace_fork.forks import first_action_difference,events,action


def export(name,rows):
    with gzip.open(OUT/name,'wt') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')


def clustered(rows,field,reps=3000):
    groups=defaultdict(list)
    for r in rows:groups[r['base_task_id']].append(float(r[field]))
    keys=sorted(groups);rng=np.random.default_rng(20261112)
    means=np.array([np.mean(groups[k]) for k in keys]);sums=np.array([sum(groups[k]) for k in keys]);sizes=np.array([len(groups[k]) for k in keys])
    index=rng.integers(len(keys),size=(reps,len(keys)))
    line=sums[index].sum(1)/sizes[index].sum(1);task=means[index].mean(1)
    return dict(rows=len(rows),base_tasks=len(keys),row_mean=sum(sums)/sum(sizes),task_mean=float(means.mean()),
        row_cluster_ci95=np.quantile(line,[.025,.975]).tolist(),task_cluster_ci95=np.quantile(task,[.025,.975]).tolist(),bootstrap_repetitions=reps)


def totals(r):
    h={int(k):v for k,v in r['normal_submissions_histogram'].items()}
    return dict(R=r['forward_count'],O=r['normal_token_count'],S=sum(max(k-1,0)*v for k,v in h.items()),Z=h.get(0,0),expand=r['expand_count'],delete=r['delete_count'])


def paired_replay(rows,a,b):
    aa={r['task_id']:r for r in rows if r['config']==a};bb={r['task_id']:r for r in rows if r['config']==b}
    table=Counter();deltas=[];increases=[];head_dest=Counter();differences=[]
    for case in sorted(aa.keys()&bb.keys()):
        x,y=aa[case],bb[case];tx,ty=totals(x),totals(y)
        same=x['completion_token_ids']==y['completion_token_ids'];d=y['forward_count']-x['forward_count']
        table[('same' if same else 'different')+'_'+('decrease' if d<0 else 'increase' if d>0 else 'equal')]+=1
        delta=dict(task_id=case,base_task_id=x['base_task_id'],delta_pass=int(y['passed'])-int(x['passed']),delta_nfe=d,**{'delta_'+k:ty[k]-tx[k] for k in tx})
        deltas.append(delta)
        if d>0:increases.append(dict(delta,output_equal=same,before=tx,after=ty))
        xv,yv=trace_read(a,case),trace_read(b,case)
        for t,e in events(yv):
            if e['head_called']:head_dest[e['reason']]+=1
        first_behavior=None;first_canvas=None;first_action=None
        for t,u in zip(xv['action_trace'],yv['action_trace']):
            if first_behavior is None and (t['selected']!=u['selected'] or [(e.get('raw'),e.get('final'),e.get('head_called'),e.get('reason')) for e in t['events']]!=[(e.get('raw'),e.get('final'),e.get('head_called'),e.get('reason')) for e in u['events']]):first_behavior=t['round']
            if first_canvas is None and t['x_after']!=u['x_after']:first_canvas=t['round']
            if first_action is None and (t['selected']!=u['selected'] or [action(e) for e in t['events']]!=[action(e) for e in u['events']]):first_action=t['round']
        if first_canvas is None and len(xv['action_trace'])!=len(yv['action_trace']):first_canvas=min(len(xv['action_trace']),len(yv['action_trace']))
        ext=next((t for t in yv['action_trace'] if t['expand']),None)
        before_ext=[(t,e) for t,e in events(yv) if ext and t['round']<=ext['round'] and (t['round']<ext['round'] or e['position']<ext['expand'][0])]
        differences.append(dict(case=case,base_task_id=x['base_task_id'],first_behavior_round=first_behavior,first_actual_action_round=first_action,first_canvas_round=first_canvas,
            first_expansion_round=ext['round'] if ext else None,delete_before_expansion=bool(ext and any(t['delete'] for t in yv['action_trace'][:ext['round']])),
            rejection_before_expansion=any(e.get('rejected') for _,e in before_ext),candidate_order_changed_before_expansion=bool(ext and any(t['selected']!=u['selected'] for t,u in zip(xv['action_trace'][:ext['round']+1],yv['action_trace'][:ext['round']+1])))))
    aggregate={k:sum(r['delta_'+k] for r in deltas) for k in ('R','O','S','Z','expand','delete')}
    assert aggregate['R']==aggregate['O']-aggregate['S']+aggregate['Z']
    return dict(baseline=a,comparison=b,paired_table={k:table.get(k,0) for k in ('same_decrease','same_equal','same_increase','different_decrease','different_equal','different_increase')},
        delta_totals=aggregate,increase_cases=increases,wins=sum(r['delta_pass']>0 for r in deltas),losses=sum(r['delta_pass']<0 for r in deltas),
        quality=clustered(deltas,'delta_pass'),efficiency=clustered(deltas,'delta_nfe'),head_destinations=dict(head_dest),differences=differences)


def fork_pairs(rows):
    groups=defaultdict(dict)
    for r in rows:groups[(r['sample_group'],r['kind'],r['state'],r['continuation_seed'])][r['intervention']]=r
    paired=[]
    for key,g in groups.items():
        contrasts=[('PP','PQ'),('PP','QP'),('PP','QQ'),('PQ','QQ'),('QP','QQ')] if 'PP' in g else [('original',k) for k in g if k!='original']
        for a,b in contrasts:
            if a not in g or b not in g:continue
            x,y=g[a],g[b]
            paired.append(dict(sample_group=key[0],kind=key[1],state=key[2],continuation_seed=key[3],task_id=x['task_id'],base_task_id=x['base_task_id'],
                baseline=a,comparison=b,delta_nfe=y['remaining_forward_count']-x['remaining_forward_count'],delta_pass=int(y['passed'])-int(x['passed']),
                delta_expand=y['remaining_expand_count']-x['remaining_expand_count'],delta_delete=y['remaining_delete_positions']-x['remaining_delete_positions'],
                tokens_changed=x['completion_token_ids']!=y['completion_token_ids'],baseline_passed=x['passed'],comparison_passed=y['passed'],
                baseline_incomplete=x['final_remaining_masks']>0,comparison_incomplete=y['final_remaining_masks']>0))
    return paired


def summarize_pairs(pairs):
    grouped=defaultdict(list)
    for r in pairs:grouped[(r['sample_group'],r['kind'],r['baseline'],r['comparison'])].append(r)
    result=[]
    for key,rows in grouped.items():
        result.append(dict(sample_group=key[0],kind=key[1],baseline=key[2],comparison=key[3],logical_pairs=len(rows),states=len({r['state'] for r in rows}),
            quality=clustered(rows,'delta_pass'),efficiency=clustered(rows,'delta_nfe'),wins=sum(r['delta_pass']>0 for r in rows),losses=sum(r['delta_pass']<0 for r in rows),
            changed_tokens=sum(r['tokens_changed'] for r in rows),incomplete_comparisons=sum(r['comparison_incomplete'] for r in rows)))
    return result


def analyze(db):
    replay=db.rows('replay');forks=db.rows('fork');teacher=db.rows('teacher');multiseed=db.rows('multiseed')
    export('fork_per_case.jsonl.gz',forks);export('teacher_per_state.jsonl.gz',teacher);export('from_scratch_multiseed.jsonl.gz',multiseed)
    comparisons=[paired_replay(replay,'global_k2_nohead_095','global_k4_nohead_095')]
    for t in ('090','095'):comparisons.append(paired_replay(replay,'global_k4_nohead_'+t,'global_k4_trained_'+t))
    atomic_json(OUT/'replay_pairs.json',comparisons)
    pairs=fork_pairs(forks);export('fork_paired_effects.jsonl.gz',pairs)
    summaries=summarize_pairs(pairs)
    atomic_json(OUT/'head_effects.json',[s for s in summaries if s['kind']=='head'])
    atomic_json(OUT/'decoder_effects.json',[s for s in summaries if s['kind']!='head'])
    # Report repetition by state, so path-seeds are never treated as independent tasks.
    by_state=defaultdict(list)
    for p in pairs:by_state[(p['sample_group'],p['kind'],p['state'],p['baseline'],p['comparison'])].append(p)
    stability=[]
    for key,pp in by_state.items():
        stability.append(dict(sample_group=key[0],kind=key[1],state=key[2],baseline=key[3],comparison=key[4],
            seeds=len(pp),nfe_deltas=[p['delta_nfe'] for p in pp],pass_deltas=[p['delta_pass'] for p in pp],tokens_changed=sum(p['tokens_changed'] for p in pp),
            constant_nfe_delta=len({p['delta_nfe'] for p in pp})==1,constant_quality_delta=len({p['delta_pass'] for p in pp})==1))
    atomic_json(OUT/'continuation_stability.json',stability)
    panel_rows=[r for r in forks if r['sample_group']=='panel' and r['intervention']=='PP' and r['continuation_seed']==42]
    teacher_summary=dict(states=len(teacher),kl_closer=sum(r['kl_fresh_q']<r['kl_fresh_p'] for r in teacher),tv_closer=sum(r['tv_fresh_q']<r['tv_fresh_p'] for r in teacher),
        mean_kl_p=float(np.mean([r['kl_fresh_p'] for r in teacher])),mean_kl_q=float(np.mean([r['kl_fresh_q'] for r in teacher])),
        mean_tv_p=float(np.mean([r['tv_fresh_p'] for r in teacher])),mean_tv_q=float(np.mean([r['tv_fresh_q'] for r in teacher])),
        candidate_changed=sum(r['candidate_changed'] for r in panel_rows),score_crossed_same_candidate=sum((r['p_yp']>=r['threshold'])!=(r['q_yp']>=r['threshold']) for r in panel_rows))
    atomic_json(OUT/'conditional_distribution_summary.json',teacher_summary)
    study=['# 焦点案例与质量翻转专项\n','所有轮次/slot均从0开始。单次干预效果不能相加；共同状态由完整控制状态、logits、hidden逐项比对。\n']
    for case in FOCUS:
        study.append('## '+case+'\n')
        study.append('|配置|seed|NFE|头|扩展|删除位置|通过|残mask|\n|---|---:|---:|---:|---:|---:|---|---:|')
        for r in [r for r in replay+multiseed if r['task_id']==case]:study.append(f"|{r['config']}|{r['seed']}|{r['forward_count']}|{r['head_calls']}|{r['expand_count']}|{r['delete_count']}|{r['passed']}|{r['final_remaining_masks']}|")
        study.append('\n干预结果（seed42；其它配对延续见fork_per_case与continuation_stability）：\n')
        study.append('|分组/状态|干预|余NFE|余头|余扩展|余删除位置|通过|残mask|\n|---|---|---:|---:|---:|---:|---|---:|')
        for r in forks:
            if r['task_id']==case and r['sample_group']!='panel' and r['continuation_seed']==42:study.append(f"|{r['sample_group']}/{r['kind']}/r{r['intervention_round']}s{r['intervention_slot']}|{r['intervention']}|{r['remaining_forward_count']}|{r['remaining_head_calls']}|{r['remaining_expand_count']}|{r['remaining_delete_positions']}|{r['passed']}|{r['final_remaining_masks']}|")
        study.append('')
    manifest=json.loads((OUT/'special_state_manifest.json').read_text())
    study.append('## K4 .90全部质量翻转\n')
    for f in manifest['quality_flips']:
        case=f['case'];study.append(f"\n{case}：无头{f['nohead_passed']}→有头{f['trained_passed']}。\n")
        for cid in ('global_k4_nohead_090','global_k4_trained_090'):
            v=trace_read(cid,case);study.append(cid+'：')
            for t in v['action_trace']:
                study.append(f"- r{t['round']}：选{t['selected']}，普通提交{t['normal_submissions']}，扩展{t['expand']}，删除{t['delete']}；"+'；'.join(f"s{e['slot']} {e.get('raw')}→{e.get('final')} {e['reason']} p={e.get('candidate_probability')}" for e in t['events']))
    study.append('\n## 不适用\n\n'+json.dumps(manifest['not_applicable'],ensure_ascii=False,indent=2))
    (OUT/'case_studies.zh.md').write_text('\n'.join(study)+'\n')
    report=['# 阈值轨迹回放与受控分叉诊断\n',
        f"完成主轨迹{sum(r['config'] in IDS for r in replay)}/5165，焦点补充{sum(r['config'] not in IDS for r in replay)}/3；分叉{len(forks)}，教师{len(teacher)}/64，从头多种子{len(multiseed)}/54。",
        '\n来源285f8f1；历史执行660f296。保持原bf16模型、float32旧rank256 KL/full/best头、lambda1、seed42键、T=.2/top_p=.9、合法全词表T1阈值、初始/扩展64、总256轮和官方3秒评分。单模型共享H200，详细日志耗时不用于加速主张。',
        '\n数据是反复使用的1033行/164基础题开发诊断人口。面板与专项分开汇总；8延续种子是给定固定历史候选的路径总效应，不是8次从头生成或8道独立题。',
        '\n## 全量复现与分解\n']
    for c in comparisons:
        report.append(f"\n{c['comparison']} 相对 {c['baseline']}：{json.dumps({k:c[k] for k in ('paired_table','delta_totals','wins','losses','quality','efficiency','head_destinations')},ensure_ascii=False)}\n")
    report.extend(['\nR=O−S+Z逐例与总和均检查；K4/K2同时报告ΔO，不能只解释S/Z。首行为差异、首实际动作差异、首画布差异和扩展前删/拒绝/顺序变化保存在replay_pairs.json，首差异本身不作为因果归因。',
        '\n## 四组单次头干预\n',
        'PP原候选原评分；PQ固定原候选改评分；QP改候选原评分；QQ同时改候选和评分。头提出结构词仅刷新。以下均为面板条件效应，不估计全量通过率或总体加速。\n'])
    for s in summaries:report.append(json.dumps(s,ensure_ascii=False)+'\n')
    report.extend(['\n## 条件分布\n',json.dumps(teacher_summary,ensure_ascii=False),
        '\n教师只写同链合法已提交普通前驱，T1合法全词表比较；更接近fresh不等于代码正确或概率校准。',
        '\n## 限制与后续决策\n',
        '当前证据需结合单次分叉、质量和多种子共同判断。处理顺序/豁免的2×2只分离这两个局部因素，不分解全部K2/K4差异。延后结构动作需要下轮重新判断，否决只屏蔽当前结构词，均不构成可部署控制器。普通失败和3秒超时不补抽；完全相同case/output在同环境复用评分，复用记录逐逻辑分支保留。',
        '支持训练的证据应是修正更接近fresh且跨种子有质量收益；反对证据包括固定候选评分已足以改变路径、头候选改坏或头结构屏障增成本。支持概率校准对照的是同词跨阈值而非已经校准；支持解码规则调整的是一次规则干预改变路径且质量不受损，反对证据是收益依赖特定seed；支持结构研究的是早期删除/扩展对长路径有可干预影响，反对证据是延后/否决留下mask或损害通过。下一步建议由以下实测病例结论补充，不自动启动任何新实验。',
        'reviewer_gate_disabled：本地diff review + CPU/GPU逐轮/恢复真实性验证，无独立reviewer。单seed历史精确复现、多种子延续和从头生成分别记录；不是独立测试。\n',
        '\n完整逐例、逻辑分叉、候选池、状态清单、评分复用和成本分别见同目录索引；原始完整轨迹/快照/SQLite留local，不进普通git。'])
    (OUT/'report.zh.md').write_text('\n'.join(report)+'\n')
    cost=dict(prefix_replay=db.rows('capture_cost'),shadow=db.rows('shadow_cost'),teacher_forwards=len(teacher),
        fork_actual_forwards=sum(r['actual_forward_count'] for r in forks),fork_actual_head_calls=sum(r['actual_head_calls'] for r in forks),
        replay_forwards=sum(r['forward_count'] for r in replay),logical_forks=len(forks),scores_executed=db.db.execute('SELECT COUNT(*) FROM scores').fetchone()[0])
    atomic_json(OUT/'runtime_cost.json',cost)

if __name__=='__main__':
    from experiments.threshold_trace_fork.run import Ledger
    analyze(Ledger())
