#!/usr/bin/env python3
"""Portable v2 merge, repair attribution, fixed population strata and cluster CI."""
from pathlib import Path
import collections,csv,json,sys,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from experiments.structure_coordinate_common import OUT,OLD,STAGES,influence,old_rows,read_jsonl,write_gzip
from experiments.structure_coordinate_common import atomic_write_json


def csvout(path,rows):
    with path.open('w') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bootstrap_pair(left,right):
    ids=sorted(set(left)&set(right));groups=sorted({t.split('/')[2] for t in ids})
    diffs=np.array([sum(int(left[t]['passed'])-int(right[t]['passed']) for t in ids if t.split('/')[2]==g) for g in groups],dtype=float)
    sizes=np.array([sum(t.split('/')[2]==g for t in ids) for g in groups],dtype=float)
    selected=np.random.default_rng(20261013).integers(0,len(groups),size=(10000,len(groups)))
    line=diffs[selected].sum(1)/sizes[selected].sum(1);macro=(diffs/sizes)[selected].mean(1)
    return dict(cluster_count=len(groups),bootstrap_replicates=10000,bootstrap_seed=20261013,
                line_delta_pp=100*diffs.sum()/sizes.sum(),line_ci95_low_pp=100*np.quantile(line,.025),line_ci95_high_pp=100*np.quantile(line,.975),
                base_task_equal_delta_pp=100*(diffs/sizes).mean(),base_task_equal_ci95_low_pp=100*np.quantile(macro,.025),base_task_equal_ci95_high_pp=100*np.quantile(macro,.975))


def pair(left,right,**labels):
    assert set(left)==set(right)
    ids=sorted(left)
    helped=[t for t in ids if left[t]['passed'] and not right[t]['passed']]
    harmed=[t for t in ids if right[t]['passed'] and not left[t]['passed']]
    return dict(**labels,N=len(ids),help=len(helped),harm=len(harmed),net=len(helped)-len(harmed),
                help_ids=json.dumps(helped),harm_ids=json.dumps(harmed),**bootstrap_pair(left,right))


def recovery(trained,baseline,k1):
    ids=sorted(trained);assert set(ids)==set(baseline)==set(k1)
    groups=sorted({t.split('/')[2] for t in ids})
    numerator=np.array([sum(int(trained[t]['passed'])-int(baseline[t]['passed']) for t in ids if t.split('/')[2]==g) for g in groups],dtype=float)
    denominator=np.array([sum(int(k1[t]['passed'])-int(baseline[t]['passed']) for t in ids if t.split('/')[2]==g) for g in groups],dtype=float)
    sizes=np.array([sum(t.split('/')[2]==g for t in ids) for g in groups],dtype=float)
    selected=np.random.default_rng(20261013).integers(0,len(groups),size=(10000,len(groups)))
    result={}
    for name,num,den in [('line',numerator,denominator),('base_task_equal',numerator/sizes,denominator/sizes)]:
        point=100*num.sum()/den.sum() if den.sum()>0 else None
        dn=den[selected].sum(1);nn=num[selected].sum(1);valid=dn>0
        samples=100*nn[valid]/dn[valid]
        result[name+'_recovery_pct']=point
        result[name+'_ci95_low_pct']=float(np.quantile(samples,.025)) if point is not None and len(samples) else None
        result[name+'_ci95_high_pct']=float(np.quantile(samples,.975)) if point is not None and len(samples) else None
        result[name+'_bootstrap_positive_denominator']=int(valid.sum())
    return result


def span(task):
    bounds=task.rsplit('/',1)[1].split('_')
    return int(bounds[1][1:])-int(bounds[0][1:])+1


def analyze():
    manifest=influence();proof={r['stable_id']:r for r in manifest['rows']};old=old_rows()
    generated=read_jsonl(OUT/'new_rows.jsonl.gz') if (OUT/'new_rows.jsonl.gz').exists() else []
    new={f"{r['stage']}::{r['policy']}::{r['arm']}::{r['task_id']}":r for r in generated}
    assert len(new)==len(generated),'duplicate new key'
    expected={k for k,r in proof.items() if r['regenerate']}
    assert set(new)<=expected
    missing=expected-{k for k,r in new.items() if type(r.get('passed')) is bool}
    if missing:
        atomic_write_json(OUT/'completion_matrix.json',dict(status='incomplete',expected_new=len(expected),completed_new=len(expected)-len(missing),missing_ids=sorted(missing)))
        return False
    merged={};changes=[];matrix={}
    for key,source in old.items():
        p=proof[key]
        if p['regenerate']:
            result=new[key];assert result['row_seed']==source['row_seed'] and result['seed']==42
            changes.append(dict(stable_id=key,stage=p['stage'],policy=p['policy'],arm=p['arm'],task_id=p['task_id'],old_passed=source['passed'],new_passed=result['passed'],
                                old_result=source['failure_reason'],new_result=result['failure_reason'],first_step=p['first_step'],token_ids_changed=source['completion_token_ids']!=result['completion_token_ids']))
            merged[key]=dict(result,v2_provenance='regenerated',v1_source_sha256=p['source_sha256'])
        else:
            assert p['status']=='proved_equivalent'
            merged[key]=dict(source,v2_provenance='proved_v1_reuse',influence_id=key,v1_source_sha256=p['source_sha256'])
    assert set(merged)==set(proof) and len(merged)==37722
    csvout(OUT/'repair_score_changes.csv',changes)
    for stage in 'ABC':
        rows=[r for k,r in merged.items() if r['stage']==stage]
        ids={r['task_id'] for r in read_jsonl(OLD/('multi-line_contexts.jsonl.gz' if stage=='C' else 'single-line_contexts.jsonl.gz'))}
        for p in STAGES[stage]['policies']:
            for a in STAGES[stage]['arms']:
                subset=[r for r in rows if (r['policy'],r['arm'])==(p,a)]
                assert len(subset)==len(ids) and {r['task_id'] for r in subset}==ids
        out=OUT/stage;out.mkdir(exist_ok=True)
        write_gzip(out/'merged_per_case_v2.jsonl.gz',rows)
        matrix[stage]=dict(status='complete',logical_rows=len(rows),new_generated=sum(r['v2_provenance']=='regenerated' for r in rows),proved_reused=sum(r['v2_provenance']=='proved_v1_reuse' for r in rows),missing=0,duplicates=0)
    pairs=[];percentages=[];ratios=[];strata_rows=[];repair_pairs=[]
    old_anchors={stage:read_jsonl(OLD/('dreamon_old_anchors.jsonl.gz' if stage=='A' else 'dreamcoder_base_old_anchors.jsonl.gz')) for stage in ['A','B']}
    for stage in 'ABC':
        policies=STAGES[stage]['policies']
        subsets=['all'] if stage!='C' else ['all','single_span_1','cross_line_2plus','span_2_3','span_4_7','span_8plus']
        def included(task,label):
            if label=='all':return True
            length=span(task)
            return {'single_span_1':length==1,'cross_line_2plus':length>=2,'span_2_3':2<=length<=3,'span_4_7':4<=length<=7,'span_8plus':length>=8}[label]
        for policy in policies:
            for subset in subsets:
                index={a:{r['task_id']:r for r in merged.values() if r['stage']==stage and r['policy']==policy and r['arm']==a and included(r['task_id'],subset)} for a in STAGES[stage]['arms']}
                before={a:{r['task_id']:r for r in old.values() if r['stage']==stage and r['policy']==policy and r['arm']==a and included(r['task_id'],subset)} for a in STAGES[stage]['arms']}
                n=len(index['D']);assert n>0
                if stage=='C':
                    if subset=='all':assert n==5815
                    if subset=='single_span_1':assert n==1033
                    if subset=='cross_line_2plus':assert n==4782
                    k1=index['A'];nohead=index['B']
                    comparisons=[('B','A'),('C','B'),('D','C'),('D','B'),('D','A')]
                else:
                    k1={r['task_id']:r for r in old_anchors[stage] if r['policy']==policy and r['arm']=='A'};nohead=index['A']
                    comparisons=[('B','A'),('C','B'),('D','C'),('D','A'),('D','B')] if stage=='A' else [('C','A'),('D','C'),('D','A')]
                overall=recovery(index['D'],nohead,k1);pure=recovery(index['D'],index['C'],k1)
                for kind,ratio in [('overall',overall),('pure_learning',pure)]:ratios.append(dict(stage=stage,policy=policy,population=subset,kind=kind,N=n,**ratio))
                for a,group in index.items():
                    passed=sum(r['passed'] for r in group.values())
                    percentages.append(dict(stage=stage,policy=policy,population=subset,arm=a,N=n,base_tasks=len({t.split('/')[2] for t in group}),passed=passed,pass_rate_pct=100*passed/n,
                        versus_nohead_pp=100*(passed-sum(r['passed'] for r in nohead.values()))/n,
                        versus_zero_pp=100*(passed-sum(r['passed'] for r in index['C'].values()))/n,
                        versus_K1_pp=100*(passed-sum(r['passed'] for r in k1.values()))/n,
                        overall_recovery_pct=overall['line_recovery_pct'] if a=='D' else None,pure_learning_recovery_pct=pure['line_recovery_pct'] if a=='D' else None,
                        regenerated=sum(r['v2_provenance']=='regenerated' for r in group.values()),proved_reused=sum(r['v2_provenance']=='proved_v1_reuse' for r in group.values()),
                        actual_submitted_per_forward=sum(t['actual_submitted_count'] for r in group.values() for t in r['action_trace'])/sum(r['forward_count'] for r in group.values()),NFE_mean=np.mean([r['forward_count'] for r in group.values()])))
                    repair_pairs.append(pair(group,before[a],stage=stage,policy=policy,population=subset,comparison=a+'_v2-v1'))
                for left,right in comparisons:pairs.append(pair(index[left],index[right],stage=stage,policy=policy,population=subset,comparison=left+'-'+right))
                if stage!='C':pairs.append(pair(index['D'],k1,stage=stage,policy=policy,population=subset,comparison='D-old_K1'))
                strata_rows.append(dict(stage=stage,policy=policy,population=subset,N=n,base_tasks=len({t.split('/')[2] for t in index['D']})))
    csvout(OUT/'percentage_table.csv',percentages);csvout(OUT/'paired_comparisons.csv',pairs)
    csvout(OUT/'repair_paired_comparisons.csv',repair_pairs);csvout(OUT/'recovery_bootstrap.csv',ratios);csvout(OUT/'population_strata.csv',strata_rows)
    # Subset generation time only: never mix this with reused full-population timing.
    times=[]
    for group in sorted({(r['stage'],r['arm']) for r in generated}):
        rows=[r for r in generated if (r['stage'],r['arm'])==group]
        times.append(dict(stage=group[0],arm=group[1],regenerated_N=len(rows),observed_subset_mean_generation_sec=np.mean([r['generation_sec'] for r in rows]),head_serial_cuda_sec=np.mean([r['head_serial_cuda_sec'] for r in rows]),scope='shared GPU affected subset only; not full population timing or fair speed ratio'))
    csvout(OUT/'observed_subset_timing.csv',times)
    helped=sum(not r['old_passed'] and r['new_passed'] for r in changes);harmed=sum(r['old_passed'] and not r['new_passed'] for r in changes)
    manifest_run=json.loads((OUT/'run_manifest.json').read_text())
    matrix.update(status='complete',logical_rows=37722,new_generated=len(generated),proved_reused=37722-len(generated),score_help=helped,score_harm=harmed,score_changed=helped+harmed,fix_commit=manifest_run['fix_commit'],new_rows_sha256=hashlib.sha256((OUT/'new_rows.jsonl.gz').read_bytes()).hexdigest())
    atomic_write_json(OUT/'completion_matrix.json',matrix)
    lines=['# 2026-10-15 并行扩删坐标修补 v2 完成','',f'基线dbdd0bd；修正提交{manifest_run["fix_commit"]}。全37722逻辑行，重生成{len(generated)}、全程等价复用{37722-len(generated)}；缺失/重复0。修补本身救回{helped}、改错{harmed}，共{helped+harmed}个评分改变。旧v1输出/报告保持不变。','',
           '|批次/策略/人口|组|通过/总数|通过率|头−无头pp|头−零头pp|总体恢复|纯学习恢复|相对K1pp|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '不适用' if v is None else f'{v:.2f}'
    for r in percentages:
        if r['arm']=='D':lines.append(f'|{r["stage"]}/{r["policy"]}/{r["population"]}|D|{r["passed"]}/{r["N"]}|{r["pass_rate_pct"]:.2f}%|{r["versus_nohead_pp"]:+.2f}|{r["versus_zero_pp"]:+.2f}|{fmt(r["overall_recovery_pct"])}%|{fmt(r["pure_learning_recovery_pct"])}%|{r["versus_K1_pp"]:+.2f}|')
    lines+=['','A总体=(D4−原生A4)/(K1−原生A4)，链B4贡献另列；C总体=(D−链无头B)/(K1−链无头B)，纯学习均=(D−C零头)/(K1−C零头)。两种分母不同，不直接相加；分母非正不适用。多行B/C均为sequential_pair路径，C额外零头调用/局部重采样，二者没有结构规则差别。', '',
       '配对和区间见paired_comparisons.csv、repair_paired_comparisons.csv、recovery_bootstrap.csv。沿用10000次、seed20261013、基础题整组bootstrap，行加权及基础题等权；条件于单生成seed42和重复开发164基础题，不是独立测试泛化。恢复比例bootstrap只对正分母重采样取百分位，保留有效次数，不能忽略不稳定分母。', '',
       '人口按官方代码区间端点跨度：全5815、单行跨度1033、跨行4782，以及预定跨行2–3/4–7/至少8行；不是生成长度，各层基础题构成不同。旧global K2锚点1012条无扩展子集及旧单行K2结果仅历史参照，未扩大补跑或冒称修复版同协议。', '',
       '修补与学习贡献分开。K4链D−C经全程等价证明不受本问题影响；多行修正后学习净增是否维持见下方。EOS残留不是全部错误的根因证据，修正分布重算不等于答案必然正确。']
    for r in pairs:
        if r['comparison']=='D-C' and (r['stage']!='C' or r['population'] in ['all','cross_line_2plus']):
            lines.append(f'{r["stage"]}/{r["policy"]}/{r["population"]} D−C：救回{r["help"]}、改错{r["harm"]}、净{r["net"]:+d}；行加权差{r["line_delta_pp"]:+.2f}pp，基础题bootstrap95%区间[{r["line_ci95_low_pp"]:+.2f},{r["line_ci95_high_pp"]:+.2f}]。')
    c=next(r for r in percentages if (r['stage'],r['population'],r['arm'])==('C','all','D'))
    net=next(r['net'] for r in pairs if (r['stage'],r['population'],r['comparison'])==('C','all','D-C'))
    lines += ['', f'判断：多行原v1头−零头净+27，修正后净{net:+d}。'+('学习增益方向保留。' if net>0 else '原多行正学习增益结论须更正。')+('训练头仍未追平K1，部分补偿结论保留。' if c['versus_K1_pp']<0 else '训练头相对K1结论需更正。')+'K4链学习配对保持；原生K4对照及其总体恢复比例以v2重算为准。此坐标补验到此关闭，不继续无边界排查。', '',
       '仅126受影响配置行按原seed恢复生成与官方3秒隔离评分；技术自检行并入最终表不重复生成。未改主模型/头/词表/采样/预算/广播/提交顺序，无新头、common-u、训练、参数搜索、思维导图更新。四个上下文新头方向仍为下阶段待办，共同训练状态/预算与旧结构同库重训对照留待用户决定。', '',
       '耗时只见observed_subset_timing.csv，为共享H200补验子集观测，不能当全人口速度或与旧计时混合计算公平加速。离散轨迹、官方原评分、CPU断言、影响清单、首分歧局部画布、复用证明和失败/重试覆盖独立保存。']
    (OUT/'report.zh.md').write_text('\n'.join(lines)+'\n')
    return True

if __name__=='__main__':sys.exit(0 if analyze() else 2)
