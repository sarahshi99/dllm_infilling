#!/usr/bin/env python3
"""Paired development-population evidence; never interpret shared cost as speed."""
from __future__ import annotations
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from analysis.analyze_dreamon_markov_k2_diagnosis import clustered_difference, write_csv, write_json
from experiments.dreamon_singleline_adapter import derive_row_seed
POLICIES=('global_confidence','left_to_right_frontier')
PAIRS=(('D','C'),('C','B'),('D','B'),('B','A'),('C','A'),('D','A'))


def analyze(out):
    manifest=json.loads((out/'run_manifest.json').read_text())
    with gzip.open(out/'per_case.jsonl.gz','rt') as f: rows=[json.loads(line) for line in f]
    # Original population membership is checked by runner; expected IDs also come from its source config table.
    import csv
    original=json.loads((REPO/'analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/run_manifest.json').read_text())
    with (Path(original['original_config_path']).parent/'paired_deltas.csv').open() as f: ids={r['task_id'] for r in csv.DictReader(f)}
    counts=Counter((r['target'],r['policy'],r['arm'],r['task_id'],r['seed']) for r in rows)
    expected={(manifest['target'],p,a,t,42) for p in POLICIES for a in 'ABCD' for t in ids}
    errors=[]
    if set(counts)!=expected: errors.append('missing or unexpected rows')
    if any(v!=1 for v in counts.values()): errors.append('duplicate rows')
    for r in rows:
        if r['row_seed']!=derive_row_seed(42,r['task_id']) or r['base_task_id']!=r['task_id'].split('/')[2]: errors.append('seed or cluster mismatch')
        if r['expand_count'] or r['delete_count'] or r['final_remaining_masks'] or r['final_dynamic_length']!=64: errors.append('fixed canvas violated')
        if r['markov_enabled_count']>r['adjacent_opportunity_count']: errors.append('trigger denominator mismatch')
    smoke=json.loads((out/'smoke_checks.json').read_text()) if (out/'smoke_checks.json').exists() else {'status':'not_run'}
    if smoke['status']!='passed': errors.append('technical checks not passed')
    failure_path=out/'failure_journal.jsonl'
    failures=[json.loads(l) for l in failure_path.read_text().splitlines()] if failure_path.exists() else []
    failed_keys={(r['target'],r['policy'],r['arm'],r['task_id'],r['seed']) for r in failures}
    audit={'status':'passed' if not errors else 'incomplete_or_invalid','errors':sorted(set(errors)),'rows':len(rows),'expected':8264,'unique':len(counts),
           'missing':len(expected-set(counts)),'duplicates':sum(v-1 for v in counts.values()),'historical_failure_events':len(failures),
           'unrecovered_failures':len(failed_keys-set(counts)),'technical_check':smoke['status'],'base_task_count':len({r['base_task_id'] for r in rows})}
    write_json(out/'integrity_audit.json',audit)
    if errors: return audit
    summary=[]; paired=[]
    for p in POLICIES:
        subset=[r for r in rows if r['policy']==p]
        by_id={(r['arm'],r['task_id']):r for r in subset}
        for a in 'ABCD':
            arm=[r for r in subset if r['arm']==a]
            groups=defaultdict(list)
            for r in arm: groups[r['base_task_id']].append(int(r['passed']))
            entry={'target':manifest['target'],'policy':p,'arm':a,'passed':sum(r['passed'] for r in arm),'denominator':1033,
                   'pass_rate':float(np.mean([r['passed'] for r in arm])),'base_task_equal_pass_rate':float(np.mean([np.mean(v) for v in groups.values()])),
                   'test_failures':sum(not r['passed'] for r in arm),'evaluation_timeouts':sum('timed out' in r['failure_reason'].lower() for r in arm),
                   'generation_cost_sec':sum(r['generation_sec'] for r in arm),'evaluator_cost_sec':sum(r['evaluator_sec'] for r in arm)}
            for key in ('forward_count','adjacent_opportunity_count','nonadjacent_fallback_count','markov_enabled_count','structural_stop_count','corrected_structural_stop_count','special_output_count'):
                entry[key]=sum(r.get(key,0) for r in arm)
            opportunities=entry['adjacent_opportunity_count']
            pair_rounds=opportunities+entry['nonadjacent_fallback_count']
            entry['head_trigger_rate_per_adjacent_opportunity']=entry['markov_enabled_count']/opportunities if opportunities else None
            entry['nonadjacent_fallback_rate_per_pair_round']=entry['nonadjacent_fallback_count']/pair_rounds if pair_rounds else None
            entry['structural_action_opportunity_count']=0
            entry['structural_stop_rate']=None  # Base has no structural actions; denominator is zero.
            summary.append(entry)
        for left,right in PAIRS:
            help_count=sum(by_id[left,t]['passed'] and not by_id[right,t]['passed'] for t in ids)
            harm_count=sum(by_id[right,t]['passed'] and not by_id[left,t]['passed'] for t in ids)
            paired.append({'target':manifest['target'],'policy':p,'comparison':left+'-'+right,'help':help_count,'harm':harm_count,'net':help_count-harm_count,**clustered_difference(subset,left,right)})
    write_csv(out/'summary.csv',summary); write_csv(out/'paired_comparisons.csv',paired)
    lines=['# '+manifest['target']+' 固定64辅助迁移','',
           '原始Base辅助行，不能填补缺失DreamOn微调权重；1033行/164基础题重复开发人口、单生成种子42。共享GPU耗时仅成本，没有正式加速比。','',
           'A=K1无头，B=K2无头，C=K2同类同秩零输出头λ1，D=K2来源v2 KL头λ1。固定画布64，官方原生参考使用明确登记的K/顺序最小补丁。','',
           '|顺序|A|B|C|D|','|---|---|---|---|---|']
    for p in POLICIES:
        lines.append('|'+p+'|'+'|'.join(str(next(r['passed'] for r in summary if r['policy']==p and r['arm']==a))+'/1033' for a in 'ABCD')+'|')
    lines+=['','|顺序|配对|帮助/伤害|净行数|行加权Δ95%CI(pp)|基础题等权Δ95%CI(pp)|','|---|---|---|---|---|---|']
    for r in paired:
        lines.append(f"|{r['policy']}|{r['comparison']}|{r['help']}/{r['harm']}|{r['net']:+d}|{100*r['line_delta']:+.2f} [{100*r['line_ci95_low']:+.2f},{100*r['line_ci95_high']:+.2f}]|{100*r['base_task_equal_delta']:+.2f} [{100*r['base_task_equal_ci95_low']:+.2f},{100*r['base_task_equal_ci95_high']:+.2f}]|")
    lines+=['','D−C用于训练贡献；C−B用于路径贡献；D−B为组合效应。只优于B、不优于C不能归因为学习。未增益说明现有权重迁移未成功，不能自动否定以后逐模型重训；固定长度失败也可能由长度误差主导。','',
            '触发/回退/结构动作分母见summary.csv；Base结构动作禁用，结构停止计数为0，不赋EOS扩删语义。失败/超时和历史未恢复错误见完整性审计。命令、模型/代码/头版本与成本见run_manifest和controller/gpu_budget.json。']
    (out/'report.zh.md').write_text('\n'.join(lines)+'\n')
    return audit

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--result-dir',type=Path,required=True)
    print(json.dumps(analyze(parser.parse_args().result_dir)))
