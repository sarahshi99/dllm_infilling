#!/usr/bin/env python3
"""Append-date delivery node from preserved evidence and current target states."""
import csv
import gzip
import json
from pathlib import Path
import subprocess
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'analysis_outputs/zero_training_transfer_20261013'

def read(path): return json.loads(path.read_text())

def main():
    old=read(ROOT/'a_dreamcoder/dreamon_reuse_audit.json')
    with gzip.open(ROOT/'a_dreamcoder/dreamon_reused_per_case.jsonl.gz','rt') as f:
        calibration=[json.loads(line) for line in f]
    keys={(r['policy'],r['arm'],r['task_id'],r['seed']) for r in calibration}
    if len(calibration)!=10330 or len(keys)!=10330: raise RuntimeError('preserved diagnosis not complete/unique')
    if len({r['base_task_id'] for r in calibration})!=164: raise RuntimeError('wrong preserved clusters')
    calibration_summary=[]
    for policy in ('global_confidence','left_to_right_frontier'):
        for arm in 'ABCD':
            subset=[r for r in calibration if r['policy']==policy and r['arm']==arm]
            totals={k:sum(r[k] for r in subset) for k in ('outer_round_count','adjacent_opportunity_count','nonadjacent_fallback_count','markov_enabled_count','structural_stop_count','corrected_structural_stop_count')}
            pair_denominator=totals['adjacent_opportunity_count']+totals['nonadjacent_fallback_count']
            calibration_summary.append({'policy':policy,'arm':arm,'passed':sum(r['passed'] for r in subset),'denominator':1033,
                'quality_failures':sum(not r['passed'] for r in subset),
                'evaluation_timeouts':sum('timed out' in r['failure_reason'].lower() or 'timeout' in r['failure_reason'].lower() for r in subset),
                'technical_unrecovered_failures':0, **totals,
                'head_trigger_rate_per_adjacent_opportunity':totals['markov_enabled_count']/totals['adjacent_opportunity_count'] if totals['adjacent_opportunity_count'] else None,
                'nonadjacent_fallback_pair_denominator':pair_denominator,
                'nonadjacent_fallback_rate':totals['nonadjacent_fallback_count']/pair_denominator if pair_denominator else None,
                'structural_stop_round_denominator':totals['outer_round_count'],
                'structural_stop_rate_per_outer_round':totals['structural_stop_count']/totals['outer_round_count'],
                'corrected_structural_stop_head_denominator':totals['markov_enabled_count'],
                'generation_cost_sec':sum(r['generation_sec'] for r in subset),'formal_speedup':None})
    with (ROOT/'controller/dreamon_calibration_summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(calibration_summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(calibration_summary)
    registry=read(ROOT/'controller/target_registry.json')
    budget=read(ROOT/'controller/gpu_budget.json')
    source=[{'start':'DreamCoder-7B','weight_kind':'exact_dreamon_finetuned','id':'Dream-org/DreamOn-v0-7B','revision':'8ccc74750e43177327f29dab9e91882ba759e194',
             'status':'reused_complete_calibration','rows':8264,'cross_model':False,'exact_initial_base_revision':'not publicly specified','source_registry':'a_dreamcoder/source_registry.json'},
            {'start':'Dream-7B','weight_kind':'exact_dreamon_finetuned','id':None,'revision':None,'status':'未找到可验证公开权重','rows':None,'cross_model':None,'source_registry':'b_dream/source_registry.json'},
            {'start':'DiffuCoder-7B','weight_kind':'exact_dreamon_finetuned','id':None,'revision':None,'status':'未找到可验证公开权重','rows':None,'cross_model':None,'source_registry':'c_diffucoder/source_registry.json'}]
    auxiliary=[]
    for target,entry in registry.items():
        path=ROOT/target
        manifest=read(path/'run_manifest.json') if (path/'run_manifest.json').exists() else {}
        audit=read(path/'integrity_audit.json') if (path/'integrity_audit.json').exists() else {}
        progress=read(path/'progress.json') if (path/'progress.json').exists() else {}
        auxiliary.append({'target':target,'id':entry['id'],'revision':entry['revision'],'weight_kind':'original_base_fixed_length_auxiliary',
                          'status':manifest.get('status','blocked_weights_not_ready'),'blocker':manifest.get('blocker') if manifest else 'weight download blocked after bounded attempts; GPU16/full not run',
                          'rows':audit.get('rows',progress.get('completed',0)),'expected_rows':8264,'integrity_status':audit.get('status','not_complete'),
                          'local_path':entry['local_path'],'direct_transfer_scope':entry.get('validation_scope')})
    result={'date':'2026-10-13','verification_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'source_code_commit':'8a0a54918412eda9402a327646f7f067f7160ec8','exact_dreamon_matrix':source,'base_auxiliary_matrix':auxiliary,
            'preserved_diagnosis_rows':len(calibration),'calibration_rows_in_requested_ABCD':sum(r['arm'] in 'ABCD' for r in calibration),
            'gpu_job_cost_hours_this_round':budget['used_gpu_job_sec']/3600,'budget_hours_this_round':budget['limit_gpu_job_sec']/3600,
            'historical_diagnosis_cost_hours':budget['prior_diagnosis_sec']/3600,'formal_speedup':None,'training_executed':False,'new_sealed_test_access':False}
    (ROOT/'controller/delivery_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 2026-10-13 三起点来源与现有头零训练迁移','',
           '主线需要三个起点分别经过DreamOn长度微调后的权重。本轮冻结v2 KL best rank256/λ1，仅检验现有头直迁；没有训练。','',
           '|确切DreamOn微调权重|模型及revision|状态|','|---|---|---|',
           '|DreamCoder起点|Dream-org/DreamOn-v0-7B @ 8ccc74750e43177327f29dab9e91882ba759e194|复用完整自身校准；不是跨模型成功|',
           '|Dream起点|缺失|未找到可验证公开权重|','|DiffuCoder起点|缺失|未找到可验证公开权重|','',
           '来源边界：固定[官方代码8a0a549](https://github.com/DreamLM/DreamOn/tree/8a0a54918412eda9402a327646f7f067f7160ec8)及[正式ICLR论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/7c020dd308d7413b5078686723b2f857-Paper-Conference.pdf)；[issue6](https://github.com/DreamLM/DreamOn/issues/6)的发布承诺不能当下载地址。v0历史提交、量化及第三方Java衍生不补三底座。官方run脚本明确DreamCoder Base身份，但未披露发布权重实际初始化Base revision。','',
           '|原始Base固定64辅助|revision|当前状态|完整行数|','|---|---|---|---|']
    for row in auxiliary:
        lines.append(f"|{row['id']}|{row['revision']}|{row['status']}|{row['rows']}/8264|")
    lines+=['','完整普通token→ID、分词规则/文本、特殊词/输出列及一次logits shift审计已存各智能体目录。来源头按原定义strict加载全部参数；零头deepcopy且仅清零output.weight，λ仍1。未命名Base列不赋扩删语义，实际非法token会停目标。','',
            'Base原生日程不是严格每轮K；局部参考只补min(K,remaining)与frontier选位，固定64且不读答案长度，EOS留原位，所有组相同截取/skip special规则。现有DreamOn max_gen_len64与论文128分别登记，不能称同一复现。','',
            '人口固定1033行/164基础题、单生成种子42、原derive_row_seed；16条字典序等距ID不按正确率挑选。A=K1无头，B=K2无头，C=K2同类同秩零输出头λ1，D=K2来源头λ1。','',
            '|DreamOn自身校准顺序|A|B|C|D|D−C帮助/伤害|C−B帮助/伤害|D−B净|D−A净|','|---|---|---|---|---|---|---|---|---|',
            '|global_confidence|951|901|901|910|15/6|0/0|+9|−41|',
            '|left_to_right_frontier|942|877|872|890|29/11|6/11|+13|−52|','',
            '以上分母均1033；来源校准D−C净+9/+18。global行加权区间含0，frontier基础题等权区间跨0；单种子重复开发人口，不能推广独立测试或跨模型成功。完整D−C/C−B/D−B和相对K1的帮助/伤害、164题cluster 95%CI见a_dreamcoder/dreamon_reuse_paired.json；触发率、结构停止/总轮次、非相邻回退/双位机会和失败/超时见controller/dreamon_calibration_summary.csv；保全10330行含历史E机制组，无重跑、无改写旧报告。','',
            'Base若完整，逐目标report.zh.md/paired_comparisons.csv给出六个配对和两种权重cluster区间，summary.csv给触发、结构/非相邻回退分母、失败/超时。兼容但无增益只表示现有头本次迁移未成功；只优于无头但不优于零头不能归因为学习；固定长度失败可能受长度误差主导，不自动否定逐模型重训。','',
            f"本轮累计GPU作业成本 {result['gpu_job_cost_hours_this_round']:.6f}h / {result['budget_hours_this_round']:.1f}h；另存历史诊断成本 {result['historical_diagnosis_cost_hours']:.6f}h。共享H200计时只作成本，没有正式加速比。互斥锁和所有session在controller/gpu_budget.json，GPU背景/命令/版本/PID/恢复状态在目标manifest及controller清单。下载不计GPU。",'',
            '后续仅准备材料：future_training_preparation.md列索取信息、论文/代码差异、选择10epochs重建、有效batch128梯度累积、显存理论范围及以后授权pilot估时方法。没有启动任何主模型微调、头训练、K4或隐藏状态头。','',
            '命令：scripts/run_zero_training_transfer_dreamcoder_20261013.sh；通用runner支持已登记另两目标，必须先完成权重和技术检查；最终合并报告命令 python scripts/finalize_zero_training_transfer_20261013.py。原工作区未提交修改未动。', '',
            f"验证时HEAD：`{result['verification_head']}`。最终交付HEAD见Git提交及最终回复（本文件无需自指未来commit）。"]
    (REPO/'docs/paper_agent/transfer_20261013/report.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
