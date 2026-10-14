#!/usr/bin/env python3
"""Portable CPU aggregation; never imports model or executes candidate code."""
from pathlib import Path
import argparse,collections,csv,gzip,hashlib,json,statistics,sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from experiments.k4_batch_common import ROOT,STAGES,read_jsonl


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvout(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def mean(rows,key):
    vals=[r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None

def analyze(stage,root=ROOT):
    out=root/stage;spec=STAGES[stage]
    path=out/'per_case.jsonl.gz'
    if not path.exists():
        dump(out/'integrity_audit.json',dict(status='not_generated',missing=spec['population']*len(spec['arms'])*len(spec['policies'])))
        return False
    rows=read_jsonl(path)
    contexts=read_jsonl(root/(spec['dataset']+'_contexts.jsonl.gz'));ids={r['task_id'] for r in contexts}
    assert len(ids)==spec['population'] and {r['task_id'].split('/')[2] for r in contexts}=={str(i) for i in range(164)}
    expected={(p,a,t) for p in spec['policies'] for a in spec['arms'] for t in ids}
    seen=set();pending=[];groups=collections.defaultdict(list)
    for r in rows:
        key=(r['policy'],r['arm'],r['task_id'])
        assert key in expected and key not in seen,'unexpected/duplicate stable configuration ID'
        seen.add(key)
        row_seed=int.from_bytes(hashlib.sha256(f"42\0{r['task_id']}".encode()).digest()[:8],'big')&((1<<63)-1)
        assert r['seed']==42 and r['row_seed']==row_seed
        if type(r.get('passed')) is not bool:pending.append(key);continue
        assert r['base_forward_count']==len(r['action_trace'])
        assert r['markov_enabled_count']==sum(len(t['chain_edges']) for t in r['action_trace'])
        assert all(0<=t['actual_submitted_count']<=spec['k'] for t in r['action_trace'])
        groups[r['policy'],r['arm']].append(r)
    summary=[]
    for p in spec['policies']:
        for a in spec['arms']:
            group=groups[p,a];n=len(group);passed=sum(r['passed'] for r in group)
            submit=sum(t['actual_submitted_count'] for r in group for t in r['action_trace'])
            nfe=sum(r['forward_count'] for r in group)
            summary.append(dict(policy=p,arm=a,N=n,base_tasks=len({r['base_task_id'] for r in group}),passed=passed,pass_rate=passed/n if n else None,
                new_generated=sum(r['provenance']=='new' for r in group),reused=sum(r['provenance']=='reused_historical_no_head_K4' for r in group),
                logical_aliases=sum(r['provenance'].startswith('alias_') for r in group),
                missing=spec['population']-n,actual_submitted_per_forward=submit/nfe if nfe else None,NFE_mean=mean(group,'forward_count'),
                output_tokens_mean=mean(group,'output_tokens'),output_chars_mean=mean(group,'output_chars'),canvas_length_mean=mean(group,'final_dynamic_length'),
                expand_total=sum(r['expand_count'] for r in group),delete_total=sum(r['delete_count'] for r in group),barrier_total=sum(r['structural_stop_count'] for r in group),
                head_calls_total=sum(r['markov_enabled_count'] for r in group),head_calls_mean=mean(group,'markov_enabled_count'),
                generation_sec_mean=mean([r for r in group if r['provenance']=='new'],'generation_sec'),
                head_projection_cuda_sec_mean=mean([r for r in group if r['provenance']=='new'],'head_projection_cuda_sec'),
                head_serial_cuda_sec_mean=mean([r for r in group if r['provenance']=='new'],'head_serial_cuda_sec'),
                trace_record_cpu_sec_mean=mean([r for r in group if r['provenance']=='new'],'trace_record_cpu_sec'),
                final_remaining_masks_total=sum(r['final_remaining_masks'] for r in group)))
    csvout(out/'summary.csv',summary)
    paired=[];index={(p,a):{r['task_id']:r for r in group} for (p,a),group in groups.items()}
    def pair(p,new,old,newrows,oldrows):
        common=set(newrows)&set(oldrows)
        help_ids=sorted(t for t in common if newrows[t]['passed'] and not oldrows[t]['passed'])
        harm_ids=sorted(t for t in common if oldrows[t]['passed'] and not newrows[t]['passed'])
        paired.append(dict(policy=p,comparison=new+'-'+old,N=len(common),help=len(help_ids),harm=len(harm_ids),net=len(help_ids)-len(harm_ids),
            both_pass=sum(newrows[t]['passed'] and oldrows[t]['passed'] for t in common),both_fail=sum(not newrows[t]['passed'] and not oldrows[t]['passed'] for t in common),
            help_ids=json.dumps(help_ids),harm_ids=json.dumps(harm_ids)))
    for p in spec['policies']:
        for new,old in ([('B','A'),('C','B'),('D','C'),('D','A'),('D','B')] if stage!='B' else [('C','A'),('D','C'),('D','A')]):
            pair(p,new,old,index.get((p,new),{}),index.get((p,old),{}))
    if stage in ['A','B']:
        anchors=read_jsonl(root/('dreamon_old_anchors.jsonl.gz' if stage=='A' else 'dreamcoder_base_old_anchors.jsonl.gz'))
        for p in spec['policies']:
            for oldarm,label in [('A','old_K1_no_head'),('B','old_K2_no_head')]:
                oldrows={r['task_id']:r for r in anchors if (r['policy'],r['arm'])==(p,oldarm)}
                for new in spec['arms']:pair(p,new,label,index.get((p,new),{}),oldrows)
    csvout(out/'paired_comparisons.csv',paired)
    missing=sorted(expected-seen)
    integrity=dict(status='complete' if not missing and not pending else 'incomplete',logical_expected=len(expected),records=len(rows),
        evaluated=sum(len(g) for g in groups.values()),missing_count=len(missing),pending_evaluation_count=len(pending),duplicates=0,
        exact_configuration_stable_id_sets=not missing and not pending,missing_ids=missing,pending_ids=pending,
        actual_new_generated=sum(r['provenance']=='new' for r in rows),reused=sum(r['provenance']=='reused_historical_no_head_K4' for r in rows),
        logical_aliases=sum(r['provenance'].startswith('alias_') for r in rows),technical_failures=int((out/'failure_journal.jsonl').exists()))
    dump(out/'integrity_audit.json',integrity)
    def fmt(x):return '无独立计时' if x is None else f'{x:.4f}'
    lines=[f'# 2026-10-14 批次{stage}：{spec["model"]}', '',
      f'官方{spec["dataset"]}数据，当前诊断生成设置。全人口{spec["population"]}填空行/164基础题；共享164基础题开发评测，不是独立未见测试。seed42，主模型与v2 KL/full/best rank256头冻结，lambda1。', '',
      f'状态：{integrity["status"]}；逻辑行{len(rows)}/{len(expected)}，实际新生成{integrity["actual_new_generated"]}，旧证据复用{integrity["reused"]}，逻辑别名{integrity["logical_aliases"]}；未完成{len(missing)+len(pending)}。', '',
      '|策略|组|N|通过|通过率|实际提交/前向|平均NFE|新生成秒/行|平均头调用|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'|{r["policy"]}|{r["arm"]}|{r["N"]}|{r["passed"]}|{fmt(r["pass_rate"])}|{fmt(r["actual_submitted_per_forward"])}|{fmt(r["NFE_mean"])}|{fmt(r["generation_sec_mean"])}|{fmt(r["head_calls_mean"])}|')
    lines += ['', '组定义：批次A的A/B/C/D分别为原生K4/链规则无头K4/零头K4/训练头K4；批次B的A/C/D分别为原生无头/零头/来源头K4；批次C的A/B/C/D分别为K1无头/K2无头/K2零头/K2训练头。C的K2无头与零头还包含既有提交规则及局部重采样差别，主要学习收益看D-C。', '', '配对救回/改错与ID见paired_comparisons.csv；旧K2 DreamOn锚点只纳入可证明不受共享扩展边界修复影响的行，表内N明确披露。普通断言、语法、运行异常及官方3秒超时均保留为有效评分，没有反复重采样。', '',
      '新生成耗时在CUDA同步边界测量，排除加载、官方执行和写出，包含离散轨迹缓冲及事件记录；trace_record_cpu_sec只覆盖Python记录组装，不隔离整个日志传输成本。head_projection_cuda_sec只测小头前向，head_serial_cuda_sec包含小头及局部采样。共享H200负载为匿名时序观测，不是公平加速测量；不能仅据K或NFE宣称加速。旧A4耗时含官方评测且无同步边界，未混进新计时均值。', '',
      '实际每前向提交数仅计选中位置直接提交，广播EOS额外删除单列delete_count；名义K不等于实际提交数。新行全部轮记录离散选位、草稿/最终提交ID、实际前驱链边/局部seed及动作/停止原因；不保存全词表logits/隐状态。旧复用轨迹缺口逐行注明，不能据此声称完整机制归因。', '',
      'Base保持固定64画布、EOS/pad原位和原生清理，完整画布top-k同分规则；64步时间表/eps0.001/EOS罚项3.0不变。K4约16次和K2约32次填满时的时钟与EOS罚项进度不同，这是跨K混杂。DreamOn64/64是参数；仍沿用扩展预算来自max_gen_len的语义，未改生成协议为另一套多行论文设置。']
    lines += ['', '观察到的配置耗时差异（只比较本轮有新计时的行；各组可用N不同）：']
    for p in spec['policies']:
        times=[r for r in summary if r['policy']==p and r['generation_sec_mean'] is not None]
        if times:lines.append(p+'：'+'；'.join(f'{r["arm"]}={r["generation_sec_mean"]:.4f}s/行(新N={r["new_generated"]})' for r in times))
    lines+=['','证据：per_case.jsonl.gz、../single-line_contexts.jsonl.gz或../multi-line_contexts.jsonl.gz、run_manifest.json、summary.csv、paired_comparisons.csv、integrity_audit.json、smoke_checks.json及匿名gpu_background.jsonl。']
    (out/'report.zh.md').write_text('\n'.join(lines)+'\n')
    return integrity['status']=='complete'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=STAGES,required=True);parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    sys.exit(0 if analyze(args.stage,args.root) else 2)
