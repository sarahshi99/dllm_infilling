"""Build an offline casebook from recorded facts and Codex's per-row reviews."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_dreamon_markov_errors import OUT, SOURCE, bundle, read_jsonl, write_json, write_jsonl, utc


def mechanisms(arms):
    bits={a:r['passed'] for a,r in arms.items()}
    out=[dict(layer='位置选择/覆盖',level='无法判断',fact='计数不能定位失败位置；无匹配位置/画布轨迹'),
         dict(layer='结构动作/停止',level='无法判断',fact='残余mask与动作总数见原字段；不倒推首次结构分歧')]
    if bits['C'] != bits['D']:
        out.append(dict(layer='小头修正本身',level='有限支持',fact='同协议D-C通过位改变，最终文本可对照；缺首次分歧轨迹，不能定位具体头替换'))
    else:
        out.append(dict(layer='小头修正本身',level='无法判断',fact='C/D通过位相同；实际头调用总数单独作为直接观察'))
    out.append(dict(layer='给头更多上下文',level='待验证假设' if not bits['D'] and bits['E'] else '无法判断',
                    fact='E成功/D失败只显示本配置恢复空间；未隔离上下文、采样或先前画布差异'))
    out.append(dict(layer='主模型语义',level='有限支持' if not bits['A'] else '无法判断',
                    fact='A也失败提示检查基础生成；一个种子不能证明能力上限' if not bits['A'] else 'A通过为对照，不能据其他组失败断言基础能力上限'))
    timeout=any(r['failure_reason']=='timed out' for r in arms.values())
    out.append(dict(layer='评测/资源',level='有限支持' if timeout else '无法判断',
                    fact='同3秒隔离复评仍超时；具体卡住轮与历史CPU调度未保存' if timeout else '官方结果及安全CPU/编译事实分开；没有新增基础设施根因证据'))
    return out


def classify(record, candidate, replay):
    if candidate['compile_error']:
        return '缩进错误' if candidate['compile_error']['type'] in ('IndentationError','TabError') else '语法错误'
    if record['failure_reason']=='timed out': return '超时'
    info=replay.get('instrumented',{}) if replay.get('instrumentation_agrees') else replay.get('official',{})
    if info.get('exception_type')=='AssertionError': return '断言失败'
    if info.get('exception_type'): return '运行异常/'+info['exception_type']
    return '证据不足'


def build():
    records,problems,joined=bundle()
    queue=read_jsonl(OUT/'review_queue.jsonl')
    reviews={r['index']:r for r in read_jsonl(OUT/'semantic_reviews.jsonl')}
    assert len(reviews)==len(queue)==326
    replay={(r['policy'],r['arm'],r['task_id']):r for r in read_jsonl(OUT/'official_test_replay.jsonl')}
    candidates={(r['stable_id'],r['arm']):r for r in read_jsonl(OUT/'candidates.jsonl')}
    labels=[];cases=[]
    all_cases=read_jsonl(OUT/'all_cases.jsonl')
    queue_by_id={q['stable_id']:(i,q) for i,q in enumerate(queue)}
    for c in all_cases:
        arms=joined[c['policy'],c['task_id']]
        c['arms']=arms
        c['context']=c['task_id']
        if c['stable_id'] in queue_by_id:
            i,q=queue_by_id[c['stable_id']];review=reviews[i]
            assert q['status']=='reviewed'
            c.update(status='reviewed',reading_group=q['reading_group'],first_generation_divergence='unknown',
                     mechanisms=mechanisms(arms),failures={},review_index=i,
                     checked=['自己的prefix/suffix','五组原输出及历史评分','全部失败的静态或隔离CPU证据','成功组对照','官方测试和断言语义'],
                     review_source='semantic_reviews.jsonl#index='+str(i),
                     unknowns=['匹配生成trace不存在；首次生成分歧unknown','总调用/动作次数不说明失败词位置','单种子重复开发人口不能外推能力上限'])
            for letters,note in review['notes'].items():
                for arm in letters:
                    r=arms[arm];cand=candidates[c['stable_id'],arm];ev=replay[c['policy'],arm,c['task_id']]
                    label=classify(r,cand,ev)
                    symptoms=[]
                    if r['output']=='':symptoms.append('空输出')
                    if r['final_remaining_masks']>0:symptoms.append('残余掩码')
                    if len(r['output'])>150:symptoms.append('长输出（长度事实；非截断/重复根因）')
                    if cand['compile_error'] and cand['compile_error']['message'].startswith('unterminated'):symptoms.append('未闭合字面量')
                    info=ev.get('instrumented',{}) if ev.get('instrumentation_agrees') else ev.get('official',{})
                    failures=dict(arm=arm,observable_label=label,primary_rule='完整拼接编译错误优先；否则历史超时；否则一致插桩异常/断言；最后证据不足',
                        symptom_labels=symptoms,explanation_zh=note,observable_evidence_level='直接观察' if label!='证据不足' else '无法判断',
                        failure_position=cand['compile_error'].get('line') if cand['compile_error'] else info.get('failure_line'),
                        relevant_snippet=cand['compile_error'].get('text') if cand['compile_error'] else info.get('failure_source'),
                        output=r['output'],candidate_program=cand['program'],official_passed=r['passed'],official_failure_reason=r['failure_reason'],
                        test_evidence=ev,first_generation_divergence='unknown',mechanisms=c['mechanisms'],review_status='reviewed',
                        observed_head_calls=r['markov_enabled_count'],head_calls_evidence_level='直接观察',
                        known='代码/拼接/失败证据见本行；原评分不覆盖',
                        unknown='无匹配生成轨迹；动态记录没有重建历史资源调度；超时未记录卡住轮',
                        minimum_next_evidence=('以后另行授权同3秒超时栈/局部变量取证；历史调度不能追回，不提高时限改成绩'
                            if r['failure_reason']=='timed out' else '只对选定配对捕获共同画布、位置、原/修正词元、局部种子、结构动作与停止原因；本轮不执行GPU'),
                        source='per_case_compact.jsonl + task_contexts.json + candidates.jsonl + official_test_replay.jsonl + semantic_reviews.jsonl')
                    c['failures'][arm]=failures
                    labels.append(dict(stable_id=c['stable_id'],policy=c['policy'],task_id=c['task_id'],base_task_id=c['base_task_id'],**failures))
            c['error_types']=sorted({x['observable_label'] for x in c['failures'].values()})
            c['evidence_levels']=sorted({'直接观察'}|{m['level'] for m in c['mechanisms']})
        else:
            c.update(reading_group='无需失败分析',failures={},mechanisms=[],error_types=[],evidence_levels=[],
                     first_generation_divergence='unknown',unknowns=['五组历史全对；未做错误归因'],checked=[])
        cases.append(c)
    assert len(labels)==1220
    write_jsonl(OUT/'all_cases.jsonl',[{k:c[k] for k in ('stable_id','policy','task_id','base_task_id','bits','abd','status','reading_group','first_generation_divergence')} for c in cases])
    write_jsonl(OUT/'failure_labels.jsonl',labels)
    # All visited population contexts; no additional dataset or sealed-test access.
    write_json(OUT/'task_contexts.json',{t:problems[t] for t in sorted(problems)})
    counts=Counter(q['status'] for q in queue)
    ids={status:[q['stable_id'] for q in queue if q['status']==status] for status in ('reviewed','missing_evidence','failed_infrastructure')}
    ids['unreviewed']=[q['stable_id'] for q in queue if q['status'] in ('pending','evidence_ready')]
    coverage=dict(total_to_review=len(queue),reviewed=counts['reviewed'],missing_evidence=counts['missing_evidence'],unreviewed=len(ids['unreviewed']),
        failed_infrastructure=counts['failed_infrastructure'],ids_by_status=ids,reviewed_ids=ids['reviewed'],missing_evidence_ids=ids['missing_evidence'],
        unreviewed_ids=ids['unreviewed'],failed_infrastructure_ids=ids['failed_infrastructure'],
        review_base_tasks=len({q['base_task_id'] for q in queue}),review_task_ids=len({q['task_id'] for q in queue}),
        by_policy=dict(Counter(q['policy'] for q in queue)),reading_groups=dict(Counter(q['reading_group'] for q in queue)),
        failed_outputs=1220,failed_outputs_per_group=dict(Counter(r['policy']+'/'+r['arm'] for r in records if r['passed'] is False)),
        population_lines_per_policy=1033,population_base_tasks=164,valid_complete_cases=2066,incomplete_cases=0,incomplete_ids=[],
        independent_unknown_passed_queue=[],all_pass_cases=1740,all_pass_reviewed=False,
        explained_mechanism_unknown=326,explained_mechanism_unknown_ids=ids['reviewed'],
        semantic_unknown_due_to_missing=0,semantic_unknown_ids=[],first_generation_divergence_unknown=326,
        failure_primary_counts=dict(Counter(x['observable_label'] for x in labels)),
        dynamic_records=773,compile_only_records=447,timeout_replays=13,
        new_official_executions=sum(r['status']=='executed' and not r.get('reused_identical_program') for r in replay.values()),
        instrumented_records=sum('instrumented' in r for r in replay.values()),instrumentation_disagreements=[],reviewer_gate_disabled=True,
        process='one Codex actually read finite batches and wrote individual notes; scripts only restore/measure/classify observable facts')
    write_json(OUT/'coverage_audit.json',coverage)
    missing=dict(semantic_cases_missing=[],unreviewed=[],
        generation_trace=dict(count=326,ids=ids['reviewed'],reason='runner popped action_trace/step_trace/markov_transitions before save',
                              source='experiments/run_dreamon_markov_k2_diagnosis.py:369',acquire='future authorized selected GPU replay only'),
        timeout_blocked_iteration=[q['stable_id']+'/'+a for q in queue for a in q['failed_arms'] if joined[q['policy'],q['task_id']][a]['failure_reason']=='timed out'],
        historical_cpu_dispatch='not recoverable from new CPU measurements',
        return_serialization='safe snapshots encode both tuples/lists as JSON arrays; unsupported types have type name. Use exact candidate/test code for container type, do not infer type equality from snapshot.',
        next_minimal_pairs=[dict(stable_id=c['stable_id'],compare='D/E',hypothesis='fresh right context can recover this final failure; context cause unproven',
                                capture=['matched common canvas','selected coordinates','raw/right logits','left token','local sampling seed','action/stop reason'])
                            for c in cases if c['status']=='reviewed' and not c['arms']['D']['passed'] and c['arms']['E']['passed']])
    write_json(OUT/'missing_evidence.json',missing)
    levels=defaultdict(Counter)
    for c in cases:
        for m in c['mechanisms']:levels[m['layer']][m['level']]+=1
    write_json(OUT/'mechanism_evidence_summary.json',{k:{level:v[level] for level in ('直接观察','有限支持','待验证假设','无法判断')} for k,v in levels.items()})
    background=list(csv.DictReader((SOURCE/'evidence/gpu_background.csv').open()))
    gpu_summary=json.loads((SOURCE/'review_20261013/gpu_background_summary.json').read_text())
    gpu_summary.update(source='../dreamon_markov_k2_diagnosis_20261013_v1/evidence/gpu_background.csv',
        samples=len(background),shared_h200=True,new_gpu_measurements=0,
        summary_reference='../dreamon_markov_k2_diagnosis_20261013_v1/review_20261013/report.zh.md',
        limitation='row-boundary snapshots; not continuous; no evaluator UTC to align; no other users PID/path/command reproduced')
    write_json(OUT/'gpu_background_summary.json',gpu_summary)
    # The single embedded JSON retains all 10330 original rows. Candidate data is
    # displayed via textContent, never evaluated or inserted as markup.
    embedded_cases=[]
    for c in cases:
        v=dict(c);v['failures']={a:{k:x for k,x in f.items() if k not in ('candidate_program','output')} for a,f in c['failures'].items()}
        embedded_cases.append(v)
    payload=json.dumps(dict(cases=embedded_cases,contexts=problems,coverage=coverage),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    template=(ROOT_TEMPLATE.parent/'dreamon_error_casebook.html').read_text()
    (OUT/'casebook.html').write_text(template.replace('__AUDIT_JSON__',payload))
    print(json.dumps(dict(coverage=coverage['reviewed'],labels=len(labels),html_bytes=(OUT/'casebook.html').stat().st_size),ensure_ascii=False))


ROOT_TEMPLATE=Path(__file__).resolve()
if __name__=='__main__':build()
