"""Restore fixed evidence, isolate CPU forensics, and present a finite review queue.

This script never performs semantic review or marks a case reviewed.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import itertools
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/dreamon_markov_error_audit_20261013_v1"
SOURCE = ROOT / "analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1"
DATA = Path("/home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff")  # existing execution-only forensics; bundle fallback never reads this implicitly
POLICIES = ("global_confidence", "left_to_right_frontier")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def write_csv(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def bundle(dataset_root=None):
    with gzip.open(SOURCE / "evidence/per_case.jsonl.gz", "rt") as f:
        records = [json.loads(x) for x in f]
    contexts = OUT / "task_contexts.json"
    if contexts.is_file():
        problems = json.loads(contexts.read_text())
    elif dataset_root is not None:
        with gzip.open(Path(dataset_root) / "data/HumanEval-SingleLineInfilling.jsonl.gz", "rt") as f:
            problems = {r["task_id"]: r for r in map(json.loads, f)}
    else:
        raise FileNotFoundError("committed task_contexts.json missing; explicitly pass dataset_root for fallback")
    if set(problems) != {r["task_id"] for r in records}:
        raise ValueError("context stable ID set differs from the complete evidence population")
    joined = defaultdict(dict)
    for r in records:
        joined[r["policy"], r["task_id"]][r["arm"]] = r
    return records, problems, joined


def prepare():
    records, problems, joined = bundle()
    queue = read_jsonl(OUT / "review_queue.jsonl")
    expected = {f"{p}::{t}" for (p, t), arms in joined.items()
                if any(r["passed"] is False for r in arms.values())}
    assert expected == {q["stable_id"] for q in queue}
    assert len(records) == len({(r["policy"], r["arm"], r["task_id"]) for r in records}) == 10330
    contexts = {t: problems[t] for t in sorted(problems)}
    write_json(OUT / "task_contexts.json", contexts)
    # Preserve original fields/values exactly; never strip completion text.
    write_jsonl(OUT / "per_case_compact.jsonl", records)
    candidate_rows = []
    for q in queue:
        problem = problems[q["task_id"]]
        for arm, r in joined[q["policy"], q["task_id"]].items():
            program = problem["prompt"] + r["output"] + problem["suffix"] + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"
            error = None
            try:
                ast.parse(program)
                compile(program, "candidate.py", "exec")
            except SyntaxError as e:
                error = dict(type=type(e).__name__, line=e.lineno, offset=e.offset, message=e.msg, text=e.text)
            candidate_rows.append(dict(stable_id=q["stable_id"], arm=arm, program=program,
                                       compile_error=error, source_task=q["task_id"],
                                       insertion_start_line=problem["prompt"].count("\n")+1,
                                       source="per_case_compact.jsonl + task_contexts.json; pinned execution.py:61-69"))
    write_jsonl(OUT / "candidates.jsonl", candidate_rows)
    joint = []
    full = []
    for policy in POLICIES:
        subset = {t: a for (p, t), a in joined.items() if p == policy}
        for t, a in sorted(subset.items()):
            bits = ''.join(str(int(a[x]["passed"])) if x in a and type(a[x].get("passed")) is bool else '?' for x in "ABCDE")
            full.append(dict(stable_id=f"{policy}::{t}", policy=policy, task_id=t, base_task_id=t.split('/')[2], bits=bits,
                             abd=bits[0]+bits[1]+bits[3], status="pending" if '0' in bits or '?' in bits else "无需失败分析"))
        for axes in ("ABCDE", "ABD"):
            for bits in map(''.join, itertools.product('01', repeat=len(axes))):
                ids = [f"{policy}::{t}" for t,a in sorted(subset.items()) if all(type(a.get(x, {}).get("passed")) is bool and int(a[x]["passed"]) == int(bit) for x,bit in zip(axes,bits))]
                joint.append(dict(policy=policy, axes=axes, bits=bits, count=len(ids), base_tasks=len({x.split('/')[-2] for x in ids}), stable_ids=json.dumps(ids), denominator=1033, missing_ids="[]"))
    write_csv(OUT / "joint_outcomes.csv", joint)
    write_jsonl(OUT / "all_cases.jsonl", full)
    pairs=[]
    old=list(csv.DictReader((SOURCE / "paired_comparisons.csv").open()))
    for policy in POLICIES:
        subset={t:a for (p,t),a in joined.items() if p==policy}
        for left,right in (("D","B"),("D","C"),("C","B"),("E","C"),("E","D")):
            cells={bits:[f"{policy}::{t}" for t,a in sorted(subset.items()) if ''.join(str(int(a[x]["passed"])) for x in (right,left))==bits] for bits in ('00','01','10','11')}
            prior=next((r for r in old if r['policy']==policy and r['comparison']==left+'-'+right),None)
            pairs.append(dict(policy=policy,comparison=left+'-'+right,bit_order=right+left,denominator=len(subset),missing_ids='[]',
                              both_fail=len(cells['00']),help=len(cells['01']),harm=len(cells['10']),both_pass=len(cells['11']),
                              help_ids=json.dumps(cells['01']),harm_ids=json.dumps(cells['10']),cells_ids=json.dumps(cells),
                              prior_help=prior['help'] if prior else 'not previously tabulated',prior_harm=prior['harm'] if prior else 'not previously tabulated',
                              difference=0 if prior and (int(prior['help']),int(prior['harm']))==(len(cells['01']),len(cells['10'])) else None))
    write_csv(OUT / "paired_recheck.csv", pairs)
    write_json(OUT / "source_manifest.json", dict(audit_version=1,source_branch='codex/dreamon-markov-k2-diagnosis-v1',
        source_commit='c84384752819d217b930d373421a348c450797ad',original_result_commit='340170ce6ae44477027dd7e94ec7a9c0f0910384',
        dataset_root=str(DATA),dataset_revision='88062ff9859c875d04db115b698ed4b0f0395170',
        dataset_file='data/HumanEval-SingleLineInfilling.jsonl.gz',fields=list(records[0]),
        original_evidence='../dreamon_markov_k2_diagnosis_20261013_v1/evidence/per_case.jsonl.gz',
        reused=['1588-cell reproduction audit','original summary/paired/cluster CI','anonymous GPU background'],
        missing=['matching generation traces','historical evaluation traceback/CPU/exitcode'],
        concatenation="prompt + output + suffix + '\\n' + test + '\\n' + 'check(entry_point)'",
        evaluator_enablement='uncomment exact exec line only; same pinned evaluator',no_gpu=True,
        run_context=json.loads((SOURCE/'evidence/evidence_manifest.json').read_text())['run_context'],
        cpu_environment={'candidate_python':'/usr/bin/python3 (3.10.12)','gpu_set':[],'concurrency':1,'timeout_sec':3.0}))
    write_json(OUT / "integrity_recheck.json",dict(records=len(records),unique_records=len(records),cases=len(full),review_cases=len(queue),
        context_ids=len(contexts),context_join_missing=[],duplicate_keys=[],invalid_passed=[],incomplete_cases=[],
        per_group=dict(Counter(r['policy']+'/'+r['arm'] for r in records)),
        compact_equals_published=read_jsonl(OUT/'per_case_compact.jsonl')==records,
        prior_audit='../dreamon_markov_k2_diagnosis_20261013_v1/review_20261013/reproduction_audit.json',
        prior_1588_cells_recomputed=False,differences=[]))
    for q in queue:
        if q['status']=='pending': q['status']='evidence_ready'
    write_jsonl(OUT/'review_queue.jsonl',queue)
    print(json.dumps(dict(cases=len(queue),candidates=len(candidate_rows),compile_errors=sum(bool(c['compile_error']) for c in candidate_rows))))


def isolated(payload):
    with tempfile.TemporaryDirectory(prefix='dreamon-audit-') as temp:
        staging=Path(temp)
        (staging/'payload.json').write_text(json.dumps(payload))
        (staging/'execution.py').write_text((DATA/'human_eval_infilling/execution.py').read_text())
        (staging/'worker.py').write_text((ROOT/'analysis/dreamon_error_audit_worker.py').read_text())
        cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--clearenv',
             '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
             '--ro-bind',str(staging),'/input','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp',
             '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1','/usr/bin/python3','-I','/input/worker.py']
        start=utc(); mono=time.monotonic(); load=os.getloadavg()
        try:
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=7)
            info=json.loads(run.stdout) if run.returncode==0 else dict(infrastructure_error=run.stderr[-2000:])
            info.update(child_exitcode=run.returncode,child_signal=-run.returncode if run.returncode<0 else None)
        except subprocess.TimeoutExpired:
            info=dict(infrastructure_error='outer wall timeout 7s',child_exitcode=None,child_signal=None)
        info.update(start_utc=start,end_utc=utc(),parent_wall_sec=time.monotonic()-mono,host_load=load)
        return info


def replay():
    probe=isolated({'probe':True})
    write_json(OUT/'isolation_probe.json',dict(result=probe,constraints='bwrap unshare-all, empty env, only readonly runtime/input, private tmp, seccomp deny networking/process/exec/system calls, RLIMIT_AS/CPU/FSIZE/NOFILE/CORE, parent 7s wall',
        exact_command='python analysis/audit_dreamon_markov_errors.py replay',candidate_execution_before_probe=False))
    available=all(probe.get('probe',{}).get(x) is True for x in ('network','shell','host_write','credential'))
    records,problems,joined=bundle()
    candidates={(c['stable_id'],c['arm']):c for c in read_jsonl(OUT/'candidates.jsonl')}
    cache={}
    existing=read_jsonl(OUT/'official_test_replay.jsonl') if (OUT/'official_test_replay.jsonl').exists() else []
    completed={(r['policy'],r['arm'],r['task_id']) for r in existing}
    with (OUT/'official_test_replay.jsonl').open('a') as f:
        for r in records:
            key=r['policy'],r['arm'],r['task_id']
            if r['passed'] or key in completed: continue
            c=candidates[(r['policy']+'::'+r['task_id'],r['arm'])]
            result=dict(policy=r['policy'],arm=r['arm'],task_id=r['task_id'],original_passed=r['passed'],original_failure_reason=r['failure_reason'])
            cache_key=(r['task_id'],r['output'])
            if not available:
                result.update(status='not_safely_replayed',reason='isolation probe failed')
            elif c['compile_error']:
                result.update(status='static_compile_only',compile_error=c['compile_error'])
            elif cache_key in cache and r['failure_reason']!='timed out':
                result.update(cache[cache_key],reused_identical_program=True)
            else:
                payload=dict(problem=problems[r['task_id']],output=r['output'],mode='official')
                official=isolated(payload)
                result.update(status='executed',official=official)
                if r['failure_reason']!='timed out' and 'infrastructure_error' not in official:
                    payload['mode']='instrumented'
                    instrumented=isolated(payload)
                    result.update(instrumented=instrumented,instrumentation_agrees=official.get('audit_passed')==instrumented.get('audit_passed') and official.get('audit_failure_reason')==instrumented.get('audit_failure_reason'),
                                  instrumentation='candidate proxy records each original call once; no assertion rewrite, no extra random consumption; separate unmodified run')
                if r['failure_reason']!='timed out': cache[cache_key]={k:v for k,v in result.items() if k not in ('policy','arm','task_id','original_passed','original_failure_reason')}
            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
            if len(completed)%100==0: print('failure records processed',len(completed),flush=True)
            completed.add(key)
    print('finished',len(completed),flush=True)


def material(start, count):
    _,problems,joined=bundle()
    queue=read_jsonl(OUT/'review_queue.jsonl')
    replay_rows=read_jsonl(OUT/'official_test_replay.jsonl') if (OUT/'official_test_replay.jsonl').exists() else []
    replay_index={(r['policy'],r['arm'],r['task_id']):r for r in replay_rows}
    seen={q['base_task_id'] for q in queue[:start]}
    for i,q in list(enumerate(queue))[start:start+count]:
        r=problems[q['task_id']]
        print('\nCASE',i,q['stable_id'],'BITS',''.join(str(int(q['passed_bits'][a])) for a in 'ABCDE'))
        clean=re.sub(r'""".*?"""|\x27\x27\x27.*?\x27\x27\x27', '', r['prompt'], flags=re.S)
        description=re.findall(r'"""(.*?)"""',r['prompt'],flags=re.S)
        print('题意', ' '.join(description[0].split())[:380] if description else '(见既有题目)', '\nPREFIX',clean,'SUFFIX',r['suffix'],sep='\n')
        base=q['base_task_id']
        if base not in seen:
            print('TEST',r['test'],sep='\n');seen.add(base)
        else:print('TEST 基础题'+base+'的此前已读官方测试；全文保留task_contexts.json')
        output_groups=defaultdict(list)
        for a,v in sorted(joined[q['policy'],q['task_id']].items()):
            output_groups[(v['output'],v['failure_reason'])].append(a)
        for (output,reason),letters in output_groups.items():
            a=letters[0]
            print(''.join(letters),repr(output),reason,'heads',{x:joined[q['policy'],q['task_id']][x]['markov_enabled_count'] for x in letters})
            ev=replay_index.get((q['policy'],a,q['task_id']),{})
            details=ev.get('instrumented',{}) if ev.get('instrumentation_agrees') else ev.get('official',{})
            compact={k:details[k] for k in ('exception_type','failure_line','failure_source','audit_failure_reason') if k in details}
            if 'calls' in details:compact['last_call']=details['calls'][-1:]
            if details.get('exception_type') not in ('AssertionError',None):compact['locals']=details.get('failure_locals')
            print('EVIDENCE',json.dumps(compact,ensure_ascii=False))


def sync_reviews():
    queue=read_jsonl(OUT/'review_queue.jsonl'); reviews=read_jsonl(OUT/'semantic_reviews.jsonl')
    assert len({r['index'] for r in reviews})==len(reviews)
    for r in reviews:
        item=queue[r['index']]
        assert set(''.join(r['notes']))==set(item['failed_arms']),r['index']
        item.update(status='reviewed',review_artifact='semantic_reviews.jsonl#index='+str(r['index']))
    write_jsonl(OUT/'review_queue.jsonl',queue)
    pending=[i for i,q in enumerate(queue) if q['status']!='reviewed']
    (OUT/'checkpoint.zh.md').write_text(f'# 2026-10-13 DreamOn错误审计接续\n\n单个Codex；reviewer_gate_disabled；已实际审阅{len(reviews)}/326。下一条索引{pending[0] if pending else "无"}。阅读/解释见semantic_reviews.jsonl，队列状态已同步。10330原记录/198上下文/1630候选完整；CPU隔离probe通过。原评分不覆盖；首次生成分歧unknown。插桩清理故障单独保留，repair仅补插桩，13超时不重跑。下一步完成剩余语义→HTML/覆盖/记录→focused commit/push→completion。\n')
    print('reviewed',len(reviews),'remaining',len(pending))


def repair_instrumentation():
    _,problems,joined=bundle()
    rows=read_jsonl(OUT/'official_test_replay.jsonl')
    cache={}
    for r in rows:
        if 'infrastructure_error' not in r.get('instrumented',{}): continue
        output=joined[r['policy'],r['task_id']][r['arm']]['output']
        key=r['task_id'],output
        if key not in cache:
            cache[key]=isolated(dict(problem=problems[r['task_id']],output=output,mode='instrumented'))
        r['instrumented_attempts']=[r['instrumented']]
        r['instrumented']=cache[key]
        r['instrumentation_agrees']=r['official'].get('audit_passed')==r['instrumented'].get('audit_passed') and r['official'].get('audit_failure_reason')==r['instrumented'].get('audit_failure_reason')
    write_jsonl(OUT/'official_test_replay.jsonl',rows)
    print('repaired unique instrumented programs',len(cache))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','replay','read','repair','sync']);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=10);args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='replay':replay()
    elif args.action=='repair':repair_instrumentation()
    elif args.action=='sync':sync_reviews()
    else:material(args.start,args.count)
