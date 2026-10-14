#!/usr/bin/env python3
"""Detached finite A->B->C queue; failures isolated, no training or further batches."""
from pathlib import Path
import fcntl,json,os,shlex,subprocess,sys,time,traceback
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from experiments.k4_batch_common import ROOT,PYTHON,STAGES

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def command(args,log):
    with log.open('a') as f:
        f.write('\n'+utc()+' '+shlex.join(args)+'\n');f.flush()
        return subprocess.run(args,cwd=REPO,stdout=f,stderr=subprocess.STDOUT).returncode

def publish(label):
    # Explicit compact portable evidence list. Never stage SQLite, raw runtime
    # logs, unrelated worktree modifications, weights, or third-party process IDs.
    files=[]
    for path in ROOT.rglob('*'):
        if not path.is_file():continue
        if path.suffix in ['.gz','.csv'] or path.name in ['run_manifest.json','integrity_audit.json','report.zh.md','smoke_checks.json',
            'reuse_audit.json','anchor_manifest.json','population_use_20261014.json','completion_matrix.json',
            'technical_tests.json','cpu_test_results.txt','launch.json','recovery.zh.md','portable_manifest.json','final_report.zh.md','prior_publication.json',
            'official_evaluator_execution.py','official_dataset_loader.py','gpu_background.jsonl']:
            files.append(str(path.relative_to(REPO)))
    files+=['docs/paper_agent/pause_checkpoint.current.md','docs/paper_agent/paper_agent_dashboard.zh.md','docs/paper_agent/activity_ledger.zh.md']
    log=ROOT/'publication.log'
    if command(['git','add','--',*files],log):return {'status':'local_stage_failed'}
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode:
        if command(['git','commit','-m',label],log):return {'status':'local_commit_failed'}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    push=['git','-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1','push','origin',branch]
    status=command(['timeout','45',*push],log)
    remote=None
    if status==0:
        try:
            remote=subprocess.check_output(['timeout','25','git','-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True,stderr=subprocess.DEVNULL).split()[0]
        except (subprocess.CalledProcessError,IndexError):pass
    return dict(status='pushed_verified' if remote==head else 'local_saved_push_or_readback_failed',local_commit=head,remote_head=remote,
        retry_command=shlex.join(push),remote_readback_command=f'git ls-remote origin refs/heads/{branch}')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=open(ROOT/'queue.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(ROOT/'queue_status.json',dict(status='running',pid=os.getpid(),start_utc=utc(),resume_command='bash scripts/launch_k4_multiline_queue.sh'))
    matrix={}
    for stage in STAGES:
        out=ROOT/stage;out.mkdir(exist_ok=True)
        manifest=out/'run_manifest.json'
        complete=manifest.exists() and json.loads(manifest.read_text()).get('status')=='completed'
        status=0 if complete else command([str(PYTHON),'experiments/run_k4_multiline_stage.py','--stage',stage],out/'runner.log')
        analysis_status=command([str(PYTHON),'analysis/analyze_k4_multiline_batch.py','--stage',stage],out/'analysis.log')
        audit=json.loads((out/'integrity_audit.json').read_text()) if (out/'integrity_audit.json').exists() else {}
        matrix[stage]=dict(worker_exit=status,analysis_exit=analysis_status,**audit)
        save(ROOT/'completion_matrix.json',matrix)
        # Preserve old research records; append a concise date entry per stage.
        text=f'\n## 2026-10-14 批次{stage}自动队列记录\n\n状态{audit.get("status","technical_failure")}；评分{audit.get("evaluated",0)}；新生成{audit.get("actual_new_generated",0)}，复用{audit.get("reused",0)}，别名{audit.get("logical_aliases",0)}。证据analysis_outputs/k4_multiline_20261014_v1/{stage}/report.zh.md。未通过技术阶段的配置保留事务恢复点；普通评分失败不触发重生成。单worker继续下一独立阶段，无训练。\n'
        for name in ['pause_checkpoint.current.md','paper_agent_dashboard.zh.md','activity_ledger.zh.md']:
            with (REPO/'docs/paper_agent'/name).open('a') as f:f.write(text)
        save(ROOT/'publication_status.json',publish(f'results: archive authorized K4 multiline stage {stage}'))
        print(f'Finished stage {stage}: {matrix[stage].get("status")}',flush=True)
    # Bounded network failures never stop local evaluation. A final retry covers
    # both pre-existing transfer/audit branches and this finite batch branch.
    prior_publication={}
    for branch in ['codex/dreamon-zero-training-transfer-20261013','codex/dreamon-markov-error-audit-v1']:
        retry=['git','-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12','push','origin',branch]
        status=command(['timeout','45',*retry],ROOT/'publication.log')
        local=subprocess.check_output(['git','rev-parse',branch],cwd=REPO,text=True).strip();remote=None
        if status==0:
            try:remote=subprocess.check_output(['timeout','25','git','-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True,stderr=subprocess.DEVNULL).split()[0]
            except (subprocess.CalledProcessError,IndexError):pass
        prior_publication[branch]=dict(status='pushed_verified' if local==remote else 'local_saved_push_or_readback_failed',local_commit=local,remote_head=remote,retry_command=shlex.join(retry))
    save(ROOT/'prior_publication.json',prior_publication)
    report=['# 2026-10-14 有限批次最终记录','', '状态/通过数/实际新生成见下表，各批配对表和报告保留完整分母；共享GPU耗时只作观察。', '', '|批次|状态|已评分逻辑行|新生成|复用|别名|','|---|---|---:|---:|---:|---:|']
    for stage,item in matrix.items():
        report.append(f'|{stage}|{item.get("status","technical_failure")}|{item.get("evaluated",0)}|{item.get("actual_new_generated",0)}|{item.get("reused",0)}|{item.get("logical_aliases",0)}|')
        summary=ROOT/stage/'summary.csv'
        if summary.exists():
            import csv
            for row in csv.DictReader(summary.open()):
                report.append(f'{stage} {row["policy"]}/{row["arm"]}：{row["passed"]}/{row["N"]}，新生成秒/行={row["generation_sec_mean"]}，实际提交/前向={row["actual_submitted_per_forward"]}。')
    report+=['','分支codex/dreamon-k4-multiline-eval-20261014-v1，最终提交见本报告所在Git HEAD；远端核验及准确重试命令另存本地publication_status.json。旧迁移/错误审计推送状态见prior_publication.json。网络失败不冒充已上传。','', '仍缺旧326案例的匹配生成轨迹与13超时卡点；本批新行不补造旧机制证据。Dream/DiffuCoder下载保持待办，无训练、K8、重复计时或其它后续实验。']
    (ROOT/'final_report.zh.md').write_text('\n'.join(report)+'\n')
    save(ROOT/'publication_status.json',publish('docs: finalize finite K4 multiline completion matrix'))
    save(ROOT/'queue_status.json',dict(status='completed' if all(r.get('status')=='complete' for r in matrix.values()) else 'finished_with_technical_blocks',pid=os.getpid(),end_utc=utc(),matrix=matrix,
        publication=json.loads((ROOT/'publication_status.json').read_text())))
    print('Finite queue exited. No training or additional experiments scheduled.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        ROOT.mkdir(parents=True,exist_ok=True)
        save(ROOT/'queue_failure.json',dict(time_utc=utc(),traceback=traceback.format_exc(),resume_command='bash scripts/launch_k4_multiline_queue.sh'))
        raise
