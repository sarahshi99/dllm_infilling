#!/usr/bin/env python3
"""Finite correction queue: GPU subset -> CPU merge/statistics -> archive/push -> exit."""
from pathlib import Path
import fcntl,json,os,shlex,subprocess,sys,time,traceback
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from experiments.structure_coordinate_common import OUT,PYTHON
from experiments.structure_coordinate_common import atomic_write_json

def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def command(args,log):
    with log.open('a') as f:
        f.write('\n'+utc()+' '+shlex.join(args)+'\n');f.flush()
        return subprocess.run(args,cwd=REPO,stdout=f,stderr=subprocess.STDOUT).returncode

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    lock=open(OUT/'queue.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_write_json(OUT/'queue_status.json',dict(status='running',pid=os.getpid(),started_utc=utc(),resume='bash scripts/launch_structure_coordinate_queue.sh'))
    worker=command([str(PYTHON),'-u','experiments/run_structure_coordinate_fix.py'],OUT/'worker.log')
    if worker!=0:
        raw=OUT/'failure_retry_log.jsonl'
        if not raw.exists() or raw.stat().st_size==0:
            raw.write_text(json.dumps(dict(time_utc=utc(),error='worker infrastructure/technical exit '+str(worker),worker_log_tail=(OUT/'worker.log').read_text()[-5000:]),ensure_ascii=False)+'\n')
        if not (OUT/'failure_retry_coverage.json').exists():
            atomic_write_json(OUT/'failure_retry_coverage.json',dict(end_utc=utc(),events=sum(1 for _ in raw.open()),empty=False,scope='driver captured worker failure before its own coverage record'))
    analysis=command([str(PYTHON),'-u','analysis/analyze_structure_coordinate_fix.py'],OUT/'analysis.log')
    matrix=json.loads((OUT/'completion_matrix.json').read_text()) if (OUT/'completion_matrix.json').exists() else dict(status='technical_failure')
    matrix.update(worker_exit=worker,analysis_exit=analysis)
    atomic_write_json(OUT/'completion_matrix.json',matrix)
    text=f'\n## 2026-10-15 并行扩删坐标补验 v2 自动完成记录\n\n状态{matrix["status"]}；新生成{matrix.get("new_generated",0)}，有证明复用{matrix.get("proved_reused",0)}；修补评分救回/改错{matrix.get("score_help",0)}/{matrix.get("score_harm",0)}。证据analysis_outputs/structure_coordinate_fix_20261015_v2/report.zh.md。旧v1表不改，旧单行K2仅历史参照，修补贡献与学习贡献分开；无思维导图修改或后续四头训练。\n'
    docs=['pause_checkpoint.current.md','paper_agent_dashboard.zh.md','activity_ledger.zh.md','experiment_queue.md']
    for name in docs:
        with (REPO/'docs/paper_agent'/name).open('a') as f:f.write(text)
    # Raw failures are retained locally; publish all events with machine paths
    # replaced, alongside explicit empty-log/coverage evidence when none occurred.
    raw=OUT/'failure_retry_log.jsonl'
    if not raw.exists():raw.write_text('')
    (OUT/'failure_retry_log.portable.jsonl').write_text(raw.read_text().replace(str(REPO),'[repo]').replace('/home/shx','[user-home]'))
    blocked={'.sqlite','.log','.lock','.tmp'}
    files=[str(p.relative_to(REPO)) for p in OUT.rglob('*') if p.is_file() and p.suffix not in blocked
           and p.name not in ['publication_status.json','queue_status.json','progress.json','failure_retry_log.jsonl']
           and '.sqlite' not in p.name]
    files += [str(Path('docs/paper_agent')/name) for name in docs]
    log=OUT/'publication.log'
    assert command(['git','add','--',*files],log)==0,'evidence stage failed'
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode:
        assert command(['git','commit','-m','results: finish finite structure coordinate v2 correction and paired audit'],log)==0,'commit failed'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    push=['git','-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1','push','origin',branch]
    pushed=command(['timeout','45',*push],log);remote=None
    if pushed==0:
        try:remote=subprocess.check_output(['timeout','25','git','-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True,stderr=subprocess.DEVNULL).split()[0]
        except (subprocess.CalledProcessError,IndexError):pass
    publication=dict(status='pushed_verified' if remote==head else 'local_saved_remote_unverified',local_commit=head,remote_head=remote,retry_command=shlex.join(push))
    atomic_write_json(OUT/'publication_status.json',publication)
    atomic_write_json(OUT/'queue_status.json',dict(status='completed' if worker==analysis==0 and matrix['status']=='complete' else 'finished_with_technical_block',ended_utc=utc(),publication=publication,matrix=matrix))
    print('Finite coordinate correction queue exited; no training or further investigations scheduled.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        atomic_write_json(OUT/'queue_failure.json',dict(time_utc=utc(),traceback=traceback.format_exc(),resume='bash scripts/launch_structure_coordinate_queue.sh'))
        raise
