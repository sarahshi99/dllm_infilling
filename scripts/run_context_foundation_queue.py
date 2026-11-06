#!/usr/bin/env python3
"""Finite background controller, independent of an interactive Codex session."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import signal
import json
import os
import shlex
import subprocess
import time
import traceback

from experiments.context_foundation.common import *


def command(args,log,env=None,pass_fds=(),timeout=None,cpu=False,gpu=False):
    print(utc(),shlex.join(args),flush=True)
    with log.open('a') as f:
        f.write('\n'+utc()+' '+shlex.join(args)+'\n')
        f.flush()
        def isolate():
            if hasattr(os,'sched_setaffinity'):
                os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-2:]))
        if not gpu:
            return subprocess.run(args,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=env,
                pass_fds=pass_fds,timeout=timeout,preexec_fn=isolate if cpu else None).returncode
        child=subprocess.Popen(args,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=env,pass_fds=pass_fds)
        atomic_json(OUT/'worker_pid.local.json',dict(pid=child.pid,at=utc()))
        start=time.monotonic()
        signalled=None
        while child.poll() is None:
            path=OUT/'budget.local.json'
            budget=json.loads(path.read_text()) if path.exists() else {}
            resident=budget.get('resident',False) and budget.get('pid')==child.pid
            estimate=budget.get('used_sec',0)+max(0,time.time()-budget.get('heartbeat_epoch',time.time()))
            exceeded=(resident and estimate>=43200) or time.monotonic()-start>86400+43200+120
            if exceeded and signalled is None:
                child.send_signal(signal.SIGTERM)
                signalled=time.monotonic()
            elif signalled is not None and time.monotonic()-signalled>60:
                child.kill()  # This queue's child only; never any external GPU process.
            time.sleep(1)
        # Persist a crash exit bound so a later recovery cannot forget resident time.
        path=OUT/'budget.local.json'
        if path.exists():
            budget=json.loads(path.read_text())
            if budget.get('resident') and budget.get('pid')==child.pid:
                elapsed=max(0,time.time()-budget['heartbeat_epoch'])
                budget['used_sec']+=elapsed
                stage=budget.get('active_stage','smoke')
                budget['stage_sec'][stage]=budget['stage_sec'].get(stage,0)+elapsed
                budget['resident']=False
                budget['exit_charged_by_supervisor']=True
                atomic_json(path,budget)
        return child.returncode


def update_protocol_code():
    manifest = json.loads((OUT/'protocol_manifest.json').read_text())
    files = [p for root in ['experiments/context_foundation','tests/context_foundation'] for p in (REPO/root).glob('*.py')]
    files += [REPO/p for p in ['scripts/prepare_context_foundation.py','scripts/run_context_foundation_queue.py',
        'scripts/launch_context_foundation_queue.sh','analysis/analyze_context_foundation.py',
        'experiments/dreamon_singleline_order_parallel.py','experiments/dreamon_singleline_adapter.py',
        'experiments/dreamon_markov_head_training.py','experiments/train_dreamon_markov_head.py',
        'experiments/run_k4_multiline_stage.py','experiments/k4_official_eval_worker.py',
        'experiments/lrdllm_dreamcoder_adapter.py','experiments/k4_batch_common.py']]
    code = {str(p.relative_to(REPO)):sha(p) for p in files}
    if 'code_sha256' in manifest:
        assert code==manifest['code_sha256'], 'protocol implementation changed: record explicit version difference before resuming'
    else:
        manifest.update(code_sha256=code,source_commit=head_commit())
        atomic_json(OUT/'protocol_manifest.json',manifest)
    publish_ready('protocol_ready.json',OUT/'protocol_manifest.json','frozen')


def publish(store_status,lock):
    matrix=json.loads((OUT/'completion_matrix.json').read_text())
    text=(f'\n## 2026-10-26 第一批A context foundation 自动收尾\n\n'
          f"状态{matrix['status']}；A1={matrix['A1']['actual']}/3072、A2={matrix['A2']['actual']}/512、DCD={matrix['DCD']['actual']}/256、评分={matrix['generation']['scored']}/4132。"
          '证据analysis_outputs/dreamon_context_foundation_20261015_v1/report.zh.md；单H200有限预算，无训练/第二批/思维导图修改。\n')
    ledger=REPO/'docs/paper_agent/activity_ledger.zh.md'
    # Append once for this precise exported matrix, preserving all historical entries.
    marker=stable(matrix)
    if not (OUT/'ledger_marker.local.json').exists() or json.loads((OUT/'ledger_marker.local.json').read_text())['marker']!=marker:
        with ledger.open('a') as f:
            f.write(text)
        atomic_json(OUT/'ledger_marker.local.json',dict(marker=marker))
    allowed=[]
    for p in OUT.rglob('*'):
        if not p.is_file() or '.local.' in p.name or p.suffix in ['.log','.sqlite','.lock'] or '.sqlite-' in p.name or '.tmp-' in p.name:
            continue
        if p.name in ['publication_status.json','queue_status.json','queue_failure.json','launch.json']:
            continue
        allowed.append(p)
    # Manifest covers compact evidence only; large tensor cache has its own logical index.
    manifest=OUT/'A_manifest.json'
    allowed=[p for p in allowed if p!=manifest]
    atomic_json(manifest,dict(contract_id=CONTRACT,schema_version=1,status=matrix['status'],source_commit=head_commit(),
        producer_branch=BRANCH,data=manifest_entries(allowed)))
    allowed.append(manifest)
    stage=[str(p.relative_to(REPO)) for p in allowed]+['docs/paper_agent/activity_ledger.zh.md']
    atomic_json(OUT/'staged_files.local.json',stage)
    log=OUT/'publication.log'
    if command(['git','add','--',*stage],log)!=0:
        raise RuntimeError('evidence git add failed')
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode:
        assert command(['git','commit','--only','-m','results: archive finite context foundation batch A and coverage gaps','--',*stage],log)==0
    commit=head_commit()
    push=['git','-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1','push','origin',BRANCH]
    rc=command(['timeout','45',*push],log)
    remote=None
    if rc==0:
        try:
            remote=subprocess.check_output(['timeout','25','git','-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12',
                'ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True,stderr=subprocess.DEVNULL).split()[0]
        except (subprocess.CalledProcessError,IndexError):
            pass
    publication=dict(status='pushed_verified' if remote==commit else 'local_commit_saved_remote_unverified',
        local_commit=commit,remote_head=remote,retry_command=shlex.join(push),at=utc())
    atomic_json(OUT/'publication_status.json',publication)
    publish_ready('A_done.json',manifest,matrix['status'])
    atomic_json(OUT/'queue_status.json',dict(status='finished',completion_status=matrix['status'],at=utc(),publication=publication,
        resume=f'cd {REPO} && bash scripts/launch_context_foundation_queue.sh',pid=os.getpid(),stages=store_status))
    print('Batch A archived; finite queue exited. No second batch or training scheduled.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--restore-missing',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        lock=acquire(common_dir()/'A_queue.lock')
    except BlockingIOError:
        print('Existing batch A queue owns A_queue.lock; no duplicate launched.',flush=True)
        return
    if (OUT/'completion_matrix.json').exists() and not args.restore_missing:
        matrix=json.loads((OUT/'completion_matrix.json').read_text())
        if matrix['status']=='complete' and (OUT/'A_manifest.json').exists():
            publish_ready('A_done.json',OUT/'A_manifest.json','complete')
            print('Exact completed batch verified; no duplicate GPU work.',flush=True)
            return
    atomic_json(OUT/'queue_status.json',dict(status='starting',at=utc(),pid=os.getpid(),resume='bash scripts/launch_context_foundation_queue.sh'))
    update_protocol_code()
    frozen=json.loads((OUT/'frozen_protocol.json').read_text())
    for category in ['model_shards','model_runtime_sha256','tokenizer_sha256']:
        for name,digest in frozen[category].items():
            assert sha(SNAPSHOT/name)==digest, 'frozen model/tokenizer changed: '+name
    assert sha(HEAD)==frozen['head']['sha256']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    if args.restore_missing:
        gpu_cmd=[PYTHON,'-u','-m','experiments.context_foundation.restore']
    else:
        gpu_cmd=[PYTHON,'-u','-m','experiments.context_foundation.worker']
    atomic_json(OUT/'queue_status.json',dict(status='gpu_worker_or_waiting',at=utc(),pid=os.getpid(),resume='bash scripts/launch_context_foundation_queue.sh',command=gpu_cmd))
    worker=command(gpu_cmd,OUT/'worker.log',env=env,pass_fds=(lock.fileno(),),gpu=True)
    # Subprocess exit is the CUDA release boundary, not a promise to free a live model.
    cpu_env=dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    atomic_json(OUT/'queue_status.json',dict(status='cpu_scoring_analysis',worker_exit=worker,at=utc(),pid=os.getpid()))
    analysis=command([PYTHON,'-u','analysis/analyze_context_foundation.py'],OUT/'analysis.log',env=cpu_env,cpu=True)
    if analysis!=0:
        raise RuntimeError('CPU analysis failed; durable generation remains; see analysis.log')
    publish(dict(gpu_worker_exit=worker,cpu_analysis_exit=analysis),lock)


if __name__=='__main__':
    try:
        main()
    except Exception:
        atomic_json(OUT/'queue_failure.json',dict(at=utc(),traceback=traceback.format_exc(),
            resume='bash scripts/launch_context_foundation_queue.sh',policy='no budget reset, no ordinary-score regeneration'))
        raise
