#!/usr/bin/env python3
"""One finite CPU consumer. No model imports, GPU work, training or generation."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.dreamon_copula_cpu import (CONTRACT, InputError, analyze_package, atomic_json,
                                         sha256, utc, verify_package)

BRANCH = 'codex/dreamon-copula-diagnostic-20261015-v1'
OUT = REPO/'analysis_outputs/dreamon_copula_diagnostic_20261015_v1'
DOC = REPO/'docs/paper_agent/copula_diagnostic_20261015_v1'


def git(*args, timeout=30):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()


def relay_dir():
    return Path(git('rev-parse', '--path-format=absolute', '--git-common-dir'))/'research_runs/dreamon_context_20261015_v1'


def restore_command(new_window=False):
    command = [sys.executable, '-u', str(REPO/'scripts/consume_dreamon_copula_cpu.py'),
               '--wait-seconds', '86400', '--poll-seconds', '60']
    if new_window:
        command.append('--new-wait-window')
    return 'cd '+shlex.quote(str(REPO))+' && '+shlex.join(command)


def inspect_producer(relay):
    """Return readiness, explicit failure, or waiting; no tensor existence heuristic."""
    ready = relay/'dcd_ready.json'
    if ready.is_file():
        value = json.loads(ready.read_text())
        if value.get('status') in ('failed', 'blocked', 'error'):
            return 'failed', 'producer dcd_ready explicitly reports '+value['status']
        return 'ready', ready
    done = relay/'A_done.json'
    if done.is_file():
        value = json.loads(done.read_text())
        if value.get('contract_id') == CONTRACT:
            return 'failed', 'producer A has exited without publishing dcd_ready; status='+str(value.get('status'))
    protocol = relay/'protocol_ready.json'
    if protocol.is_file():
        value = json.loads(protocol.read_text())
        path = Path(value.get('manifest_absolute_path', ''))
        if (value.get('contract_id') == CONTRACT and path.is_file()
                and sha256(path) == value.get('manifest_sha256')):
            failure = path.parent/'queue_failure.json'
            if failure.is_file():
                detail = json.loads(failure.read_text())
                status_path = path.parent/'queue_status.json'
                current = json.loads(status_path.read_text()) if status_path.is_file() else {}
                if detail.get('at', '') >= current.get('at', ''):
                    return 'failed', 'producer explicit queue_failure: '+json.dumps(detail, ensure_ascii=False)
    return 'waiting', 'A has not atomically published dcd_ready'


def wait_window(out, wait_seconds, new_window=False, now=None):
    require_seconds = 0 <= wait_seconds <= 86400
    if not require_seconds:
        raise ValueError('wait-seconds must be between 0 and 86400')
    now = time.time() if now is None else now
    path = out/'consumer_window.local.json'
    if path.exists() and not new_window:
        value = json.loads(path.read_text())
        # A restart cannot extend the original deadline.
        value['deadline_epoch'] = min(value['deadline_epoch'], value['started_epoch']+wait_seconds)
    else:
        value = dict(started_epoch=now, deadline_epoch=now+wait_seconds, started_at=utc())
        with (out/'wait_history.local.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
    atomic_json(path, value)
    return value


def checkpoint(status, reason, **extra):
    value = dict(contract_id=CONTRACT, schema_version=1, status=status, reason=reason, at=utc(),
                 pid=os.getpid(), command=restore_command(status == 'blocked'), **extra)
    atomic_json(OUT/'consumer_status.local.json', value)
    lines = ['# 第一批 C 独立恢复入口', '', f'时间：{value["at"]}', f'状态：**{status}**',
             f'原因：{reason}', '', f'分支：`{BRANCH}`',
             '输出：`analysis_outputs/dreamon_copula_diagnostic_20261015_v1/`。',
             '本批仅CPU，不训练、不调用模型或生成、不启动第二批。等待不代表诊断完成。', '',
             '恢复命令（仅手动新等待窗口允许重设24小时期限；中断恢复默认保留原期限）：', '',
             '```bash', value['command'], '```', '',
             '若partial manifest缺字段：由A发布者补充实际count、四桶missing_buckets（global/single、global/crossline、frontier/single、frontier/crossline的缺数），',
             '以及budget_stop={stopped:boolean,reason:非空字符串}，核验所有文件hash后原子重发dcd_ready；C不能自行补写A的声明。',
             '字段缺失、hash不符或数值支持不可恢复时，不得仅重新运行C绕过校验；先恢复producer包。', '']
    if extra:
        lines += ['运行记录：', '```json', json.dumps(extra, ensure_ascii=False, indent=2), '```', '']
    (DOC/'checkpoint.zh.md').write_text('\n'.join(lines))
    return value


def archive(status, relay, input_ready=None, reason=None):
    """Commit only C-owned files, push without force, verify actual remote HEAD.

    C_done references a stable C_manifest; publication status stays outside that
    manifest to avoid a self-referential commit/hash cycle. Local commits survive
    push failures. Re-entry uses the durable analysis rows, never repeats states.
    """
    allowed = [REPO/'analysis/dreamon_copula_cpu.py', REPO/'scripts/consume_dreamon_copula_cpu.py',
               REPO/'tests/test_dreamon_copula_cpu.py']
    allowed += sorted(DOC.glob('*.md'))
    evidence = sorted(p for p in OUT.iterdir() if p.is_file() and '.local.' not in p.name
                      and p.name not in ('C_manifest.json', 'publication_status.json') and p.suffix != '.log')
    manifest = dict(contract_id=CONTRACT, schema_version=1, status=status,
                    source_commit=git('rev-parse', 'HEAD'), producer_branch=BRANCH, created_at=utc(),
                    input_ready=input_ready, input_manifest_sha256=(input_ready or {}).get('manifest_sha256'),
                    blocked_reason=reason, data=[dict(path=str(p.relative_to(REPO)), sha256=sha256(p)) for p in evidence])
    atomic_json(OUT/'C_manifest.json', manifest)
    allowed += evidence+[OUT/'C_manifest.json']
    paths = sorted({str(p.relative_to(REPO)) for p in allowed})
    atomic_json(OUT/'staged_files.local.json', paths)
    print('C staged files:', json.dumps(paths, ensure_ascii=False), flush=True)
    git('add', '--', *paths)
    if subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *paths], cwd=REPO).returncode:
        git('commit', '--only', '-m', 'results: archive first-batch CPU copula '+status, '--', *paths, timeout=60)
    commit = git('rev-parse', 'HEAD')
    ssh = 'ssh -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1'
    push = ['git', '-c', 'core.sshCommand='+ssh, 'push', 'origin', BRANCH]
    remote, push_error = None, None
    try:
        subprocess.run(push, cwd=REPO, check=True, timeout=45, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        remote = git('-c', 'core.sshCommand='+ssh, 'ls-remote', 'origin', 'refs/heads/'+BRANCH, timeout=25).split()[0]
    except (subprocess.SubprocessError, IndexError) as exc:
        push_error = str(exc)
    publication = dict(status='pushed_verified' if remote == commit else 'local_commit_saved_remote_unverified',
                       local_commit=commit, remote_head=remote, at=utc(), error=push_error,
                       retry_command='cd '+shlex.quote(str(REPO))+' && '+shlex.join(push),
                       verification_command='cd '+shlex.quote(str(REPO))+' && git ls-remote origin refs/heads/'+BRANCH)
    atomic_json(OUT/'publication_status.json', publication)
    for entry in manifest['data']:
        if sha256(REPO/entry['path']) != entry['sha256']:
            raise InputError('C evidence changed after archive: '+entry['path'])
    ready = dict(contract_id=CONTRACT, schema_version=1, status=status, source_commit=commit,
                 producer_branch=BRANCH, manifest_absolute_path=str(OUT/'C_manifest.json'),
                 manifest_sha256=sha256(OUT/'C_manifest.json'), created_at=utc(),
                 input_manifest_sha256=manifest['input_manifest_sha256'], publication=publication,
                 blocked_reason=reason)
    atomic_json(relay/'C_done.json', ready)
    atomic_json(OUT/'consumer_status.local.json', dict(status=status, at=utc(), publication=publication,
                                                      command=restore_command(status == 'blocked')))
    print(json.dumps(ready, ensure_ascii=False, indent=2), flush=True)
    return ready


def existing_done(relay):
    path = relay/'C_done.json'
    if not path.exists():
        return None
    done = json.loads(path.read_text())
    if (done.get('contract_id') != CONTRACT or done.get('schema_version') != 1
            or done.get('producer_branch') != BRANCH):
        raise InputError('C_done from incompatible contract; do not overwrite')
    if done.get('status') not in ('complete', 'partial'):
        return None
    manifest_path = Path(done['manifest_absolute_path'])
    if not manifest_path.is_file() or sha256(manifest_path) != done['manifest_sha256']:
        raise InputError('existing completed C manifest hash mismatch; no duplicate analysis')
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['data']:
        p = REPO/entry['path']
        if not p.is_file() or sha256(p) != entry['sha256']:
            raise InputError('existing completed C evidence hash mismatch; no duplicate analysis')
    current = relay/'dcd_ready.json'
    if current.exists() and json.loads(current.read_text()).get('manifest_sha256') != done.get('input_manifest_sha256'):
        raise InputError('new A input differs from completed C; do not silently rerun first batch')
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=float, default=86400)
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--new-wait-window', action='store_true', help='manual recovery only; append a new bounded window')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('poll-seconds must be between 1 and 60')
    if not 0 <= args.wait_seconds <= 86400:
        parser.error('wait-seconds must be between 0 and 86400')
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    relay = relay_dir()
    relay.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode forever; never unlink any lock or anyone else's ready.
    with (relay/'C_consumer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing C consumer owns C_consumer.lock; no duplicate launched.', flush=True)
            return 0
        if git('branch', '--show-current') != BRANCH:
            raise InputError('consumer must run on its dedicated C branch')
        return consume(args, relay)


def consume(args, relay):
    previous = existing_done(relay)
    if previous:
        # A push failure is recoverable without recomputing or changing C evidence.
        print('Verified completed C; no duplicate CPU analysis.', flush=True)
        print(json.dumps(previous.get('publication'), ensure_ascii=False), flush=True)
        return 0
    window = wait_window(OUT, args.wait_seconds, args.new_wait_window)
    input_ready = None
    try:
        checkpoint('waiting', 'Awaiting validated A dcd_ready; no empirical conclusion yet.',
                   deadline_epoch=window['deadline_epoch'], poll_seconds=args.poll_seconds)
        print(utc(), 'finite CPU consumer started; deadline epoch', window['deadline_epoch'], flush=True)
        while True:
            state, detail = inspect_producer(relay)
            if state == 'ready':
                input_ready = json.loads(Path(detail).read_text())
                checkpoint('validating', 'Verify ready, manifest and every shard before analysis.')
                package = verify_package(detail)
                if package['manifest']['count'] == 0:
                    raise InputError('validated package contains N=0; no empirical diagnostic is possible')
                checkpoint('analyzing', 'Validated complete files; analyzing all actual states.',
                           input_manifest_sha256=package['ready']['manifest_sha256'], actual_n=package['manifest']['count'])
                summary = analyze_package(package, OUT)
                status = package['manifest']['status']
                checkpoint(status, summary['verdict'], actual_n=summary['actual_n'],
                           input_manifest_sha256=package['ready']['manifest_sha256'])
                archive(status, relay, package['ready'])
                return 0
            if state == 'failed':
                raise InputError(str(detail))
            remaining = window['deadline_epoch']-time.time()
            if remaining <= 0:
                raise TimeoutError('A dcd_ready wait deadline reached (at most 24 hours); no diagnosis completed')
            time.sleep(min(args.poll_seconds, remaining))
    except Exception as exc:
        reason = type(exc).__name__+': '+str(exc)
        print(traceback.format_exc(), flush=True)
        atomic_json(OUT/'blocked.json', dict(contract_id=CONTRACT, schema_version=1, status='blocked',
                                            reason=reason, at=utc(), input_ready=input_ready,
                                            restore_command=restore_command(True)))
        checkpoint('blocked', reason)
        try:
            archive('blocked', relay, input_ready, reason)
        except Exception:
            # Preserve local results/commit even if archive itself cannot finish.
            atomic_json(OUT/'archive_failure.local.json', dict(status='blocked', traceback=traceback.format_exc(),
                                                               at=utc(), restore_command=restore_command(True)))
            print('Archive failed; local evidence retained. See archive_failure.local.json.', flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
