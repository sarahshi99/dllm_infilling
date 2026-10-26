from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import time

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'analysis_outputs/dreamon_context_foundation_20261015_v1'
CACHE = Path('/home/shx/.cache/dllm_infilling/dreamon_context_foundation_20261015_v1')
CONTRACT = 'dreamon-context-v1-20261015'
BRANCH = 'codex/dreamon-context-foundation-20261015-v1'
BASE = 'd39a51d494efc2a1d1661eb28622b18b3b741fa6'
PYTHON = '/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python'
OLD_DATA = REPO / 'analysis_outputs/dreamon_markov_head_training_20260910_v2'
PREPARED = Path('/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/data/prepared_records.jsonl.gz')
SNAPSHOT = Path('/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194')
HEAD = Path('/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/checkpoints/kl/full/best.pt')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}-{threading.get_ident()}')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(tmp, 'wt') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    tmp.replace(path)


def common_dir():
    git = subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=REPO, text=True).strip()
    path = Path(git) / 'research_runs/dreamon_context_20261015_v1'
    path.mkdir(parents=True, exist_ok=True)
    return path


def head_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def publish_ready(name, manifest, status):
    manifest = Path(manifest).resolve()
    value = json.loads(manifest.read_text())
    assert value['contract_id'] == CONTRACT and value['schema_version'] == 1
    for entry in value['data']:
        p = REPO / entry['path']
        assert not Path(entry['path']).is_absolute() and p.is_file() and sha(p) == entry['sha256']
    atomic_json(common_dir() / name, dict(contract_id=CONTRACT, schema_version=1, status=status,
        source_commit=head_commit(), producer_branch=BRANCH, manifest_absolute_path=str(manifest),
        manifest_sha256=sha(manifest), created_at=utc()))


def manifest_entries(paths):
    result = []
    for p in paths:
        p = Path(p)
        if p.name.endswith(('.jsonl', '.jsonl.gz')):
            count = sum(1 for _ in read_jsonl(p))
        elif p.suffix == '.npz':
            import numpy as np
            with np.load(p,allow_pickle=False) as z:
                count = len(z['state_id'])
        elif p.suffix == '.csv':
            import csv
            with p.open() as f:
                count = sum(1 for _ in csv.DictReader(f))
        else:
            count = None
        result.append(dict(path=str(p.relative_to(REPO)), sha256=sha(p), rows=count))
    return result


class Store:
    """Transactions make completed rows idempotent; raw shards publish before references."""
    def __init__(self, path=None):
        self.db = sqlite3.connect(path or OUT / 'queue.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS rows(kind TEXT,key TEXT,value TEXT,PRIMARY KEY(kind,key))')
        self.db.commit()

    def put(self, kind, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO rows VALUES(?,?,?)', (kind, key, json.dumps(value, allow_nan=False)))

    def get(self, kind, key):
        r = self.db.execute('SELECT value FROM rows WHERE kind=? AND key=?', (kind, key)).fetchone()
        return json.loads(r[0]) if r else None

    def rows(self, kind):
        return [json.loads(r[0]) for r in self.db.execute('SELECT value FROM rows WHERE kind=? ORDER BY key', (kind,))]

    def count(self, kind):
        return self.db.execute('SELECT COUNT(*) FROM rows WHERE kind=?',(kind,)).fetchone()[0]


class BudgetExceeded(Exception):
    pass


class Budget:
    """Resident wall time including load/writes; monotonic session clock, durable heartbeat.

    Supervisor checkpoints every second even when a model call is running. A crashed
    supervisor is conservatively charged until the next recovery's process check.
    """
    def __init__(self):
        self.path = OUT / 'budget.local.json'
        previous = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.used = float(previous.get('used_sec', 0))
        self.stages = previous.get('stage_sec', {})
        if previous.get('resident'):
            # Crash with no exit timestamp: charge the full unknown interval.
            unknown = max(0, time.time() - previous['heartbeat_epoch'])
            self.used += unknown
            stage = previous.get('active_stage','smoke')
            self.stages[stage] = self.stages.get(stage,0)+unknown
        self.mutex = threading.RLock()
        self.start = time.monotonic()
        self.stage_start = self.start
        self.stage = 'smoke'

    def snapshot(self):
        now = time.monotonic()
        stages = dict(self.stages)
        stages[self.stage] = stages.get(self.stage, 0) + now - self.stage_start
        return dict(used_sec=self.used + now - self.start, stage_sec=stages, heartbeat_at=utc(),
                    heartbeat_epoch=time.time(), pid=os.getpid(), resident=True,active_stage=self.stage)

    def save(self):
        with self.mutex:
            atomic_json(self.path, self.snapshot())

    def switch(self, stage):
        with self.mutex:
            now = time.monotonic()
            self.stages[self.stage] = self.stages.get(self.stage, 0) + now - self.stage_start
            self.stage_start = now
            self.stage = stage
            self.save()

    def check(self):
        s = self.snapshot()
        if s['used_sec'] >= 43200:
            raise BudgetExceeded('total_12_gpu_job_hours')
        limits = {'mechanism': 14400, 'dcd': 7200}
        if s['stage_sec'].get(self.stage, 0) >= limits.get(self.stage, float('inf')):
            raise BudgetExceeded(self.stage + '_stage_cap')

    def close(self):
        self.save()
        v = json.loads(self.path.read_text())
        v['resident'] = False
        atomic_json(self.path, v)


def acquire(path):
    f = open(path, 'a')
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return f
