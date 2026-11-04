"""Task specification and transactional evidence; no GPU scheduling."""
from pathlib import Path
import collections
import json
import os
import sqlite3
import subprocess

from experiments.context_foundation.common import atomic_json, utc, sha, read_jsonl

REPO = Path(__file__).resolve().parents[2]
SHARED = Path('/home/shx/projects/dllm_infilling/threshold_eval_20261104_v1')
OUT = REPO/'analysis_outputs/dreamon_threshold_20261104_v1'
PYTHON = '/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python'
TASK = 'dreamon_threshold_20261104_v1'


def configs():
    result = []
    for order in ('global','left'):
        result.append(dict(id=f'{order}_k1_nohead_off',order=order,k=1,method='nohead',threshold=None))
        for k in (2,4):
            for method in ('nohead','trained'):
                for threshold,label in ((None,'off'),(.80,'080'),(.90,'090'),(.95,'095')):
                    result.append(dict(id=f'{order}_k{k}_{method}_{label}',order=order,k=k,method=method,threshold=threshold))
    return result


def spec():
    return json.loads((SHARED/'run_spec.json').read_text())


def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()


def gpu_observation():
    result = dict(at=utc(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for name,query in [('devices','index,uuid,name,memory.total,memory.used,utilization.gpu'),
                       ('compute_processes','pid,gpu_uuid,used_gpu_memory')]:
        option = '--query-gpu=' if name=='devices' else '--query-compute-apps='
        p = subprocess.run(['nvidia-smi',option+query,'--format=csv,noheader,nounits'],capture_output=True,text=True)
        result[name] = p.stdout.strip().splitlines()
        if p.returncode:
            result[name+'_error'] = p.stderr[-1000:]
    return result


class Ledger:
    def __init__(self, role):
        self.path = SHARED/f'{role}.sqlite'
        self.db = sqlite3.connect(self.path,timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS results(kind TEXT, config TEXT, case_key TEXT, repetition INTEGER, payload TEXT NOT NULL, PRIMARY KEY(kind,config,case_key,repetition))')
        self.db.execute('CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, at TEXT, phase TEXT, key TEXT, error TEXT)')
        self.db.commit()

    def has(self,kind,config,case,repetition=0):
        return self.db.execute('SELECT 1 FROM results WHERE kind=? AND config=? AND case_key=? AND repetition=?',(kind,config,case,repetition)).fetchone() is not None

    def put(self,kind,config,case,payload,repetition=0):
        # INSERT deliberately rejects duplicate keys. Resume skips committed rows.
        with self.db:
            self.db.execute('INSERT INTO results VALUES(?,?,?,?,?)',(kind,config,case,repetition,json.dumps(payload,ensure_ascii=False,allow_nan=False)))

    def rows(self,kind):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM results WHERE kind=? ORDER BY config,case_key,repetition',(kind,))]

    def count(self,kind):
        return self.db.execute('SELECT COUNT(*) FROM results WHERE kind=?',(kind,)).fetchone()[0]

    def attempt(self,phase,key,error):
        with self.db:
            self.db.execute('INSERT INTO attempts(at,phase,key,error) VALUES(?,?,?,?)',(utc(),phase,key,error))


def prepare():
    from experiments.k4_batch_common import population
    from experiments.context_foundation.common import SNAPSHOT
    OUT.mkdir(parents=True,exist_ok=True)
    SHARED.mkdir(parents=True,exist_ok=True)
    if (SHARED/'run_spec.json').exists():
        assert spec()['task_id']==TASK
        return
    frozen = json.loads((REPO/'analysis_outputs/dreamon_context_foundation_20261015_v1/frozen_protocol.json').read_text())
    registry_path = REPO/'analysis_outputs/dreamon_markov_head_training_20260910_v2/checkpoint_registry.json'
    registry = json.loads(registry_path.read_text())
    matches = [r for r in registry if (r['head'],r['phase'],r['label'])==('kl','full','best')]
    assert len(matches)==1 and matches[0]['sha256']==frozen['head']['sha256']
    assert sha(matches[0]['path'])==frozen['head']['sha256']
    rows = sorted(population('single-line'),key=lambda r:r['task_id'])
    assert len(rows)==1033 and len({r['task_id'].split('/')[2] for r in rows})==164
    # Fixed before any outcomes; one lexicographically first case per base task.
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['task_id'].split('/')[2]].append(r['task_id'])
    timing = [min(groups[g]) for g in sorted(groups,key=int)]
    old = json.loads((REPO/'analysis_outputs/k4_multiline_20261014_v1/A/run_manifest.json').read_text())
    record = dict(task_id=TASK,created_at=utc(),source_commit=commit(),reference_commit='43a52fd5fb43392e2ede7b6be659d3261dd6818a',
        worktree=str(REPO),report_directory=str(OUT),shared_directory=str(SHARED),python=PYTHON,
        model=frozen['model'],snapshot=str(SNAPSHOT),model_dtype='bfloat16',head_dtype='float32',head_lambda=1,
        head=matches[0],head_registry=str(registry_path),protocol=frozen['official_protocol'],seed=42,
        sampler=frozen['sampler'],configs=configs(),cases=[r['task_id'] for r in rows],
        smoke_cases=old['smoke_ids'],timing_cases=timing,timing_repetitions=3,
        expected_quality=35122,expected_timing=16728,base_tasks=164,
        gpu_policy='two role workers share current single mapped GPU, no GPU lock or idle wait; unified timing after both generation processes exit',
        gpu_initial=gpu_observation(),sealed_external_test_opened=False,
        retries=dict(oom=1,infrastructure=3,ordinary_failure=0),
        threshold_definition='FP32 full-vocabulary T1 probability of actual sampled token after legal action mask before T=.2/top-p=.9; only slot>0 ordinary; reject breaks same spatial chain')
    atomic_json(SHARED/'run_spec.json',record)
    atomic_json(OUT/'run_spec.json',record)
    atomic_json(SHARED/'status.json',dict(task_id=TASK,status='implementation',quality_completed=0,timing_completed=0,at=utc()))
    atomic_json(SHARED/'task_ledger.json',dict(configs=[dict(c,role=c['method'],expected=1033) for c in configs()],quality_expected=35122,timing_expected=16728,storage='role SQLite PRIMARY KEY(kind,config,case_key,repetition), WAL/FULL'))
    (SHARED/'commands.txt').write_text(f'{PYTHON} -m experiments.threshold_eval.runner --phase smoke\nbash {SHARED}/start_worker.sh --role nohead\nbash {SHARED}/start_worker.sh --role trained\n{PYTHON} -m experiments.threshold_eval.runner --phase pipeline\n')
    (OUT/'experiment_brief.zh.md').write_text('''# 阈值评测实施说明

本轮按用户固定矩阵执行，不扩展方向。候选实现：修改历史入口、独立继承共同采样解码器、重写完整模型。选择独立继承，保留历史入口用于 off 逐 token/画布对照；K显式传1/2/4。

baseline 为两策略独立 K1 和同方法 off；数据为官方已授权单行1033例/164基础题；模型和旧rank256头冻结。比较为本轮共同采样重跑，不与951/817/831历史行配对。阈值失败只停止当前连续链，第一空间候选唯一豁免。结构规则继承坐标修补版。头只读实际已提交普通前驱。

验证：CPU边界/结构/链/事务测试→固定16真实例技术smoke→提交执行版本→原子READY→双角色同卡→顺序官方3秒评分→统一164题×34×3计时→人口/版本审计、整题bootstrap、报告与push。技术smoke不是效果筛选；普通失败和超时保留。OOM本进程最多恢复一次；其它基础设施最多重试3次。

当前GPU映射不重设；不等待空闲、不设GPU锁、不干预外部任务。真实时延必须标为共享环境，不能凭质量并发阶段耗时声称加速。成功标准：35122唯一质量行、16728独立计时行、34全配置完整。终止条件为可复现实现错误或重试耗尽，保留其它正常角色和已提交事务。

reviewer_gate_disabled：按项目协议做本地diff审查和fresh tests；无独立reviewer，残余风险在报告注明。
''')


if __name__=='__main__':
    prepare()
