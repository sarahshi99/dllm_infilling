"""Task specification and transactional evidence; no GPU scheduling."""
from pathlib import Path
import collections
import json
import os
import sqlite3
import subprocess

from experiments.context_foundation.common import atomic_json, utc, sha, read_jsonl, stable

REPO = Path(__file__).resolve().parents[2]
TASK = os.environ.get('THRESHOLD_TASK_ID', 'dreamon_threshold_20261104_v1')
assert TASK in ('dreamon_threshold_20261104_v1', 'dreamon_multiline_threshold_20261111_v1')
MULTILINE = TASK == 'dreamon_multiline_threshold_20261111_v1'
SHARED = Path('/home/shx/projects/dllm_infilling') / ('threshold_multiline_eval_20261111_v1' if MULTILINE else 'threshold_eval_20261104_v1')
OUT = REPO/'analysis_outputs'/TASK
PYTHON = '/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python'
DATASET = 'multi-line' if MULTILINE else 'single-line'


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


def case_metadata(key):
    import re
    match=re.fullmatch(r'L(\d+)(?:_L(\d+))?',key.split('/')[-1])
    if not match:raise ValueError(key)
    start,end=match.groups();span=int(end or start)-int(start)+1
    if span<1:raise ValueError(key)
    return dict(base_task_id=key.split('/')[2],hole_span_lines=span,
                span_bucket='1' if span==1 else '2-3' if span<=3 else '4-7' if span<=7 else '8+',crossline=span>1)


def population_counts(rows):
    meta=[case_metadata(r['task_id']) for r in rows]
    return dict(cases=len(rows),base_tasks=len({r['base_task_id'] for r in meta}),
        crossline_cases=sum(r['crossline'] for r in meta),crossline_base_tasks=len({r['base_task_id'] for r in meta if r['crossline']}),
        span_layers={b:dict(cases=sum(r['span_bucket']==b for r in meta),base_tasks=len({r['base_task_id'] for r in meta if r['span_bucket']==b})) for b in ('1','2-3','4-7','8+')})


def select_timing(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['task_id'].split('/')[2]].append(r['task_id'])
    return [min([k for k in groups[g] if case_metadata(k)['crossline']] or groups[g],
                key=lambda k:stable(['multiline_timing_v1',k])) for g in sorted(groups,key=int)]


# Git blob comparison allows report commits while rejecting changed executable inputs.
GENERATION_PATHS=['experiments/threshold_eval/adapter.py','experiments/threshold_eval/common.py',
    'experiments/threshold_eval/runner.py','experiments/context_foundation/adapter.py',
    'experiments/dreamon_singleline_order_parallel.py','experiments/dreamon_singleline_adapter.py',
    'experiments/train_dreamon_markov_head.py','experiments/dreamon_markov_head_training.py',
    'experiments/k4_batch_common.py','scripts/start_threshold_worker.sh']


def verify_execution(version):
    subprocess.run(['git','diff','--exit-code',version,'--',*GENERATION_PATHS],cwd=REPO,check=True,capture_output=True)
    return version


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
    rows = sorted(population(DATASET),key=lambda r:r['task_id'])
    counts = population_counts(rows)
    timing = select_timing(rows) if MULTILINE else [min(r['task_id'] for r in rows if r['task_id'].split('/')[2]==str(g)) for g in range(164)]
    if MULTILINE:
        planned=json.loads((OUT/'task_pack/planning_spec.json').read_text())
        assert counts['cases']==planned['dataset']['population']
        assert counts['span_layers']==planned['dataset']['span_layers']
        planned_timing=json.loads((OUT/'task_pack/timing_cases.planned.json').read_text())
        assert timing==[r['case_key'] for r in planned_timing]
        smoke=[]
        for bucket in ('1','2-3','4-7','8+'):
            eligible=sorted(r['task_id'] for r in rows if case_metadata(r['task_id'])['span_bucket']==bucket)
            smoke.extend(eligible[i*(len(eligible)-1)//3] for i in range(4))
    else:
        old=json.loads((REPO/'analysis_outputs/k4_multiline_20261014_v1/A/run_manifest.json').read_text())
        smoke=old['smoke_ids']
    record = dict(task_id=TASK,created_at=utc(),source_commit=commit(),reference_commit='43a52fd5fb43392e2ede7b6be659d3261dd6818a',
        worktree=str(REPO),report_directory=str(OUT),shared_directory=str(SHARED),python=PYTHON,
        model=frozen['model'],snapshot=str(SNAPSHOT),model_dtype='bfloat16',head_dtype='float32',head_lambda=1,
        head=matches[0],head_registry=str(registry_path),protocol=frozen['official_protocol'],seed=42,
        sampler=frozen['sampler'],configs=configs(),cases=[r['task_id'] for r in rows],
        smoke_cases=smoke,timing_cases=timing,timing_repetitions=3,
        dataset=DATASET,population=counts,expected_quality=len(configs())*len(rows),expected_timing=len(configs())*len(timing)*3,base_tasks=164,
        role_expected={role:sum(c['method']==role for c in configs())*len(rows) for role in ('nohead','trained')},
        case_metadata={r['task_id']:case_metadata(r['task_id']) for r in rows},
        gpu_policy='two role workers share current single mapped GPU, no GPU lock or idle wait; unified timing after both generation processes exit',
        gpu_initial=gpu_observation(),sealed_external_test_opened=False,
        retries=dict(oom=1,infrastructure=3,ordinary_failure=0),
        threshold_definition='FP32 full-vocabulary T1 probability of actual sampled token after legal action mask before T=.2/top-p=.9; only slot>0 ordinary; reject breaks same spatial chain')
    atomic_json(SHARED/'run_spec.json',record)
    atomic_json(OUT/'run_spec.json',record)
    atomic_json(SHARED/'status.json',dict(task_id=TASK,status='implementation',quality_completed=0,timing_completed=0,at=utc()))
    atomic_json(SHARED/'task_ledger.json',dict(configs=[dict(c,role=c['method'],expected=len(rows)) for c in configs()],quality_expected=record['expected_quality'],timing_expected=record['expected_timing'],storage='role SQLite PRIMARY KEY(kind,config,case_key,repetition), WAL/FULL'))
    (SHARED/'commands.txt').write_text(f'THRESHOLD_TASK_ID={TASK} {PYTHON} -m experiments.threshold_eval.runner --phase smoke\nbash {SHARED}/start_worker.sh --role nohead\nbash {SHARED}/start_worker.sh --role trained\nTHRESHOLD_TASK_ID={TASK} {PYTHON} -m experiments.threshold_eval.runner --phase pipeline\n')
    (OUT/'experiment_brief.zh.md').write_text(f"""# {TASK} 实施说明

按固定任务书执行。选择集中参数化已有入口；不重写采样、阈值和结构解码器，不修改旧结果。数据为官方{DATASET}，{len(rows)}案例；冻结DreamOn与rank256头。baseline为本轮同策略K1、同方法off及同阈值另一方法。
CPU边界与事务测试→四跨度层固定16例真实模型smoke及旧单行回归→执行版本提交与READY→两个角色同卡并发→自然退出→官方bwrap每例3秒评分→独立164×34×3计时→整题bootstrap10000次→报告与push核实。
GPU使用实时映射；不抢占、不等待外部任务空闲、不设GPU锁。OOM仅失败进程退出后恢复一次；基础设施至多重试3次。普通代码失败和超时保留。成功条件：{record['expected_quality']}质量唯一行与{record['expected_timing']}独立计时行，跨行及跨度分母一致，零重复、非法行、混合生成版本与计时token分歧。风险：固定64预算及外部GPU争用；不能声称独立新题泛化或确定加速达标。
reviewer_gate_disabled：本地diff review与fresh verification；不调用subagent。
""")



if __name__=='__main__':
    prepare()
