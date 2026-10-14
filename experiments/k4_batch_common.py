"""Exact authorized batch populations/protocols; CPU-only portable helpers."""
from pathlib import Path
import gzip
import importlib.util
import json

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'analysis_outputs/k4_multiline_20261014_v1'
DIAG = REPO / 'analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/run_manifest.json'
EVALUATOR = Path('/home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff')
PYTHON = Path('/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python')
STAGES = {
 'A': dict(dataset='single-line', population=1033, model='dreamon', k=4,
           arms='ABCD', policies=['global_confidence','left_to_right_frontier']),
 'B': dict(dataset='single-line', population=1033, model='dreamcoder_base', k=4,
           arms='ACD', policies=['global_confidence','left_to_right_frontier']),
 'C': dict(dataset='multi-line', population=5815, model='dreamon', k=2,
           arms='ABCD', policies=['global_confidence']),
}
CODE_PATHS = ['experiments/k4_batch_common.py','experiments/run_k4_multiline_stage.py',
 'experiments/dreamon_singleline_order_parallel.py','experiments/dreamcoder_fixed_transfer_adapter.py',
 'experiments/dreamon_singleline_adapter.py','experiments/run_dreamon_singleline_order_parallel.py',
 'experiments/lrdllm_dreamcoder_adapter.py','experiments/dreamon_markov_head_training.py',
 'experiments/train_dreamon_markov_head.py','experiments/k4_official_eval_worker.py',
 'analysis/analyze_k4_multiline_batch.py','scripts/run_k4_multiline_queue.py',
 'scripts/prepare_k4_multiline_evidence.py','scripts/launch_k4_multiline_queue.sh']

def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_gzip(path, rows):
    # One atomic export, never replace a complete archive with an interrupted write.
    temp = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp,'wt',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    temp.replace(path)

def population(dataset):
    spec=importlib.util.spec_from_file_location('k4_pinned_data',EVALUATOR/'human_eval_infilling/data.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    problems=module.read_problems(dataset)  # official normal loader, no old controller exclusions
    rows=list(problems.values())
    assert len(rows)==(1033 if dataset=='single-line' else 5815)
    assert len({r['task_id'] for r in rows})==len(rows)
    assert {r['task_id'].split('/')[2] for r in rows}=={str(i) for i in range(164)}
    return rows
