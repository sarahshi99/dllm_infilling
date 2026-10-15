"""Finite correction batch locations, frozen inputs and portable source helpers."""
from pathlib import Path
import hashlib,json
from experiments.k4_batch_common import REPO,ROOT as OLD,DIAG,EVALUATOR,PYTHON,STAGES,read_jsonl,write_gzip
OUT=REPO/'analysis_outputs/structure_coordinate_fix_20261015_v2'
SOURCE_COMMIT='dbdd0bd5cbbf21a049a989213136aa2650844aea'
CODE_PATHS=['experiments/dreamon_singleline_order_parallel.py','analysis/dreamon_structure_replay.py',
 'scripts/prepare_structure_coordinate_fix.py','experiments/structure_coordinate_common.py',
 'experiments/run_structure_coordinate_fix.py','analysis/analyze_structure_coordinate_fix.py',
 'scripts/run_structure_coordinate_queue.py','scripts/launch_structure_coordinate_queue.sh',
 'experiments/run_k4_multiline_stage.py','experiments/k4_official_eval_worker.py',
 'experiments/dreamon_singleline_adapter.py','experiments/train_dreamon_markov_head.py',
 'experiments/dreamon_markov_head_training.py']
def influence():return json.loads((OUT/'influence_manifest.json').read_text())
def old_rows():
    return {f"{s}::{r['policy']}::{r['arm']}::{r['task_id']}":r for s in 'ABC' for r in read_jsonl(OLD/s/'per_case.jsonl.gz')}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    temp.replace(path)
