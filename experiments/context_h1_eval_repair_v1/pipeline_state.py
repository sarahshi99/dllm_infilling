import os
from pathlib import Path
from experiments.context_foundation.common import REPO,atomic_json,utc
RUN=Path('/home/shx/.cache/dllm_infilling/context_h1_eval_repair_20261111_v1')
SOURCE=RUN.parent/'context_h1_training_20261107_v1'
OUT=REPO/'analysis_outputs/dreamon_context_h1_eval_repair_20261111_v1'
RESUME=f'cd {REPO} && bash experiments/context_h1_eval_repair_v1/launch.sh'
def progress(stage,**values):
    row=dict(stage=stage,at=utc(),pid=os.getpid(),**values)
    atomic_json(RUN/'progress.json',row)
    print(row,flush=True)
