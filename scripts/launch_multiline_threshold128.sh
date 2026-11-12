#!/usr/bin/env bash
set -euo pipefail
export THRESHOLD_TASK_ID=dreamon_multiline_threshold_128_20261112_v1
export THRESHOLD_SHARED=/home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$WORKTREE"
mkdir -p "$THRESHOLD_SHARED"
exec 8>"$THRESHOLD_SHARED/launch.lock"
if ! flock -n 8; then echo '128 launcher already running'; exit 0; fi
"$PYTHON" -m experiments.threshold_eval.common
if [[ ! -f "$THRESHOLD_SHARED/READY.json" ]]; then
  # One capacity observation, never a GPU-idle polling queue. Existing roles use
  # about 18 GiB each; allow 40,000 MiB for two additional concurrent roles.
  "$PYTHON" - <<'PY'
import json
import subprocess
from experiments.threshold_eval.common import SHARED, OUT, TASK, atomic_json, gpu_observation, spec, utc
gpu = gpu_observation()
free = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
free_mib = int(free.strip())
if free_mib < 40000:
    value = dict(task_id=TASK, status='blocked_resources', at=utc(), free_mib=free_mib,
                 required_free_mib=40000, quality_completed=0, timing_completed=0,
                 quality_expected=spec()['expected_quality'], gpu=gpu,
                 reason='Insufficient capacity for two additional model roles; no GPU process started, no idle polling.')
    atomic_json(SHARED/'status.json', value)
    atomic_json(OUT/'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    raise SystemExit(4)
PY
  export THRESHOLD_EXECUTION_COMMIT=$(git rev-parse HEAD)
  "$PYTHON" -u -m experiments.threshold_eval.runner --phase smoke > "$THRESHOLD_SHARED/smoke.log" 2>&1
  # The smoke child has exited and released its model before READY and roles.
  "$PYTHON" - <<'PY'
import json
import os
from experiments.threshold_eval.common import SHARED, OUT, REPO, TASK, atomic_json, utc, verify_execution
version = verify_execution(os.environ['THRESHOLD_EXECUTION_COMMIT'])
assert json.loads((OUT/'smoke.json').read_text())['status'] == 'passed'
value = dict(task_id=TASK, at=utc(), worktree=str(REPO), execution_commit=version,
             smoke_passed=True, smoke_process_exited=True)
atomic_json(SHARED/'READY.json', value)
atomic_json(OUT/'execution_manifest.json', value)
PY
fi
exec "$PYTHON" -u -m experiments.threshold_eval.runner --phase pipeline
