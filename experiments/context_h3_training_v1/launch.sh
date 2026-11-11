#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_root=$(git -C "$task_dir" rev-parse --show-toplevel)
task_run=/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1
mkdir -p "$task_run"
cd "$task_root"
# Own outer lock also protects shell exit files against duplicate launches.
exec 9>"$task_run/launch.lock"
flock -n 9 || exit 73
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
set +e
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -u -m experiments.context_h3_training_v1.pipeline "$@" >> "$task_run/pipeline.log" 2>&1
task_exit=$?
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python - "$task_run" "$task_exit" <<'PYEXIT'
import json,sys
from pathlib import Path
from datetime import datetime,timezone
(Path(sys.argv[1])/'shell_exit.json').write_text(json.dumps(dict(exit_code=int(sys.argv[2]),at=datetime.now(timezone.utc).isoformat()))+'\n')
PYEXIT
exit "$task_exit"
