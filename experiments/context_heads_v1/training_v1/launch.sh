#!/usr/bin/env bash
set -euo pipefail
task_source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_repo_root=$(git -C "$task_source_dir" rev-parse --show-toplevel)
task_run_dir=/home/shx/.cache/dllm_infilling/context_heads_training_20261028_v1
mkdir -p "$task_run_dir"
exec 9>"$task_run_dir/worker.lock"
flock -n 9 || { echo 'This pilot already has a worker'; exit 1; }
cd "$task_repo_root"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu --format=csv > "$task_run_dir/gpu_at_launch.txt"
task_exit=0
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/shx/miniconda3/envs/dllm_env/bin/python -u -m experiments.context_heads_v1.training_v1.run \
    --device cuda:0 --batch-size 16 --learning-rate 0.0003 --seed 20261028 \
    --threads 4 --memory-limit-gib 8 --checkpoint-every 64 --max-wall-seconds 3600 \
    >> "$task_run_dir/train.log" 2>&1 || task_exit=$?
/home/shx/miniconda3/envs/dllm_env/bin/python - "$task_run_dir" "$task_exit" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
root=Path(sys.argv[1])
(root/'exit.json').write_text(json.dumps(dict(exit_code=int(sys.argv[2]),at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
PY
exit "$task_exit"
