#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
SESSION=dreamon_k4_multiline_20261014_v1
OUTPUT=analysis_outputs/k4_multiline_20261014_v1
mkdir -p "$OUTPUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Queue already active: %s\n' "$SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  '/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -u scripts/run_k4_multiline_queue.py >> analysis_outputs/k4_multiline_20261014_v1/queue.log 2>&1'
printf 'Started queue: %s\n' "$SESSION"
