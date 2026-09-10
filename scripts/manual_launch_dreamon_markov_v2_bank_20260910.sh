#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python"
MODEL="/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194"
SOURCE="/home/shx/projects/dllm_infilling/.worktrees/dreamon-source-8a0a549"
RUN_ROOT="/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2"
DATA_ROOT="$RUN_ROOT/data"
BANK_DB="$RUN_ROOT/transition_bank.sqlite"
OUTPUT_ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-markov-k2-eval-v1"
RESULT_DIR="$OUTPUT_ROOT/analysis_outputs/dreamon_markov_head_training_20260910_v2"
LOG="$OUTPUT_ROOT/logs/paper_agent/20260910_dreamon_markov_v2_bank.log"

mkdir -p "$RUN_ROOT" "$RESULT_DIR" "$(dirname "$LOG")"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

printf '[bank-launch] %s\n' "$(date -u +%FT%TZ)"
printf '[execution-head] %s\n' "$(git rev-parse HEAD)"
printf '[bank-db] %s\n' "$BANK_DB"
printf '[result-dir] %s\n' "$RESULT_DIR"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
"$PYTHON" experiments/build_dreamon_markov_transition_bank.py \
  --prepared-records "$DATA_ROOT/prepared_records.jsonl.gz" \
  --source-root "$SOURCE" \
  --model-snapshot "$MODEL" \
  --bank-db "$BANK_DB" \
  --result-dir "$RESULT_DIR" \
  --device cuda \
  --per-problem-cap 16 \
  --oversample 1.25 \
  --train-target 200000 \
  --validation-target 20000 \
  --external-test-target 20000

printf '[bank-exit] 0 %s\n' "$(date -u +%FT%TZ)"
