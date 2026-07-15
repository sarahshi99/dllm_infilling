#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
PYTHON="/home/shx/miniconda3/envs/dllm_env/bin/python"
DATASET="/home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl"
STAMP="20260715_v1"
LOG_DIR="$ROOT/logs/paper_agent"
OUT_ROOT="$ROOT/outputs_clean/m1_randomspanlight_${STAMP}"
COMPACT="$ROOT/analysis_outputs/m1_randomspanlight_${STAMP}"

mkdir -p "$LOG_DIR" "$OUT_ROOT" "$COMPACT"
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface \
  "$PYTHON" experiments/m1_abductive_program_state_bridge.py \
  --dataset-jsonl "$DATASET" \
  --stage1-output-dir "$OUT_ROOT/stage1" \
  --generic-output-dir "$OUT_ROOT/generic" \
  --m1-output-dir "$OUT_ROOT/dependency_cone" \
  --compact-dir "$COMPACT" \
  --smoke-cases 12 \
  --auto-full
