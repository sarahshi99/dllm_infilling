#!/usr/bin/env bash
set -o pipefail

cd /home/shx/projects/dllm_infilling/git_workspace

LOG=logs/paper_agent/20260707_h200_controller_action_bank_offline.log
mkdir -p logs/paper_agent

{
  date -u
  echo "RUN=controller_action_bank_h200_20260707_tier1_offline"
  echo "CUDA_VISIBLE_DEVICES=0"
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_OFFLINE=1 \
  HF_HOME=/home/shx/.cache/huggingface \
  TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 \
  TOKENIZERS_PARALLELISM=false \
  /home/shx/miniconda3/envs/dllm_env/bin/python \
    experiments/action_ceiling/frozen_canvas_controller.py \
    --mode bank \
    --timestamp h200_20260707_tier1_offline \
    --primary-results /home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_midcons_20260706_tier1_20260706_042029/results.jsonl
  code=$?
  echo "COMMAND_EXIT_CODE=$code"
  date -u
  exit "$code"
} 2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"
