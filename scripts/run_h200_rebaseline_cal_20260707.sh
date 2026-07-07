#!/usr/bin/env bash
set -o pipefail

cd /home/shx/projects/dllm_infilling/git_workspace

LOG=logs/paper_agent/20260707_h200_rebaseline_cal_tmux.log
mkdir -p logs/paper_agent

{
  date -u
  echo "RUN=h200_rebaseline_cal_tmux"
  echo "CUDA_VISIBLE_DEVICES=0"
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  HF_HOME=/home/shx/.cache/huggingface \
  CUDA_VISIBLE_DEVICES=0 \
  TOKENIZERS_PARALLELISM=false \
  /home/shx/miniconda3/envs/dllm_env/bin/python \
    clean_scripts/run_cal_official_lcas_v3.py \
    --output-dir /home/shx/projects/dllm_infilling/outputs_clean \
    --experiment-name h200_rebaseline_cal_20260707_tier1_tmux
  code=$?
  echo "COMMAND_EXIT_CODE=$code"
  date -u
  exit "$code"
} 2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"
