#!/usr/bin/env bash
set -o pipefail

cd /home/shx/projects/dllm_infilling/git_workspace

LOG=logs/paper_agent/20260707_h200_rebaseline_v6_tmux.log
mkdir -p logs/paper_agent

{
  date -u
  echo "RUN=h200_rebaseline_v6_short_override_tmux"
  echo "CUDA_VISIBLE_DEVICES=0"
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  HF_HOME=/home/shx/.cache/huggingface \
  CUDA_VISIBLE_DEVICES=0 \
  TOKENIZERS_PARALLELISM=false \
  /home/shx/miniconda3/envs/dllm_env/bin/python \
    clean_scripts/run_route2_rescue_quality_v5.py \
    --candidate-set cheap \
    --selector anchor_len32_short_trace_override \
    --anchor-switch-margin 0.10 \
    --short-override-gap-margin 0.285156 \
    --short-override-top1-margin 0.283203 \
    --route2-policy precision_top1_conf \
    --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_midcons_20260706_tier1_20260706_042029/results.jsonl \
    --route2-reference-results /home/shx/projects/dllm_infilling/outputs_clean/h200_rebaseline_route2_20260707_tier1_rerun_tmux_20260707_032537/results.jsonl \
    --output-dir /home/shx/projects/dllm_infilling/outputs_clean \
    --experiment-name h200_rebaseline_v6_20260707_tier1_tmux
  code=$?
  echo "COMMAND_EXIT_CODE=$code"
  date -u
  exit "$code"
} 2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"
