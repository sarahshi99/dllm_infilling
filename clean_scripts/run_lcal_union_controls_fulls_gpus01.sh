#!/usr/bin/env bash
set -euo pipefail

cd /home/shx/projects/dllm_infilling

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

PYTHON=/home/shx/miniconda3/envs/dllm_env/bin/python
RUNNER=clean_scripts/run_lcal_official_bounded_repair.py
BASELINE=outputs_clean/202604/full_lcas_v3b_alpha006_compact_sl_20260429_180958/results.jsonl

COMMON_ARGS=(
  --baseline-results "$BASELINE"
  --repair-max-s3-len 5
  --repair-min-official-len 6
  --repair-max-official-len 9
  --repair-min-delta 1
  --repair-max-delta 8
  --suspicion-min-official-len 16
  --suspicion-max-official-len 64
  --suspicion-min-delta 1
)

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus01_control \
  "${COMMON_ARGS[@]}"

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_union_eval12_nomiddle_s3_off6_9_delta1_8_susp16_gpus01_control \
  "${COMMON_ARGS[@]}" \
  --official-eval-max-s3-len 12
