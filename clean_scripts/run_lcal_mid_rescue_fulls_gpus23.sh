#!/usr/bin/env bash
set -euo pipefail

cd /home/shx/projects/dllm_infilling

export CUDA_VISIBLE_DEVICES=2,3
export TOKENIZERS_PARALLELISM=false

PYTHON=/home/shx/miniconda3/envs/dllm_env/bin/python
RUNNER=clean_scripts/run_lcal_official_bounded_repair.py
BASELINE=outputs_clean/202604/full_lcas_v3b_alpha006_compact_sl_20260429_180958/results.jsonl

COMMON_ARGS=(
  --baseline-results "$BASELINE"
  --official-eval-max-s3-len 12
  --repair-max-s3-len 5
  --repair-min-official-len 6
  --repair-max-official-len 9
  --repair-min-delta 1
  --repair-max-delta 8
  --suspicion-max-s3-len 5
  --suspicion-min-official-len 16
  --suspicion-max-official-len 64
  --suspicion-min-delta 1
  --mid-rescue-max-s3-len 12
  --mid-rescue-source base
)

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_gpus23 \
  "${COMMON_ARGS[@]}" \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 13 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 7 \
  --mid-rescue-min-long-ratio 0.8

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_union_midaggr_off11_15_d3_8_r08_gpus23 \
  "${COMMON_ARGS[@]}" \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 15 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 8 \
  --mid-rescue-min-long-ratio 0.8
