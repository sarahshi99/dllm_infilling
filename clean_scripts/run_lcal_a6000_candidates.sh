#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

PYTHON=${PYTHON:-/home/shx/miniconda3/envs/dllm_env/bin/python}
RUNNER=clean_scripts/run_lcal_official_bounded_repair_a6000_v2.py
BASELINE=${BASELINE:-outputs_clean/202604/full_lcas_v3b_alpha006_compact_sl_20260429_180958/results.jsonl}

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
)

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000 \
  "${COMMON_ARGS[@]}" \
  --mid-rescue-max-s3-len 12 \
  --mid-rescue-source base \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 13 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 7 \
  --mid-rescue-min-long-ratio 0.8 \
  --mid-rescue-min-support-count 2 \
  --mid-rescue-best-long-lens 13,14,15,16 \
  --mid-rescue-veto-s3-le 5 \
  --mid-rescue-veto-official-len 13

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000 \
  "${COMMON_ARGS[@]}" \
  --true-long-max-s3-len 12 \
  --true-long-min-official-len 17 \
  --true-long-max-official-len 64 \
  --true-long-min-delta 8 \
  --true-long-min-long-ratio 0.85 \
  --true-long-min-raw-ratio 0.85 \
  --true-long-min-support-count 2 \
  --true-long-best-long-lens 16,20,24,28,32,40 \
  --true-long-source any

"$PYTHON" "$RUNNER" \
  --experiment-name full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000 \
  "${COMMON_ARGS[@]}" \
  --mid-rescue-max-s3-len 12 \
  --mid-rescue-source base \
  --mid-rescue-min-official-len 11 \
  --mid-rescue-max-official-len 13 \
  --mid-rescue-min-delta 3 \
  --mid-rescue-max-delta 7 \
  --mid-rescue-min-long-ratio 0.8 \
  --mid-rescue-min-support-count 2 \
  --mid-rescue-best-long-lens 13,14,15,16 \
  --mid-rescue-veto-s3-le 5 \
  --mid-rescue-veto-official-len 13 \
  --true-long-max-s3-len 12 \
  --true-long-min-official-len 17 \
  --true-long-max-official-len 64 \
  --true-long-min-delta 8 \
  --true-long-min-long-ratio 0.85 \
  --true-long-min-raw-ratio 0.85 \
  --true-long-min-support-count 2 \
  --true-long-best-long-lens 16,20,24,28,32,40 \
  --true-long-source any
