#!/usr/bin/env bash
set -euo pipefail

WAIT_LOG="logs/paper_agent/20260609_1628_wait_dream_base_smoke_gpu23.log"
SMOKE_LOG="logs/paper_agent/20260609_1542_smoke_dream_base_lcal_official_bounded_repair_gpu2_unsandboxed.log"
MODEL_DIR="/tmp/dream_base_repo_probe_20260609"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$WAIT_LOG"
}

query_gpu_field() {
  local gpu="$1"
  local field="$2"
  local value
  value="$(nvidia-smi --id="$gpu" --query-gpu="$field" --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
  if [[ -z "$value" ]]; then
    value="999999"
  fi
  printf '%s' "$value"
}

log "waiting for GPU 2/3 to run Dream-v0-Base smoke"

while true; do
  for gpu in 2 3; do
    used="$(query_gpu_field "$gpu" memory.used)"
    util="$(query_gpu_field "$gpu" utilization.gpu)"
    log "gpu=$gpu used=${used}MiB util=${util}%"
    if [[ "$used" -lt 1000 && "$util" -lt 20 ]]; then
      log "selected GPU $gpu"
      set +e
      DLLM_DISABLE_FLASH_ATTN=1 \
      PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages \
      HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609 \
      HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 \
      TRANSFORMERS_OFFLINE=1 \
      HF_HUB_OFFLINE=1 \
      CUDA_VISIBLE_DEVICES="$gpu" \
      TOKENIZERS_PARALLELISM=false \
      /home/shx/miniconda3/envs/dllm_env/bin/python \
        clean_scripts/run_dreamcoder_official_infilling.py \
        --model-path "$MODEL_DIR" \
        --max-samples 2 \
        --mask-length-source lcal_official_bounded_repair \
        --output-dir /home/shx/projects/dllm_infilling/outputs_clean \
        --experiment-name "smoke_dream_base_lcal_official_bounded_repair_gpu${gpu}_unsandboxed" \
        --official-eval-max-s3-len 12 \
        --repair-max-s3-len 5 \
        --repair-min-official-len 6 \
        --repair-max-official-len 9 \
        --repair-min-delta 1 \
        --repair-max-delta 8 \
        --suspicion-max-s3-len 5 \
        --suspicion-min-official-len 16 \
        --suspicion-max-official-len 64 \
        --suspicion-min-delta 1 \
        --mid-rescue-max-s3-len 12 \
        --mid-rescue-source base \
        --mid-rescue-min-official-len 11 \
        --mid-rescue-max-official-len 13 \
        --mid-rescue-min-delta 3 \
        --mid-rescue-max-delta 7 \
        --mid-rescue-min-long-ratio 0.8 \
        > "$SMOKE_LOG" 2>&1
      rc="$?"
      set -e
      log "smoke exit $rc"
      exit "$rc"
    fi
  done
  sleep 60
done
