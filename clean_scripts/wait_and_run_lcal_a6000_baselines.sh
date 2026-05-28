#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_UTIL=${MAX_UTIL:-20}
MAX_USED_MB=${MAX_USED_MB:-14000}
POLL_SEC=${POLL_SEC:-180}
STABLE_POLLS=${STABLE_POLLS:-3}
GPU_IDS=${GPU_IDS:-0,1}

echo "Waiting for GPUs ${GPU_IDS} to be idle enough for A6000 baseline runs."
echo "Thresholds: util<=${MAX_UTIL}%, memory<=${MAX_USED_MB}MiB, stable_polls=${STABLE_POLLS}, poll_sec=${POLL_SEC}"

stable_count=0
while true; do
  mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  ready=1
  for gpu_id in ${GPU_IDS//,/ }; do
    row=""
    for candidate in "${rows[@]}"; do
      index=$(awk -F',' '{gsub(/ /, "", $1); print $1}' <<<"$candidate")
      if [[ "$index" == "$gpu_id" ]]; then
        row="$candidate"
        break
      fi
    done
    if [[ -z "$row" ]]; then
      echo "Could not find GPU ${gpu_id} in nvidia-smi output"
      ready=0
      continue
    fi
    used_mb=$(awk -F',' '{gsub(/ /, "", $2); print $2}' <<<"$row")
    util=$(awk -F',' '{gsub(/ /, "", $3); print $3}' <<<"$row")
    echo "$(date '+%F %T') gpu=${gpu_id} used_mb=${used_mb} util=${util}"
    if (( used_mb > MAX_USED_MB || util > MAX_UTIL )); then
      ready=0
    fi
  done

  if (( ready == 1 )); then
    stable_count=$((stable_count + 1))
    echo "Idle poll ${stable_count}/${STABLE_POLLS}"
  else
    stable_count=0
  fi

  if (( stable_count >= STABLE_POLLS )); then
    echo "GPUs ${GPU_IDS} are ready. Starting A6000 baseline runs."
    exec clean_scripts/run_lcal_a6000_baselines.sh
  fi

  sleep "$POLL_SEC"
done
