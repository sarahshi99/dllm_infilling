#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/shx/miniconda3/envs/dllm_env/bin/python}
LOG_DIR=${LOG_DIR:-logs/a6000_resume}
MIDCONS_GPU_IDS=${MIDCONS_GPU_IDS:-0,1}
MID_PRECISION_GPU_IDS=${MID_PRECISION_GPU_IDS:-2,3}
TRUE_LONG_GPU_IDS=${TRUE_LONG_GPU_IDS:-0,1}
COMBINED_GPU_IDS=${COMBINED_GPU_IDS:-2,3}

MIDCONS_RUN_DIR=${MIDCONS_RUN_DIR:-outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626}
MID_PRECISION_RUN_DIR=${MID_PRECISION_RUN_DIR:-outputs_clean/full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517}
TRUE_LONG_RUN_DIR=${TRUE_LONG_RUN_DIR:-outputs_clean/full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755}
COMBINED_RUN_DIR=${COMBINED_RUN_DIR:-outputs_clean/full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756}

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_DISABLE_XET=1

mkdir -p "$LOG_DIR"

launch_resume() {
  local label=$1
  local gpu_ids=$2
  local run_dir=$3
  local log_path="$LOG_DIR/${label}.log"
  local pid_path="$LOG_DIR/${label}.pid"

  if [[ ! -f "$run_dir/results.jsonl" ]]; then
    echo "Missing results.jsonl for $label: $run_dir" >&2
    return 1
  fi

  nohup env CUDA_VISIBLE_DEVICES="$gpu_ids" \
    "$PYTHON" clean_scripts/resume_lcal_a6000_policy.py --run-dir "$run_dir" \
    >"$log_path" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$pid_path"
  echo "$label pid=$pid gpu_ids=$gpu_ids run_dir=$run_dir log=$log_path"
}

launch_resume midcons "$MIDCONS_GPU_IDS" "$MIDCONS_RUN_DIR"
launch_resume mid_precision "$MID_PRECISION_GPU_IDS" "$MID_PRECISION_RUN_DIR"
launch_resume true_long "$TRUE_LONG_GPU_IDS" "$TRUE_LONG_RUN_DIR"
launch_resume combined "$COMBINED_GPU_IDS" "$COMBINED_RUN_DIR"
