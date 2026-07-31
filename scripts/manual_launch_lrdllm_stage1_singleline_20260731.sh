#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/shx/miniconda3/envs/dllm_env/bin/python}
OFFICIAL_CAL_ROOT=${OFFICIAL_CAL_ROOT:-/home/shx/.cache/dllm_infilling/calibrated_adaptive_length-741e8418}
HUMANEVAL_ROOT=${HUMANEVAL_ROOT:-/home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff}
CURRENT_DATASET=${CURRENT_DATASET:-$HUMANEVAL_ROOT/data/HumanEval-SingleLineInfilling.jsonl.gz}
MANIFEST_ROOT=$ROOT/analysis_outputs/baseline_manifests_20260731_v1
SMOKE_MANIFEST=$MANIFEST_ROOT/cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl
FULL_MANIFEST=$MANIFEST_ROOT/cal_singleline_rest_nonfrozen_manifest.jsonl
OUTPUT_DIR=$ROOT/outputs_clean/lrdllm_stage1_singleline_smoke_20260731_v1
ANALYSIS_DIR=$ROOT/analysis_outputs/lrdllm_stage1_singleline_smoke_20260731_v1
LOG_PATH=$ROOT/logs/paper_agent/20260731_lrdllm_stage1_smoke.log
ADAPTER=$ROOT/experiments/p1_lrdllm_stage1_adapter.py
ARM=local_lrdllm_stage1_fixed_decode

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/shx/.cache/huggingface

preflight() {
  "$PYTHON" "$ADAPTER" \
    --official-cal-root "$OFFICIAL_CAL_ROOT" \
    --humaneval-root "$HUMANEVAL_ROOT" \
    --current-dataset "$CURRENT_DATASET" \
    --manifest-jsonl "$SMOKE_MANIFEST" \
    --full-manifest-jsonl "$FULL_MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --analysis-output-dir "$ANALYSIS_DIR" \
    --arm "$ARM" \
    --max-probe-length 128 \
    --seed 42 \
    --smoke-cases 12 \
    --preflight-only
}

gpu_gate() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "gpu_gate=failed reason=nvidia_smi_missing"
    return 3
  fi
  local processes
  processes=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    printf '%s\n' "gpu_gate=blocked_by_existing_gpu_process"
    printf '%s\n' "$processes"
    return 4
  fi
  local uncorrectable corrected
  uncorrectable=$(nvidia-smi -i 0 --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
  corrected=$(nvidia-smi -i 0 --query-gpu=ecc.errors.corrected.volatile.total --format=csv,noheader,nounits)
  if [[ "$uncorrectable" != "0" || "$corrected" != "0" ]]; then
    printf '%s\n' "gpu_gate=failed reason=volatile_ecc corrected=$corrected uncorrectable=$uncorrectable"
    return 5
  fi
  printf '%s\n' "gpu_gate=passed physical_gpu=0"
}

workspace_gate() {
  if [[ ! -f "$ROOT/docs/paper_agent/experiments/20260731_lrdllm_stage1_protocol_freeze.zh.md" ]]; then
    printf '%s\n' "workspace_gate=failed reason=protocol_freeze_missing"
    return 7
  fi
  if [[ -n "$(git -C "$ROOT" diff --name-only -- outputs_clean logs analysis_outputs)" ]]; then
    printf '%s\n' "workspace_gate=failed reason=tracked_raw_log_or_analysis_diff"
    return 8
  fi
  if [[ -d "$OUTPUT_DIR" && ! -f "$OUTPUT_DIR/${ARM}_raw.jsonl" ]]; then
    printf '%s\n' "workspace_gate=failed reason=output_directory_conflict path=$OUTPUT_DIR"
    return 9
  fi
  printf '%s\n' "workspace_gate=passed"
}

smoke_once() {
  "$PYTHON" "$ADAPTER" \
    --official-cal-root "$OFFICIAL_CAL_ROOT" \
    --humaneval-root "$HUMANEVAL_ROOT" \
    --current-dataset "$CURRENT_DATASET" \
    --manifest-jsonl "$SMOKE_MANIFEST" \
    --full-manifest-jsonl "$FULL_MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --analysis-output-dir "$ANALYSIS_DIR" \
    --arm "$ARM" \
    --max-probe-length 128 \
    --seed 42 \
    --smoke-cases 12 \
    --progress-every 1
}

run_smoke() {
  preflight
  workspace_gate
  gpu_gate
  mkdir -p "$(dirname "$LOG_PATH")"
  {
    printf '%s\n' "stage1_smoke_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    smoke_once
    printf '%s\n' "stage1_resume_noop_check_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    smoke_once
    "$PYTHON" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); assert x["status"]=="completed" and x["new_rows_written"]==0 and x["resume_noop"] is True; print(json.dumps({"resume_noop": True, "new_rows_written": 0}, sort_keys=True))' "$OUTPUT_DIR/${ARM}_smoke_progress.json"
    printf '%s\n' "stage1_smoke_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee -a "$LOG_PATH"
}

run_full() {
  if [[ "${ALLOW_LRDLLM_STAGE1_FULL:-0}" != "1" ]]; then
    printf '%s\n' "full_gate=disabled set ALLOW_LRDLLM_STAGE1_FULL=1 for an explicit later run"
    return 6
  fi
  gpu_gate
  "$PYTHON" "$ADAPTER" \
    --official-cal-root "$OFFICIAL_CAL_ROOT" \
    --humaneval-root "$HUMANEVAL_ROOT" \
    --current-dataset "$CURRENT_DATASET" \
    --manifest-jsonl "$FULL_MANIFEST" \
    --full-manifest-jsonl "$FULL_MANIFEST" \
    --output-dir "$ROOT/outputs_clean/lrdllm_stage1_singleline_full_20260731_v1" \
    --analysis-output-dir "$ROOT/analysis_outputs/lrdllm_stage1_singleline_full_20260731_v1" \
    --arm "$ARM" \
    --max-probe-length 128 \
    --seed 42 \
    --auto-full
}

case "${1:-help}" in
  preflight)
    preflight
    ;;
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  help|--help|-h)
    printf '%s\n' "usage: $0 {preflight|smoke|full}"
    ;;
  *)
    printf '%s\n' "unknown mode: $1" >&2
    exit 2
    ;;
esac
