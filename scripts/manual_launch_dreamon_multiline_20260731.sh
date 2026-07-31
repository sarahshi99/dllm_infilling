#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-}"
MIN_GEN_LEN="${2:-4}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: $0 smoke|full 4|8|16|32|64" >&2
  exit 2
fi
if [[ "$MIN_GEN_LEN" != "4" && "$MIN_GEN_LEN" != "8" && "$MIN_GEN_LEN" != "16" && "$MIN_GEN_LEN" != "32" && "$MIN_GEN_LEN" != "64" ]]; then
  echo "invalid min_gen_len: $MIN_GEN_LEN" >&2
  exit 2
fi

ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
PYTHON="/home/shx/miniconda3/envs/dllm_env/bin/python"
SOURCE_ROOT="/tmp/dllm_infilling_protocol_audit_20260728/DreamOn"
EVALUATOR_ROOT="/home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff"
MODEL_SNAPSHOT="/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194"
MANIFEST_ROOT="$ROOT/analysis_outputs/baseline_manifests_20260731_v1"
FULL_MANIFEST="$MANIFEST_ROOT/lrdllm_common_multiline_nonfrozen_manifest.jsonl"
SMOKE_MANIFEST="$MANIFEST_ROOT/dreamon_multiline_technical_smoke12_manifest.jsonl"
SUMMARY="$MANIFEST_ROOT/lrdllm_common_manifest_summary.json"
OUTPUT="$ROOT/outputs_clean/dreamon_multiline_20260731_v1"
LOG="$ROOT/logs/paper_agent/20260731_dreamon_multiline_min${MIN_GEN_LEN}.log"
MANIFEST="$FULL_MANIFEST"
if [[ "$MODE" == "smoke" ]]; then MANIFEST="$SMOKE_MANIFEST"; fi

mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#GPU_PIDS[@]} > 0 )) && [[ "${ALLOW_SHARED_GPU:-0}" != "1" ]]; then
  echo "refusing to launch: physical GPU 0 already has compute PIDs: ${GPU_PIDS[*]}" >&2
  exit 3
fi

mkdir -p "$OUTPUT" "$(dirname "$LOG")" /tmp/hf_modules_dreamon_20260731
cd "$ROOT"
COMMAND=(
  "$PYTHON" experiments/dreamon_singleline_adapter.py
  --source-root "$SOURCE_ROOT"
  --evaluator-root "$EVALUATOR_ROOT"
  --model-snapshot "$MODEL_SNAPSHOT"
  --model-profile dreamon
  --dataset-profile multiline
  --manifest-jsonl "$MANIFEST"
  --full-manifest-jsonl "$FULL_MANIFEST"
  --manifest-summary-json "$SUMMARY"
  --output-dir "$OUTPUT"
  --min-gen-len "$MIN_GEN_LEN"
  --mode "$MODE"
  --seed 42
  --progress-every 1
)
ENVIRONMENT=(
  CUDA_VISIBLE_DEVICES=0
  TOKENIZERS_PARALLELISM=false
  DLLM_DISABLE_FLASH_ATTN=1
  PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages
  HF_MODULES_CACHE=/tmp/hf_modules_dreamon_20260731
  TRANSFORMERS_OFFLINE=1
  HF_HUB_OFFLINE=1
  HF_HOME=/home/shx/.cache/huggingface
)
{
  echo "RUN_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "MODE=$MODE"
  echo "MIN_GEN_LEN=$MIN_GEN_LEN"
  echo "ALLOW_SHARED_GPU=${ALLOW_SHARED_GPU:-0}"
  echo "PREEXISTING_GPU_PIDS=${GPU_PIDS[*]:-none}"
  printf 'COMMAND='
  printf '%q ' env "${ENVIRONMENT[@]}" "${COMMAND[@]}"
  printf '\n'
} >>"$LOG"

set +e
env "${ENVIRONMENT[@]}" "${COMMAND[@]}" >>"$LOG" 2>&1
CODE=$?
set -e
{
  echo "COMMAND_EXIT_CODE=$CODE"
  echo "RUN_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >>"$LOG"
exit "$CODE"
