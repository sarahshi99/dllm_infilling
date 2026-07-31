#!/usr/bin/env bash
set -uo pipefail

ARM="${1:-}"
MODE="${2:-}"
DATASET="${3:-singleline}"
if [[ "$ARM" != "lrdllm_primary" && "$ARM" != "fixed64_common_protocol" ]]; then
  echo "usage: $0 lrdllm_primary|fixed64_common_protocol technical-smoke|mechanism-smoke|full singleline|randomspan|multiline" >&2
  exit 2
fi
if [[ "$MODE" != "technical-smoke" && "$MODE" != "mechanism-smoke" && "$MODE" != "full" ]]; then
  echo "invalid mode: $MODE" >&2
  exit 2
fi
if [[ "$DATASET" != "singleline" && "$DATASET" != "randomspan" && "$DATASET" != "multiline" ]]; then
  echo "invalid dataset: $DATASET" >&2
  exit 2
fi
if [[ "$MODE" != "full" && "$DATASET" != "singleline" ]]; then
  echo "$MODE is frozen only for singleline" >&2
  exit 2
fi

ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
PYTHON="/home/shx/miniconda3/envs/dllm_env/bin/python"
MANIFEST_ROOT="$ROOT/analysis_outputs/baseline_manifests_20260731_v1"
EVALUATOR_ROOT="/home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff"
MODEL_SNAPSHOT="/home/shx/.cache/huggingface/hub/models--Dream-org--Dream-Coder-v0-Base-7B/snapshots/2346ccd3be517d0d314152b988a3b9bafa7d6d63"
FULL_MANIFEST="$MANIFEST_ROOT/lrdllm_common_${DATASET}_nonfrozen_manifest.jsonl"
SUMMARY="$MANIFEST_ROOT/lrdllm_common_manifest_summary.json"
OUTPUT="$ROOT/outputs_clean/lrdllm_dreamcoder_${DATASET}_20260731_v1"
LOG="$ROOT/logs/paper_agent/20260731_lrdllm_dreamcoder_${DATASET}_${ARM}.log"

if [[ "$MODE" == "technical-smoke" ]]; then
  MANIFEST="$MANIFEST_ROOT/lrdllm_common_singleline_technical_smoke12_manifest.jsonl"
elif [[ "$MODE" == "mechanism-smoke" ]]; then
  MANIFEST="$MANIFEST_ROOT/lrdllm_common_singleline_mechanism_smoke64_manifest.jsonl"
else
  MANIFEST="$FULL_MANIFEST"
fi

mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#GPU_PIDS[@]} > 0 )); then
  echo "refusing to launch: physical GPU 0 already has compute PIDs: ${GPU_PIDS[*]}" >&2
  exit 3
fi

mkdir -p "$OUTPUT" "$(dirname "$LOG")" /tmp/hf_modules_lrdllm_dreamcoder_20260731
cd "$ROOT"
COMMAND=(
  "$PYTHON" experiments/lrdllm_dreamcoder_adapter.py
  --dataset "$DATASET"
  --manifest-jsonl "$MANIFEST"
  --full-manifest-jsonl "$FULL_MANIFEST"
  --manifest-summary-json "$SUMMARY"
  --evaluator-root "$EVALUATOR_ROOT"
  --model-snapshot "$MODEL_SNAPSHOT"
  --output-dir "$OUTPUT"
  --arm "$ARM"
  --mode "$MODE"
  --seed 42
  --progress-every 1
)
ENVIRONMENT=(
  CUDA_VISIBLE_DEVICES=0
  TOKENIZERS_PARALLELISM=false
  DLLM_DISABLE_FLASH_ATTN=1
  PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages
  HF_MODULES_CACHE=/tmp/hf_modules_lrdllm_dreamcoder_20260731
  TRANSFORMERS_OFFLINE=1
  HF_HUB_OFFLINE=1
  HF_HOME=/home/shx/.cache/huggingface
)

{
  echo "RUN_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "ARM=$ARM"
  echo "MODE=$MODE"
  echo "DATASET=$DATASET"
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
