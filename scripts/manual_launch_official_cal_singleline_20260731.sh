#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: $0 smoke|full" >&2
  exit 2
fi

ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
PYTHON="/home/shx/miniconda3/envs/dllm_env/bin/python"
CAL_ROOT="/home/shx/.cache/dllm_infilling/calibrated_adaptive_length-741e8418"
EVAL_ROOT="/home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff"
DATASET="$EVAL_ROOT/data/HumanEval-SingleLineInfilling.jsonl.gz"
MANIFEST_ROOT="$ROOT/analysis_outputs/baseline_manifests_20260731_v1"
FULL_MANIFEST="$MANIFEST_ROOT/cal_singleline_rest_nonfrozen_manifest.jsonl"
SMOKE_MANIFEST="$MANIFEST_ROOT/cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl"
OUTPUT="$ROOT/outputs_clean/official_cal_singleline_20260731_v1"
LOG="$ROOT/logs/paper_agent/20260731_official_cal_singleline.log"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$ROOT"

COMMON=(
  "$PYTHON" experiments/p1_official_cal_adapter.py
  --official-cal-root "$CAL_ROOT"
  --humaneval-root "$EVAL_ROOT"
  --current-dataset "$DATASET"
  --benchmark-name single-line
  --full-manifest-jsonl "$FULL_MANIFEST"
  --output-dir "$OUTPUT"
  --arm official_cal_primary
)

if [[ "$MODE" == "smoke" ]]; then
  COMMAND=("${COMMON[@]}" --manifest-jsonl "$SMOKE_MANIFEST" --smoke-cases 12)
else
  COMMAND=("${COMMON[@]}" --manifest-jsonl "$FULL_MANIFEST" --auto-full)
fi

{
  echo "RUN_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "MODE=$MODE"
  printf 'COMMAND='
  printf '%q ' env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface "${COMMAND[@]}"
  printf '\n'
} >>"$LOG"

set +e
env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface "${COMMAND[@]}" >>"$LOG" 2>&1
CODE=$?
set -e

{
  echo "COMMAND_EXIT_CODE=$CODE"
  echo "RUN_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >>"$LOG"
exit "$CODE"
