#!/usr/bin/env bash
set -uo pipefail

ARM="${1:-}"
BENCHMARK="${2:-}"
MODE="${3:-}"
if [[ "$ARM" != "daedal_dynamic" && "$ARM" != "daedal_fixed8_control" ]]; then
  echo "usage: $0 daedal_dynamic|daedal_fixed8_control singleline|multiline smoke|full" >&2
  exit 2
fi
if [[ "$BENCHMARK" != "singleline" && "$BENCHMARK" != "multiline" ]]; then
  echo "invalid benchmark: $BENCHMARK" >&2
  exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "invalid mode: $MODE" >&2
  exit 2
fi

ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
PYTHON="/home/shx/miniconda3/envs/dllm_env/bin/python"
CAL_ROOT="/home/shx/.cache/dllm_infilling/calibrated_adaptive_length-741e8418"
EVALUATOR_ROOT="/home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff"
MODEL_SNAPSHOT="/home/shx/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
OUTPUT="$ROOT/outputs_clean/daedal_fim_${BENCHMARK}_20260731_v1"
LOG="$ROOT/logs/paper_agent/20260731_daedal_fim_${BENCHMARK}_${ARM}.log"

if [[ "$BENCHMARK" == "singleline" ]]; then
  FULL_MANIFEST="$ROOT/analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_manifest.jsonl"
  SMOKE_MANIFEST="$ROOT/analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl"
else
  FULL_MANIFEST="$ROOT/analysis_outputs/official_cal_corrected_protocol_20260715_v1/cal_rest_common_manifest.jsonl"
  SMOKE_MANIFEST="$ROOT/analysis_outputs/official_cal_corrected_protocol_20260715_v1/smoke_manifest.jsonl"
fi
MANIFEST="$FULL_MANIFEST"
if [[ "$MODE" == "smoke" ]]; then MANIFEST="$SMOKE_MANIFEST"; fi

mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#GPU_PIDS[@]} > 0 )) && [[ "${ALLOW_SHARED_GPU:-0}" != "1" ]]; then
  echo "refusing to launch: physical GPU 0 already has compute PIDs: ${GPU_PIDS[*]}" >&2
  exit 3
fi

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$ROOT"
COMMAND=(
  "$PYTHON" experiments/daedal_fim_adapter.py
  --benchmark "$BENCHMARK"
  --manifest-jsonl "$MANIFEST"
  --full-manifest-jsonl "$FULL_MANIFEST"
  --cal-root "$CAL_ROOT"
  --evaluator-root "$EVALUATOR_ROOT"
  --model-snapshot "$MODEL_SNAPSHOT"
  --output-dir "$OUTPUT"
  --arm "$ARM"
  --mode "$MODE"
  --seed 42
  --progress-every 1
)
ENVIRONMENT=(CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HOME=/home/shx/.cache/huggingface)
{
  echo "RUN_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
