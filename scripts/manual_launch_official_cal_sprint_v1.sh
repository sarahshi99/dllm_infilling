#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
PYTHON="/home/shx/miniconda3/envs/dllm_env/bin/python"
AUDIT="$ROOT/analysis_outputs/official_cal_corrected_protocol_20260715_v1"
OUT="$ROOT/outputs_clean/official_cal_primary_20260715_sprint_v1"
LOG="$ROOT/logs/paper_agent/20260715_official_cal_primary_sprint_v1.log"
mkdir -p "$(dirname "$LOG")" "$OUT"
# Pinned upstream's `length_bias.py` fitting utility imports SciPy, but the
# audited runtime path `llada_cal.llada_cal.generate` does not.  Do not block
# the reproduction on an unused fitting dependency; the adapter smoke proves
# the actual pinned decoder/evaluator import path.
cd "$ROOT"
env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface \
  "$PYTHON" experiments/p1_official_cal_adapter.py \
  --official-cal-root /home/shx/.cache/dllm_infilling/calibrated_adaptive_length-741e8418 \
  --humaneval-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff \
  --current-dataset /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-MultiLineInfilling.jsonl \
  --manifest-jsonl "$AUDIT/smoke_manifest.jsonl" --output-dir "$OUT" --arm official_cal_primary --smoke-cases 12 >"$LOG" 2>&1
env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface \
  "$PYTHON" experiments/p1_official_cal_adapter.py \
  --official-cal-root /home/shx/.cache/dllm_infilling/calibrated_adaptive_length-741e8418 \
  --humaneval-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff \
  --current-dataset /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-MultiLineInfilling.jsonl \
  --manifest-jsonl "$AUDIT/cal_rest_common_manifest.jsonl" --output-dir "$OUT" --arm official_cal_primary --auto-full >>"$LOG" 2>&1
