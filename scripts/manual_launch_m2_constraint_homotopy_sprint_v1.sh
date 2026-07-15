#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
PYTHON="/home/shx/miniconda3/envs/dllm_env/bin/python"
LOG="$ROOT/logs/paper_agent/20260715_m2_constraint_homotopy_sprint_v1.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface \
  "$PYTHON" experiments/m2_constraint_homotopy.py \
  --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl \
  --vanilla-output-dir outputs_clean/m2_vanilla_randomspanlight_20260715_sprint_v1 \
  --gradual-output-dir outputs_clean/m2_gradual_randomspanlight_20260715_sprint_v1 \
  --abrupt-output-dir outputs_clean/m2_abrupt_randomspanlight_20260715_sprint_v1 \
  --compact-dir analysis_outputs/m2_constraint_homotopy_20260715_sprint_v1 \
  --smoke-cases 12 --auto-full >"$LOG" 2>&1
