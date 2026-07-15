#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
PYTHON="/home/shx/miniconda3/envs/dllm_env/bin/python"
LOG="$ROOT/logs/paper_agent/20260715_m4_semantic_particle_assembly_sprint_v1.log"
BANK="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/phase5-method-falsification/outputs_clean/phase5_randomspanlight_candidate_bank_20260711_v1/candidate_bank_raw.jsonl"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface \
  "$PYTHON" experiments/m4_semantic_particle_assembly.py \
  --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl \
  --candidate-bank-raw "$BANK" \
  --best-output-dir outputs_clean/m4_best_randomspanlight_20260715_sprint_v1 \
  --assembly-output-dir outputs_clean/m4_assembly_randomspanlight_20260715_sprint_v1 \
  --repair-output-dir outputs_clean/m4_repair_randomspanlight_20260715_sprint_v1 \
  --compact-dir analysis_outputs/m4_semantic_particle_assembly_20260715_sprint_v1 \
  --smoke-cases 12 --repair --auto-full >"$LOG" 2>&1
