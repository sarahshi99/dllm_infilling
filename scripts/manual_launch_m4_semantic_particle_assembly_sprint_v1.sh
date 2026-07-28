#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
PYTHON="/home/shx/miniconda3/envs/dllm_env/bin/python"
LOG="$ROOT/logs/paper_agent/20260728_m4_semantic_particle_assembly_multilinecore_v1.log"
BANK="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/phase5-method-falsification/outputs_clean/phase6_multiline_candidate_bank_20260712_v1/candidate_bank_raw.jsonl"
MANIFEST="$ROOT/analysis_outputs/m4_semantic_particle_assembly_20260728_multilinecore_manifest_v1/selection_manifest.json"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface \
  "$PYTHON" experiments/m4_semantic_particle_assembly.py \
  --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-MultiLineInfilling.jsonl \
  --candidate-bank-raw "$BANK" \
  --selection-manifest "$MANIFEST" \
  --source-config HumanEval-MultiLineInfilling \
  --population-label multiline_core_148_hash_selected \
  --best-output-dir outputs_clean/m4_best_multilinecore_20260728_v1 \
  --assembly-output-dir outputs_clean/m4_assembly_multilinecore_20260728_v1 \
  --repair-output-dir outputs_clean/m4_repair_multilinecore_20260728_v1 \
  --compact-dir analysis_outputs/m4_semantic_particle_assembly_20260728_multilinecore_v1 \
  --smoke-cases 12 --repair --auto-full >"$LOG" 2>&1
