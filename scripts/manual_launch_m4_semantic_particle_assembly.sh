#!/usr/bin/env bash
set -euo pipefail

REPO=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-method-portfolio-v0
SESSION=m4-semantic-particle-assembly-v0
LOG="$REPO/logs/paper_agent/20260713_m4_semantic_particle_assembly_v0.log"
BANK=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/phase5-method-falsification/outputs_clean/phase5_randomspanlight_candidate_bank_20260711_v1/candidate_bank_raw.jsonl

mkdir -p "$(dirname "$LOG")"

tmux new-session -d -s "$SESSION" -c "$REPO" \
  "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/m4_semantic_particle_assembly.py --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl --candidate-bank-raw '$BANK' --best-output-dir outputs_clean/m4_best_randomspanlight_20260713_v0 --assembly-output-dir outputs_clean/m4_assembly_randomspanlight_20260713_v0 --repair-output-dir outputs_clean/m4_repair_randomspanlight_20260713_v0 --compact-dir analysis_outputs/m4_semantic_particle_assembly_20260713_v0 --smoke-cases 12 --repair --auto-full 2>&1 | tee '$LOG'"

echo "Started tmux session: $SESSION"
echo "Log: $LOG"
