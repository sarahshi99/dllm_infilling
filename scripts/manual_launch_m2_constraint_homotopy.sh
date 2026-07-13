#!/usr/bin/env bash
set -euo pipefail

REPO=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-method-portfolio-v0
SESSION=m2-constraint-homotopy-v0
LOG="$REPO/logs/paper_agent/20260713_m2_constraint_homotopy_v0.log"

mkdir -p "$(dirname "$LOG")"

tmux new-session -d -s "$SESSION" -c "$REPO" \
  "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/m2_constraint_homotopy.py --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl --gradual-output-dir outputs_clean/m2_gradual_randomspanlight_20260713_v0 --abrupt-output-dir outputs_clean/m2_abrupt_randomspanlight_20260713_v0 --compact-dir analysis_outputs/m2_constraint_homotopy_20260713_v0 --smoke-cases 12 --auto-full 2>&1 | tee '$LOG'"

echo "Started tmux session: $SESSION"
echo "Log: $LOG"
