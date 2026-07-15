#!/usr/bin/env bash
set -euo pipefail

REPO=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-method-portfolio-v0
SESSION=m2-constraint-homotopy-20260715-v1
LOG="$REPO/logs/paper_agent/20260715_m2_constraint_homotopy_v1.log"

mkdir -p "$(dirname "$LOG")"

# --auto-full is guarded in the runner and can only promote the 12-case smoke
# to the 148-task RandomSpanLight population.  It cannot reach 927 or 5079.
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/m2_constraint_homotopy.py --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl --gradual-output-dir outputs_clean/m2_gradual_randomspanlight_20260715_v1 --abrupt-output-dir outputs_clean/m2_abrupt_randomspanlight_20260715_v1 --compact-dir analysis_outputs/m2_constraint_homotopy_20260715_v1 --smoke-cases 12 --auto-full 2>&1 | tee '$LOG'"

echo "Started tmux session: $SESSION"
echo "Log: $LOG"
