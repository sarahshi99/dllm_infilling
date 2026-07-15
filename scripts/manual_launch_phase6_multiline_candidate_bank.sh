#!/usr/bin/env bash
set -euo pipefail

REPO=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-method-portfolio-v0
SESSION=phase6-m1-multiline-v3
LOG="$REPO/logs/paper_agent/20260713_m1_multiline_v3.log"

mkdir -p "$(dirname "$LOG")"

tmux new-session -d -s "$SESSION" -c "$REPO" \
  "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/phase6_multiline_candidate_bank.py run --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-MultiLineInfilling.jsonl --output-dir outputs_clean/m1_stage1_multiline_20260713_v3 --generic-output-dir outputs_clean/m1_generic_multiline_20260713_v3 --m1-output-dir outputs_clean/m1_dependency_cone_multiline_20260713_v3 --compact-dir analysis_outputs/m1_multiline_20260713_v3 --smoke-cases 12 2>&1 | tee '$LOG'"

echo "Started tmux session: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"
