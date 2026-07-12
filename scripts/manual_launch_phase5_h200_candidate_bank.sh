#!/usr/bin/env bash
set -euo pipefail

REPO=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/phase5-method-falsification
SESSION=phase5-randomspanlight-bank
LOG="$REPO/logs/paper_agent/20260712_phase5_randomspanlight_candidate_bank.log"

mkdir -p "$(dirname "$LOG")"

tmux new-session -d -s "$SESSION" -c "$REPO" \
  "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/phase5_randomspanlight_candidate_bank.py run --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl --output-dir outputs_clean/phase5_randomspanlight_candidate_bank_20260711_v1 --compact-dir analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1 --smoke-cases 12 --auto-full 2>&1 | tee '$LOG'"

echo "Started tmux session: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"
