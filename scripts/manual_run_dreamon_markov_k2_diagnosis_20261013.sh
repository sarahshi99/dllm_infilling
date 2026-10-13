#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
ORIGINAL_CONFIG=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-markov-k2-eval-v1/analysis_outputs/dreamon_markov_k2_full_20261011_v1/run_config.json
OUTPUT=analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1
mkdir -p "$OUTPUT"
# Occupied H200 exits before loading. No background watcher or additional methods.
set +e
"$PYTHON" experiments/run_dreamon_markov_k2_diagnosis.py \
  --original-config "$ORIGINAL_CONFIG" --output-dir "$OUTPUT" --budget-gpu-hours 12
status=$?
set -e
if [[ -f "$OUTPUT/run_manifest.json" ]]; then
  "$PYTHON" analysis/analyze_dreamon_markov_k2_diagnosis.py --result-dir "$OUTPUT"
fi
exit "$status"
