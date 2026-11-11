#!/usr/bin/env bash
set -euo pipefail
export THRESHOLD_TASK_ID=dreamon_multiline_threshold_20261111_v1
export THRESHOLD_SHARED=/home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
WORKTREE=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["worktree"])' "$THRESHOLD_SHARED/run_spec.json")
exec bash "$WORKTREE/scripts/start_threshold_worker.sh" "$@"
