#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=dreamon_context_foundation_20261015_v1
OUTPUT=analysis_outputs/dreamon_context_foundation_20261015_v1
mkdir -p "$OUTPUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Existing queue: tmux %s; inspect %s/queue_status.json and worker.log; no duplicate.\n' "$SESSION" "$OUTPUT"
  exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --restore-missing ) ]]; then
  printf 'Usage: bash scripts/launch_context_foundation_queue.sh [--restore-missing]\n' >&2
  exit 2
fi
ARG=${1:-}
tmux new-session -d -s "$SESSION" -c "$PWD" \
 "/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -u scripts/run_context_foundation_queue.py $ARG >> $OUTPUT/queue.log 2>&1"
printf 'Started finite batch A queue: %s\n' "$SESSION"
