#!/usr/bin/env bash
# One model process; bounded infrastructure/OOM retry, no restart on validity failures.
set -u
TASK_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$TASK_ROOT"
TASK_LOCAL=analysis_outputs/dreamon_threshold_trace_fork_20261112_v1/local
mkdir -p "$TASK_LOCAL"
TASK_OOM_RETRIES=0
TASK_INFRA_RETRIES=0
while true; do
    TOKENIZERS_PARALLELISM=false /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -u -m experiments.threshold_trace_fork.run
    TASK_EXIT=$?
    printf '%s exit=%s oom_retries=%s infrastructure_retries=%s\n' "$(date -u +%FT%TZ)" "$TASK_EXIT" "$TASK_OOM_RETRIES" "$TASK_INFRA_RETRIES" >> "$TASK_LOCAL/attempts.log"
    if [ "$TASK_EXIT" -eq 42 ] && [ "$TASK_OOM_RETRIES" -lt 1 ]; then
        TASK_OOM_RETRIES=$((TASK_OOM_RETRIES+1))
    elif [ "$TASK_EXIT" -eq 2 ] && [ "$TASK_INFRA_RETRIES" -lt 3 ]; then
        TASK_INFRA_RETRIES=$((TASK_INFRA_RETRIES+1))
    else
        exit "$TASK_EXIT"
    fi
done
