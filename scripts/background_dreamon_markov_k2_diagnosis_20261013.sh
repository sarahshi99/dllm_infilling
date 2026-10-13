#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
bash scripts/manual_run_dreamon_markov_k2_diagnosis_20261013.sh
status=$?
"$PYTHON" scripts/finalize_dreamon_markov_k2_diagnosis_20261013.py "$status"
finalize_status=$?
printf 'Diagnosis exit=%s, finalization exit=%s\n' "$status" "$finalize_status"
exit "$status"
