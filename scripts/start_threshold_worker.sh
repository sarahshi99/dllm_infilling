#!/usr/bin/env bash
set -euo pipefail
SHARED=${THRESHOLD_SHARED:-/home/shx/projects/dllm_infilling/threshold_eval_20261104_v1}
ROLE=""
if [[ ${1:-} == --role ]]; then ROLE=${2:-}; fi
case "$ROLE" in nohead|trained|timing) ;; *) echo 'usage: start_worker.sh --role nohead|trained|timing' >&2; exit 2;; esac
# READY is code publication only. There is no GPU lock or free-memory check.
if [[ ! -f "$SHARED/READY.json" ]]; then echo 'public code not READY' >&2; exit 3; fi
exec 9>"$SHARED/$ROLE.lock"
if ! flock -n 9; then echo "$ROLE already running; no second model loaded"; exit 0; fi
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
WORKTREE="$($PYTHON -c 'import json,sys;print(json.load(open(sys.argv[1]))["worktree"])' "$SHARED/READY.json")"
cd "$WORKTREE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# Preserve CUDA_VISIBLE_DEVICES exactly as inherited. cuda:0 is logical only.
PHASE=quality
if [[ "$ROLE" == timing ]]; then PHASE=timing; fi
OOM_RETRIES=0
INFRA_RETRIES=0
while :; do
  args=("$PYTHON" -u -m experiments.threshold_eval.runner --phase "$PHASE")
  if [[ "$ROLE" != timing ]]; then args+=(--role "$ROLE"); fi
  set +e
  "${args[@]}" >> "$SHARED/$ROLE.worker.log" 2>&1
  CODE=$?
  set -e
  if [[ "$CODE" == 0 ]]; then break; fi
  # Runner uses 2 for code/contract errors; those are not infrastructure retries.
  if [[ "$CODE" == 2 ]]; then break; fi
  if [[ "$CODE" == 42 && "$OOM_RETRIES" -lt 1 ]]; then
    OOM_RETRIES=$((OOM_RETRIES+1))
  elif [[ "$CODE" != 42 && "$INFRA_RETRIES" -lt 3 ]]; then
    INFRA_RETRIES=$((INFRA_RETRIES+1))
  else
    break
  fi
  # Failed child has exited, releasing only its own CUDA context; immediately resume.
  echo "$(date -u +%FT%TZ) recovery code=$CODE oom_retries=$OOM_RETRIES infra_retries=$INFRA_RETRIES" >> "$SHARED/$ROLE.worker.log"
done
"$PYTHON" - "$SHARED" "$ROLE" "$CODE" "$OOM_RETRIES" "$INFRA_RETRIES" <<'PY'
import json,os,sys,time
from pathlib import Path
root,role,code,oom,infra=sys.argv[1:]
p=Path(root)/(role+'.exit.json');tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps(dict(role=role,exit_code=int(code),oom_retries=int(oom),infrastructure_retries=int(infra),launcher_pid=os.getppid(),at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))+'\n');tmp.replace(p)
PY
exit "$CODE"
