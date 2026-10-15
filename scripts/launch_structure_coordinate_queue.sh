#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
SESSION=dreamon_structure_coordinate_fix_20261015_v1
OUTPUT=analysis_outputs/structure_coordinate_fix_20261015_v2
mkdir -p "$OUTPUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Existing correction queue; inspect %s/queue_status.json and %s/worker.log\n' "$OUTPUT" "$OUTPUT"
  exit 0
fi
if [[ -f "$OUTPUT/completion_matrix.json" ]]; then
  set +e
  /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python - <<'PY_CHECK'
from pathlib import Path
import json,hashlib,subprocess
root=Path('analysis_outputs/structure_coordinate_fix_20261015_v2')
m=json.loads((root/'completion_matrix.json').read_text())
if m.get('status')=='complete':
    spec=json.loads((root/'run_manifest.json').read_text())
    assert hashlib.sha256((root/'new_rows.jsonl.gz').read_bytes()).hexdigest()==m['new_rows_sha256']
    assert hashlib.sha256((root/'influence_manifest.json').read_bytes()).hexdigest()==spec['influence_sha256']
    for path,blob in spec['code_versions'].items():
        assert subprocess.check_output(['git','hash-object',path],text=True).strip()==blob
    print('Exact-version correction already complete; evidence summary verified; no second queue.')
    raise SystemExit(42)
PY_CHECK
  status=$?
  set -e
  if [[ "$status" == 42 ]]; then exit 0; fi
  if [[ "$status" != 0 ]]; then exit "$status"; fi
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
 '/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -u scripts/run_structure_coordinate_queue.py >> analysis_outputs/structure_coordinate_fix_20261015_v2/queue.log 2>&1'
printf 'Started finite correction queue: %s\n' "$SESSION"
