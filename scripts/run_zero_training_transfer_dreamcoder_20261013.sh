#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
OUTPUT=analysis_outputs/zero_training_transfer_20261013/dreamcoder_base
mkdir -p "$OUTPUT"
set +e
"$PYTHON" experiments/run_zero_training_transfer.py --target dreamcoder_base \
  --target-registry analysis_outputs/zero_training_transfer_20261013/controller/target_registry.json \
  --output-dir "$OUTPUT" > "$OUTPUT/runner.log" 2>&1
status=$?
set -e
if [[ -f "$OUTPUT/per_case.jsonl.gz" ]]; then
  "$PYTHON" analysis/analyze_zero_training_transfer.py --result-dir "$OUTPUT" > "$OUTPUT/analysis.log" 2>&1
fi
"$PYTHON" - "$OUTPUT" "$status" <<'PY'
import datetime,json,pathlib,sys
out=pathlib.Path(sys.argv[1]); status=int(sys.argv[2])
(out/'exit.json').write_text(json.dumps({'runner_exit':status,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
exit "$status"
