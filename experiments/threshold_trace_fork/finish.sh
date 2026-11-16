#!/usr/bin/env bash
# Sequential continuation after the generation/fork model exits.
set -u
TASK_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$TASK_ROOT"
TASK_OUT=analysis_outputs/dreamon_threshold_trace_fork_20261112_v1
TASK_LOCAL="$TASK_OUT/local"
TASK_PY=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
while tmux has-session -t threshold_trace_fork_20261112 2>/dev/null; do
    [ -f "$TASK_LOCAL/PAUSE_REQUESTED" ] && exit 4
    [ "$(tmux display-message -p -t threshold_trace_fork_20261112 '#{pane_dead}' 2>/dev/null)" = 1 ] && break
    sleep 15
done
[ -f "$TASK_LOCAL/PAUSE_REQUESTED" ] && exit 4
TASK_VERIFY_NEEDED=$($TASK_PY - <<'PY'
import json,sqlite3
from pathlib import Path
p=Path('analysis_outputs/dreamon_threshold_trace_fork_20261112_v1')
db=sqlite3.connect(p/'local/results.sqlite')
def count(kind):return db.execute('SELECT COUNT(*) FROM results WHERE kind=?',(kind,)).fetchone()[0]
started=db.execute("SELECT 1 FROM results WHERE kind='meta' AND key='original_validation_version'").fetchone() is not None
assert json.loads((p/'status.json').read_text())['phase']=='completed' or started,'primary pipeline did not exit successfully'
assert (count('replay'),count('fork'),count('multiseed'),count('teacher'))==(5168,2920,54,64),'primary pipeline incomplete; do not start dependent validation'
print(int(count('original_full_validation')!=5168 or count('teacher_logspace_validation')!=64))
PY
) || exit 3
if [ "$TASK_VERIFY_NEEDED" = 1 ]; then
    TOKENIZERS_PARALLELISM=false "$TASK_PY" -u -m experiments.threshold_trace_fork.verify_all > "$TASK_LOCAL/verify_all.log" 2>&1
    TASK_EXIT=$?
    printf '%s original_validation_exit=%s\n' "$(date -u +%FT%TZ)" "$TASK_EXIT" >> "$TASK_LOCAL/validation_attempts.log"
    [ "$TASK_EXIT" -eq 0 ] || exit "$TASK_EXIT"
fi
"$TASK_PY" -u -m experiments.threshold_trace_fork.enrich > "$TASK_LOCAL/enrich.log" 2>&1
TASK_EXIT=$?
printf '%s enrichment_exit=%s\n' "$(date -u +%FT%TZ)" "$TASK_EXIT" >> "$TASK_LOCAL/validation_attempts.log"
exit "$TASK_EXIT"
