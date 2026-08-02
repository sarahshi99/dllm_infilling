#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-progressive-v3-budgeted
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
RUNNER="$ROOT/repro_scripts/run_dreamon_progressive_v2.py"
MODEL=/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194
PROTOCOL="$ROOT/repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_protocol/protocol.json"
FULL_POPULATION="$ROOT/repro_results/dreamon_progressive_v2_protocol/generation_population.jsonl"
AFFECTED6_POPULATION="$ROOT/repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_protocol/affected6_generation_population.jsonl"
METHOD=v3_hard_budgeted_nonempty_oracle

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_MODULES_CACHE=/home/shx/projects/dllm_infilling/.cache/hf_modules_dreamon_repro
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/shx/projects/dllm_infilling/human-eval-infilling${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"

run_verification() {
    "$PYTHON" -m pytest -q \
        tests/test_dreamon_v3_nonempty.py \
        tests/test_dreamon_v3_budgeted.py \
        tests/test_dreamon_v3_runner.py \
        tests/test_dreamon_hard_v2_boundary.py \
        tests/test_dreamon_slot_generator.py \
        tests/test_dreamon_progressive_v2_runner.py
    "$PYTHON" -m py_compile \
        repro_scripts/dreamon_slot_generator.py \
        repro_scripts/run_dreamon_progressive_v2.py
    bash -n repro_scripts/run_dreamon_progressive_v3_hard_budgeted_nonempty_oracle.sh
    "$PYTHON" - <<'PY'
import json
from pathlib import Path

protocol = json.loads(Path(
    "repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_protocol/protocol.json"
).read_text(encoding="utf-8"))
assert protocol["protocol_version"] == 3
assert protocol["method"] == "v3_hard_budgeted_nonempty_oracle"
assert protocol["generation"]["initial_expand_budget"] == 64
assert protocol["generation"]["nonempty_guard"] is True

generator = Path("repro_scripts/dreamon_slot_generator.py").read_text(encoding="utf-8")
generation = Path("repro_scripts/run_dreamon_progressive_v2.py").read_text(encoding="utf-8").split(
    "def run_generation", 1
)[1].split("def score_completion", 1)[0]
for forbidden in (
    "canonical_solution",
    "reference_middle",
    "reference_lines",
    "check_correctness",
):
    assert forbidden not in generation
assert "select_nonempty_guarded_update" in generator
assert "clone_canvas_state" in generator
assert "nonempty_guard_no_valid_action" in generator
assert "BoundaryShift" not in generator
PY
}

output_for_stage() {
    case "$1" in
        affected6) printf '%s\n' "$ROOT/repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_affected6" ;;
        pilot) printf '%s\n' "$ROOT/repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_pilot30" ;;
        full) printf '%s\n' "$ROOT/repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_all642" ;;
        *) echo "unknown stage: $1" >&2; return 2 ;;
    esac
}

population_for_stage() {
    case "$1" in
        affected6) printf '%s\n' "$AFFECTED6_POPULATION" ;;
        pilot|full) printf '%s\n' "$FULL_POPULATION" ;;
        *) echo "unknown stage: $1" >&2; return 2 ;;
    esac
}

run_stage() {
    local stage=$1
    local output population
    output=$(output_for_stage "$stage")
    population=$(population_for_stage "$stage")
    mkdir -p "$output"
    printf 'stage_started_at=%s method=%s stage=%s head=%s\n' \
        "$(date --iso-8601=seconds)" "$METHOD" "$stage" "$(git rev-parse HEAD)" \
        | tee -a "$output/run.log"
    for mode in generate score audit gate; do
        "$PYTHON" "$RUNNER" \
            --mode "$mode" \
            --method "$METHOD" \
            --stage "$stage" \
            --output-dir "$output" \
            --model-path "$MODEL" \
            --protocol "$PROTOCOL" \
            --population "$population" \
            --device cuda \
            --workers 8 \
            --timeout 3 \
            2>&1 | tee -a "$output/run.log"
    done
    local before_hash before_rows after_hash after_rows
    before_hash=$(sha256sum "$output/predictions.jsonl" | awk '{print $1}')
    before_rows=$(wc -l < "$output/predictions.jsonl")
    "$PYTHON" "$RUNNER" \
        --mode generate \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --model-path "$MODEL" \
        --protocol "$PROTOCOL" \
        --population "$population" \
        --device cuda \
        2>&1 | tee -a "$output/run.log"
    after_hash=$(sha256sum "$output/predictions.jsonl" | awk '{print $1}')
    after_rows=$(wc -l < "$output/predictions.jsonl")
    if [[ "$before_hash" != "$after_hash" || "$before_rows" != "$after_rows" ]]; then
        echo "deterministic row-level resume/dedup equality failed" >&2
        return 1
    fi
    printf 'deterministic_resume_equality=passed rows=%s sha256=%s\n' \
        "$after_rows" "$after_hash" | tee -a "$output/run.log"
    printf 'stage_finished_at=%s method=%s stage=%s\n' \
        "$(date --iso-8601=seconds)" "$METHOD" "$stage" | tee -a "$output/run.log"
}

pilot_authorizes_full() {
    "$PYTHON" - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path(
    "repro_results/dreamon_progressive_v3_hard_budgeted_nonempty_oracle_pilot30/gate.json"
).read_text(encoding="utf-8"))
raise SystemExit(0 if gate["full_authorized"] else 1)
PY
}

case "${1:-all}" in
    verify) run_verification ;;
    affected6) run_verification; run_stage affected6 ;;
    pilot) run_verification; run_stage pilot ;;
    full) run_verification; run_stage full ;;
    all)
        run_verification
        run_stage affected6
        run_stage pilot
        if pilot_authorizes_full; then
            run_stage full
        else
            printf 'full_not_authorized reason=pilot_pass_below_18_or_engineering_gate_failed\n'
        fi
        ;;
    *) echo "usage: $0 {verify|affected6|pilot|full|all}" >&2; exit 2 ;;
esac
