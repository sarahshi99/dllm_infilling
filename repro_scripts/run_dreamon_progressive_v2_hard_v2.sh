#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-progressive-v2-slots
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
RUNNER="$ROOT/repro_scripts/run_dreamon_progressive_v2.py"
MODEL=/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194
PROTOCOL="$ROOT/repro_results/dreamon_progressive_v2_hard_v2_protocol/protocol.json"
POPULATION="$ROOT/repro_results/dreamon_progressive_v2_protocol/generation_population.jsonl"
METHOD=v2_hard_v2_boundary

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
        tests/test_dreamon_hard_v2_boundary.py \
        tests/test_dreamon_slot_generator.py \
        tests/test_dreamon_progressive_v2_runner.py
    "$PYTHON" -m py_compile \
        repro_scripts/dreamon_slot_generator.py \
        repro_scripts/run_dreamon_progressive_v2.py
    bash -n repro_scripts/run_dreamon_progressive_v2_hard_v2.sh
    "$PYTHON" - <<'PY'
from pathlib import Path

source = Path("repro_scripts/dreamon_slot_generator.py").read_text(encoding="utf-8")
constrained = source.split("def constrained_active_logits", 1)[1].split("def select_update", 1)[0]
if "newline_token_ids" in constrained or "newline_token_map" in constrained:
    raise SystemExit("blanket hard-slot newline logit ban remains")
PY
}

output_for_stage() {
    case "$1" in
        smoke) printf '%s\n' "$ROOT/repro_results/dreamon_progressive_v2_hard_v2_smoke5" ;;
        pilot) printf '%s\n' "$ROOT/repro_results/dreamon_progressive_v2_hard_v2_pilot30" ;;
        full) printf '%s\n' "$ROOT/repro_results/dreamon_progressive_v2_hard_v2_all642" ;;
        *) echo "unknown stage: $1" >&2; return 2 ;;
    esac
}

run_stage() {
    local stage=$1
    local output
    output=$(output_for_stage "$stage")
    mkdir -p "$output"
    printf 'stage_started_at=%s method=%s stage=%s head=%s\n' \
        "$(date --iso-8601=seconds)" "$METHOD" "$stage" "$(git rev-parse HEAD)" \
        | tee -a "$output/run.log"
    "$PYTHON" "$RUNNER" \
        --mode generate \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --model-path "$MODEL" \
        --protocol "$PROTOCOL" \
        --population "$POPULATION" \
        --device cuda \
        2>&1 | tee -a "$output/run.log"
    "$PYTHON" "$RUNNER" \
        --mode score \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --model-path "$MODEL" \
        --protocol "$PROTOCOL" \
        --population "$POPULATION" \
        --workers 8 \
        --timeout 3 \
        2>&1 | tee -a "$output/run.log"
    "$PYTHON" "$RUNNER" \
        --mode audit \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --protocol "$PROTOCOL" \
        --population "$POPULATION" \
        2>&1 | tee -a "$output/run.log"
    "$PYTHON" "$RUNNER" \
        --mode gate \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --protocol "$PROTOCOL" \
        --population "$POPULATION" \
        2>&1 | tee -a "$output/run.log"
    if [[ "$stage" == smoke ]]; then
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
            --population "$POPULATION" \
            --device cuda \
            2>&1 | tee -a "$output/run.log"
        after_hash=$(sha256sum "$output/predictions.jsonl" | awk '{print $1}')
        after_rows=$(wc -l < "$output/predictions.jsonl")
        if [[ "$before_hash" != "$after_hash" || "$before_rows" != "$after_rows" ]]; then
            echo "smoke deterministic resume/dedup equality failed" >&2
            return 1
        fi
        printf 'smoke_deterministic_resume_equality=passed rows=%s sha256=%s\n' \
            "$after_rows" "$after_hash" | tee -a "$output/run.log"
    fi
    printf 'stage_finished_at=%s method=%s stage=%s\n' \
        "$(date --iso-8601=seconds)" "$METHOD" "$stage" | tee -a "$output/run.log"
}

case "${1:-all}" in
    verify) run_verification ;;
    smoke) run_verification; run_stage smoke ;;
    pilot) run_verification; run_stage pilot ;;
    full) run_verification; run_stage full ;;
    all) run_verification; run_stage smoke; run_stage pilot; run_stage full ;;
    *) echo "usage: $0 {verify|smoke|pilot|full|all}" >&2; exit 2 ;;
esac
