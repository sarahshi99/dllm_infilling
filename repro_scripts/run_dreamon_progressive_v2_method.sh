#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-progressive-v2-slots
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
RUNNER="$ROOT/repro_scripts/run_dreamon_progressive_v2.py"
MODEL=/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194

if [[ $# -ne 1 ]]; then
    echo "usage: $0 {v2_hard|v2_opentail|joint_opentail}" >&2
    exit 2
fi

METHOD=$1
case "$METHOD" in
    v2_hard)
        STEM=dreamon_progressive_v2_hard
        ;;
    v2_opentail)
        STEM=dreamon_progressive_v2_opentail
        ;;
    joint_opentail)
        STEM=dreamon_joint_opentail
        ;;
    *)
        echo "unknown method: $METHOD" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_MODULES_CACHE=/home/shx/projects/dllm_infilling/.cache/hf_modules_dreamon_repro
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/shx/projects/dllm_infilling/human-eval-infilling${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
"$PYTHON" -m pytest -q \
    tests/test_dreamon_slot_generator.py \
    tests/test_dreamon_progressive_v2_runner.py
"$PYTHON" -m py_compile \
    repro_scripts/dreamon_slot_generator.py \
    repro_scripts/run_dreamon_progressive_v2.py

run_stage() {
    local stage=$1
    local output
    case "$stage" in
        smoke) output="$ROOT/repro_results/${STEM}_smoke5_v1" ;;
        pilot) output="$ROOT/repro_results/${STEM}_pilot30_v1" ;;
        full) output="$ROOT/repro_results/${STEM}_all642" ;;
    esac
    mkdir -p "$output"
    printf 'stage_started_at=%s method=%s stage=%s\n' "$(date --iso-8601=seconds)" "$METHOD" "$stage" | tee -a "$output/run.log"
    "$PYTHON" "$RUNNER" \
        --mode generate \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --model-path "$MODEL" \
        --device cuda \
        2>&1 | tee -a "$output/run.log"
    "$PYTHON" "$RUNNER" \
        --mode score \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --workers 8 \
        --timeout 3 \
        2>&1 | tee -a "$output/run.log"
    "$PYTHON" "$RUNNER" \
        --mode audit \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        2>&1 | tee -a "$output/run.log"
    if [[ "$stage" == "smoke" ]]; then
        local before_hash after_hash before_rows after_rows
        before_hash=$(sha256sum "$output/predictions.jsonl" | awk '{print $1}')
        before_rows=$(wc -l < "$output/predictions.jsonl")
        "$PYTHON" "$RUNNER" \
            --mode generate \
            --method "$METHOD" \
            --stage "$stage" \
            --output-dir "$output" \
            --model-path "$MODEL" \
            --device cuda \
            2>&1 | tee -a "$output/run.log"
        after_hash=$(sha256sum "$output/predictions.jsonl" | awk '{print $1}')
        after_rows=$(wc -l < "$output/predictions.jsonl")
        if [[ "$before_hash" != "$after_hash" || "$before_rows" != "$after_rows" ]]; then
            echo "smoke resume/dedup check failed" >&2
            exit 1
        fi
        printf 'smoke_resume_check=passed rows=%s sha256=%s\n' "$after_rows" "$after_hash" | tee -a "$output/run.log"
    fi
    printf 'stage_finished_at=%s method=%s stage=%s\n' "$(date --iso-8601=seconds)" "$METHOD" "$stage" | tee -a "$output/run.log"
}

run_stage smoke
run_stage pilot
run_stage full
