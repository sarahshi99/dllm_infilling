#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-progressive-v3-budgeted
PYTHON=/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python
RUNNER="$ROOT/repro_scripts/run_dreamon_progressive_v2.py"
REPRO_AUDIT="$ROOT/repro_scripts/audit_dreamon_v3c_reproducibility.py"
MODEL=/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194
FULL_POPULATION="$ROOT/repro_results/dreamon_progressive_v2_protocol/generation_population.jsonl"
PURE30_POPULATION="$ROOT/manifests/v3_pure_newline_blank30.jsonl"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 {v3_c0_budgeted_oneshot_pure_newline_veto|v3_c_budgeted_nonconsuming_blankline} [verify|v3c_smoke|pure30|pilot|full]" >&2
    exit 2
fi

METHOD=$1
COMMAND=${2:-verify}
case "$METHOD" in
    v3_c0_budgeted_oneshot_pure_newline_veto)
        STEM=dreamon_v3_c0_pure_newline_veto
        PROTOCOL="$ROOT/repro_results/dreamon_v3_c0_pure_newline_veto_protocol/protocol.json"
        ;;
    v3_c_budgeted_nonconsuming_blankline)
        STEM=dreamon_v3_c_nonconsuming_blankline
        PROTOCOL="$ROOT/repro_results/dreamon_v3_c_nonconsuming_blankline_protocol/protocol.json"
        ;;
    *)
        echo "unsupported V3-C method: $METHOD" >&2
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

run_verification() {
    "$PYTHON" repro_scripts/prepare_dreamon_v3c_protocol.py
    "$PYTHON" -m pytest -q \
        tests/test_dreamon_v3c.py \
        tests/test_dreamon_v3_nonempty.py \
        tests/test_dreamon_v3_budgeted.py \
        tests/test_dreamon_v3_runner.py \
        tests/test_dreamon_hard_v2_boundary.py \
        tests/test_dreamon_slot_generator.py \
        tests/test_dreamon_progressive_v2_runner.py
    "$PYTHON" -m py_compile \
        repro_scripts/dreamon_slot_generator.py \
        repro_scripts/run_dreamon_progressive_v2.py \
        repro_scripts/prepare_dreamon_v3c_protocol.py \
        repro_scripts/audit_dreamon_v3c_reproducibility.py
    bash -n repro_scripts/run_dreamon_v3c_method.sh
    bash -n repro_scripts/run_dreamon_v3c.sh
    "$PYTHON" - "$METHOD" "$PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path

method, protocol_path = sys.argv[1:]
protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
assert protocol["protocol_version"] == 4
assert protocol["method"] == method
assert protocol["generation"]["initial_expand_budget"] == 64
assert protocol["generation"]["nonempty_guard"] is False
assert protocol["generation"]["pure_newline_guard_budget_per_slot"] == 1
assert protocol["generation"]["pure_newline_guard_global_budget"] == 3

generator = Path("repro_scripts/dreamon_slot_generator.py").read_text(encoding="utf-8")
generation = Path("repro_scripts/run_dreamon_progressive_v2.py").read_text(encoding="utf-8").split(
    "def run_generation", 1
)[1].split("def score_completion", 1)[0]
for forbidden in ("canonical_solution", "reference_middle", "reference_lines", "check_correctness"):
    assert forbidden not in generation
assert "BoundaryShift" not in generator
assert "ast.parse" not in generator
assert "method_uses_nonempty_guard(method)" in generator
assert "return method == Method.V3_HARD_BUDGETED_NONEMPTY_ORACLE" in generator
assert "newline_token_ids" not in generator.split("def constrained_active_logits", 1)[1].split("def select_update", 1)[0]
PY
}

output_for_stage() {
    case "$1" in
        v3c_smoke) printf '%s\n' "$ROOT/repro_results/${STEM}_smoke5" ;;
        pure30) printf '%s\n' "$ROOT/repro_results/${STEM}_pure30" ;;
        pilot) printf '%s\n' "$ROOT/repro_results/${STEM}_pilot30" ;;
        full) printf '%s\n' "$ROOT/repro_results/${STEM}_all642" ;;
        *) echo "unknown stage: $1" >&2; return 2 ;;
    esac
}

population_for_stage() {
    case "$1" in
        v3c_smoke|pure30) printf '%s\n' "$PURE30_POPULATION" ;;
        pilot|full) printf '%s\n' "$FULL_POPULATION" ;;
        *) echo "unknown stage: $1" >&2; return 2 ;;
    esac
}

run_generate() {
    local stage=$1 output=$2 population=$3
    "$PYTHON" "$RUNNER" \
        --mode generate \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --model-path "$MODEL" \
        --protocol "$PROTOCOL" \
        --population "$population" \
        --device cuda
}

assert_full_authorized() {
    "$PYTHON" - "$STEM" <<'PY'
import json
import sys
from pathlib import Path

stem = sys.argv[1]
root = Path("repro_results")
smoke = json.loads((root / f"{stem}_smoke5/gate.json").read_text(encoding="utf-8"))
pure = json.loads((root / f"{stem}_pure30/gate.json").read_text(encoding="utf-8"))
pilot = json.loads((root / f"{stem}_pilot30/gate.json").read_text(encoding="utf-8"))
passed = (
    smoke.get("engineering_gate_passed") is True
    and pure.get("engineering_gate_passed") is True
    and pure.get("authorizes_full") is True
    and pilot.get("engineering_gate_passed") is True
)
raise SystemExit(0 if passed else 1)
PY
}

run_smoke_reproducibility_audits() {
    local stage=$1 output=$2 population=$3
    local deterministic="$output/deterministic_rerun"
    local resumed="$output/checkpoint_resume_rerun"
    run_generate "$stage" "$deterministic" "$population"
    "$PYTHON" "$REPRO_AUDIT" compare \
        --reference-dir "$output" \
        --candidate-dir "$deterministic" \
        --output "$output/deterministic_rerun_audit.json" \
        --label deterministic_fresh_rerun
    "$PYTHON" "$REPRO_AUDIT" seed-resume \
        --source-dir "$output" \
        --resume-dir "$resumed" \
        --rows 2
    run_generate "$stage" "$resumed" "$population"
    "$PYTHON" "$REPRO_AUDIT" compare \
        --reference-dir "$output" \
        --candidate-dir "$resumed" \
        --output "$output/checkpoint_resume_audit.json" \
        --label durable_row_checkpoint_resume
}

run_stage() {
    local stage=$1
    if [[ "$stage" == "full" ]]; then
        assert_full_authorized
    fi
    local output population
    output=$(output_for_stage "$stage")
    population=$(population_for_stage "$stage")
    mkdir -p "$output"
    printf 'stage_started_at=%s method=%s stage=%s head=%s\n' \
        "$(date --iso-8601=seconds)" "$METHOD" "$stage" "$(git rev-parse HEAD)" \
        | tee -a "$output/run.log"
    run_generate "$stage" "$output" "$population" 2>&1 | tee -a "$output/run.log"
    "$PYTHON" "$RUNNER" \
        --mode score \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --model-path "$MODEL" \
        --protocol "$PROTOCOL" \
        --population "$population" \
        --workers 8 \
        --timeout 3 2>&1 | tee -a "$output/run.log"
    "$PYTHON" "$RUNNER" \
        --mode audit \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --model-path "$MODEL" \
        --protocol "$PROTOCOL" \
        --population "$population" 2>&1 | tee -a "$output/run.log"

    local before_hash before_rows after_hash after_rows
    before_hash=$(sha256sum "$output/predictions.jsonl" | awk '{print $1}')
    before_rows=$(wc -l < "$output/predictions.jsonl")
    run_generate "$stage" "$output" "$population" 2>&1 | tee -a "$output/run.log"
    after_hash=$(sha256sum "$output/predictions.jsonl" | awk '{print $1}')
    after_rows=$(wc -l < "$output/predictions.jsonl")
    if [[ "$before_hash" != "$after_hash" || "$before_rows" != "$after_rows" ]]; then
        echo "durable row-level resume/dedup equality failed" >&2
        return 1
    fi
    printf 'durable_resume_equality=passed rows=%s sha256=%s\n' \
        "$after_rows" "$after_hash" | tee -a "$output/run.log"

    if [[ "$stage" == "v3c_smoke" ]]; then
        run_smoke_reproducibility_audits "$stage" "$output" "$population" \
            2>&1 | tee -a "$output/run.log"
    fi

    "$PYTHON" "$RUNNER" \
        --mode gate \
        --method "$METHOD" \
        --stage "$stage" \
        --output-dir "$output" \
        --model-path "$MODEL" \
        --protocol "$PROTOCOL" \
        --population "$population" 2>&1 | tee -a "$output/run.log"
    printf 'stage_finished_at=%s method=%s stage=%s\n' \
        "$(date --iso-8601=seconds)" "$METHOD" "$stage" | tee -a "$output/run.log"
}

case "$COMMAND" in
    verify) run_verification ;;
    v3c_smoke|pure30|pilot|full) run_verification; run_stage "$COMMAND" ;;
    *) echo "unknown command: $COMMAND" >&2; exit 2 ;;
esac
