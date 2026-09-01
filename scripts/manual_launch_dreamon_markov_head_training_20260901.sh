#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-markov-head-training-v1"
PYTHON="/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python"
MODEL="/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194"
SOURCE="/home/shx/projects/dllm_infilling/.worktrees/dreamon-source-8a0a549"
EVALUATOR="/home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff"
RUN_ROOT="/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260901_v1"
DATA_ROOT="$RUN_ROOT/data"
BANK_DB="$RUN_ROOT/transition_bank.sqlite"
COMMON_INIT="$RUN_ROOT/common_initialization.pt"
COMMON_TRAINING="$RUN_ROOT/common_training_config.json"
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"
RESULT_DIR="$ROOT/analysis_outputs/dreamon_markov_head_training_20260901_v1"
LOG="$ROOT/logs/paper_agent/20260901_dreamon_markov_head_training_v1.log"
COMMANDS="$RESULT_DIR/commands.log"
BRANCH="codex/dreamon-markov-head-training-v1"

mkdir -p "$RUN_ROOT" "$CHECKPOINT_ROOT" "$RESULT_DIR" "$(dirname "$LOG")"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

record() {
  printf '%s\n' "$*" >> "$COMMANDS"
  printf '[command] %s\n' "$*"
}

wait_for_gpu() {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; do
    printf '[gpu-wait] %s unrelated compute process present; waiting 60s\n' "$(date -u +%FT%TZ)"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    sleep 60
  done
}

commit_if_needed() {
  local message="$1"
  shift
  local paths=()
  local path
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      paths+=("$path")
    fi
  done
  if [[ ${#paths[@]} -eq 0 ]]; then
    return
  fi
  git add -- "${paths[@]}"
  if ! git diff --cached --quiet; then
    git diff --cached --name-only
    git commit -m "$message"
  fi
}

printf 'DreamOn Markov head training v1\nstart=%s\nbranch=%s\nbase=77f0572b1ca4fe031ab6bbf29b3a4d8740f38802\n' "$(date -u +%FT%TZ)" "$BRANCH" > "$COMMANDS"

record "$PYTHON -m unittest tests.test_dreamon_markov_head_training tests.test_dreamon_markov_premise tests.test_dreamon_order_parallelism tests.test_dreamon_singleline_adapter"
$PYTHON -m unittest tests.test_dreamon_markov_head_training tests.test_dreamon_markov_premise tests.test_dreamon_order_parallelism tests.test_dreamon_singleline_adapter

record "$PYTHON analysis/prepare_dreamon_markov_training_data.py --model-snapshot $MODEL --evaluator-root $EVALUATOR --result-dir $RESULT_DIR --cache-dir $DATA_ROOT --split-seed 20260901"
$PYTHON analysis/prepare_dreamon_markov_training_data.py --model-snapshot "$MODEL" --evaluator-root "$EVALUATOR" --result-dir "$RESULT_DIR" --cache-dir "$DATA_ROOT" --split-seed 20260901

wait_for_gpu
record "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON experiments/build_dreamon_markov_transition_bank.py --prepared-records $DATA_ROOT/prepared_records.jsonl.gz --source-root $SOURCE --model-snapshot $MODEL --bank-db $BANK_DB --result-dir $RESULT_DIR --device cuda --per-problem-cap 16 --oversample 1.25"
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON experiments/build_dreamon_markov_transition_bank.py --prepared-records "$DATA_ROOT/prepared_records.jsonl.gz" --source-root "$SOURCE" --model-snapshot "$MODEL" --bank-db "$BANK_DB" --result-dir "$RESULT_DIR" --device cuda --per-problem-cap 16 --oversample 1.25

record "$PYTHON experiments/train_dreamon_markov_head.py init-common --model-snapshot $MODEL --result-dir $RESULT_DIR --common-init $COMMON_INIT"
$PYTHON experiments/train_dreamon_markov_head.py init-common --model-snapshot "$MODEL" --result-dir "$RESULT_DIR" --common-init "$COMMON_INIT"

wait_for_gpu
record "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON experiments/train_dreamon_markov_head.py determine-microbatch --model-snapshot $MODEL --result-dir $RESULT_DIR --bank-db $BANK_DB --common-training-config $COMMON_TRAINING --device cuda"
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON experiments/train_dreamon_markov_head.py determine-microbatch --model-snapshot "$MODEL" --result-dir "$RESULT_DIR" --bank-db "$BANK_DB" --common-training-config "$COMMON_TRAINING" --device cuda
cp "$COMMON_TRAINING" "$RESULT_DIR/common_training_config.json"

commit_if_needed "data: freeze OpenCoder split and shared Markov bank" \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/data_manifest_summary.json \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/split_and_dedup_audit.json \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/split_manifest.jsonl.zst \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/external_test_manifest.jsonl.zst \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/data_audit_samples.json \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/transition_bank_summary.json \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/transition_bank_audit_samples.json \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/common_initialization.json \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/common_training_config.json

wait_for_gpu
record "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON experiments/train_dreamon_markov_head.py run-head --kind tv --model-snapshot $MODEL --result-dir $RESULT_DIR --bank-db $BANK_DB --common-init $COMMON_INIT --common-training-config $COMMON_TRAINING --checkpoint-root $CHECKPOINT_ROOT --device cuda"
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON experiments/train_dreamon_markov_head.py run-head --kind tv --model-snapshot "$MODEL" --result-dir "$RESULT_DIR" --bank-db "$BANK_DB" --common-init "$COMMON_INIT" --common-training-config "$COMMON_TRAINING" --checkpoint-root "$CHECKPOINT_ROOT" --device cuda
nvidia-smi > "$RESULT_DIR/nvidia_smi_after_tv.txt"
commit_if_needed "results: record sequential TV-head training" \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/tv_training_status.json \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/training_curves_tv.csv \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/pilot_validation_diagnostics_tv.jsonl.gz \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/validation_diagnostics_tv.jsonl.gz \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/nvidia_smi_after_tv.txt

wait_for_gpu
record "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON experiments/train_dreamon_markov_head.py run-head --kind kl --model-snapshot $MODEL --result-dir $RESULT_DIR --bank-db $BANK_DB --common-init $COMMON_INIT --common-training-config $COMMON_TRAINING --checkpoint-root $CHECKPOINT_ROOT --device cuda"
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON experiments/train_dreamon_markov_head.py run-head --kind kl --model-snapshot "$MODEL" --result-dir "$RESULT_DIR" --bank-db "$BANK_DB" --common-init "$COMMON_INIT" --common-training-config "$COMMON_TRAINING" --checkpoint-root "$CHECKPOINT_ROOT" --device cuda
nvidia-smi > "$RESULT_DIR/nvidia_smi_after_kl.txt"
commit_if_needed "results: record independent KL-head training" \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/kl_training_status.json \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/training_curves_kl.csv \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/pilot_validation_diagnostics_kl.jsonl.gz \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/validation_diagnostics_kl.jsonl.gz \
  analysis_outputs/dreamon_markov_head_training_20260901_v1/nvidia_smi_after_kl.txt

wait_for_gpu
record "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON analysis/finalize_dreamon_markov_head_training.py --result-dir $RESULT_DIR --bank-db $BANK_DB --model-snapshot $MODEL --common-training-config $COMMON_TRAINING --device cuda"
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false $PYTHON analysis/finalize_dreamon_markov_head_training.py --result-dir "$RESULT_DIR" --bank-db "$BANK_DB" --model-snapshot "$MODEL" --common-training-config "$COMMON_TRAINING" --device cuda

record "$PYTHON -m unittest tests.test_dreamon_markov_head_training tests.test_dreamon_markov_premise tests.test_dreamon_order_parallelism tests.test_dreamon_singleline_adapter"
$PYTHON -m unittest tests.test_dreamon_markov_head_training tests.test_dreamon_markov_premise tests.test_dreamon_order_parallelism tests.test_dreamon_singleline_adapter
$PYTHON -m py_compile experiments/dreamon_markov_head_training.py experiments/build_dreamon_markov_transition_bank.py experiments/train_dreamon_markov_head.py analysis/prepare_dreamon_markov_training_data.py analysis/finalize_dreamon_markov_head_training.py
git diff --check

commit_if_needed "results: compare Markov objectives and finalize research audit" \
  analysis_outputs/dreamon_markov_head_training_20260901_v1 \
  docs/paper_agent/current_action.md \
  docs/paper_agent/experiment_queue.md \
  docs/paper_agent/decision_log.md \
  docs/paper_agent/idea_board.md \
  docs/paper_agent/codex_handoff.latest.zh.md \
  docs/paper_agent/review_manifest.latest.json \
  docs/paper_agent/evidence_snapshot.json \
  docs/paper_agent/experiments/20260901_dreamon_markov_head_training_result.zh.md \
  docs/results/run_registry.md

git push -u origin HEAD:"$BRANCH"
printf 'completed=%s\nfinal_head=%s\n' "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" | tee "$RUN_ROOT/completed.txt"
