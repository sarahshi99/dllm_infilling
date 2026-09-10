#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python"
MODEL="/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194"
RUN_ROOT="/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2"
DATA_ROOT="$RUN_ROOT/data"
BANK_DB="$RUN_ROOT/transition_bank.sqlite"
COMMON_INIT="$RUN_ROOT/common_initialization.pt"
COMMON_TRAINING="$RUN_ROOT/common_training_config.json"
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"
OUTPUT_ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-markov-k2-eval-v1"
RESULT_DIR="$OUTPUT_ROOT/analysis_outputs/dreamon_markov_head_training_20260910_v2"
LOG="$OUTPUT_ROOT/logs/paper_agent/20260910_dreamon_markov_v2_training.log"

mkdir -p "$CHECKPOINT_ROOT" "$RESULT_DIR" "$(dirname "$LOG")"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

printf '[training-watcher-launch] %s\n' "$(date -u +%FT%TZ)"
printf '[execution-head] %s\n' "$(git rev-parse HEAD)"

while [[ ! -f "$RESULT_DIR/transition_bank_summary.json" ]]; do
  if ! pgrep -f 'build_dreamon_markov_transition_bank.py.*dreamon_markov_head_training_20260910_v2' >/dev/null; then
    printf '[training-watcher-error] bank summary absent and bank process not running %s\n' "$(date -u +%FT%TZ)"
    exit 3
  fi
  printf '[training-watcher-wait] bank still running %s\n' "$(date -u +%FT%TZ)"
  sleep 60
done

"$PYTHON" - "$RESULT_DIR/transition_bank_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
if summary.get("status") != "completed" or summary.get("total_selected") != 240000:
    raise SystemExit(f"bank did not complete exactly: {summary.get('status')} {summary.get('total_selected')}")
PY

"$PYTHON" analysis/audit_dreamon_markov_v2_bank.py \
  --bank-db "$BANK_DB" \
  --prepared-records "$DATA_ROOT/prepared_records.jsonl.gz" \
  --exclusion-audit "$RESULT_DIR/humaneval_exclusion_audit.json" \
  --output "$RESULT_DIR/actual_bank_membership_audit_v2.json" \
  --train-target 200000 \
  --validation-target 20000 \
  --external-test-target 20000 \
  --per-problem-cap 16 \
  --oversample 1.25

if [[ ! -f "$COMMON_INIT" ]]; then
  "$PYTHON" experiments/train_dreamon_markov_head.py init-common \
    --model-snapshot "$MODEL" \
    --result-dir "$RESULT_DIR" \
    --common-init "$COMMON_INIT"
fi

if [[ ! -f "$COMMON_TRAINING" ]]; then
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" experiments/train_dreamon_markov_head.py determine-microbatch \
    --model-snapshot "$MODEL" \
    --result-dir "$RESULT_DIR" \
    --bank-db "$BANK_DB" \
    --common-training-config "$COMMON_TRAINING" \
    --device cuda
fi
cp "$COMMON_TRAINING" "$RESULT_DIR/common_training_config.json"

tv_exit=0
if [[ ! -f "$RESULT_DIR/tv_training_status.json" ]]; then
  set +e
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" experiments/train_dreamon_markov_head.py run-head \
    --kind tv \
    --model-snapshot "$MODEL" \
    --result-dir "$RESULT_DIR" \
    --bank-db "$BANK_DB" \
    --common-init "$COMMON_INIT" \
    --common-training-config "$COMMON_TRAINING" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --device cuda
  tv_exit=$?
  set -e
fi
printf '[tv-exit] %s %s\n' "$tv_exit" "$(date -u +%FT%TZ)"
nvidia-smi > "$RESULT_DIR/nvidia_smi_after_tv.txt"

kl_exit=0
if [[ ! -f "$RESULT_DIR/kl_training_status.json" ]]; then
  set +e
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" experiments/train_dreamon_markov_head.py run-head \
    --kind kl \
    --model-snapshot "$MODEL" \
    --result-dir "$RESULT_DIR" \
    --bank-db "$BANK_DB" \
    --common-init "$COMMON_INIT" \
    --common-training-config "$COMMON_TRAINING" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --device cuda
  kl_exit=$?
  set -e
fi
printf '[kl-exit] %s %s\n' "$kl_exit" "$(date -u +%FT%TZ)"
nvidia-smi > "$RESULT_DIR/nvidia_smi_after_kl.txt"

if [[ ! -f "$RESULT_DIR/tv_training_status.json" || ! -f "$RESULT_DIR/kl_training_status.json" ]]; then
  printf '[training-watcher-error] missing head status tv_exit=%s kl_exit=%s\n' "$tv_exit" "$kl_exit"
  exit 4
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
"$PYTHON" analysis/finalize_dreamon_markov_head_training.py \
  --result-dir "$RESULT_DIR" \
  --bank-db "$BANK_DB" \
  --model-snapshot "$MODEL" \
  --common-training-config "$COMMON_TRAINING" \
  --device cuda \
  --run-id dreamon_markov_head_training_20260910_v2 \
  --run-date-utc 2026-09-10 \
  --branch codex/dreamon-markov-k2-eval-v1 \
  --base-head "$(git rev-parse HEAD)" \
  --report-title 'DreamOn 外部一阶 Markov 头重新训练 v2' \
  --data-isolation-status v2_prepared_and_actual_bank_audits_passed \
  --data-isolation-note '训练数据来自固定 OpenCoder educational_instruct；完整源数据先按保守规则排除 HumanEval 候选关联组，prepared 与实际 transition bank 的成员审计均通过。该结论不证明基础模型预训练未见过 HumanEval。' \
  --skip-research-record-updates \
  --resume-existing-external-diagnostics

printf '[training-watcher-exit] 0 %s\n' "$(date -u +%FT%TZ)"
