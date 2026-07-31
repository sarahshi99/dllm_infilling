#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/shx/miniconda3/envs/dllm_env/bin/python}
OFFICIAL_CAL_ROOT=${OFFICIAL_CAL_ROOT:-/home/shx/.cache/dllm_infilling/calibrated_adaptive_length-741e8418}
HUMANEVAL_ROOT=${HUMANEVAL_ROOT:-/home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff}
CURRENT_DATASET=${CURRENT_DATASET:-$HUMANEVAL_ROOT/data/HumanEval-SingleLineInfilling.jsonl.gz}
HF_HOME=${HF_HOME:-/home/shx/.cache/huggingface}
MANIFEST_ROOT=$ROOT/analysis_outputs/baseline_manifests_20260731_v1
SMOKE_MANIFEST=$MANIFEST_ROOT/cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl
FULL_MANIFEST=$MANIFEST_ROOT/cal_singleline_rest_nonfrozen_manifest.jsonl
PROBE_OUTPUT=$ROOT/outputs_clean/lrdllm_stage1_singleline_probe1_20260731_v1
PROBE_ANALYSIS=$ROOT/analysis_outputs/lrdllm_stage1_singleline_probe1_20260731_v1
SMOKE_OUTPUT=$ROOT/outputs_clean/lrdllm_stage1_singleline_smoke_20260731_v1
SMOKE_ANALYSIS=$ROOT/analysis_outputs/lrdllm_stage1_singleline_smoke_20260731_v1
FULL_OUTPUT=$ROOT/outputs_clean/lrdllm_stage1_singleline_full_20260731_v1
FULL_ANALYSIS=$ROOT/analysis_outputs/lrdllm_stage1_singleline_full_20260731_v1
OFFICIAL_FIXED32_OUTPUT=$ROOT/outputs_clean/official_cal_singleline_20260731_v1
LOG_ROOT=$ROOT/logs/paper_agent
ADAPTER=$ROOT/experiments/p1_lrdllm_stage1_adapter.py
ARM=local_lrdllm_stage1_fixed_decode

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME

common_args() {
  printf '%s\0' \
    --official-cal-root "$OFFICIAL_CAL_ROOT" \
    --humaneval-root "$HUMANEVAL_ROOT" \
    --current-dataset "$CURRENT_DATASET" \
    --full-manifest-jsonl "$FULL_MANIFEST" \
    --hf-home "$HF_HOME" \
    --arm "$ARM" \
    --max-probe-length 128 \
    --seed 42
}

cpu_preflight() {
  local mode=${1:-smoke}
  local -a extra=()
  local manifest=$SMOKE_MANIFEST
  local output=$SMOKE_OUTPUT
  if [[ "$mode" == "probe_only" ]]; then
    extra+=(--probe-only)
    output=$PROBE_OUTPUT
  elif [[ "$mode" == "full" ]]; then
    extra+=(--auto-full)
    manifest=$FULL_MANIFEST
    output=$FULL_OUTPUT
  fi
  mapfile -d '' -t common < <(common_args)
  "$PYTHON" "$ADAPTER" "${common[@]}" \
    --manifest-jsonl "$manifest" \
    --output-dir "$output" \
    --smoke-cases 12 \
    --preflight-only \
    "${extra[@]}"
}

gpu_gate() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "gpu_gate=failed reason=nvidia_smi_missing"
    return 3
  fi
  local processes
  processes=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    printf '%s\n' "gpu_gate=blocked_by_existing_gpu_process"
    printf '%s\n' "$processes"
    return 4
  fi
  local uncorrectable corrected
  uncorrectable=$(nvidia-smi -i 0 --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
  corrected=$(nvidia-smi -i 0 --query-gpu=ecc.errors.corrected.volatile.total --format=csv,noheader,nounits)
  if [[ "$uncorrectable" != "0" || "$corrected" != "0" ]]; then
    printf '%s\n' "gpu_gate=failed reason=volatile_ecc corrected=$corrected uncorrectable=$uncorrectable"
    return 5
  fi
  printf '%s\n' "gpu_gate=passed physical_gpu=0"
}

workspace_gate() {
  if [[ ! -f "$ROOT/docs/paper_agent/experiments/20260731_lrdllm_stage1_protocol_freeze.zh.md" ]]; then
    printf '%s\n' "workspace_gate=failed reason=protocol_freeze_missing"
    return 7
  fi
  if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
    printf '%s\n' "workspace_gate=failed reason=tracked_worktree_not_clean"
    return 8
  fi
  if [[ -n "$(git -C "$ROOT" diff --name-only -- outputs_clean logs analysis_outputs)" ]]; then
    printf '%s\n' "workspace_gate=failed reason=tracked_raw_log_or_analysis_diff"
    return 9
  fi
  printf '%s\n' "workspace_gate=passed commit=$(git -C "$ROOT" rev-parse HEAD)"
}

output_gate() {
  local output=$1
  local raw_name=$2
  local failure_name=$3
  if [[ -e "$output/$failure_name" ]]; then
    printf '%s\n' "output_gate=failed reason=existing_failure_journal path=$output/$failure_name use_new_output_version=true"
    return 10
  fi
  if [[ -d "$output" && -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" && ! -f "$output/$raw_name" ]]; then
    printf '%s\n' "output_gate=failed reason=output_directory_conflict path=$output"
    return 11
  fi
  printf '%s\n' "output_gate=passed path=$output"
}

probe_only_gate() {
  "$PYTHON" -c 'import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); arm=sys.argv[2]; audit=json.loads((root/f"{arm}_probe_only_final_audit.json").read_text()); assert audit["passed"] is True and audit["expected_count"]==audit["observed_count"]==1 and audit["probe_forward_error_count"]==0 and audit["finite_diagnostic_error_count"]==0 and audit["evaluator_completed_count"]==0 and audit["formal_decode_forwards_total"]==0; print(json.dumps({"probe_only_gate": True, "rows": 1, "forwards": 8}, sort_keys=True))' "$PROBE_OUTPUT" "$ARM"
}

run_probe_only() {
  cpu_preflight probe_only
  workspace_gate
  output_gate "$PROBE_OUTPUT" "${ARM}_probe_only_raw.jsonl" "${ARM}_probe_only_failure_journal.jsonl"
  gpu_gate
  mkdir -p "$LOG_ROOT"
  mapfile -d '' -t common < <(common_args)
  "$PYTHON" "$ADAPTER" "${common[@]}" \
    --manifest-jsonl "$SMOKE_MANIFEST" \
    --output-dir "$PROBE_OUTPUT" \
    --analysis-output-dir "$PROBE_ANALYSIS" \
    --smoke-cases 12 \
    --progress-every 1 \
    --probe-only 2>&1 | tee -a "$LOG_ROOT/20260731_lrdllm_stage1_probe1.log"
  probe_only_gate
}

smoke_once() {
  mapfile -d '' -t common < <(common_args)
  "$PYTHON" "$ADAPTER" "${common[@]}" \
    --manifest-jsonl "$SMOKE_MANIFEST" \
    --output-dir "$SMOKE_OUTPUT" \
    --analysis-output-dir "$SMOKE_ANALYSIS" \
    --probe-output-dir "$PROBE_OUTPUT" \
    --smoke-cases 12 \
    --progress-every 1
}

run_smoke() {
  cpu_preflight smoke
  workspace_gate
  probe_only_gate
  output_gate "$SMOKE_OUTPUT" "${ARM}_raw.jsonl" "${ARM}_failure_journal.jsonl"
  gpu_gate
  mkdir -p "$LOG_ROOT"
  {
    printf '%s\n' "stage1_smoke_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    smoke_once
    printf '%s\n' "stage1_resume_noop_check_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    smoke_once
    "$PYTHON" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); assert x["status"]=="completed" and x["new_rows_written"]==0 and x["resume_noop"] is True; print(json.dumps({"resume_noop": True, "new_rows_written": 0}, sort_keys=True))' "$SMOKE_OUTPUT/${ARM}_smoke_progress.json"
    printf '%s\n' "stage1_smoke_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee -a "$LOG_ROOT/20260731_lrdllm_stage1_smoke.log"
}

run_full() {
  if [[ "${ALLOW_LRDLLM_STAGE1_FULL:-0}" != "1" ]]; then
    printf '%s\n' "full_gate=disabled explicit_operator_intent_missing"
    return 6
  fi
  cpu_preflight full
  workspace_gate
  probe_only_gate
  output_gate "$FULL_OUTPUT" "${ARM}_raw.jsonl" "${ARM}_failure_journal.jsonl"
  gpu_gate
  mapfile -d '' -t common < <(common_args)
  "$PYTHON" "$ADAPTER" "${common[@]}" \
    --manifest-jsonl "$FULL_MANIFEST" \
    --output-dir "$FULL_OUTPUT" \
    --analysis-output-dir "$FULL_ANALYSIS" \
    --probe-output-dir "$PROBE_OUTPUT" \
    --smoke-output-dir "$SMOKE_OUTPUT" \
    --official-fixed32-output-dir "$OFFICIAL_FIXED32_OUTPUT" \
    --auto-full
}

case "${1:-help}" in
  preflight)
    cpu_preflight "${2:-smoke}"
    ;;
  probe-only)
    run_probe_only
    ;;
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  help|--help|-h)
    printf '%s\n' "usage: $0 {preflight [probe_only|smoke|full]|probe-only|smoke|full}"
    ;;
  *)
    printf '%s\n' "unknown mode: $1" >&2
    exit 2
    ;;
esac
