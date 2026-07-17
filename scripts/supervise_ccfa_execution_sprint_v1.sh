#!/usr/bin/env bash
# Non-destructive, auditable GPU sequencer for the authorized execution sprint.
set -euo pipefail

ROOT="/home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1"
CAL_OUT="$ROOT/outputs_clean/official_cal_primary_20260715_sprint_v1"
CAL_RAW="$CAL_OUT/official_cal_primary_raw.jsonl"
CAL_FULL_MANIFEST="$CAL_OUT/official_cal_primary_full_run_manifest.json"
M4_SESSION="m4-semantic-particle-assembly-sprint-v1"
M4_LAUNCHER="$ROOT/scripts/manual_launch_m4_semantic_particle_assembly_sprint_v1.sh"
M4_DIRS=(
  "$ROOT/outputs_clean/m4_best_randomspanlight_20260715_sprint_v1"
  "$ROOT/outputs_clean/m4_assembly_randomspanlight_20260715_sprint_v1"
  "$ROOT/outputs_clean/m4_repair_randomspanlight_20260715_sprint_v1"
)
STATE_JSONL="$ROOT/logs/paper_agent/20260717_ccfa_execution_sprint_supervisor.jsonl"
STATE_MD="$ROOT/logs/paper_agent/20260717_ccfa_execution_sprint_supervisor.md"
mkdir -p "$(dirname "$STATE_JSONL")"

last_state=""
last_rows=0

record() {
  local event="$1"
  local rows="$2"
  local free_mib="$3"
  local util="$4"
  local compute_count="$5"
  local utc
  utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if [[ "$event" == "$last_state" ]]; then
    return
  fi
  last_state="$event"
  printf '{"utc":"%s","event":"%s","cal_raw_rows":%s,"gpu_free_mib":%s,"gpu_utilization_percent":%s,"compute_process_count":%s}\n' "$utc" "$event" "$rows" "$free_mib" "$util" "$compute_count" >>"$STATE_JSONL"
  printf -- '- `%s` `%s`: CAL raw=%s, free MiB=%s, util=%s%%, compute processes=%s\n' "$utc" "$event" "$rows" "$free_mib" "$util" "$compute_count" >>"$STATE_MD"
}

cal_rows() {
  if [[ -f "$CAL_RAW" ]]; then
    wc -l <"$CAL_RAW" | tr -d ' '
  else
    printf '0'
  fi
}

gpu_snapshot() {
  local line
  line="$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$line" ]]; then
    return 1
  fi
  printf '%s\n' "$line" | awk -F',' 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
}

compute_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {print $1}' | sort -u | wc -l | tr -d ' '
}

ecc_zero() {
  local table gpu_row
  table="$(nvidia-smi 2>/dev/null || true)"
  gpu_row="$(printf '%s\n' "$table" | awk '/^\|   0  / {print; exit}')"
  [[ -n "$gpu_row" && "$gpu_row" == *"|                    0 |"* ]]
}

m4_outputs_absent() {
  local directory
  for directory in "${M4_DIRS[@]}"; do
    if [[ -e "$directory" ]]; then
      return 1
    fi
  done
  return 0
}

record "supervisor_started" "$(cal_rows)" -1 -1 -1
while true; do
  rows="$(cal_rows)"
  if ! snapshot="$(gpu_snapshot)"; then
    record "gpu_query_unavailable_waiting" "$rows" -1 -1 -1
    sleep 60
    continue
  fi
  read -r free_mib utilization <<<"$snapshot"
  compute_count="$(compute_process_count)"

  if [[ ! -f "$CAL_FULL_MANIFEST" ]]; then
    record "cal_full_not_started_waiting" "$rows" "$free_mib" "$utilization" "$compute_count"
    sleep 60
    continue
  fi
  full_age=$(( $(date +%s) - $(stat -c %Y "$CAL_FULL_MANIFEST") ))
  if (( full_age < 600 || rows <= last_rows )); then
    record "cal_t_plus_10_or_growth_not_yet_satisfied" "$rows" "$free_mib" "$utilization" "$compute_count"
    last_rows="$rows"
    sleep 60
    continue
  fi
  last_rows="$rows"

  if tmux has-session -t "$M4_SESSION" 2>/dev/null; then
    record "m4_already_running" "$rows" "$free_mib" "$utilization" "$compute_count"
    sleep 60
    continue
  fi
  if ! m4_outputs_absent; then
    record "m4_existing_output_requires_manual_audit" "$rows" "$free_mib" "$utilization" "$compute_count"
    sleep 60
    continue
  fi
  if (( free_mib < 25600 )); then
    record "m4_waiting_for_25gib_memory_reserve" "$rows" "$free_mib" "$utilization" "$compute_count"
    sleep 60
    continue
  fi
  if (( compute_count > 1 )); then
    record "m4_waiting_no_third_research_process" "$rows" "$free_mib" "$utilization" "$compute_count"
    sleep 60
    continue
  fi
  if ! ecc_zero; then
    record "m4_waiting_for_ecc_zero" "$rows" "$free_mib" "$utilization" "$compute_count"
    sleep 60
    continue
  fi

  tmux new-session -d -s "$M4_SESSION" -c "$ROOT" "bash $M4_LAUNCHER"
  record "m4_started_after_cal_t_plus_10_safe_slot" "$rows" "$free_mib" "$utilization" "$compute_count"
  sleep 60
done
