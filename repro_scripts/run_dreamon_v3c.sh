#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-progressive-v3-budgeted
LAUNCHER="$ROOT/repro_scripts/run_dreamon_v3c_method.sh"
C0=v3_c0_budgeted_oneshot_pure_newline_veto
C=v3_c_budgeted_nonconsuming_blankline

cd "$ROOT"

case "${1:-targeted}" in
    verify)
        "$LAUNCHER" "$C0" verify
        "$LAUNCHER" "$C" verify
        ;;
    smoke)
        "$LAUNCHER" "$C0" v3c_smoke
        "$LAUNCHER" "$C" v3c_smoke
        ;;
    pure30)
        "$LAUNCHER" "$C0" pure30
        "$LAUNCHER" "$C" pure30
        ;;
    pilot)
        "$LAUNCHER" "$C0" pilot
        "$LAUNCHER" "$C" pilot
        ;;
    full)
        if "$LAUNCHER" "$C0" full; then
            printf 'full_completed method=%s\n' "$C0"
        else
            printf 'full_not_run_or_failed method=%s\n' "$C0" >&2
        fi
        if "$LAUNCHER" "$C" full; then
            printf 'full_completed method=%s\n' "$C"
        else
            printf 'full_not_run_or_failed method=%s\n' "$C" >&2
        fi
        ;;
    targeted)
        "$0" smoke
        "$0" pure30
        "$0" pilot
        ;;
    *) echo "usage: $0 {verify|smoke|pure30|pilot|full|targeted}" >&2; exit 2 ;;
esac
