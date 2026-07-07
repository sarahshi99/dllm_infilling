# H200 Material Drift Triage

triage_verdict: `material_drift_confirmed_controller_v2_blocked`
server_repro_verdict: `h200_material_outcome_drift`
controller_v2_allowed: `False`
branch: `codex/risk-controlled-dynamic-rescue`
commit: `c90457d76c49dc34288eb956a3f5b80dfa3c78ba`

## Privacy / Protocol Guard

- Frozen test remains sealed; this report does not write row-level test flips.
- Core full-run aggregate counts still use the existing 1033-row reproduction audit.
- Row-level triage CSVs are restricted to train/calibration/validation splits.

## Core Outcome Drift

| Run | Old Pass | H200 Pass | Delta | H200 Wins | H200 Losses | Agreement | Trigger Delta | Hash Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `control` | 787 | 787 | 0 | 4 | 4 | 0.9923 | 0 | 0.9622 |
| `midcons` | 795 | 794 | -1 | 4 | 5 | 0.9913 | 1 | 0.9613 |
| `route2` | 801 | 795 | -6 | 3 | 9 | 0.9884 | 2 | 0.9555 |
| `v6` | 802 | 796 | -6 | 3 | 9 | 0.9884 | 2 | 0.9555 |
| `cal` | 774 | 769 | -5 | 4 | 9 | 0.9874 | 0 | 0.9613 |

## Action Bank Drift

- outcome agreement: `0.9495145631067962`
- pass delta: `2`
- benefit labels non-KEEP old/H200: `191` / `193`
- harm labels non-KEEP old/H200: `1242` / `1237`
- candidate hash agreement: `0.801294498381877`

## Interpretation

- The H200 deltas are small relative to 1033 rows but materially affect Route2/V6/CAL baselines and controller labels.
- Controller V1 replay still selects zero intervention, but validation primary/V6 moved from 90 to 89 on the H200 bank.
- Controller V2 remains blocked until the research owner decides whether to accept H200 as the new source of truth, rerun additional migration checks, or investigate kernel/model drift further.
