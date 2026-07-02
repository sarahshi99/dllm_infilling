# True-Long Action-Ceiling Matrix Dry Run

Decision: `ready_for_small_pilot_after_researcher_approval`

## Pre-Registration

- hypothesis: true-long failures are jointly limited by canvas adequacy, rescue generation quality, selector quality, and trigger recall.
- intervention: compare current primary, current Route2 rescue, oracle-sufficient canvas, and oracle-sufficient canvas with a fixed 96-step conservative schedule.
- control: current `midcons` primary output and current Route2 precision len32 output.
- expected outcomes: C/D reveal whether correct candidates exist when canvas is sufficient; A/B show current deployable gap.
- decision rule: if C/D do not create correct candidates in triggered and missed long pools, stop blind true-long length-control and pivot to backbone/generation limitation; if C/D create candidates but B misses them, focus selector/action selection; if missed pool has C/D wins, focus trigger recall.
- stop condition: no full run until a pilot report shows positive ceiling or clear negative anatomy on the preselected small case set.
- leakage guard: C/D use oracle length for offline ceiling only and must not be described as deployable.

## Dry-Run Summary

- case_count: `9`
- action_count: `36`
- planned_decode_step_units: `2592`
- case_pool_counts: `{'missed_failed_long': 3, 'positive_control_rescued': 3, 'triggered_failed_long': 3}`
- oracle_bucket_counts: `{'17-24': 2, '25+': 6, '9-12': 1}`
- route2_rescue_len_ge_oracle_rate_in_manifest: `0.6666666666666666`

## Planned Actions

| Action | Deployability | Canvas | Steps |
|---|---|---|---:|
| `A_primary` | `deployable_current_primary` | `use_midcons_selected_length` | `64` |
| `B_route2_len32` | `deployable_current_route2` | `use_existing_route2_precision_len32_output` | `64` |
| `C_oracle_sufficient` | `offline_ceiling_only_oracle_canvas` | `max(primary_len, route2_len, oracle_len)` | `64` |
| `D_oracle_sufficient_conservative` | `offline_ceiling_only_oracle_canvas` | `max(primary_len, route2_len, oracle_len)` | `96` |

## Cases

| Pool | Task | Bucket | Oracle | Primary Len | Route2 Triggered | Route2 Len | Ceiling Len |
|---|---|---|---:|---:|---|---:|---:|
| positive_control_rescued | `SingleLineInfilling/HumanEval/116/L0` | `17-24` | 21 | 3 | `True` | 32 | 32 |
| positive_control_rescued | `SingleLineInfilling/HumanEval/66/L1` | `17-24` | 21 | 3 | `True` | 32 | 32 |
| positive_control_rescued | `SingleLineInfilling/HumanEval/60/L0` | `9-12` | 10 | 3 | `True` | 32 | 32 |
| triggered_failed_long | `SingleLineInfilling/HumanEval/51/L0` | `25+` | 34 | 3 | `True` | 32 | 34 |
| triggered_failed_long | `SingleLineInfilling/HumanEval/88/L0` | `25+` | 34 | 3 | `True` | 32 | 34 |
| triggered_failed_long | `SingleLineInfilling/HumanEval/85/L0` | `25+` | 32 | 3 | `True` | 32 | 32 |
| missed_failed_long | `SingleLineInfilling/HumanEval/153/L1` | `25+` | 48 | 7 | `False` |  | 48 |
| missed_failed_long | `SingleLineInfilling/HumanEval/113/L3` | `25+` | 44 | 6 | `False` |  | 44 |
| missed_failed_long | `SingleLineInfilling/HumanEval/153/L3` | `25+` | 41 | 7 | `False` |  | 41 |

## Next Command

Run `--execute-pilot` only for this manifest or an explicitly reviewed `--task-ids-csv` subset. Do not use this scaffold for a full 1033-row run.
