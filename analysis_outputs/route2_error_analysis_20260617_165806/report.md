# Route2 Error Analysis

Decision: `mixed_rescue_quality_and_gate_recall`

Recommended next path: `rescue_generation_quality+gate_recall`

## Overview

- Joined rows: `1033`
- Pairwise W/L/TP/TF: `6/0/795/232`
- Route2 triggers: `57`
- Triggered failed-long rows: `33`
- Missed failed-long rows: `56`
- Rescue length >= oracle among triggered failed-long: `31/33`

## Bucket Pairwise

| Bucket | Win | Loss | Tie pass | Tie fail | Triggers |
|---|---:|---:|---:|---:|---:|
| `13-16` | `0` | `0` | `53` | `37` | `10` |
| `17-24` | `2` | `0` | `17` | `63` | `24` |
| `25+` | `0` | `0` | `5` | `26` | `11` |
| `9-12` | `2` | `0` | `182` | `48` | `6` |
| `<=8` | `2` | `0` | `538` | `58` | `6` |

## Interpretation

The diagnostic separates gate recall from rescue quality. If most triggered failed-long rows already have rescue length greater than or equal to oracle length, another blind length increase is not the default next step.

The current result supports a mixed next direction when both triggered failed-long and missed failed-long counts are substantial: inspect rescue generation/selection quality and expand probe-trace fusion for missed failed-long rows.

## Example Rows

| Class | Task | Bucket | Pairwise | Triggered | Primary len | Rescue len | Final len |
|---|---|---|---|---:|---:|---:|---:|
| `route2_win` | `SingleLineInfilling/HumanEval/116/L0` | `17-24` | `win` | `True` | `3` | `32` | `32` |
| `route2_win` | `SingleLineInfilling/HumanEval/16/L0` | `<=8` | `win` | `True` | `3` | `32` | `32` |
| `route2_win` | `SingleLineInfilling/HumanEval/34/L0` | `<=8` | `win` | `True` | `3` | `32` | `32` |
| `triggered_failed_long` | `SingleLineInfilling/HumanEval/108/L6` | `25+` | `tie_fail` | `True` | `3` | `32` | `32` |
| `triggered_failed_long` | `SingleLineInfilling/HumanEval/11/L6` | `17-24` | `tie_fail` | `True` | `3` | `32` | `32` |
| `triggered_failed_long` | `SingleLineInfilling/HumanEval/115/L0` | `17-24` | `tie_fail` | `True` | `3` | `32` | `32` |
| `missed_failed_long` | `SingleLineInfilling/HumanEval/10/L5` | `17-24` | `tie_fail` | `False` | `16` | `None` | `16` |
| `missed_failed_long` | `SingleLineInfilling/HumanEval/104/L2` | `17-24` | `tie_fail` | `False` | `6` | `None` | `6` |
| `missed_failed_long` | `SingleLineInfilling/HumanEval/107/L7` | `17-24` | `tie_fail` | `False` | `10` | `None` | `10` |
| `short_or_medium_win` | `SingleLineInfilling/HumanEval/16/L0` | `<=8` | `win` | `True` | `3` | `32` | `32` |
| `short_or_medium_win` | `SingleLineInfilling/HumanEval/34/L0` | `<=8` | `win` | `True` | `3` | `32` | `32` |
| `short_or_medium_win` | `SingleLineInfilling/HumanEval/60/L0` | `9-12` | `win` | `True` | `3` | `32` | `32` |
