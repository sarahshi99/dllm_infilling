# clean_scripts index

This directory contains small entrypoint scripts for length-selection and stopping experiments.
The names are historical, so use this file as the current map.

## Current main line

| Script | Use | Status |
| --- | --- | --- |
| `run_cal_lite_lcas_v3.py` | Base CAL-lite length selection plus LCAS-v3 stopping. No long-aware correction. | Stable baseline. |
| `run_cal_lite_lcal_v3.py` | LCAL-v3: base CAL-lite selection, then long-aware correction only when `base_selected_length >= 13`. | Stable LCAL-v3 baseline. |
| `run_cal_lite_lcal_v3_t2_ratio.py` | LCAL-v3 experimental T2: adds a ratio trigger branch using `best_long_score / best_score`. | Experimental; useful for analysis, not recommended as new default. |

## Older or auxiliary scripts

| Script | Use | Status |
| --- | --- | --- |
| `run_cal_lite.py` | Plain CAL-lite entrypoint. | Legacy/base utility. |
| `run_cal_lite_lcas.py` | Earlier LCAS entrypoint. | Legacy. |
| `run_cal_lite_lcas_b.py` | Earlier LCAS-B entrypoint. | Legacy. |
| `run_cal_lite_stop.py` | CAL-lite with stopping controls. | Auxiliary. |
| `run_fixed_stop.py` | Fixed-length decode with stopping controls. | Auxiliary. |
| `run_vanilla_fixed.py` | Vanilla fixed-length decode. | Baseline utility. |
| `run_vanilla_oracle.py` | Vanilla oracle-length decode. | Diagnostic baseline. |
| `run_lcas_v3_model_sweep.py` | LCAS-v3 model sweep entrypoint. | Auxiliary. |
| `analyze_length_failure.py` | Failure analysis helper. | Analysis utility. |

## Terms

- CAL-lite: selects one mask length from a probe grid.
- LCAS: length-conditional adaptive stopping. It controls when decoding stops after a length is chosen.
- LCAL: long-aware correction on top of CAL-lite. It tries a stronger long grid only for samples that look long.
- T2 ratio branch: an additional LCAL trigger condition. Besides `base_selected_length >= 13`, it also triggers if the best long candidate score is close enough to the best overall score.

## Current recommendation

For reproducible comparisons, keep the old filenames and use experiment names to encode variants.
Do not make `run_cal_lite_lcal_v3_t2_ratio.py` the default: it improves some medium/long cases but introduces short-sample losses.
