# A6000 Mid/Long Rescue Report

Updated: 2026-05-29 Asia/Shanghai

## Status

The four A6000 recovery runs completed on the full `HumanEval-SingleLineInfilling` test split (`1033` tasks each). Raw outputs remain local under `outputs_clean/`; compact pairwise summaries and the generated scoreboard are tracked under `analysis_outputs/a6000_midcons_longrescue/`.

## Commands

Recovery launcher committed in `clean_scripts/resume_lcal_a6000_four_policies_offline.sh`:

```bash
bash clean_scripts/resume_lcal_a6000_four_policies_offline.sh
```

Pairwise analysis:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_lcal_pairwise.py \
  --base-results outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529/results.jsonl \
  --candidate-results <candidate-run-dir>/results.jsonl \
  --output-dir analysis_outputs/a6000_midcons_longrescue/<label>_vs_a6000_control
```

Scoreboard generation:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_a6000_scoreboard_section.py
```

## Main Result

| Run | Pass | Delta vs A6000 control | <=8 | 9-12 | 13-16 | 17-24 | 25+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6000 control | `787/1033` `76.19%` | baseline | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` |
| midcons | `795/1033` `76.96%` | `+8` (8W/0L) | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` |
| mid_precision | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` |
| true_long | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` |
| combined | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` |

## Interpretation

`midcons` is the current A6000 best checkpoint. It improves the A6000 control by `+8` wins with `0` losses. The gains are concentrated in medium lengths:

- `<=8`: +1 net, no losses.
- `9-12`: +3 net, no losses.
- `13-16`: +4 net, no losses.
- `17-24` and `25+`: unchanged.

The eight wins all use `official_mid_rescue` as the final source. This validates the conservative medium-length rescue rule under same-hardware conditions.

The `mid_precision`, `true_long`, and `combined` candidates are not improvements. In these runs, the added precision guards remove all mid-rescue triggers, and the true-long branch triggers zero times. They are useful negative evidence: the current official-CAL based true-long trigger is too conservative and not a viable long-tail fix.

## Why True-Long Did Not Help

The true-long branch required all of these to pass: official length at least 17, large delta, high adjusted/raw long ratio, support count, allowed best-long length, and source gate.

Diagnostics from `true_long`:

- `official_repair_true_long_len_passed`: 17 tasks.
- `official_repair_true_long_delta_passed`: 47 tasks.
- `official_repair_true_long_support_passed`: 2 tasks.
- all gates passed: 0 tasks.
- all gates except support passed: 4 tasks, but all four were already-passing short/medium cases.

That means relaxing the support guard would mostly add false positives, not solve the long bucket. The core problem is different: among `midcons` long failures (`oracle >= 17`), `90/91` under-select the length and `71/91` still come from `base`. Official-CAL often selects very short lengths (`1`, `2`, `6`, `7`, `9`) on true-long failures, so official-CAL is not reliable enough to serve as the long trigger.

## Output Paths

- A6000 control: `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529`
- midcons: `outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626`
- mid_precision: `outputs_clean/full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517`
- true_long: `outputs_clean/full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755`
- combined: `outputs_clean/full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756`

## Next Decision

Use `midcons` as the current A6000 checkpoint for short/medium lengths. For the long bucket, stop iterating on the current official-CAL true-long trigger and design a separate long-underestimation detector that uses long-curve evidence and failure signatures without sacrificing `<=8` and `9-12`.

Follow-up diagnostic:

- `analysis_outputs/long_underestimate_detector/a6000_midcons/sweep.md`
- `docs/results/long_underestimate_detector_report.md`

The offline sweep found no safe heuristic long-underestimation rule from the current result fields. This strengthens the decision to move toward trajectory features, learned length classification, or literature-style length regularization for the true-long branch.
