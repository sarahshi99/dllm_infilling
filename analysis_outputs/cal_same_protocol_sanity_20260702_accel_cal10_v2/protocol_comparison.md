# CAL Same-Protocol Sanity

verdict: `protocol_matched_cal`

## Checks

| Check | Pass |
|---|---:|
| `checkpoint_same` | `True` |
| `dataset_subset_same` | `True` |
| `split_same` | `True` |
| `prompt_format_same` | `True` |
| `mask_initialization_same` | `True` |
| `denoising_budget_comparable` | `True` |
| `evaluation_harness_same` | `True` |
| `seed_protocol_explicit` | `True` |
| `no_oracle_length_used_for_generation` | `True` |
| `no_test_result_used_for_generation` | `True` |
| `output_schema_paired_compare` | `True` |
| `official_cal_metadata_present` | `True` |
| `fixed_case_filter_wrapper` | `True` |

## Notes

- This sanity uses the local `run_cal_official_lcas_v3.py` implementation and a wrapper-level fixed 10-case filter.
- It is a same-repository protocol sanity, not an external official-code reproduction claim.
- If verdict is `protocol_matched_cal`, a full run may be compared under the exact same local protocol; otherwise no misleading full comparison should be launched.

## Artifacts

- run dir: `analysis_outputs/cal_same_protocol_sanity_20260702_accel_cal10_v2`
- results: `analysis_outputs/cal_same_protocol_sanity_20260702_accel_cal10_v2/results.csv`
- summary: `analysis_outputs/cal_same_protocol_sanity_20260702_accel_cal10_v2/summary.json`
