# Controller Action Bank

- row_count: `4635`
- task_count: `927`
- split_counts: `{'calibration': 775, 'train': 3225, 'validation': 635}`
- action_counts: `{'EXPAND_16': 927, 'EXPAND_24': 927, 'EXPAND_32': 927, 'EXPAND_48': 927, 'KEEP_PRIMARY': 927}`
- pass_count: `2504`
- benefit_labels_non_keep: `191`
- harm_labels_non_keep: `1242`
- cost: `{'mean': 3.52723535730289, 'p50': 3.3891470790840685, 'p95': 5.8281581461196765}`

## Merge Coverage

- expected_action_rows: `4635`
- observed_action_rows: `4635`
- missing_action_rows: `0`
- extra_action_rows: `0`
- duplicate_source_action_rows: `10`

The bank is generated only for train/calibration/validation. Test rows remain sealed unless the validation gate authorizes one frozen test run.
