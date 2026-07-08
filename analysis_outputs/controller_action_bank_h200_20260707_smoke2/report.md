# Controller Action Bank

- row_count: `10`
- task_count: `2`
- split_counts: `{'train': 10}`
- action_counts: `{'EXPAND_16': 2, 'EXPAND_24': 2, 'EXPAND_32': 2, 'EXPAND_48': 2, 'KEEP_PRIMARY': 2}`
- pass_count: `8`
- benefit_labels_non_keep: `0`
- harm_labels_non_keep: `2`
- cost: `{'mean': 1.1693930370965973, 'p50': 1.164764957036823, 'p95': 1.4274737990344875}`

The bank is generated only for train/calibration/validation. Test rows remain sealed unless the validation gate authorizes one frozen test run.
