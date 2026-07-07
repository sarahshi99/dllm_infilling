# Controller Action Bank

- row_count: `4635`
- task_count: `927`
- split_counts: `{'calibration': 775, 'train': 3225, 'validation': 635}`
- action_counts: `{'EXPAND_16': 927, 'EXPAND_24': 927, 'EXPAND_32': 927, 'EXPAND_48': 927, 'KEEP_PRIMARY': 927}`
- pass_count: `2506`
- benefit_labels_non_keep: `193`
- harm_labels_non_keep: `1237`
- cost: `{'mean': 1.1893461289886307, 'p50': 1.1560063179931603, 'p95': 1.5792336129234172}`

The bank is generated only for train/calibration/validation. Test rows remain sealed unless the validation gate authorizes one frozen test run.
