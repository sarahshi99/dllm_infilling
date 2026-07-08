# Dream-Coder Expanded Diagnostic Manifest

Verdict: `dreamcoder_expanded_manifest_ready`.
Cases: `37`.
Strata: `{'missed_failed_long': 12, 'triggered_failed_long': 4, 'medium_near_long_underselection': 8, 'short_primary_pass_harmable': 6, 'positive_control_recoverable': 7}`.
Splits: `{'validation': 6, 'train': 30, 'calibration': 1}`.
Oracle buckets: `{'25+': 14, '17-24': 10, '<=8': 8, '13-16': 2, '9-12': 3}`.
Frozen test rows: `0`.

This is a CPU-only manifest construction step. It prepares a bounded 37-case Dream-Coder oracle-sufficient diagnostic on train/calibration/validation-derived SingleLine rows; it does not run model generation and does not open the frozen test. The intended cap was 45, but only 4 triggered failed-long rows matched the current train/calibration/validation proxy definition.

Expected GPU command:

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/shx/miniconda3/envs/dllm_env/bin/python experiments/phase4_continuation.py --mode dream-oracle --timestamp 20260708_dreamcoder_expanded37_v1 --manifest analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/case_manifest.csv
```
