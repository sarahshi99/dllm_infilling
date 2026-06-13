# Trace Long Rescue Full-Trace Plan

Timestamp: 2026-06-11 CST

## Action Name

Collect full LLaDA-Base generation traces and evaluate three pure inference-time long-length recovery routes.

## Current Phase And Plan Version

Post-backbone-matrix long-length recovery design under experiment plan `v3`.

## Reviewer Motivation

Current methods improve short/medium behavior but still under-select true-long infilling. A CCF-A reviewer will need evidence that any long-length recovery is not just a short-bucket tradeoff.

## Hypothesis

Generation trace dynamics can distinguish true-long under-selection from safe short/medium cases better than probe-curve scalar fields alone.

## Fixed Success Gate

Gate A: oracle `>=17` bucket improves, overall pass count does not decrease, and oracle `<=8` short-bucket net loss is at most `2` tasks.

Gate B: long-bucket gains with small overall regression are exploratory only.

## Routes

1. Trace-only true-long detector.
2. Risk-controlled long rescue.
3. Multi-canvas trace reranking without verifier.

## Baselines And Comparisons

- Literature reported values remain external reported numbers.
- Previous local method: A6000 control `787/1033 = 76.19%`.
- Current local method: `midcons` `795/1033 = 76.96%`.
- New trace policies must compare against both previous local method and current method.

## Full Trace Collection Commands

Previous local method trace run on GPU `2`:

```bash
tmux new-session -d -s trace_llada_base_prev_20260611 -c /home/shx/projects/dllm_infilling/git_workspace "script -q -e -c \"HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_cal_lite_lcas_v3.py --model-path GSAI-ML/LLaDA-8B-Base --probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24 --length-alpha 0.06 --lcas-policy lcas_v3b --save-step-traces --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_trace_llada_base_cal_lite_lcas_v3b_gpu2\" logs/paper_agent/20260611_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log"
```

Current method trace run on GPU `3`:

```bash
tmux new-session -d -s trace_llada_base_midcons_20260611 -c /home/shx/projects/dllm_infilling/git_workspace "script -q -e -c \"HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path GSAI-ML/LLaDA-8B-Base --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_trace_llada_base_midcons_gpu3 --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8 --save-step-traces\" logs/paper_agent/20260611_full_trace_llada_base_midcons_gpu3.log"
```

## Success Criteria

- Both logs end with `COMMAND_EXIT_CODE="0"`.
- Both outputs have `1033` valid `results.jsonl` rows.
- Both outputs have non-empty `step_traces.jsonl`.
- Trace analysis can report route 1/2/3 Gate A and Gate B summaries.

## Kill Criteria

- Import failure, OOM, verifier failure, malformed rows, missing `summary.json`, empty traces, or pass-rate collapse relative to previous local method/current method.

## Expected Documentation Outputs

- Updated `experiment_results.*.md`.
- Updated dashboard, checkpoint, and activity ledger.
- Terminal table: literature reported numbers / previous local method / current method / trace route candidates.
