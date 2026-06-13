# DreamCoder Instruct LCAL Official Bounded-Repair Smoke

Timestamp: 2026-06-09 12:25 CST

## Purpose

Validate that `Dream-org/Dream-Coder-v0-Instruct-7B` can run through the same DreamCoder official fixed-canvas LCAL/S3 + official-CAL bounded-repair adapter before launching full runs.

## Comparison Context

| Category | Anchor |
|---|---|
| Local same-backbone baseline | DreamCoder-Instruct official-canvas cal_lite `848/1033 = 82.09%` |
| Literature anchor | DreamCoder-7B appears in CAL, LR-DLLM, and DreamOn, but paper tables usually do not separate this local instruct checkpoint from base |

This smoke only tests validity of the runner and output schema. It is not a claim-bearing comparison.

## Exact Command

```bash
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path Dream-org/Dream-Coder-v0-Instruct-7B --max-samples 2 --mask-length-source lcal_official_bounded_repair --baseline-results /home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_instruct_alpha010_cap24_official_canvas_20260513_232724/results.jsonl --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260609_1225_smoke_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed.log
```

## Environment

- Workdir: `/home/shx/projects/dllm_infilling/git_workspace`.
- GPU: `CUDA_VISIBLE_DEVICES=3`.
- Python: `/home/shx/miniconda3/envs/dllm_env/bin/python`.
- Compatibility: `PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages`, `DLLM_DISABLE_FLASH_ATTN=1`.
- HuggingFace mirror/cache: `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`, offline cached model mode, `HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609`, `HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205`.

## Acceptance Criteria

- Process exits `0`.
- `summary.json` exists with `num_samples=2`.
- `results.jsonl` has exactly two valid JSON rows.
- Rows contain verifier outputs and LCAL/official-CAL metadata.
- Reconstruction is not obviously collapsed by the prompt/canvas format.

## Result

- Output: `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_122945`.
- Exit status: `0`.
- Rows: `2`.
- Summary: `2/2 = 100%`.
- Avg total sec including probe: `3.0555`.
- Baseline pairwise over the two common samples: `0` wins, `0` losses, `2` tie-pass.
- Verifier keys: `tier1_parse_compile`, `tier2_smoke_exec`, `tier3_unit_tests`.
- LCAL source: both rows `final_source=base`; `official_repair_trigger_count=0`.

Interpretation: acceptance criteria passed for runner/schema/verifier/GPU-runtime sanity. This is not a performance estimate and should not be compared as a paper result.
