# DreamCoder Base LCAL Official Bounded-Repair Smoke

Timestamp: 2026-06-09 12:03 CST

## Purpose

Validate that `Dream-org/Dream-Coder-v0-Base-7B` can run through the DreamCoder official fixed-canvas runner with the adapted LCAL/S3 + official-CAL bounded-repair policy before launching a full `1033`-sample run.

## Comparison Context

| Category | Anchor |
|---|---|
| Local same-backbone baseline | DreamCoder-Base official-canvas cal_lite `825/1033 = 79.86%` |
| CAL literature anchor | DreamCoder-Base CAL `70.2` avg, best shown `76.2` |
| LR-DLLM literature anchor | DreamCoder-7B single-line `81.6` |
| DreamOn anchor | DreamCoder-7B + DreamOn `92.1`, training-based and not a clean inference-only baseline |

This smoke only tests validity of the runner and output schema. It is not a claim-bearing comparison.

## Exact Command

```bash
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path Dream-org/Dream-Coder-v0-Base-7B --max-samples 2 --mask-length-source lcal_official_bounded_repair --baseline-results /home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721/results.jsonl --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260609_1212_smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log
```

## Environment

- Workdir: `/home/shx/projects/dllm_infilling/git_workspace`.
- GPU: `CUDA_VISIBLE_DEVICES=2`.
- Python: `/home/shx/miniconda3/envs/dllm_env/bin/python`.
- Package path: `PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages`, yielding Transformers `4.52.3` and torch `2.7.0+cu126` in the probe.
- Optional attention: `DLLM_DISABLE_FLASH_ATTN=1`, because the visible `dllm_env` flash-attn binary requires unavailable `GLIBC_2.32`.
- HuggingFace mirror/cache: `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`, offline cached model mode, `HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609`, `HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205`.

## Acceptance Criteria

- Process exits `0`.
- `summary.json` exists with `num_samples=2`.
- `results.jsonl` has exactly two valid JSON rows.
- Rows contain verifier outputs and LCAL/official-CAL metadata.
- Reconstruction is not obviously collapsed by the prompt/canvas format.

## Result

First launch failed before decoding because `datasets` attempted to create a lock file under read-only `/home/shx/.cache/huggingface/datasets`. The dataset cache was copied to `/tmp/hf_datasets_dreamcoder_20260609_1205`, and a dataset-load probe succeeded with two tasks.

Second launch used the `/tmp` dataset and module caches. It loaded the model, loaded two tasks, and entered the first task, but failed inside the local HumanEval verifier:

```text
PermissionError: [Errno 1] Operation not permitted
...
EOFError
```

Root cause: the IDE sandbox blocks the `multiprocessing.Manager()` listener socket used by the HumanEval verifier. Separate CUDA probes also showed `torch.cuda.is_available() == False` in the sandbox, so the attempted smoke is invalid for both runtime and pass-rate evidence.

Next valid action: rerun the same 2-sample smoke outside the sandbox, with GPU 2 visible and the same `/tmp` HF caches. This requires explicit user approval because the automatic escalation reviewer returned `503 Service Unavailable`.

The first approved unsandboxed launch failed before model loading because Transformers imported the optional `flash_attn` package from `dllm_env`, whose binary requires unavailable `GLIBC_2.32`. The runner now supports `DLLM_DISABLE_FLASH_ATTN=1`; a local import probe with that flag succeeded.

Final approved unsandboxed smoke result:

- Output: `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_122358`.
- Exit status: `0`.
- Rows: `2`.
- Summary: `2/2 = 100%`.
- Avg total sec including probe: `3.1439`.
- Baseline pairwise over the two common samples: `0` wins, `0` losses, `2` tie-pass.
- Verifier keys: `tier1_parse_compile`, `tier2_smoke_exec`, `tier3_unit_tests`.
- LCAL source: both rows `final_source=base`; `official_repair_trigger_count=0`.

Interpretation: acceptance criteria passed for runner/schema/verifier/GPU-runtime sanity. This is not a performance estimate and should not be compared as a paper result.
