# LLaDA-MoE Smoke Gate Action Brief

Timestamp: 2026-06-11 11:12 CST

## Action

Run the `inclusionAI/LLaDA-MoE-7B-A1B-Base` local baseline/candidate smoke gate on GPUs `2/3`.

## Reviewer Motivation

The backbone appears in LR-DLLM and should be covered by a local same-backbone baseline/candidate pair before any paper claim compares across related literature.

## Hypothesis

The current LCAL official bounded repair method may be a near-tie or small positive on LLaDA-MoE, consistent with the backbone-dependent behavior observed on LLaDA-1.5, Dream, DreamCoder, and DiffuCoder. The smoke gate only validates runner compatibility; it is not evidence of pass-rate superiority.

## Environment

- Repo: `/home/shx/projects/dllm_infilling/git_workspace`
- Model path: `/tmp/lladamoe_probe_20260610`
- Python entrypoint: `/home/shx/miniconda3/envs/dllm_env/bin/python`
- Compatibility path: `PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages`
- Flash-attn shim: `/tmp/no_flash_attn`, used after the first baseline smoke showed the `dllm_env` `flash_attn_2_cuda` extension requires unavailable `GLIBC_2.32`.
- Offline mode: `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`
- GPUs: baseline on `2`, candidate on `3`
- Logs: `logs/paper_agent/20260611_1112_smoke_lladamoe_*.log`
- Output root: `/home/shx/projects/dllm_infilling/outputs_clean`

## Exact Commands

Baseline smoke:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_lladamoe_20260610 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_cal_lite_lcas_v3.py --model-path /tmp/lladamoe_probe_20260610 --max-samples 2 --probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24 --length-alpha 0.06 --lcas-policy lcas_v3b --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_lladamoe_cal_lite_lcas_v3b_gpu2_shared" logs/paper_agent/20260611_1112_smoke_lladamoe_cal_lite_lcas_v3b_gpu2_shared.log
```

Baseline retry with flash-attn hidden:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/tmp/no_flash_attn:/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_lladamoe_20260610 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_cal_lite_lcas_v3.py --model-path /tmp/lladamoe_probe_20260610 --max-samples 2 --probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24 --length-alpha 0.06 --lcas-policy lcas_v3b --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared" logs/paper_agent/20260611_1118_smoke_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared.log
```

Candidate smoke:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_lladamoe_20260610 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path /tmp/lladamoe_probe_20260610 --max-samples 2 --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_lladamoe_lcal_official_bounded_repair_gpu3_shared --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260611_1112_smoke_lladamoe_lcal_official_bounded_repair_gpu3_shared.log
```

Candidate retry with flash-attn hidden:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/tmp/no_flash_attn:/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_lladamoe_20260610 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path /tmp/lladamoe_probe_20260610 --max-samples 2 --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260611_1118_smoke_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared.log
```

## First Failure Root Cause

The first baseline smoke exited `1` before loading weights. Root cause: Transformers `4.52.3` imported `transformers.integrations.flash_attention`, detected `dllm_env`'s `flash_attn` distribution, and imported `flash_attn_2_cuda`, which requires unavailable `GLIBC_2.32`. A local shim at `/tmp/no_flash_attn` makes `is_flash_attn_2_available()` return false without modifying the environment.

## Success Criteria

Both smokes must exit `0`, produce two valid rows, and write valid summaries before any full run starts.

## Kill Criteria

Stop and investigate before full runs on model import failure, OOM, verifier failure, malformed JSON, missing summary, or suspiciously empty/corrupted decoded outputs.
