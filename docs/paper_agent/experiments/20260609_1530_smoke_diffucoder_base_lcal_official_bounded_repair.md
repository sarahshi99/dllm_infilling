# DiffuCoder Base LCAL Official Bounded-Repair Smoke

Timestamp: 2026-06-09 15:30 CST

## Purpose

Start the next literature backbone after DreamCoder: `apple/DiffuCoder-7B-Base`. This is a compatibility smoke, not a claim-bearing experiment. The goal is to verify that the Dream-style official-canvas runner can load DiffuCoder, build the correct mask canvas, generate with `diffusion_generate`, run the local HumanEval verifier, and write the expected schema.

## Comparison Context

| Backbone | Local same-backbone baseline | Literature anchors | Claim status |
|---|---:|---|---|
| `apple/DiffuCoder-7B-Base` | none found yet | CAL DiffuCoder-Base `68.0` avg / `74.8` best shown; DreamOn DiffuCoder `92.2` training-based | smoke only; no protocol-matched claim |

## Command

Run outside the IDE sandbox because the model is not cached, and valid CUDA/verifier evidence requires non-sandboxed execution.

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_HOME=/tmp/hf_diffucoder_20260609 HF_MODULES_CACHE=/tmp/hf_modules_diffucoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_diffucoder_20260609_1530 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path apple/DiffuCoder-7B-Base --max-samples 2 --mask-length-source lcal_official_bounded_repair --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_diffucoder_base_lcal_official_bounded_repair_gpu2_unsandboxed --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260609_1530_smoke_diffucoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log
```

## Environment

- GPU: `CUDA_VISIBLE_DEVICES=2`.
- Python: `/home/shx/miniconda3/envs/dllm_env/bin/python`.
- Newer Transformers shim: `PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages`.
- Flash-attn guard: `DLLM_DISABLE_FLASH_ATTN=1`.
- HuggingFace mirror/cache: `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`, `HF_HOME=/tmp/hf_diffucoder_20260609`, `HF_MODULES_CACHE=/tmp/hf_modules_diffucoder_20260609`, `HF_DATASETS_CACHE=/tmp/hf_datasets_diffucoder_20260609_1530`.

## Success Criteria

- Command exits `0`.
- Output directory is created under `/home/shx/projects/dllm_infilling/outputs_clean/`.
- `results.jsonl` has `2` rows and `summary.json` exists.
- Rows include `metrics`, `verification`, `length_probe`, `lcal_v3`, and `official_cal`.
- Canvas/backend are compatible with Dream-style fixed-canvas infilling.
- Runtime fields include `total_sec_including_probe`.

## Kill Criteria

Do not launch a full run if model loading fails, `diffusion_generate` signature is incompatible, CUDA OOMs, the verifier fails for environment reasons, rows are malformed, or outputs collapse systematically.

## Result

Failed before model loading on 2026-06-09 15:31 CST.

- Log: `logs/paper_agent/20260609_1530_smoke_diffucoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log`.
- Exit code: `1`.
- Failure point: `AutoTokenizer.from_pretrained(cfg.model.model_path, trust_remote_code=True)`.
- Root cause: HuggingFace mirror metadata resolution for `apple/DiffuCoder-7B-Base` failed. The runner could not retrieve `config.json`, and the model was not present in local cache.
- Independent probe: `curl -I https://hf-mirror.com/apple/DiffuCoder-7B-Base/resolve/main/config.json` returned a `308` redirect to `https://huggingface.co/...`, while direct `huggingface.co` returned a valid repo commit/etag response. `huggingface-cli download` with `HF_ENDPOINT=https://hf-mirror.com` failed with the same `FileMetadataError` / `LocalEntryNotFoundError`.

Interpretation: this is a download/mirror blocker, not a DiffuCoder model result and not a method result. No GPU generation or verifier evidence was produced.

Next action: continue the backbone matrix with `Dream-org/Dream-v0-Base-7B` mirror download/API probe. Return to DiffuCoder later if the model is cached manually or a working mirror path is available.
