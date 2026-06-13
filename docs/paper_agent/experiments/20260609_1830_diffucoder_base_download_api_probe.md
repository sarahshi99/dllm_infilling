# DiffuCoder Base Download And API Probe

Timestamp: 2026-06-09 18:30 CST

## Purpose

Continue the literature-backbone matrix with `apple/DiffuCoder-7B-Base`, which appears in CAL and DreamOn. This stage is not a performance result. It only checks whether the exact checkpoint can be downloaded through the mirror/Git-LFS path and whether its tokenizer/model API matches the Dream-style official-canvas runner.

## Context

Prior `huggingface_hub` mirror metadata attempts for DiffuCoder failed before model loading. The Dream-7B Git/LFS path worked, so this stage uses the same cautious path:

1. `GIT_LFS_SKIP_SMUDGE=1 git ls-remote` against `https://hf-mirror.com/apple/DiffuCoder-7B-Base`.
2. `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1` into `/tmp/diffucoder_base_repo_probe_20260609`.
3. Inspect `config.json`, tokenizer metadata, remote modeling files, and LFS shard sizes.
4. Pull LFS weights only after confirming disk space.
5. Run a tiny API/verifier smoke before any full run.

## Metadata Observed Before Weight Pull

- HEAD: `a2c33054cbe99b9ab7af074f09b1d13943157ff1`.
- Local metadata path: `/tmp/diffucoder_base_repo_probe_20260609`.
- Architecture: `DreamModel`.
- `model_type`: `Dream`.
- `mask_token_id`: `151666`.
- Tokenizer mask token: `<|mask|>`.
- Canvas runner candidate: `clean_scripts/run_dreamcoder_official_infilling.py`.
- LFS shard sizes: `4.88GB`, `4.93GB`, `4.33GB`, `1.09GB`, total about `15.2GB`.
- `/tmp` free before weight pull: about `93GB`.

## Planned Weight Pull

```bash
git -C /tmp/diffucoder_base_repo_probe_20260609 lfs pull
```

## Success Criteria

- LFS pull completes and four safetensors shards are materialized.
- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` loads from the local path.
- `AutoModel.from_pretrained(..., trust_remote_code=True)` loads on one GPU without OOM.
- The model exposes the Dream-style `diffusion_generate` path used by the existing official-canvas runner.
- A tiny verifier smoke writes valid rows with verifier outputs, backend, canvas, mask length, oracle length, and runtime.

## Kill Criteria

Stop before full run if the LFS pull fails, model API differs from Dream/DreamCoder, tokenizer special tokens are inconsistent, GPU memory exceeds a safe single-GPU footprint, verifier fields are missing, or the tiny smoke shows prompt/canvas collapse.

## Result

Completed on 2026-06-09 19:23 CST.

- LFS pull completed and four safetensors shards were materialized:
  - `model-00001-of-00004.safetensors`: `4.6G`
  - `model-00002-of-00004.safetensors`: `4.6G`
  - `model-00003-of-00004.safetensors`: `4.1G`
  - `model-00004-of-00004.safetensors`: `1.1G`
- `git lfs ls-files --debug` reports `checkout: true` and `download: true` for all four shards.
- A valid 2-sample smoke completed:
  - Log: `logs/paper_agent/20260609_1922_smoke_diffucoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log`
  - Output: `/home/shx/projects/dllm_infilling/outputs_clean/smoke_diffucoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_192200`
  - Result: `2/2 PASS`
  - Avg total sec including probe: `3.1293`
  - Backend: `dreamcoder_native_diffusion_generate_fixed_canvas`
  - Canvas: `bos_prefix_masks_suffix_eos`
  - Summary and verifier fields are present.

## Interpretation

DiffuCoder-Base appears compatible with the existing Dream-style official fixed-canvas runner. The smoke is not a performance result, but it passes the API/schema/verifier gate required before full runs. The next claim-relevant step is a local same-backbone pair: `cal_lite` baseline and `lcal_official_bounded_repair` candidate.
