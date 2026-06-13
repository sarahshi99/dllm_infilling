# LLaDA-1.5 Download And API Probe

Timestamp: 2026-06-09 21:05 CST

## Purpose

Continue the literature-backbone matrix with `GSAI-ML/LLaDA-1.5`, which appears in LR-DLLM. This stage is a download/API compatibility check, not a performance result and not a protocol-matched comparison.

## Context

The user confirmed the configured proxy can access HuggingFace. The shell environment currently exposes:

- `HTTP_PROXY=http://127.0.0.1:7890`
- `HTTPS_PROXY=http://127.0.0.1:7890`
- `ALL_PROXY=http://127.0.0.1:7890`

For consistency with prior successful runs, the first attempt uses `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1` plus `/tmp` caches. If the mirror fails but direct HuggingFace works through the proxy, direct HuggingFace may be used for this probe and documented explicitly.

## Reviewer Motivation

LR-DLLM reports separate rows for `LLaDA-8B`, `LLaDA-1.5`, and `LLaDA-MoE`. A credible cross-backbone paper table should avoid mixing those rows under one "LLaDA" label. This probe starts the LLaDA-1.5 row so later results can be compared against a same-backbone local baseline.

## Planned Checks

1. Download small metadata files into `/tmp/llada15_probe_20260609`.
2. Inspect config architecture, model type, tokenizer class, mask token, and mask token id.
3. Decide whether the existing LLaDA-style runners are appropriate:
   - candidate: `clean_scripts/run_lcal_official_bounded_repair.py`;
   - local baseline: `clean_scripts/run_cal_lite_lcas_v3.py`.
4. Only after metadata/API compatibility is confirmed, run a tiny verifier smoke on GPU `2` or `3`.

## Commands

Small-file download:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/tmp/hf_llada15_20260609 TRANSFORMERS_CACHE=/tmp/hf_llada15_20260609 /home/shx/miniconda3/envs/llmxy/bin/python -m huggingface_hub.commands.huggingface_cli download GSAI-ML/LLaDA-1.5 config.json tokenizer_config.json special_tokens_map.json --cache-dir /tmp/hf_llada15_20260609 --local-dir /tmp/llada15_probe_20260609 --max-workers 1
```

Local metadata/API inspection:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -c "from transformers import AutoConfig, AutoTokenizer; p='/tmp/llada15_probe_20260609'; c=AutoConfig.from_pretrained(p, trust_remote_code=True); t=AutoTokenizer.from_pretrained(p, trust_remote_code=True); print(type(c)); print(getattr(c, 'architectures', None), getattr(c, 'model_type', None)); print('mask', getattr(t, 'mask_token', None), getattr(t, 'mask_token_id', None)); print('vocab', len(t))"
```

Future smoke candidate, only after the probe passes and GPU availability is rechecked:

```bash
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false HF_MODULES_CACHE=/tmp/hf_modules_llada15_20260609 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path /tmp/llada15_probe_20260609 --max-samples 2 --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_llada15_lcal_official_bounded_repair_gpu2 --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8
```

Future local baseline smoke, only after model API smoke is stable:

```bash
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false HF_MODULES_CACHE=/tmp/hf_modules_llada15_20260609 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_cal_lite_lcas_v3.py --model-path /tmp/llada15_probe_20260609 --max-samples 2 --probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24 --length-alpha 0.06 --lcas-policy lcas_v3b --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_llada15_cal_lite_lcas_v3b_gpu3
```

## Success Criteria

- Metadata download exits `0`.
- Local config/tokenizer inspection exits `0`.
- Mask token and mask token id are resolved.
- Runner family is documented before any smoke or full run.

## Kill Criteria

Stop before GPU smoke if metadata is incomplete, tokenizer mask token is missing, remote code cannot load, the model family is not LLaDA-compatible, or the required dependencies are absent.

## Result

Completed as a metadata/API/local-weight probe on 2026-06-09 21:55 CST.

### Network And Download Path

- The user's configured proxy can reach direct HuggingFace:
  - `curl -I https://huggingface.co/GSAI-ML/LLaDA-1.5/resolve/main/config.json` returned HTTP `307` with commit `84346fd91ba60252d260022201ad6fc5a3468fb2`.
  - `curl -I https://huggingface.co/api/models/GSAI-ML/LLaDA-1.5` returned HTTP `200`.
- `hf-mirror.com` redirects this repo back to `huggingface.co`, and `huggingface_hub` with `HF_ENDPOINT=https://hf-mirror.com` failed on the small-file path with `FileMetadataError`; direct HuggingFace through the proxy worked better.
- Small files were downloaded with direct `curl -sS -L` into `/tmp/llada15_probe_20260609`.
- Weight shards were downloaded with direct `curl -sS -fL`; the sixth shard needed several `curl -C -` resumes because the `cas-bridge.xethub.hf.co` TLS path repeatedly ended with `unexpected eof while reading`.

### Files And Sizes

Local model path: `/tmp/llada15_probe_20260609`.

Expected total shard size from HF API / `model.safetensors.index.json`: `16,031,197,144` bytes. Local byte-size verification passed:

| File | Size bytes | Status |
|---|---:|---|
| `model-00001-of-00006.safetensors` | `2,982,245,448` | ok |
| `model-00002-of-00006.safetensors` | `2,986,466,392` | ok |
| `model-00003-of-00006.safetensors` | `2,952,895,528` | ok |
| `model-00004-of-00006.safetensors` | `2,919,357,568` | ok |
| `model-00005-of-00006.safetensors` | `2,952,912,144` | ok |
| `model-00006-of-00006.safetensors` | `1,237,320,064` | ok |

### API Observations

- `architectures`: `["LLaDAModelLM"]`.
- `model_type`: `llada`.
- `AutoConfig`: `configuration_llada.LLaDAConfig`.
- `AutoModel`: `modeling_llada.LLaDAModelLM`.
- Config `mask_token_id`: `126336`.
- Tokenizer `mask_token` attribute: `None`; `tokenizer.convert_tokens_to_ids("<|mdm_mask|>") == 126336`.
- EOS/PAD token id: `126081`.
- Tokenizer vocab length observed by `AutoTokenizer`: `126349`.

### Local Load Smoke

Command shape:

```bash
HF_MODULES_CACHE=/tmp/hf_modules_llada15_20260609 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 /home/shx/miniconda3/envs/dllm_env/bin/python -c "..."
```

Observed:

- Checkpoint shards loaded from `/tmp/llada15_probe_20260609`.
- Loaded class: `LLaDAModelLM`.
- Parameter dtype: `torch.bfloat16`.
- Mask id agreement: config `126336`, tokenizer `<|mdm_mask|>` `126336`.
- This smoke loaded on CPU because the current sandboxed Python process reported `torch.cuda.is_available() == False` and `visible_count == 0`.

### GPU Status

GPU smoke was not launched. A fresh `nvidia-smi` at 2026-06-09 21:53 CST showed GPUs `2` and `3` occupied:

- GPU 2: about `30.7GB` used, including a Python process using about `25.3GB`.
- GPU 3: about `38.5GB` used, including a Python process using about `32.4GB`.

Do not interrupt these processes. The next step is a sandbox-outside tiny verifier smoke only after GPU `2` or `3` is actually free and the user or environment permits unsandboxed CUDA/verifier execution.

## Interpretation

LLaDA-1.5 is compatible with the LLaDA-style model family and can be represented by the existing LLaDA runner family, with one caveat: mask-token resolution should rely on config `mask_token_id` or the explicit `<|mdm_mask|>` token rather than `tokenizer.mask_token`.

No pass rate, runtime, or paper comparison should be reported from this probe. The next claim-relevant stage must generate a same-backbone local `cal_lite` baseline and a candidate smoke/full run under the same verifier/canvas/protocol.
