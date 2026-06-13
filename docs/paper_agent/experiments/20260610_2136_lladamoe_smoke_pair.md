# LLaDA-MoE Baseline/Candidate Smoke Pair Plan

Timestamp: 2026-06-10 21:36 CST

## Purpose

Run tiny verifier smokes for `inclusionAI/LLaDA-MoE-7B-A1B-Base` after the local weight download and API checks pass. This is a runner/schema/verifier/memory gate, not a performance result.

## Comparison Design

| Backbone | Local baseline | Candidate | Literature anchor |
|---|---|---|---|
| `inclusionAI/LLaDA-MoE-7B-A1B-Base` | `cal_lite` LCAS-v3b, alpha `0.06` | LCAL official bounded repair with the current `midcons` thresholds | LR-DLLM LLaDA-MoE single-line `71.3`; reported baseline `48.8` |

The literature numbers are anchors only. A claim requires the local same-backbone baseline/candidate pair.

## Preconditions

- Local path `/tmp/lladamoe_probe_20260610` contains `config.json`, tokenizer files, remote-code files, `model.safetensors.index.json`, and all three safetensors shards.
- No `.aria2` files remain in `/tmp/lladamoe_probe_20260610`.
- Shard byte sizes sum to `14,713,761,792` bytes according to the safetensors index/API probe.
- `AutoConfig`, `AutoTokenizer`, and at least a local model-class/load probe pass with `trust_remote_code=True`.
- GPU `2` and `3` have enough free memory despite the existing shared jobs.

## Launch Order

Start the baseline smoke first on GPU `2`. After it has loaded the model and emitted the run directory or first task output, start the candidate smoke on GPU `3`. This follows the user's instruction that two parallel experiments do not have to start simultaneously when backbone connection behavior may differ.

## Baseline Smoke Command

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_lladamoe_20260610 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_cal_lite_lcas_v3.py --model-path /tmp/lladamoe_probe_20260610 --max-samples 2 --probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24 --length-alpha 0.06 --lcas-policy lcas_v3b --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_lladamoe_cal_lite_lcas_v3b_gpu2_shared" logs/paper_agent/20260611_1112_smoke_lladamoe_cal_lite_lcas_v3b_gpu2_shared.log
```

## Candidate Smoke Command

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_lladamoe_20260610 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path /tmp/lladamoe_probe_20260610 --max-samples 2 --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_lladamoe_lcal_official_bounded_repair_gpu3_shared --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260611_1112_smoke_lladamoe_lcal_official_bounded_repair_gpu3_shared.log
```

## Kill Criteria

Stop before full runs if either smoke has unresolved mask-token handling, remote-code import failure, OOM, verifier failure, malformed JSON rows, missing `summary.json`, near-zero pass collapse on the first two examples, or suspiciously empty/corrupted generated text.

## Full-Run Gate

Only after both smokes exit `0`, produce two valid rows each, and write valid summaries should the full local same-backbone pair be launched.
