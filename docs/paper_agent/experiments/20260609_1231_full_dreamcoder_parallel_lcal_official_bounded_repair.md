# Full DreamCoder Base/Instruct LCAL Official Bounded-Repair Runs

Timestamp: 2026-06-09 12:31 CST

## Purpose

Launch full `1033`-sample runs for the two cached DreamCoder backbones after both official-canvas smoke tests passed. These are the first claim-relevant DreamCoder adapted-policy runs, but they still require full completion, sanity checks, pairwise comparisons, and literature-positioning before any claim.

## Comparison Context

| Backbone | Local same-backbone baseline | Literature anchors |
|---|---:|---|
| `Dream-org/Dream-Coder-v0-Base-7B` | official-canvas cal_lite `825/1033 = 79.86%` | CAL DreamCoder-Base `70.2` avg / `76.2` best shown; LR-DLLM DreamCoder-7B `81.6`; DreamOn DreamCoder `92.1` training-based |
| `Dream-org/Dream-Coder-v0-Instruct-7B` | official-canvas cal_lite `848/1033 = 82.09%` | no perfectly separated paper row for this local instruct checkpoint; use DreamCoder anchors cautiously |

## Base Command

Session: `dreamcoder_base_full_20260609_1231`

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path Dream-org/Dream-Coder-v0-Base-7B --mask-length-source lcal_official_bounded_repair --baseline-results /home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721/results.jsonl --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260609_1231_full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log
```

## Instruct Command

Session: `dreamcoder_instruct_full_20260609_1231`

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path Dream-org/Dream-Coder-v0-Instruct-7B --mask-length-source lcal_official_bounded_repair --baseline-results /home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_instruct_alpha010_cap24_official_canvas_20260513_232724/results.jsonl --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260609_1231_full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed.log
```

## Environment

- GPUs: Base on `CUDA_VISIBLE_DEVICES=2`; Instruct on `CUDA_VISIBLE_DEVICES=3`.
- Python: `/home/shx/miniconda3/envs/dllm_env/bin/python`.
- Compatibility: `PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages`, `DLLM_DISABLE_FLASH_ATTN=1`.
- HuggingFace mirror/cache: `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`, offline cached models, `/tmp` HF modules/datasets cache.

## Success Criteria

- Each run exits `0`.
- Each `results.jsonl` has `1033` valid rows.
- Each `summary.json` exists and has `num_samples=1033`.
- Pairwise baseline comparisons are present.
- Bucket metrics, repair-source histograms, verifier outputs, and runtime fields are present.

## Kill Criteria

Stop and investigate if either run OOMs, produces malformed JSON rows, loses verifier fields, uses the wrong canvas/backend, or shows early systematic reconstruction collapse.

## Result

Completed and checked on 2026-06-09 14:28 CST.

- Base output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`.
- Instruct output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`.
- Both `results.jsonl` files have `1033` rows and both `summary.json` files exist.
- `nvidia-smi` showed GPUs 2/3 free after completion.
- `tmux ls` no longer listed the DreamCoder full-run sessions.

### Main Same-Backbone Comparison

| Backbone | Candidate | Local same-backbone baseline | Delta | Pairwise W/L/TP/TF | Avg sec/sample incl. probe |
|---|---:|---:|---:|---:|---:|
| DreamCoder-Base | `832/1033 = 80.54%` | `825/1033 = 79.86%` | `+7` tasks / `+0.68pp` | `27/20/805/181` | `3.7763` vs `3.7847` |
| DreamCoder-Instruct | `834/1033 = 80.74%` | `848/1033 = 82.09%` | `-14` tasks / `-1.36pp` | `21/35/813/164` | `3.8472` vs `3.8657` |

### Bucket Results

DreamCoder-Base:

| Oracle bucket | N | Candidate | Baseline | Delta tasks | W/L |
|---|---:|---:|---:|---:|---:|
| `<=8` | `672` | `91.22%` | `90.18%` | `+7` | `18/11` |
| `9-12` | `195` | `78.46%` | `80.51%` | `-4` | `4/8` |
| `13-16` | `80` | `53.75%` | `48.75%` | `+4` | `4/0` |
| `17-24` | `59` | `32.20%` | `32.20%` | `0` | `1/1` |
| `25+` | `27` | `14.81%` | `14.81%` | `0` | `0/0` |

DreamCoder-Instruct:

| Oracle bucket | N | Candidate | Baseline | Delta tasks | W/L |
|---|---:|---:|---:|---:|---:|
| `<=8` | `672` | `88.24%` | `89.58%` | `-9` | `12/21` |
| `9-12` | `195` | `86.15%` | `86.67%` | `-1` | `6/7` |
| `13-16` | `80` | `60.00%` | `66.25%` | `-5` | `0/5` |
| `17-24` | `59` | `30.51%` | `32.20%` | `-1` | `1/2` |
| `25+` | `27` | `25.93%` | `18.52%` | `+2` | `2/0` |

### Repair Diagnostics

| Backbone | Repair triggers | True-long precision among triggers | Trigger source histogram |
|---|---:|---:|---|
| DreamCoder-Base | `115/1033 = 11.13%` | `7.83%` | `official_bounded_repair=66`, `official_long_suspicion=20`, `official_mid_rescue=29` |
| DreamCoder-Instruct | `126/1033 = 12.20%` | `7.94%` | `official_bounded_repair=37`, `official_long_suspicion=27`, `official_mid_rescue=62` |

Canvas/backend sanity:

- Candidate canvas: `bos_prefix_masks_suffix_eos` for all `1033` rows in both runs.
- Candidate backend: `dreamcoder_native_diffusion_generate_fixed_canvas` for all `1033` rows in both runs.
- Verifier schema note: Base has `1024` rows with tier1/2/3 verifier keys and `9` compile-only failures; Instruct has `992` rows with tier1/2/3 verifier keys and `41` compile-only failures. These compile-only rows have `passed=False`, so they are verifier outcomes rather than missing successful verification.

## Reviewer Interpretation

DreamCoder-Base shows a small local same-backbone positive result: `+7` tasks over the official-canvas cal_lite baseline, with no runtime cost increase. This is useful evidence, but it is not yet a strong paper claim because the net gain is small, the pairwise loss count is nonzero (`20` losses), and true-long buckets remain unchanged.

DreamCoder-Instruct is negative transfer evidence: the same policy loses `14` tasks against its own local baseline. It should not be combined with Base to claim cross-backbone improvement.

Literature positioning must remain separated:

- Base candidate `80.54%` is above the CAL DreamCoder-Base anchors (`70.2` average, `76.2` best shown), but this is not protocol-matched until prompt/canvas/decoding budget and local rerun controls are reconciled.
- Base candidate is near but below LR-DLLM DreamCoder-7B `81.6`.
- DreamOn DreamCoder `92.1` is a training-based dynamic-canvas result, not an inference-only local baseline.
- Instruct has no clean matching DreamCoder-Instruct literature row in the current anchor table; only its local same-backbone baseline should drive the claim.
