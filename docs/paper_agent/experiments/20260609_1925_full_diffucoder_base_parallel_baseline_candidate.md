# Full DiffuCoder Base Parallel Baseline/Candidate Runs

Timestamp: 2026-06-09 19:25 CST

## Purpose

Launch a protocol-matched local same-backbone pair for `apple/DiffuCoder-7B-Base` after its download/API probe and 2-sample LCAL official bounded-repair smoke passed. This is the first DiffuCoder claim-relevant local comparison stage. Literature anchors must remain separate until both full runs finish and pairwise/bucket/runtime analysis is complete.

## Comparison Context

| Backbone | Local baseline being generated | Candidate being generated | Literature anchors |
|---|---|---|---|
| `apple/DiffuCoder-7B-Base` | official-canvas `cal_lite`, `alpha=0.10` | official-canvas `lcal_official_bounded_repair` | CAL DiffuCoder-Base `68.0` avg / `74.8` best shown; DreamOn DiffuCoder-7B `92.2` training-based |

## Baseline Command

Session: `diffucoder_base_baseline_full_20260609_1925`

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_diffucoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path /tmp/diffucoder_base_repo_probe_20260609 --mask-length-source cal_lite --length-alpha 0.10 --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_diffucoder_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed" logs/paper_agent/20260609_1925_full_diffucoder_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed.log
```

## Candidate Command

Session: `diffucoder_base_candidate_full_20260609_1925`

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_diffucoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path /tmp/diffucoder_base_repo_probe_20260609 --mask-length-source lcal_official_bounded_repair --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_diffucoder_base_lcal_official_bounded_repair_gpu3_unsandboxed --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260609_1925_full_diffucoder_base_lcal_official_bounded_repair_gpu3_unsandboxed.log
```

## Environment

- GPUs: baseline on `CUDA_VISIBLE_DEVICES=2`; candidate on `CUDA_VISIBLE_DEVICES=3`.
- Python: `/home/shx/miniconda3/envs/dllm_env/bin/python`.
- Compatibility: `PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages`, `DLLM_DISABLE_FLASH_ATTN=1`.
- Local model: `/tmp/diffucoder_base_repo_probe_20260609`.
- Offline cache: `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules_diffucoder_20260609`, `HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205`.
- Proxy environment available for future HF access: `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` point to `127.0.0.1:7890`.

## Success Criteria

- Both runs exit `0`.
- Each `results.jsonl` has `1033` rows and each `summary.json` exists.
- Pairwise comparison is computed from raw rows after both runs complete.
- Bucket metrics, verifier outputs, backend/canvas, and runtime fields are present.

## Kill Criteria

Stop and investigate if either run OOMs, produces malformed JSON rows, loses verifier fields, uses the wrong canvas/backend, or shows early systematic reconstruction collapse.

## Current Status

Completed and analyzed on 2026-06-09 20:50 CST.

- Baseline session: `diffucoder_base_baseline_full_20260609_1925`, exited after command completion.
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_192508`.
- Candidate session: `diffucoder_base_candidate_full_20260609_1925`, exited after command completion.
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_192533`.
- Final sanity: both `results.jsonl` files have `1033` rows, `0` malformed JSON rows, `1033` common task ids, and `summary.json`.
- Canvas/backend: both runs used `bos_prefix_masks_suffix_eos` and `dreamcoder_native_diffusion_generate_fixed_canvas`.
- Both logs record `COMMAND_EXIT_CODE="0"`.

### Same-Backbone Main Result

| Run | Pass | Rate | Delta | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---:|---:|---:|---:|---:|
| DiffuCoder-Base official-canvas `cal_lite`, `alpha=0.10` baseline | `838/1033` | `81.12%` | baseline | `3.6562` | n/a |
| DiffuCoder-Base LCAL official bounded repair candidate | `839/1033` | `81.22%` | `+1` task / `+0.10pp` | `3.7538` | `25/24/814/170` |

### Bucket Result

| Oracle bucket | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `<=8` | `603/672 = 89.73%` | `608/672 = 90.48%` | `+5` tasks / `+0.74pp` |
| `9-12` | `160/195 = 82.05%` | `162/195 = 83.08%` | `+2` tasks / `+1.03pp` |
| `13-16` | `45/80 = 56.25%` | `41/80 = 51.25%` | `-4` tasks / `-5.00pp` |
| `17-24` | `23/59 = 38.98%` | `22/59 = 37.29%` | `-1` task / `-1.69pp` |
| `25+` | `7/27 = 25.93%` | `6/27 = 22.22%` | `-1` task / `-3.70pp` |

### Pairwise Bucket Diagnostics

| Pairwise class | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| Candidate wins | `18` | `5` | `1` | `1` | `0` |
| Candidate losses | `13` | `3` | `5` | `2` | `1` |
| Tie-pass | `590` | `157` | `40` | `21` | `6` |
| Tie-fail | `51` | `30` | `34` | `35` | `20` |

### Repair Diagnostics

- Candidate triggered repair on `113/1033 = 10.94%` rows.
- Triggered-row pass rate: `86/113 = 76.11%`.
- Triggered true-long precision: `9/113 = 7.96%` when true-long is oracle `>=17`.
- Repair-trigger bucket histogram: `<=8: 89`, `9-12: 12`, `13-16: 3`, `17-24: 7`, `25+: 2`.
- Repair-trigger pairwise role: `13` wins, `14` losses, `73` tie-pass, `13` tie-fail.
- Repair reason histogram: `official_bounded_repair=59`, `official_long_suspicion=25`, `official_mid_rescue=29`, `official_no_trigger=797`, `s3_len_above_official_eval_max=123`.

### Interpretation

This is another near-tie, very small local same-backbone positive result. The candidate improves DiffuCoder-Base by only `+1` task over its own `cal_lite` baseline, with nearly balanced pairwise wins/losses (`25` versus `24`) and a modest runtime increase (`3.7538s` versus `3.6562s` per sample including probe).

Unlike Dream-7B, the candidate's net gains are in short and short-medium buckets (`<=8` and `9-12`, `+7` total tasks), while `13-16` and long buckets regress by `-6` total tasks. The repair trigger is still not a precise true-long detector: only `7.96%` of triggered rows are oracle `>=17`, and most triggered rows are in `<=8`.

Literature positioning: the local baseline and candidate are above CAL's DiffuCoder-Base anchors (`68.0` average, `74.8` best shown), but this is not a protocol-matched external rerun. DreamOn DiffuCoder-7B `92.2` is training-based and remains much higher. This result supports a strong backbone baseline observation for DiffuCoder, but not a strong bounded-repair improvement claim.
