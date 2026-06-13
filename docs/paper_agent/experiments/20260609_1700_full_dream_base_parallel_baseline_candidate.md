# Full Dream Base Parallel Baseline/Candidate Runs

Timestamp: 2026-06-09 17:00 CST

## Purpose

Launch a protocol-matched local same-backbone pair for `Dream-org/Dream-v0-Base-7B` after its 2-sample LCAL official bounded-repair smoke passed. This is the first Dream-7B claim-relevant local comparison stage, but no literature claim should be made until both full runs finish and pairwise/bucket/runtime analysis is complete.

## Comparison Context

| Backbone | Local baseline being generated | Candidate being generated | Literature anchors |
|---|---|---|---|
| `Dream-org/Dream-v0-Base-7B` | official-canvas `cal_lite`, `alpha=0.10` | official-canvas `lcal_official_bounded_repair` | LR-DLLM Dream-7B `76.7`; DreamOn Dream-7B `88.6` training-based |

## Baseline Command

Session: `dream_base_baseline_full_20260609_1700`

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path /tmp/dream_base_repo_probe_20260609 --mask-length-source cal_lite --length-alpha 0.10 --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed" logs/paper_agent/20260609_1700_full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed.log
```

## Candidate Command

Session: `dream_base_candidate_full_20260609_1700`

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path /tmp/dream_base_repo_probe_20260609 --mask-length-source lcal_official_bounded_repair --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260609_1700_full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed.log
```

## Environment

- GPUs: baseline on `CUDA_VISIBLE_DEVICES=2`; candidate on `CUDA_VISIBLE_DEVICES=3`.
- Python: `/home/shx/miniconda3/envs/dllm_env/bin/python`.
- Compatibility: `PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages`, `DLLM_DISABLE_FLASH_ATTN=1`.
- Local model: `/tmp/dream_base_repo_probe_20260609`.
- Offline cache: `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609`, `HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205`.

## Success Criteria

- Both runs exit `0`.
- Each `results.jsonl` has `1033` rows and each `summary.json` exists.
- Pairwise comparison is computed from raw rows after both runs complete.
- Bucket metrics, verifier outputs, and runtime fields are present.

## Kill Criteria

Stop and investigate if either run OOMs, produces malformed JSON rows, loses verifier fields, uses the wrong canvas/backend, or shows early systematic reconstruction collapse.

## Result

Completed and analyzed on 2026-06-09 18:27 CST.

- Baseline session: `dream_base_baseline_full_20260609_1700`.
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_170219`.
- Candidate session: `dream_base_candidate_full_20260609_1700`.
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_170219`.
- Final sanity: both `results.jsonl` files have `1033` rows, `0` malformed JSON rows, `1033` common task ids, and `summary.json`.
- Canvas/backend: both runs used `bos_prefix_masks_suffix_eos` and `dreamcoder_native_diffusion_generate_fixed_canvas`.
- Both tmux sessions exited after command completion; final logs record `COMMAND_EXIT_CODE="0"`.

### Same-Backbone Main Result

| Run | Pass | Rate | Delta | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---:|---:|---:|---:|---:|
| Dream-7B official-canvas `cal_lite`, `alpha=0.10` baseline | `802/1033` | `77.64%` | baseline | `3.6494` | n/a |
| Dream-7B LCAL official bounded repair candidate | `803/1033` | `77.73%` | `+1` task / `+0.10pp` | `3.7337` | `28/27/775/203` |

### Bucket Result

| Oracle bucket | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `<=8` | `591/672 = 87.95%` | `594/672 = 88.39%` | `+3` tasks / `+0.45pp` |
| `9-12` | `155/195 = 79.49%` | `151/195 = 77.44%` | `-4` tasks / `-2.05pp` |
| `13-16` | `39/80 = 48.75%` | `38/80 = 47.50%` | `-1` task / `-1.25pp` |
| `17-24` | `14/59 = 23.73%` | `15/59 = 25.42%` | `+1` task / `+1.69pp` |
| `25+` | `3/27 = 11.11%` | `5/27 = 18.52%` | `+2` tasks / `+7.41pp` |

### Pairwise Bucket Diagnostics

| Pairwise class | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| Candidate wins | `21` | `2` | `1` | `2` | `2` |
| Candidate losses | `18` | `6` | `2` | `1` | `0` |
| Tie-pass | `573` | `149` | `37` | `13` | `3` |
| Tie-fail | `60` | `38` | `40` | `43` | `22` |

### Repair Diagnostics

- Candidate triggered repair on `101/1033 = 9.78%` rows.
- Triggered-row pass rate: `60/101 = 59.41%`.
- Triggered true-long precision: `13/101 = 12.87%` when true-long is defined as oracle `>=17`.
- Repair-trigger bucket histogram: `<=8: 75`, `9-12: 8`, `13-16: 5`, `17-24: 7`, `25+: 6`.
- Repair-trigger pairwise role: `14` wins, `16` losses, `46` tie-pass, `25` tie-fail.
- Repair reason histogram: `official_bounded_repair=65`, `official_long_suspicion=17`, `official_mid_rescue=19`, `official_no_trigger=836`, `s3_len_above_official_eval_max=96`.

### Interpretation

This is a near-tie, very small local same-backbone positive result. The candidate improves Dream-7B by only `+1` task over its own `cal_lite` baseline, with almost balanced pairwise wins and losses (`28` versus `27`) and a modest runtime increase (`3.7337s` versus `3.6494s` per sample including probe). The long buckets improve by `+3` total tasks across oracle `>=17`, but the `9-12` and `13-16` buckets regress by `-5` total tasks.

The repair trigger remains a weak true-long detector: only `12.87%` of triggered rows are oracle `>=17`, and most triggered rows are actually in `<=8`. This is useful evidence that Dream-7B benefits slightly from the current bounded-repair policy, but it is not a strong claim and should not be used as a standalone SOTA statement.

Literature positioning: the candidate `77.73%` is above the LR-DLLM Dream-7B single-line anchor `76.7`, but that is a literature anchor rather than a protocol-matched local rerun. DreamOn Dream-7B `88.6` is training-based and remains much higher. Therefore this result is best described as local protocol-matched near-tie/slight positive evidence, not external SOTA.
