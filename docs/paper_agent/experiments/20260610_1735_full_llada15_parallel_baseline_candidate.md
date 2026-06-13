# Full LLaDA-1.5 Parallel Baseline/Candidate Runs

Timestamp: 2026-06-10 17:35 CST

## Purpose

Launch the first protocol-matched local same-backbone full pair for `GSAI-ML/LLaDA-1.5`. This follows the completed metadata/API/local-weight probe and the 2-sample smoke pair.

## Comparison Context

| Backbone | Local baseline being generated | Candidate being generated | Literature anchor |
|---|---|---|---|
| `GSAI-ML/LLaDA-1.5` | `cal_lite` LCAS-v3b, `alpha=0.06` | LCAL official bounded repair | LR-DLLM LLaDA-1.5 single-line `68.9`; LR-DLLM baseline single-line `48.8` |

Literature anchors are not protocol-matched local comparisons. This run creates the local comparison needed before making any LLaDA-1.5 claim.

## Pre-Run Gates

- Local model path `/tmp/llada15_probe_20260609` exists.
- Six safetensors shards match expected byte sizes, total `16,031,197,144` bytes.
- LLaDA-1.5 API probe passed: `LLaDAModelLM`, `model_type=llada`, config `mask_token_id=126336`.
- Candidate smoke: `2/2` pass, exit `0`.
- Baseline smoke: `2/2` pass, exit `0`.
- Smoke pairwise: `0` wins, `0` losses, `2` tie-pass, `0` tie-fail.
- User approved running on GPUs `2/3` despite other jobs because memory fits.

## Commands

Baseline on GPU 2:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "HF_MODULES_CACHE=/tmp/hf_modules_llada15_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_cal_lite_lcas_v3.py --model-path /tmp/llada15_probe_20260609 --probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24 --length-alpha 0.06 --lcas-policy lcas_v3b --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_llada15_cal_lite_lcas_v3b_gpu2_shared" logs/paper_agent/20260610_1735_full_llada15_cal_lite_lcas_v3b_gpu2_shared.log
```

Candidate on GPU 3:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "HF_MODULES_CACHE=/tmp/hf_modules_llada15_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path /tmp/llada15_probe_20260609 --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_llada15_lcal_official_bounded_repair_gpu3_shared --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260610_1735_full_llada15_lcal_official_bounded_repair_gpu3_shared.log
```

## Success Criteria

- Both logs record `COMMAND_EXIT_CODE="0"`.
- Each output dir contains `summary.json` and `1033` valid rows in `results.jsonl`.
- Pairwise comparison has `1033` common task ids.
- Bucket pass rates, selected-length metrics, stopping metadata, and runtime fields are present.

## Kill Criteria

Stop and investigate if either run OOMs, fails verifier execution, emits malformed rows, uses the wrong model path, or shows systematic collapse in early rows.

## Final Status

Launched on 2026-06-10 17:27 CST. Both runs completed on 2026-06-10.

- Baseline tmux session: `llada15_baseline_full_20260610_1735`.
- Candidate tmux session: `llada15_candidate_full_20260610_1735`.
- Baseline output directory observed in log: `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_cal_lite_lcas_v3b_gpu2_shared_20260610_172705`.
- Candidate output directory observed in log: `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_lcal_official_bounded_repair_gpu3_shared_20260610_172720`.
- Early health check: both runs loaded 6 checkpoint shards, used cached `HumanEval-SingleLineInfilling`, loaded `1033` tasks, and started decoding rows.
- Early GPU memory: GPU2/GPU3 total memory use was about `25.8GB` and `25.9GB` after both runs loaded, leaving ample headroom on A6000.

Completion checks:

- Baseline log ended with `COMMAND_EXIT_CODE="0"` at `2026-06-10 19:01:10+08:00`.
- Candidate log ended with `COMMAND_EXIT_CODE="0"` at `2026-06-10 19:23:03+08:00`.
- Both `results.jsonl` files contain `1033` valid JSON rows.
- Both output directories contain `summary.json`.
- Pairwise analysis output: `analysis_outputs/llada15_full_pair_20260610_1923`.

## Result

| Run | Pass | Rate | Avg sec/sample incl. probe |
|---|---:|---:|---:|
| `cal_lite` LCAS-v3b local baseline | `817/1033` | `79.09%` | `5.4224` |
| LCAL official bounded repair candidate | `818/1033` | `79.19%` | `6.6453` |

Pairwise against the local baseline: `18` wins, `17` losses, `800` tie-pass, and `198` tie-fail. Net delta is `+1` task / `+0.10pp`.

Bucket summary by oracle length:

| Bucket | Baseline | Candidate | Pairwise net |
|---|---:|---:|---:|
| `<=8` | `91.47%` | `90.47%` | `-6` |
| `9-12` | `83.19%` | `83.19%` | `0` |
| `13-16` | `64.44%` | `67.78%` | `+3` |
| `17-24` | `18.29%` | `21.95%` | `+3` |
| `25+` | `12.90%` | `16.13%` | `+1` |

Repair diagnostics:

- Official repair triggered on `110/1033 = 10.65%` rows.
- Triggered-row true-long precision is `12/110 = 10.91%`.
- Triggered rows are mostly short: `82/110` are oracle `<=8`.
- `official_long_suspicion` triggered `33` rows, of which `29` are oracle `<=8`.
- `official_mid_rescue` triggered `26` rows, with true-long precision `3.85%`.

## Interpretation

This is a near-tie/slight local positive result, not a strong claim upgrade. The candidate recovers a few medium/long tasks, but the gain is almost fully offset by short-bucket losses and higher runtime. The result reinforces the current paper-agent diagnosis: the existing official-CAL repair family is too imprecise as a true-long detector, even when it produces a small net positive on this backbone.

The literature comparison remains anchor-only: the local candidate is above LR-DLLM's LLaDA-1.5 single-line `68.9`, but that is not a protocol-matched local comparison.
