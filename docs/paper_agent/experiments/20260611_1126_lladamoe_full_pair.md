# LLaDA-MoE Full Pair Action Brief

Timestamp: 2026-06-11 11:26 CST

## Action

Run the full local same-backbone pair for `inclusionAI/LLaDA-MoE-7B-A1B-Base`.

## Reviewer Motivation

This run fills the LLaDA-MoE row in the local protocol-matched matrix. It is required because LR-DLLM reports a LLaDA-MoE result, but that paper number is only a literature anchor unless the local baseline and candidate are evaluated under the same runner, dataset, verifier, and hardware protocol.

## Hypothesis

Based on the completed backbone matrix, LCAL official bounded repair is expected to be a near-tie or small backbone-dependent change, not a guaranteed improvement. The main value of this run is completing the protocol-matched row.

## Smoke Gate Evidence

| Run | Output | Rows | Pass rate | Avg sec/sample incl. probe |
|---|---|---:|---:|---:|
| baseline smoke | `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112427` | 2 | 100.00% | 10.1910 |
| candidate smoke | `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112452` | 2 | 100.00% | 11.7759 |

## Environment

- Repo: `/home/shx/projects/dllm_infilling/git_workspace`
- Model path: `/tmp/lladamoe_probe_20260610`
- Python entrypoint: `/home/shx/miniconda3/envs/dllm_env/bin/python`
- Compatibility path: `/tmp/no_flash_attn:/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages`
- Offline mode: `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`
- GPUs: baseline on `2`, candidate on `3`
- Logs:
  - `logs/paper_agent/20260611_1126_full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared.log`
  - `logs/paper_agent/20260611_1126_full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared.log`
- Output root: `/home/shx/projects/dllm_infilling/outputs_clean`

## Exact Commands

Baseline:

```bash
tmux new-session -d -s lladamoe_full_base_20260611_1126 -c /home/shx/projects/dllm_infilling/git_workspace "script -q -e -c \"DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/tmp/no_flash_attn:/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_lladamoe_20260610 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_cal_lite_lcas_v3.py --model-path /tmp/lladamoe_probe_20260610 --probe-lengths 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24 --length-alpha 0.06 --lcas-policy lcas_v3b --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared\" logs/paper_agent/20260611_1126_full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared.log"
```

Candidate:

```bash
tmux new-session -d -s lladamoe_full_candidate_20260611_1126 -c /home/shx/projects/dllm_infilling/git_workspace "script -q -e -c \"DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/tmp/no_flash_attn:/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_lladamoe_20260610 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path /tmp/lladamoe_probe_20260610 --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8\" logs/paper_agent/20260611_1126_full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared.log"
```

## Success Criteria

Both runs must exit `0`, produce `1033` rows, write valid summaries, and support pairwise/bucket/runtime analysis.

## Kill Criteria

Stop and investigate on import failure, OOM, verifier failure, malformed JSON, missing summary, or severe pass-rate collapse relative to the smoke gate.

## Final Result

Completed at 2026-06-11 14:30 CST.

Verification:

- Baseline log: `logs/paper_agent/20260611_1126_full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared.log`, ended with `COMMAND_EXIT_CODE="0"`.
- Candidate log: `logs/paper_agent/20260611_1126_full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared.log`, ended with `COMMAND_EXIT_CODE="0"`.
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719`.
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740`.
- Pairwise analysis: `analysis_outputs/lladamoe_full_pair_20260611_1438`.
- Both rows files have `1033` valid rows and `0` malformed rows; both summaries exist.

| Run | Pass | Rate | Avg sec/sample incl. probe |
|---|---:|---:|---:|
| `cal_lite` LCAS-v3b baseline | `777/1033` | `75.22%` | `8.7025` |
| LCAL official bounded repair candidate | `801/1033` | `77.54%` | `10.6107` |

Pairwise against local baseline: `31` wins, `7` losses, `770` tie-pass, and `225` tie-fail. Net delta is `+24` tasks / `+2.32pp`.

Bucket pairwise by oracle length:

| Oracle bucket | Count | Wins | Losses | Net | Baseline pass | Candidate pass |
|---|---:|---:|---:|---:|---:|---:|
| `<=8` | `598` | `16` | `7` | `+9` | `88.46%` | `89.97%` |
| `9-12` | `232` | `5` | `0` | `+5` | `77.59%` | `79.74%` |
| `13-16` | `90` | `4` | `0` | `+4` | `57.78%` | `62.22%` |
| `17-24` | `82` | `6` | `0` | `+6` | `14.63%` | `21.95%` |
| `25+` | `31` | `0` | `0` | `0` | `12.90%` | `12.90%` |

Candidate source histogram:

| Source | Count |
|---|---:|
| `base` | `835` |
| `strong_correction` | `92` |
| `official_bounded_repair` | `66` |
| `official_mid_rescue` | `23` |
| `official_long_suspicion` | `15` |
| `weak_correction` | `2` |

Repair diagnostics:

- `official_repair_trigger_count = 104`, trigger rate `10.07%`.
- `official_long_suspicion_trigger_count = 15`, trigger rate `1.45%`.
- `official_mid_rescue_trigger_count = 23`, trigger rate `2.23%`.
- `official_repair_true_long_precision = 11.54%`.
- `official_long_suspicion_true_long_precision = 40.00%`.
- `official_mid_rescue_true_long_precision = 13.04%`.
- `under_select_rate_17plus = 91.15%`.

Interpretation: LLaDA-MoE is the strongest current local transfer result. It improves every oracle bucket except `25+`, where it ties, and the aggregate gain is much larger than the near-ties on Dream-7B, DiffuCoder-Base, and LLaDA-1.5. The cost is nontrivial: candidate runtime is about `+1.91s` per sample including probe. This supports a local same-backbone improvement claim for LLaDA-MoE, but not an external SOTA claim.

Literature positioning: the local candidate is above LR-DLLM's LLaDA-MoE single-line `71.3`, but that is a literature anchor rather than a protocol-matched local comparison.
