# Paper Agent Pause Checkpoint

Timestamp: 2026-06-14 02:40 CST

## Current Branch

`paper-agent-overnight`

## Current Phase

The literature-backbone local same-backbone matrix has completed through `inclusionAI/LLaDA-MoE-7B-A1B-Base`. Trace-long-rescue v1 Task 1/2/3/4/5 completed with negative Route 1/2/3 evidence. CPU-only `trace_feature_audit_v2` completed with `diagnostic_only`; after user approval to continue, Route 2 trace-gated long-rescue full follow-up runs completed. The latest GPU3-only precision `len32` run is `801/1033 = 77.54%`, pairwise `6/0/795/232`. The full-run signal is modestly positive and low-risk, but true-long recovery remains weak, especially oracle `25+`.

## Completed Items

- Initialized bilingual paper-agent docs under `docs/paper_agent/`.
- Built and tested a compact evidence snapshot builder.
- Generated `docs/paper_agent/evidence_snapshot.md` and `.json` from local A6000 raw outputs.
- Implemented and tested `analysis/analyze_probe_curve_long_signals.py`.
- Generated bilingual probe-curve audit docs and JSON.
- Updated plan history through `v3`, including the GPU `2,3` allocation constraint.
- Verified the current milestone with focused unit tests, compile checks, audit regeneration, JSON assertions, and `git diff --check`.
- Entered pause flow and updated the dashboard, logs, evidence snapshot, and this checkpoint.
- Resumed from the stale checkpoint in low-token mode and reconciled the uncommitted strict-split diagnostic files.
- Implemented and tested `analysis/analyze_probe_curve_split_score.py`.
- Generated bilingual strict-split probe-score audit docs and JSON.
- Updated the dashboard, evidence snapshot, experiment results, open questions, activity ledger, and overnight logs with the negative strict-split result.
- Reconciled the checkpoint workflow table with the dashboard/results/log evidence: earlier probe-curve audit verification was completed, while the strict-split diagnostic has focused 3-test verification and still needs final milestone-level verification before commit/push.
- Completed fresh focused verification for the strict-split diagnostic on 2026-06-04: unit test, py_compile, audit regeneration, JSON assertions, and `git diff --check` passed.
- Wrote an Action Brief for the LLaDA-Instruct `midcons` full run to `docs/paper_agent/current_action.md` and `docs/paper_agent/experiments/20260604_2022_llada_instruct_midcons_full.md`.
- Started the LLaDA-Instruct full run. The first launch attempted direct `huggingface.co` access and was interrupted before samples ran; the active run uses `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`.
- Confirmed the active run loaded model checkpoint shards, loaded `1033` tasks, and created output directory `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`.
- Completed the LLaDA-Instruct run: `815/1033 = 78.90%`, below the historical same-backbone LCAS-v3 baseline `817/1033 = 79.09%`; pairwise `17` wins and `19` losses.
- Wrote `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md` with CAL, LR-DLLM, DreamOn backbone anchors and the next rerun plan.
- Implemented the DreamCoder official-canvas LCAL/S3 + official-CAL bounded-repair adapter in `clean_scripts/run_dreamcoder_official_infilling.py`; `py_compile` passed.
- Wrote the DreamCoder Base smoke action brief to `docs/paper_agent/current_action.md` and `docs/paper_agent/experiments/20260609_1203_smoke_dreamcoder_base_lcal_official_bounded_repair.md`.
- Debugged the smoke environment: copied HF module/dataset caches to `/tmp`, confirmed dataset loading works from `/tmp`, and identified the remaining blocker as sandbox-disabled CUDA plus sandbox-disabled verifier multiprocessing.
- After user approval, ran sandbox-outside DreamCoder Base smoke successfully: `2/2` pass, `2` tie-pass versus its same-backbone baseline over the first two tasks, output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_122358`.
- Ran sandbox-outside DreamCoder Instruct smoke successfully: `2/2` pass, `2` tie-pass versus its same-backbone baseline over the first two tasks, output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_122945`.
- Launched and completed full DreamCoder Base/Instruct parallel runs in tmux after smoke success.
- Recomputed DreamCoder same-backbone metrics from raw `results.jsonl`: Base candidate `832/1033 = 80.54%` versus baseline `825/1033 = 79.86%`, pairwise `27` wins and `20` losses; Instruct candidate `834/1033 = 80.74%` versus baseline `848/1033 = 82.09%`, pairwise `21` wins and `35` losses.
- Updated DreamCoder full-run experiment brief, experiment results, dashboards, activity ledgers, and this checkpoint with local baseline and literature-anchor separation.
- Downloaded/probed `Dream-org/Dream-v0-Base-7B` via local Git/LFS checkout under `/tmp`, ran a valid 2-sample smoke, launched a full same-backbone pair, monitored both tmux sessions to completion, and recomputed Dream-7B same-backbone metrics from raw `results.jsonl`: candidate `803/1033 = 77.73%` versus local cal_lite baseline `802/1033 = 77.64%`, pairwise `28` wins and `27` losses.
- Downloaded/probed `apple/DiffuCoder-7B-Base` via proxy/mirror Git/LFS path under `/tmp`, ran a valid 2-sample smoke, launched a full same-backbone pair, monitored both tmux sessions to completion, and recomputed DiffuCoder-Base same-backbone metrics from raw `results.jsonl`: candidate `839/1033 = 81.22%` versus local cal_lite baseline `838/1033 = 81.12%`, pairwise `25` wins and `24` losses.
- Wrote the next action brief for `GSAI-ML/LLaDA-1.5` download/API probe to `docs/paper_agent/current_action.md` and `docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`.
- Completed the `GSAI-ML/LLaDA-1.5` metadata/API/local-weight probe through direct HuggingFace with the configured proxy. All six safetensors shards are present under `/tmp/llada15_probe_20260609` and byte-size checked against HF API/index. Local API/config/tokenizer/model-class inspection passed; CPU/local checkpoint loading passed. GPU smoke was not launched because GPUs `2/3` were occupied and sandboxed Python could not see CUDA.
- Completed the `GSAI-ML/LLaDA-1.5` 2-sample smoke pair and full local same-backbone pair on shared GPUs `2/3`. Baseline `cal_lite` LCAS-v3b is `817/1033 = 79.09%`; LCAL official bounded-repair candidate is `818/1033 = 79.19%`, pairwise `18` wins and `17` losses. This is a near-tie/slight local positive, not a strong claim upgrade.
- Completed `inclusionAI/LLaDA-MoE-7B-A1B-Base` proxy/mirror download recovery under `/tmp/lladamoe_probe_20260610`; all three safetensors shards are present and `safe_open` validation passed.
- Debugged LLaDA-MoE environment compatibility: `dllm_env` Transformers was too old for `modeling_rope_utils`, while the `dllm_env` `flash_attn_2_cuda` extension failed on `GLIBC_2.32`. The active runs therefore use `PYTHONPATH=/tmp/no_flash_attn:/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages` with `/home/shx/miniconda3/envs/dllm_env/bin/python`.
- Completed LLaDA-MoE 2-sample smoke gate. Baseline smoke output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112427` is `2/2`; candidate smoke output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112452` is `2/2`; both logs ended with `COMMAND_EXIT_CODE="0"`.
- Completed the full LLaDA-MoE local same-backbone pair on GPUs `2/3`. Baseline `cal_lite` LCAS-v3b is `777/1033 = 75.22%`; LCAL official bounded-repair candidate is `801/1033 = 77.54%`, pairwise `31` wins, `7` losses, `770` tie-pass, and `225` tie-fail. Both logs ended with `COMMAND_EXIT_CODE="0"` and both outputs have `1033` valid rows plus `summary.json`.
- Completed trace-long-rescue Task 1/2/3 under the user override to avoid Superpowers skills, subagents, reviewer discovery, and Goal tools. The existing full-trace action brief and current action passed markdown diff hygiene.
- Implemented `analysis/trace_long_rescue_features.py` and `tests/test_trace_long_rescue_features.py`; verified the initial Task 2 test set with `Ran 3 tests` / `OK`.
- Implemented `analysis/analyze_trace_long_rescue_routes.py` and `analysis/print_trace_long_rescue_report.py`; verified the expanded Task 3 test set with `Ran 6 tests` / `OK`, py_compile for all three trace-long-rescue analysis scripts, and `git diff --check`.
- Completed serial Task 4 full trace collection. Previous local method trace run on GPU `2` verified at `769/1033 = 74.44%` with `35257` trace rows; current `midcons` trace run on GPU `3` verified at `795/1033 = 76.96%` with `35768` trace rows.
- Completed Task 5 offline Route 1/2/3 analysis on both full trace outputs. All routes failed Gate A/B; Route 1/2 had `0` triggers and Route 3 stopped without route signal, so no policy runner is justified.
- Implemented and ran CPU-only `trace_feature_audit_v2` under `superpowers:executing-plans`. Final output is `analysis_outputs/trace_feature_audit_v2_20260613_204721`; decision is `diagnostic_only`. Previous source has policy-level candidates, but current `midcons` source is diagnostic-only, so no direct full GPU policy runner is justified.
- Completed two user-approved Route 2 trace-gated long-rescue full follow-up runs using `clean_scripts/run_route2_trace_rescue.py`. Broad plateau output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958` is `801/1033 = 77.54%`, pairwise `7/1/794/231`; precision top1/conf output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958` is `800/1033 = 77.44%`, pairwise `5/0/795/233`.
- Completed the user-requested GPU3-only Route2 precision `len32` full run. The earlier GPU1 partial run was interrupted at about `405/1033` and is excluded. Clean output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516` is `801/1033 = 77.54%`, pairwise `6/0/795/232`; log `logs/paper_agent/20260614_full_route2_precision_len32_gpu3.log` ended with `COMMAND_EXIT_CODE=0`.

## Current Central Claim

Inference-time length control for DLLM code infilling can safely recover medium-length under-selection by separating medium rescue from true-long detection; however, true-long infilling remains dominated by length underestimation and likely requires a stronger length-modeling signal than the current official-CAL gate family.

## Current Experiment Plan Version

`v3`: probe-curve-first long-length modeling with future GPU experiments restricted to `CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false` unless the user changes the allocation, now extended by the literature-backbone rerun matrix and Route 2 trace-gated follow-up evidence.

## Latest Evidence And Results

- A6000 control: `787/1033 = 76.19%`.
- A6000 `midcons`: `795/1033 = 76.96%`, same-hardware `+8` wins and `0` losses.
- Long buckets unchanged: `17-24 = 20.73%`, `25+ = 16.13%`.
- Long-underestimate sweep: `16776` rules, `0` strict viable.
- Probe-curve audit: `1033/1033` rows have probe-curve features; `0/1033` rows have stopping traces.
- Probe-curve threshold sweep: `4106` single-feature thresholds, `0` strict viable.
- Best single-feature threshold: `long_score_max <= 0.229253`, `63.04%` true-long precision, `31.87%` failed-long recall, `8.70%` short-risk, `2.17%` current-pass risk.
- Interpretation: single-feature probe thresholds are informative but not GPU-safe under the current `5%` short-risk gate.
- Strict-split probe-score audit: `5` deterministic SHA256 task-id folds, `24` features, aggregate held-out `63` triggers, `47.62%` true-long precision, `32.97%` failed-long recall, `22.22%` short-risk, `7.94%` current-pass risk.
- strict_heldout_pass: `False`.
- Interpretation: the simple multivariate probe score is not GPU-safe. GPU work remains blocked until a safer offline signal or trace-enabled smoke rationale exists.
- LLaDA-Instruct cross-model validation: candidate `815/1033 = 78.90%`; historical same-backbone baseline `817/1033 = 79.09%`; interpretation is negative transfer evidence.
- DreamCoder Base smoke: valid sandbox-outside smoke passed, `2/2`, avg total sec including probe `3.1439`, pairwise against same-backbone baseline over first two tasks `0` wins / `0` losses / `2` tie-pass.
- DreamCoder Instruct smoke: valid sandbox-outside smoke passed, `2/2`, avg total sec including probe `3.0555`, pairwise against same-backbone baseline over first two tasks `0` wins / `0` losses / `2` tie-pass.
- DreamCoder Base full run: `832/1033 = 80.54%`; local same-backbone official-canvas cal_lite baseline `825/1033 = 79.86%`; pairwise `27` wins, `20` losses, `805` tie-pass, `181` tie-fail; avg total sec including probe `3.7763` versus `3.7847`. Interpretation: small local positive, not yet a strong claim.
- DreamCoder Instruct full run: `834/1033 = 80.74%`; local same-backbone official-canvas cal_lite baseline `848/1033 = 82.09%`; pairwise `21` wins, `35` losses, `813` tie-pass, `164` tie-fail; avg total sec including probe `3.8472` versus `3.8657`. Interpretation: negative transfer evidence.
- DreamCoder literature anchors: Base is above CAL DreamCoder-Base anchors (`70.2` average, `76.2` best shown) and below LR-DLLM DreamCoder-7B `81.6`; DreamOn DreamCoder `92.1` is training-based. These are anchors, not protocol-matched claims.
- Dream-7B full local pair: candidate `803/1033 = 77.73%`; local same-backbone official-canvas cal_lite baseline `802/1033 = 77.64%`; pairwise `28` wins, `27` losses, `775` tie-pass, `203` tie-fail; avg total sec including probe `3.7337` versus `3.6494`. Interpretation: near-tie/slight local positive, not a strong claim.
- Dream-7B literature anchors: candidate is above LR-DLLM Dream-7B single-line `76.7`, but DreamOn Dream-7B `88.6` is training-based and much higher. These are anchors, not protocol-matched claims.
- DiffuCoder-Base full local pair: candidate `839/1033 = 81.22%`; local same-backbone official-canvas cal_lite baseline `838/1033 = 81.12%`; pairwise `25` wins, `24` losses, `814` tie-pass, `170` tie-fail; avg total sec including probe `3.7538` versus `3.6562`. Interpretation: near-tie/slight local positive, not a strong bounded-repair improvement claim.
- DiffuCoder literature anchors: both local DiffuCoder rows are above CAL DiffuCoder-Base anchors (`68.0` average, `74.8` best shown), but DreamOn DiffuCoder-7B `92.2` is training-based and much higher. These are anchors, not protocol-matched claims.
- LLaDA-1.5 probe: local path `/tmp/llada15_probe_20260609`; architecture `LLaDAModelLM`; `model_type=llada`; config `mask_token_id=126336`; tokenizer `<|mdm_mask|>` resolves to `126336` while `tokenizer.mask_token` is `None`; six shards match expected byte sizes with total `16,031,197,144` bytes. Interpretation: runner family is likely compatible, but this is not a GPU/verifier result and gives no pass rate.
- LLaDA-1.5 full local pair: candidate `818/1033 = 79.19%`; local same-backbone `cal_lite` LCAS-v3b baseline `817/1033 = 79.09%`; pairwise `18` wins, `17` losses, `800` tie-pass, `198` tie-fail; avg total sec including probe `6.6453` versus `5.4224`. Interpretation: near-tie/slight local positive, with short-bucket regression and low official-repair true-long precision (`10.91%`).
- LLaDA-MoE full local pair: candidate `801/1033 = 77.54%`; local same-backbone `cal_lite` LCAS-v3b baseline `777/1033 = 75.22%`; pairwise `31` wins, `7` losses, `770` tie-pass, `225` tie-fail; avg total sec including probe `10.6107` versus `8.7025`. Bucket deltas are nonnegative in every oracle bucket: `<=8 +9`, `9-12 +5`, `13-16 +4`, `17-24 +6`, `25+ 0`. Interpretation: strongest current local transfer result, but slower and still not external SOTA.
- Trace feature audit v2: output `analysis_outputs/trace_feature_audit_v2_20260613_204721`; decision `diagnostic_only`; previous source `policy_candidate`; midcons source `diagnostic_only`. Strongest midcons held-out candidate: `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16`, with `18` triggers, `9` failed-long, `2` short-risk, `0` current-pass risk, and `0.500` true-long precision. This was useful diagnostic signal, but not enough by itself for automatic full GPU policy execution.
- Route 2 broad plateau full follow-up: output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`; log `logs/paper_agent/20260613_full_route2_broad_gpu2.log`; `801/1033 = 77.54%` versus `midcons` `795/1033 = 76.96%`; pairwise `7` wins / `1` loss / `794` tie-pass / `231` tie-fail; triggers `73`; trigger true-long precision `53.42%`; avg sec including probe `5.0852`.
- Route 2 precision top1/conf full follow-up: output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`; log `logs/paper_agent/20260613_full_route2_precision_gpu3.log`; `800/1033 = 77.44%` versus `midcons` `795/1033 = 76.96%`; pairwise `5` wins / `0` losses / `795` tie-pass / `233` tie-fail; triggers `57`; trigger true-long precision `61.40%`; avg sec including probe `5.0945`.
- Route 2 precision len32 GPU3-only follow-up: output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`; log `logs/paper_agent/20260614_full_route2_precision_len32_gpu3.log`; `801/1033 = 77.54%` versus `midcons` `795/1033 = 76.96%`; pairwise `6` wins / `0` losses / `795` tie-pass / `232` tie-fail; triggers `57`; trigger true-long precision `61.40%`; avg sec including probe `5.4622`. Bucket net: `<=8 +2`, `9-12 +2`, `13-16 0`, `17-24 +2`, `25+ 0`.
- Interpretation: precision is the cleaner paper-safe incremental result because it has no losses; broad has slightly larger net gain but one short loss. Neither resolves true-long: `17-24` gains only `+1/+2`, and `25+` is unchanged. Among `91` baseline failed-long rows, broad triggers `39` and rescues only `2`; precision triggers `35` and rescues only `1`, so fixed `len=24` rescue quality/length choice remains the key bottleneck.

## Running Or Just-Ended Commands

No trace-long-rescue or Route 2 GPU command is running as of 2026-06-14 02:40 CST. The latest GPU action was the completed GPU3-only Route2 precision len32 follow-up:

- precision len32 log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260614_full_route2_precision_len32_gpu3.log`
- precision len32 output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`
- precision len32 final sanity: log ended with `COMMAND_EXIT_CODE=0`; `results.jsonl` has `1033` valid rows; `summary.json` exists; `step_traces.jsonl` is nonempty; result is `801/1033 = 77.54%`, pairwise `6/0/795/232`.

Previous Route 2 full follow-up runs:

- broad log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260613_full_route2_broad_gpu2.log`
- broad output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`
- broad final sanity: log ended with `COMMAND_EXIT_CODE=0`; `results.jsonl` has `1033` valid rows and `0` malformed rows; `summary.json` exists; `step_traces.jsonl` has `39740` rows.
- precision log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260613_full_route2_precision_gpu3.log`
- precision output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`
- precision final sanity: log ended with `COMMAND_EXIT_CODE=0`; `results.jsonl` has `1033` valid rows and `0` malformed rows; `summary.json` exists; `step_traces.jsonl` has `38872` rows.

The latest CPU-only audit before those GPU follow-ups was `trace_feature_audit_v2`:

- command: `/home/shx/miniconda3/envs/dllm_env/bin/python analysis/trace_feature_audit_v2.py ... --output-dir analysis_outputs/trace_feature_audit_v2_20260613_204721 --folds 5`
- output: `analysis_outputs/trace_feature_audit_v2_20260613_204721`
- result: compact JSON printed `decision=diagnostic_only`, `previous=policy_candidate`, `midcons=diagnostic_only`
- decision at that time: do not automatically start a full GPU policy runner from the audit alone. The later full runs were user-approved follow-up trials.

Completed current `midcons` trace run:

- tmux session: `trace_llada_base_midcons_20260612` exited.
- log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260612_full_trace_llada_base_midcons_gpu3.log`
- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`
- verification: log ended with `COMMAND_EXIT_CODE="0"`; `results.jsonl` has `1033` valid rows; `step_traces.jsonl` has `35768` rows linked to `1033` task ids; `summary.json` exists; pass count is `795/1033 = 76.96%`.

Completed offline route analysis:

- previous analysis: `analysis_outputs/trace_long_rescue_llada_base_prev_20260612_192611`
- midcons analysis: `analysis_outputs/trace_long_rescue_llada_base_midcons_20260612_192611`
- report: `analysis_outputs/trace_long_rescue_report_20260612`
- result: Route 1 and Route 2 trigger `0` rows on both trace sources; Route 3 stops because single-canvas traces plus no Route 1/2 signal do not justify extra multi-canvas cost. Gate A/B both fail for all routes.
- decision: no route-specific GPU policy full run should be launched from this trace batch.

Completed previous local method trace run:

- tmux session: `trace_llada_base_prev_20260612` exited.
- log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260612_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log`
- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`
- verification: log ended with `COMMAND_EXIT_CODE="0"`; `results.jsonl` has `1033` valid rows; `step_traces.jsonl` has `35257` rows linked to `1033` task ids; `summary.json` exists; pass count is `769/1033 = 74.44%`.

Completed local verification commands before the GPU launch:

- `/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_long_rescue_features.py`
- `/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/trace_long_rescue_features.py analysis/analyze_trace_long_rescue_routes.py analysis/print_trace_long_rescue_report.py`
- `git diff --check -- analysis/trace_long_rescue_features.py analysis/analyze_trace_long_rescue_routes.py analysis/print_trace_long_rescue_report.py tests/test_trace_long_rescue_features.py docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md docs/paper_agent/current_action.md`

Completed LLaDA-MoE full local same-backbone pair:

- Baseline log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260611_1126_full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared.log`
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719`
- Candidate log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260611_1126_full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared.log`
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740`
- Pairwise analysis: `/home/shx/projects/dllm_infilling/git_workspace/analysis_outputs/lladamoe_full_pair_20260611_1438`
- Final sanity: both logs ended with `COMMAND_EXIT_CODE="0"`, both rows files have `1033` valid rows and `0` malformed rows, both summaries exist, and the pairwise analysis has `1033` common task ids.
- Result: candidate `801/1033 = 77.54%` versus baseline `777/1033 = 75.22%`, pairwise `31/7/770/225`, avg sec including probe `10.6107` versus `8.7025`.

Historical blocked command:

- Log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1203_smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2.log`
- Partial output dirs:
  - `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_20260609_120429`
  - `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_20260609_120635`
- Status: invalid evidence; do not use for paper comparison.
- Root causes:
  - first attempt: read-only HF dataset cache lock;
  - second attempt: sandbox blocked `multiprocessing.Manager()` listener socket in HumanEval verifier;
  - CUDA probes inside sandbox: `torch.cuda.is_available() == False`.

Required next command after explicit user approval:

```bash
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path Dream-org/Dream-Coder-v0-Base-7B --max-samples 2 --mask-length-source lcal_official_bounded_repair --baseline-results /home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721/results.jsonl --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260609_1212_smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log
```

Completed DreamCoder full commands:

- Base log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1231_full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log`
- Base output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`
- Base final rows: `1033`
- Instruct log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1231_full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed.log`
- Instruct output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`
- Instruct final rows: `1033`
- GPU status after completion: `nvidia-smi` showed no running GPU processes on 2026-06-09 14:28 CST.

Completed Dream-7B full commands:

- Baseline log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1700_full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed.log`
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_170219`
- Baseline final rows: `1033`
- Candidate log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1700_full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed.log`
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_170219`
- Candidate final rows: `1033`
- Final sanity: both rows files have `0` malformed JSON rows, `1033` common task ids, `summary.json`, canvas `bos_prefix_masks_suffix_eos`, and backend `dreamcoder_native_diffusion_generate_fixed_canvas`.

Completed LLaDA-1.5 download/API/local-weight probe:

- Probe doc: `/home/shx/projects/dllm_infilling/git_workspace/docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`
- Local model path: `/tmp/llada15_probe_20260609`
- HF commit observed: `84346fd91ba60252d260022201ad6fc5a3468fb2`
- Network note: direct HuggingFace through proxy worked; `hf-mirror.com` redirected this repo back to HuggingFace and `huggingface_hub` mirror mode failed.
- Shard byte-size check: all six shards match expected sizes; total `16,031,197,144` bytes.
- API note: `LLaDAModelLM`, `model_type=llada`, config `mask_token_id=126336`, tokenizer `<|mdm_mask|>` id `126336`, `tokenizer.mask_token=None`.
- CPU/local load smoke: passed with `torch.bfloat16`; not a GPU smoke.
- GPU status at 2026-06-09 21:53 CST: GPU2 and GPU3 occupied; do not interrupt.

Completed LLaDA-1.5 full local same-backbone pair:

- Baseline log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260610_1735_full_llada15_cal_lite_lcas_v3b_gpu2_shared.log`
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_cal_lite_lcas_v3b_gpu2_shared_20260610_172705`
- Baseline final rows: `1033`
- Candidate log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260610_1735_full_llada15_lcal_official_bounded_repair_gpu3_shared.log`
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_lcal_official_bounded_repair_gpu3_shared_20260610_172720`
- Candidate final rows: `1033`
- Pairwise analysis: `/home/shx/projects/dllm_infilling/git_workspace/analysis_outputs/llada15_full_pair_20260610_1923`
- Final sanity: both logs ended with `COMMAND_EXIT_CODE="0"`, both outputs have `summary.json`, both `results.jsonl` files have `1033` valid rows, and both have `1033` common task ids.
- Result: candidate `818/1033 = 79.19%` versus baseline `817/1033 = 79.09%`, pairwise `18/17/800/198`, avg sec including probe `6.6453` versus `5.4224`.

Completed DiffuCoder-Base full commands:

- Baseline log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1925_full_diffucoder_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed.log`
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_192508`
- Baseline final rows: `1033`
- Candidate log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1925_full_diffucoder_base_lcal_official_bounded_repair_gpu3_unsandboxed.log`
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_192533`
- Candidate final rows: `1033`
- Final sanity: both rows files have `0` malformed JSON rows, `1033` common task ids, `summary.json`, canvas `bos_prefix_masks_suffix_eos`, and backend `dreamcoder_native_diffusion_generate_fixed_canvas`.

Previously completed command:

- tmux session: `llada_instruct_midcons_20260604`
- Log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260604_2022_llada_instruct_midcons_full_hfmirror.log`
- Output dir: `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`
- Final observed progress: `1033` rows in `results.jsonl`; `summary.json` exists.

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path GSAI-ML/LLaDA-8B-Instruct --baseline-results /home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_llada-instruct_resume_20260512_141851/results.jsonl --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23 --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8
```

Just-ended strict-split diagnostic commands:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_split_score.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/analyze_probe_curve_split_score.py
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_split_score.py
/home/shx/miniconda3/envs/dllm_env/bin/python -c "import json; p='docs/paper_agent/probe_curve_split_score_audit.json'; d=json.load(open(p)); a=d['cross_validation']['aggregate_heldout']; assert a['strict_heldout_pass'] is False; assert a['trigger_count']==63; assert round(a['short_risk_rate'], 4)==0.2222; assert round(a['current_pass_risk_rate'], 4)==0.0794; assert round(a['true_long_precision'], 4)==0.4762; assert round(a['failed_long_recall'], 4)==0.3297; print('strict_split_assertions_ok', a)"
git diff --check -- analysis/analyze_probe_curve_split_score.py tests/test_analyze_probe_curve_split_score.py docs/paper_agent/probe_curve_split_score_audit.json docs/paper_agent/probe_curve_split_score_audit.md docs/paper_agent/probe_curve_split_score_audit.zh.md docs/paper_agent/current_action.md docs/paper_agent/pause_checkpoint.current.md docs/paper_agent/activity_ledger.en.md docs/paper_agent/activity_ledger.zh.md
```

Observed output:

- strict-split unit tests: `Ran 3 tests` and `OK`
- audit generation: `strict_heldout_pass=False heldout_triggers=63 short_risk=22.22%`
- JSON assertions: `strict_split_assertions_ok` with `trigger_count=63`, `short_risk_rate=0.2222`, `current_pass_risk_rate=0.0794`, `true_long_precision=0.4762`, `failed_long_recall=0.3297`, and `strict_heldout_pass=False`
- diff hygiene: exit `0`

## Modified Files

Intended paper-agent milestone files:

- `analysis/analyze_probe_curve_long_signals.py`
- `tests/test_analyze_probe_curve_long_signals.py`
- `docs/paper_agent/evidence_snapshot.md`
- `docs/paper_agent/experiment_plan.current.en.md`
- `docs/paper_agent/experiment_plan.current.zh.md`
- `docs/paper_agent/experiment_plan.history.en.md`
- `docs/paper_agent/experiment_plan.history.zh.md`
- `docs/paper_agent/experiment_results.en.md`
- `docs/paper_agent/experiment_results.zh.md`
- `docs/paper_agent/open_questions.en.md`
- `docs/paper_agent/open_questions.zh.md`
- `docs/paper_agent/overnight_log.en.md`
- `docs/paper_agent/overnight_log.zh.md`
- `docs/paper_agent/paper_agent_dashboard.en.md`
- `docs/paper_agent/paper_agent_dashboard.zh.md`
- `docs/paper_agent/probe_curve_signal_audit.json`
- `docs/paper_agent/probe_curve_signal_audit.md`
- `docs/paper_agent/probe_curve_signal_audit.zh.md`
- `docs/paper_agent/research_design.current.en.md`
- `docs/paper_agent/research_design.current.zh.md`
- `docs/paper_agent/pause_checkpoint.current.md`
- `analysis/analyze_probe_curve_split_score.py`
- `analysis/trace_long_rescue_features.py`
- `analysis/analyze_trace_long_rescue_routes.py`
- `analysis/print_trace_long_rescue_report.py`
- `tests/test_analyze_probe_curve_split_score.py`
- `tests/test_trace_long_rescue_features.py`
- `docs/paper_agent/activity_ledger.en.md`
- `docs/paper_agent/activity_ledger.zh.md`
- `docs/paper_agent/probe_curve_split_score_audit.json`
- `docs/paper_agent/probe_curve_split_score_audit.md`
- `docs/paper_agent/probe_curve_split_score_audit.zh.md`
- `docs/superpowers/plans/2026-05-31-probe-curve-learned-diagnostic.md`

User/unrelated dirty files to preserve and not stage:

- `AGENTS.md`
- `AGENTS.zh.md`

## Uncommitted Files At Checkpoint Capture

```text
## paper-agent-overnight...origin/paper-agent-overnight [ahead 1]
 M AGENTS.md
 D AGENTS.zh.md
 M docs/paper_agent/evidence_snapshot.md
 M docs/paper_agent/experiment_results.en.md
 M docs/paper_agent/experiment_results.zh.md
 M docs/paper_agent/open_questions.en.md
 M docs/paper_agent/open_questions.zh.md
 M docs/paper_agent/overnight_log.en.md
 M docs/paper_agent/overnight_log.zh.md
 M docs/paper_agent/paper_agent_dashboard.en.md
 M docs/paper_agent/paper_agent_dashboard.zh.md
?? analysis/analyze_probe_curve_split_score.py
?? docs/paper_agent/activity_ledger.en.md
?? docs/paper_agent/activity_ledger.zh.md
?? docs/paper_agent/probe_curve_split_score_audit.json
?? docs/paper_agent/probe_curve_split_score_audit.md
?? docs/paper_agent/probe_curve_split_score_audit.zh.md
?? docs/superpowers/plans/2026-05-31-probe-curve-learned-diagnostic.md
?? tests/test_analyze_probe_curve_split_score.py
```

Captured older pause-state status for the prior probe-curve audit milestone:

```text
## paper-agent-overnight...origin/paper-agent-overnight
 M AGENTS.md
 D AGENTS.zh.md
 M docs/paper_agent/evidence_snapshot.md
 M docs/paper_agent/experiment_plan.current.en.md
 M docs/paper_agent/experiment_plan.current.zh.md
 M docs/paper_agent/experiment_plan.history.en.md
 M docs/paper_agent/experiment_plan.history.zh.md
 M docs/paper_agent/experiment_results.en.md
 M docs/paper_agent/experiment_results.zh.md
 M docs/paper_agent/open_questions.en.md
 M docs/paper_agent/open_questions.zh.md
 M docs/paper_agent/overnight_log.en.md
 M docs/paper_agent/overnight_log.zh.md
 M docs/paper_agent/paper_agent_dashboard.en.md
 M docs/paper_agent/paper_agent_dashboard.zh.md
 M docs/paper_agent/research_design.current.en.md
 M docs/paper_agent/research_design.current.zh.md
?? analysis/analyze_probe_curve_long_signals.py
?? docs/paper_agent/pause_checkpoint.current.md
?? docs/paper_agent/probe_curve_signal_audit.json
?? docs/paper_agent/probe_curve_signal_audit.md
?? docs/paper_agent/probe_curve_signal_audit.zh.md
?? tests/test_analyze_probe_curve_long_signals.py
```

## Known Risks

- Current improvement is small and heuristic; it is not a CCF-A-level central claim yet.
- Long buckets remain unchanged despite `midcons`.
- Current outputs lack stopping traces, so trajectory diagnostics require a trace-enabled smoke run later.
- Single-feature probe-curve thresholds fail the short-risk gate.
- The first simple strict-split multivariate probe score also fails the short-risk gate: `22.22%` held-out short-risk versus the `5%` gate.
- Independent subagent code review is blocked in this environment; only local diff review plus tests were used.
- Future GPU experiments must use cards `2,3` and must not interrupt other jobs.
- Current IDE sandbox cannot produce valid GPU/verifier results for HumanEval experiments; long GPU/verifier runs must be launched outside the sandbox, as the current DreamCoder full runs were.

## Open Questions

- Can constrained, nonlinear, trace-aware, or cross-run probe scoring reduce short-risk to `<=5%` while retaining at least `10` failed-long triggers? The first simple strict-split linear score failed with `22.22%` held-out short-risk.
- Can trajectory features detect true-long under-selection, or is training-time length regularization required?
- Is the paper best framed as a positive method paper, a diagnostic-plus-method paper, or a rigorous negative result motivating dynamic canvas or length regularization?
- For missing backbones without local baselines, should the first full run be treated only as exploratory until a same-backbone local baseline is generated?
- How should paper-critical raw result files be stored if a future run becomes central?

## Workflow / Skill Status

| Workflow / Skill | Status | Evidence | Output files | Notes |
|---|---|---|---|---|
| gstack `/office-hours` | completed | `research_design.initial.*.md` and `research_design.current.*.md` contain the research-community user, need, and minimum publishable contribution mapping; log entry `2026-05-31 12:39 CST` records paper-agent document initialization | `docs/paper_agent/research_design.initial.en.md`, `docs/paper_agent/research_design.initial.zh.md`, `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | No separate CLI transcript; completion is evidenced by the output research design docs |
| gstack `/plan-ceo-review` | completed | `research_design.current.en.md` includes `CEO-Style Stress Review` covering novelty, importance, reviewer appeal, scope, central claim, weakest assumption, and CCF-A realism | `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | Current conclusion: credible foothold, not yet a CCF-A claim |
| gstack `/plan-eng-review` | completed | `experiment_plan.current.en.md` includes engineering review, datasets, baselines, metrics, compute budget, reproducibility, failure modes, and kill criteria | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | Current version is `v3` |
| Superpowers `brainstorming` | not_started | no evidence found | none | The current direction was carried by the gstack-style design docs; run before any future creative spec change |
| Superpowers `writing-plans` | completed | Current context records the skill was read; `experiment_plan.current.*.md` and history provide the executable plan | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | No separate Superpowers plan file was created |
| Superpowers `systematic-debugging` | completed | The first 2026-06-04 JSON assertion failed with `KeyError: 'strict_heldout_pass'`; inspecting the JSON and script showed the assertion used the wrong schema path | `docs/paper_agent/current_action.md` | Root cause: aggregate metrics live under `cross_validation.aggregate_heldout`; no metric changed |
| Superpowers `verification-before-completion` | completed | Log entries `2026-05-31 21:47 CST` and `2026-05-31 22:11 CST` record completed probe-curve audit verification; 2026-06-04 records strict-split unit test, py_compile, audit regeneration, corrected JSON assertions, and `git diff --check` | `docs/paper_agent/current_action.md`, `docs/paper_agent/paper_agent_dashboard.en.md`, `docs/paper_agent/probe_curve_split_score_audit.md` | Key verification: probe-curve audit `Ran 8 tests` / `OK`; strict-split diagnostic `Ran 3 tests` / `OK` and `strict_heldout_pass=False` |
| Superpowers `requesting-code-review` | blocked | Log entry `2026-05-31 21:47 CST` records that no independent Task/subagent reviewer tool was visible; local diff review plus fresh tests were used as fallback | `docs/paper_agent/overnight_log.en.md`, `docs/paper_agent/overnight_log.zh.md` | Independent reviewer was not completed; residual risk is documented |

## Next Resume: First 3 Actions

1. Preserve unrelated `AGENTS.md` / `AGENTS.zh.md` user changes and do not stage them with paper-agent docs.
2. Recheck GPU `2/3` availability with `nvidia-smi`; do not interrupt the currently observed external GPU jobs.
3. If GPU `2` or `3` is free and sandbox-outside CUDA/verifier execution is permitted, run tiny verifier smokes for both LLaDA-1.5 candidate and same-backbone `cal_lite` baseline before any full run.

## Recommended Resume Prompt

```text
Continue the paper-agent work in /home/shx/projects/dllm_infilling/git_workspace on branch paper-agent-overnight in low-token mode. Do not call create_goal/update_goal/get_goal. First read AGENTS.md, git status, docs/paper_agent/pause_checkpoint.current.md, docs/paper_agent/current_action.md, paper_agent_dashboard.zh.md, activity_ledger.zh.md tail, and docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md. Preserve unrelated AGENTS.md / AGENTS.zh.md user changes and do not recreate AGENTS.zh.md. LLaDA-1.5 metadata/API/local-weight probe has completed at `/tmp/llada15_probe_20260609`; do not repeat the full download unless files are missing. Recheck GPU 2/3 availability. If GPU 2 or 3 is free and sandbox-outside CUDA/verifier execution is permitted, run tiny verifier smokes for candidate and same-backbone `cal_lite` baseline, analyze outputs, update GitHub-readable docs, and only then consider full local same-backbone runs. Do not claim SOTA; keep literature anchors separate from local protocol-matched evidence.
```
