# Paper-Agent Activity Ledger

## 2026-06-13 23:14 CST

- action: closed out the two user-approved LLaDA-Base Route 2 trace-gated long-rescue full follow-up runs and updated the recovery path.
- evidence: broad plateau log `logs/paper_agent/20260613_full_route2_broad_gpu2.log`, output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`; precision top1/conf log `logs/paper_agent/20260613_full_route2_precision_gpu3.log`, output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`. Both logs ended with `COMMAND_EXIT_CODE=0`, and both outputs have `1033` valid rows, `0` malformed rows, and `summary.json`.
- result: against the current `midcons` baseline `795/1033 = 76.96%`, broad plateau reaches `801/1033 = 77.54%`, pairwise `7` wins / `1` loss / `794` tie-pass / `231` tie-fail; precision top1/conf reaches `800/1033 = 77.44%`, pairwise `5` wins / `0` losses / `795` tie-pass / `233` tie-fail.
- diagnostics: broad triggers `73` rows with `53.42%` trigger true-long precision, and bucket net `<=8 0`, `9-12 +2`, `13-16 +2`, `17-24 +2`, `25+ 0`. Precision triggers `57` rows with `61.40%` trigger true-long precision, and bucket net `<=8 +1`, `9-12 +1`, `13-16 +2`, `17-24 +1`, `25+ 0`. Among `91` baseline failed-long rows, Broad triggers `39` and rescues only `2`; Precision triggers `35` and rescues only `1`.
- interpretation: the Route 2 trace gate has a real but modest positive full-run signal. The precision policy is cleaner, with no losses and no short loss; broad has a larger net gain but `1` short loss. Neither solves true-long, especially because `25+` is unchanged. The current bottleneck looks more like fixed `len=24` rescue quality/length choice than only gate recall.
- next: treat the precision policy as paper-cleaner incremental evidence. Do not keep threshold-tuning full runs without a new design. First run triggered-but-still-failed / missed failed-long error analysis; a stronger CCF-A claim needs adaptive rescue length, stronger generation-side rescue, or a stronger length signal.

## 2026-06-13 20:55 CST

- action: implemented and ran CPU-only `trace_feature_audit_v2` to test whether richer trace signals exist after the v1 Route 1/2 formulas produced zero triggers.
- evidence: implementation `analysis/trace_feature_audit_v2.py`, tests `tests/test_trace_feature_audit_v2.py`, final output directory `analysis_outputs/trace_feature_audit_v2_20260613_204721`, and report `analysis_outputs/trace_feature_audit_v2_20260613_204721/report.md`.
- result: overall decision is `diagnostic_only`. The previous source has `1033` rows, `113` true-long rows, `96` failed-long rows, and source decision `policy_candidate`; the current `midcons` source has `1033` rows, `113` true-long rows, `91` failed-long rows, and source decision `diagnostic_only`.
- diagnostics: the strongest midcons candidate, `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16`, has `18` held-out triggers, `9` failed-long, `2` short-risk, `0` current-pass risk, and `0.500` true-long precision. This shows trace features carry signal, but cross-source stability and fixed-rule risk are still insufficient.
- next: do not launch a full GPU policy runner directly. If Route 2 continues, first write a small GPU smoke action brief around the low-top1 / late-plateau / low-confidence family.

## 2026-06-12 19:31 CST

- action: completed Task 4/5 full trace collection and offline route analysis for trace-long-rescue.
- evidence: previous trace output `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`; midcons trace output `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`; route analysis directories `analysis_outputs/trace_long_rescue_llada_base_prev_20260612_192611` and `analysis_outputs/trace_long_rescue_llada_base_midcons_20260612_192611`; report `analysis_outputs/trace_long_rescue_report_20260612`.
- result: previous trace run verified `769/1033 = 74.44%` with `35257` trace rows; current `midcons` trace run verified `795/1033 = 76.96%` with `35768` trace rows. Route 1/2 triggered `0` rows on both trace sources; Route 3 stopped with no trace signal. Gate A/B failed for all routes.
- interpretation: this is negative diagnostic evidence for the current trace-only/risk-controlled route family. No route-specific GPU policy full run should be launched from this trace batch.
- next: stop here for policy-runner work unless a new action brief defines a stronger trace feature family or a different route.

## 2026-06-12 18:10 CST

- action: verified the previous local method full trace run, then launched the current `midcons` full trace run as the only active trace-long-rescue GPU job.
- evidence: previous output `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`; current tmux session `trace_llada_base_midcons_20260612`; current log `logs/paper_agent/20260612_full_trace_llada_base_midcons_gpu3.log`; current output `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`.
- result: previous trace verification passed with log exit `0`, `1033` valid result rows, `35257` trace rows linked to `1033` task ids, `summary.json`, and pass count `769/1033 = 74.44%`. Current `midcons` early health passed with at least `26/1033` rows and non-empty traces on GPU3.
- next: monitor `trace_llada_base_midcons_20260612` to completion; verify exit `0`, `summary.json`, `1033` valid rows, and non-empty linked traces before any offline Route 1/2/3 analysis.

## 2026-06-12 17:17 CST

- action: launched Task 4 previous local method full trace collection as the only active GPU job for trace-long-rescue.
- evidence: tmux session `trace_llada_base_prev_20260612`; log `logs/paper_agent/20260612_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log`; output `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`.
- result: early health passed. The run loaded `1033` tasks, created `config.json`, `results.jsonl`, and non-empty `step_traces.jsonl`, and reached at least `204/1033` rows with `6989` trace rows while GPU2 was active.
- next: monitor this tmux session to completion; verify exit `0`, `summary.json`, `1033` valid rows, and non-empty traces before launching the current `midcons` trace run.

## 2026-06-12 16:59 CST

- action: completed trace-long-rescue Task 1/2/3 in serial mode before any GPU full trace run.
- evidence: `docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md` and `docs/paper_agent/current_action.md` passed markdown diff hygiene; added `analysis/trace_long_rescue_features.py`, `analysis/analyze_trace_long_rescue_routes.py`, `analysis/print_trace_long_rescue_report.py`, and `tests/test_trace_long_rescue_features.py`.
- result: focused verification passed with `/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_long_rescue_features.py` reporting `Ran 6 tests` / `OK`; py_compile passed for all three trace-long-rescue analysis scripts; `git diff --check` passed for the touched Task 1/2/3 files.
- interpretation: offline trace analysis tooling is ready, but no route-specific policy runner exists and no GPU trace run has started.
- next: print exact command/log/output/success/kill criteria, then launch only the previous local method full trace run on GPU `2`; verify it before starting the current `midcons` trace run on GPU `3`.

## 2026-06-11 14:42 CST

- action: monitored and closed out the full `inclusionAI/LLaDA-MoE-7B-A1B-Base` local same-backbone pair, generated pairwise/bucket analysis, and synchronized the result into public experiment docs.
- evidence: baseline log `logs/paper_agent/20260611_1126_full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared.log` and candidate log `logs/paper_agent/20260611_1126_full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared.log` both ended with `COMMAND_EXIT_CODE="0"`. Baseline output `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719` and candidate output `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740` each have `1033` valid rows, `0` malformed rows, and `summary.json`. Pairwise analysis is `analysis_outputs/lladamoe_full_pair_20260611_1438`.
- result: candidate `801/1033 = 77.54%` versus local `cal_lite` LCAS-v3b baseline `777/1033 = 75.22%`; pairwise `31` wins, `7` losses, `770` tie-pass, `225` tie-fail; avg sec including probe `10.6107` versus `8.7025`.
- diagnostics: oracle-bucket deltas are `<=8 +9`, `9-12 +5`, `13-16 +4`, `17-24 +6`, and `25+ 0`. Official repair trigger precision remains weak for true-long detection: `official_repair_true_long_precision = 11.54%`, `official_long_suspicion_true_long_precision = 40.00%`, and `official_mid_rescue_true_long_precision = 13.04%`; `under_select_rate_17plus = 91.15%`.
- interpretation: strongest current local transfer result and a valid local same-backbone improvement claim for LLaDA-MoE, but not an external SOTA claim. Literature anchors remain separate from protocol-matched local controls.
- next: do not launch another GPU full run without a new action brief. The next research step should design a stronger length signal or ablation plan, or explicitly choose an uncovered checkpoint variant.

## 2026-06-11 11:30 CST

- action: completed the `inclusionAI/LLaDA-MoE-7B-A1B-Base` smoke gate and launched the full local same-backbone baseline/candidate pair.
- evidence: resumed download log `logs/paper_agent/20260611_1100_lladamoe_aria2_resume.log` exited `0`; smoke baseline output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112427` and candidate output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112452` are both `2/2` with exit `0`. Full baseline tmux session `lladamoe_full_base_20260611_1126` writes `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719`; full candidate tmux session `lladamoe_full_candidate_20260611_1126` writes `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740`.
- result: the full runs are still running, so there is no final pass rate yet. Early health showed about `16/1033` baseline tasks and `11/1033` candidate tasks, GPUs `2/3` each at about `15-16GB`, with no OOM or import failure.
- diagnostics: LLaDA-MoE needs the `modeling_rope_utils` path from `llmxy` Transformers `4.52.3`, while the `dllm_env` `flash_attn_2_cuda` extension fails on `GLIBC_2.32`; the active commands therefore use a `/tmp/no_flash_attn` shim plus `dllm_env` verifier dependencies.
- next: monitor both tmux sessions to completion; after completion, verify exit codes, `1033` rows, summaries, then generate pairwise/bucket/runtime analysis and local/history/literature comparison tables.

## 2026-06-10 21:36 CST

- action: continued the `inclusionAI/LLaDA-MoE-7B-A1B-Base` download/API gate and prepared the smoke-pair brief before any GPU experiment.
- evidence: active tmux session `lladamoe_aria2_20260610_1950`; active log `logs/paper_agent/20260610_1950_lladamoe_aria2_download.log`; download/probe brief `docs/paper_agent/experiments/20260610_1941_lladamoe_download_api_probe.md`; smoke plan `docs/paper_agent/experiments/20260610_2136_lladamoe_smoke_pair.md`.
- result: download is still in progress. The apparent `14G` local shard size is due to `aria2c` preallocation; `.aria2` sidecar files are still present, so the checkpoint is not complete.
- next: continue monitoring until the tmux session exits and `.aria2` files disappear, then validate shard bytes, run a local API/load probe, and only then launch the two 2-sample smokes.

## 2026-06-10 19:25 CST

- action: monitored the full `GSAI-ML/LLaDA-1.5` local same-backbone pair to completion, verified row counts and exit codes, recomputed raw-row pairwise/bucket/runtime metrics, and updated the public result docs.
- evidence: baseline output `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_cal_lite_lcas_v3b_gpu2_shared_20260610_172705`; candidate output `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_lcal_official_bounded_repair_gpu3_shared_20260610_172720`; pairwise analysis `analysis_outputs/llada15_full_pair_20260610_1923`; final brief `docs/paper_agent/experiments/20260610_1735_full_llada15_parallel_baseline_candidate.md`.
- result: candidate is `818/1033 = 79.19%` versus local same-backbone `cal_lite` LCAS-v3b baseline `817/1033 = 79.09%`, pairwise `18` wins and `17` losses, net `+1` task. Avg sec including probe is `6.6453` versus `5.4224`.
- diagnostics: candidate loses `6` net tasks in oracle `<=8`, ties `9-12`, and gains `+7` total across oracle `>=13`. Official repair triggers `110/1033 = 10.65%` rows, but true-long precision is only `10.91%`; `82/110` triggers are oracle `<=8`.
- interpretation: near-tie/slight local positive, not a strong claim upgrade. The current official-CAL repair family remains an imprecise true-long detector.
- next: continue the literature-backbone matrix with `inclusionAI/LLaDA-MoE-7B-A1B-Base`; first download/probe the checkpoint through the configured proxy/mirror path, then run tiny baseline/candidate smokes before any full pair.

## 2026-06-10 17:35 CST

- action: launched the full `GSAI-ML/LLaDA-1.5` local same-backbone pair after both 2-sample smokes passed.
- evidence: full-run brief `docs/paper_agent/experiments/20260610_1735_full_llada15_parallel_baseline_candidate.md`; baseline log `logs/paper_agent/20260610_1735_full_llada15_cal_lite_lcas_v3b_gpu2_shared.log`; candidate log `logs/paper_agent/20260610_1735_full_llada15_lcal_official_bounded_repair_gpu3_shared.log`.
- result: early health is good. Both runs loaded local LLaDA-1.5 shards, loaded `1033` HumanEval-SingleLineInfilling tasks, and began decoding. GPU2/GPU3 memory stayed around `25.8GB`/`25.9GB` total after load despite the existing shared jobs.
- next: monitor both tmux sessions to completion, then run row-count/JSON sanity checks and raw-row pairwise/bucket/runtime analysis before making any claim.

## 2026-06-09 21:55 CST

- action: completed the `GSAI-ML/LLaDA-1.5` metadata/API/local-weight probe through the configured proxy path and documented the result.
- evidence: local model path `/tmp/llada15_probe_20260609`; probe brief `docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`; current action `docs/paper_agent/current_action.md`.
- result: direct HuggingFace through the proxy worked; `hf-mirror.com` redirected this repo back to HuggingFace and `huggingface_hub` mirror mode failed. All six safetensors shards were downloaded and byte-size checked against the HF API/index with total size `16,031,197,144` bytes. Local `AutoConfig`, `AutoTokenizer`, `AutoModel.from_config`, and CPU/local checkpoint loading passed. LLaDA-1.5 uses `LLaDAModelLM`, `model_type=llada`, config `mask_token_id=126336`, and tokenizer token `<|mdm_mask|>` also resolves to `126336`; `tokenizer.mask_token` itself is `None`.
- caveat: this is not a GPU smoke and not a performance result. A fresh `nvidia-smi` showed GPUs `2` and `3` occupied, and the sandboxed Python process reported `torch.cuda.is_available() == False`.
- next: when GPU `2` or `3` is free and unsandboxed CUDA/verifier execution is permitted, run tiny candidate and same-backbone `cal_lite` baseline smokes before any full LLaDA-1.5 pair.

## 2026-06-09 21:05 CST

- action: synchronized the just-completed DiffuCoder-Base full local pair into the recovery path and opened the next literature-backbone action brief for `GSAI-ML/LLaDA-1.5`.
- evidence: DiffuCoder full-pair brief `docs/paper_agent/experiments/20260609_1925_full_diffucoder_base_parallel_baseline_candidate.md`; next-action brief `docs/paper_agent/current_action.md`; LLaDA-1.5 probe brief `docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`.
- result: DiffuCoder-Base candidate is `839/1033 = 81.22%` versus its local same-backbone `cal_lite` baseline `838/1033 = 81.12%`, pairwise `25` wins and `24` losses. This remains a near-tie/slight local positive, not a strong bounded-repair claim.
- next: run the proxy-enabled LLaDA-1.5 metadata/download probe, inspect config/tokenizer/mask token, then run tiny GPU smoke only if the API contract is compatible.

## 2026-06-09 18:27 CST

- action: monitored the full `Dream-org/Dream-v0-Base-7B` baseline/candidate pair to completion, verified row counts/schema/backend/canvas, recomputed same-backbone pairwise/bucket/runtime metrics from raw `results.jsonl`, and updated the public experiment docs.
- evidence: baseline output `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_170219`; candidate output `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_170219`; both have `1033` rows, `summary.json`, `0` malformed JSON rows, and the expected backend/canvas.
- result: candidate is `803/1033 = 77.73%` versus local same-backbone cal_lite baseline `802/1033 = 77.64%`, pairwise `28` wins and `27` losses, net `+1` task. Avg sec including probe is `3.7337` versus `3.6494`. This is a near-tie/slight local positive, not a strong claim.
- diagnostics: long buckets improve by `+3` total tasks across oracle `>=17`, but `9-12` and `13-16` regress by `-5` total tasks. Repair triggers `101/1033 = 9.78%` rows, but only `13/101 = 12.87%` triggered rows are true-long; most triggers are short.
- literature positioning: candidate is above LR-DLLM Dream-7B single-line `76.7`, while DreamOn Dream-7B `88.6` is training-based and much higher. These are anchors only, not protocol-matched claims.
- next: continue the backbone matrix with `apple/DiffuCoder-7B-Base`. Use HuggingFace mirror or Git/LFS download/API probe first, then tiny smoke; do not launch a full run until the runner/canvas/verifier contract is checked.

## 2026-06-09 14:35 CST

- action: checked the completed DreamCoder Base/Instruct full runs, recomputed same-backbone pairwise/bucket/runtime metrics from raw `results.jsonl`, and updated the public experiment docs.
- evidence: Base output `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`; Instruct output `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`; both have `1033` rows and `summary.json`. GPU check after completion showed no running GPU processes.
- result: Base candidate is `832/1033 = 80.54%` versus local same-backbone baseline `825/1033 = 79.86%`, pairwise `27` wins and `20` losses, net `+7` tasks. Instruct candidate is `834/1033 = 80.74%` versus local same-backbone baseline `848/1033 = 82.09%`, pairwise `21` wins and `35` losses, net `-14` tasks. Base is a small local positive; Instruct is negative transfer evidence.
- literature positioning: Base is above CAL DreamCoder-Base anchors (`70.2` average, `76.2` best shown) and below LR-DLLM DreamCoder-7B `81.6`; DreamOn DreamCoder `92.1` is training-based. These are anchors only, not protocol-matched claims.
- next: continue the backbone matrix with `apple/DiffuCoder-7B-Base`. Use the HuggingFace mirror, first run a download/API probe and tiny smoke on GPU 2/3, and do not launch a full run until the runner/canvas/verifier contract is checked.

## 2026-06-09 12:12 CST

- action: started and debugged the `Dream-org/Dream-Coder-v0-Base-7B` official-canvas LCAL/S3 + official-CAL bounded-repair smoke.
- evidence: the first smoke log `logs/paper_agent/20260609_1203_smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2.log` shows model loading succeeded but `datasets` failed when creating a lock under read-only `/home/shx/.cache/huggingface/datasets`. After copying the HF module and dataset caches to `/tmp`, a dataset-load probe succeeded with `2` tasks. The second smoke loaded the model and tasks, then failed in the HumanEval verifier because sandboxed `multiprocessing.Manager()` cannot create its listener socket; CUDA probes inside the sandbox also showed `torch.cuda.is_available() == False`.
- result: this is an environment/sandbox blocker, not a DreamCoder method result. There is no valid pass rate or runtime yet.
- next: obtain explicit user approval for sandbox-outside execution, then run the 2-sample Base smoke on GPU 2. If it passes, continue to Instruct smoke/full runs. Do not use the failed sandbox output directories for paper comparison.

## 2026-06-09 12:31 CST

- action: after user approval, completed sandbox-outside DreamCoder Base and DreamCoder Instruct 2-sample smokes and prepared full parallel runs.
- evidence: Base output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_122358`; Instruct output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_122945`. Both have `2` JSON rows, `summary.json`, tier1/2/3 verifier outputs, and `lcal_v3`/`official_cal` metadata. Both smokes are `2/2` pass and `2` tie-pass against their own baseline over the first two common tasks.
- result: the official-canvas adapter passed the runner/schema/verifier/GPU-runtime sanity gate; this is not a full performance result. Wrote the full-run action brief with Base on GPU2 and Instruct on GPU3.
- next: launch the two full `1033` sample runs, then compare each against its same-backbone baseline and literature anchors.

## 2026-06-09 12:34 CST

- action: launched the DreamCoder Base/Instruct full parallel runs.
- evidence: Base tmux session `dreamcoder_base_full_20260609_1231`, output `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`, log `logs/paper_agent/20260609_1231_full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log`; Instruct tmux session `dreamcoder_instruct_full_20260609_1231`, output `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`, log `logs/paper_agent/20260609_1231_full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed.log`.
- result: early health check passed; Base had `15` observed rows and Instruct had `7` observed rows; GPUs 2/3 were both at `95%+` utilization with about `15-16GB` memory used and no early OOM.
- next: after completion, run row-count/JSON sanity checks, read `summary.json`, compare each run against its same-backbone baseline with pairwise/bucket/runtime analysis, and report literature anchors separately.

## 2026-06-09 11:04 CST

- action: inspected the completed `GSAI-ML/LLaDA-8B-Instruct + midcons` run and designed the literature-backbone rerun matrix.
- evidence: candidate summary at `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834/summary.json`; historical baseline summary at `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_llada-instruct_resume_20260512_141851/summary.json`; source PDFs for CAL, LR-DLLM, and DreamOn.
- result: candidate is `815/1033 = 78.90%`, below the same-backbone historical baseline `817/1033 = 79.09%`; pairwise `17` wins and `19` losses. Wrote `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`.
- next: do not upgrade the claim; next implementation target is a DreamCoder official-canvas bounded-repair adapter with smoke-run criteria before any full GPU run.

## 2026-06-04 20:31 CST

- action: started the user-approved `GSAI-ML/LLaDA-8B-Instruct` + current `midcons` bounded-repair full run on GPUs `2,3`.
- evidence: the first direct `huggingface.co` launch failed before model loading because the network was unreachable and was interrupted with Ctrl-C; the run was relaunched with `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`. Log `logs/paper_agent/20260604_2022_llada_instruct_midcons_full_hfmirror.log` shows checkpoint shards loaded, `Loaded 1033 tasks`, and output directory `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`.
- result: at this timestamp the run had not finished yet; the latest check found `62` rows in `results.jsonl` and GPUs 2/3 in use. This interim state is superseded by the 2026-06-09 final result above.
- next: after completion, run row-count/JSON sanity checks, read `summary.json`, compare against the historical LLaDA-Instruct baseline `817/1033 = 79.09%`, and update results/dashboard/checkpoint.

## 2026-06-04 15:40 CST

- action: ran final focused verification for the strict-split probe-score diagnostic.
- evidence: `/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_split_score.py`, py_compile, `analysis/analyze_probe_curve_split_score.py`, JSON assertions, and `git diff --check`.
- result: fresh verification passed: `Ran 3 tests` / `OK`, compile exit `0`, audit regeneration reproduced `strict_heldout_pass=False heldout_triggers=63 short_risk=22.22%`, JSON assertions confirmed key metrics under `cross_validation.aggregate_heldout`, and diff hygiene passed. The first JSON assertion failed because the schema path was wrong; root cause was the assertion command, not changed audit metrics.
- next: keep the result as negative evidence; default next step is a CPU-only conservative high-precision score kill-test, with GPU work still blocked.

## 2026-06-04 15:31 CST

- action: after resuming the long paper-agent session, reconciled the verification status across the dashboard, checkpoint, experiment results, and overnight log.
- evidence: `AGENTS.md`, `docs/paper_agent/research_agent_protocol.md`, `git status --short --branch`, the tails of `pause_checkpoint.current.md`, `paper_agent_dashboard.zh.md`, `experiment_results.zh.md`, `open_questions.zh.md`, `activity_ledger.zh.md`, and verification-related `rg` hits.
- result: confirmed that the dashboard's `verification-before-completion=completed` refers to the earlier probe-curve audit with `Ran 8 tests` / `OK`; the strict-split diagnostic only has focused `Ran 3 tests` / `OK` and still needs final milestone-level verification, diff review, commit, and push. Updated the checkpoint workflow wording; launched no GPU work or new experiment.
- next: run final focused verification for the strict-split diagnostic before deciding whether to run a CPU-only conservative high-precision score kill-test; GPU work remains blocked.

## 2026-06-01 01:50 CST

- action: resumed in low-token mode and reconciled the stale pause checkpoint with current dirty files.
- evidence: `git status --short --branch`, `docs/paper_agent/pause_checkpoint.current.md`, `docs/paper_agent/paper_agent_dashboard.zh.md`, `docs/paper_agent/evidence_snapshot.md`, `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/open_questions.en.md`, and the tail of `overnight_log.*.md`.
- result: confirmed that the uncommitted strict-split probe diagnostic is the current planned CPU-only next step; no gstack workflow or GPU experiment was launched.
- next: finish the strict-split diagnostic evidence and record whether it passes the offline GPU gate.

## 2026-06-01 01:52 CST

- action: ran the strict-split probe-score unit test and generated the CPU-only audit from the existing A6000 `midcons` result.
- evidence: `tests/test_analyze_probe_curve_split_score.py`, `analysis/analyze_probe_curve_split_score.py`, `docs/paper_agent/probe_curve_split_score_audit.json`, `docs/paper_agent/probe_curve_split_score_audit.md`.
- result: tests passed; aggregate held-out gate failed with `63` triggers, `47.62%` true-long precision, `32.97%` failed-long recall, `22.22%` short-risk, and `7.94%` current-pass risk.
- next: document the negative result bilingually and keep GPU work blocked.
