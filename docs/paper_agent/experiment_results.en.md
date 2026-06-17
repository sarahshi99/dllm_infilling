# Experiment Results

Updated: 2026-06-17 17:10 CST

## Three-Way Comparison: Paper-Reported Numbers vs Our Previous Methods vs Current Method

Important terminology correction: the "our previous/local method" column below is not a local reproduction of the corresponding paper method. It is the earlier local method or local control version from this project. Paper-reported numbers are placed in the same table only to show external positioning; they are not the local protocol-matched control.

| Backbone / checkpoint | Relevant paper-reported numbers | Our previous/local method | Current method | Current vs previous | Current relative to paper reports |
|---|---|---:|---:|---:|---|
| `GSAI-ML/LLaDA-8B-Base` | CAL: avg `65.5`, best shown `73.6`; LR-DLLM LLaDA-8B: `69.4` | A6000 control `787/1033 = 76.19%` | Mainline `midcons` `795/1033 = 76.96%`; Route2 precision len24 `800/1033 = 77.44%`; Route2 broad len24 `801/1033 = 77.54%`; Route2 precision len32 `801/1033 = 77.54%` | `midcons +8` tasks / `+0.77pp`; Route2 precision len24 `+13` tasks / `+1.26pp`; Route2 broad len24 and precision len32 are both `+14` tasks / `+1.36pp` vs A6000 control | Above CAL best `73.6` and LR-DLLM `69.4`; Route2 is small follow-up evidence, not an external SOTA claim |
| `GSAI-ML/LLaDA-8B-Instruct` | CAL: avg `69.9`, best shown `76.9` | historical LCAS-v3 `817/1033 = 79.09%` | `midcons` `815/1033 = 78.90%` | `-2` tasks / `-0.19pp` | Above CAL best `76.9`, but below our previous method |
| `Dream-org/Dream-Coder-v0-Base-7B` | CAL: avg `70.2`, best shown `76.2`; LR-DLLM DreamCoder: `81.6`; DreamOn DreamCoder: `92.1` | official-canvas `cal_lite` `825/1033 = 79.86%` | bounded repair `832/1033 = 80.54%` | `+7` tasks / `+0.68pp` | Above CAL best `76.2`, below LR-DLLM `81.6` and DreamOn `92.1` |
| `Dream-org/Dream-Coder-v0-Instruct-7B` | No exact matching paper-reported row | official-canvas `cal_lite` `848/1033 = 82.09%` | bounded repair `834/1033 = 80.74%` | `-14` tasks / `-1.36pp` | No direct paper-number comparison; local negative transfer |
| `Dream-org/Dream-v0-Base-7B` | LR-DLLM Dream-7B: `76.7`; DreamOn Dream-7B: `88.6` | `cal_lite` `802/1033 = 77.64%` | bounded repair `803/1033 = 77.73%` | `+1` task / `+0.10pp` | Slightly above LR-DLLM `76.7`, below DreamOn `88.6` |
| `apple/DiffuCoder-7B-Base` | CAL: avg `68.0`, best shown `74.8`; DreamOn DiffuCoder: `92.2` | `cal_lite` `838/1033 = 81.12%` | bounded repair `839/1033 = 81.22%` | `+1` task / `+0.10pp` | Above CAL best `74.8`, below DreamOn `92.2` |
| `GSAI-ML/LLaDA-1.5` | LR-DLLM LLaDA-1.5: `68.9` | `cal_lite` LCAS-v3b `817/1033 = 79.09%` | bounded repair `818/1033 = 79.19%` | `+1` task / `+0.10pp` | Above LR-DLLM `68.9` |
| `inclusionAI/LLaDA-MoE-7B-A1B-Base` | LR-DLLM LLaDA-MoE: `71.3` | `cal_lite` LCAS-v3b `777/1033 = 75.22%` | bounded repair `801/1033 = 77.54%` | `+24` tasks / `+2.32pp` | Above LR-DLLM `71.3`; strongest current local gain |

Use this table as follows: paper claims can report both the current method's position relative to paper-reported numbers and the local gain over our previous method. The writing must still state that the paper-reported numbers are not local controls under our exact protocol.

## LLaDA-Base Full Trace Long-Rescue Diagnostics

Full trace collection completed for the previous local method and the current `midcons` method. These diagnostics use trace/decode dynamics for triggers and labels only for offline Gate A/B accounting; they are not a new SOTA claim.

| Run | Output | Rows | Trace rows | Pass rate |
| --- | --- | --- | --- | --- |
| previous local method trace | /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552 | 1033 | 35257 | 769/1033 = 74.44% |
| current midcons trace | /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846 | 1033 | 35768 | 795/1033 = 76.96% |

Offline route analysis:

| Trace source | Route | Triggers | Failed-long | Short | Current-pass risk | Gate A | Gate B | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| previous local method | Route 1 trace-only detector | 0 | 0 | 0 | 0 | no | no | stop |
| previous local method | Route 2 risk-controlled rescue | 0 | 0 | 0 | 0 | no | no | stop |
| previous local method | Route 3 multi-canvas trace rerank | 0 | 0 | 0 | 0 | no | no | stop_no_trace_signal |
| current midcons | Route 1 trace-only detector | 0 | 0 | 0 | 0 | no | no | stop |
| current midcons | Route 2 risk-controlled rescue | 0 | 0 | 0 | 0 | no | no | stop |
| current midcons | Route 3 multi-canvas trace rerank | 0 | 0 | 0 | 0 | no | no | stop_no_trace_signal |

Interpretation: no route met the offline continuation rule. This is diagnostic negative evidence; no route-specific GPU policy full run should be launched from these traces.

## Trace Feature Audit V2

`trace_feature_audit_v2` is a CPU-only offline diagnostic. It does not report a new pass rate, and the audit itself did not launch GPU work. It reuses the two full trace outputs above and searches richer trace-shape, stop-reason, motif, and model-assisted discovery families for a readable long-rescue gate. The final valid output directory is `analysis_outputs/trace_feature_audit_v2_20260613_204721`.

Decision: `diagnostic_only`. The reason is that the previous trace source produced policy-level candidates, but the current `midcons` trace source reached only diagnostic-only; the signal is not stable enough across sources to launch a Route 2 GPU policy runner directly.

| Source | Rows | True-long | Failed-long | Short | Decision |
|---|---:|---:|---:|---:|---|
| previous | 1033 | 113 | 96 | 598 | policy_candidate |
| midcons | 1033 | 113 | 91 | 598 | diagnostic_only |

Top held-out train-selected candidates:

| Source | Candidate | Decision | Fold | Train rank | Triggers | Failed-long | Short risk | Current-pass risk | Precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| previous | `confidence_first_le_0p875_AND_gap_last_le_0p84375` | policy_candidate | 1 | 2 | 21 | 10 | 4 | 4 | 0.476 |
| previous | `confidence_first_le_0p882812_AND_gap_last_le_0p84375` | policy_candidate | 4 | 3 | 25 | 11 | 5 | 3 | 0.440 |
| previous | `top1_last_le_0p859375_AND_confidence_first_le_0p875` | policy_candidate | 1 | 3 | 20 | 10 | 3 | 4 | 0.500 |
| previous | `top1_last_le_0p605469_AND_confidence_min_le_0p730469` | policy_candidate | 3 | 4 | 15 | 10 | 3 | 0 | 0.667 |
| midcons | `top1_last_le_0p667969_AND_max_remaining_plateau_steps_ge_16` | diagnostic_only | 4 | 1 | 18 | 9 | 2 | 0 | 0.500 |
| midcons | `top1_last_le_0p625_AND_max_remaining_plateau_steps_ge_16` | diagnostic_only | 3 | 1 | 12 | 8 | 2 | 0 | 0.667 |

Interpretation: v2 shows that trace features are not signal-free. Low `top1_last`, low confidence/gap, and late plateau candidates can capture failed-long rows. However, fixed-rule cross-source/full-data transfer still has too much short-risk, so the audit's original conclusion was that the result should guide the next stricter gate or smoke design rather than automatically trigger a full GPU policy run. After the user explicitly asked to continue, two Route 2 full follow-up runs were completed; see the next section.

## LLaDA-Base Route2 Trace-Gated Long Rescue Full Runs

After the user explicitly preferred full runs, two Route2 trace-gated long-rescue full policy runs completed on GPUs `2/3`. The user then required a `len32` follow-up to avoid GPU0/GPU1 and use GPU3 only; that clean GPU3-only full run also completed. The policy uses only inference-time trace/decode features to trigger fixed rescue. Oracle/verifier labels are used only for offline accounting and are not used to choose between primary and rescue outputs.

Outputs:

- Broad: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`
- Precision: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`
- Precision len32: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`
- Comparison baseline: current `midcons` trace run `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`

Verification: all three logs ended with `COMMAND_EXIT_CODE=0`; all three `results.jsonl` files have `1033` rows; Broad has `39740` step-trace rows, Precision len24 has `38872`, and Precision len32 has a nonempty `step_traces.jsonl`; all three output directories contain `summary.json`.

| Run | Pass | Rate | Delta vs current `midcons` | Avg sec incl. probe |
|---|---:|---:|---:|---:|
| current `midcons` baseline | `795/1033` | `76.96%` | baseline | n/a |
| Route2 broad len24 | `801/1033` | `77.54%` | `+6` tasks / `+0.58pp` | `5.0852` |
| Route2 precision len24 | `800/1033` | `77.44%` | `+5` tasks / `+0.48pp` | `5.0945` |
| Route2 precision len32 | `801/1033` | `77.54%` | `+6` tasks / `+0.58pp` | `5.4622` |

Trigger / pairwise:

| Policy | Triggers | Trigger pass | True-long precision | Pairwise W/L/TP/TF vs `midcons` | Short triggers | Primary-pass risk |
|---|---:|---:|---:|---:|---:|---:|
| Route2 broad len24 | `73` (`7.07%`) | `9.59%` | `53.42%` | `7/1/794/231` | `10` | `1` |
| Route2 precision len24 | `57` (`5.52%`) | `8.77%` | `61.40%` | `5/0/795/233` | `6` | `0` |
| Route2 precision len32 | `57` (`5.52%`) | `10.53%` | `61.40%` | `6/0/795/232` | `6` | `0` |

Oracle bucket pass rates:

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| current `midcons` baseline | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` |
| Route2 broad len24 | `89.97%` | `79.31%` | `61.11%` | `23.17%` | `16.13%` |
| Route2 precision len24 | `90.13%` | `78.88%` | `61.11%` | `21.95%` | `16.13%` |
| Route2 precision len32 | `90.30%` | `79.31%` | `58.89%` | `23.17%` | `16.13%` |

Bucket pairwise vs current `midcons`:

| Policy | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| Route2 broad len24 | `W1/L1/net+0` | `W2/L0/net+2` | `W2/L0/net+2` | `W2/L0/net+2` | `W0/L0/net+0` |
| Route2 precision len24 | `W1/L0/net+1` | `W1/L0/net+1` | `W2/L0/net+2` | `W1/L0/net+1` | `W0/L0/net+0` |
| Route2 precision len32 | `W2/L0/net+2` | `W2/L0/net+2` | `W0/L0/net+0` | `W2/L0/net+2` | `W0/L0/net+0` |

Long-failure coverage:

| Policy | Failed-long total | Failed-long triggered | Rescue wins in failed-long | Triggered but still fail | Failed-long not triggered |
|---|---:|---:|---:|---:|---:|
| Route2 broad len24 | `91` | `39` | `2` | `37` | `52` |
| Route2 precision len24 | `91` | `35` | `1` | `34` | `56` |
| Route2 precision len32 | `91` | `35` | `2` | `33` | `56` |

Interpretation: the Route2 full runs are small positive results, not a solution to the long-length bottleneck. Broad len24 gains `+6` tasks over current `midcons`, but has `1` primary-pass loss and more short triggers. Precision len24 gains `+5` tasks with no observed primary-pass loss. Precision len32 gains `+6` tasks with no observed primary-pass loss, making it the cleaner follow-up so far. The key diagnostic is that the gate finds some failed-long rows, but fixed rescue usually cannot recover them. Precision len32 gains `+2` in oracle `17-24`, but `25+` is unchanged and triggered oracle `25+` rows are `0/11` pass. The next step should analyze triggered-but-still-failed and missed failed-long rows before deciding whether training-free adaptive rescue length, better rescue decoding, or trace/probe fusion is justified.

## Route2 Error Analysis Discovery V3

`route2_error_analysis` is a CPU-only diagnostic that explains why Route2 precision `len32` produces only a small gain and whether the next Discovery layer should optimize gate recall, rescue generation/selection, or adaptive length. It launches no GPU work and adds no new pass-rate claim.

Output directory: `analysis_outputs/route2_error_analysis_20260617_165806`; report: `analysis_outputs/route2_error_analysis_20260617_165806/report.md`.

| Metric | Value |
|---|---:|
| Joined rows | `1033` |
| Pairwise W/L/TP/TF | `6/0/795/232` |
| Route2 triggers | `57` |
| Triggered failed-long rows | `33` |
| Missed failed-long rows | `56` |
| Triggered failed-long with rescue length >= oracle | `31/33` |
| Dominant bottleneck | `mixed_rescue_quality_and_gate_recall` |
| Recommended next path | `rescue_generation_quality+gate_recall` |

Interpretation: the result supports continued signal search rather than abandoning true-long rescue. All `6` Route2 wins come from triggered rescue, so the gate has real signal. However, `31/33` triggered failed-long rows already have rescue length at least oracle length, so blind canvas-length increases are not the default answer. At the same time, `56` failed-long rows are missed entirely, so gate recall is still insufficient. The next CPU-first designs should inspect rescue generation/selection failures and expand probe-trace fusion for missed failed-long recall.

## LLaDA-MoE Local Same-Backbone Pair

The full `inclusionAI/LLaDA-MoE-7B-A1B-Base` local baseline/candidate pair completed on GPUs `2` and `3`.

Outputs:

- Baseline: `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719`
- Candidate: `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740`
- Pairwise analysis: `analysis_outputs/lladamoe_full_pair_20260611_1438`

Verification:

- Baseline log ended with `COMMAND_EXIT_CODE="0"`.
- Candidate log ended with `COMMAND_EXIT_CODE="0"`.
- Both `results.jsonl` files have `1033` valid rows and `0` malformed rows.
- Both output directories contain `summary.json`.

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `inclusionAI/LLaDA-MoE-7B-A1B-Base` | `cal_lite` LCAS-v3b local baseline | `777/1033` | `75.22%` | baseline | `8.7025` | n/a |
| `inclusionAI/LLaDA-MoE-7B-A1B-Base` | LCAL official bounded repair | `801/1033` | `77.54%` | `+24` tasks / `+2.32pp` | `10.6107` | `31/7/770/225` |

Bucket pairwise by oracle length:

| Oracle bucket | Count | Wins | Losses | Net | Baseline pass | Candidate pass |
|---|---:|---:|---:|---:|---:|---:|
| `<=8` | `598` | `16` | `7` | `+9` | `88.46%` | `89.97%` |
| `9-12` | `232` | `5` | `0` | `+5` | `77.59%` | `79.74%` |
| `13-16` | `90` | `4` | `0` | `+4` | `57.78%` | `62.22%` |
| `17-24` | `82` | `6` | `0` | `+6` | `14.63%` | `21.95%` |
| `25+` | `31` | `0` | `0` | `0` | `12.90%` | `12.90%` |

Repair diagnostics: candidate triggered official repair on `104/1033 = 10.07%` rows, `official_long_suspicion` on `15/1033 = 1.45%`, and `official_mid_rescue` on `23/1033 = 2.23%`. Trigger precision remains weak for true-long detection: `official_repair_true_long_precision = 11.54%`, `official_long_suspicion_true_long_precision = 40.00%`, and `official_mid_rescue_true_long_precision = 13.04%`. The candidate remains strongly under-selective on true-long rows: `under_select_rate_17plus = 91.15%`.

Interpretation: this is the strongest current local transfer result. Unlike the near-tie results on Dream-7B, DiffuCoder-Base, and LLaDA-1.5, LLaDA-MoE gains `+24` tasks with only `7` losses, and gains are positive or neutral in every oracle bucket. The cost is higher runtime: `10.6107s` versus `8.7025s` per sample including probe. This supports a local protocol-matched improvement claim for LLaDA-MoE, but it is still not an external SOTA claim because DreamOn is training-based and much higher on Dream/DiffuCoder/DreamCoder, and LR-DLLM/CAL are not local protocol-matched controls.

Literature positioning: the candidate `77.54%` is above the LR-DLLM LLaDA-MoE single-line anchor `71.3` and its reported baseline anchor `48.8`, but this remains a suggestive anchor rather than a protocol-matched comparison.

Detailed brief: `docs/paper_agent/experiments/20260611_1126_lladamoe_full_pair.md`.

## LLaDA-1.5 Local Same-Backbone Pair

The full `GSAI-ML/LLaDA-1.5` local baseline/candidate pair completed on shared GPUs `2` and `3`.

Outputs:

- Baseline: `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_cal_lite_lcas_v3b_gpu2_shared_20260610_172705`
- Candidate: `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_lcal_official_bounded_repair_gpu3_shared_20260610_172720`

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `GSAI-ML/LLaDA-1.5` | `cal_lite` LCAS-v3b local baseline | `817/1033` | `79.09%` | baseline | `5.4224` | n/a |
| `GSAI-ML/LLaDA-1.5` | LCAL official bounded repair | `818/1033` | `79.19%` | `+1` task / `+0.10pp` | `6.6453` | `18/17/800/198` |

Bucket summary:

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| LLaDA-1.5 candidate | `90.47%` | `83.19%` | `67.78%` | `21.95%` | `16.13%` |
| LLaDA-1.5 baseline | `91.47%` | `83.19%` | `64.44%` | `18.29%` | `12.90%` |

Bucket pairwise: candidate loses `6` net tasks in oracle `<=8`, ties `9-12`, and gains `+3`, `+3`, and `+1` in `13-16`, `17-24`, and `25+`.

Repair diagnostics: candidate triggered official repair on `110/1033 = 10.65%` rows. Triggered-row true-long precision is only `12/110 = 10.91%`; `82/110` triggers are oracle `<=8`. `official_long_suspicion` is especially noisy: `29/33` of its triggers are oracle `<=8`.

Interpretation: this is a near-tie/slight local positive result, not a strong claim upgrade. The candidate recovers a few medium/long tasks but pays with short-bucket regressions and higher cost: `6.6453s` versus `5.4224s` per sample including probe. The result supports the broader diagnosis that true-long recovery remains weak and the current official-CAL trigger family is not a precise true-long detector.

Literature positioning: the local LLaDA-1.5 rows are above LR-DLLM's LLaDA-1.5 single-line anchor `68.9` and its reported LLaDA-1.5 baseline anchor `48.8`, but those are literature anchors, not protocol-matched local comparisons.

Detailed brief: `docs/paper_agent/experiments/20260610_1735_full_llada15_parallel_baseline_candidate.md`.

## LLaDA-Instruct Cross-Model Result

The user-approved `GSAI-ML/LLaDA-8B-Instruct + midcons` full run completed on GPUs `2,3`.

Output:

`/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`

| Model | Run | Pass | Rate | Avg sec/sample incl. probe | Comparison |
|---|---|---:|---:|---:|---|
| `GSAI-ML/LLaDA-8B-Instruct` | historical LCAS-v3 | `817/1033` | `79.09%` | `6.8661` | same-backbone baseline |
| `GSAI-ML/LLaDA-8B-Instruct` | current `midcons` bounded repair | `815/1033` | `78.90%` | `4.1766` | `-2` tasks vs baseline |

Pairwise comparison: `17` wins, `19` losses, `798` tie-pass, `199` tie-fail. The losses are concentrated in short buckets: `17` in oracle `<=8` and `2` in `9-12`, while the candidate has `4` long-bucket wins among oracle `>=17`.

Interpretation: this is negative transfer evidence. `midcons` remains a useful LLaDA-Base checkpoint, but it does not transfer cleanly to LLaDA-Instruct and should not be used to upgrade the claim. The next experiment phase must compare each literature backbone against its own baseline.

Planning doc: `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`.

## DreamCoder Official-Canvas Cross-Backbone Result

The sandbox-outside DreamCoder Base/Instruct full runs completed on GPU `2` and GPU `3`.

Outputs:

- Base: `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`
- Instruct: `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `Dream-org/Dream-Coder-v0-Base-7B` | official-canvas cal_lite baseline | `825/1033` | `79.86%` | baseline | `3.7847` | n/a |
| `Dream-org/Dream-Coder-v0-Base-7B` | LCAL official bounded repair | `832/1033` | `80.54%` | `+7` tasks / `+0.68pp` | `3.7763` | `27/20/805/181` |
| `Dream-org/Dream-Coder-v0-Instruct-7B` | official-canvas cal_lite baseline | `848/1033` | `82.09%` | baseline | `3.8657` | n/a |
| `Dream-org/Dream-Coder-v0-Instruct-7B` | LCAL official bounded repair | `834/1033` | `80.74%` | `-14` tasks / `-1.36pp` | `3.8472` | `21/35/813/164` |

Bucket summary:

| Model | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| DreamCoder-Base candidate | `91.22%` | `78.46%` | `53.75%` | `32.20%` | `14.81%` |
| DreamCoder-Base baseline | `90.18%` | `80.51%` | `48.75%` | `32.20%` | `14.81%` |
| DreamCoder-Instruct candidate | `88.24%` | `86.15%` | `60.00%` | `30.51%` | `25.93%` |
| DreamCoder-Instruct baseline | `89.58%` | `86.67%` | `66.25%` | `32.20%` | `18.52%` |

Repair diagnostics: Base triggered repair on `115/1033 = 11.13%` rows with only `7.83%` true-long precision among triggers; Instruct triggered on `126/1033 = 12.20%` rows with `7.94%` true-long precision. This confirms that the current trigger family is still mostly not a precise true-long detector.

Interpretation: DreamCoder-Base is a small local same-backbone positive result, but not a strong claim upgrade because the effect is only `+7` tasks, pairwise losses remain (`20`), and long buckets do not improve. DreamCoder-Instruct is negative transfer evidence and should not be used to claim cross-backbone robustness.

Literature positioning: the Base result is above CAL's DreamCoder-Base anchors (`70.2` average, `76.2` best shown) and below LR-DLLM DreamCoder-7B `81.6`; DreamOn DreamCoder `92.1` is training-based. These are suggestive anchors, not protocol-matched evidence. There is no clean literature row for this exact local DreamCoder-Instruct checkpoint.

Detailed brief: `docs/paper_agent/experiments/20260609_1231_full_dreamcoder_parallel_lcal_official_bounded_repair.md`.

## Dream-7B Official-Canvas Local Pair

The sandbox-outside Dream-7B local same-backbone pair completed on GPU `2` and GPU `3`.

Outputs:

- Baseline: `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_170219`
- Candidate: `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_170219`

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `Dream-org/Dream-v0-Base-7B` | official-canvas cal_lite baseline | `802/1033` | `77.64%` | baseline | `3.6494` | n/a |
| `Dream-org/Dream-v0-Base-7B` | LCAL official bounded repair | `803/1033` | `77.73%` | `+1` task / `+0.10pp` | `3.7337` | `28/27/775/203` |

Bucket summary:

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| Dream-7B candidate | `88.39%` | `77.44%` | `47.50%` | `25.42%` | `18.52%` |
| Dream-7B baseline | `87.95%` | `79.49%` | `48.75%` | `23.73%` | `11.11%` |

Repair diagnostics: the candidate triggered repair on `101/1033 = 9.78%` rows, with `60/101 = 59.41%` triggered-row pass rate and only `13/101 = 12.87%` true-long precision among triggers. Triggered rows were mostly short: `75` of `101` triggers were in oracle `<=8`.

Interpretation: Dream-7B is a near-tie/slight local positive result. The candidate gains only `+1` task, has nearly balanced pairwise wins/losses, and is slightly slower than the local baseline. Long buckets improve by `+3` total tasks across oracle `>=17`, but `9-12` and `13-16` regress by `-5` total tasks. This is useful protocol-matched evidence, but not a strong claim.

Literature positioning: candidate `77.73%` is above the LR-DLLM Dream-7B single-line anchor `76.7`, while DreamOn Dream-7B `88.6` is training-based and much higher. These are literature anchors, not protocol-matched local comparisons.

Detailed brief: `docs/paper_agent/experiments/20260609_1700_full_dream_base_parallel_baseline_candidate.md`.

## DiffuCoder-Base Official-Canvas Local Pair

The sandbox-outside DiffuCoder-Base local same-backbone pair completed on GPU `2` and GPU `3`.

Outputs:

- Baseline: `/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_192508`
- Candidate: `/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_192533`

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `apple/DiffuCoder-7B-Base` | official-canvas cal_lite baseline | `838/1033` | `81.12%` | baseline | `3.6562` | n/a |
| `apple/DiffuCoder-7B-Base` | LCAL official bounded repair | `839/1033` | `81.22%` | `+1` task / `+0.10pp` | `3.7538` | `25/24/814/170` |

Bucket summary:

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| DiffuCoder-Base candidate | `90.48%` | `83.08%` | `51.25%` | `37.29%` | `22.22%` |
| DiffuCoder-Base baseline | `89.73%` | `82.05%` | `56.25%` | `38.98%` | `25.93%` |

Repair diagnostics: candidate triggered repair on `113/1033 = 10.94%` rows, with `86/113 = 76.11%` triggered-row pass rate but only `9/113 = 7.96%` true-long precision among triggers. Triggered rows are mostly short: `89` of `113` triggers are in oracle `<=8`.

Interpretation: DiffuCoder-Base is a strong local backbone, but the bounded-repair candidate is only a near-tie/slight local positive over its own baseline. Gains are in `<=8` and `9-12`; `13-16`, `17-24`, and `25+` regress. This is useful protocol-matched evidence, but not a strong improvement claim.

Literature positioning: both local DiffuCoder rows are above CAL's DiffuCoder-Base anchors (`68.0` average, `74.8` best shown), while DreamOn DiffuCoder-7B `92.2` is training-based and much higher. These are literature anchors, not protocol-matched local comparisons.

Detailed brief: `docs/paper_agent/experiments/20260609_1925_full_diffucoder_base_parallel_baseline_candidate.md`.

## Current A6000 Checkpoint

Baseline: A6000 union control.

Environment: local A6000 environment recorded in prior result reports.

GPU set: A6000 control and candidate runs were reported as A6000 runs; future reports must name exact `CUDA_VISIBLE_DEVICES`. Future GPU experiments in this agent plan should use `CUDA_VISIBLE_DEVICES=2,3` unless the user changes the allocation.

Model: `GSAI-ML/LLaDA-8B-Base`.

Dataset: `HumanEval-SingleLineInfilling`, test split, `1033` tasks.

Command references:

- Full recovery launcher: `bash clean_scripts/resume_lcal_a6000_four_policies_offline.sh`.
- Pairwise analysis: `/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_lcal_pairwise.py ...`.
- Scoreboard generation: `/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_a6000_scoreboard_section.py`.
- Paper-agent evidence snapshot: `/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_paper_agent_evidence_snapshot.py`.

Output directories:

- Control: `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529`.
- `midcons`: `outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626`.
- `mid_precision`: `outputs_clean/full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517`.
- `true_long`: `outputs_clean/full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755`.
- `combined`: `outputs_clean/full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756`.

## Main Table

| Run | Pass | Rate | Delta vs A6000 control | `<=8` | `9-12` | `13-16` | `17-24` | `25+` | Comparison |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A6000 control | `787/1033` | `76.19%` | baseline | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |
| `midcons` | `795/1033` | `76.96%` | `+8` wins, `0` losses | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` | same-hardware |
| `mid_precision` | `787/1033` | `76.19%` | `0` wins, `0` losses | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |
| `true_long` | `787/1033` | `76.19%` | `0` wins, `0` losses | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |
| `combined` | `787/1033` | `76.19%` | `0` wins, `0` losses | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |

## Interpretation

`midcons` is the current A6000 best checkpoint. It gives a same-hardware improvement with no pairwise losses, concentrated in short-to-medium and medium buckets.

The true-long line is negative evidence. Existing true-long gates trigger zero useful changes under safety constraints, and the offline sweep found no safe heuristic rule from the current scalar result fields.

The fresh paper-agent evidence snapshot independently recomputed the central metrics from raw local `results.jsonl` files and wrote:

- `docs/paper_agent/evidence_snapshot.json`
- `docs/paper_agent/evidence_snapshot.md`

It confirms:

- `midcons` pairwise result: `8` wins, `0` losses, `+8` net.
- `midcons` long failures: `91` failed `oracle >= 17` rows.
- Under-selected failed long rows: `90/91 = 98.90%`.
- Failed long rows still ending from `base`: `71`.
- Long-underestimate sweep: `16776` evaluated rules, `0` strict viable rules.

## Probe-Curve Signal Audit

Command:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_long_signals.py
```

Tracked outputs:

- `docs/paper_agent/probe_curve_signal_audit.json`
- `docs/paper_agent/probe_curve_signal_audit.md`
- `docs/paper_agent/probe_curve_signal_audit.zh.md`

Result:

- rows with probe-curve features: `1033/1033`.
- rows with stopping traces: `0/1033`.
- evaluated single-feature thresholds: `4106`.
- strict viable thresholds: `0`.
- best threshold: `long_score_max <= 0.229253`, with `63.04%` true-long precision, `31.87%` failed-long recall, `8.70%` short-risk, and `2.17%` current-pass risk.

Interpretation: existing probe-curve scalar features are informative but not safe enough as a direct GPU policy. The next CPU step should be multivariate or learned scoring; trajectory analysis requires a trace-enabled smoke run.

## Probe-Curve Strict-Split Score Audit

Command:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_split_score.py
```

Tracked outputs:

- `docs/paper_agent/probe_curve_split_score_audit.json`
- `docs/paper_agent/probe_curve_split_score_audit.md`
- `docs/paper_agent/probe_curve_split_score_audit.zh.md`

Result:

- split discipline: `5` deterministic SHA256 task-id folds with train-thresholds only.
- rows: `1033`.
- feature_count: `24`.
- aggregate held-out trigger_count: `63`.
- true_long_precision: `47.62%`.
- failed_long_recall: `32.97%`.
- short_risk_rate: `22.22%`.
- current_pass_risk_rate: `7.94%`.
- strict_heldout_pass: `False`.

Fresh verification (2026-06-04):

- unit test: `/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_split_score.py` -> `Ran 3 tests` / `OK`.
- compile: `/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/analyze_probe_curve_split_score.py` -> exit `0`.
- audit regeneration: `/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_split_score.py` -> `strict_heldout_pass=False heldout_triggers=63 short_risk=22.22%`.
- JSON assertions: confirmed `trigger_count=63`, `short_risk_rate=0.2222`, `current_pass_risk_rate=0.0794`, `true_long_precision=0.4762`, `failed_long_recall=0.3297`, and `strict_heldout_pass=False` under `cross_validation.aggregate_heldout`.
- diff hygiene: `git diff --check` passed for the intended files.

Interpretation: the simple dependency-free multivariate probe-curve score fails the offline GPU gate. It has much higher short-risk than both the required `5%` gate and the single-feature best threshold's `8.70%` short-risk. This is negative evidence against launching a GPU smoke run from the current probe-curve fields alone.

## Paper Relevance

This supports a narrow but honest paper claim: medium-length under-selection can be repaired safely by confidence-curve agreement. It does not yet support a broad CCF-A claim or a SOTA claim.

## Next Result Needed

The next result should be one of:

- a diagnostic feature snapshot proving a stronger long-tail signal exists;
- a safer diagnostic signal, trace-enabled evidence, or a smoke GPU run showing no short-bucket regression;
- a full same-hardware run improving long buckets;
- or a rigorous negative result that justifies pivoting toward dynamic canvas or length regularization.
