# Experiment Results

更新时间：2026-06-14 02:40 CST

## 三方对比总表：论文报告值 vs 我们之前的方法 vs 当前方法

重要口径修正：下表中的“我们之前的方法 / previous local method”不是对应论文方法的本地复现，而是本项目早前已经跑出的本地方法或本地控制版本。论文报告值只作为外部 reported numbers 放在同一张表里，便于判断相对位置；它们不等同于本地同协议 baseline。

| Backbone / checkpoint | 相关论文报告值 | 我们之前的方法或本地旧方法 | 当前方法 | 当前 vs 之前 | 当前相对论文报告值的位置 |
|---|---|---:|---:|---:|---|
| `GSAI-ML/LLaDA-8B-Base` | CAL: avg `65.5`, best shown `73.6`; LR-DLLM LLaDA-8B: `69.4` | A6000 control `787/1033 = 76.19%` | 主线 `midcons` `795/1033 = 76.96%`; Route2 precision len24 `800/1033 = 77.44%`; Route2 broad len24 `801/1033 = 77.54%`; Route2 precision len32 `801/1033 = 77.54%` | `midcons +8` tasks / `+0.77pp`; Route2 precision len24 `+13` tasks / `+1.26pp`; Route2 broad len24 和 precision len32 均为 `+14` tasks / `+1.36pp` vs A6000 control | 高于 CAL best `73.6` 和 LR-DLLM `69.4`；Route2 是小幅 follow-up evidence，不是 external SOTA claim |
| `GSAI-ML/LLaDA-8B-Instruct` | CAL: avg `69.9`, best shown `76.9` | historical LCAS-v3 `817/1033 = 79.09%` | `midcons` `815/1033 = 78.90%` | `-2` tasks / `-0.19pp` | 高于 CAL best `76.9`，但低于我们之前方法 |
| `Dream-org/Dream-Coder-v0-Base-7B` | CAL: avg `70.2`, best shown `76.2`; LR-DLLM DreamCoder: `81.6`; DreamOn DreamCoder: `92.1` | official-canvas `cal_lite` `825/1033 = 79.86%` | bounded repair `832/1033 = 80.54%` | `+7` tasks / `+0.68pp` | 高于 CAL best `76.2`，低于 LR-DLLM `81.6` 和 DreamOn `92.1` |
| `Dream-org/Dream-Coder-v0-Instruct-7B` | 无精确匹配的论文 reported row | official-canvas `cal_lite` `848/1033 = 82.09%` | bounded repair `834/1033 = 80.74%` | `-14` tasks / `-1.36pp` | 不能做直接论文数值比较；本地为 negative transfer |
| `Dream-org/Dream-v0-Base-7B` | LR-DLLM Dream-7B: `76.7`; DreamOn Dream-7B: `88.6` | `cal_lite` `802/1033 = 77.64%` | bounded repair `803/1033 = 77.73%` | `+1` task / `+0.10pp` | 略高于 LR-DLLM `76.7`，低于 DreamOn `88.6` |
| `apple/DiffuCoder-7B-Base` | CAL: avg `68.0`, best shown `74.8`; DreamOn DiffuCoder: `92.2` | `cal_lite` `838/1033 = 81.12%` | bounded repair `839/1033 = 81.22%` | `+1` task / `+0.10pp` | 高于 CAL best `74.8`，低于 DreamOn `92.2` |
| `GSAI-ML/LLaDA-1.5` | LR-DLLM LLaDA-1.5: `68.9` | `cal_lite` LCAS-v3b `817/1033 = 79.09%` | bounded repair `818/1033 = 79.19%` | `+1` task / `+0.10pp` | 高于 LR-DLLM `68.9` |
| `inclusionAI/LLaDA-MoE-7B-A1B-Base` | LR-DLLM LLaDA-MoE: `71.3` | `cal_lite` LCAS-v3b `777/1033 = 75.22%` | bounded repair `801/1033 = 77.54%` | `+24` tasks / `+2.32pp` | 高于 LR-DLLM `71.3`；当前最强本地提升 |

读表方式：如果写论文 claim，可以同时报告“高于论文 reported number”和“相对我们之前方法的本地提升”。但需要在文字里说明评测协议不完全同源，避免把论文 reported number 误写成本地控制组。

## LLaDA-Base Full Trace Long-Rescue Diagnostics

Previous local method 和 current `midcons` 的 full trace collection 均已完成。以下 diagnostics 使用 trace/decode dynamics 做 trigger，oracle/pass labels 只用于 offline Gate A/B accounting；这不是新的 SOTA claim。

| Run | Output | Rows | Trace rows | Pass rate |
| --- | --- | --- | --- | --- |
| previous local method trace | /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552 | 1033 | 35257 | 769/1033 = 74.44% |
| current midcons trace | /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846 | 1033 | 35768 | 795/1033 = 76.96% |

Offline route analysis：

| Trace source | Route | Triggers | Failed-long | Short | Current-pass risk | Gate A | Gate B | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| previous local method | Route 1 trace-only detector | 0 | 0 | 0 | 0 | no | no | stop |
| previous local method | Route 2 risk-controlled rescue | 0 | 0 | 0 | 0 | no | no | stop |
| previous local method | Route 3 multi-canvas trace rerank | 0 | 0 | 0 | 0 | no | no | stop_no_trace_signal |
| current midcons | Route 1 trace-only detector | 0 | 0 | 0 | 0 | no | no | stop |
| current midcons | Route 2 risk-controlled rescue | 0 | 0 | 0 | 0 | no | no | stop |
| current midcons | Route 3 multi-canvas trace rerank | 0 | 0 | 0 | 0 | no | no | stop_no_trace_signal |

Interpretation：没有任何 route 满足 offline continuation rule。这是 diagnostic negative evidence；不应基于这批 traces 启动 route-specific GPU policy full run。

## Trace Feature Audit V2

`trace_feature_audit_v2` 是 CPU-only offline diagnostic，不报告新的 pass rate；该 audit 本身没有启动 GPU 工作。它复用上面的两个 full trace outputs，尝试用更丰富的 trace-shape、stop-reason、motif 和 model-assisted discovery family 找到可读的 long-rescue gate。最终有效输出目录为 `analysis_outputs/trace_feature_audit_v2_20260613_204721`。

Decision：`diagnostic_only`。原因是 previous trace source 出现 policy-level 候选，但 current `midcons` trace source 只达到 diagnostic-only；跨源稳定性不足以直接启动 Route 2 GPU policy runner。

| Source | Rows | True-long | Failed-long | Short | Decision |
|---|---:|---:|---:|---:|---|
| previous | 1033 | 113 | 96 | 598 | policy_candidate |
| midcons | 1033 | 113 | 91 | 598 | diagnostic_only |

Top held-out train-selected candidates：

| Source | Candidate | Decision | Fold | Train rank | Triggers | Failed-long | Short risk | Current-pass risk | Precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| previous | `confidence_first_le_0p875_AND_gap_last_le_0p84375` | policy_candidate | 1 | 2 | 21 | 10 | 4 | 4 | 0.476 |
| previous | `confidence_first_le_0p882812_AND_gap_last_le_0p84375` | policy_candidate | 4 | 3 | 25 | 11 | 5 | 3 | 0.440 |
| previous | `top1_last_le_0p859375_AND_confidence_first_le_0p875` | policy_candidate | 1 | 3 | 20 | 10 | 3 | 4 | 0.500 |
| previous | `top1_last_le_0p605469_AND_confidence_min_le_0p730469` | policy_candidate | 3 | 4 | 15 | 10 | 3 | 0 | 0.667 |
| midcons | `top1_last_le_0p667969_AND_max_remaining_plateau_steps_ge_16` | diagnostic_only | 4 | 1 | 18 | 9 | 2 | 0 | 0.500 |
| midcons | `top1_last_le_0p625_AND_max_remaining_plateau_steps_ge_16` | diagnostic_only | 3 | 1 | 12 | 8 | 2 | 0 | 0.667 |

Interpretation：v2 说明 trace features 并非完全无信号，特别是 low `top1_last`、low confidence/gap、late plateau 这类候选能抓到 failed-long rows。但固定规则做跨源/全量 transfer 时 short-risk 仍偏高，因此该 audit 的原始结论是：证据应作为下一轮更严格 gate/smoke 设计依据，而不是自动启动 full GPU policy run。后续在用户明确要求继续推进后，额外完成了两条 Route 2 full follow-up runs，见下一节。

## LLaDA-Base Route2 Trace-Gated Long Rescue Full Runs

用户明确偏好 full run 后，已在 GPU `2/3` 上完成两条 Route2 trace-gated long-rescue full policy runs。之后用户要求 `len32` follow-up 必须只使用 GPU3，已完成一条干净的 GPU3-only full run。策略只用 inference-time trace/decode features 触发 fixed rescue；oracle/verifier labels 只用于离线统计，不参与选择 primary 或 rescue output。

输出目录：

- Broad：`/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`
- Precision：`/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`
- Precision len32：`/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`
- 对照 baseline：current `midcons` trace run `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`

验证：三条日志均以 `COMMAND_EXIT_CODE=0` 结束；三条 `results.jsonl` 均有 `1033` 行；Broad 有 `39740` 条 step traces，Precision len24 有 `38872` 条 step traces，Precision len32 有非空 `step_traces.jsonl`；三个输出目录均有 `summary.json`。

| Run | Pass | Rate | Delta vs current `midcons` | Avg sec incl. probe |
|---|---:|---:|---:|---:|
| current `midcons` baseline | `795/1033` | `76.96%` | baseline | n/a |
| Route2 broad len24 | `801/1033` | `77.54%` | `+6` tasks / `+0.58pp` | `5.0852` |
| Route2 precision len24 | `800/1033` | `77.44%` | `+5` tasks / `+0.48pp` | `5.0945` |
| Route2 precision len32 | `801/1033` | `77.54%` | `+6` tasks / `+0.58pp` | `5.4622` |

Trigger / pairwise：

| Policy | Triggers | Trigger pass | True-long precision | Pairwise W/L/TP/TF vs `midcons` | Short triggers | Primary-pass risk |
|---|---:|---:|---:|---:|---:|---:|
| Route2 broad len24 | `73` (`7.07%`) | `9.59%` | `53.42%` | `7/1/794/231` | `10` | `1` |
| Route2 precision len24 | `57` (`5.52%`) | `8.77%` | `61.40%` | `5/0/795/233` | `6` | `0` |
| Route2 precision len32 | `57` (`5.52%`) | `10.53%` | `61.40%` | `6/0/795/232` | `6` | `0` |

Oracle bucket pass rates：

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| current `midcons` baseline | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` |
| Route2 broad len24 | `89.97%` | `79.31%` | `61.11%` | `23.17%` | `16.13%` |
| Route2 precision len24 | `90.13%` | `78.88%` | `61.11%` | `21.95%` | `16.13%` |
| Route2 precision len32 | `90.30%` | `79.31%` | `58.89%` | `23.17%` | `16.13%` |

Bucket pairwise vs current `midcons`：

| Policy | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| Route2 broad len24 | `W1/L1/net+0` | `W2/L0/net+2` | `W2/L0/net+2` | `W2/L0/net+2` | `W0/L0/net+0` |
| Route2 precision len24 | `W1/L0/net+1` | `W1/L0/net+1` | `W2/L0/net+2` | `W1/L0/net+1` | `W0/L0/net+0` |
| Route2 precision len32 | `W2/L0/net+2` | `W2/L0/net+2` | `W0/L0/net+0` | `W2/L0/net+2` | `W0/L0/net+0` |

Long-failure coverage：

| Policy | Failed-long total | Failed-long triggered | Rescue wins in failed-long | Triggered but still fail | Failed-long not triggered |
|---|---:|---:|---:|---:|---:|
| Route2 broad len24 | `91` | `39` | `2` | `37` | `52` |
| Route2 precision len24 | `91` | `35` | `1` | `34` | `56` |
| Route2 precision len32 | `91` | `35` | `2` | `33` | `56` |

Interpretation：Route2 full run 是小幅正收益，而不是长长度瓶颈被解决。Broad len24 相比 current `midcons` 净增 `+6` tasks，但有 `1` 个 primary-pass loss 和更多 short triggers；Precision len24 净增 `+5` tasks 且没有 observed primary-pass loss；Precision len32 净增 `+6` tasks 且没有 observed primary-pass loss，是当前更干净的 Route2 follow-up。最关键诊断是：gate 能抓到一部分 failed-long rows，但 fixed rescue 大多救不回来。Precision len32 在 oracle `17-24` 净增 `+2`，但 oracle `25+` 仍为 `0` 增益；triggered oracle `25+` 行为 `0/11` pass。下一步应分析 triggered-but-still-failed 和 missed failed-long rows，再决定是否设计 training-free adaptive rescue length、better rescue decoding 或更强 trace/probe fusion gate。

## LLaDA-MoE Local Same-Backbone Pair

完整 `inclusionAI/LLaDA-MoE-7B-A1B-Base` local baseline/candidate pair 已在 GPU `2` 和 `3` 上完成。

输出目录：

- Baseline：`/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719`
- Candidate：`/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740`
- Pairwise analysis：`analysis_outputs/lladamoe_full_pair_20260611_1438`

验证：

- Baseline 日志以 `COMMAND_EXIT_CODE="0"` 结束。
- Candidate 日志以 `COMMAND_EXIT_CODE="0"` 结束。
- 两个 `results.jsonl` 都有 `1033` 个 valid rows，`0` 个 malformed rows。
- 两个输出目录都有 `summary.json`。

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `inclusionAI/LLaDA-MoE-7B-A1B-Base` | `cal_lite` LCAS-v3b local baseline | `777/1033` | `75.22%` | baseline | `8.7025` | n/a |
| `inclusionAI/LLaDA-MoE-7B-A1B-Base` | LCAL official bounded repair | `801/1033` | `77.54%` | `+24` tasks / `+2.32pp` | `10.6107` | `31/7/770/225` |

按 oracle length 分桶的 pairwise：

| Oracle bucket | Count | Wins | Losses | Net | Baseline pass | Candidate pass |
|---|---:|---:|---:|---:|---:|---:|
| `<=8` | `598` | `16` | `7` | `+9` | `88.46%` | `89.97%` |
| `9-12` | `232` | `5` | `0` | `+5` | `77.59%` | `79.74%` |
| `13-16` | `90` | `4` | `0` | `+4` | `57.78%` | `62.22%` |
| `17-24` | `82` | `6` | `0` | `+6` | `14.63%` | `21.95%` |
| `25+` | `31` | `0` | `0` | `0` | `12.90%` | `12.90%` |

Repair diagnostics：candidate 在 `104/1033 = 10.07%` 行触发 official repair，在 `15/1033 = 1.45%` 行触发 `official_long_suspicion`，在 `23/1033 = 2.23%` 行触发 `official_mid_rescue`。True-long trigger precision 仍偏弱：`official_repair_true_long_precision = 11.54%`，`official_long_suspicion_true_long_precision = 40.00%`，`official_mid_rescue_true_long_precision = 13.04%`。Candidate 在 true-long rows 上仍强烈 under-select：`under_select_rate_17plus = 91.15%`。

Interpretation：这是目前最清楚的 local transfer 正结果。不同于 Dream-7B、DiffuCoder-Base、LLaDA-1.5 的 near-tie，LLaDA-MoE 净增 `+24` tasks，只有 `7` 个 losses，并且所有 oracle buckets 都是正增益或持平。代价是更慢：`10.6107s` vs `8.7025s` per sample including probe。这个结果可以支持 LLaDA-MoE 上的 local protocol-matched improvement claim，但仍不能称为 external SOTA，因为 DreamOn 是 training-based 且在 Dream/DiffuCoder/DreamCoder 上显著更高，LR-DLLM/CAL 也不是本地 protocol-matched controls。

文献位置：candidate `77.54%` 高于 LR-DLLM LLaDA-MoE single-line anchor `71.3` 和其 reported baseline anchor `48.8`，但这仍只是 suggestive anchor，不是 protocol-matched comparison。

详细记录：`docs/paper_agent/experiments/20260611_1126_lladamoe_full_pair.md`。

## LLaDA-1.5 Local Same-Backbone Pair

完整 `GSAI-ML/LLaDA-1.5` local baseline/candidate pair 已在共享 GPU `2` 和 `3` 上完成。

输出目录：

- Baseline：`/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_cal_lite_lcas_v3b_gpu2_shared_20260610_172705`
- Candidate：`/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_lcal_official_bounded_repair_gpu3_shared_20260610_172720`

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `GSAI-ML/LLaDA-1.5` | `cal_lite` LCAS-v3b local baseline | `817/1033` | `79.09%` | baseline | `5.4224` | n/a |
| `GSAI-ML/LLaDA-1.5` | LCAL official bounded repair | `818/1033` | `79.19%` | `+1` task / `+0.10pp` | `6.6453` | `18/17/800/198` |

Bucket summary：

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| LLaDA-1.5 candidate | `90.47%` | `83.19%` | `67.78%` | `21.95%` | `16.13%` |
| LLaDA-1.5 baseline | `91.47%` | `83.19%` | `64.44%` | `18.29%` | `12.90%` |

Bucket pairwise：candidate 在 oracle `<=8` 净损失 `6` 个任务，在 `9-12` 持平，并在 `13-16`、`17-24`、`25+` 分别净增 `+3`、`+3`、`+1`。

Repair diagnostics：candidate 在 `110/1033 = 10.65%` 行触发 official repair。Triggered-row true-long precision 只有 `12/110 = 10.91%`；`110` 个 triggers 中 `82` 个是 oracle `<=8`。其中 `official_long_suspicion` 尤其嘈杂：`33` 个 triggers 中 `29` 个是 oracle `<=8`。

Interpretation：这是 near-tie / slight local positive result，不是强 claim upgrade。Candidate 挽回了少量 medium/long 任务，但代价是 short-bucket regression 和更高耗时：`6.6453s` vs `5.4224s` per sample including probe。该结果支持当前主要诊断：true-long recovery 仍弱，当前 official-CAL trigger family 不是精确的 true-long detector。

文献位置：本地 LLaDA-1.5 两条结果高于 LR-DLLM 的 LLaDA-1.5 single-line anchor `68.9` 和其 reported LLaDA-1.5 baseline anchor `48.8`，但这些是 literature anchors，不是 protocol-matched local comparisons。

详细记录：`docs/paper_agent/experiments/20260610_1735_full_llada15_parallel_baseline_candidate.md`。

## LLaDA-Instruct Cross-Model Result

用户确认的 `GSAI-ML/LLaDA-8B-Instruct + midcons` full run 已在 GPU `2,3` 上完成。

输出目录：

`/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`

| Model | Run | Pass | Rate | Avg sec/sample incl. probe | Comparison |
|---|---|---:|---:|---:|---|
| `GSAI-ML/LLaDA-8B-Instruct` | historical LCAS-v3 | `817/1033` | `79.09%` | `6.8661` | same-backbone baseline |
| `GSAI-ML/LLaDA-8B-Instruct` | current `midcons` bounded repair | `815/1033` | `78.90%` | `4.1766` | 相比 baseline `-2` tasks |

Pairwise comparison：`17` wins、`19` losses、`798` tie-pass、`199` tie-fail。损失主要集中在 short buckets：oracle `<=8` 有 `17` 个 losses，`9-12` 有 `2` 个 losses；candidate 在 oracle `>=17` 有 `4` 个 long-bucket wins。

Interpretation：这是 negative transfer evidence。`midcons` 仍是有价值的 LLaDA-Base checkpoint，但不能干净迁移到 LLaDA-Instruct，因此不能据此升级论文 claim。下一阶段必须让每个 literature backbone 都和自己的 baseline 对比。

计划文档：`docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`。

## DreamCoder Official-Canvas Cross-Backbone Result

沙箱外 DreamCoder Base/Instruct full runs 已在 GPU `2` 和 GPU `3` 上完成。

输出目录：

- Base：`/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`
- Instruct：`/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `Dream-org/Dream-Coder-v0-Base-7B` | official-canvas cal_lite baseline | `825/1033` | `79.86%` | baseline | `3.7847` | n/a |
| `Dream-org/Dream-Coder-v0-Base-7B` | LCAL official bounded repair | `832/1033` | `80.54%` | `+7` tasks / `+0.68pp` | `3.7763` | `27/20/805/181` |
| `Dream-org/Dream-Coder-v0-Instruct-7B` | official-canvas cal_lite baseline | `848/1033` | `82.09%` | baseline | `3.8657` | n/a |
| `Dream-org/Dream-Coder-v0-Instruct-7B` | LCAL official bounded repair | `834/1033` | `80.74%` | `-14` tasks / `-1.36pp` | `3.8472` | `21/35/813/164` |

Bucket summary：

| Model | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| DreamCoder-Base candidate | `91.22%` | `78.46%` | `53.75%` | `32.20%` | `14.81%` |
| DreamCoder-Base baseline | `90.18%` | `80.51%` | `48.75%` | `32.20%` | `14.81%` |
| DreamCoder-Instruct candidate | `88.24%` | `86.15%` | `60.00%` | `30.51%` | `25.93%` |
| DreamCoder-Instruct baseline | `89.58%` | `86.67%` | `66.25%` | `32.20%` | `18.52%` |

Repair diagnostics：Base 在 `115/1033 = 11.13%` 行触发 repair，trigger 中 true-long precision 只有 `7.83%`；Instruct 在 `126/1033 = 12.20%` 行触发，true-long precision 为 `7.94%`。这说明当前 trigger family 仍不是精确的 true-long detector。

Interpretation：DreamCoder-Base 是小幅 local same-backbone 正结果，但还不足以升级成强 claim，因为净提升只有 `+7` tasks，仍有 `20` 个 pairwise losses，且 long buckets 没有改善。DreamCoder-Instruct 是 negative transfer evidence，不能拿来宣称 cross-backbone robustness。

文献位置：Base 结果高于 CAL 的 DreamCoder-Base anchors（`70.2` average、`76.2` best shown），低于 LR-DLLM DreamCoder-7B `81.6`；DreamOn DreamCoder `92.1` 是 training-based 方法。这些只能作为 suggestive anchors，不是 protocol-matched evidence。当前 anchor table 中没有与本地 DreamCoder-Instruct checkpoint 精确匹配的文献行。

详细记录：`docs/paper_agent/experiments/20260609_1231_full_dreamcoder_parallel_lcal_official_bounded_repair.md`。

## Dream-7B Official-Canvas Local Pair

沙箱外 Dream-7B local same-backbone pair 已在 GPU `2` 和 GPU `3` 上完成。

输出目录：

- Baseline：`/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_170219`
- Candidate：`/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_170219`

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `Dream-org/Dream-v0-Base-7B` | official-canvas cal_lite baseline | `802/1033` | `77.64%` | baseline | `3.6494` | n/a |
| `Dream-org/Dream-v0-Base-7B` | LCAL official bounded repair | `803/1033` | `77.73%` | `+1` task / `+0.10pp` | `3.7337` | `28/27/775/203` |

Bucket summary：

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| Dream-7B candidate | `88.39%` | `77.44%` | `47.50%` | `25.42%` | `18.52%` |
| Dream-7B baseline | `87.95%` | `79.49%` | `48.75%` | `23.73%` | `11.11%` |

Repair diagnostics：candidate 在 `101/1033 = 9.78%` 行触发 repair，triggered-row pass rate 为 `60/101 = 59.41%`，trigger 中 true-long precision 只有 `13/101 = 12.87%`。触发行大多是 short：`101` 个 triggers 中有 `75` 个位于 oracle `<=8`。

Interpretation：Dream-7B 是 near-tie / slight local positive result。candidate 只多 `+1` task，pairwise wins/losses 基本持平，且比本地 baseline 稍慢。oracle `>=17` 的 long buckets 总共多 `+3` tasks，但 `9-12` 和 `13-16` 合计少 `-5` tasks。这是有用的 protocol-matched 证据，但不是强 claim。

文献位置：candidate `77.73%` 高于 LR-DLLM Dream-7B single-line anchor `76.7`；DreamOn Dream-7B `88.6` 是 training-based 且明显更高。这些只是 literature anchors，不是 protocol-matched local comparisons。

详细记录：`docs/paper_agent/experiments/20260609_1700_full_dream_base_parallel_baseline_candidate.md`。

## DiffuCoder-Base Official-Canvas Local Pair

沙箱外 DiffuCoder-Base local same-backbone pair 已在 GPU `2` 和 GPU `3` 上完成。

输出目录：

- Baseline：`/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_192508`
- Candidate：`/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_192533`

| Model | Run | Pass | Rate | Delta vs local baseline | Avg sec/sample incl. probe | Pairwise W/L/TP/TF |
|---|---|---:|---:|---:|---:|---:|
| `apple/DiffuCoder-7B-Base` | official-canvas cal_lite baseline | `838/1033` | `81.12%` | baseline | `3.6562` | n/a |
| `apple/DiffuCoder-7B-Base` | LCAL official bounded repair | `839/1033` | `81.22%` | `+1` task / `+0.10pp` | `3.7538` | `25/24/814/170` |

Bucket summary：

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| DiffuCoder-Base candidate | `90.48%` | `83.08%` | `51.25%` | `37.29%` | `22.22%` |
| DiffuCoder-Base baseline | `89.73%` | `82.05%` | `56.25%` | `38.98%` | `25.93%` |

Repair diagnostics：candidate 在 `113/1033 = 10.94%` 行触发 repair，triggered-row pass rate 为 `86/113 = 76.11%`，但 triggered rows 中 true-long precision 只有 `9/113 = 7.96%`。触发行大多是 short：`113` 个 triggers 中有 `89` 个位于 oracle `<=8`。

Interpretation：DiffuCoder-Base 是很强的本地 backbone，但 bounded-repair candidate 相比自己的 baseline 只是 near-tie / slight local positive。收益来自 `<=8` 和 `9-12`；`13-16`、`17-24`、`25+` 均有回退。这是有用的 protocol-matched 证据，但不是强 improvement claim。

文献位置：本地两条 DiffuCoder 结果均高于 CAL 的 DiffuCoder-Base anchors（`68.0` average、`74.8` best shown）；DreamOn DiffuCoder-7B `92.2` 是 training-based 且明显更高。这些只是 literature anchors，不是 protocol-matched local comparisons。

详细记录：`docs/paper_agent/experiments/20260609_1925_full_diffucoder_base_parallel_baseline_candidate.md`。

## 当前 A6000 Checkpoint

Baseline：A6000 union control。

Environment：prior result reports 中记录的本地 A6000 environment。

GPU set：A6000 control 与 candidate runs 已报告为 A6000 runs；未来报告必须写明精确 `CUDA_VISIBLE_DEVICES`。除非用户更改 allocation，本 agent plan 的未来 GPU experiments 应使用 `CUDA_VISIBLE_DEVICES=2,3`。

Model：`GSAI-ML/LLaDA-8B-Base`。

Dataset：`HumanEval-SingleLineInfilling`，test split，`1033` tasks。

Command references：

- Full recovery launcher：`bash clean_scripts/resume_lcal_a6000_four_policies_offline.sh`。
- Pairwise analysis：`/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_lcal_pairwise.py ...`。
- Scoreboard generation：`/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_a6000_scoreboard_section.py`。
- Paper-agent evidence snapshot：`/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_paper_agent_evidence_snapshot.py`。

Output directories：

- Control：`outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529`。
- `midcons`：`outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626`。
- `mid_precision`：`outputs_clean/full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517`。
- `true_long`：`outputs_clean/full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755`。
- `combined`：`outputs_clean/full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756`。

## Main Table

| Run | Pass | Rate | Delta vs A6000 control | `<=8` | `9-12` | `13-16` | `17-24` | `25+` | Comparison |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A6000 control | `787/1033` | `76.19%` | baseline | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |
| `midcons` | `795/1033` | `76.96%` | `+8` wins, `0` losses | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` | same-hardware |
| `mid_precision` | `787/1033` | `76.19%` | `0` wins, `0` losses | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |
| `true_long` | `787/1033` | `76.19%` | `0` wins, `0` losses | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |
| `combined` | `787/1033` | `76.19%` | `0` wins, `0` losses | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |

## Interpretation

`midcons` 是当前 A6000 best checkpoint。它在 same-hardware comparison 下带来提升且没有 pairwise losses，收益集中于 short-to-medium 和 medium buckets。

True-long 结果是 negative evidence。现有 true-long gates 在 safety constraints 下没有触发有用变化，offline sweep 也没有从当前 scalar result fields 中找到安全 heuristic rule。

最新 paper-agent evidence snapshot 从本地 raw `results.jsonl` files 独立重新计算了核心 metrics，并写入：

- `docs/paper_agent/evidence_snapshot.json`
- `docs/paper_agent/evidence_snapshot.md`

它确认：

- `midcons` pairwise result：`8` wins、`0` losses、`+8` net。
- `midcons` long failures：`91` 个 failed `oracle >= 17` rows。
- Under-selected failed long rows：`90/91 = 98.90%`。
- 仍从 `base` 结束的 failed long rows：`71`。
- Long-underestimate sweep：评估 `16776` 条 rules，`0` 条 strict viable rules。

## Probe-Curve Signal Audit

Command：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_long_signals.py
```

Tracked outputs：

- `docs/paper_agent/probe_curve_signal_audit.json`
- `docs/paper_agent/probe_curve_signal_audit.md`
- `docs/paper_agent/probe_curve_signal_audit.zh.md`

Result：

- rows with probe-curve features：`1033/1033`。
- rows with stopping traces：`0/1033`。
- evaluated single-feature thresholds：`4106`。
- strict viable thresholds：`0`。
- best threshold：`long_score_max <= 0.229253`，对应 `63.04%` true-long precision、`31.87%` failed-long recall、`8.70%` short-risk 和 `2.17%` current-pass risk。

Interpretation：现有 probe-curve scalar features 有信息量，但作为直接 GPU policy 仍不够安全。下一步 CPU 工作应转向 multivariate 或 learned scoring；trajectory analysis 需要 trace-enabled smoke run。

## Probe-Curve Strict-Split Score Audit

Command：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_split_score.py
```

Tracked outputs：

- `docs/paper_agent/probe_curve_split_score_audit.json`
- `docs/paper_agent/probe_curve_split_score_audit.md`
- `docs/paper_agent/probe_curve_split_score_audit.zh.md`

Result：

- split discipline：`5` 个 deterministic SHA256 task-id folds，且 thresholds 只在 train folds 上选择。
- rows：`1033`。
- feature_count：`24`。
- aggregate held-out trigger_count：`63`。
- true_long_precision：`47.62%`。
- failed_long_recall：`32.97%`。
- short_risk_rate：`22.22%`。
- current_pass_risk_rate：`7.94%`。
- strict_heldout_pass：`False`。

Fresh verification（2026-06-04）：

- unit test：`/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_split_score.py` -> `Ran 3 tests` / `OK`。
- compile：`/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/analyze_probe_curve_split_score.py` -> exit `0`。
- audit regeneration：`/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_split_score.py` -> `strict_heldout_pass=False heldout_triggers=63 short_risk=22.22%`。
- JSON assertions：确认 `cross_validation.aggregate_heldout` 下的 `trigger_count=63`、`short_risk_rate=0.2222`、`current_pass_risk_rate=0.0794`、`true_long_precision=0.4762`、`failed_long_recall=0.3297`、`strict_heldout_pass=False`。
- diff hygiene：`git diff --check` 对 intended files 通过。

Interpretation：简单 dependency-free multivariate probe-curve score 未通过 offline GPU gate。它的 short-risk 明显高于要求的 `5%` gate，也高于 single-feature 最佳 threshold 的 `8.70%` short-risk。因此，这是一条反对仅凭当前 probe-curve fields 启动 GPU smoke run 的 negative evidence。

## Paper Relevance

这支持一个狭窄但诚实的 paper claim：medium-length under-selection 可以通过 confidence-curve agreement 安全修复。它还不支持 broad CCF-A claim 或 SOTA claim。

## Next Result Needed

下一项结果应是以下之一：

- diagnostic feature snapshot 证明存在更强 long-tail signal；
- 更安全的 diagnostic signal、trace-enabled evidence，或显示没有 short-bucket regression 的 smoke GPU run；
- full same-hardware run 改善 long buckets；
- 或严谨 negative result，支撑转向 dynamic canvas 或 length regularization。
