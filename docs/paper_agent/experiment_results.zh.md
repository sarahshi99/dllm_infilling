# Experiment Results

更新时间：2026-07-01 15:40 CST

## Codex Phase 0 审计说明（2026-07-02）

本轮未新增 pass-rate 结果，也未启动 GPU。Codex 已审计当前结果链、关键 runner 的 gate/action/selector 和可复现性缺口；详见 `docs/paper_agent/codex_repository_audit.zh.md`。

审计后结果口径不变：V6 short override `802/1033 = 77.64%` 是当前 LLaDA-Base 最高 full result；Route2 precision len32 `801/1033 = 77.54%` 是更干净的低风险 rescue 证据；V7/V8 是比例放长负结果；这些仍然都来自同一 `HumanEval-SingleLineInfilling/test` 1033 rows 上的多轮探索，不能替代 held-out controller evaluation。

## 三方对比总表：论文报告值 vs 我们之前的方法 vs 当前方法

重要口径修正：下表中的“我们之前的方法 / previous local method”不是对应论文方法的本地复现，而是本项目早前已经跑出的本地方法或本地控制版本。论文报告值只作为外部 reported numbers 放在同一张表里，便于判断相对位置；它们不等同于本地同协议 baseline。

| Backbone / checkpoint | 相关论文报告值 | 我们之前的方法或本地旧方法 | 当前方法 | 当前 vs 之前 | 当前相对论文报告值的位置 |
|---|---|---:|---:|---:|---|
| `GSAI-ML/LLaDA-8B-Base` | CAL: avg `65.5`, best shown `73.6`; LR-DLLM LLaDA-8B: `69.4` | A6000 control `787/1033 = 76.19%` | 主线 `midcons` `795/1033 = 76.96%`; Route2 precision len24 `800/1033 = 77.44%`; Route2 broad len24 `801/1033 = 77.54%`; Route2 precision len32 `801/1033 = 77.54%`; V5.1 anchor m002/m010 `801/1033 = 77.54%`; V6 short override `802/1033 = 77.64%`; V7 proportional widening `792/1033 = 76.67%`，negative；V8 proportional CAL score `786/781/782`，均 negative | `midcons +8` tasks / `+0.77pp`; Route2 precision len24 `+13` tasks / `+1.26pp`; Route2 broad len24、precision len32、V5.1 anchor 均为 `+14` tasks / `+1.36pp` vs A6000 control；V6 为 `+15` tasks / `+1.45pp` vs A6000 control，且 vs Route2 precision len32 为 `1/0/801/231`；V7 相对 `midcons` 为 `-3` tasks，pairwise `1/4/791/237`；V8a/b/c 相对 `midcons` 分别为 `-9/-14/-13` tasks | 高于 CAL best `73.6` 和 LR-DLLM `69.4`；Route2/V5/V6 是小幅 follow-up evidence，不是 external SOTA claim；V7/V8 是比例放长路线的负结果 |
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

## LLaDA-Base Route2 V5.1 Rescue-Quality Full Runs

V5.1 是 Route2 precision `len32` 之后的 rescue-quality follow-up。它不重新挖 trigger，而是在 Route2 已触发的 `57` 行上生成多个 deterministic candidates：`len24_s64`、`len32_s64`、`len32_s96`。策略边界是 training-free / inference-time / verifier-free；oracle/pass labels 只用于离线统计。V5.1 的 `anchor_len32_confidence` selector 默认保护 `len32_s64`，只允许 `len32_s96` 在 inference-visible score 超过 margin 时替换；`len24_s64` 保留为 diagnostic-only，避免重现 smoke 中较短候选误伤已知 Route2 win 的问题。

输出目录：

- margin `0.02`：`outputs_clean/full_route2_rescue_quality_v5_anchor_m002_gpu2_20260618_175641`
- margin `0.10`：`outputs_clean/full_route2_rescue_quality_v5_anchor_m010_gpu3_20260618_175642`

验证：两条 GPU jobs exit code 均为 `0`；两条 `results.jsonl` 均为 `1033` 行；两条均生成 `summary.json`、`step_traces.jsonl`、`candidate_upper_bound.csv`。

| Run | Pass | Rate | Avg sec incl. probe | Triggered | Selected candidates | Candidate upper-bound on triggered | Pairwise vs `midcons` | Pairwise vs Route2 precision `len32` |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| V5.1 anchor margin `0.02` | `801/1033` | `77.54%` | `4.9208` | `57` | `primary=976`, `len32_s64=56`, `len32_s96=1` | `9/57 = 15.79%` | `6/0/795/232` | `0/0/801/232` |
| V5.1 anchor margin `0.10` | `801/1033` | `77.54%` | `4.9618` | `57` | `primary=976`, `len32_s64=57` | `9/57 = 15.79%` | `6/0/795/232` | `0/0/801/232` |

Oracle bucket pass rates：

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| V5.1 anchor margin `0.02` | `540/598 = 90.30%` | `184/232 = 79.31%` | `53/90 = 58.89%` | `19/82 = 23.17%` | `5/31 = 16.13%` |
| V5.1 anchor margin `0.10` | `540/598 = 90.30%` | `184/232 = 79.31%` | `53/90 = 58.89%` | `19/82 = 23.17%` | `5/31 = 16.13%` |

Triggered-row diagnostic：

| Oracle bucket | Triggered | Selected pass | Oracle upper-bound pass |
|---|---:|---:|---:|
| `<=8` | `6` | `2` | `2` |
| `9-12` | `6` | `2` | `2` |
| `13-16` | `10` | `0` | `2` |
| `17-24` | `24` | `2` | `3` |
| `25+` | `11` | `0` | `0` |

Interpretation：V5.1 没有超过 Route2 precision `len32`，因此不是新的 pass-rate claim。它的价值是诊断性的：anchor selector 成功保护了已知 Route2 len32 行为，`0` losses vs Route2 precision `len32`；但 candidate upper-bound 只有 `9/57`，selected policy 只有 `6/57` triggered pass，说明主要瓶颈仍是 rescue candidate 生成质量。唯一一次 `len32_s96` 替换发生在 `SingleLineInfilling/HumanEval/122/L0`，但所有候选均失败。另有三行 upper-bound-only rows 需要 `len24_s64` 才能通过：`HumanEval/7/L0`、`HumanEval/11/L6`、`HumanEval/128/L2`。这提示短候选 override 有潜力，但必须先设计严格保护规则，不能直接全量放开 `len24_s64`。下一步不应盲目继续 GPU full run，而应 CPU-first 设计一个 reviewer-readable shorter-candidate override 或更强 rescue-generation candidate family。

## LLaDA-Base Route2 V6 Short-Override Full Run

V6 是 V5.1 之后的 conservative selector polish：默认保持 `len32_s64` anchor，只在 trace gap/top1 同时强烈支持 `len24_s64` 时允许短候选 override。它仍是 training-free / inference-time / verifier-free；oracle/pass labels 只用于离线统计。

输出目录：

- `outputs_clean/full_route2_v6_short_override_gpu2_20260620_124754`

验证：

- full run exit code 为 `0`。
- `results.jsonl` 为完整 `1033` 行。
- 生成 `summary.json`、`step_traces.jsonl`、`candidate_upper_bound.csv`。

| Run | Pass | Rate | Avg sec incl. probe | Triggered | Selected candidates | Pairwise vs `midcons` | Pairwise vs Route2 precision `len32` |
|---|---:|---:|---:|---:|---|---:|---:|
| V6 short override | `802/1033` | `77.64%` | `4.8768` | `57` | `primary=976`, `len32_s64=56`, `len24_s64=1` | `7/0/795/231` | `1/0/801/231` |

Oracle bucket pass rates：

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| V6 short override | `540/598 = 90.30%` | `184/232 = 79.31%` | `53/90 = 58.89%` | `20/82 = 24.39%` | `5/31 = 16.13%` |

唯一的 `len24_s64` override：

| task_id | oracle length | selected length | pass | role |
|---|---:|---:|---|---|
| `SingleLineInfilling/HumanEval/11/L6` | `22` | `24` | true | V6 相对 Route2 precision `len32` 的唯一 win |

Interpretation：V6 满足 full-run success criteria：相对 Route2 precision `len32` 为 `+1` task、`0` losses，相对 `midcons` 为 `+7` tasks、`0` losses。这个结果可以作为 reviewer-readable 的小幅 selector polish evidence，但不是长长度问题的根本解决。最关键的诊断没有变：candidate upper-bound 仍只有 `9/57`，oracle `25+` bucket 仍为 `5/31 = 16.13%`，说明 selector-only 方向的剩余收益很小。下一步应该转向更强的 rescue candidate generation / rescue decoding，而不是继续做类似阈值选择器 sweep。

## LLaDA-Base V8 Proportional CAL Score Full Runs

V8 是对用户“按比例放长长度估计”想法的忠实公式级实验。不同于 V7 的 post-hoc near-best 长候选覆盖，V8 直接修改 CAL-like length score：

```text
score(L) = raw_score(L) * L^(alpha + beta * log(max(min(L, cap) / ref, 1)))
```

这意味着长度越长，额外指数奖励越大；`cap` 只限制额外奖励的计算，不是禁止候选长度超过该值。

输出目录：

- V8a：`outputs_clean/full_v8a_propcal_beta002_gpu1_20260701_102654`
- V8b：`outputs_clean/full_v8b_propcal_beta004_gpu1_20260701_120525`
- V8c：`outputs_clean/full_v8c_propcal_beta004_cap32_gpu1_20260701_134849`

所有 V8 full runs 均在 GPU `1` 串行完成，均写出完整 `1033` 行和 `summary.json`。

| Run | Setting | Pass | Rate | Pairwise vs `midcons` | Short losses | Long wins |
|---|---|---:|---:|---:|---:|---:|
| current `midcons` | baseline | `795/1033` | `76.96%` | baseline | n/a | n/a |
| Route2 precision `len32` | trace-gated rescue | `801/1033` | `77.54%` | `6/0/795/232` | `0` | `2` |
| V6 short override | selector polish | `802/1033` | `77.64%` | `7/0/795/231` | `0` | `2` |
| V7 proportional widening | post-hoc widening | `792/1033` | `76.67%` | `1/4/791/237` | `2` | `0` |
| V8a proportional score | beta `0.02`, no cap | `786/1033` | `76.09%` | `3/12/783/235` | `7` | `2` |
| V8b proportional score | beta `0.04`, no cap | `781/1033` | `75.61%` | `7/21/774/231` | `13` | `3` |
| V8c proportional score | beta `0.04`, reward cap `32` | `782/1033` | `75.70%` | `6/19/776/232` | `11` | `2` |

Oracle bucket pass rates：

| Run | `<=8` | `9-12` | `13-16` | `17-24` | `25+` |
|---|---:|---:|---:|---:|---:|
| V8a | `88.80%` | `77.16%` | `57.78%` | `21.95%` | `19.35%` |
| V8b | `87.79%` | `76.72%` | `58.89%` | `21.95%` | `22.58%` |
| V8c | `88.13%` | `76.72%` | `58.89%` | `21.95%` | `19.35%` |

Interpretation：V8 说明“全局比例式放长”确实能带来少量 long wins，V8b 的 long wins 到 `3`；但 short/medium regressions 更大，导致三条 full runs 全部低于 `midcons`、Route2、V6 和 V7。V8c 的 reward cap 没有救回整体结果，而且由于 cap 不是 hard candidate cap，结果里仍会出现 `40/48` 候选。结论是：直接把比例奖励放进 CAL-like score 不是当前主线；如果继续比例思想，必须先有 under-selection detector 或风险 guard，而不是全局加长。

## Route2 Error Analysis Discovery V3

`route2_error_analysis` 是 CPU-only 诊断，用来解释 Route2 precision `len32` 为什么只有小幅增益，以及下一轮 Discovery layer 应该优化 gate recall、rescue generation/selection，还是 adaptive length。它不启动 GPU，也不报告新的 pass-rate claim。

输出目录：`analysis_outputs/route2_error_analysis_20260617_165806`；报告：`analysis_outputs/route2_error_analysis_20260617_165806/report.md`。

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

Interpretation：这支持继续找信号，而不是放弃 true-long rescue。Route2 的 `6` 个 wins 都来自 triggered rescue，说明 gate 有真实信号；但 `33` 个 triggered failed-long 中有 `31` 个 rescue length 已经不小于 oracle length，说明盲目继续加长 canvas 不是默认解。与此同时，还有 `56` 个 failed-long rows 完全没被 trigger，说明 gate recall 仍然不足。因此下一步应做两条 CPU-first 设计：一是分析 rescue generation/selection 为什么在长度足够时失败，二是用 probe-trace fusion 扩展 missed failed-long 的召回。

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
