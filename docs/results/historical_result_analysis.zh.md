# 历史结果分析

更新时间：2026-05-29 Asia/Shanghai

本文在不提交原始 `outputs_clean/` artifact 的前提下，总结有保留价值的历史实验结果。机器可读索引为 `docs/results/run_registry.json`，英文人类可读索引为 `docs/results/run_registry.md`。

## 盘点

当前 registry 包含 `49` 个有意义的本地 runs。默认排除 smoke runs 和缺少 summary 的 incomplete runs。

原始输出保留在本地：

- `/home/shx/projects/dllm_infilling/outputs_clean`
- `/home/shx/projects/dllm_infilling/outputs_clean/202604`
- `/home/shx/projects/dllm_infilling/model_generalization_runs`

## 最佳本地 LLaDA-Base 演进线

| 阶段 | 代表 run | Pass | 重要性 |
|---|---|---:|---|
| Oracle 上界参考 | `full_oracle_sl_20260410_145452` | `900/1033 = 87.12%` | 显示长度选择是主要瓶颈；已知缺失长度时生成成功率明显更高。 |
| Fixed-length baseline | `full_fixed_sl_20260410_161152` | `478/1033 = 46.27%` | 固定 canvas 长度在 single-line infilling 上不具竞争力。 |
| 早期 CAL-lite | `full_cal_lite_sl_20260414_164141` | `550/1033 = 53.24%` | 初始 confidence probing 超过 fixed length，但长度选择错误仍多。 |
| CAL-lite v1 | `full_cal_lite_v1_sl_20260415_194824` | `718/1033 = 69.51%` | 更好的 mask-length probing 带来主要跃升。 |
| CAL-lite v2 alpha sweep | `full_cal_lite_v2_alpha_008_sl_20260417_131256` | `770/1033 = 74.54%` | 早期 alpha family 的最佳点，后来被 superseded。 |
| LCAS/LCAL family | `full_lcas_v3b_alpha006_compact_sl_20260429_180958` | `769/1033 = 74.44%` | Adaptive stopping 稳定，但单独不足以超过 bounded repair。 |
| Official bounded repair | `full_lcal_official_bounded_repair_s3_off6_11_delta1_8_gpus01_20260515_163902` | `784/1033 = 75.90%` | 修复短长度 under-selection 成为可靠路径。 |
| Union checkpoint | `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus23_20260519_175826` | `787/1033 = 76.19%` | A6000 前稳定全局 checkpoint。 |
| A6000 control | `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529` | `787/1033 = 76.19%` | 与旧 union 同分，是可靠 A6000 baseline。 |
| A6000 midcons | `full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626` | `795/1033 = 76.96%` | 当前最佳同硬件 checkpoint；相对 A6000 control 为 `+8` wins、`0` losses。 |

## 历史结果说明了什么

项目已经越过了简单 fixed-length decoding 阶段。一致模式是：

1. 更好的长度选择带来最大收益。
2. 有界 short/medium repair 比宽松 long repair 更安全。
3. Conservative mid rescue 能改善 `9-12` 和 `13-16` buckets，同时没有明显 aggregate short damage。
4. 与 oracle run 的剩余差距主要来自长样本 under-selection。

当前最佳 LLaDA-Base 结果 `795/1033` 仍比 oracle-length 参考少 `105` 个 pass。这既是研究机会，也是主要风险：继续堆叠手调 rescue 规则很难单独补齐这个缺口。

## 应保留为 canonical 的 runs

- `full_oracle_sl_20260410_145452`：oracle-length 上界参考。
- `full_fixed_sl_20260410_161152`：fixed-length 下界 baseline。
- `full_cal_lite_v1_sl_20260415_194824`：第一次显著 CAL-lite 提升。
- `full_cal_lite_v2_alpha_008_sl_20260417_131256`：早期 alpha sweep 最佳点。
- `full_lcas_v3b_alpha006_compact_sl_20260429_180958`：LCAS v3 参考。
- `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus23_20260519_175826`：旧 union checkpoint。
- `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529`：A6000 control。
- `full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626`：当前 A6000 best。

其他 full runs 应继续被索引，因为它们记录了失败方向并避免重复实验；但普通 git 不需要保留它们的 raw JSONL。

## Cross-Model 历史记录

Cross-model registry 包含 `7` 个完整本地记录：

| 模型/run | Pass | 解释 |
|---|---:|---|
| Dream-Coder Instruct official canvas `alpha010_cap24` | `848/1033 = 82.09%` | 最强本地 cross-model 记录，但 protocol 与 LLaDA LCAL stack 不同。 |
| Dream-Coder Base official canvas `alpha010_cap24` | `825/1033 = 79.86%` | official-canvas prompting 上有强 transfer signal。 |
| Dream-Coder Base official canvas `alpha020` | `818/1033 = 79.19%` | 弱于 `alpha010_cap24`，提供 alpha sensitivity 证据。 |
| LLaDA-8B-Instruct LCAS v3 resume | `817/1033 = 79.09%` | 说明 instruct checkpoint 在本地 protocol 下可能较强。 |
| Dream-Coder Instruct official canvas `alpha020` | `785/1033 = 75.99%` | Alpha sensitivity 很大，不能据此判断模型弱。 |
| Dream-Coder LCAS v3 runs | `0/1033` 和 `1/1033` | 更可能是 prompt/canvas mismatch，不是模型质量结论。 |

Cross-model 结论：已有历史证据足以支持 model-generalization track，但这些记录还不是干净的 SOTA 对比，因为 prompt format 和 canvas 假设不同。

## 与论文相关的 takeaway

对 CCF-A 质量论文而言，历史结果支持如下叙事：

- DLLM code infilling 的核心技术瓶颈是未知长度 infilling。
- Inference-time confidence probing 与 bounded repair 能恢复许多 short/medium cases。
- 同硬件 A6000 证据验证了 conservative mid rescue。
- True-long cases 仍然困难，因为当前 signals 会把 long failures 与 short/medium false positives 混淆。

当前 claim 可以作为经验诊断和 inference-time medium rescue method，但若要达到顶会论文强度，下一阶段还需要更强 long-tail method、cross-model validation，或更尖锐的理论/算法贡献。

## 归档策略

不要删除历史 raw outputs。不要把 raw `results.jsonl` 提交到普通 git，除非某个 run 被明确选入 Git LFS 或外部 artifact 存储。继续保留：

- 紧凑 run registries；
- pairwise summaries；
- bucket summaries；
- experiment plans/specs；
- 本文件这类 interpretation reports。
