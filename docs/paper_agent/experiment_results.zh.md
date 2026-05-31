# Experiment Results

更新时间：2026-05-31 15:56 CST

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

## Paper Relevance

这支持一个狭窄但诚实的 paper claim：medium-length under-selection 可以通过 confidence-curve agreement 安全修复。它还不支持 broad CCF-A claim 或 SOTA claim。

## Next Result Needed

下一项结果应是以下之一：

- diagnostic feature snapshot 证明存在更强 long-tail signal；
- smoke GPU run 显示没有 short-bucket regression；
- full same-hardware run 改善 long buckets；
- 或严谨 negative result，支撑转向 dynamic canvas 或 length regularization。
