# Experiment Plan v3：Probe-Curve-First Length Modeling

更新时间：2026-05-31 15:24 CST

## Codex Phase 0 更新（2026-07-02）

V7/V8 full runs 已将“全局比例放长 reward”路线降级为负结果。Codex 审计后建议把下一阶段优先级从继续 probe/threshold sweep 转为 `True-long action-ceiling matrix`：用小规模、预注册、可 dry-run 的 action matrix 区分 canvas adequacy、rescue generation、candidate selection 和 trigger recall。该实验仍必须先写 action brief 和 dry-run manifest，不应直接启动 full GPU。

## Objective

从当前 `midcons` A6000 checkpoint 出发，推进到 CCF-A 级别的 DLLM code infilling length-control method。本计划优先进行可复现 diagnostics，再启动昂贵 GPU runs。

## Baseline And Environment

- Dataset：`HumanEval-SingleLineInfilling`，test split，`1033` tasks。
- Current model：`GSAI-ML/LLaDA-8B-Base`。
- Same-hardware baseline：A6000 union control，`787/1033 = 76.19%`。
- Current checkpoint：A6000 `midcons`，`795/1033 = 76.96%`。
- GPU policy：除非用户显式更改分配，未来实验必须只使用 GPU `2,3`。不要 kill、抢占或中断已有进程；如果卡 `2,3` 被占用，应等待或排队。本计划继续从 CPU-only analysis 推进，直到 offline gates 证明值得启动 smoke run。

## Engineering Review Summary

Architecture：保持 legacy `expvision_dllm/` 和 `scripts/` 冻结。新工作放入 `expvision_dllm_clean/`、`clean_scripts/`、`analysis/`、`analysis_outputs/` 和 `docs/paper_agent/`。

Data flow：

```text
existing results.jsonl
  -> compact evidence builder / diagnostic script
  -> paper-agent evidence snapshot and probe-curve audit
  -> experiment_results and dashboard
  -> GPU runner only if offline criteria pass
```

Failure posture：任何 GPU policy 都不应基于未通过 offline short-risk 与 recall checks 的 heuristic 启动。Raw outputs 保留在本地；tracked artifacts 只包含 compact summaries。

## Datasets

Primary：

- `HumanEval-SingleLineInfilling`，`1033` tasks。

Required next validation：

- 如果 protocol alignment 可行，加入 HumanEval-Infilling multi-line。
- 只有当 single-line story 稳定后，才加入 SantaCoder-FIM 或另一个 FIM-style benchmark。

## Baselines

Internal baselines：

- fixed length。
- oracle length。
- CAL-lite。
- LCAS-v3b。
- LCAL。
- official-CAL bounded repair。
- A6000 union control。
- A6000 `midcons`。

External anchors：

- DreamOn dynamic canvas。
- LR-DLLM length regularization。
- Autoregressive code infilling models 只在 explicitly matched settings 下比较。

## Metrics

Primary：

- pass@1。
- 相对 same-hardware control 的 pairwise wins/losses。
- oracle-length bucket pass rates：`<=8`、`9-12`、`13-16`、`17-24`、`25+`。

Secondary：

- selected-minus-oracle length error。
- under-selection rate。
- 按 source 统计 trigger count 和 trigger precision。
- proposed long detectors 的 short-risk rate。
- current-pass risk rate。
- decode/probe overhead。

## Experiment Phases

### E0：Evidence Snapshot

从已有 raw outputs 构建紧凑 paper-agent evidence snapshot。必须验证 row counts、pass rates、bucket metrics、wins/losses 和 long-failure under-selection counts。

Success：生成 summary 与已有报告一致，并作为 compact documentation 跟踪。

### E1：Long-Signal Diagnostic Expansion

将 CPU-only diagnostics 扩展到现有 scalar result fields 之外，并从 probe-curve shape features 开始，因为当前 A6000 full-run outputs 没有保存 `stopping_trace` 或 `step_traces`。

当前可用：

- length-probe curve shape features；
- base、official-CAL 和 long probe selections 之间的不一致；
- 不需要 inference-time oracle 的 failure signatures。

当前 full run 不可用：

- 从 `stopping_trace` 或 step traces 提取的 denoising trajectory summaries。

最新 CPU audit result：

- `analysis/analyze_probe_curve_long_signals.py` 评估了 `4106` 个 single-feature probe-curve thresholds。
- `strict_viable_thresholds = 0`。
- 最佳 threshold 有 `63.04%` true-long precision 和 `31.87%` failed-long recall，但 `8.70%` short-risk，高于 `5%` safety gate。

Success：找到 low short-risk 且有足够 failed-long recall 的 candidate signal 或 learned score，值得启动 smoke GPU run。

Kill：没有 candidate 同时满足 short-risk `<=5%` 和至少 `10` 个 failed-long triggers，除非有清晰记录的 lower-precision/high-recall tradeoff。

### E2：Learned Length Classifier Or Scorer

如果 E1 single-feature rules 失败，在已有 diagnostic fields 上训练或拟合 lightweight length-risk classifier，并使用严格 split discipline 验证。Inference time 不能使用 oracle。

Success：held-out diagnostic precision/recall 超过 hand rules，并保持 short safety。

Kill：classifier 依赖不能跨 model family 或 environment transfer 的 run-specific artifacts。

### E3：Trace-Enabled GPU Smoke Then Full Run

只有在 E1 或 E2 通过 offline gates 后，才在 GPU `2,3` 可用或可以安全等待时运行 smoke experiment。如果下一项 hypothesis 依赖 trajectory information，则 smoke run 必须启用 `--save-step-traces`，确保实际捕获缺失的 trajectory signal。只有 smoke results 未显示 short regression 时，才运行 full `1033`。

Command pattern：

```bash
CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false <runner command>
```

Success：full candidate 改善 total pass rate 和 long buckets，且无 short-bucket regression。

Kill：任何 controlled short-bucket regression，除非被强且有记录的 long-bucket gain 抵消。

### E4：Cross-Model Protocol Alignment

在可用缓存上，用 stable protocol 重跑 Dream-Coder Base/Instruct 和 LLaDA Instruct。只有在模型缓存或网络权限可用时才加入 Dream/DiffuCoder。

Success：method effect transfer，或产生可解释的 non-transfer result。

Kill：prompt/canvas mismatch 使 apples-to-apples interpretation 不成立。

## Compute Budget

CPU-only diagnostics 应先运行，并且现在可以继续。GPU runs 应等待卡可用或使用 wait/queue scripts。任何覆盖 `1033` samples 的 full run 都必须有 manifest，注明 model、command、GPU set、output directory 和 comparison baseline。

## Reproducibility Requirements

每个报告实验必须包含：

- baseline；
- environment；
- GPU set；
- model；
- command；
- output directory；
- total pass rate；
- bucket metrics；
- wins/losses；
- same-hardware 或 cross-hardware comparison label。

## Priority

1. Evidence snapshot 和 paper-agent docs。
2. Long under-selection 的 probe-curve multivariate 或 learned diagnostic scoring。
3. 只有 diagnostic gates 通过后，才启动 trace-enabled GPU smoke experiment。
4. Full A6000 candidate。
5. Cross-model protocol-matched validation。

## Current Decision

使用 `midcons` 作为当前 checkpoint。不要从已有 official-CAL gate family 或 single-feature probe-curve threshold 启动另一个 full true-long GPU run。
