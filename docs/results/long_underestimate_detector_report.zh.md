# Long Underestimate Detector Sweep 报告

更新时间：2026-05-29 Asia/Shanghai

## 目的

本报告检查当前 A6000 `midcons` 结果是否包含足够的 inference-time signals，可以安全触发 long-tail rescue rule。这是 diagnostic-only 分析，不改变任何生成代码结果。

## 命令

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/diagnose_long_underestimate_policy.py \
  --results outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626/results.jsonl \
  --output-dir analysis_outputs/long_underestimate_detector/a6000_midcons
```

## 结果

Sweep 在 `1033` 行上评估了 `16776` 条候选规则。

最佳规则：

```text
sel<=3|best>=13|gap>=4|ratio>=0.45|raw>=0.4|supp>=0|src=base
```

指标：

- triggers: `93`
- true-long precision: `35.48%`
- failed-long recall: `36.26%`
- short-risk rate: `40.86%`
- current-pass risk: `27.96%`

严格成功标准：

- trigger count >= `10`
- true-long precision >= `60%`
- short-risk rate <= `5%`
- failed-long triggered rows >= `10`

结果：`0` 条规则通过。

即使把 short-risk 放宽到 `10%` 或 `20%`，也没有出现至少五个 triggers 的可行规则。第一个非空 low-risk family 只在 `short_risk <= 40%` 左右出现，这对 GPU policy 太危险。

## 解释

当前 result fields 不包含安全的 long-underestimation trigger。Long failures 确实存在，而且大多是 under-selected；但同样弱的 long-curve signals 也出现在 short/medium cases 中。直接 heuristic rescue 很可能损害使 `midcons` 安全的 short buckets。

## 决策

不要再基于当前 gate family 启动 GPU 实验。下一条可信 long-tail 路线应加入更强信号：

- denoising trajectory/step-trace features；
- learned length classifier；
- DreamOn-style dynamic canvas method；
- LR-DLLM-style length regularization。

这个负面结果支持论文叙事：medium rescue 可以通过 confidence-curve agreement 做到 short-safe，但 true long infilling 需要更强 length modeling。
