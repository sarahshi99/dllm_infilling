# Research Design v1：Length-Controlled Diffusion Code Infilling

创建时间：2026-05-31 12:36 CST

## Problem Statement

用于代码 infilling 的 diffusion language models 在 denoising 前需要 infill length 或 canvas。当前项目显示，length choice 不是次要工程细节；它是 `HumanEval-SingleLineInfilling` 的主导 failure mode，尤其当真实 middle span 长于 policy-selected mask length 时更明显。

审稿人能够认可的研究问题是：

> DLLM 如何在没有 oracle length information 的情况下选择或自适应 infilling length，同时保持 short-completion safety，并改善 long 或 medium under-selection？

## Research Community As User

这里的有效“用户”是 code-generation researchers、DLLM researchers、reviewers，以及 length-control baseline authors。他们需要的不是另一个本地 heuristic，而是对一个已知缺口的可复现回答：fixed canvases 和 oracle lengths 都不现实，而朴素 confidence-based length selection 会低估困难 long cases。

## Minimum Publishable Contribution

minimum publishable contribution 是一个 method 与 evaluation package，在 matched protocol 下证明以下至少一项：

1. 有原则的 inference-time length-control method 相比 fixed、oracle-ablated、CAL-like 和 LCAS/LCAL internal baselines 改善 DLLM code infilling，且没有 short-bucket regression。
2. 一个严谨的 negative result 证明 confidence-curve-only inference rescue 不足以解决 true-long code infilling，并引入更强的 trajectory-based 或 learned length signal 来缩小部分差距。
3. 一个 dynamic-canvas 或 length-regularized method 在 apples-to-apples settings 下达到有竞争力的 literature-level performance。

## Current Central Claim

Same-hardware A6000 evidence 支持以下较窄 claim：

> Inference-time confidence-curve agreement 可以安全恢复 medium-length DLLM code infilling cases，但 true-long code infilling 仍主要受 length underestimation 支配，需要比当前 official-CAL trigger family 更强的 length modeling。

这还不是 CCF-A central claim。它是可信的 empirical foothold，也是有用的论文叙事约束。

## Evidence Base

- A6000 control：`787/1033 = 76.19%`。
- A6000 `midcons`：`795/1033 = 76.96%`，相对 same-hardware control 为 `+8` wins、`0` losses。
- Gains 集中在 `9-12` 和 `13-16` oracle buckets。
- Long buckets 没有变化：`17-24 = 20.73%`，`25+ = 16.13%`。
- 在 A6000 `midcons` run 中，多数 failed `oracle >= 17` cases 是 under-selected；已有报告记录 `90/91` failed long cases 为 under-selected，且 `71/91` 仍从 `base` 结束。
- Offline long-underestimate sweep 评估了 `16776` 条 rules，没有找到同时满足 true-long precision、short-risk 与 recall criteria 的安全 rule。
- Probe-curve single-feature audit 评估了 `4106` 个 thresholds，发现 `0` 个 strict viable thresholds。当前 full run 的所有 rows 都有 probe-curve features，但没有保存 stopping traces。
- 文献记录指出 DreamOn 和 LR-DLLM 是更强的 length-control anchors；在 protocol matching 前，当前本地结果不能称为 SOTA。

## Novelty Hypothesis

最强 novelty path 不是当前 `midcons` rule 本身。Novelty 必须来自解释并修复 safe medium rescue 与 unsafe long rescue 之间的边界：

- 当存在 short-safe confidence agreement 时，medium under-selection 可以被纠正；
- true-long under-selection 不能由同一批 confidence fields 捕获；
- multivariate probe-curve scoring、denoising trajectories、learned length classification、dynamic canvas resizing 或 length regularization 可能提供缺失 signal。

在新实验验证前，这仍是 hypothesis。

## Approaches Considered

### Approach A：巩固 Medium Rescue

把 `midcons` 作为核心贡献，并将其表述为 short-safe inference-time rescue。

Pros：

- 已有 same-hardware 正向证据。
- 易于复现和 ablate。
- 实现风险低。

Cons：

- 提升幅度较小。
- 没有解决 long infilling。
- 如果单独使用，对强 CCF-A 论文来说过于 heuristic。

### Approach B：继续 Heuristic True-Long Gates

放宽或重调 official-CAL true-long gates。

Pros：

- 代码改动最小。
- 使用已有 runner 和 result fields。

Cons：

- 当前证据为负。
- Offline sweep 显示，在可用 trigger count 下 short-risk 很高。
- 可能浪费 GPU budget。

### Approach C：构建更强 Length Modeling

加入 trajectory features、learned length classifier、dynamic-canvas behavior 或 length regularization，再与 `midcons` 和 literature anchors 比较。

Pros：

- 最可能形成 paper-level contribution。
- 直接针对已观察到的 failure mode。
- 可以先 diagnostic-first 评估，再运行 full GPU。

Cons：

- 工程和实验风险更高。
- 可能需要更广泛的 protocol alignment 或 training/fine-tuning 决策。
- 需要谨慎 ablation，避免变成黑箱 patch。

Recommendation：选择 Approach C 作为论文方向，同时保留 Approach A 作为当前可复现 checkpoint，并拒绝 Approach B，除非新 diagnostics 推翻当前证据。

## CEO-Style Stress Review

Novelty：有条件成立。当前 medium rescue 单独不够新；medium-vs-true-long separation 加上更强 length modeling 才可能有 novelty。

Importance：如果 framing 是 unknown-length DLLM infilling，则重要性高，因为 fixed-canvas 和 oracle-length assumptions 是核心实践限制。

Reviewer appeal：目前中等；如果论文包含 protocol-matched DreamOn/LR-DLLM comparisons、强 ablations 和清晰 failure taxonomy，则可能较高。

Strongest contribution：项目已有少见的具体 same-hardware evidence，显示 medium-length rescue 可以 short-safe，而 true-long rescue 因不同原因失败。

Weakest assumption：不通过 training-time length regularization 也能找到更强 inference-time signal。如果该假设失败，论文必须 pivot 到 negative result 加 length-regularized method。

CCF-A realism：目前不足，但如果下一阶段产生有原则的 long-length method 或强 cross-model validation，则现实可行。

## Success Criteria

- Same-hardware A6000 pass@1 超过 `76.96%`，且没有 `<=8` 或 `9-12` regression。
- 改善 `17-24` 和/或 `25+` buckets，或用强 diagnostics 证明 inference-only rescue 为什么做不到。
- 为 medium rescue、long signal、stopping policy 和 length-selection features 提供 ablations。
- 在至少一个额外 model family 上以 matched prompt/canvas/evaluation settings 复现。
- 区分 apples-to-apples comparisons 与 suggestive literature comparisons。

## Kill Criteria

- Proposed long detector 在 offline diagnostics 中 short-risk 高于 `5%`，且没有令人信服的 precision/recall tradeoff。
- Full GPU candidate 只通过 short-bucket regressions 改善 aggregate pass rate。
- Cross-model transfer 失败，且没有出现可解释 failure pattern。
- Literature protocol matching 表明本地 setting 过窄，不足以支撑所声明贡献。

## Scope Boundary

下一里程碑不应声称 SOTA。它应产出可复现的 paper-planning package：evidence snapshot、current method claim、next experiment plan 和 GPU-safe queue strategy。
