# Discovery V4 Literature Brainstorm

时间：2026-06-17 CST

## 目的

这份 brief 回答用户的问题：下一步不能只靠枚举 feature，也不能因为 v2/Route2 没有解决 true-long 就放弃。我们需要重新设计三层 Discovery layer，让它能更聪明地发现信号，并把信号转成可审稿的 inference-time/training-free 策略。

本轮没有启动 GPU 实验。当前 GPU2/3 有其他用户任务，后续任何 GPU 动作都必须重新检查 `nvidia-smi`，并写 action brief。

## 当前事实

Route2 precision `len32`：

- pass：`801/1033 = 77.54%`
- pairwise vs `midcons`：`6/0/795/232`
- triggers：`57`
- triggered failed-long：`33`
- missed failed-long：`56`
- triggered failed-long 中 rescue length >= oracle：`31/33`

直觉：

- 不是没有信号：`6` 个 wins 都是 triggered rescue。
- 不是已经解决 true-long：oracle `25+` 仍不动。
- 不是简单继续加长：大多数 triggered failed-long 的长度已经够。
- 关键是混合问题：gate recall + rescue generation/selection quality。

## 文献启发总结

## 审稿人视角的方法谱系

你的质疑是合理的：如果只说“借鉴很多方向”，很容易变成经验拼盘。V4 应该明确回答三个问题：

1. 哪些部分是成熟方法，可以直接复用？
2. 哪些部分只是帮助发现信号的 microscope，不能直接写成最终方法？
3. 哪些部分才是这个项目自己的新融合？

结论：

- V4 不是从零发明一个全新算法。
- V4 的骨架来自成熟的 constrained selection、slice discovery、rule distillation。
- V4 的 discovery layer 融合 time-series shape、calibration/OOD、weak supervision、partial uplift、MBR/self-consistency 等思路，用来找非线性组合和时序形状。
- V4 的项目特异性创新在于：把 DLLM code infilling 的一次输出拆成 row-action taxonomy，并把问题拆成 `MissedLongHead` 和 `RescueQualityHead` 两个头，然后只接受 inference-visible、training-free、可审稿的小规则或小 score。

| 方向 | 可直接借用 | 在本项目中的改造 | 是否是最终方法 |
|---|---|---|---|
| Risk-controlled selection | risk-coverage、false-positive constraint、held-out threshold | failed-long coverage 受 short/current-pass risk 约束 | 是，作为评估和 gate 框架 |
| Slice/subgroup discovery | 找可解释的异常/失败数据切片 | 找 missed-long、rescued-long、triggered-failure、risk slices | 是，作为候选规则来源 |
| Rule list / RuleFit / Anchors | 小规则、rule precision/coverage、规则蒸馏 | 限制到最多三条 inference-visible clauses | 是，若规则稳定 |
| Time-series feature discovery | catch22/tsfresh/shapelet/ROCKET 风格的轨迹摘要 | decode trace 的 plateau、collapse、stagnation、disagreement | 主要是 microscope，需蒸馏 |
| Calibration / OOD | confidence residual、misclassification detection | stop reason / selected length 条件下的置信残差 | 可成为规则特征 |
| Weak supervision | labeling functions 和弱信号冲突/重叠分析 | 将 probe、trace、stop reason、policy disagreement 当作 noisy signals | 主要是 microscope |
| Uplift / logged-policy learning | action-outcome 思维、反事实风险警告 | primary/len24/len32/broad 的 partial action evidence | 诊断工具，不做因果 claim |
| MBR / self-consistency | 多候选一致性作为无 verifier 的质量 proxy | 只在 rescue quality slice 明确时作为备选 action | 可能成为后续 GPU action |

这意味着 V4 的“新”不是某个通用 ML 模型，而是任务结构化和决策协议：

- 先构造 `row_action_table`，把每个任务在不同 policy 下的行为摆平；
- 用 `MissedLongHead` 寻找该触发而没触发的 true-long；
- 用 `RescueQualityHead` 分析触发后为什么仍失败，尤其是长度足够但生成/选择失败；
- 所有 learned/discovery 模型只能提出候选，最终必须回到 training-free 或明确转向 learned-controller 论文方向。

如果这样仍找不到稳定信号，负结果也是可信的：我们可以说已经用 risk-control、slice discovery、trajectory shape、calibration residual、weak signal fusion、partial action diagnostics 多条路线做过 CPU audit，而不是只扫了几个手写阈值。

### 1. Risk-Controlled Selection

代表方向：

- selective classification / risk-coverage；
- Neyman-Pearson classification；
- conformal risk control。

对本项目的启发：

- 不要用 accuracy/AUC 作为主目标。
- 要显式约束 short-risk 和 current-pass-risk。
- candidate gate 必须报告 coverage-risk frontier。

映射到 layer：

- Layer 2 用 constrained utility 排序候选。
- Layer 3 用 held-out risk gate 决定是否能上 GPU。

### 2. Slice / Subgroup Discovery

代表方向：

- automated data slicing；
- subgroup discovery；
- interpretable decision sets/rule lists。

对本项目的启发：

我们不是在找“一个全局特征”，而是在找模型失败/成功的局部区域：

- missed failed-long slice；
- triggered but still failed slice；
- rescued long slice；
- short/current-pass risk slice。

映射到 layer：

- Layer 1 先构建完整 row-action taxonomy。
- Layer 2 搜索小 conjunction / slice，而不是只扫单 feature。
- Layer 3 把稳定 slice 蒸馏成 gate 或 action rule。

### 3. Time-Series Trace Shape

代表方向：

- catch22 / compact time-series feature bank；
- shapelets；
- ROCKET/random convolution kernels。

对本项目的启发：

v1/v2 可能不是 trace 没信号，而是公式形状太窄。Trace 应该看：

- late plateau；
- early confidence collapse；
- high-confidence stagnation；
- progress-then-stall；
- trace/probe disagreement。

映射到 layer：

- Layer 2 用 shape/motif discovery 找候选。
- 但 final policy 只能使用可命名 motif 或少量统计量。

### 4. Calibration / OOD / Misclassification Detection

代表方向：

- neural calibration；
- max-softmax / OOD baselines；
- deep ensembles / uncertainty estimation。

对本项目的启发：

当前 missed failed-long 里很多看上去 trace confidence 很高。说明 raw confidence/top1 不能直接当真实可信度，要做 residual：

- confidence conditioned on selected length；
- confidence conditioned on stop reason；
- confidence-probe disagreement；
- energy-like uncertainty aggregate；
- late-vs-early confidence change。

映射到 layer：

- Layer 2 添加 calibration residual family。
- 目标是抓高置信但实际 under-length 的 missed rows。

### 5. Counterfactual / Uplift Modeling

代表方向：

- heterogeneous treatment effect；
- causal trees；
- off-policy evaluation；
- counterfactual risk minimization。

对本项目的启发：

Route2 是 action，不只是 label：

- primary；
- precision len24；
- precision len32；
- broad len24；
- 未来可能的 alternate rescue decoding。

我们真正想知道的是：对某个 row，哪个 action 有正收益且低风险。

限制：

- 当前 rescue 只在 triggered rows 上观察到，存在 treatment bias。
- 所以 uplift 现在只能当 diagnostic，不能当因果 claim。

映射到 layer：

- Layer 1 合并多策略 action outcome。
- Layer 2 做 partial action/uplift diagnostics。
- Layer 3 只有在可解释 action family 出现后才允许 GPU 验证。

### 6. Weak Supervision / Heuristic Fusion

代表方向：

- data programming；
- weak supervision；
- labeling functions。

对本项目的启发：

我们已经有许多弱信号：

- low top1 / plateau；
- probe long bump；
- selected length short；
- stop reason；
- broad/precision disagreement；
- confidence residual。

不要硬写一个 AND 公式。可以先把这些作为 labeling functions，分析哪些组合稳定，再蒸馏成规则。

映射到 layer：

- Layer 2 用 weak-signal vote / label model 做 microscope。
- Layer 3 只接受蒸馏后的规则或小 score。

## V4 三层设计

### Layer 1：Error And Action Anatomy

目标：把所有可用结果合并成 row-action table。

输入：

- current `midcons`；
- Route2 precision `len32`；
- Route2 precision `len24`；
- Route2 broad `len24`；
- full trace；
- probe fields。

输出：

- 每个 task 的 action outcome；
- 触发/未触发；
- wins/losses/tie-pass/tie-fail；
- missed failed-long；
- triggered failed-long；
- rescued long；
- short/current-pass risk；
- trace/probe/stop reason features。

意义：

这一步避免“只盯最终 pass rate”，把 gate recall 和 rescue quality 分开。

### Layer 2：Discovery Model Layer

目标：用多个 microscope 找信号，而不是只枚举 feature。

模块：

1. constrained slice/subgroup miner；
2. shallow tree / sparse score / rule extraction；
3. trace shape motif discovery；
4. calibration residual features；
5. weak heuristic fusion；
6. partial uplift/action diagnostics。

所有模块都必须输出：

- top candidates；
- held-out risk；
- fold stability；
- 是否使用了 forbidden labels；
- 是否能蒸馏成 rule。

### Layer 3：Policy Distillation

目标：把信号转成可审稿策略。

候选机制：

1. Probe-Trace Fusion Gate
   用于补 `56` 个 missed failed-long，但必须低 short/current-pass risk。

2. Rescue Quality Selector
   用于处理 `33` 个 triggered failed-long。若长度够但失败，应换 rescue decoding/selection，而不是继续加长。

3. Conservative Route2 Polish
   如果没有强新信号，就把 precision `len32` 作为低风险小幅正结果。

4. Stop Current True-Long Branch
   如果所有候选都不稳定或依赖 oracle/pass labels，就停止这一支。

## 推荐下一步

先实现 CPU-only `discovery_v4_signal_audit.py`，不启动 GPU。

最低产物：

- `row_action_table.csv`
- `slice_candidates.csv`
- `rule_candidates.csv`
- `trace_shape_candidates.csv`
- `calibration_residuals.csv`
- `uplift_diagnostics.csv`
- `policy_shortlist.md`
- `report.md`

GPU 只有在以下条件满足时才能考虑：

- held-out missed failed-long coverage 至少 `8`，或 triggered rescue-failure slice 至少 `10`；
- short-risk <= `3`；
- current-pass-risk <= `3`；
- 规则稳定出现在至少 `4/5` folds；
- 机制不是简单“加长”；
- GPU2/3 没有他人任务。

## References

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks", 2017. https://arxiv.org/abs/1705.08500
- Angelopoulos et al., "Conformal Risk Control", 2022. https://arxiv.org/abs/2208.02814
- Tong, Feng, and Li, "Neyman-Pearson Classification Algorithms and NP Receiver Operating Characteristics", 2018. https://arxiv.org/abs/1608.03109
- Chung et al., "Slice Finder: Automated Data Slicing for Model Validation", 2019. https://arxiv.org/abs/1807.06068
- Friedman and Popescu, "Predictive Learning via Rule Ensembles", 2008. https://arxiv.org/abs/0811.1679
- Angelino et al., "Learning Certifiably Optimal Rule Lists for Categorical Data", 2017. https://arxiv.org/abs/1704.01701
- Lubba et al., "catch22: CAnonical Time-series CHaracteristics", 2019. https://arxiv.org/abs/1901.10200
- Dempster, Petitjean, and Webb, "ROCKET", 2019. https://arxiv.org/abs/1910.13051
- Grabocka et al., "Learning Time-Series Shapelets", 2014. https://arxiv.org/abs/1503.03238
- Guo et al., "On Calibration of Modern Neural Networks", 2017. https://arxiv.org/abs/1706.04599
- Hendrycks and Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", 2016. https://arxiv.org/abs/1610.02136
- Lakshminarayanan, Pritzel, and Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", 2017. https://arxiv.org/abs/1612.01474
- Athey and Imbens, "Recursive Partitioning for Heterogeneous Causal Effects", 2016. https://www.pnas.org/doi/10.1073/pnas.1510489113
- Dudik, Langford, and Li, "Doubly Robust Policy Evaluation and Learning", 2011. https://arxiv.org/abs/1103.4601
- Ratner et al., "Snorkel: Rapid Training Data Creation with Weak Supervision", 2017. https://arxiv.org/abs/1711.10160
- Benjamini and Hochberg, "Controlling the False Discovery Rate", 1995. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Meinshausen and Buhlmann, "Stability Selection", 2010. https://doi.org/10.1111/j.1467-9868.2010.00740.x
- Ribeiro, Singh, and Guestrin, "Nothing Else Matters: Model-Agnostic Explanations By Identifying Prediction Invariance", 2016. https://arxiv.org/abs/1611.05817
- Liu, Rosen, and G.C., "AutoSlicer: Scalable Automated Data Slicing for ML Model Analysis", 2022. https://arxiv.org/abs/2212.09032
- Christ, Kempa-Liehr, and Feindt, "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests", 2016. https://arxiv.org/abs/1610.07717
- Swaminathan and Joachims, "Counterfactual Risk Minimization: Learning from Logged Bandit Feedback", 2015. https://arxiv.org/abs/1502.02362
- Eikema and Aziz, "Sampling-Based Approximations to Minimum Bayes Risk Decoding for Neural Machine Translation", 2021. https://arxiv.org/abs/2108.04718
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", 2022. https://arxiv.org/abs/2203.11171
- Manakul, Liusie, and Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models", 2023. https://arxiv.org/abs/2303.08896
