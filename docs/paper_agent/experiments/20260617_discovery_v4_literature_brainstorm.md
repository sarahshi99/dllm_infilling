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
