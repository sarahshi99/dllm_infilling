# CCF-A Roadmap 与下一步实验实施计划

> **给 agentic worker：** 英文版是完整执行计划。本中文版本翻译研究路线、实验设计、风险和执行更新。

**目标：** 将当前 DLLM code-infilling 项目推进为可复现、具备 CCF-A 投稿路径的研究工作。

**架构：** 将工作拆成 evidence、method 和 validation 三条线。Evidence 通过 compact reports 和 registries 跟踪；method work 放在新的 `clean_scripts/` 文件；validation 先做同硬件 A6000 controls，再做 cross-model reruns。

**技术栈：** `clean_scripts/` Python runners、`expvision_dllm_clean/` reusable modules、本地 raw outputs `outputs_clean/`、compact reports `analysis_outputs/` 与 `docs/results/`、Git worktrees。

---

## Superpowers Phase Log

- `using-git-worktrees`：实验分支为 `exp/a6000-midcons-longrescue`，归档分支为 `docs/result-archive`。
- `receiving-code-review`：用于 scoreboard label、bucket registry key、GPU launcher behavior 的 P2 fixes。
- `test-driven-development`：用于 resumable A6000 policy runner。
- `verification-before-completion`：用于实验完成声明、commit 前和报告 pass rates 前。
- `writing-plans`：用于把研究方向变成可执行步骤。
- `brainstorming`：用于下一步实验设计；默认继续推进，除非遇到破坏性操作或真正决策阻塞。

## 需求追踪

- 阅读仓库并解释当前项目：已完成；项目聚焦 DLLM code infilling 的 dynamic mask length selection 与 LCAS stopping。
- 保留历史 outputs：raw `outputs_clean/` 保留本地；提交 compact records。
- 使用 Git worktrees：已使用 `git_workspace/.worktrees/a6000-midcons-longrescue`。
- 结果可复现：每份报告记录命令、输出目录、baseline 和 pairwise path。
- 使用 A6000：已完成 A6000 control 与四个 candidates。
- 不杀其他用户进程：遵守。
- 尽量写新代码文件：新增 resumable runner 和 launcher。
- 文献对比后再谈 SOTA：已记录 DreamOn、CAL、LR-DLLM anchors；不声称 SOTA。
- Cross-model validation：历史 registry 已完成，新 GPU runs 暂停。
- 历史结果整理：已通过 result archive 完成。

## 当前项目

项目研究 diffusion language models 的 code middle infilling，目前中心模型是 `GSAI-ML/LLaDA-8B-Base`，数据为 `HumanEval-SingleLineInfilling`。核心问题是 DLLM 在 denoising 前需要 mask/canvas length，而错误长度选择是主要失败模式。

现有方法栈：

- Fixed length 与 oracle length 给出下界/上界。
- CAL-lite 用 confidence probe 选择候选 mask length。
- LCAS 按 length bucket 自适应停止 denoising。
- LCAL 从 confidence curves 增加 long-aware correction。
- official-CAL bounded repair 安全处理 short under-selection。
- `midcons` 增加 conservative medium-length rescue。

## 既有结果

最新 A6000 受控结果：

- A6000 control：`787/1033 = 76.19%`。
- `midcons`：`795/1033 = 76.96%`，`+8` wins，`0` losses。
- `mid_precision`：`787/1033 = 76.19%`，无 pairwise change。
- `true_long`：`787/1033 = 76.19%`，无 pairwise change。
- `combined`：`787/1033 = 76.19%`，无 pairwise change。

意义：

- `midcons` 是真实的同硬件改进，不是环境噪声。
- 改进主要来自 `9-12` 和 `13-16`。
- Long buckets 仍然差：`17-24 = 20.73%`，`25+ = 16.13%`。
- 当前 official-CAL true-long triggering 无效；安全 gates 后触发 0 次。

## 距离 CCF-A 的差距

当前项目尚未达到 CCF-A 完整度。它已经对 length selection 有强 empirical handle，但 publishable claim 仍偏 heuristic，需要加强。

缺失部分：

- 更 principled 的方法，而不只是 hand-tuned rescue rule。
- 强 literature-positioned baselines，尤其 DreamOn-style variable length generation 与 LR-DLLM-style length regularization。
- 在 Dream-Coder、DiffuCoder、Dream、LLaDA variants 上 cross-model validation。
- Multi-line 或更广 benchmark 证据，除非论文明确限定 single-line length control。
- Ablations：official-CAL、long curve、support、raw ratio、stop policy、repair bounds。
- Error analysis：解释 long-tail under-generation 为什么难。

## 核心 claim 候选

推荐 claim：

> DLLM code infilling 的 inference-time length control 可以通过分离 medium-length rescue 与 true-long detection 来做到 short-safe；confidence-curve agreement 能在不牺牲 short tasks 的情况下改善 medium infilling，而 true-long cases 需要不同的 underestimation detector，因为 official-CAL 在 long failures 上不可靠。

这是目前诚实的 claim。更强 CCF-A claim 需要下一步 long detector 或 cross-model transfer 成功。

## Baselines

内部 baselines：

- fixed64；
- oracle length；
- CAL-lite；
- LCAS-v3b；
- LCAL；
- official-CAL bounded repair；
- A6000 control；
- `midcons` A6000 best。

外部 baselines：

- DreamOn on Dream-7B、DiffuCoder-7B、DreamCoder-7B；
- LR-DLLM fully unknown length setting；
- 在 evaluation settings 匹配时，DeepSeek-Coder、Seed-Coder、Qwen2.5-Coder 等 autoregressive code infilling models。

## Metrics

Primary：

- `HumanEval-SingleLineInfilling` pass@1；
- 相对同硬件 control 的 pairwise wins/losses；
- oracle-length bucket pass rates：`<=8`、`9-12`、`13-16`、`17-24`、`25+`。

Secondary：

- selected length error；
- under/over-selection rate；
- trigger count 与 trigger precision by source；
- decode time 与 probe overhead；
- short-bucket loss rate。

## Paper Tables

- Table 1：内部 baselines 与 literature anchors 的 main pass@1。
- Table 2：同硬件 A6000 pairwise results。
- Table 3：Oracle-length bucket breakdown。
- Table 4：Rescue signals ablation。
- Table 5：Cross-model generalization。
- Table 6：Long under-generation failure taxonomy。

## 下一步实验

### Task 1：完成当前结果归档

已完成：A6000 pairwise summaries、scoreboard、compact reports；result archive registry 已刷新。

### Task 2：将 `midcons` 设为当前 A6000 checkpoint

已完成：`midcons` 是当前 A6000 best；`mid_precision`、`true_long`、`combined` 记录为负面证据。

### Task 3：设计独立 Long-Underestimation Detector

方向：

- 不再把 official-CAL high length 作为 primary long trigger；
- 从 long-tail under-selection signatures 触发；
- 先做 record-only/diagnostic mode；
- 保护 short buckets，要求使用不会大量出现在 already-passing short samples 中的证据。

### Task 4：Cross-Model Validation

暂停到 GPU 可用后执行：

- 在 cached Dream-Coder Base/Instruct 上重跑 stable protocol；
- 在 cache 允许时重跑 LLaDA Base/Instruct；
- 仅在 model cache/network 可用时加入 DiffuCoder-7B 或 Dream-7B；
- 只有在 prompt/canvas/evaluation settings 匹配时才比较。

## 失败风险

- `midcons` 改进可能太小，不能单独支撑 CCF-A claim。
- Long-tail improvement 可能需要训练期 length control，更接近 DreamOn，而不是纯 inference rescue。
- 如果模型 confidence curves 不同，cross-model gains 可能不迁移。
- 文献数字可能因为 prompt format、model family 和 unknown-length assumptions 不同而不可直接比较。

## 执行更新，2026-05-29

当前用户约束：不要启动新 GPU 实验，因为显卡需要留给其他用户。GPU-dependent cross-model task 因此暂停，不是取消。

已完成：

- Task 1 result archival：A6000 pairwise summaries、scoreboard 和 compact reports 已在 `analysis_outputs/a6000_midcons_longrescue/` 与 `docs/results/`。
- Task 2 checkpoint decision：`midcons` 是当前 A6000 best，`795/1033 = 76.96%`，相对 control `+8` wins、`0` losses。
- Task 3 diagnostic design：long-underestimate offline sweep 已完成，记录在 `docs/results/long_underestimate_detector_report.md`。
- Review follow-up：missing comparison labels、bucket registry keys 和 GPU launcher overrides 的 P2 issues 已有 regression tests。

新 CPU-only report：

- `docs/results/2026-05-29_completed_experiment_analysis.md`

GPU 可用前暂停：

- Dream-Coder/DiffuCoder/Dream/LLaDA 的 fresh cross-model runs。

## 立即建议

把 A6000 `midcons` 作为当前 best checkpoint；下一步以 diagnostic-first 方式启动 long-underestimation detector。除非分析发现新的 discriminative signal，不要继续在当前 official-CAL true-long trigger 上消耗 GPU。
