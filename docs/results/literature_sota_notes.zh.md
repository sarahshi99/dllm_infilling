# 文献与 SOTA 笔记

更新时间：2026-05-29

本文记录经过 source-check 的比较锚点，用于未来 SOTA 主张。当前项目尚不声称 SOTA。

## 主要 Infilling 锚点

DreamOn 是目前找到的、与 diffusion-model code infilling 最相关的直接对比：

- 论文："DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas"
- arXiv: https://arxiv.org/abs/2602.01326
- OpenReview PDF: https://openreview.net/pdf?id=EQTPmqukiU
- 来源显示的 venue/status：ICLR 2026。
- Benchmarks：HumanEval-Infilling single-line 和 multi-line pass@1，以及 SantaCoder-FIM exact match。
- 论文报告的 evaluation details：official HumanEval-Infilling evaluation scripts；diffusion baselines 使用 fixed mask length 64；DreamOn 动态调整长度。

论文 Table 1 报告：

| Model | HumanEval-Infilling Single-Line Pass@1 | HumanEval-Infilling Multi-Line Pass@1 | SantaCoder-FIM EM |
|---|---:|---:|---:|
| Deepseek-Coder-6.7B | 73.0 | 45.7 | 76.3 |
| Seed-Coder-8B | 89.7 | 59.3 | 77.2 |
| Qwen2.5-Coder-7B | 92.6 | 58.7 | 79.8 |
| LLaDA-8B | 48.3 | 21.1 | 35.1 |
| Dream-7B | 48.2 | 21.9 | 60.3 |
| Dream-7B + DreamOn | 88.6 | 53.3 | 73.8 |
| DiffuCoder-7B | 53.7 | 45.0 | 58.0 |
| DiffuCoder-7B + DreamOn | 92.2 | 63.1 | 77.4 |
| DreamCoder-7B | 55.5 | 43.2 | 59.3 |
| DreamCoder-7B + DreamOn | 92.1 | 63.8 | 79.0 |

## 直接 CAL 锚点

CAL 是当前本地项目最接近的方法学前身：

- 论文："Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly"
- arXiv: https://arxiv.org/abs/2602.00476
- 核心思想：通过 denoising confidence 推断 infilling length，而不是要求 oracle length 或 fixed canvas。
- 摘要层面报告结果：CAL 在 code infilling 中相对 fixed-length baselines 最高提升 `47.7%` Pass@1。

这与本地 LCAS/LCAL stack 直接相关：两者都从 confidence/probe behavior 推断 length choice，然后通过 bounded repair 和 bucket-specific controls 提高安全性。

## 如何比较我们的 Runs

在确认 exact benchmark split、prompt format、decoding budget、model base 和 metric 与文献设置一致之前，当前 LCAL/LCAS 实验不应被称为 SOTA。

当前本地 runs 应被视为：

- 当模型、数据集、硬件、runner 和环境匹配时，是直接 internal comparisons。
- 当只有 benchmark family 匹配时，是启发性的 literature comparisons。
- 当 model family、prompt/canvas format、oracle length availability 或 evaluation subset 不同时，不可直接比较。

未来任何 claim 都必须记录：

- Source paper 或 official repository URL。
- Dataset 和 split。
- Metric 和 pass@k。
- Model name、parameter count 和 checkpoint type。
- Infilling format，以及使用 oracle length、fixed mask length、learned length 还是 dynamic length control。
- Decode steps、temperature/top-p 和 max mask/canvas length。
- 我们的结果是 single-line、multi-line 还是 mixed。

## 当前本地 Cross-Model 证据

本地 registry `docs/results/model_generalization_registry.md` 记录了既有 full Dream-Coder 和 LLaDA runs。这些结果对 transfer planning 有用，但还不足以支撑论文级 SOTA claim，因为 prompt/canvas format 在不同模型族之间并不一致。

重要本地记录：

- `Dream-org/Dream-Coder-v0-Instruct-7B`, official-canvas `alpha010_cap24`: `848/1033 = 82.09%`。
- `Dream-org/Dream-Coder-v0-Base-7B`, official-canvas `alpha010_cap24`: `825/1033 = 79.86%`。
- `GSAI-ML/LLaDA-8B-Instruct`, LCAS v3 resume: `817/1033 = 79.09%`。

这些结果应在统一 A6000 protocol 下重跑后，再与新的 A6000 LCAL candidates 比较。

## 当前本地 A6000 锚点

当前最佳同硬件 LLaDA-Base run：

- `full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626`
- model: `GSAI-ML/LLaDA-8B-Base`
- local protocol: `HumanEval-SingleLineInfilling`, `1033` tasks
- result: `795/1033 = 76.96%`
- pairwise control: 相对 A6000 union control 为 `+8` wins、`0` losses

这是强 internal checkpoint，但不是 external SOTA claim。DreamOn 在 DreamCoder/DiffuCoder 上通过 dynamic canvas control 报告了更高 single-line numbers；LR-DLLM 则更直接处理 unknown-length generation。因此有用的比较是方法学层面的：我们的结果支持显式 length modeling 的必要性，但下一步 paper-level 实验必须先对齐 prompt/canvas/evaluation settings。

## 另一个 2026 锚点：LR-DLLM

LR-DLLM 是另一个相关的 variable-length inference 论文：

- 论文："Improving Variable-Length Generation in Diffusion Language Models via Length Regularization"
- arXiv: https://arxiv.org/abs/2602.07546
- 摘要报告设置：fully unknown-length HumanEvalInfilling 和四语言 McEval。
- 摘要 headline result：在 fully unknown lengths 的 HumanEvalInfilling 上 `51.3%` Pass@1，相对 DreamOn `+13.4%`。

这篇论文相关性很高，因为它把 length selection 视为 inference-time confidence-bias problem，接近本项目核心问题。但在匹配 dataset split、prompt/canvas format 和 unknown-length assumptions 前，不能直接与当前 `HumanEval-SingleLineInfilling` 本地 protocol 比较。
