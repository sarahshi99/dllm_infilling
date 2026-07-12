# 文献与 SOTA 笔记

更新时间：2026-07-12

## Phase 5 方法组合的 source-checked 边界

1. FIM/InCoder 已建立双向 infilling conditioning：`https://arxiv.org/abs/2207.14255`、`https://arxiv.org/abs/2204.05999`。Semantic Bridge 的 novelty 不能是“看 prefix 和 suffix”，而必须是可审计 semantic obligations 与 candidate ranking advantage。
2. Synchromesh/CSD 与 Constrained Discrete Diffusion 已覆盖 constrained generation：`https://arxiv.org/abs/2201.11227`、`https://arxiv.org/abs/2503.09790`。Constraint-Homotopy 必须证明 gradual schedule 的独立作用；本轮不实现。
3. Birth–death sampling 与 diffusion SMC guidance 已有先例：`https://arxiv.org/abs/2211.00450`、`https://arxiv.org/abs/2601.21104`。Birth–Death Canvas Diffusion 目前只是内部 hypothesis；本轮不实现。
4. MBR-EXEC 说明 candidate semantics/selection 常通过 execution 获益：`https://arxiv.org/abs/2204.11454`。本项目 deployable method 禁止 execution/unit-test outcome，因此 Semantic Particle Assembly 必须与 execution-aware selection 分开；本轮不实现。
5. ReCode 使用 semantics-preserving transformations 测 robustness：`https://arxiv.org/abs/2212.10264`。Phase 5 alpha-renaming F2 只测 controlled equivariance predictive increment；稳定性不等于正确性。

Phase 5 full shared bank 已完成：M1-D0 fixed AST/def-use proxy 的 cross-canvas pairwise accuracy 为 `0.6273`，选择相对 fixed64/confidence 的 net 为 `+15/+14`，但没有通过相对所有 deterministic baselines 的 preregistered grouped-bootstrap lower-bound gate。A1 equivariance 为负结果；M4-D0 只提供有限 assembly premise。不得把这些 diagnostics 写成四个正式方法已经实现。

截至 2026-07-12，official CAL、rho-EOS 和 DreamOn 均有公开代码：

- CAL：`https://github.com/NiuHechang/Calibrated_Adaptive_Length`；
- rho-EOS：`https://github.com/yjyddq/rho-EOS`；
- DreamOn：`https://github.com/DreamLM/DreamOn`。

因此 external baseline gap 的 blocker 已从“普遍无代码”变为“尚未完成本地 protocol audit/adapter/full reproduction”。LR-DLLM 仍是 paper-audited、code/adapter-blocked。Local CAL/CAL-lite 不能称为 official CAL。

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

## Remasking / Candidate Exploration 锚点（2026-07-02 source-check）

这些文献直接约束本项目的 novelty 边界：不能把“把 token 重新置回 mask 再 refine”本身写成本项目原创。

- Token-to-Mask / T2M remasking:
  - "Remask, Don't Replace: Token-to-Mask Refinement in Masked Diffusion Language Models"
  - arXiv: https://arxiv.org/abs/2604.18738
  - 核心相关性：将可疑 committed token reset 到 mask state，再让 diffusion process 重新预测；训练外、修改 decoding/editing rule 的方向。
- Targeted Remasking:
  - "Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement in Discrete Diffusion Language Models"
  - arXiv: https://arxiv.org/abs/2605.26436
  - 核心相关性：把 T2M 做成 targeted remasking，并讨论 probability-based、trigger-mirrored、temporal-difference-based 等 error detection strategies。
- Self-reflective remasking / RemeDi:
  - "Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models"
  - arXiv: https://arxiv.org/abs/2509.23653
  - HF model card: https://huggingface.co/maple-research-lab/RemeDi-Instruct
  - 核心相关性：把 remasking 作为 mask-based DLM 的机制之一，并通过 per-token confidence / remask-aware pipeline 支持 iterative refinement。
- Candidate quality-exploration tradeoff:
  - "Locally Confident, Globally Stuck: The Quality-Exploration Dilemma in Diffusion Language Models"
  - arXiv: https://arxiv.org/abs/2604.00375
  - 核心相关性：指出 low-confidence / confidence-prioritized remasking 可提升 single-sample quality，但可能压制 exploration 和 Pass@k；这与本项目 Phase 1b 的 multi-seed/candidate-diversity ceiling 直接相关。

本项目 2026-07-02 的 `F_oracle_sufficient_trace_remask` 只能定位为 diagnostic baseline：它用内部 trace 中的 token flip count 和 final confidence 做固定 remask rule，以测试 hard cases 是否存在 candidate exploration ceiling；不能宣称 remasking 是本项目原创贡献。

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

## Novelty Boundary After Phase 1b

截至 2026-07-02，source-checked 文献边界如下：

- Remasking / token-to-mask refinement 已有 T2M、Targeted Remasking 和 RemeDi 等明确先例；本项目不能把 remasking 本身作为原创点。
- Length calibration / variable-length inference 已有 CAL 与 LR-DLLM；本项目不能把“用 confidence/probe 做长度校准”单独作为原创点。
- Dynamic canvas / beyond fixed-size canvas 已有 DreamOn，并且直接面向 code infilling；本项目不能简单声称“动态 canvas”本身新颖。
- 本项目潜在原创性应集中在：joint modeling of canvas adequacy and rescue adequacy，以及 risk-controlled selective action under bounded inference cost。
- Phase 1b negative/diagnostic evidence 支持把论文问题从“选更长长度”转为“何时需要 canvas、何时 rescue 真的会产生可选正确候选、何时 intervention risk 可控”。
