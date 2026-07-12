# 文献与 Novelty 边界笔记

更新时间：2026-07-12

## Phase 5 独立方法的最近邻与 Novelty 边界

- Semantic Bridge Projection：FIM 与 InCoder 已明确证明 prefix/suffix bidirectional context 对 infilling 有用，因此“使用双向上下文”本身不新。最近邻：`Efficient Training of Language Models to Fill in the Middle`（https://arxiv.org/abs/2207.14255）与 `InCoder: A Generative Model for Code Infilling and Synthesis`（https://arxiv.org/abs/2204.05999）。Phase 5 只有在 inference-visible prefix-forward/suffix-backward semantic obligations 能在同一 candidate bank 上产生独立 ranking advantage 时才有方法价值。
- Constraint-Homotopy Infilling：constraint-aware generation 已有 Synchromesh 的 Constrained Semantic Decoding（https://arxiv.org/abs/2201.11227）与 Constrained Discrete Diffusion（https://arxiv.org/abs/2503.09790）。Novelty 不能是“给 diffusion 加约束”，必须是 code-infilling-specific gradual constraint schedule。本轮只登记，不实现。
- Birth–Death Canvas Diffusion：birth–death interacting-particle sampling 已有理论工作（https://arxiv.org/abs/2211.00450），training-free diffusion guidance 也已有 SMC/MLMC framework（https://arxiv.org/abs/2601.21104）。内部名称不能暗示这些理论自动适用于 discrete code canvas；未来必须给 equal-compute accounting。本轮只登记，不实现。
- Semantic Particle Assembly：candidate-set semantic selection 的直接近邻包括 MBR-EXEC（https://arxiv.org/abs/2204.11454），但它依赖 execution/simulated execution。Phase 5 禁止 unit tests/oracle/reference 进入 deployable selection，因此未来 assembly 必须来自纯 inference-visible landmarks。本轮只登记，不实现。
- Metamorphic Equivariance：ReCode（https://arxiv.org/abs/2212.10264）使用 semantics-preserving transformations 评估 code-generation robustness。Phase 5 仅使用严格 local alpha-renaming，并明确 `equivariance/stability != correctness`；它只能是 auxiliary evaluator。

Phase 5 empirical boundary：shared bank 已完成。Current fixed AST/def-use proxy 虽有 cross-canvas within-task accuracy `0.6273` 与正 paired selection net，但没有通过相对所有 deterministic baselines 的 preregistered grouped-bootstrap lower-bound gate；alpha equivariance delta AUC 的 95% CI 也跨 `0`。因此本轮不形成新的 numerical method-performance claim，current V0 被 kill。Novelty 判断仍只能支持“program-structure signal 值得以新机制继续研究”，不能声称已得到 standalone Semantic Bridge method。

本文件是 `docs/results/literature_sota_notes.zh.md` 的 paper-agent compact 版本，记录 Phase 1b 后最需要约束论文表述的 source-checked 文献。不要把这里的 reported numbers 当成本地同协议 baseline。

## 长度与 Dynamic Canvas 锚点

- CAL: "Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly"
  - arXiv: https://arxiv.org/abs/2602.00476
  - Official code: https://github.com/NiuHechang/Calibrated_Adaptive_Length
  - 相关性：training-free calibrated adaptive length，用 denoising confidence / search 近似 optimal infilling length。
- rho-EOS: "Training-free Bidirectional Variable-Length Control for Masked Diffusion LLMs"
  - arXiv: https://arxiv.org/abs/2601.22527
  - Official code: https://github.com/yjyddq/rho-EOS
  - 相关性：在统一 denoising loop 中用 implicit EOS density 做双向 expansion/contraction；公开 quick-start 主要是 completion/generation setting，是否能公平适配 infilling 仍需 protocol audit。
- LR-DLLM: "Improving Variable-Length Generation in Diffusion Language Models via Length Regularization"
  - arXiv: https://arxiv.org/abs/2602.07546
  - 相关性：把 generation length 显式作为 variable，并在 inference time 做 length-regularized decision。
- DreamOn: "Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas"
  - arXiv: https://arxiv.org/abs/2602.01326
  - OpenReview: https://openreview.net/pdf?id=EQTPmqukiU
  - Code: https://github.com/DreamLM/DreamOn
  - 相关性：直接面向 code infilling 的 dynamic variable-length generation；不能把“beyond fixed-size canvas”作为本项目未限定的新颖点。

Baseline execution boundary：CAL、rho-EOS、DreamOn 的代码可用性已经确认，但当前仓库尚未完成同 model/data/prompt/budget/evaluator 的 official baseline pack。DreamOn 是 training-based，应与 training-free CAL/rho-EOS/LR-DLLM 分层。LR-DLLM 当前仍缺 executable official/local Stage I/II adapter。

## Remasking / Candidate Exploration 锚点

- T2M: "Remask, Don't Replace: Token-to-Mask Refinement in Masked Diffusion Language Models"
  - arXiv: https://arxiv.org/abs/2604.18738
  - 相关性：将 suspect token reset 到 mask state 后重新预测，是 token-to-mask refinement 的直接先例。
- Targeted Remasking: "Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement in Discrete Diffusion Language Models"
  - arXiv: https://arxiv.org/abs/2605.26436
  - 相关性：提出 targeted T2M remasking，并使用 probability/trigger/temporal-difference 类 detection strategies。
- RemeDi: "Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models"
  - arXiv: https://arxiv.org/abs/2509.23653
  - Model card: https://huggingface.co/maple-research-lab/RemeDi-Instruct
  - 相关性：把 remasking 作为 mask-based DLM 的 self-reflective refinement 机制。
- Quality-exploration tradeoff: "Locally Confident, Globally Stuck: The Quality-Exploration Dilemma in Diffusion Language Models"
  - arXiv: https://arxiv.org/abs/2604.00375
  - 相关性：low-confidence / confidence-prioritized remasking 可能提高 Pass@1 但压制 exploration；这直接关系到 Phase 1b 的 candidate-diversity ceiling。

## Phase 1b Novelty 边界

- `F_oracle_sufficient_trace_remask` 只是 diagnostic baseline，不是原创 remasking 方法。
- Remasking、self-reflective remasking、T2M refinement 均已有明确相关工作；本项目不能以此作为主贡献。
- CAL/LR-DLLM 已覆盖 length calibration / variable-length inference 的重要部分；本项目不能只声称“长度校准”新颖。
- DreamOn 已直接处理 code infilling beyond fixed-size canvas；本项目不能把 dynamic canvas 本身写成空泛原创点。
- 更稳妥的潜在贡献位置是：joint modeling of canvas adequacy and rescue adequacy，以及 risk-controlled selective action，在固定预算下控制 rescue harm 并解释 candidate-existence ceiling。

## 与当前证据的连接

Phase 1b `analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/` 的结果为 `candidate_diversity_without_correctness`：

- `113/L3` 中 no-early-commit 产生新 hash，但仍为 `UnitTestFailure`。
- `85/L0` 中 C/E/F 三个 seed 均同 hash，均为 `SyntaxError`。
- trace-remask 执行真实 remask/refinement，但没有改变最终 hash。

因此当前证据只能说：当前 Route2-based generation family 存在有限候选探索空间，但 hard true-long cases 未出现正确候选；不能据此宣称 deployable controller、remasking novelty、true-long 解决或 backbone 最终无能力。
