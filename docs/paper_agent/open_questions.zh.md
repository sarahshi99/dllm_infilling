# Open Questions

更新时间：2026-05-31 15:24 CST

## Research Direction

1. True-long under-selection 能否用 inference-time trajectory features 检测，还是必须使用 training-time length regularization？
2. 最终论文更适合作为 positive method paper、diagnostic-plus-method paper，还是一个推动 dynamic canvas/length regularization 的 negative result？
3. Central benchmark 是否应保持 single-line infilling，还是必须加入 multi-line/FIM evidence 以提升 CCF-A reviewer appeal？

## Experimental Design

1. 什么 multivariate probe-curve score 能把最佳 single-feature 的 short-risk 从 `8.70%` 降到至多 `5%`，同时保留至少 `10` 个 failed-long triggers？
2. Learned length classifier 应使用什么 split discipline，才能避免对 `1033` HumanEval tasks 过拟合？
3. Trace-enabled smoke size 应多大，才足以收集 trajectory features 并在 full run 前检测 short-bucket regression？
4. 第一个 apples-to-apples cross-model comparison 应选择哪个目标：Dream-Coder official canvas、LLaDA Instruct、Dream，还是 DiffuCoder？

## Literature Alignment

1. DreamOn 在 HumanEval-Infilling single-line 上使用的精确 prompt/canvas/evaluation setting 是什么？
2. LR-DLLM 报告的 fully unknown-length setting 具体是什么，本地 runner 能否复现？
3. DreamOn 与 LR-DLLM results 是 direct comparable，还是只能作为 suggestive anchors？

## Engineering And Reproducibility

1. 下一项 diagnostic script 应写入 `analysis_outputs/paper_agent/`，还是直接写入 `docs/paper_agent/`？
2. 在卡 `2,3` 上排队的 GPU experiments 应使用已有 wait script，还是新增支持 `--save-step-traces` 的 manifest-driven launcher？
3. 如果未来某个 run 成为 paper-critical，raw result files 应如何保存：Git LFS、external artifact store，还是只保存 compact reproduction script？

## User Decisions Deferred

当前不需要用户立即决策。如果项目从 inference-only rescue pivot 到 training/fine-tuning 或 length-regularized modeling，则需要用户决策。
