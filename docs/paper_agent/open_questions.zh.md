# Open Questions

更新时间：2026-06-01 01:52 CST

## Research Direction

1. True-long under-selection 能否用 inference-time trajectory features 检测，还是必须使用 training-time length regularization？
2. 最终论文更适合作为 positive method paper、diagnostic-plus-method paper，还是一个推动 dynamic canvas/length regularization 的 negative result？
3. Central benchmark 是否应保持 single-line infilling，还是必须加入 multi-line/FIM evidence 以提升 CCF-A reviewer appeal？

## Experimental Design

1. 是否存在 constrained、nonlinear、trace-aware 或 cross-run probe-curve score，能把 short-risk 降到至多 `5%`，同时保留至少 `10` 个 failed-long triggers？第一个 simple strict-split linear score 已失败，held-out short-risk 为 `22.22%`。
2. Deterministic task-id folding 对 learned length classifier 是否足够，还是必须使用 cross-run/cross-model validation 后才能把该 signal 视为 paper-grade evidence？
3. Trace-enabled smoke size 应多大，才足以收集 trajectory features 并在 full run 前检测 short-bucket regression？
4. 第一个 apples-to-apples cross-model comparison 应选择哪个目标：Dream-Coder official canvas、LLaDA Instruct、Dream，还是 DiffuCoder？

## Literature Alignment

1. DreamOn 在 HumanEval-Infilling single-line 上使用的精确 prompt/canvas/evaluation setting 是什么？
2. LR-DLLM 报告的 fully unknown-length setting 具体是什么，本地 runner 能否复现？
3. DreamOn 与 LR-DLLM results 是 direct comparable，还是只能作为 suggestive anchors？

## Engineering And Reproducibility

1. 下一项 diagnostic 应从当前 probe-curve fields 转向 trace-enabled features，还是先在现有 fields 上尝试更保守的 high-precision score？
2. 在卡 `2,3` 上排队的 GPU experiments 应使用已有 wait script，还是新增支持 `--save-step-traces` 的 manifest-driven launcher？
3. 如果未来某个 run 成为 paper-critical，raw result files 应如何保存：Git LFS、external artifact store，还是只保存 compact reproduction script？

## User Decisions Deferred

当前不需要用户立即决策。如果项目从 inference-only rescue pivot 到 training/fine-tuning 或 length-regularized modeling，则需要用户决策。
