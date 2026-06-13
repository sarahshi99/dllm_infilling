# Open Questions

Updated: 2026-06-01 01:52 CST

## Research Direction

1. Can true-long under-selection be detected with inference-time trajectory features, or is training-time length regularization required?
2. Is the final paper best framed as a positive method paper, a diagnostic-plus-method paper, or a negative result that motivates dynamic canvas/length regularization?
3. Should the central benchmark remain single-line infilling, or must multi-line/FIM evidence be added for CCF-A reviewer appeal?

## Experimental Design

1. Can any constrained, nonlinear, trace-aware, or cross-run probe-curve score reduce short-risk to at most `5%` while keeping at least `10` failed-long triggers? The first simple strict-split linear score failed with `22.22%` held-out short-risk.
2. Is deterministic task-id folding sufficient for a learned length classifier, or is cross-run/cross-model validation required before treating the signal as paper-grade evidence?
3. What trace-enabled smoke size is enough to collect trajectory features and detect short-bucket regression before a full run?
4. Which cross-model comparison is the first apples-to-apples target: Dream-Coder official canvas, LLaDA Instruct, Dream, or DiffuCoder?

## Literature Alignment

1. What exact prompt/canvas/evaluation setting does DreamOn use for HumanEval-Infilling single-line?
2. What exact fully unknown-length setting does LR-DLLM report, and can the local runner reproduce it?
3. Are DreamOn and LR-DLLM results directly comparable, or only suggestive anchors?

## Engineering And Reproducibility

1. Should the next diagnostic move beyond current probe-curve fields to trace-enabled features, or first try a more conservative high-precision score on the existing fields?
2. Should queued GPU experiments on cards `2,3` use an existing wait script or a new manifest-driven launcher with `--save-step-traces` support?
3. How should raw result files be stored if one future run becomes paper-critical: Git LFS, external artifact store, or compact reproduction script only?

## User Decisions Deferred

No immediate user decision is required. A decision will be needed if the project pivots from inference-only rescue to training/fine-tuning or length-regularized modeling.
