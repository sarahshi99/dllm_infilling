# Open Questions

Updated: 2026-05-31 12:36 CST

## Research Direction

1. Can true-long under-selection be detected with inference-time trajectory features, or is training-time length regularization required?
2. Is the final paper best framed as a positive method paper, a diagnostic-plus-method paper, or a negative result that motivates dynamic canvas/length regularization?
3. Should the central benchmark remain single-line infilling, or must multi-line/FIM evidence be added for CCF-A reviewer appeal?

## Experimental Design

1. Which trace fields are available in existing raw outputs, and are they sufficient for trajectory diagnostics without rerunning generation?
2. What split discipline should be used for a learned length classifier so that it does not overfit the `1033` HumanEval tasks?
3. Which cross-model comparison is the first apples-to-apples target: Dream-Coder official canvas, LLaDA Instruct, Dream, or DiffuCoder?
4. What minimum smoke size is enough to detect short-bucket regression before a full run?

## Literature Alignment

1. What exact prompt/canvas/evaluation setting does DreamOn use for HumanEval-Infilling single-line?
2. What exact fully unknown-length setting does LR-DLLM report, and can the local runner reproduce it?
3. Are DreamOn and LR-DLLM results directly comparable, or only suggestive anchors?

## Engineering And Reproducibility

1. Should the next diagnostic script write to `analysis_outputs/paper_agent/` or directly into `docs/paper_agent/`?
2. Should queued GPU experiments use an existing wait script or a new manifest-driven launcher?
3. How should raw result files be stored if one future run becomes paper-critical: Git LFS, external artifact store, or compact reproduction script only?

## User Decisions Deferred

No immediate user decision is required. A decision will be needed if the project pivots from inference-only rescue to training/fine-tuning or length-regularized modeling.
