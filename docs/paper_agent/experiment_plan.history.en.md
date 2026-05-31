# Experiment Plan History

## Plan Revision v0 -> v1

- Timestamp: 2026-05-31 12:36 CST
- Trigger: Start of the long autonomous paper-agent task and required initialization of versioned paper-agent documents.
- Previous plan: No `docs/paper_agent/` plan existed. The nearest prior plan was `docs/superpowers/plans/2026-05-29-ccfa-roadmap-and-next-experiments.md`.
- New plan: Diagnostic-first length modeling plan using A6000 `midcons` as the current checkpoint, stopping further official-CAL true-long GPU runs until a stronger offline signal is found.
- Evidence: `docs/results/a6000_midcons_longrescue_report.md`, `docs/results/long_underestimate_detector_report.md`, `analysis_outputs/experiment_scoreboard.md`, `docs/results/literature_sota_notes.md`, `docs/results/model_generalization_registry.md`, and inspected clean runner/analysis code.
- Reason for change: The A6000 `midcons` result is positive for medium lengths, but true-long gates and offline heuristic sweeps are negative evidence. A CCF-A path needs a stronger length-modeling mechanism.
- Expected benefit: Avoid wasting GPU on known weak trigger families; preserve current positive evidence while moving toward a paper-level method.
- Risk introduced: The plan may require larger engineering work, learned components, or protocol alignment before the next positive result appears.
- What remains unchanged: Raw outputs stay local; compact summaries are tracked; legacy code remains reference-only; new experiments use `expvision_dllm_clean/`, `clean_scripts/`, and `analysis/`.
- Kill criteria affected: Any new long detector must pass offline short-risk and recall gates before GPU evaluation.
- Related commits or files: `docs/paper_agent/*`, existing reports under `docs/results/` and `analysis_outputs/`.
