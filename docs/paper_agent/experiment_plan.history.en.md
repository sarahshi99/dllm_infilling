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

## Plan Revision v1 -> v2

- Timestamp: 2026-05-31 12:58 CST
- Trigger: Fresh inspection of the A6000 `midcons` raw output showed that full-run results contain probe-curve fields but no saved `stopping_trace` or `step_traces`; a CPU-only probe-curve audit was then run.
- Previous plan: E1 listed trajectory summaries and probe-curve shape features as candidate diagnostics, with trajectory diagnostics appearing first.
- New plan: Prioritize probe-curve diagnostics and learned scoring from existing outputs before trajectory diagnostics. Trajectory diagnostics now require a future trace-enabled smoke run because the current full run has `rows_with_stopping_trace = 0`.
- Evidence: `analysis/analyze_probe_curve_long_signals.py`, `docs/paper_agent/probe_curve_signal_audit.md`, `docs/paper_agent/probe_curve_signal_audit.json`, and raw output key inspection for the A6000 `midcons` run.
- Reason for change: Existing `results.jsonl` files support probe-curve analysis immediately, but do not contain trajectory traces. The probe-curve audit found `4106` evaluated thresholds and `0` strict viable single-feature thresholds.
- Expected benefit: Avoids planning a trajectory analysis that cannot be done from current artifacts; keeps progress CPU-only and evidence-driven while GPUs are occupied.
- Risk introduced: Single-feature probe-curve diagnostics may be too weak; the next method may need multivariate learned scoring or a trace-enabled smoke rerun.
- What remains unchanged: No GPU run should launch before offline gates pass; `midcons` remains the current A6000 checkpoint.
- Kill criteria affected: A single-feature probe-curve threshold is not GPU-eligible unless short-risk is at most `5%` and failed-long trigger count is at least `10`.
- Related commits or files: `analysis/analyze_probe_curve_long_signals.py`, `tests/test_analyze_probe_curve_long_signals.py`, `docs/paper_agent/probe_curve_signal_audit.*`.

## Plan Revision v2 -> v3

- Timestamp: 2026-05-31 15:24 CST
- Trigger: User clarified that future experiments should use GPU cards `2,3`, not `0,1,2,3`, while avoiding interruption of other users' running jobs.
- Previous plan: Trace-enabled smoke and any later GPU run would use GPU `0,1,2,3` when available.
- New plan: Future GPU experiments must use `CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false` unless the user explicitly changes the allocation. The agent should wait or queue instead of killing, preempting, or interrupting existing processes.
- Evidence: Direct user instruction in this session; prior `nvidia-smi` check showed existing Python jobs on all four cards, so safe scheduling remains necessary.
- Reason for change: The compute allocation constraint is now narrower than the initial plan. Respecting GPU ownership is part of reproducibility and collaboration safety.
- Expected benefit: Avoids interfering with work on cards `0,1` and keeps future experiment manifests aligned with the user's intended hardware allocation.
- Risk introduced: Full runs on two cards may be slower than the earlier four-card plan, and any same-hardware comparison must clearly name the `2,3` GPU set.
- What remains unchanged: No GPU run should launch before offline gates pass; `midcons` remains the current A6000 checkpoint; trace-enabled smoke runs still require a stronger offline signal.
- Kill criteria affected: None of the scientific gates change. The operational gate now additionally requires GPU `2,3` availability or a non-disruptive wait/queue mechanism.
- Related commits or files: `docs/paper_agent/experiment_plan.current.*.md`, `docs/paper_agent/paper_agent_dashboard.*.md`, `docs/paper_agent/open_questions.*.md`, `docs/paper_agent/overnight_log.*.md`.
