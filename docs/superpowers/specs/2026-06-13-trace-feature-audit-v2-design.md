# Trace Feature Audit V2 Design

Date: 2026-06-13
Branch: `paper-agent-overnight`
Scope: CPU-only offline design for the next long-length recovery diagnostic.

## Objective

Design a smarter `trace_feature_audit_v2` after the first full trace-long-rescue batch produced negative evidence. The goal is not to run another GPU policy immediately. The goal is to determine whether the existing full generation traces contain usable, inference-time visible signals for true-long under-selection that the first hand-written Route 1/2 formulas failed to expose.

The immediate research question is:

> Can trace dynamics identify failed `oracle >= 17` rows with bounded `oracle <= 8` and current-pass risk, using a rule that is explainable enough for a CCF-A review?

The design keeps the main method family pure inference-time and training-free. Learned or fitted models are allowed only as offline discovery instruments unless we explicitly change the paper claim and run a proper train/validation/test protocol.

## Current Evidence

The current `LLaDA-Base` trace batch is healthy but negative for the first route formulas:

- Previous local method trace:
  - output: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`
  - `1033` rows, `35257` trace rows, `769/1033 = 74.44%`
  - `true_long = 113`, `failed_long = 96`
- Current `midcons` trace:
  - output: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`
  - `1033` rows, `35768` trace rows, `795/1033 = 76.96%`
  - `true_long = 113`, `failed_long = 91`

Route 1 and Route 2 both triggered `0` rows on both trace sources. This does not prove the trace features are useless. It proves that the first formula family was too narrow. In particular, the first formula required unstable remaining-mask behavior and decode uncertainty at the same time. A quick diagnostic showed that those two signals exist separately in failed-long rows, but their intersection is empty under the current thresholds.

The important lesson is:

> We should not conclude "no signal"; we should conclude "the first Boolean formula has the wrong shape."

## Why The Learned Model Is A Discovery Tool, Not The Default Method

If a model is trained with labels derived from `oracle_mask_length`, verifier pass/fail, or bucket outcomes and then used at inference time to decide whether to trigger a longer canvas, the resulting method is no longer strictly training-free. That may still be a valid paper direction, but it changes the claim, the baselines, and the evidence burden.

For the current paper direction, a learned model should serve as a microscope:

- It can reveal nonlinear feature interactions that hand-written rules missed.
- It can rank feature families by stability and utility.
- It can suggest compact rules, thresholds, or monotone interactions.
- It cannot by itself become the main inference-time policy unless we reframe the method as a learned length controller.

The main candidate policy from this audit must therefore be a distilled rule or small rule set that uses only inference-time visible features. Labels are allowed only for offline evaluation and model-assisted discovery.

If the learned model is dramatically better than any distilled rule, it should be reported as a diagnostic branch or a future learned-predictor direction, not silently folded into the training-free method.

## Literature-Inspired Design Principles

The design borrows ideas from several adjacent literatures, but adapts them to a small, high-risk, reviewer-facing setting.

### Time-Series Feature Libraries

Trace rows are short time series. Rather than only reading final-step fields, the audit should summarize whole trajectories.

Useful ideas:

- `catch22` / highly comparative time-series analysis: use compact, interpretable time-series characteristics such as trend, distribution, outlier, autocorrelation, and fluctuation descriptors.
- `tsfresh`: generate a broad feature bank and then filter for statistically relevant features.
- Shapelets: search for short local trajectory patterns that distinguish classes, such as late plateau or confidence collapse.

Adaptation:

- Use a small hand-auditable version of these ideas, not a huge dependency-heavy feature factory.
- Features must be named in reviewer-readable language.
- Every feature should specify whether it is endpoint, aggregate, slope, windowed, or shapelet-like.

### Interpretable Model-Assisted Discovery

The audit should use simple models to reveal interactions without making the final method opaque.

Useful ideas:

- RuleFit-style rule ensembles: trees can discover conjunctions, then useful rules can be extracted.
- Explainable Boosting Machine / GA2M-style additive and pairwise terms: reveal smooth single-feature and pairwise effects.
- Sparse logistic or sparse linear models: expose stable feature weights under regularization.
- Shallow decision trees: give a directly readable candidate rule.
- Stability selection and all-relevant feature selection: prefer features that recur across splits, not one lucky threshold.

Adaptation:

- Models are diagnostics, not final policies.
- The final report must include both model score and distilled rule score.
- A discovered rule is not credible unless it survives task-id splits and both trace sources.

### Risk-Controlled Selection

This is not a normal classification task. A high AUC can still be useless if it triggers many short rows. The audit should be formulated as constrained selection:

> Maximize failed-long coverage subject to short-risk and current-pass-risk limits.

Useful ideas:

- Neyman-Pearson style classification: optimize detection under a false-positive constraint.
- Selective classification: report risk-coverage tradeoffs instead of only accuracy.
- Conformal risk control: treat risk constraints explicitly rather than relying on a single point estimate.

Adaptation:

- Primary objective is not accuracy.
- Primary tables should sort by constrained utility:
  - failed-long trigger count,
  - true-long precision,
  - short-risk count and rate,
  - current-pass-risk count and rate,
  - cross-run stability.

## Feature Families

The v2 audit should build feature families from each task's `results.jsonl` row plus its grouped `step_traces.jsonl` rows. Features may use selected length and decode trace dynamics. They must not use oracle length or pass/fail as policy inputs.

### Family A: Endpoint Features

These are direct extensions of the existing v1 features:

- selected length.
- final remaining masks.
- final remaining-mask ratio.
- final mean gap.
- final mean top1.
- last mean confidence.
- stop reason.
- trace step count.

Purpose: preserve interpretability and provide anchors for comparisons with v1.

### Family B: Trajectory Shape Features

These summarize how the trace evolves:

- remaining-mask slope over all steps.
- remaining-mask slope over the last `k` steps.
- confidence slope over all steps.
- confidence slope over the last `k` steps.
- area under remaining-mask curve.
- area under uncertainty curve.
- max, min, median, and interquartile range for confidence/gap/top1 if present.
- number of plateaus in remaining masks.
- longest late plateau.
- last step index where remaining masks decreased.
- fraction of steps with no remaining-mask decrease.

Purpose: detect under-selection patterns that are not visible at the final step.

### Family C: Stop-Reason Conditioned Features

The same numeric trace can mean different things under different stop reasons. The audit should compute split diagnostics by:

- `before_min_stop_step`
- `too_many_remaining_masks`
- `mean_gap_below_threshold`
- `mean_top1_below_threshold`
- any other observed stop reason.

Derived features:

- stop reason as categorical input for model diagnostics.
- stop-reason-specific thresholds.
- interaction terms such as `mean_gap_below_threshold AND selected_len <= 12`.

Purpose: avoid wrong global formulas like the v1 hard `AND`.

### Family D: Relative And Normalized Features

Raw remaining masks may not compare fairly across selected lengths. Add:

- final remaining masks divided by selected length.
- area under remaining-mask curve divided by selected length and step count.
- confidence change divided by initial confidence.
- remaining-mask decay ratio between first half and second half.
- late instability relative to early instability.

Purpose: prevent long selected canvases and short selected canvases from being mixed under one raw scale.

### Family E: Probe-Trace Fusion Features

The project already has probe-curve diagnostics that were not sufficient alone. The v2 audit should test whether trace dynamics become useful when conditioned on probe fields.

Examples:

- trace uncertainty high while probe long score is high.
- selected length short while probe curve has a long-mode bump.
- final trace uncertainty high and long-score margin above threshold.
- disagreement between base/S3 selected length and trace-derived unresolvedness.

Purpose: discover "weak signals become useful in combination" cases.

### Family F: Shapelet-Like Boolean Motifs

Instead of generic trees only, define a small library of reviewer-readable motifs:

- late plateau: remaining masks fail to decrease for the last `m` steps.
- early collapse: confidence drops sharply in early steps.
- unresolved finish: final remaining-mask ratio high despite many decode steps.
- uncertainty-only stop: final gap/top1 is poor even when remaining masks are not high.
- remaining-only stop: remaining masks are high even when confidence looks normal.
- two-phase behavior: early progress followed by late stagnation.

Purpose: recover signals that failed because v1 used only one global conjunction.

## Discovery Layer

The discovery layer should answer not only "what works" but "why it works and whether it is robust."

### Data Splitting

Use deterministic task-id splits so results are reproducible:

- five SHA256 task-id folds.
- discovery folds for model/rule search.
- held-out fold for risk reporting.
- aggregate held-out summary across folds.

The audit should run separately for:

- previous local method trace.
- current `midcons` trace.
- optionally their intersection as a stability check.

### Labels For Offline Evaluation

Labels are not policy inputs. They are offline accounting targets:

- `failed_long`: `oracle >= 17` and current run failed.
- `true_long`: `oracle >= 17`.
- `short_risk`: `oracle <= 8`.
- `current_pass_risk`: current run passed.
- optional `medium`: `9-16` for diagnosing tradeoffs.

### Model Set

The model set should deliberately mix interpretable and discovery-oriented tools:

1. Single-feature constrained thresholds.
2. Pairwise constrained thresholds.
3. Stop-reason-specific thresholds.
4. Shallow decision trees with max depth `2` or `3`.
5. Sparse logistic regression over standardized numeric features.
6. RuleFit-like rule extraction from a small tree ensemble.
7. GA2M/EBM-style additive plus pairwise interaction diagnostics, if dependency cost is acceptable.
8. Stability selection over folds.
9. Shapelet motif scoring over the hand-defined motif library.

The final method candidate should come from items 1-4 or a distilled rule from 6-9. Items 5-9 are mainly for discovery and ablation.

### Objective Function

The audit should not optimize accuracy. Candidate ranking should use constrained utility:

```text
 failed_long_recall_weight * failed_long_trigger_count
+ true_long_precision_weight * true_long_precision
- short_risk_penalty * short_risk_count
- current_pass_risk_penalty * current_pass_risk_count
- instability_penalty * cross_split_variance
```

But this score is only for ranking. Gate decisions should remain thresholded and auditable:

- minimum failed-long triggers.
- maximum short-risk count.
- maximum current-pass-risk count.
- minimum held-out stability.
- no oracle-only inputs.

### Discovery Outputs

For every candidate family, write:

- full candidate table.
- Pareto frontier by failed-long coverage vs short risk.
- top rules per split.
- top stable features.
- top unstable features to reject.
- per-bucket trigger summary.
- "why v1 failed" diagnostic table.
- a small list of distilled policy candidates.

## Candidate Policy Gate

A distilled rule is worth a GPU policy runner only if it satisfies all primary gates on held-out accounting:

- triggers at least `10` failed-long rows on at least one trace source, or at least `8` on both sources.
- short-risk count is at most `5`, with a preferred target at most `2`.
- current-pass-risk count is at most `5`, with a preferred target at most `3`.
- true-long precision is at least `0.45`, or at least `0.35` with strong failed-long recall and very low short risk.
- same qualitative rule works on both previous and `midcons` trace sources.
- the rule can be written in at most three readable clauses.

The strict Gate A for a later GPU policy remains:

- oracle `>=17` bucket improves.
- overall pass count does not decrease.
- oracle `<=8` short-bucket net loss is at most `2` tasks.

The offline audit cannot prove pass-rate improvement. It can only justify or reject a GPU policy run.

## Alternative Paths To Evaluate

The design should not force Route 3 if v2 fails. Instead, the audit should end with one of these decisions:

### Path A: Distilled Trace Gate

Use a compact rule discovered by v2 as a Route 2 long-rescue gate. This is the preferred positive outcome.

### Path B: Trace-Conditioned Length Predictor Diagnostic

If a small learned model is clearly better than all distilled rules, treat it as evidence for a future lightweight length predictor. This changes the paper framing and should require a separate design.

### Path C: Probe-Trace Fusion

If trace alone remains weak but probe-trace interactions are stable, build a fusion gate rather than a trace-only gate.

### Path D: Backbone-Specific Transfer Analysis

If the signal is weak on LLaDA-Base but appears in LLaDA-MoE or LLaDA-1.5 outputs where fields permit, investigate whether true-long detection is model-family dependent.

### Path E: Stop True-Long Rescue Under Current Data

If no stable low-risk signal appears, record the negative result and shift the paper contribution toward medium-length rescue plus protocol-matched backbone evidence.

### Non-Default Path: Multi-Canvas Trace Audit

Route 3 should not be the default fallback. It should only be reconsidered if v2 discovers a trace quality score that is stable enough to justify collecting counterfactual traces for multiple canvas lengths. Without that quality score, multi-canvas reranking is expensive and harder to defend.

## Proposed Implementation Shape

This spec does not implement code. If approved, the implementation plan should create:

- `analysis/trace_feature_audit_v2.py`
- `tests/test_trace_feature_audit_v2.py`
- `analysis_outputs/trace_feature_audit_v2_STAMP/`
- `docs/paper_agent/experiments/TIMESTAMP_trace_feature_audit_v2.md`

The script should accept:

```bash
--prev-results PREV_RESULTS_JSONL
--prev-traces PREV_STEP_TRACES_JSONL
--midcons-results MIDCONS_RESULTS_JSONL
--midcons-traces MIDCONS_STEP_TRACES_JSONL
--output-dir ANALYSIS_OUTPUT_DIR
--folds 5
```

It should be CPU-only and should not launch GPU experiments.

## Expected Report Tables

The terminal and docs report should include:

1. Trace source overview:
   - rows,
   - trace rows,
   - true-long count,
   - failed-long count,
   - short count.
2. Feature coverage:
   - available fields,
   - missing fields,
   - numeric/categorical counts.
3. Top single features.
4. Top pairwise rules.
5. Top stop-reason-conditioned rules.
6. Top shallow-tree distilled rules.
7. Top stable model-discovered features.
8. Pareto frontier:
   - failed-long triggers,
   - short risk,
   - current-pass risk,
   - true-long precision.
9. Candidate decision:
   - `policy_candidate`,
   - `diagnostic_only`,
   - `reject`,
   - `needs_new_data`.

## Success Criteria

The design is successful if the implementation can answer these questions without ambiguity:

- Did v1 fail because the formula was too strict, because the features are weak, or because the available trace is missing the decisive signal?
- Which trace feature families have stable held-out signal?
- Is there a readable rule worth a GPU Route 2 policy runner?
- If not, is a learned length predictor direction justified?
- Should Route 3 remain deprioritized?

## Kill Criteria

Stop and document negative evidence if:

- no candidate family triggers at least `5` failed-long rows under acceptable short risk;
- all useful candidates depend on oracle/pass/verifier labels as inputs;
- held-out performance collapses across task-id folds;
- the only strong signal is an opaque learned model that cannot be distilled;
- implementation would require new GPU data before answering the CPU-only question.

## References

- Lubba et al., "catch22: CAnonical Time-series CHaracteristics", 2019. https://arxiv.org/abs/1901.10200
- Christ et al., "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests", 2016. https://arxiv.org/abs/1610.07717
- Grabocka et al., "Learning Time-Series Shapelets", 2014. https://arxiv.org/abs/1503.03238
- Friedman and Popescu, "Predictive Learning via Rule Ensembles", 2008. https://arxiv.org/abs/0811.1679
- Lou et al., "Accurate Intelligible Models with Pairwise Interactions", 2013. https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf
- Meinshausen and Buhlmann, "Stability Selection", 2010. https://doi.org/10.1111/j.1467-9868.2010.00740.x
- Tong, Feng, and Li, "Neyman-Pearson Classification Algorithms and NP Receiver Operating Characteristics", 2018. https://arxiv.org/abs/1608.03109
- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks", 2017. https://arxiv.org/abs/1705.08500
- Angelopoulos et al., "Conformal Risk Control", 2022. https://arxiv.org/abs/2208.02814
- Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead", 2019. https://doi.org/10.1038/s42256-019-0048-x

## Self-Review

- No placeholder requirements remain.
- The scope is CPU-only offline audit design, not a GPU runner.
- Oracle/pass labels are clearly restricted to offline evaluation and discovery.
- Learned models are positioned as discovery instruments unless the project explicitly changes claim scope.
- Route 3 is explicitly non-default and requires a stable trace quality score before reconsideration.
- The design answers the user's concern that useful features may require better formulas, functions, or discovery models.
