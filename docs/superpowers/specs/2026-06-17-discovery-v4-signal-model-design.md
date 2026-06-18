# Discovery V4 Signal Model Design

Date: 2026-06-17
Branch: `paper-agent-overnight`
Scope: CPU-first research design for finding and exploiting useful true-long signals after Route2 error analysis.

## Objective

Continue the true-long signal search instead of stopping at Route2's small gain. The goal is to redesign the three-layer Discovery stack so it can find useful signals and convert them into defensible inference-time mechanisms.

The current evidence is mixed:

- Route2 precision `len32` is a clean low-risk gain: `801/1033 = 77.54%`, pairwise `6/0/795/232`.
- All `6` wins are triggered rescue cases, so the trigger has real signal.
- `33` triggered failed-long rows remain failures.
- `31/33` triggered failed-long rows already have rescue length at least oracle length, so blind length increases are not the default answer.
- `56` failed-long rows are missed entirely, so gate recall remains weak.

The immediate research question is:

> Can we discover inference-visible signals that separately predict missed true-long rows and rescue-success/rescue-failure rows, then distill them into a risk-controlled training-free policy?

## Superpowers Alignment

- `superpowers:brainstorming`: used as local protocol fallback to explore multiple signal-discovery paradigms.
- `superpowers:using-git-worktrees`: checked. The current branch is already the dedicated `paper-agent-overnight` branch. This design action does not need another worktree.
- `superpowers:writing-plans`: used as local protocol fallback. The executable plan is `docs/superpowers/plans/2026-06-17-discovery-v4-signal-model-plan.md`.

The current environment does not expose callable `superpowers:*` skill files, so the project protocol is used directly.

## Literature-Inspired Principles

### Reviewer Audit: Method Lineage

The V4 design is not a brand-new learning algorithm from nowhere. It is a project-specific fusion whose individual control and discovery components should be recognizable to reviewers.

| Method family | What can be borrowed directly | Project adaptation | Why it is reasonable | Main limitation |
|---|---|---|---|---|
| Risk-controlled selection | risk-coverage accounting, false-positive constraints, held-out thresholding | optimize failed-long coverage under short-risk and current-pass-risk constraints | the experiment objective is constrained rescue, not ordinary classification | CPU audit can justify a GPU run but cannot prove pass-rate gain |
| Slice / subgroup discovery | readable conjunction search, slice quality functions, coverage/precision tradeoff | mine missed-long, rescued-long, rescue-failure, and risk slices | the current failure is local and rare; global AUC can hide it | multiple testing and hindsight bias must be controlled |
| Rule lists / rule extraction | shallow trees, rule lists, RuleFit-style extracted clauses | distill microscope outputs into at most three inference-visible clauses | CCF-A reviewers can inspect and falsify small rules | tiny gains can overfit without fold stability |
| Time-series feature discovery | trajectory summaries, shapelets, random-kernel probes | summarize decode traces as plateau, collapse, stagnation, and disagreement motifs | v1 failed partly because it used one rigid Boolean shape | random-kernel scores are not final-policy material unless distilled |
| Calibration / OOD / failure detection | confidence residuals, max-probability baselines, disagreement signals | condition confidence on selected length, stop reason, and probe curve | raw trace confidence can be misleading for high-confidence under-length rows | calibration may be backbone-specific |
| Weak supervision | labeling functions, heuristic overlap/conflict analysis | treat probe bumps, trace motifs, stop reasons, and policy disagreement as noisy weak signals | avoids forcing a brittle hand-written AND formula too early | label model itself is not training-free if deployed |
| Uplift / logged-policy diagnostics | action/outcome tables, treatment-effect caution, off-policy caveats | compare primary, precision len24/len32, and broad len24 as partial action evidence | rescue is an action-choice problem, not only a label problem | logs are deterministic and non-random, so causal claims are not allowed |
| MBR / self-consistency style generation | multi-candidate agreement as a verifier-free quality proxy | use only as a possible rescue-quality action if CPU evidence identifies a slice | directly addresses length-sufficient failures where "make it longer" is wrong | extra compute must be justified by a precise gate |

Boundary of novelty:

- Not novel: constrained risk reporting, slice mining, rule extraction, time-series summaries, calibration residuals, and weak-signal fusion as general techniques.
- Novel for this project: the row-action taxonomy for diffusion-code infilling; the split into `MissedLongHead` and `RescueQualityHead`; the use of Route2 variants as partial action evidence; and the distillation of decode/probe dynamics into training-free long-rescue gates.
- Claim discipline: if V4 produces a learned predictor that cannot be distilled, it becomes a separate learned-controller direction rather than evidence for the current training-free method.

### 1. Treat This As Risk-Controlled Selection, Not Accuracy

Selective classification, Neyman-Pearson classification, and conformal risk control all point to the same lesson: the policy should maximize useful coverage subject to explicit false-positive/risk constraints.

Adaptation:

- Primary score is not AUC or accuracy.
- Main tables should report failed-long coverage, short risk, current-pass risk, and policy utility.
- Candidate gates should be calibrated on held-out folds under constraints, not picked from full-data hindsight.

### 2. Mine Error Slices, Not Just Features

Slice discovery and subgroup discovery are closer to the current problem than generic classification. We need to find coherent regions where a mechanism works or fails:

- missed failed-long slices;
- triggered-but-still-failed slices;
- rescued long slices;
- short/medium accidental-win slices;
- current-pass risk slices.

Adaptation:

- Use slice quality functions: coverage, excess failed-long rate, rescue-success rate, short-risk count, current-pass-risk count.
- Search readable conjunctions and small rule lists, not large opaque models.
- Validate slice stability across task-id folds and across available policy variants.

### 3. Use Discovery Models As Microscopes

Models may help find non-linear interactions and time-series motifs, but they should not silently become the paper's main inference-time policy.

Allowed discovery tools:

- shallow trees and RuleFit-like extracted rules;
- sparse logistic/linear scores;
- additive/pairwise models;
- subgroup/rule-list search;
- shapelet or random-kernel time-series probes;
- weak-supervision label models over heuristics;
- uplift/treatment-effect models for rescue usefulness.

Policy constraint:

- final candidate must be distillable into a small readable rule, score, or calibrated selector using inference-visible fields only.

### 4. Separate Gate Recall From Rescue Quality

Route2 error analysis shows two bottlenecks:

- `56` missed failed-long rows: gate recall problem.
- `33` triggered failed-long rows, `31/33` length-sufficient by oracle: rescue generation/selection problem.

V4 must not collapse these into one classifier. It should build two linked discovery heads:

- `MissedLongHead`: find rows that should trigger some additional action.
- `RescueQualityHead`: decide which rescue action or selection rule is likely to help once triggered.

### 5. Be Honest About Counterfactual Limits

The current Route2 logs only observe rescue outcomes for triggered rows. That creates treatment-assignment bias. Uplift/counterfactual ideas are useful, but without randomized or multi-action logs they are diagnostic, not causal proof.

Adaptation:

- Use existing broad/precision/len24/len32 variants as partial multi-action evidence.
- Treat action-value estimates as hypotheses.
- Require new GPU data only after a CPU audit identifies a credible low-risk action family.

## Three-Layer V4 Stack

### Layer 1: Error And Action Anatomy

Layer 1 should build a unified row table across:

- current `midcons` baseline;
- Route2 precision `len32`;
- Route2 precision `len24`;
- Route2 broad `len24`;
- full trace outputs;
- available probe fields.

For each task row, compute:

- baseline pass/fail;
- each policy pass/fail;
- pairwise W/L/TP/TF against baseline;
- trigger status by policy;
- oracle bucket for offline accounting only;
- short/current-pass risk flags for offline accounting only;
- selected length, primary length, rescue length;
- trace endpoint and trajectory features;
- probe curve features;
- stop reason and final source;
- rescue output metadata where available.

Layer 1 outputs:

- `row_action_table.csv`
- `class_taxonomy.csv`
- `action_overlap.csv`
- `policy_delta_by_bucket.csv`
- `length_sufficiency_diagnostic.csv`

Purpose:

- expose where the existing policies agree/disagree;
- distinguish gate recall from rescue generation;
- create a clean substrate for discovery.

### Layer 2: Discovery Model Layer

Layer 2 should run multiple discovery families, each framed as a microscope. It should not ask "which model has highest accuracy?" It should ask "which stable, interpretable signal family survives risk constraints?"

#### Family A: Constrained Slice/Subgroup Discovery

Search small conjunctions over binned inference-visible fields.

Candidate examples:

- selected length short + probe long-mode bump + high trace confidence;
- `no_remaining_masks` stop + low top1 + long plateau;
- high-confidence missed rows with selected length below probe long mode;
- route2 trigger rows where rescue confidence profile predicts failure.

Ranking:

```text
utility =
  failed_long_coverage
  + rescue_win_coverage
  - short_risk_penalty
  - current_pass_risk_penalty
  - instability_penalty
```

This family is the main path for reviewer-readable gates.

#### Family B: Rule Extraction From Shallow Models

Train shallow trees, sparse logistic scores, and small rule ensembles on discovery folds, then extract candidate rules.

Targets:

- `missed_failed_long` among untriggered rows;
- `triggered_failed_long` versus `triggered_rescued_long` among triggered rows;
- `route2_win` versus triggered non-win;
- short/current-pass risk.

Use:

- only to propose rules;
- require held-out fold validation;
- reject rules that use oracle/pass/verifier labels as inputs.

#### Family C: Time-Series Trace Shape Discovery

The existing trace formulas were too narrow. V4 should search shape families:

- late plateau motifs;
- early confidence collapse;
- high-confidence stagnation;
- two-phase progress then stall;
- trace/probe disagreement;
- random-kernel or shapelet-style summaries as discovery aids.

Policy use:

- final signal must be distilled to named motifs or small numeric summaries.

#### Family D: Calibration And OOD-Style Signals

Modern neural calibration and OOD literature suggests that raw confidence is often misleading. V4 should add normalized residuals:

- confidence relative to selected length;
- top1/gap residual conditioned on stop reason;
- energy-like negative log confidence aggregates;
- confidence drop from early to late steps;
- disagreement between probe long-score and trace confidence.

Purpose:

- identify high-confidence missed-long rows that v2's low-confidence trigger missed.

#### Family E: Weak-Supervision Heuristic Fusion

Treat candidate heuristics as noisy labeling functions:

- low top1/plateau;
- probe long bump;
- selected length short relative to probe mode;
- stop reason family;
- trace confidence residual;
- broad-vs-precision policy disagreement.

The fusion model can rank heuristics and discover correlated failures, but the final policy should be a distilled rule set or calibrated score, not an opaque label model.

#### Family F: Counterfactual/Uplift Diagnostics

Frame rescue as an action:

- action `0`: keep primary;
- action `1`: fixed rescue len24;
- action `2`: fixed rescue len32;
- future actions: alternate rescue decoding/selection.

Current data is biased because actions are not randomized. Still, partial evidence can rank hypotheses:

- rows helped by len32 but not len24;
- rows triggered by broad but not precision;
- rows triggered but still failed despite length sufficiency;
- rows where short/medium rescue helps accidentally.

Output should be diagnostic:

- candidate action families;
- required counterfactual data;
- whether another GPU run is justified.

## Layer 3: Policy Distillation And Gates

Layer 3 converts discovered signals into candidate mechanisms:

### Candidate 1: Probe-Trace Fusion Gate

Goal: recover missed failed-long rows without harming short/current-pass rows.

Offline gate:

- triggers at least `8` missed failed-long rows on held-out folds;
- short-risk count at most `3`;
- current-pass-risk count at most `3`;
- stable on at least `4/5` folds;
- rule uses at most three readable clauses.

### Candidate 2: Rescue Quality Selector

Goal: avoid wasting rescue on rows that are likely to fail, or choose a different rescue mode.

Offline gate:

- among triggered rows, separates rescued wins from triggered failures better than current precision gate;
- identifies at least one rescue-failure slice with `>=10` rows;
- proposes an inference-visible alternative action for that slice.

Possible alternatives:

- different decode schedule;
- multiple rescue candidates with trace-confidence selection;
- MBR/consensus-like selection without verifier;
- smaller/larger rescue only if length insufficiency is supported.

### Candidate 3: Conservative Route2 Polish

Goal: preserve the clean `+6` / `0` loss evidence if no stronger signal is found.

Use when:

- no stable missed-long gate appears;
- rescue-quality features do not imply a credible new action;
- Route2 remains only an incremental positive result.

### Candidate 4: Stop Current True-Long Branch

Use when:

- discovered signals are unstable;
- all useful rules depend on oracle/pass labels;
- improvement requires expensive multi-canvas policy without a stable quality score.

## Implementation Shape For Next CPU Audit

Proposed script:

- `analysis/discovery_v4_signal_audit.py`

Proposed tests:

- `tests/test_discovery_v4_signal_audit.py`

Inputs:

```bash
--baseline-results outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl
--baseline-traces outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/step_traces.jsonl
--route2-len32-results outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl
--route2-len24-results outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958/results.jsonl
--route2-broad-results outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958/results.jsonl
--output-dir analysis_outputs/discovery_v4_signal_audit_TIMESTAMP
--folds 5
```

Expected outputs:

- `summary.json`
- `row_action_table.csv`
- `slice_candidates.csv`
- `rule_candidates.csv`
- `trace_shape_candidates.csv`
- `calibration_residuals.csv`
- `uplift_diagnostics.csv`
- `policy_shortlist.md`
- `report.md`

## Success Criteria

The CPU audit succeeds if it can answer:

- Is there a stable inference-visible slice for missed failed-long rows?
- Is there a stable inference-visible slice for triggered rescue failures?
- Is rescue failure mostly generation/selection quality rather than length?
- Which candidate policy should be written as the next GPU action brief, if any?

## GPU Gate

Do not launch GPU experiments from V4 unless all are true:

- a CPU candidate passes held-out risk constraints;
- the mechanism is named and not just "try more length";
- success and kill criteria are written;
- GPU `2/3` are checked with `nvidia-smi` and are not occupied by other users' tasks, unless the user explicitly overrides.

## References

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks", 2017. https://arxiv.org/abs/1705.08500
- Angelopoulos et al., "Conformal Risk Control", 2022. https://arxiv.org/abs/2208.02814
- Tong, Feng, and Li, "Neyman-Pearson Classification Algorithms and NP Receiver Operating Characteristics", 2018. https://arxiv.org/abs/1608.03109
- Lubba et al., "catch22: CAnonical Time-series CHaracteristics", 2019. https://arxiv.org/abs/1901.10200
- Dempster, Petitjean, and Webb, "ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels", 2019. https://arxiv.org/abs/1910.13051
- Grabocka et al., "Learning Time-Series Shapelets", 2014. https://arxiv.org/abs/1503.03238
- Friedman and Popescu, "Predictive Learning via Rule Ensembles", 2008. https://arxiv.org/abs/0811.1679
- Angelino et al., "Learning Certifiably Optimal Rule Lists for Categorical Data", 2017. https://arxiv.org/abs/1704.01701
- Lakkaraju, Bach, and Leskovec, "Interpretable Decision Sets", 2016. https://dl.acm.org/doi/10.1145/2939672.2939874
- Chung et al., "Slice Finder: Automated Data Slicing for Model Validation", 2019. https://arxiv.org/abs/1807.06068
- Ribeiro, Singh, and Guestrin, "Nothing Else Matters: Model-Agnostic Explanations By Identifying Prediction Invariance", 2016. https://arxiv.org/abs/1611.05817
- Liu, Rosen, and G.C., "AutoSlicer: Scalable Automated Data Slicing for ML Model Analysis", 2022. https://arxiv.org/abs/2212.09032
- Christ, Kempa-Liehr, and Feindt, "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests", 2016. https://arxiv.org/abs/1610.07717
- Guo et al., "On Calibration of Modern Neural Networks", 2017. https://arxiv.org/abs/1706.04599
- Hendrycks and Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", 2016. https://arxiv.org/abs/1610.02136
- Lakshminarayanan, Pritzel, and Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", 2017. https://arxiv.org/abs/1612.01474
- Athey and Imbens, "Recursive Partitioning for Heterogeneous Causal Effects", 2016. https://www.pnas.org/doi/10.1073/pnas.1510489113
- Dudik, Langford, and Li, "Doubly Robust Policy Evaluation and Learning", 2011. https://arxiv.org/abs/1103.4601
- Swaminathan and Joachims, "Counterfactual Risk Minimization: Learning from Logged Bandit Feedback", 2015. https://arxiv.org/abs/1502.02362
- Ratner et al., "Snorkel: Rapid Training Data Creation with Weak Supervision", 2017. https://arxiv.org/abs/1711.10160
- Benjamini and Hochberg, "Controlling the False Discovery Rate", 1995. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Meinshausen and Buhlmann, "Stability Selection", 2010. https://doi.org/10.1111/j.1467-9868.2010.00740.x
- Eikema and Aziz, "Sampling-Based Approximations to Minimum Bayes Risk Decoding for Neural Machine Translation", 2021. https://arxiv.org/abs/2108.04718
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", 2022. https://arxiv.org/abs/2203.11171
- Manakul, Liusie, and Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models", 2023. https://arxiv.org/abs/2303.08896

## Self-Review

- The design does not abandon true-long signal search.
- The design avoids treating learned discovery models as the final policy by default.
- The design separates gate recall from rescue generation/selection quality.
- The design includes cross-domain methods beyond the prior v2 feature enumeration.
- The design preserves the training-free/inference-time claim unless the project explicitly changes direction.
