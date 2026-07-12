# Research Design v1: Length-Controlled Diffusion Code Infilling

Created: 2026-05-31 12:36 CST

## Canonical Route Correction (2026-07-12)

`docs/paper_agent/ccfa_master_roadmap.zh.md` is authoritative. Track A closes protocol-matched baselines, grouped statistics, external evaluation, and paper-integrity gaps; Track B incubates M1--M4 independently. F1--F4 map only to M4-D0/A1/M1-D0. The central claim is diagnostic and action-specific, not a unique backbone-cause claim. The 1033 spans are development-only and the 6707 spans correspond to 148 base-task clusters.

## Phase 5 Result Update (2026-07-12)

The shared RandomSpanLight bank completed with base `1332/1332` and alpha auxiliary `728/728`, while frozen test evaluation count remained `0`. The fixed combined AST/def-use proxy reached cross-canvas within-task accuracy `0.6273` and positive paired selection nets, but failed the preregistered requirement that every deterministic-baseline grouped-bootstrap delta lower bound exceed zero. The current proxy V0 is killed; Phase 5 decision is `iterate`. This does not create a positive standalone-method claim or change the diagnostic/mixed central claim. Any continuation requires a new independent preregistration and may not tune the current formula on these outcomes.

## Problem Statement

Diffusion language models for code infilling need an infill length or canvas before denoising. The current project shows that length choice is not a secondary implementation detail: it is a dominant failure mode for `HumanEval-SingleLineInfilling`, especially when the correct middle span is longer than the policy-selected mask length.

The reviewer-recognizable research problem is:

> How can a DLLM choose or adapt its infilling length without oracle length information, while preserving short-completion safety and improving long or medium under-selection?

## Research Community As User

The effective users are code-generation researchers, DLLM researchers, reviewers, and authors of length-control baselines. Their need is not another local heuristic, but a reproducible answer to a recognized gap: fixed canvases and oracle lengths are unrealistic, while naive confidence-based length selection underestimates hard long cases.

## Minimum Publishable Contribution

The minimum publishable contribution is a method and evaluation package that proves one of the following under a matched protocol:

1. A principled inference-time length-control method improves DLLM code infilling over fixed, oracle-ablated, CAL-like, and LCAS/LCAL internal baselines without short-bucket regression.
2. A negative but rigorous result shows that confidence-curve-only inference rescue is insufficient for true-long code infilling, and introduces a stronger trajectory-based or learned length signal that closes part of the gap.
3. A dynamic-canvas or length-regularized method reaches competitive literature-level performance under apples-to-apples settings.

## Current Central Claim

Same-hardware A6000 evidence supports the following narrow claim:

> Inference-time confidence-curve agreement can safely recover medium-length DLLM code infilling cases, but true-long code infilling remains dominated by length underestimation and requires stronger length modeling than the current official-CAL trigger family.

This is not yet a CCF-A central claim. It is a credible empirical foothold and a useful paper narrative constraint.

## Evidence Base

- A6000 control: `787/1033 = 76.19%`.
- A6000 `midcons`: `795/1033 = 76.96%`, `+8` wins and `0` losses against the same-hardware control.
- Gains are concentrated in `9-12` and `13-16` oracle buckets.
- Long buckets remain unchanged: `17-24 = 20.73%`, `25+ = 16.13%`.
- In the A6000 `midcons` run, most failed `oracle >= 17` cases are under-selected; prior reports record `90/91` failed long cases as under-selected and `71/91` still ending from `base`.
- The offline long-underestimate sweep evaluated `16776` rules and found no safe rule satisfying true-long precision, short-risk, and recall criteria.
- The probe-curve single-feature audit evaluated `4106` thresholds and found `0` strict viable thresholds. The current full run has probe-curve features for all rows but no saved stopping traces.
- Literature notes identify DreamOn and LR-DLLM as stronger length-control anchors; current local results must not be called SOTA before protocol matching.

## Novelty Hypothesis

The strongest novelty path is not the current `midcons` rule by itself. The novelty must come from explaining and fixing the boundary between safe medium rescue and unsafe long rescue:

- medium under-selection can be corrected when short-safe confidence agreement exists;
- true-long under-selection is not captured by the same confidence fields;
- multivariate probe-curve scoring, denoising trajectories, learned length classification, dynamic canvas resizing, or length regularization may provide the missing signal.

This is a hypothesis until validated by new experiments.

## Approaches Considered

### Approach A: Consolidate Medium Rescue

Use `midcons` as the core contribution and present it as a short-safe inference-time rescue.

Pros:

- Already has same-hardware positive evidence.
- Easy to reproduce and ablate.
- Low implementation risk.

Cons:

- Improvement is small.
- Does not solve long infilling.
- Too heuristic for a strong CCF-A paper if used alone.

### Approach B: Continue Heuristic True-Long Gates

Loosen or retune official-CAL true-long gates.

Pros:

- Minimal code change.
- Uses existing runner and result fields.

Cons:

- Current evidence is negative.
- Offline sweep shows high short-risk for usable trigger counts.
- Likely wastes GPU budget.

### Approach C: Build Stronger Length Modeling

Add trajectory features, a learned length classifier, dynamic-canvas behavior, or length regularization, then compare against `midcons` and literature anchors.

Pros:

- Most plausible path to a paper-level contribution.
- Directly targets the observed failure mode.
- Can be evaluated diagnostic-first before full GPU runs.

Cons:

- Higher engineering and experimental risk.
- May require broader protocol alignment or training/fine-tuning decisions.
- Needs careful ablation to avoid becoming a black-box patch.

Recommendation: choose Approach C as the paper direction, while preserving Approach A as the current reproducible checkpoint and rejecting Approach B unless new diagnostics contradict the current evidence.

## CEO-Style Stress Review

Novelty: conditional. The current medium rescue is not novel enough alone; the medium-vs-true-long separation plus stronger length modeling could be novel.

Importance: high if framed as unknown-length DLLM infilling, because fixed-canvas and oracle-length assumptions are central practical limitations.

Reviewer appeal: medium today, potentially high if the paper includes protocol-matched DreamOn/LR-DLLM comparisons, strong ablations, and a clear failure taxonomy.

Strongest contribution: the project has unusually concrete same-hardware evidence that medium-length rescue can be short-safe while true-long rescue fails for a different reason.

Weakest assumption: that a stronger inference-time signal exists without training-time length regularization. If this fails, the paper must pivot to a negative result plus length-regularized method.

CCF-A realism: not sufficient today, but plausible if the next phase produces a principled long-length method or strong cross-model validation.

## Success Criteria

- Improve same-hardware A6000 pass@1 beyond `76.96%` without `<=8` or `9-12` regression.
- Improve `17-24` and/or `25+` buckets, or prove with strong diagnostics why inference-only rescue cannot do so.
- Provide ablations for medium rescue, long signal, stopping policy, and length-selection features.
- Reproduce on at least one additional model family under matched prompt/canvas/evaluation settings.
- Distinguish apples-to-apples comparisons from suggestive literature comparisons.

## Kill Criteria

- A proposed long detector has short-risk above `5%` in offline diagnostics without a compelling precision/recall tradeoff.
- A full GPU candidate improves aggregate pass rate only through short-bucket regressions.
- Cross-model transfer fails and no interpretable failure pattern emerges.
- Literature protocol matching shows that the local setting is too narrow to support the claimed contribution.

## Scope Boundary

The next milestone should not claim SOTA. It should produce a reproducible paper-planning package: evidence snapshot, current method claim, next experiment plan, and a GPU-safe queue strategy.
