# CCF-A Roadmap And Next Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current DLLM code-infilling project into a reproducible research program with a credible CCF-A submission path.

**Architecture:** Separate the work into evidence, method, and validation tracks. Evidence is tracked in compact reports and registries; method work happens in new `clean_scripts/` files; validation uses same-hardware A6000 controls first, then cross-model reruns.

**Tech Stack:** Python runners in `clean_scripts/`, reusable modules in `expvision_dllm_clean/`, local raw outputs in `outputs_clean/`, compact reports in `analysis_outputs/` and `docs/results/`, Git worktrees for branch isolation.

---

## Superpowers Phase Log

- `using-git-worktrees`: active work is isolated on `exp/a6000-midcons-longrescue`; result archival is isolated on `docs/result-archive`.
- `receiving-code-review`: used for the P2 review fixes around scoreboard labeling, bucket registry keys, and GPU launcher behavior.
- `test-driven-development`: used for the resumable A6000 policy runner before implementation.
- `verification-before-completion`: used before claiming experiment completion, before committing, and before reporting pass rates.
- `writing-plans`: used here to turn the research direction into executable next steps.
- `brainstorming`: used for next-experiment design; the standing user instruction is to continue unless there is a destructive action or a genuine decision blocker.

## Requirement Tracker

- Read the repository and explain current project: done in prior analysis; current work is DLLM code infilling with dynamic mask length selection and LCAS stopping.
- Preserve prior outputs: raw `outputs_clean/` remains local; compact records are committed instead of committing heavy raw outputs.
- Use Git worktrees: current experiment work is in `git_workspace/.worktrees/a6000-midcons-longrescue`.
- Keep results reproducible: every result report names command, output directory, baseline, and pairwise path.
- Use A6000 GPUs: completed A6000 control and four A6000 candidates.
- Do not kill other users' processes: complied; tmux runs used available memory alongside existing processes.
- Use new code files when possible: new resumable runner and launcher were added instead of rewriting existing runners.
- Compare to literature before SOTA claims: current notes identify DreamOn and LR-DLLM anchors; no SOTA claim is made.
- Cross-model validation: not complete yet; planned after the A6000 method checkpoint is stable.
- Historical result整理: partial; current A6000 report is complete, result registry still needs refresh on the `docs/result-archive` branch.

## Current Project

The project studies code middle infilling for diffusion language models, currently centered on `GSAI-ML/LLaDA-8B-Base` and `HumanEval-SingleLineInfilling`. The core problem is that DLLMs need a mask/canvas length before denoising, and wrong length selection is a dominant failure mode.

Existing method stack:

- Fixed length and oracle length establish lower/upper bounds.
- CAL-lite probes candidate mask lengths using confidence.
- LCAS stops denoising adaptively by length bucket.
- LCAL adds long-aware correction from confidence curves.
- official-CAL bounded repair handles short under-selection safely.
- `midcons` adds a conservative medium-length rescue.

## Existing Results

The latest A6000 controlled result is:

- A6000 control: `787/1033 = 76.19%`.
- `midcons`: `795/1033 = 76.96%`, `+8` wins, `0` losses.
- `mid_precision`: `787/1033 = 76.19%`, no pairwise change.
- `true_long`: `787/1033 = 76.19%`, no pairwise change.
- `combined`: `787/1033 = 76.19%`, no pairwise change.

Why this matters:

- `midcons` is real same-hardware improvement, not environment noise.
- The improvement is mostly `9-12` and `13-16`.
- Long buckets remain poor: `17-24 = 20.73%`, `25+ = 16.13%`.
- Current official-CAL true-long triggering is not useful; it fired zero times after safety guards.

## Distance To CCF-A

The current project is not yet CCF-A complete. It has a strong empirical handle on length selection, but the publishable claim is still too heuristic unless strengthened.

Missing pieces:

- A principled method that is more than a hand-tuned rescue rule.
- Strong literature-positioned baselines, especially DreamOn-style variable length generation and LR-DLLM-style length regularization.
- Cross-model validation on Dream-Coder, DiffuCoder, Dream, and LLaDA variants.
- Multi-line or broader benchmark evidence, unless the paper is intentionally scoped as single-line length control.
- Ablations showing which signals matter: official-CAL, long curve, support, raw ratio, stop policy, and repair bounds.
- Error analysis that explains why long-tail under-generation remains hard.

## Core Claim Candidate

Recommended claim:

> Inference-time length control for DLLM code infilling can be made short-safe by separating medium-length rescue from true-long detection; confidence-curve agreement improves medium infilling without sacrificing short tasks, while true-long cases require a different underestimation detector because official-CAL is unreliable on long failures.

This is currently an honest claim. A stronger CCF-A claim needs the next long detector or cross-model transfer to work.

## Baselines

Internal baselines:

- fixed64.
- oracle length.
- CAL-lite.
- LCAS-v3b.
- LCAL.
- official-CAL bounded repair.
- A6000 control.
- `midcons` A6000 best.

External baselines to position against:

- DreamOn on Dream-7B, DiffuCoder-7B, DreamCoder-7B.
- LR-DLLM for fully unknown length settings.
- Autoregressive code infilling models such as DeepSeek-Coder, Seed-Coder, and Qwen2.5-Coder when evaluation settings match.

## Metrics

Primary:

- pass@1 on `HumanEval-SingleLineInfilling`.
- pairwise wins/losses vs same-hardware control.
- oracle-length bucket pass rates: `<=8`, `9-12`, `13-16`, `17-24`, `25+`.

Secondary:

- selected length error.
- under/over-selection rate.
- trigger count and trigger precision by source.
- decode time and probe overhead.
- short-bucket loss rate.

## Paper Tables

- Table 1: Main pass@1 against internal baselines and literature anchors.
- Table 2: Same-hardware A6000 controlled pairwise results.
- Table 3: Oracle-length bucket breakdown.
- Table 4: Ablation of rescue signals.
- Table 5: Cross-model generalization.
- Table 6: Failure taxonomy for long under-generation.

## Next Experiments

### Task 1: Finish Current Result Archival

**Files:**
- Modify: `analysis_outputs/a6000_midcons_longrescue/*`
- Modify: `docs/results/a6000_midcons_longrescue_report.md`
- Modify on result-archive branch: `docs/results/run_registry.md`

- [ ] Verify all four candidate runs have `1033` rows and `summary.json`.
- [ ] Generate pairwise summaries against A6000 control.
- [ ] Generate `analysis_outputs/a6000_midcons_longrescue/a6000_scoreboard_section.md`.
- [ ] Commit and push the compact report artifacts.
- [ ] Refresh the result archive registry on `docs/result-archive`.

### Task 2: Make `midcons` The Current A6000 Checkpoint

**Files:**
- Modify: `analysis_outputs/experiment_scoreboard.md`
- Modify: `docs/results/a6000_midcons_longrescue_report.md`

- [ ] Record `midcons` as the current A6000 best checkpoint.
- [ ] Record why `mid_precision`, `true_long`, and `combined` are negative evidence.
- [ ] Keep raw outputs local and only commit compact summaries.

### Task 3: Design A Separate Long-Underestimation Detector

**Files:**
- Create: `docs/superpowers/specs/2026-05-29-long-underestimate-detector-design.md`
- Create later: `clean_scripts/run_lcal_long_underestimate_detector.py`
- Test later: `tests/test_long_underestimate_detector.py`

Design direction:

- Do not use official-CAL high length as the primary long trigger.
- Trigger from long-tail under-selection signatures: selected length far below high-confidence long probe, repeated low selected lengths on long-like curves, and failure-prone stopping behavior.
- Run first in record-only/diagnostic mode to estimate precision before allowing it to change final length.
- Protect short buckets by requiring evidence that does not appear in already-passing short samples.

Why:

- In the completed true-long run, all strict gates passed for zero samples.
- If support is removed, the remaining candidates are mostly already-passing short/medium samples.
- Long failures are overwhelmingly under-selected by `base`, not missed by the current official trigger.

### Task 4: Cross-Model Validation

**Files:**
- Modify or create: `clean_scripts/run_lcas_v3_model_sweep.py`
- Modify: `docs/results/model_generalization_registry.md`

- [ ] Re-run the stable protocol on cached Dream-Coder Base/Instruct.
- [ ] Re-run on LLaDA Base/Instruct where cache permits.
- [ ] Add DiffuCoder-7B or Dream-7B only if model cache/network access is available.
- [ ] Compare only under matched prompt/canvas/evaluation settings.

## Failure Risks

- The current `midcons` improvement may be too small for a standalone CCF-A claim.
- Long-tail improvement may require training-time length control, closer to DreamOn, rather than inference-only rescue.
- Cross-model gains may not transfer if model confidence curves differ.
- Literature numbers may not be apples-to-apples because prompt format, model family, and unknown-length assumptions differ.

## Immediate Recommendation

Ship the A6000 `midcons` result as the current best checkpoint and start the long-underestimation detector as a diagnostic-first experiment. Do not spend more GPU on the current official-CAL true-long trigger unless the analysis identifies a new discriminative signal.
