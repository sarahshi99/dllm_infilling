# Route2 V6 Short-Override CPU Action

Date: 2026-06-19 CST

## Action Name

Design and run a CPU-only audit for a conservative `len24_s64` override after V5.1.

## Stage And Workflow

Plan version: Route2 V6 short-override / rescue-quality follow-up.

Superpowers workflow alignment:

- `superpowers:brainstorming`: compare candidate paths after V5.1 negative full result.
- `superpowers:using-git-worktrees`: checked. Current branch `paper-agent-overnight` is already the dedicated paper-agent branch; no extra worktree is needed for a CPU-only analysis script.
- `superpowers:writing-plans`: write the executable spec and plan before implementation.
- `superpowers:executing-plans`: implement serially. Subagents remain disabled by project protocol.
- `superpowers:verification-before-completion`: run focused tests, compile, and diff hygiene before reporting.

The current Codex environment does not expose callable `superpowers:*` skill files, so this action follows the project protocol's documented Superpowers workflow as a local fallback.

## Reviewer Motivation

V5.1 preserved Route2 precision `len32` exactly but did not improve it:

- V5.1 margin `0.02`: `801/1033 = 77.54%`
- V5.1 margin `0.10`: `801/1033 = 77.54%`
- pairwise vs Route2 precision `len32`: `0/0/801/232`

The useful new evidence is that the candidate oracle upper bound on triggered rows is `9/57`, while the anchor-selected policy passes `6/57`. The three remaining upper-bound-only rows are exactly the kind of narrow opportunity a reviewer may accept if the override is conservative and auditable.

## Hypothesis

There may be an inference-visible rule that selects `len24_s64` on a small subset of Route2-triggered rows where it is better than `len32_s64`, without losing known `len32_s64` wins.

If no such rule exists, the result is still useful: it justifies not spending GPU on another selector-only full run and redirects effort toward better rescue candidate generation.

## Baseline, Dataset, Model, Metric

- dataset: HumanEval single-line infilling, `1033` rows.
- model/backbone: `GSAI-ML/LLaDA-8B-Base`.
- baseline: current `midcons`, `795/1033 = 76.96%`.
- strong local reference: Route2 precision `len32`, `801/1033 = 77.54%`.
- V5.1 inputs:
  - `outputs_clean/full_route2_rescue_quality_v5_anchor_m002_gpu2_20260618_175641/results.jsonl`
  - `outputs_clean/full_route2_rescue_quality_v5_anchor_m010_gpu3_20260618_175642/results.jsonl`
- comparison type: CPU-only offline selector audit, not a new pass-rate claim.

## Exact Work

Planned files:

- `docs/superpowers/specs/2026-06-19-route2-v6-short-override-design.md`
- `docs/superpowers/plans/2026-06-19-route2-v6-short-override-plan.md`
- `analysis/route2_v6_short_override_audit.py`
- `tests/test_route2_v6_short_override_audit.py`
- `analysis_outputs/route2_v6_short_override_audit_*/`

Planned command:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/route2_v6_short_override_audit.py \
  --v5-results outputs_clean/full_route2_rescue_quality_v5_anchor_m002_gpu2_20260618_175641/results.jsonl \
  --reference-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl \
  --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl \
  --output-dir analysis_outputs/route2_v6_short_override_audit_20260619
```

## GPU / Environment

No GPU work. This action is CPU-only and must not start GPU experiments.

## Expected Outcome

The audit should produce one of three decisions:

- `policy_candidate`: a readable inference-visible `len24_s64` override captures at least one upper-bound-only row with no anchor losses.
- `diagnostic_only`: some signal exists, but it is too narrow or unstable for GPU.
- `reject_selector_only`: selector-only improvement is not credible; move to rescue-generation candidate design.

## Success Criteria

- Join all `1033` V5.1 rows with baseline/reference rows.
- Reproduce V5.1 pass count `801/1033` and reference pairwise `0/0/801/232`.
- Reproduce triggered count `57`, selected triggered pass `6`, and oracle upper-bound `9`.
- List the upper-bound-only `len24_s64` rows and any anchor-loss rows.
- Search only inference-visible policy fields from candidate records.
- Output a candidate table, Pareto summary, and final decision.

## Kill Criteria

Stop as diagnostic-only if:

- candidate records lack enough inference-visible fields for rule search;
- any rule that captures `len24_s64` wins also loses known `len32_s64` wins;
- the only useful rule depends on oracle/pass/verifier labels as inputs;
- the result cannot beat the existing Route2 precision `len32` upper bound by offline accounting.

## Known Risks

- The sample is tiny: only `57` triggered rows and only `3` upper-bound-only `len24_s64` opportunities.
- Any apparent rule can be overfit.
- CPU offline selector accounting cannot prove a full-run pass-rate improvement unless the exact inference-time selector is later implemented and run.
- A shorter candidate may pass by chance on medium rows while harming true-long rows.

## Documentation Outputs

- update this action brief with audit results;
- write a report under `analysis_outputs/route2_v6_short_override_audit_*`;
- update `docs/paper_agent/experiment_results.zh.md` only if the audit materially changes next-step interpretation.
