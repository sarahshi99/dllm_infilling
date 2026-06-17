# Route2 Error Analysis Discovery V3 Design

Date: 2026-06-17
Branch: `paper-agent-overnight`
Scope: CPU-only diagnostic design for improving the trace/probe Discovery layer after Route2 full runs.

## Objective

Do not abandon true-long signal search just because `trace_feature_audit_v2` ended as `diagnostic_only`. Instead, use the full Route2 policy outcomes as supervised diagnostic evidence to improve the next Discovery layer.

The immediate research question is:

> What distinguishes Route2 wins, triggered true-long failures, and missed failed-long rows, and which new trace/probe/rescue-quality signals should Discovery V3 search for?

This remains training-free and inference-time oriented. Oracle length and verifier results may be used only for offline error taxonomy and candidate evaluation. They must not become policy inputs.

## Current Evidence

The relevant baseline is current LLaDA-Base `midcons`:

- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`
- result: `795/1033 = 76.96%`

The cleanest Route2 follow-up is precision `len32`:

- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`
- result: `801/1033 = 77.54%`
- pairwise vs `midcons`: `6/0/795/232`
- triggers: `57`
- trigger true-long precision: `61.40%`

A quick CPU check after the full run found:

- all `6` wins are triggered rescue cases;
- triggered-but-still-failed true-long rows: `33`;
- `31/33` triggered-but-still-failed true-long rows already have rescue length greater than or equal to oracle length;
- missed baseline failed-long rows: `56`;
- oracle `25+` has `0` net gain.

Interpretation:

> Blindly increasing the rescue length is unlikely to be the main answer. The next signal search must separate gate recall from rescue generation/selection quality.

## Why This Is A Discovery-Layer Continuation

`trace_feature_audit_v2` asked whether existing trace dynamics contain stable low-risk true-long signals. It found partial signal but not a robust policy candidate on current `midcons`.

Route2 then provided policy-level counterfactual evidence:

- Which triggered rows actually become wins.
- Which triggered rows still fail.
- Which failed-long rows were missed entirely.
- Which wins are short/medium rather than true-long.

That evidence should feed a new Discovery layer, not end the search.

## Three-Layer V3 Structure

### Layer 1: Error Anatomy Layer

Build an exact taxonomy from baseline and Route2 outputs:

- `route2_win`: Route2 passes and baseline fails.
- `route2_loss`: Route2 fails and baseline passes.
- `triggered_failed_long`: oracle `>=17`, Route2 triggered, final fails.
- `triggered_rescued_long`: oracle `>=17`, Route2 triggered, baseline fails, Route2 passes.
- `missed_failed_long`: oracle `>=17`, baseline fails, Route2 did not trigger.
- `short_or_medium_win`: oracle `<17`, baseline fails, Route2 passes.
- `unchanged_fail`: both fail.

For each class, summarize:

- oracle bucket;
- primary selected length;
- rescue length;
- whether rescue length is at least oracle length;
- trace features used by Route2;
- available probe/length fields;
- stop reason;
- verifier failure category if available.

Purpose: decide whether the next failure is length estimation, gate recall, decode quality, or selection.

### Layer 2: Discovery Expansion Layer

Search for signals that v2 did not explicitly optimize:

- rescue-success predictors among triggered rows;
- missed-failed-long predictors among untriggered rows;
- high-confidence missed-long patterns, because many missed rows have high trace confidence;
- probe-trace disagreement, especially selected length short while probe long mode is present;
- rescue length sufficiency features: `rescue_len >= oracle` for offline diagnosis only, then distilled into inference-visible proxies such as selected length, trace uncertainty, probe curve shape, and stop reason;
- failure-mode features derived from verifier output where available, only for offline diagnosis.

This layer may use simple learned models as microscopes, but final candidates must be distilled into readable inference-time rules.

### Layer 3: Policy Distillation Layer

Translate diagnostics into one of these candidate paths:

- **Path A: Better rescue decoding/selection.** Use when triggered true-long failures usually have enough length but still fail.
- **Path B: Probe-trace fusion gate.** Use when missed failed-long rows share inference-visible probe/trace patterns.
- **Path C: Adaptive rescue length.** Use only if many triggered failures still have rescue length below plausible required length.
- **Path D: Conservative Route2 polish.** Keep precision `len32` as incremental evidence and stop trying to force true-long claims.
- **Path E: Stop true-long under current signals.** Use if no low-risk inference-visible signal distinguishes errors.

## Required Diagnostic Outputs

The CPU-only diagnostic should write:

- `summary.json`
- `error_taxonomy.csv`
- `triggered_failed_long.csv`
- `missed_failed_long.csv`
- `wins.csv`
- `feature_contrast.csv`
- `report.md`

The report must include:

1. pairwise overview vs `midcons`;
2. bucket-level wins/losses/triggers;
3. triggered-but-still-failed true-long table;
4. missed failed-long table;
5. rescue length sufficiency analysis;
6. feature contrasts between wins, triggered failures, and missed failures;
7. decision among Paths A-E;
8. whether another GPU run is justified.

## Success Criteria

- The diagnostic is CPU-only and does not launch GPU work.
- It joins `1033` Route2 rows with `1033` baseline rows.
- It reproduces the known precision `len32` pairwise counts `6/0/795/232`.
- It reproduces `33` triggered-but-still-failed true-long rows and `56` missed baseline failed-long rows.
- It can answer whether the main bottleneck is gate recall, rescue length, rescue generation quality, or a mixture.
- It produces at least one concrete next-policy recommendation or a clear negative result.

## Kill Criteria

Stop and document negative evidence if:

- row joins or pairwise counts do not reproduce the known summary;
- required fields are missing from the Route2 run;
- the only useful separators are oracle/pass/verifier labels that cannot be replaced by inference-visible proxies;
- no feature contrast separates wins, triggered failures, and missed failures enough to justify another policy design;
- the recommended next action would be another blind full GPU run without a new mechanism.

## Reviewer-Facing Claim Boundary

This diagnostic cannot prove a new pass-rate improvement. It can only justify the next mechanism:

- a better rescue decoder,
- a safer fusion gate,
- an adaptive length policy,
- or a principled stop decision.

It should preserve the current conservative claim:

> Route2 precision `len32` is low-risk incremental evidence, but true-long infilling remains unsolved.

## Self-Review

- The design does not abandon signal search.
- The design explicitly uses Route2 outcomes to improve the Discovery layer.
- The design does not use oracle/pass labels as inference-time policy inputs.
- The design explains why blind length increases are not the default next step.
- The design keeps Route3 non-default unless a stable trace/probe quality score appears.
