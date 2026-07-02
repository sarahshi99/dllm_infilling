# Next-Step Brainstorm After V8

Date: 2026-07-01 CST

## Skill / Workflow Note

Requested skill: `superpowers:brainstorming`.

The current environment does not expose a readable `superpowers:brainstorming` `SKILL.md`. Following the project protocol, this document uses the Superpowers local fallback: make the creative phase explicit, compare multiple routes, reject weak routes, and end with a small set of next actions suitable for `superpowers:writing-plans`.

No GPU experiment is launched in this brainstorming step.

## Where We Are

The project has a stable but modest LLaDA-Base local improvement:

- `midcons`: `795/1033 = 76.96%`
- Route2 precision `len32`: `801/1033 = 77.54%`
- V6 short override: `802/1033 = 77.64%`

The recent proportional-length line is negative:

- V7 post-hoc proportional widening: `792/1033 = 76.67%`
- V8a formula-level proportional score: `786/1033 = 76.09%`
- V8b formula-level proportional score: `781/1033 = 75.61%`
- V8c formula-level proportional score: `782/1033 = 75.70%`

The key V8 diagnosis is:

- V8 changes too many non-true-long rows.
- V8b changes `88` rows, but only `12` are true-long and `76` are non-true-long.
- V8b gets `7` wins but `21` losses.
- Global long-length reward has poor selectivity.

The older Route2/V3/V4 diagnosis still matters:

- Route2 has real signal: all `6` Route2 precision `len32` wins come from triggered rescue.
- Route2 recall is weak: `56` failed-long rows are missed.
- Rescue quality is weak: `33` triggered failed-long rows remain failures, and `31/33` already have rescue length at least oracle.
- V4 found no low-risk non-leaking rule from existing logs that justifies another direct GPU full run.

## Constraints

Preferred paper framing:

- inference-time / training-free if possible;
- no verifier at inference time;
- labels/oracle/pass are allowed only for offline accounting;
- full runs matter, but GPU full should follow a fresh action brief and clear kill criteria;
- do not keep blindly increasing length.

Practical constraints:

- current best LLaDA-Base follow-up is V6, but gain is tiny;
- true-long `25+` remains unsolved;
- selector-only work has low ceiling;
- global length prior is unsafe.

## Brainstormed Routes

### Route A: Local Guarded Proportional Length Reward

Idea:

Keep the user's ratio intuition, but make it local instead of global. Only apply proportional length reward when an under-selection detector says the current row is likely too short.

Possible detector families:

- selected length short relative to probe long-mode bump;
- trace/probe disagreement;
- stop reason plus late plateau;
- low raw-score margin between selected length and longer candidates;
- broad-vs-precision trigger disagreement;
- current V8 changed rows that are true wins versus false losses.

Why it might work:

- V8 proved proportional reward can rescue a few long rows.
- The failure is selectivity, not zero signal.
- A gate may keep the long wins while avoiding short/medium losses.

Why it might fail:

- Prior V4 already struggled to find a safe gate from existing logs.
- Existing features may still not separate true-long enough.

Minimum next action:

CPU-only audit on V8 changed rows. Learn what distinguishes V8 wins from V8 losses and true-long changed rows from short/medium changed rows. Do not run GPU until this audit finds a rule with very low short risk.

Verdict:

Worth a small CPU audit. Not worth immediate GPU.

### Route B: Rescue Quality Instead Of Length Selection

Idea:

Focus on the triggered rows where Route2 already knows something is wrong. Since `31/33` triggered failed-long rows already have rescue length >= oracle, the problem is often generation/selection quality, not canvas length.

Candidate mechanisms:

- alternate denoising schedules for rescue rows;
- multiple deterministic seeds/schedules with verifier-free selection;
- confidence/trace-based MBR or self-consistency selection;
- preserve-prefix or suffix-aware infill constraints;
- compare `len24`, `len32`, and possibly schedule variants, not just length.

Why it might work:

- It attacks the actual V3 bottleneck.
- V6 already showed candidate selection can add one task with zero loss.
- It could create a stronger method story than another length heuristic.

Why it might fail:

- Multi-candidate generation costs more.
- Without verifier, candidate selection may remain weak.
- If the model cannot generate correct code even with enough length, gains may be limited.

Minimum next action:

CPU audit of existing candidate tables and traces to classify triggered failures:

- length-insufficient;
- length-sufficient but low confidence;
- length-sufficient but wrong structure;
- candidate-selection miss;
- all candidates fail.

Then design exactly one new rescue action, not a broad sweep.

Verdict:

Most scientifically aligned with current evidence. Recommended primary route.

### Route C: Missed-Long Gate Recall

Idea:

Find the `56` failed-long rows missed by Route2 and build a new high-precision trigger.

Possible signal families:

- probe long-score residuals;
- confidence residual conditioned on selected length;
- high-confidence-but-short selected rows;
- V8/V7 disagreement: rows that only global length reward changes and are true wins;
- selected length shorter than a robust long-mode probe estimate.

Why it might work:

- There are many missed failed-long rows, so recall is a real bottleneck.
- Even a small high-precision gate could add value if it avoids short losses.

Why it might fail:

- This has been searched several times already, and safe gates were hard to find.
- Triggering missed rows still needs a good rescue action.

Minimum next action:

Do not search gate alone. Pair gate search with a known action, such as Route2 precision `len32` or a future rescue-quality action. The target should be action utility, not true-long classification.

Verdict:

Secondary route. Useful only if tied to action outcome.

### Route D: Lightweight Learned Length Controller

Idea:

Admit that purely hand-written rules may be too weak. Train a small length/action predictor from inference-visible fields, with proper task-id splits.

Why it might work:

- Nonlinear combinations may be necessary.
- The user already noted formula/model choice may hide useful features.

Why it might fail:

- Changes the claim from training-free to learned controller.
- Requires careful train/validation/test protocol and more baselines.
- May be harder to position against CAL/LR-DLLM/DreamOn.

Minimum next action:

Only as an offline upper-bound diagnostic for now. If it strongly beats rules and is stable, write a separate design spec.

Verdict:

Keep as fallback or future paper direction, not immediate mainline.

### Route E: Backbone Transfer Instead Of More LLaDA-Base Search

Idea:

The strongest local transfer result is LLaDA-MoE `+24` tasks. Maybe the paper should emphasize cross-backbone evidence and frame LLaDA-Base true-long rescue as a limitation.

Why it might work:

- Gives a more robust empirical section.
- Avoids overfitting LLaDA-Base long-tail failures.

Why it might fail:

- Does not solve the user's current long-length concern.
- Some backbones are near-tie or negative.

Minimum next action:

Do not make this the next experiment. Keep it as paper framing support.

Verdict:

Good for paper framing; not the next research mechanism.

## Rejected Routes

### Re-run V8 With More Betas

Rejected because V8 shows monotonic tradeoff: stronger beta increases long wins slightly but increases losses more. More beta sweeps are likely low-value unless a local guard is added.

### Blindly Increase Rescue Length

Rejected because Route2 V3 found most triggered failed-long rows already have rescue length >= oracle.

### Continue Selector-Only Tweaks

Rejected as mainline because V6 improved only one task and candidate upper-bound remains `9/57`.

### Direct Multi-Canvas Reranking Everywhere

Rejected as default because it is expensive, complex to defend, and still needs a quality score. It can re-enter only as a targeted rescue-quality action.

## Recommended Next Step

Primary recommendation:

> Route B first: a rescue-quality anatomy and one new targeted rescue action.

Reason:

- It matches the strongest diagnosis: triggered rows often already have enough length but still fail.
- It is more likely to produce a method contribution than another gate threshold.
- It can stay inference-time and training-free if the selector uses trace/probe/candidate confidence only.

Parallel CPU-side companion:

> Route A-lite: analyze V8 changed rows to see whether a local proportional guard can keep wins and remove losses.

Reason:

- It directly addresses the user's ratio idea.
- It should be cheap and diagnostic.
- It should not consume GPU unless it finds a rule.

## Concrete Next Action Candidates

### Candidate 1: Rescue Failure Anatomy Audit

Inputs:

- `midcons` baseline;
- Route2 precision `len32`;
- V5/V6 candidate tables where available;
- V8 changed-row diagnostics.

Output:

- taxonomy of triggered failures;
- rows where length is enough but generation fails;
- rows where shorter candidate helps;
- rows where all current candidates fail;
- recommended one new rescue action.

Decision gate:

- if a credible new action family exists, write a `writing-plans` spec;
- otherwise stop rescue-quality direction and shift to learned controller/paper framing.

### Candidate 2: Local Proportional Guard Audit

Inputs:

- V8a/b/c results;
- midcons baseline;
- probe/trace features.

Output:

- compare V8 wins vs V8 losses;
- identify whether V8's long wins share an inference-visible signature;
- reject if no rule captures at least `2` wins with at most `1` loss in held-out accounting.

Decision gate:

- only if a low-risk rule exists should it become a GPU action.

### Candidate 3: Learned Controller Upper Bound

Inputs:

- row-action table from all completed policies.

Output:

- task-id split learned predictor upper bound;
- feature importance;
- distillability report.

Decision gate:

- if learned model is much better than rules, open a separate learned-controller framing.

## Recommended Order

1. Write `superpowers:writing-plans` plan for Candidate 1 plus Candidate 2 as a cheap companion audit.
2. Implement CPU-only audits.
3. Do not launch GPU from brainstorming.
4. If Candidate 1 identifies a new rescue action, run small targeted smoke.
5. Only after smoke, run one full GPU experiment.

## Current Best Answer To "What Do We Do Now?"

Stop treating length as the only control variable. The next serious attempt should ask:

> Given a row has been identified as risky, how do we generate and select a better infill candidate without a verifier?

And, separately:

> Can V8's few long wins be isolated by a local under-selection guard so the global short/medium losses disappear?

This keeps both intuitions alive: rescue-quality is the main route, and the ratio idea survives only as a local guarded mechanism.
