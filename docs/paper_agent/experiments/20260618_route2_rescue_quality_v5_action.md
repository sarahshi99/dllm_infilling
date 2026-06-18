# Route2 Rescue Quality V5 Action Brief

Date: 2026-06-18 CST

## Action Name

Design and implement Route2 rescue-quality V5: multi-candidate rescue generation plus inference-visible candidate selection.

## Why This Starts A New Mechanism

Discovery V4 ended as `route2_polish_only`. That does not justify another full run of the same broad/precision Route2 policy.

The new bottleneck is different:

- Route2 precision `len32` already gives a clean small gain: `801/1033 = 77.54%`, pairwise `6/0/795/232`.
- Route2 error analysis found `33` triggered failed-long rows.
- `31/33` triggered failed-long rows already have rescue length >= oracle.
- Therefore, another blind length increase is not the default answer.

V5 changes the action after a precision trigger:

```text
old Route2:
  trigger -> one fixed rescue -> always choose rescue

new V5:
  trigger -> several rescue candidates -> choose with inference-visible selector
```

## Current Inputs

- baseline `midcons`: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl`
- baseline trace: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/step_traces.jsonl`
- Route2 precision len32: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl`
- V3 report: `analysis_outputs/route2_error_analysis_20260617_165806/report.md`
- V4 report: `analysis_outputs/discovery_v4_signal_audit_20260618_000000/report.md`

## Planned Files

- spec: `docs/superpowers/specs/2026-06-18-route2-rescue-quality-v5-design.md`
- plan: `docs/superpowers/plans/2026-06-18-route2-rescue-quality-v5-plan.md`
- runner: `clean_scripts/run_route2_rescue_quality_v5.py`
- tests: `tests/test_route2_rescue_quality_v5.py`

## Plain Explanation

V5 is not a new length predictor. It is a new action after the existing Route2 trigger.

Before V5, the best Route2 policy did this:

```text
primary midcons output
if Route2 precision trigger fires:
  run one fixed rescue with length at least 32
  always use that rescue
else:
  keep primary output
```

That policy already improved `LLaDA-8B-Base` from `795/1033 = 76.96%` to `801/1033 = 77.54%`.

V5 asks a narrower question:

```text
When Route2 already says "this row is risky enough to rescue",
can we generate several rescue candidates and choose a better one
without using hidden tests or oracle labels?
```

So V5 is based on both positive and negative evidence:

- Positive evidence: Route2 precision `len32` was clean and useful, with `6` wins and `0` losses against `midcons`.
- Negative evidence: V3/V4 showed that simply finding more long-risk rows or blindly increasing length did not solve true-long failures.
- Diagnostic evidence: many triggered failed-long rows already had rescue length at least as large as the oracle length, so failure was often rescue generation/selection quality, not just length underestimation.

The candidate name `len32_s64` means:

- `len32`: force the rescue canvas to have at least `32` mask positions, i.e. `max(primary_len, 32)`;
- `s64`: decode using `64` denoising/generation steps.

Likewise:

- `len24_s64` means at least length `24`, `64` steps;
- `len32_s96` means at least length `32`, `96` steps.

The initial V5 selector `consensus_confidence` tried to choose among these candidates using only inference-visible signals. Smoke showed that this selector can wrongly choose the shorter `len24_s64` candidate and lose a known Route2 `len32` win. Therefore, the current next step is a safer V5.1 selector: `anchor_len32_confidence`.

V5.1 uses `len32_s64` as the anchor because it reproduces the already successful Route2 precision `len32` action. It keeps `len24_s64` as a diagnostic candidate but does not allow it to override the anchor by default.

## Candidate Set

Initial cheap candidate set:

| Candidate | Length | Steps | Purpose |
|---|---:|---:|---|
| `len24_s64` | `max(primary_len, 24)` | `64` | compare to existing shorter rescue |
| `len32_s64` | `max(primary_len, 32)` | `64` | reproduce current clean Route2 action |
| `len32_s96` | `max(primary_len, 32)` | `96` | test slower generation schedule |

Optional full candidate set if smoke is healthy:

- add `len40_s64`;
- add `len40_s96`.

## Selectors

### Selector A: `consensus_confidence`

Default policy candidate. It must not use hidden unit tests, oracle length, pass/fail outcome, or pairwise labels.

Inputs:

- candidate middle text;
- pairwise text consensus;
- final trace confidence/top1/gap;
- remaining mask ratio;
- selected length.

### Selector B: `syntax_aware`

Secondary ablation. It may use parse/compile diagnostics but no hidden unit tests.

Label it as compiler-assisted inference-time, not the default pure verifier-free selector.

### Selector C: `oracle_upper_bound`

Offline diagnostic only. It asks whether any generated candidate passes hidden tests.

Never use it as deployed selection.

## GPU Policy

No GPU command should start from this action brief until:

- `tests/test_route2_rescue_quality_v5.py` passes;
- `clean_scripts/run_route2_rescue_quality_v5.py` compiles;
- `git diff --check` passes;
- `nvidia-smi` confirms GPU `2/3` are free of other users' jobs, unless the user explicitly overrides;
- a tiny smoke command is written with output directory and kill criteria.

## Smoke Success Criteria

Smoke is pipeline validation only, not a performance claim.

It must show:

- command exit `0`;
- requested row count in `results.jsonl`;
- candidate metadata exists for every triggered row;
- `summary.json` reports Selector A, Selector B, and oracle upper bound;
- pairwise accounting against baseline can be computed.

## Full Run Success Criteria

Selector A is interesting only if full run satisfies:

- at least `803/1033` pass;
- losses vs `midcons` at most `2`;
- losses vs Route2 precision `len32` at most `3`;
- oracle `<=8` net loss at most `1`;
- oracle `17-24` plus `25+` net gain positive;
- runtime overhead reported.

Stronger paper signal:

- at least `808/1033` pass;
- oracle `25+` improves by at least `1`;
- candidate-set upper bound or Selector A gives a clear mechanism story.

## Kill Criteria

Stop if:

- gains only appear under oracle upper-bound selection;
- Selector A regresses below Route2 precision `len32`;
- short bucket loses more than `1`;
- candidate generation has no upper-bound gain;
- runtime is too high for the gain;
- GPU `2/3` are occupied and no explicit override is given.

## Current Status

Implementation status at 2026-06-18 12:37 CST:

- V5 runner implemented: `clean_scripts/run_route2_rescue_quality_v5.py`.
- Focused tests implemented: `tests/test_route2_rescue_quality_v5.py`.
- Selector A now scores only a whitelisted policy view. Hidden pass/fail, oracle length, verification outcomes, and pairwise labels are not passed into the deployed selector.
- Oracle upper bound is logged only as `offline_only`.
- CPU verification passed:
  - `/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_rescue_quality_v5.py tests/test_route2_trace_rescue.py`
  - `/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_route2_rescue_quality_v5.py`
  - `/home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --help`
  - `git diff --check`
- Added targeted smoke support: `--task-ids-csv`.
- GPU smoke launched only after GPU `2/3` were observed free. The first `/tmp` cache attempt was stopped because it began re-downloading the full model too slowly; the successful runs used the existing local HF cache plus `HF_ENDPOINT=https://hf-mirror.com`.

Smoke 1, primary fallback schema:

- command: `max-samples 10`, GPU `3`, existing HF cache.
- output: `outputs_clean/smoke_route2_rescue_quality_v5_cheap_consensus_gpu3_cache_20260618_124827`
- result: `10/10` pass, `route2_trigger_count = 0`.
- interpretation: basic runner, logging, summaries, and pairwise references work, but this did not exercise the candidate branch.

Smoke 2, targeted triggered branch:

- command: targeted task ids `SingleLineInfilling/HumanEval/4/L1`, `SingleLineInfilling/HumanEval/6/L12`, `SingleLineInfilling/HumanEval/16/L0`, GPU `3`.
- output: `outputs_clean/smoke_route2_rescue_quality_v5_targeted_gpu3_20260618_125123`
- result: `1/3` pass, `route2_trigger_count = 3`, candidate-count histogram `{3: 3}`.
- pairwise vs `midcons`: `1/0/0/2`.
- pairwise vs Route2 precision `len32`: `0/0/1/2`.
- oracle upper bound on triggered rows: `1/3`.
- selector chose `len24_s64` on all three targeted rows.
- interpretation: V5 candidate logging/selection works. For the two targeted true-long failures, no candidate passed, so those examples are generation failures rather than selector mistakes. The short/medium win was preserved.

Smoke 3, targeted historical Route2 wins:

- first attempt hit GPU OOM during model load because another process occupied the physical card during loading; this is a resource collision, not a V5 code error.
- retry output: `outputs_clean/smoke_route2_rescue_quality_v5_wins_gpu3_retry_20260618_130028`
- targeted task ids: `SingleLineInfilling/HumanEval/66/L1`, `SingleLineInfilling/HumanEval/116/L0`, `SingleLineInfilling/HumanEval/60/L0`.
- result: `2/3` pass, `route2_trigger_count = 3`, candidate-count histogram `{3: 3}`.
- pairwise vs `midcons`: `2/0/0/1`.
- pairwise vs Route2 precision `len32`: `0/1/2/0`.
- oracle upper bound on triggered rows: `3/3`.
- important failure: on `SingleLineInfilling/HumanEval/60/L0`, `len32_s64` and `len32_s96` passed, but `consensus_confidence` selected `len24_s64`, which failed.

Current decision:

- Do not launch a full V5 run with the current `consensus_confidence` selector.
- The next safe design is an anchor-protected selector that defaults to the existing clean Route2 precision action `len32_s64` and only switches to another candidate when the score margin is large enough. An even simpler fallback is to remove `len24_s64` from the selectable policy candidates while keeping it only as a diagnostic candidate.
- CPU-only V5.1 implementation status: `anchor_len32_confidence` has been added to `clean_scripts/run_route2_rescue_quality_v5.py`.
- CPU verification for V5.1 passed:
  - `/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_rescue_quality_v5.py tests/test_route2_trace_rescue.py`
  - `/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_route2_rescue_quality_v5.py`
  - `/home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --help`
  - `git diff --check`
- No GPU run has been launched for V5.1.

Recommended full/smoke command when GPU `2` or `3` is free:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --candidate-set cheap --selector consensus_confidence --route2-policy precision_top1_conf --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl --route2-reference-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl --experiment-name full_route2_rescue_quality_v5_cheap_consensus_gpu3
```
