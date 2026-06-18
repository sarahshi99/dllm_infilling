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
- No GPU experiment has been launched from this action.

GPU smoke is currently blocked because `nvidia-smi` shows GPU `2` and `3` are occupied by user `xy` training jobs:

```text
GPU 2 PID 3410485 /home/xy/anaconda3/envs/cgsa/bin/python3.9 ... elapsed 17:57
GPU 3 PID 3410486 /home/xy/anaconda3/envs/cgsa/bin/python3.9 ... elapsed 17:57
```

Recommended smoke command when GPU `3` is free:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/tmp/hf_route2_v5_20260618 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --max-samples 10 --candidate-set cheap --selector consensus_confidence --route2-policy precision_top1_conf --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl --route2-reference-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl --experiment-name smoke_route2_rescue_quality_v5_cheap_consensus_gpu3
```
