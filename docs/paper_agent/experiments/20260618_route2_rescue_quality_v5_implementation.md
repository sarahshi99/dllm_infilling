# Route2 Rescue Quality V5 Implementation Note

Date: 2026-06-18 CST

## Status

V5 design has moved from planning to a CPU-verified runner implementation.

Implemented files:

- `clean_scripts/run_route2_rescue_quality_v5.py`
- `tests/test_route2_rescue_quality_v5.py`

This implementation adds a new triggered action:

```text
primary midcons
if precision Route2 fires:
  generate multiple deterministic rescue candidates
  select one with an inference-visible selector
else:
  keep primary
```

## Selector Boundary

Default selector: `consensus_confidence`.

It uses only a whitelisted policy view:

- candidate middle text;
- text consensus among candidates;
- trace confidence/top1/gap;
- final remaining-mask ratio;
- selected length.

Hidden unit-test pass/fail, oracle length, verification labels, and pairwise outcomes are kept out of the selector view. They are logged only for offline accounting.

Secondary selector: `syntax_aware`.

It can use parse/compile diagnostics and is labeled as `compiler_assisted_inference_time`.

Oracle selector: `oracle_upper_bound`.

It is offline-only and never deployable.

## Verification

Passed:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_rescue_quality_v5.py tests/test_route2_trace_rescue.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_route2_rescue_quality_v5.py
/home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --help
git diff --check
```

Result:

```text
Ran 8 tests in 0.001s
OK
```

## GPU Gate

No GPU smoke was launched.

Reason: GPU `2` and `3` are currently occupied by user `xy` training jobs, so the project safety gate says to wait unless the user explicitly overrides.

When GPU `3` is free, the recommended smoke is:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/tmp/hf_route2_v5_20260618 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --max-samples 10 --candidate-set cheap --selector consensus_confidence --route2-policy precision_top1_conf --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl --route2-reference-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl --experiment-name smoke_route2_rescue_quality_v5_cheap_consensus_gpu3
```

## Next Step

Wait for GPU `2/3` to be free, run the smoke, inspect schema and candidate metadata, then decide whether a full V5 run is justified.
