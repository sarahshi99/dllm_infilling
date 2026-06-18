# Route2 Rescue Quality V5 Implementation Note

Date: 2026-06-18 CST

## Status

V5 design has moved from planning to a CPU-verified runner implementation.

Implemented files:

- `clean_scripts/run_route2_rescue_quality_v5.py`
- `tests/test_route2_rescue_quality_v5.py`

Additional implementation update:

- added `--task-ids-csv` so smoke can target known triggered rows instead of hoping early dataset rows trigger Route2;
- verified task-id filtering with focused unit tests.

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
Ran 10 tests in 0.001s
OK
```

## Smoke Results

### Smoke 1: Primary Fallback Schema

Output:

- `outputs_clean/smoke_route2_rescue_quality_v5_cheap_consensus_gpu3_cache_20260618_124827`

Result:

- `10` rows;
- `10/10` pass;
- `route2_trigger_count = 0`;
- pairwise vs `midcons`: `0/0/10/0`;
- pairwise vs Route2 precision `len32`: `0/0/10/0`.

Interpretation:

The ordinary primary fallback path works, but this smoke does not validate the candidate branch because no task triggered Route2.

### Smoke 2: Targeted Triggered Branch

Targeted task ids:

- `SingleLineInfilling/HumanEval/4/L1`
- `SingleLineInfilling/HumanEval/6/L12`
- `SingleLineInfilling/HumanEval/16/L0`

Output:

- `outputs_clean/smoke_route2_rescue_quality_v5_targeted_gpu3_20260618_125123`

Result:

- `3` rows;
- `route2_trigger_count = 3`;
- candidate-count histogram `{3: 3}`;
- pass rate `1/3`;
- pairwise vs `midcons`: `1/0/0/2`;
- pairwise vs Route2 precision `len32`: `0/0/1/2`;
- oracle upper bound on triggered rows: `1/3`;
- selector chose `len24_s64` on all three rows.

Candidate detail:

| task_id | oracle | selected | final pass | oracle upper bound | passing candidates |
|---|---:|---|---|---|---|
| `SingleLineInfilling/HumanEval/4/L1` | `19` | `len24_s64` | false | false | none |
| `SingleLineInfilling/HumanEval/6/L12` | `21` | `len24_s64` | false | false | none |
| `SingleLineInfilling/HumanEval/16/L0` | `8` | `len24_s64` | true | true | `len24_s64,len32_s64,len32_s96` |

Interpretation:

The V5 candidate branch is functional and logs the needed metadata. On the two targeted true-long failed rows, the candidate set itself has no passing candidate, so these examples are generation failures rather than selector mistakes. The short/medium Route2 win is preserved.

## GPU Gate

The first `/tmp` cache attempt was stopped because it re-downloaded model shards too slowly. Successful smoke runs used:

- `HF_ENDPOINT=https://hf-mirror.com`
- `HF_HOME=/home/shx/.cache/huggingface`

After smoke, GPU `2/3` became occupied by other users' jobs, so no full run was launched.

When GPU `2` or `3` is free and the user approves a full run, the recommended command is:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --candidate-set cheap --selector consensus_confidence --route2-policy precision_top1_conf --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl --route2-reference-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl --experiment-name full_route2_rescue_quality_v5_cheap_consensus_gpu3
```

## Next Step

Do not treat the smoke as a performance claim. It justifies that a full run is technically possible. The research decision is still open because the targeted true-long examples showed generation failure, not selector upside.
