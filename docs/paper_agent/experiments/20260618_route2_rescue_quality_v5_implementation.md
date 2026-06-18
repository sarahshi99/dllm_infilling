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
- added CPU-only selector `anchor_len32_confidence` after smoke showed that `consensus_confidence` can lose a known Route2 win by selecting `len24_s64`.

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

V5.1 selector: `anchor_len32_confidence`.

It is still pure inference-time and verifier-free. Its purpose is to protect the already successful Route2 precision `len32` action:

```text
anchor candidate = len32_s64
diagnostic candidate = len24_s64
optional switch candidate = len32_s96

select len32_s64 by default
switch to len32_s96 only if inference-visible score exceeds anchor by margin
do not let len24_s64 override the anchor
```

This design came directly from smoke evidence:

- `len32_s64` is the old Route2 precision `len32` behavior.
- `len24_s64` caused a loss on `SingleLineInfilling/HumanEval/60/L0`.
- `len32_s96` may preserve the length-safe behavior while testing whether extra denoising steps help.

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
Ran 12 tests in 0.002s
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

### Smoke 3: Targeted Historical Route2 Wins

The first attempt failed during model loading with CUDA OOM because another process occupied the physical GPU during loading. The retry succeeded.

Targeted task ids:

- `SingleLineInfilling/HumanEval/66/L1`
- `SingleLineInfilling/HumanEval/116/L0`
- `SingleLineInfilling/HumanEval/60/L0`

Output:

- failed OOM attempt: `logs/paper_agent/20260618_route2_v5_wins_gpu3.log`
- successful retry: `outputs_clean/smoke_route2_rescue_quality_v5_wins_gpu3_retry_20260618_130028`

Result:

- `3` rows;
- `route2_trigger_count = 3`;
- candidate-count histogram `{3: 3}`;
- pass rate `2/3`;
- pairwise vs `midcons`: `2/0/0/1`;
- pairwise vs Route2 precision `len32`: `0/1/2/0`;
- oracle upper bound on triggered rows: `3/3`.

Candidate detail:

| task_id | oracle | selected | final pass | passing candidates | selector diagnosis |
|---|---:|---|---|---|---|
| `SingleLineInfilling/HumanEval/66/L1` | `21` | `len32_s64` | true | `len32_s64,len32_s96` | good |
| `SingleLineInfilling/HumanEval/116/L0` | `21` | `len32_s96` | true | `len32_s64,len32_s96` | good |
| `SingleLineInfilling/HumanEval/60/L0` | `10` | `len24_s64` | false | `len32_s64,len32_s96` | bad shorter-candidate switch |

Interpretation:

This is the strongest smoke-level evidence so far. V5 can preserve true-long Route2 wins, but the current `consensus_confidence` selector is not safe enough relative to the existing Route2 precision `len32` policy because it can switch to `len24_s64` and lose a known Route2 win.

Therefore, a full run with the current selector is not justified.

Next selector design should be conservative:

- use `len32_s64` as an anchor because it is the existing clean Route2 precision action;
- select `len32_s96` only if it exceeds the anchor under inference-visible score;
- either remove `len24_s64` from policy selection or allow it only when it exceeds the anchor by a large margin and the selected length is not below a safety floor;
- keep `len24_s64` in logs as a diagnostic candidate if needed.

Implemented CPU-only follow-up:

- selector: `anchor_len32_confidence`;
- default: `len32_s64`;
- switch candidate: `len32_s96`, only with score margin;
- diagnostic-only: `len24_s64`;
- full GPU runs completed for switch margins `0.02` and `0.10`.

## Full V5.1 GPU Results

Full anchor-selector runs completed on GPU `2` and GPU `3` with exit code `0`.

Outputs:

- margin `0.02`: `outputs_clean/full_route2_rescue_quality_v5_anchor_m002_gpu2_20260618_175641`
- margin `0.10`: `outputs_clean/full_route2_rescue_quality_v5_anchor_m010_gpu3_20260618_175642`

| Run | Pass | Rate | Avg sec incl. probe | Triggered | Selected candidates | Pairwise vs `midcons` | Pairwise vs Route2 precision `len32` |
|---|---:|---:|---:|---:|---|---:|---:|
| V5.1 anchor margin `0.02` | `801/1033` | `77.54%` | `4.9208` | `57` | `primary=976`, `len32_s64=56`, `len32_s96=1` | `6/0/795/232` | `0/0/801/232` |
| V5.1 anchor margin `0.10` | `801/1033` | `77.54%` | `4.9618` | `57` | `primary=976`, `len32_s64=57` | `6/0/795/232` | `0/0/801/232` |

The full results exactly match Route2 precision `len32`. They preserve the known `len32` wins but do not improve on them.

Candidate oracle upper bound on triggered rows:

- `9/57 = 15.79%` candidate upper-bound pass rate;
- selected policy pass on triggered rows: `6/57`;
- `25+` oracle bucket remains `0/11` on triggered rows.

The only selection difference between the two margins was `SingleLineInfilling/HumanEval/122/L0`: margin `0.02` switched from `len32_s64` to `len32_s96`, but both candidates failed and no candidate passed.

There are three rows where a non-selected `len24_s64` candidate passed while the anchor failed:

- `SingleLineInfilling/HumanEval/7/L0`, oracle `14`;
- `SingleLineInfilling/HumanEval/11/L6`, oracle `22`;
- `SingleLineInfilling/HumanEval/128/L2`, oracle `15`.

This means the remaining selector upside is real but unsafe to exploit naively. A global or weakly guarded `len24_s64` override conflicts with the earlier smoke finding that `len24_s64` can lose a known Route2 precision `len32` win.

## GPU Gate / Current Decision

The first `/tmp` cache attempt was stopped because it re-downloaded model shards too slowly. Successful smoke runs used:

- `HF_ENDPOINT=https://hf-mirror.com`
- `HF_HOME=/home/shx/.cache/huggingface`

The revised anchor selector was run full. It is safe relative to Route2 precision `len32`, but it is not a new pass-rate improvement.

Do not launch another GPU full run from V5.1 alone. The next GPU run should require a new CPU-first action brief with one of these clearly justified changes:

- a conservative, reviewer-readable `len24_s64` override that protects known `len32` wins;
- a stronger rescue-generation candidate family that improves the candidate upper bound;
- an explicit switch to a learned or compiler-assisted selector claim, with separate protocol and baselines.

## Next Step

Treat V5.1 as a negative/diagnostic full result, not as a new performance claim. It confirms that anchor protection can safely reproduce Route2 precision `len32`, and it narrows the next problem: the candidate set itself only has `9/57` triggered-row oracle upper-bound passes. The next research step should target rescue quality or a tightly constrained shorter-candidate override, not another blind full run.
