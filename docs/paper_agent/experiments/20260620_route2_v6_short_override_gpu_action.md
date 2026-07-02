# Route2 V6 Short-Override GPU Action

Date: 2026-06-20 CST

## Action Name

Run targeted GPU smoke for `anchor_len32_short_trace_override`.

## Why This Action

The CPU audit found narrow selector-only upside after V5.1:

- V5.1 anchor result: `801/1033 = 77.54%`.
- Candidate upper bound on triggered rows: `9/57`.
- Selected triggered pass: `6/57`.
- Three `len24_s64`-only opportunities exist.
- Four anchor-risk rows exist where `len32_s64` passes but `len24_s64` fails.

The new V6 selector is intentionally conservative:

```text
default: len32_s64 anchor
optional same-length switch: len32_s96 if it beats anchor by score margin
short override: len24_s64 only if
  delta_short_minus_anchor_gap_median >= 0.285156
  and delta_short_minus_anchor_top1_median >= 0.283203
```

This targets the strongest trace-only rule from the CPU audit and avoids text-length features.

## CPU Verification

Completed before GPU:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_route2_rescue_quality_v5.py tests/test_route2_v6_short_override_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile clean_scripts/run_route2_rescue_quality_v5.py analysis/route2_v6_short_override_audit.py
/home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --help
git diff --check
```

Result: `Ran 17 tests`, `OK`; compile/help/diff-check passed.

## GPU Smoke

GPU: `2` only for targeted smoke.

HF mirror/cache:

- `HF_ENDPOINT=https://hf-mirror.com`
- `HF_HUB_DISABLE_XET=1`
- `HF_HOME=/home/shx/.cache/huggingface`

Target rows:

- `SingleLineInfilling/HumanEval/7/L0`
- `SingleLineInfilling/HumanEval/11/L6`
- `SingleLineInfilling/HumanEval/128/L2`
- `SingleLineInfilling/HumanEval/34/L0`
- `SingleLineInfilling/HumanEval/60/L0`
- `SingleLineInfilling/HumanEval/66/L1`
- `SingleLineInfilling/HumanEval/116/L0`

Command:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --candidate-set cheap --selector anchor_len32_short_trace_override --anchor-switch-margin 0.10 --short-override-gap-margin 0.285156 --short-override-top1-margin 0.283203 --route2-policy precision_top1_conf --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl --route2-reference-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl --task-ids-csv SingleLineInfilling/HumanEval/7/L0,SingleLineInfilling/HumanEval/11/L6,SingleLineInfilling/HumanEval/128/L2,SingleLineInfilling/HumanEval/34/L0,SingleLineInfilling/HumanEval/60/L0,SingleLineInfilling/HumanEval/66/L1,SingleLineInfilling/HumanEval/116/L0 --experiment-name smoke_route2_v6_short_override_targeted_gpu2
```

Log:

- `logs/paper_agent/20260620_route2_v6_short_override_targeted_gpu2.log`

## Expected Outcome

The selector should:

- choose `len24_s64` on `HumanEval/11/L6`;
- keep anchor or same-length candidate on the four anchor-risk rows;
- avoid broad `len24_s64` switching.

## Success Criteria

Smoke-level success:

- command exits `0`;
- all `7` rows run;
- selected candidate histogram matches a narrow override pattern;
- no anchor-risk row is switched to failing `len24_s64`;
- pairwise vs Route2 precision `len32` is at least non-negative on this targeted set.

Full-run gate:

- only consider full GPU run if smoke confirms zero anchor-risk losses and at least one recovered `len24_s64` opportunity.

## Kill Criteria

Do not launch full run if:

- smoke exits nonzero;
- any anchor-risk row switches to failing `len24_s64`;
- the selected override does not reproduce the CPU-predicted opportunity;
- GPU `2/3` become occupied by other users.

## Documentation Outputs

- update this file with smoke result;
- update V6 CPU action/spec if smoke contradicts offline audit;
- only update `experiment_results.zh.md` after a full run.

## Smoke Result

Completed on GPU `2`.

Output:

- failed first attempt due to a local argument plumbing bug: `outputs_clean/smoke_route2_v6_short_override_targeted_gpu2_20260620_124144`
- successful retry: `outputs_clean/smoke_route2_v6_short_override_targeted_gpu2_retry_20260620_124449`

Verification:

- retry exit code: `0`
- rows: `7`
- selected histogram: `len24_s64=1`, `len32_s64=6`
- pairwise vs Route2 precision `len32`: `1/0/4/2`
- pairwise vs `midcons`: `5/0/0/2`

Selection detail:

| task_id | oracle | selected | reason | pass | passing candidates |
|---|---:|---|---|---|---|
| `SingleLineInfilling/HumanEval/7/L0` | `14` | `len32_s64` | `anchor_default` | false | `len24_s64` |
| `SingleLineInfilling/HumanEval/11/L6` | `22` | `len24_s64` | `short_trace_gap_top1_override` | true | `len24_s64` |
| `SingleLineInfilling/HumanEval/128/L2` | `15` | `len32_s64` | `anchor_default` | false | `len24_s64` |
| `SingleLineInfilling/HumanEval/34/L0` | `8` | `len32_s64` | `anchor_default` | true | `len32_s64,len32_s96` |
| `SingleLineInfilling/HumanEval/60/L0` | `10` | `len32_s64` | `anchor_default` | true | `len32_s64,len32_s96` |
| `SingleLineInfilling/HumanEval/66/L1` | `21` | `len32_s64` | `anchor_default` | true | `len32_s64,len32_s96` |
| `SingleLineInfilling/HumanEval/116/L0` | `21` | `len32_s64` | `anchor_default` | true | `len32_s64,len32_s96` |

Interpretation:

The targeted smoke satisfies the full-run gate. It recovers the strongest CPU-audit `len24_s64` opportunity and preserves all four anchor-risk rows.

## Full GPU Run

Run on GPU `2` only. GPU `3` is left free because the CPU and smoke evidence justify a single parameter setting, not a parallel sweep.

Command:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_route2_rescue_quality_v5.py --candidate-set cheap --selector anchor_len32_short_trace_override --anchor-switch-margin 0.10 --short-override-gap-margin 0.285156 --short-override-top1-margin 0.283203 --route2-policy precision_top1_conf --baseline-results /home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846/results.jsonl --route2-reference-results /home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516/results.jsonl --experiment-name full_route2_v6_short_override_gpu2
```

Log:

- `logs/paper_agent/20260620_route2_v6_short_override_full_gpu2.log`

Success criteria for the full run:

- exit code `0`;
- `1033` rows;
- pass count at least `802/1033`;
- losses vs Route2 precision `len32` equal `0`;
- short bucket net loss equal `0`;
- selector histogram shows a very small number of `len24_s64` overrides.

## Full Result

Completed on GPU `2`.

Output:

- run directory: `outputs_clean/full_route2_v6_short_override_gpu2_20260620_124754`
- log: `logs/paper_agent/20260620_route2_v6_short_override_full_gpu2.log`
- exit code: `0`
- rows: `1033`

Main result:

| Run | Pass | Rate | Pairwise vs `midcons` | Pairwise vs Route2 precision `len32` |
|---|---:|---:|---:|---:|
| current `midcons` | `795/1033` | `76.96%` | baseline | n/a |
| Route2 precision `len32` | `801/1033` | `77.54%` | `6/0/795/232` | baseline |
| V6 short override | `802/1033` | `77.64%` | `7/0/795/231` | `1/0/801/231` |

Selector behavior:

| Item | Value |
|---|---:|
| Route2 triggered rows | `57` |
| Candidate upper-bound on triggered rows | `9/57` |
| Selected candidates | `primary=976`, `len32_s64=56`, `len24_s64=1` |
| `len24_s64` overrides | `1` |
| Losses vs Route2 precision `len32` | `0` |

The only `len24_s64` override was:

| task_id | oracle | selected length | pass | Pairwise role |
|---|---:|---:|---|---|
| `SingleLineInfilling/HumanEval/11/L6` | `22` | `24` | true | the single win vs Route2 precision `len32` |

Oracle bucket pass rates:

| Bucket | Pass |
|---|---:|
| `<=8` | `540/598 = 90.30%` |
| `9-12` | `184/232 = 79.31%` |
| `13-16` | `53/90 = 58.89%` |
| `17-24` | `20/82 = 24.39%` |
| `25+` | `5/31 = 16.13%` |

Interpretation:

V6 satisfies the full-run success criteria and gives a clean but tiny improvement over Route2 precision `len32`: `+1` task, `0` losses. It should be reported as conservative selector polish, not as a new broad long-length solution. The result confirms that there is a real inference-visible short-candidate signal for one true-long row, but it also confirms the selector-only ceiling is very low: candidate upper-bound remains `9/57`, and oracle `25+` is unchanged. The next useful research step should focus on better rescue candidate generation or richer rescue decoding, not another selector-only sweep.
