# V8 Proportional CAL Score Full GPU1 Action

Date: 2026-07-01 CST

## Action Name

Direct proportional length reward inside the CAL-lite score formula.

## Motivation

V7 was not a faithful test of the user's idea. It selected a longer near-best candidate after the normal base selection had already happened. V8 instead changes the CAL-lite scoring formula itself, so the length reward is applied during the first length-estimation step.

Baseline CAL-lite scoring:

```text
score(L) = raw_score(L) * L^alpha
```

V8 scoring:

```text
score(L) = raw_score(L) * L^(alpha + beta * log(max(min(L, cap) / ref, 1)))
```

This keeps the existing training-free inference-time framing, but lets longer lengths receive a stronger reward directly in the CAL-like formula.

## Variants

All runs use `GSAI-ML/LLaDA-8B-Base`, HumanEval single-line infilling, GPU1 only, and the same bounded official-repair settings as the current LLaDA-Base line. Runs are sequential, not parallel.

| Variant | beta | ref | cap | Experiment name |
|---|---:|---:|---:|---|
| V8a mild | `0.02` | `12` | none | `full_v8a_propcal_beta002_gpu1` |
| V8b medium | `0.04` | `12` | none | `full_v8b_propcal_beta004_gpu1` |
| V8c capped | `0.04` | `12` | `32` | `full_v8c_propcal_beta004_cap32_gpu1` |

## Baselines

- current `midcons`: `795/1033 = 76.96%`
- Route2 precision len32: `801/1033 = 77.54%`
- V6 short override: `802/1033 = 77.64%`
- V7 proportional widening: `792/1033 = 76.67%`

## Success / Kill Criteria

Success criteria:

- each full run writes `1033` rows and `summary.json`;
- report pass count, pairwise wins/losses vs current `midcons`, oracle bucket pass rates, selected-minus-oracle diagnostics, and runtime;
- if any V8 run exceeds V6, update the main result table as a new follow-up candidate.

Kill criteria:

- stop the sequence if a run exits nonzero for a code/config error;
- stop if GPU1 OOMs repeatedly;
- do not kill or disturb other users' GPU processes.

## Verification Before GPU

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_length_probe_proportional_score.py tests/test_lcal_official_bounded_repair_proportional.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile expvision_dllm_clean/length_probe.py expvision_dllm_clean/config.py clean_scripts/run_cal_lite_lcal_rescue_policy.py clean_scripts/run_lcal_official_bounded_repair.py
/home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --help
git diff --check -- expvision_dllm_clean/config.py expvision_dllm_clean/length_probe.py clean_scripts/run_cal_lite_lcal_rescue_policy.py clean_scripts/run_lcal_official_bounded_repair.py tests/test_length_probe_proportional_score.py
```

Result: passed. Focused unit tests: `Ran 8 tests` / `OK`.

## Status

Completed. V8a, V8b, and V8c all wrote `1033` rows and `summary.json`.

## V8a Result

Run:

- output: `outputs_clean/full_v8a_propcal_beta002_gpu1_20260701_102654`
- log: `logs/paper_agent/20260701_v8a_propcal_beta002_gpu1.log`
- rows: `1033`
- pass: `786/1033 = 76.09%`

Comparison vs current `midcons`:

| Metric | Value |
|---|---:|
| pairwise wins | `3` |
| pairwise losses | `12` |
| tie-pass | `783` |
| tie-fail | `235` |
| short-bucket losses | `7` |
| long-bucket wins | `2` |

Oracle bucket pass rates:

| Bucket | Pass rate |
|---|---:|
| `<=8` | `88.80%` |
| `9-12` | `77.16%` |
| `13-16` | `57.78%` |
| `17-24` | `21.95%` |
| `25+` | `19.35%` |

Interpretation:

V8a is a faithful formula-level test, but it is negative as a policy candidate. It gives `2` long-bucket wins, including one `25+` win, but loses `12` tasks overall and causes `7` short-bucket losses. The mild proportional reward does reduce average under-selection somewhat, but it also increases short/medium false positives and does not beat current `midcons`, Route2, V6, or even V7.

## V8b Result

Run:

- output: `outputs_clean/full_v8b_propcal_beta004_gpu1_20260701_120525`
- log: `logs/paper_agent/20260701_v8b_propcal_beta004_gpu1.log`
- rows: `1033`
- pass: `781/1033 = 75.61%`

Comparison vs current `midcons`:

| Metric | Value |
|---|---:|
| pairwise wins | `7` |
| pairwise losses | `21` |
| tie-pass | `774` |
| tie-fail | `231` |
| short-bucket losses | `13` |
| long-bucket wins | `3` |

Oracle bucket pass rates:

| Bucket | Pass rate |
|---|---:|
| `<=8` | `87.79%` |
| `9-12` | `76.72%` |
| `13-16` | `58.89%` |
| `17-24` | `21.95%` |
| `25+` | `22.58%` |

Interpretation:

V8b confirms the expected tradeoff: stronger proportional reward rescues more long rows than V8a, including `2` `25+` wins, but it hurts many more short/medium rows. The overall result is worse than V8a and far below current `midcons`. The uncapped formula is too aggressive.

## V8c Result

Run:

- output: `outputs_clean/full_v8c_propcal_beta004_cap32_gpu1_20260701_134849`
- log: `logs/paper_agent/20260701_v8c_propcal_beta004_cap32_gpu1.log`
- rows: `1033`
- pass: `782/1033 = 75.70%`

Comparison vs current `midcons`:

| Metric | Value |
|---|---:|
| pairwise wins | `6` |
| pairwise losses | `19` |
| tie-pass | `776` |
| tie-fail | `232` |
| short-bucket losses | `11` |
| long-bucket wins | `2` |

Oracle bucket pass rates:

| Bucket | Pass rate |
|---|---:|
| `<=8` | `88.13%` |
| `9-12` | `76.72%` |
| `13-16` | `58.89%` |
| `17-24` | `21.95%` |
| `25+` | `19.35%` |

Interpretation:

V8c does not rescue the formula-level proportional idea. Capping the proportional reward at length `32` reduces the worst V8b behavior only slightly, but the overall result remains below `midcons`, Route2, V6, and V7. The cap is a reward cap, not a hard candidate-length cap: candidates `40` and `48` are still allowed, but their extra proportional exponent is computed as if the length were `32`.

## Overall V8 Comparison

| Run | Formula change | Pass | Rate | Pairwise vs `midcons` | Short losses | Long wins | Position |
|---|---|---:|---:|---:|---:|---:|---|
| current `midcons` | baseline | `795/1033` | `76.96%` | baseline | n/a | n/a | current main line |
| Route2 precision `len32` | trace-gated rescue | `801/1033` | `77.54%` | `6/0/795/232` | `0` | `2` | cleaner positive follow-up |
| V6 short override | selector polish | `802/1033` | `77.64%` | `7/0/795/231` | `0` | `2` | current best LLaDA-Base follow-up |
| V7 proportional widening | post-hoc near-best widening | `792/1033` | `76.67%` | `1/4/791/237` | `2` | `0` | negative |
| V8a proportional score | beta `0.02`, no cap | `786/1033` | `76.09%` | `3/12/783/235` | `7` | `2` | negative |
| V8b proportional score | beta `0.04`, no cap | `781/1033` | `75.61%` | `7/21/774/231` | `13` | `3` | negative |
| V8c proportional score | beta `0.04`, reward cap `32` | `782/1033` | `75.70%` | `6/19/776/232` | `11` | `2` | negative |

Decision:

V8 is the faithful test of the user's intended idea: directly modifying the CAL-like scoring formula so longer lengths receive proportionally larger reward. The result is negative. The formula does expose a real tradeoff, because larger beta increases long wins from `2` to `3`, but the gain is dominated by short and medium regressions. The current best LLaDA-Base follow-up remains V6 short override `802/1033 = 77.64%`.

Next:

Do not continue this direct proportional scoring family as the main policy. If the ratio idea is revisited, it should be under a stricter under-selection detector or with candidate-level risk guards, not as a global scoring reward applied to all rows.
