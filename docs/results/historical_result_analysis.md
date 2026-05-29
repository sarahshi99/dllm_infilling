# Historical Result Analysis

Updated: 2026-05-29 Asia/Shanghai

This document summarizes the valuable historical experiment results without committing raw `outputs_clean/` artifacts. The machine-readable index remains `docs/results/run_registry.json`; the human-readable index remains `docs/results/run_registry.md`.

## Inventory

The current registry contains `49` meaningful local runs. Smoke runs and incomplete no-summary runs are excluded by default.

Raw outputs remain local:

- `/home/shx/projects/dllm_infilling/outputs_clean`
- `/home/shx/projects/dllm_infilling/outputs_clean/202604`
- `/home/shx/projects/dllm_infilling/model_generalization_runs`

## Best Local LLaDA-Base Lineage

| Stage | Representative run | Pass | Why it matters |
|---|---|---:|---|
| Oracle upper reference | `full_oracle_sl_20260410_145452` | `900/1033 = 87.12%` | Shows length choice is a major bottleneck; generation can succeed much more often when the missing length is known. |
| Fixed-length baseline | `full_fixed_sl_20260410_161152` | `478/1033 = 46.27%` | Fixed canvas length is not competitive for single-line infilling. |
| Early CAL-lite | `full_cal_lite_sl_20260414_164141` | `550/1033 = 53.24%` | Initial confidence probing beats fixed length but leaves large length-selection errors. |
| CAL-lite v1 | `full_cal_lite_v1_sl_20260415_194824` | `718/1033 = 69.51%` | Major jump from better mask-length probing. |
| CAL-lite v2 alpha sweep | `full_cal_lite_v2_alpha_008_sl_20260417_131256` | `770/1033 = 74.54%` | Best early alpha family; later superseded. |
| LCAS/LCAL family | `full_lcas_v3b_alpha006_compact_sl_20260429_180958` | `769/1033 = 74.44%` | Adaptive stopping stabilized but did not alone surpass later bounded repair. |
| Official bounded repair | `full_lcal_official_bounded_repair_s3_off6_11_delta1_8_gpus01_20260515_163902` | `784/1033 = 75.90%` | Repairing short under-selection becomes the reliable path. |
| Union checkpoint | `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus23_20260519_175826` | `787/1033 = 76.19%` | Stable pre-A6000 global checkpoint. |
| A6000 control | `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529` | `787/1033 = 76.19%` | Same score as old union; good A6000 baseline. |
| A6000 midcons | `full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626` | `795/1033 = 76.96%` | Current best same-hardware checkpoint; `+8` wins and `0` losses vs A6000 control. |

## What The History Shows

The project has already moved past simple fixed-length decoding. The consistent pattern is:

1. Better length selection gives the largest gains.
2. Bounded short/medium repair is safer than broad long repair.
3. Conservative mid rescue can improve `9-12` and `13-16` buckets without visible aggregate short damage.
4. The remaining gap to the oracle run is dominated by long under-selection.

The best current LLaDA-Base result, `795/1033`, is still `105` passes below the oracle-length reference. This is the research opportunity, but it is also the main risk: more hand-tuned rescue rules are unlikely to close the gap alone.

## Runs To Keep As Canonical

Keep these as canonical compact references:

- `full_oracle_sl_20260410_145452`: oracle-length upper reference.
- `full_fixed_sl_20260410_161152`: fixed-length lower baseline.
- `full_cal_lite_v1_sl_20260415_194824`: first large CAL-lite improvement.
- `full_cal_lite_v2_alpha_008_sl_20260417_131256`: best early alpha sweep.
- `full_lcas_v3b_alpha006_compact_sl_20260429_180958`: LCAS v3 reference.
- `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus23_20260519_175826`: old union checkpoint.
- `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529`: A6000 control.
- `full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626`: current A6000 best.

Other full runs should remain indexed because they document failed directions and prevent repeated experiments, but they do not need raw JSONL preservation in normal git.

## Cross-Model Historical Records

The cross-model registry contains `7` full local records:

| Model/run | Pass | Interpretation |
|---|---:|---|
| Dream-Coder Instruct official canvas `alpha010_cap24` | `848/1033 = 82.09%` | Strongest local cross-model record, but protocol differs from the LLaDA LCAL stack. |
| Dream-Coder Base official canvas `alpha010_cap24` | `825/1033 = 79.86%` | Strong transfer signal for official-canvas prompting. |
| Dream-Coder Base official canvas `alpha020` | `818/1033 = 79.19%` | Worse than `alpha010_cap24`; useful alpha sensitivity evidence. |
| LLaDA-8B-Instruct LCAS v3 resume | `817/1033 = 79.09%` | Suggests instruct checkpoint can be strong under this local protocol. |
| Dream-Coder Instruct official canvas `alpha020` | `785/1033 = 75.99%` | Alpha sensitivity is large; not enough to claim model weakness. |
| Dream-Coder LCAS v3 runs | `0/1033` and `1/1033` | Prompt/canvas mismatch, not a meaningful model-quality conclusion. |

Cross-model conclusion: there is enough historical evidence to justify a model-generalization track, but the existing records are not yet a clean SOTA comparison because prompt format and canvas assumptions differ.

## Paper-Relevant Takeaways

For a CCF-A-quality paper, the history supports this narrative:

- The central technical bottleneck is unknown-length infilling for DLLMs.
- Inference-time confidence probing and bounded repair recover many short/medium cases.
- Same-hardware A6000 evidence validates conservative mid rescue.
- True-long cases remain hard because current signals confuse long failures with short/medium false positives.

The current claim is credible as an empirical diagnosis plus an inference-time medium rescue method. It is not yet enough as a top-tier paper unless the next stage adds a stronger long-tail method, cross-model validation, or a sharper theoretical/algorithmic contribution.

## Archive Policy

Do not delete historical raw outputs. Do not commit raw `results.jsonl` files to normal git unless a specific run is selected for Git LFS or external artifact storage. Continue to preserve:

- compact run registries;
- pairwise summaries;
- bucket summaries;
- experiment plans/specs;
- interpretation reports like this file.
