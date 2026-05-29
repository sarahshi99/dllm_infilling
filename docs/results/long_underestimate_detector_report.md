# Long Underestimate Detector Sweep Report

Updated: 2026-05-29 Asia/Shanghai

## Purpose

This report checks whether the current A6000 `midcons` result contains enough inference-time signals to safely trigger a long-tail rescue rule. It is a diagnostic-only analysis; it does not change any generated code result.

## Command

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/diagnose_long_underestimate_policy.py \
  --results outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626/results.jsonl \
  --output-dir analysis_outputs/long_underestimate_detector/a6000_midcons
```

## Result

The sweep evaluated `16776` candidate rules over `1033` rows.

Best scoring rule:

```text
sel<=3|best>=13|gap>=4|ratio>=0.45|raw>=0.4|supp>=0|src=base
```

Metrics:

- triggers: `93`
- true-long precision: `35.48%`
- failed-long recall: `36.26%`
- short-risk rate: `40.86%`
- current-pass risk: `27.96%`

Strict success criteria:

- rules with trigger count >= `10`
- true-long precision >= `60%`
- short-risk rate <= `5%`
- failed-long triggered rows >= `10`

Result: `0` rules passed.

Even relaxing short-risk to `10%` or `20%` produced no viable rule with at least five triggers. The first non-empty low-risk family appears only around `short_risk <= 40%`, which is too dangerous for a GPU policy.

## Interpretation

The current result fields do not contain a safe long-underestimation trigger. The long failures are real and mostly under-selected, but the same weak long-curve signals also appear in short/medium cases. A direct heuristic rescue would likely damage the short buckets that made `midcons` safe.

## Decision

Do not launch another GPU experiment based only on the current gate family. The next credible long-tail direction should add a stronger signal:

- denoising trajectory/step-trace features,
- a learned length classifier,
- a DreamOn-style dynamic canvas method,
- or LR-DLLM-style length regularization.

This negative result supports the paper narrative: medium rescue can be short-safe with confidence-curve agreement, but true long infilling needs stronger length modeling.
