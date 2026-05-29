# A6000 Midcons And True-Long Rescue Design

Date: 2026-05-28  
Branch: `exp/a6000-midcons-longrescue`  
Primary hardware: A6000 GPUs `0,1`

## Objective

Re-establish the LCAL/LCAS baseline on the A6000 environment, preserve the proven mid-length gains from `midcons`, and open a separate true-long rescue path for `17-24` and `25+` oracle-length samples.

The near-term goal is not to claim a final SOTA result. The near-term goal is to produce a clean same-hardware comparison that tells us whether:

- `midcons` still improves over the A6000 union control.
- mid-rescue precision can reduce short/mid false positives without losing most gains.
- a true-long rescue branch can improve long buckets without damaging short buckets.

## Current Evidence

The latest tracked scoreboard says:

- Old global checkpoint: `old_union_gpus23`, `787/1033 = 76.19%`.
- New-environment control: `union_gpus01_control`, `785/1033 = 75.99%`.
- Best new-environment strategy: `midcons`, `791/1033 = 76.57%`.

The controlled new-environment comparison showed `midcons` at `+7 / -1`, net `+6`, over `union_gpus01_control`. Its main gain was in `13-16`; `17-24` and `25+` stayed unchanged. This means a pure mid-rescue tuning loop is unlikely to reach the next major result by itself.

## Baseline Runs

All A6000 baseline runs must use:

```bash
CUDA_VISIBLE_DEVICES=0,1
TOKENIZERS_PARALLELISM=false
```

The first two required full runs are:

1. A6000 union control:
   - runner: `clean_scripts/run_lcal_official_bounded_repair.py`
   - experiment name: `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control`
   - policy: old union settings, no mid rescue.

2. A6000 midcons rerun:
   - runner: `clean_scripts/run_lcal_official_bounded_repair.py`
   - experiment name: `full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000`
   - policy: conservative mid rescue `off11..13`, `delta3..7`, `ratio>=0.8`, `source=base`.

These establish the A6000 comparison base before any new strategy is judged.

## Candidate Strategy A: Mid-Rescue Precision

Purpose: keep most of `midcons` gains while removing direct short/mid losses.

Initial guards to evaluate offline first:

- Keep `official_len <= 13`.
- Keep `delta <= 7`.
- Add stronger long-curve support checks, such as:
  - `best_long_len in {13,14,15,16}`, or
  - adjacent support count above a threshold.
- Add a short-protection veto when `s3_len <= 5` and official jumps to `13`, unless long-curve support is strong.

Success criteria:

- Same-hardware net result over A6000 union control is positive.
- Direct losses in `<=8` and `9-12` do not exceed the A6000 `midcons` rerun.
- Most of the `13-16` gain is retained.

## Candidate Strategy B: True-Long Rescue

Purpose: target the unchanged `17-24` and `25+` buckets with a branch that is separate from mid rescue.

The true-long branch must be record-only until it proves that it does not harm short buckets. It should trigger only on high-confidence under-selection signatures, such as:

- S3/base selected length is short or medium.
- Official-CAL or long-grid evidence points clearly to a longer length.
- Long-curve support is strong and not flat/noisy.
- Candidate official length is high enough to plausibly represent true-long behavior.

Success criteria:

- Improves `17-24` or `25+` versus A6000 union control.
- Produces a clear win/loss table by oracle bucket.
- Does not become the default policy unless short-bucket regressions are bounded and explainable.

## Cross-Model Validation

Cross-model work is part of the project goal, but it should come after the A6000 single-model comparison is stable.

Priority order:

1. Re-run the best A6000 policy on the current primary base model.
2. Select one additional model for first transfer validation.
3. Expand to more models only after the run harness and metrics are stable.

Candidate models:

- `Dream-Coder-7B`
- `DiffuCoder-7B`
- `LLaDA-8B`
- `Dream-7B`

Each cross-model run must record model path, tokenizer behavior, prompt format differences, evaluation compatibility, total pass rate, bucket metrics, and whether the comparison is directly comparable to the primary model.

## Literature And SOTA Comparison

SOTA comparison requires source verification before any claim.

The comparison doc must record:

- paper or official repository link
- dataset and split
- metric
- base model and parameter count
- infilling setting and prompt format
- whether oracle length, fixed length, or learned length control is used
- whether the comparison is direct, approximate, or not comparable

No SOTA claim should be made from memory or from mismatched settings.

## Result Recording

Raw full outputs stay local under `outputs_clean/` unless explicitly selected for Git LFS or external archival.

Committed records should include:

- this spec
- run command manifests
- compact run summaries
- updated `analysis_outputs/experiment_scoreboard.md`
- bucket tables
- wins/losses versus A6000 union control
- notes explaining any environment drift

## Implementation Boundaries

- Keep legacy `expvision_dllm/` and `scripts/` frozen.
- Prefer edits under `clean_scripts/`, `analysis/`, `analysis_outputs/`, and `docs/`.
- New true-long logic should be separate from existing mid-rescue logic where possible.
- Do not make a new strategy the default until A6000 baseline comparisons are recorded.

## Open Risks

- A6000 results may differ from the earlier old-server and new-5090 runs; same-hardware comparisons must be primary.
- True-long rescue may win long cases while hurting short cases; it must start as record-only.
- Cross-model validation may require model-specific loading or prompt changes, so it should be gated behind stable primary-model results.
- Raw result files are large; preserving every `results.jsonl` in normal git would make the repo hard to use.
