# LCAL Infilling Experiment Scoreboard

Updated: 2026-05-29 Asia/Shanghai

## Current Decision

`midcons` is now also the best strategy in the A6000 rerun. It improves the A6000 control by `+8` wins and `0` losses, reaching `795/1033 = 76.96%`.

The current long-tail branch is negative evidence: `true_long` and `combined` remain exactly tied with the A6000 control because the official-CAL true-long trigger fires zero times after safety guards. The next long experiment should use a separate long-underestimation detector rather than loosening the same official-CAL trigger.

Historical 5090 decision:

`midcons` was the best strategy in the RTX 5090 / torch 2.11 cu128 environment, but it was not yet a clean global best checkpoint against the old-server union checkpoint because the environment change itself introduced a small short-bucket regression.

Use this terminology:

- **Old global checkpoint:** `old_union_gpus23`, `787/1033 = 76.19%`.
- **New-environment control:** `union_gpus01_control`, `785/1033 = 75.99%`.
- **Best new-environment strategy:** `midcons`, `791/1033 = 76.57%`.

## Main Table

| Run | Env | Method | Pass | Delta vs old union | Delta vs gpus01 union | <=8 | 9-12 | 13-16 | 17-24 | 25+ | Best status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `old_union_gpus23` | old server / gpus23 | union: S3 short-safe + bounded repair off6..9 + long suspicion off>=16 | `787/1033` `76.19%` | baseline | n/a | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | old global best |
| `union_gpus01_control` | new 5090 / torch cu128 | exact old union config rerun | `785/1033` `75.99%` | `-2` | baseline | `89.30%` | `77.59%` | `54.44%` | `20.73%` | `16.13%` | environment control |
| `eval12_nomiddle_gpus01_control` | new 5090 / torch cu128 | old union, but official probing allowed for S3<=12; no mid rescue | `785/1033` `75.99%` | `-2` | `0` | `89.30%` | `77.59%` | `54.44%` | `20.73%` | `16.13%` | gate-control only |
| `midcons` | new 5090 / torch cu128 | union + conservative mid rescue off11..13, delta3..7, ratio>=0.8 | `791/1033` `76.57%` | `+4` | `+6` | `89.30%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` | strategy-best in gpus01 env |
| `midaggr` | new 5090 / torch cu128 | union + aggressive mid rescue off11..15, delta3..8, ratio>=0.8 | `789/1033` `76.38%` | `+2` | `+4` | `88.96%` | `77.59%` | `61.11%` | `20.73%` | `16.13%` | record only; hurts short |

## A6000 Controlled Runs

| Run | Method | Pass | Delta vs A6000 control | <=8 | 9-12 | 13-16 | 17-24 | 25+ | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `a6000_control` | union control: S3 short-safe + bounded repair + long suspicion | `787/1033` `76.19%` | baseline | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | A6000 baseline |
| `midcons` | union + conservative mid rescue off11..13, delta3..7, ratio>=0.8 | `795/1033` `76.96%` | `+8` (8W/0L) | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` | A6000 best |
| `mid_precision` | mid rescue + support/best-len/short-jump guards | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | negative evidence |
| `true_long` | true-long rescue off>=17, delta>=8, ratio>=0.85, support>=2 | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | negative evidence |
| `combined` | mid precision + true-long rescue | `787/1033` `76.19%` | `0` (0W/0L) | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | negative evidence |

## Controlled Comparisons

| Comparison | Wins | Losses | Net | Interpretation |
|---|---:|---:|---:|---|
| `union_gpus01_control` vs `old_union_gpus23` | 4 | 6 | -2 | Cross-server / torch change is real but small. It mainly affects <=8 and 9-12 tasks. |
| `eval12_nomiddle_gpus01_control` vs `union_gpus01_control` | 0 | 0 | 0 | Expanding official evaluation to S3<=12 is not itself changing outputs when no mid rescue rule is enabled. |
| `midcons` vs `union_gpus01_control` | 7 | 1 | +6 | Conservative mid rescue is genuinely useful in the same environment. |
| `midaggr` vs `union_gpus01_control` | 9 | 5 | +4 | Aggressive mid rescue gets more wins but introduces too many short/mid false positives. |

## What The First Detailed Breakdown Means

The first breakdown decomposes each new result by:

- Pairwise wins/losses against a control run.
- Oracle-length bucket of each win/loss.
- Final source (`base`, `official_bounded_repair`, `official_long_suspicion`, `official_mid_rescue`, `strong_correction`, `weak_correction`).
- Length transition, for example `S3 6 -> official 13`.
- Whether the loss is caused directly by `official_mid_rescue` or by environment/run-stack drift.

This matters because the earlier quick summary compared `midcons` directly to `old_union_gpus23`, which mixed strategy change with server/PyTorch change.

## Why Offline Expectations Did Not Match Full

Offline expectation for conservative mid rescue was roughly `+8 / 0 loss` over union. Full controlled result is `+7 / 1 loss` over `union_gpus01_control`.

Reasons:

1. **Environment drift existed.** Re-running the exact old union on the 5090 stack changed 10 tasks pairwise: 4 wins and 6 losses, net -2. So a direct comparison to old `gpus23` results exaggerates or misattributes some losses.
2. **The gate-control is innocent.** `official_eval_max=12` without mid rescue produced exactly the same pass/fail set as union control. So the wider official probing window is not the source of result changes by itself.
3. **Mid rescue is too broad for short oracle cases.** Conservative mid rescue triggered 24 times: 18 pass, 6 fail. It caused 7 controlled wins and 1 controlled loss. Aggressive mid rescue triggered 43 times: 30 pass, 13 fail, with 5 controlled losses. The false positives are mostly true-short or 9-12 oracle tasks that official-CAL over-extends to 13-15.
4. **It does not solve true-long under-selection.** 17-24 and 25+ bucket pass rates remain unchanged at `20.73%` and `16.13%`. The current mid rescue mostly improves 13-16 and some 9-12 tasks, not the long tail needed for 80%.

## Design Diagnosis

The current mid rescue rule:

```text
S3 <= 12
official_len in 11..13 or 11..15
delta in 3..7 or 3..8
long_ratio >= 0.8
source == base
```

What worked:

- It recovers under-selected medium cases where S3 picked 6-9 and official-CAL picked 11-13.
- Conservative version improves the 13-16 bucket from `54.44%` to `58.89%` in the same environment.
- It does not reduce aggregate <=8 bucket pass rate relative to gpus01 union control, though it still has one direct short loss and one short win.

What failed:

- `long_ratio >= 0.8` is not selective enough. Short false positives often still have ratios around `0.83..0.97`.
- Official selected length around 13-15 is not a reliable true-long signal. It can be an over-extension for oracle <=8 or 9-12.
- Aggressive off11..15/delta3..8 improves 13-16 more but damages <=8 and 9-12, so it cannot be a best checkpoint under the “no short sacrifice” criterion.

## Next Experiments

Priority order:

1. **Mid rescue precision pass.**
   Keep conservative range but add guards to remove the direct loss without giving up most wins:
   - Require `official_len <= 13`.
   - Require `delta <= 7`.
   - Add raw/curve agreement: official rescue only if base long evidence is not flat/noisy, e.g. `best_long_len in {13,14,15,16}` or adjacent support count >= 2.
   - Add a short-protection veto when base/S3 <= 5 and official jumps to 13 unless long curve support is strong.

2. **True-long rescue branch, separate from mid rescue.**
   Current long buckets are unchanged, so reaching 80% requires a different branch for oracle >=17:
   - Do not let this branch affect <=8/9-12.
   - Trigger only on high official length, strong long curve support, and likely failure/under-selection signature.
   - Evaluate as a record-only experiment first if it sacrifices short.

3. **Same-environment checkpoint policy.**
   Use `union_gpus01_control` as the baseline for new-server strategy decisions.
   Keep `old_union_gpus23` as the old global checkpoint until a candidate improves total and does not regress short buckets under a controlled environment comparison.

## Files

- Old union: `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus23_20260519_175826`
- gpus01 union control: `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus01_control_20260521_174051`
- gpus01 eval12/no-mid control: `outputs_clean/full_lcal_official_bounded_repair_union_eval12_nomiddle_s3_off6_9_delta1_8_susp16_gpus01_control_20260521_182208`
- midcons: `outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_gpus01_20260520_201658`
- midaggr: `outputs_clean/full_lcal_official_bounded_repair_union_midaggr_off11_15_d3_8_r08_gpus01_20260520_210248`
