# Trace Feature Audit V2 Report

Decision: `diagnostic_only`

Decision reason: At least one source has signal, but the policy-candidate gate is not stable across both trace sources.

## Source Summary

| Source | Rows | True-long | Failed-long | Short | Decision |
|---|---:|---:|---:|---:|---|
| midcons | 1033 | 113 | 91 | 598 | diagnostic_only |
| previous | 1033 | 113 | 96 | 598 | policy_candidate |

## Top Candidates

| Source | Candidate | Family | Decision | Fold | Train rank | Triggers | Failed-long | Short risk | Current-pass risk | Precision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| previous | `confidence_first_le_0p875_AND_gap_last_le_0p84375` | `pairwise` | `policy_candidate` | 1 | 2 | 21 | 10 | 4 | 4 | 0.476 |
| previous | `confidence_first_le_0p882812_AND_gap_last_le_0p84375` | `pairwise` | `policy_candidate` | 4 | 3 | 25 | 11 | 5 | 3 | 0.440 |
| previous | `top1_last_le_0p859375_AND_confidence_first_le_0p875` | `pairwise` | `policy_candidate` | 1 | 3 | 20 | 10 | 3 | 4 | 0.500 |
| previous | `top1_last_le_0p605469_AND_confidence_min_le_0p730469` | `pairwise` | `policy_candidate` | 3 | 4 | 15 | 10 | 3 | 0 | 0.667 |
| previous | `top1_last_le_0p605469_AND_confidence_first_le_0p703125` | `pairwise` | `policy_candidate` | 3 | 5 | 15 | 10 | 3 | 0 | 0.667 |
| previous | `confidence_min_le_0p878906_AND_gap_last_le_0p84375` | `pairwise` | `policy_candidate` | 4 | 7 | 26 | 11 | 5 | 3 | 0.423 |
| previous | `top1_last_le_0p859375_AND_confidence_first_le_0p882812` | `pairwise` | `policy_candidate` | 4 | 9 | 25 | 11 | 5 | 3 | 0.440 |
| previous | `top1_last_le_0p605469_AND_confidence_max_le_0p921875` | `pairwise` | `policy_candidate` | 3 | 11 | 15 | 10 | 3 | 0 | 0.667 |
| previous | `top1_last_le_0p605469_AND_confidence_last_le_0p921875` | `pairwise` | `policy_candidate` | 3 | 14 | 15 | 10 | 3 | 0 | 0.667 |
| previous | `top1_last_le_0p859375_AND_confidence_min_le_0p878906` | `pairwise` | `policy_candidate` | 4 | 14 | 26 | 11 | 5 | 3 | 0.423 |
| previous | `confidence_median_le_0p804688_AND_gap_last_le_0p601562` | `pairwise` | `policy_candidate` | 3 | 15 | 16 | 10 | 4 | 1 | 0.625 |
| previous | `top1_last_le_0p605469_AND_gap_median_le_0p71875` | `pairwise` | `policy_candidate` | 3 | 20 | 15 | 10 | 3 | 0 | 0.667 |
| midcons | `top1_last_le_0p667969_AND_max_remaining_plateau_steps_ge_16` | `pairwise` | `diagnostic_only` | 4 | 1 | 18 | 9 | 2 | 0 | 0.500 |
| previous | `confidence_max_le_0p835938_AND_confidence_first_le_0p703125` | `pairwise` | `diagnostic_only` | 3 | 1 | 14 | 9 | 3 | 0 | 0.643 |
| midcons | `top1_last_le_0p625_AND_max_remaining_plateau_steps_ge_16` | `pairwise` | `diagnostic_only` | 3 | 1 | 12 | 8 | 2 | 0 | 0.667 |
| previous | `confidence_first_le_0p875_AND_top1_last_le_0p945312` | `pairwise` | `diagnostic_only` | 1 | 1 | 26 | 11 | 8 | 5 | 0.423 |
| midcons | `confidence_min_le_0p714844_AND_last_remaining_decrease_step_ge_40` | `pairwise` | `diagnostic_only` | 1 | 1 | 11 | 6 | 1 | 1 | 0.545 |
| previous | `confidence_slope_ge_0p00176711_AND_last_remaining_decrease_step_ge_40` | `pairwise` | `diagnostic_only` | 4 | 1 | 15 | 6 | 3 | 2 | 0.467 |
| midcons | `gap_median_le_0p751953_AND_remaining_last_le_1` | `pairwise` | `diagnostic_only` | 2 | 1 | 17 | 7 | 6 | 3 | 0.412 |
| previous | `top1_median_le_0p78125_AND_gap_last_le_0p84375` | `pairwise` | `diagnostic_only` | 2 | 1 | 23 | 8 | 9 | 4 | 0.348 |
