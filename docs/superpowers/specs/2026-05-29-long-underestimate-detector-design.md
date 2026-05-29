# Long Underestimate Detector Design

Date: 2026-05-29

## Objective

Design the next long-tail experiment after the A6000 `true_long` candidate failed to improve pass rate. The goal is to detect cases where the current policy under-selects mask length for `oracle >= 17` without sacrificing `<=8` and `9-12`.

## Evidence

Completed A6000 runs show:

- `midcons`: `795/1033 = 76.96%`, `+8` vs A6000 control with `0` losses.
- `true_long`: `787/1033 = 76.19%`, exactly tied with control.
- In `true_long`, all true-long gates passed for `0` samples.
- If the support gate is removed, the remaining candidates are already-passing short/medium samples, not useful long fixes.
- In `midcons`, among `oracle >= 17` failures, `90/91` are under-selected and `71/91` still come from `base`.

This means official-CAL is not a reliable primary true-long trigger. A separate detector is needed.

## Design

Add an offline analysis script first, not a GPU runner:

`analysis/diagnose_long_underestimate_policy.py`

The script will sweep simple gates over an existing `results.jsonl`:

- `selected_mask_length <= max_selected_len`
- `best_long_len >= min_best_long_len`
- `best_long_len - selected_mask_length >= min_gap`
- `long_ratio >= min_long_ratio`
- `raw_long_ratio >= min_raw_long_ratio`
- `support_count >= min_support_count`
- optional final source allow-list, defaulting to `base,strong_correction,weak_correction`

For each candidate rule, it will report:

- trigger count.
- true-long precision: triggered rows with `oracle_mask_length >= 17`.
- short-risk rate: triggered rows with `oracle_mask_length <= 8`.
- medium-risk rate: triggered rows with `oracle_mask_length <= 12`.
- long-failed recall: triggered rows among failed `oracle >= 17` rows.
- current-pass risk: triggered rows that already pass under the current policy.

The output is diagnostic only. It does not change generated code or claim a new score.

## Why This First

Running a new GPU policy without this sweep would be blind. The last true-long attempt showed that apparently reasonable gates either trigger nothing or mostly protect against short false positives. The sweep gives a cheap precision/recall map before we create a new runner.

## Success Criteria

A candidate is worth GPU evaluation only if it satisfies all of:

- true-long precision at least `0.60` among triggered rows, or a clearly explainable lower threshold with high recall.
- short-risk rate at most `0.05`.
- at least `10` triggered failed-long rows.
- no obvious dependence on oracle-only features.

If no candidate satisfies this, the next research move should shift from heuristic inference rescue to either learned length scoring or a DreamOn/LR-DLLM style length-regularized method.

## Implementation Notes

- The script is read-only over raw outputs.
- It writes compact JSON/CSV/Markdown to `analysis_outputs/long_underestimate_detector/`.
- It uses only fields already present in `results.jsonl`, so it can run on historical outputs and cross-model outputs.
- Unit tests should use tiny synthetic JSONL fixtures to verify counts and rates.

## Self-Review

- No placeholder requirements remain.
- The spec is scoped to an offline diagnostic, not the full GPU runner.
- The method does not use oracle at inference time; oracle labels are only used for offline evaluation.
- The output directly informs whether a GPU experiment is worth running.
