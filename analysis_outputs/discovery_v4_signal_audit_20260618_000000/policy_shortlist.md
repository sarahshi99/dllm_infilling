# Discovery V4 Policy Shortlist

Decision: `route2_polish_only`

Reason: No stable low-risk V4 signal was found beyond Route2 polish.

## Best Candidate

- Name: `broad_len24_triggered_ge_1`
- Family: `single`
- Clauses: `[["broad_len24_triggered", ">=", 1.0]]`
- Stable folds: `3`
- Missed failed-long: `4`
- Short risk: `10`
- Current-pass risk: `1`

## Next Action

Keep Route2 precision len32 as conservative polish and avoid a blind full policy run.
